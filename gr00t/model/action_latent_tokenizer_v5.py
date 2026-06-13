"""Action Latent Tokenizer V5 — RLA-LAM hybrid (DreamDojo pixel decoder).

V5 is V4 with the **visual element swapped from DINO/VGGT to DreamDojo's Latent
Action Model (LAM)**. Everything orthogonal to "what is the visual signal / what
is the reconstruction target" is reused verbatim from V4 (imported, not copied,
so behavior cannot drift):

  * ``TimeWiseEncoderV4`` — action encoder + RLA ``SimpleTokenTransformer`` fusion,
    **including the SD-style VAE bottleneck** (``use_vae`` / KL / free-bits). The
    only difference is ``dino_dim`` is now the LAM latent width (32), because the
    fusion's visual context is the LAM ``z_rep`` token, not DINO patch tokens.
  * ``ReconDecoderV4`` — action reconstruction decoder (self/cross attention,
    input up-proj, global/hand handling) — used unchanged.

The two V5-specific changes vs V4:

  1. **Visual context (fusion input).** Instead of ``dino_diff = x1 - x0`` patch
     tokens ``[B, Lp, C]``, the fusion consumes the LAM ``z_rep`` ``[B, 1, 32]`` —
     a single latent-action token a *frozen* LAM encoder distills from the
     ``(frame0, frame1)`` pair. The LAM encoder lives OUTSIDE this module
     (trainer/wrapper-owned, frozen, not in this state_dict), exactly mirroring the
     V4 DINO/VGGT extractor pattern.

  2. **Reconstruction target (loss 2).** Instead of predicting future DINO
     features (``dino_decoder``), V5 reconstructs the **future-frame pixels** with
     LAM's ``SpatioTransformer`` decoder (``PixelDecoderV5``). The per-timestep
     action latents (``time_out [B,T,token_dim]``) are merged into a single latent
     via a **learnable softmax weighted sum**, which plays LAM's ``z_rep`` role
     (``action_up`` input). ``patch_up`` / decoder are initialized from the
     pretrained LAM checkpoint (trainable); ``action_up`` is fresh (token_dim≠32).

Losses: ``loss = lambda_recon * action_recon + lambda_pixel * pixel_recon
(+ lambda_kl * KL)``.

Interface mirrors V4 (so wrapper / VLA code is reused): ``encode(...) ->
(global, time, hand)`` and ``decode(global, time, hand) -> actions``. Unlike V4,
``encode`` consumes the LAM ``z_rep`` instead of DINO feats, and pixel
reconstruction is via ``decode_pixel``.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse V4 building blocks verbatim (import, not copy → behavior cannot drift).
from gr00t.model.action_latent_tokenizer_v4 import (
    ReconDecoderV4,
    TimeWiseEncoderV4,
    _str_to_byte_tensor,
    byte_tensor_to_str,
)
from gr00t.model.lam_modules import SpatioTransformer, patchify, unpatchify

# Re-export so callers can ``from ..._v5 import byte_tensor_to_str`` (parity w/ v4).
__all__ = [
    "PixelDecoderV5",
    "ActionLatentTokenizerV5",
    "byte_tensor_to_str",
    "_str_to_byte_tensor",
]


# =====================================================================
# V5 pixel decoder (DreamDojo LAM SpatioTransformer + patch_up/action_up)
# =====================================================================


class PixelDecoderV5(nn.Module):
    """Future-frame pixel decoder, structurally identical to LAM's decoder path.

    Reconstructs ``frame1`` from ``frame0`` patches + a single action latent::

        z_pix      = Σ_t softmax(pool_logits)_t · time_tok[:, t]      # [B, token_dim]
        action_emb = action_up(z_pix)                                 # [B, model_dim]
        patch_emb  = patch_up(patchify(frame0))                       # [B, 1, N, model_dim]
        recon      = sigmoid(decoder(patch_emb + action_emb))         # [B, 1, N, patch_token_dim]
        frame1_hat = unpatchify(recon)                                # [B, 1, H, W, C]

    ``patch_up`` and ``decoder`` use LAM's submodule names so the pretrained LAM
    checkpoint loads key-for-key (see the trainer's ckpt-init step). ``action_up``
    maps ``token_dim → model_dim`` and is fresh-initialized whenever ``token_dim``
    differs from LAM's ``latent_dim`` (32), e.g. the default ``token_dim=64``.
    """

    def __init__(
        self,
        action_horizon: int,
        token_dim: int = 64,
        image_channels: int = 3,
        patch_size: int = 16,
        model_dim: int = 1024,
        dec_blocks: int = 24,
        num_heads: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.token_dim = token_dim
        self.patch_size = patch_size
        self.model_dim = model_dim
        patch_token_dim = image_channels * patch_size ** 2  # 3*16^2 = 768

        # Learnable softmax weighted-sum over the T per-timestep latents → 1 latent.
        # Zero-init logits ⇒ starts as a uniform mean over timesteps.
        self.pool_logits = nn.Parameter(torch.zeros(action_horizon))

        # LAM submodule names (patch_up / action_up / decoder) preserved for ckpt load.
        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.action_up = nn.Linear(token_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=patch_token_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

    def pool(self, time_tok: torch.Tensor) -> torch.Tensor:
        """[B,T,token_dim] → [B,token_dim] via learnable softmax weighted sum."""
        w = torch.softmax(self.pool_logits, dim=0)  # [T]
        return torch.einsum("t,btd->bd", w.to(time_tok.dtype), time_tok)

    def forward(self, time_tok: torch.Tensor, frame0: torch.Tensor) -> torch.Tensor:
        """time_tok [B,T,token_dim] + frame0 [B,1,H,W,C] (∈[0,1]) → frame1_hat [B,1,H,W,C]."""
        B, _, H, W, _ = frame0.shape

        z_pix = self.pool(time_tok)                       # [B, token_dim]
        action_emb = self.action_up(z_pix)                # [B, model_dim]
        action_emb = action_emb[:, None, None, :]         # [B, 1, 1, model_dim]

        patches = patchify(frame0, self.patch_size)       # [B, 1, N, patch_token_dim]
        patch_emb = self.patch_up(patches.to(action_emb.dtype))  # [B, 1, N, model_dim]

        x = patch_emb + action_emb                        # broadcast-add over N
        recon = torch.sigmoid(self.decoder(x))            # [B, 1, N, patch_token_dim]
        return unpatchify(recon, self.patch_size, H, W)   # [B, 1, H, W, C]


# =====================================================================
# V5 tokenizer
# =====================================================================


class ActionLatentTokenizerV5(nn.Module):
    """RLA-LAM hybrid action latent tokenizer (continuous latent, pixel recon).

    Losses: ``loss = lambda_recon * action_recon + lambda_pixel * pixel_recon
    (+ lambda_kl * KL when the encoder is a VAE)``.

    ``encode`` consumes the LAM ``z_rep`` ``[B,1,32]`` (the frozen LAM extractor is
    external — trainer/wrapper-owned, not in this state_dict). ``forward`` also
    needs the raw ``frame0``/``frame1`` videos (``[B,1,H,W,C]`` in ``[0,1]``) for
    pixel reconstruction.
    """

    def __init__(
        self,
        encoder: TimeWiseEncoderV4,
        recon_decoder: ReconDecoderV4,
        pixel_decoder: Optional[PixelDecoderV5] = None,
        lambda_recon: float = 1.0,
        lambda_pixel: float = 1.0,
        lambda_kl: float = 0.0,
        recon_loss_type: str = "mse",
        pixel_loss_type: str = "mse",
        # ---- LAM extractor descriptor (recorded as markers; rebuilt at inference) ----
        lam_ckpt: Optional[str] = None,
        lam_model_dim: int = 1024,
        lam_latent_dim: int = 32,
        lam_patch_size: int = 16,
        lam_enc_blocks: int = 24,
        lam_dec_blocks: int = 24,
        lam_num_heads: int = 16,
        lam_image_h: int = 240,
        lam_image_w: int = 320,
    ):
        super().__init__()
        self.encoder = encoder
        self.recon_decoder = recon_decoder
        self.pixel_decoder = pixel_decoder

        self.lambda_recon = float(lambda_recon)
        self.lambda_pixel = float(lambda_pixel)
        self.lambda_kl = float(lambda_kl)
        self.recon_loss_type = recon_loss_type
        self.pixel_loss_type = pixel_loss_type

        # VAE flag mirrors v4 (single source of truth on the encoder).
        self.use_vae = bool(getattr(encoder, "use_vae", False))

        # convenience attrs mirrored from encoder (some wrapper paths read these)
        self.num_global_tokens = encoder.num_global_tokens
        self.num_hand_tokens = encoder.num_hand_tokens
        self.hand_in_recon = False  # V5 minimal scope: no hand tokens in recon

        # ---- detection markers ----
        # V5 detection marker.
        self.register_buffer("_is_v5", torch.tensor(True))
        # VAE detection marker (only when enabled → off-path state_dict unchanged).
        if self.use_vae:
            self.register_buffer("_is_vae", torch.tensor(True))

        # LAM extractor descriptor. Stored as buffers so the inference wrapper can
        # rebuild the matching frozen LAM encoder (the extractor itself is NOT part
        # of this state_dict — trainer/wrapper-owned, frozen).
        self.lam_ckpt = lam_ckpt
        self.lam_image_h = int(lam_image_h)
        self.lam_image_w = int(lam_image_w)
        self.register_buffer("_lam_model_dim", torch.tensor(int(lam_model_dim)))
        self.register_buffer("_lam_latent_dim", torch.tensor(int(lam_latent_dim)))
        self.register_buffer("_lam_patch_size", torch.tensor(int(lam_patch_size)))
        self.register_buffer("_lam_enc_blocks", torch.tensor(int(lam_enc_blocks)))
        self.register_buffer("_lam_dec_blocks", torch.tensor(int(lam_dec_blocks)))
        self.register_buffer("_lam_num_heads", torch.tensor(int(lam_num_heads)))
        self.register_buffer("_lam_image_h", torch.tensor(int(lam_image_h)))
        self.register_buffer("_lam_image_w", torch.tensor(int(lam_image_w)))
        if lam_ckpt:
            self.register_buffer("_lam_ckpt", _str_to_byte_tensor(str(lam_ckpt)))

    @staticmethod
    def _zero(device):
        return torch.zeros((), device=device)

    def _recon_loss_fn(self, pred, target):
        if self.recon_loss_type == "l1":
            return F.l1_loss(pred, target)
        return F.mse_loss(pred, target)

    def _pixel_loss_fn(self, pred, target):
        target = target.to(dtype=pred.dtype)
        if self.pixel_loss_type == "l1":
            return F.l1_loss(pred, target)
        return F.mse_loss(pred, target)

    # ---- interface (matches V4 plus LAM z_rep / pixel recon) ----

    def encode(self, actions: torch.Tensor, z_rep: torch.Tensor):
        """[B,T,D] actions + LAM z_rep [B,1,latent] → (global, time, hand) @ token_dim."""
        z_rep = z_rep.to(dtype=actions.dtype)
        return self.encoder(actions, z_rep)

    def decode(self, global_tok, time_tok, hand_tok) -> torch.Tensor:
        """Action reconstruction from latent tokens (V4-compatible signature)."""
        g = global_tok if (global_tok is not None and global_tok.shape[1] > 0) else None
        h = hand_tok if (hand_tok is not None and hand_tok.shape[1] > 0 and self.hand_in_recon) else None
        return self.recon_decoder(time_tok, global_tokens=g, hand_tokens=h)

    def decode_pixel(self, time_tok: torch.Tensor, frame0: torch.Tensor) -> torch.Tensor:
        """Reconstruct future-frame pixels from time latents + frame0 video."""
        return self.pixel_decoder(time_tok, frame0)

    def forward(self, batch: dict = None, **kwargs) -> dict:
        if batch is None:
            batch = kwargs
        actions = batch["action"]
        actions = actions.to(dtype=self.encoder.action_proj.weight.dtype)
        z_rep = batch["z_rep"].to(dtype=actions.dtype)
        device = actions.device

        global_tok, time_tok, hand_tok = self.encode(actions, z_rep)

        # Loss 1: action reconstruction
        if self.lambda_recon > 0:
            recon = self.decode(global_tok, time_tok, hand_tok)
            loss_recon = self._recon_loss_fn(recon, actions)
        else:
            loss_recon = self._zero(device)

        # Loss 2: future-frame pixel reconstruction (conditioned on time latents)
        if self.lambda_pixel > 0 and self.pixel_decoder is not None:
            frame0 = batch["frame0"].to(dtype=actions.dtype)
            frame1 = batch["frame1"]
            frame1_hat = self.decode_pixel(time_tok, frame0)
            loss_pixel = self._pixel_loss_fn(frame1_hat, frame1)
        else:
            loss_pixel = self._zero(device)

        loss = self.lambda_recon * loss_recon + self.lambda_pixel * loss_pixel

        out = {
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_pixel": loss_pixel,
        }

        # Loss 3: VAE KL (only when the encoder is a VAE). The encoder stashes the
        # KL during encode; here we weight and add it. (Same plumbing as V4.)
        if self.use_vae and self.encoder._last_kl is not None:
            loss_kl = self.encoder._last_kl
            if self.lambda_kl > 0:
                loss = loss + self.lambda_kl * loss_kl
            out["loss"] = loss
            out["loss_kl"] = loss_kl

        return out

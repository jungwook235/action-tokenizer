"""Frozen LAM latent-action extractor (visual source for the V5 tokenizer).

Mirrors the API/behavior of :class:`gr00t.utils.dino.DINOv3FeatureExtractor` and
:class:`gr00t.utils.vggt_feature.VGGTFeatureExtractor` so the V5 action tokenizer
can use DreamDojo's Latent Action Model (LAM) as its frozen visual feature source.
Like those extractors, this is:

  - frozen params, permanent eval mode, no-grad forward,
  - returns the LAM ``z_rep`` latent-action token(s) ``[B, T-1, latent_dim]`` from a
    ``(frame0, frame1, ...)`` video tensor (the analog of per-frame DINO/VGGT feats,
    but already a motion/transition summary).

Unlike DINO/VGGT (HF-hosted), LAM lives in the in-repo DreamDojo source tree at
``<repo>/DreamDojo`` and is loaded from a local Lightning checkpoint. We inject the
DreamDojo root onto ``sys.path`` before importing (exactly as ``DreamDojo/infer.py``
does) so ``from external.lam.model import LAM`` resolves.

Only the LAM **encoder** path (``encoder`` + ``action_prompt`` + ``fc``) is used; the
decoder / ``patch_up`` / ``action_up`` are dropped after load to free memory (the V5
*trainable* pixel decoder is a separate module initialized from the same checkpoint).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _import_lam(dreamdojo_root: Path):
    """Import the inner ``LatentActionModel`` from the in-repo DreamDojo source tree.

    ``<dreamdojo_root>`` (which contains ``external/lam/...``) is prepended to
    ``sys.path`` so ``import external.lam.modules.lam`` resolves.

    NOTE: we import the inner ``LatentActionModel`` (pure ``nn.Module``) directly,
    NOT the ``LAM`` LightningModule wrapper (``external.lam.model``) — the wrapper
    pulls in ``lightning``, which is not a dependency of this project, and we only
    need the encoder path that produces ``z_rep``.
    """
    root = str(dreamdojo_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from external.lam.modules.lam import LatentActionModel  # noqa: E402

    return LatentActionModel


# Default HF location of the pretrained LAM checkpoint (DreamDojo docs:
# "Our latent action model weights can be found at huggingface.co/nvidia/DreamDojo").
LAM_HF_REPO = "nvidia/DreamDojo"
LAM_HF_FILENAME = "LAM_400k.ckpt"


def resolve_lam_ckpt(
    ckpt_path: str = None,
    hf_repo: str = LAM_HF_REPO,
    hf_filename: str = LAM_HF_FILENAME,
) -> str:
    """Return a local path to the LAM checkpoint, downloading from HF if absent.

    Mirrors the DINO/VGGT auto-download UX (``from_pretrained``): if ``ckpt_path``
    points at an existing local file it is used as-is; otherwise the checkpoint is
    pulled from the ``nvidia/DreamDojo`` HF repo and cached under ``HF_HOME``.
    """
    import os

    if ckpt_path and os.path.exists(ckpt_path):
        return ckpt_path
    from huggingface_hub import hf_hub_download

    print(
        f"[lam] local checkpoint {ckpt_path!r} not found; downloading "
        f"{hf_repo}/{hf_filename} from Hugging Face..."
    )
    return hf_hub_download(repo_id=hf_repo, filename=hf_filename)


class LAMFeatureExtractor(nn.Module):
    """Frozen LAM extractor returning latent-action tokens ``[B, T-1, latent_dim]``.

    Always runs in eval mode with no gradients.

    Args:
        ckpt_path: path to the pretrained LAM Lightning checkpoint (e.g.
            ``checkpoints/DreamDojo/LAM_400k.ckpt``).
        model_dim / latent_dim / patch_size / enc_blocks / dec_blocks / num_heads /
            image_channels: LAM hyperparameters (defaults match ``DreamDojo/infer.py``).
        image_h / image_w: expected input frame size; frames are resized here if
            needed (LAM was trained at 240x320).
        dreamdojo_root: path to the DreamDojo source tree (defaults to ``<repo>/DreamDojo``).
    """

    def __init__(
        self,
        ckpt_path: str,
        model_dim: int = 1024,
        latent_dim: int = 32,
        patch_size: int = 16,
        enc_blocks: int = 24,
        dec_blocks: int = 24,
        num_heads: int = 16,
        image_channels: int = 3,
        image_h: int = 240,
        image_w: int = 320,
        dreamdojo_root: str = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = latent_dim  # parity with DINO/VGGT extractors
        self.patch_size = patch_size
        self.image_h = image_h
        self.image_w = image_w

        if dreamdojo_root is None:
            repo_root = Path(__file__).resolve().parents[2]  # utils -> gr00t -> repo root
            dreamdojo_root = repo_root / "DreamDojo"
        LatentActionModel = _import_lam(Path(dreamdojo_root))

        # Auto-download from HF if the local path is absent (DINO/VGGT-style UX).
        ckpt_path = resolve_lam_ckpt(ckpt_path)

        print(f"Loading LAM (LatentActionModel) from {ckpt_path} (latent_dim={latent_dim})...")
        model = LatentActionModel(
            in_dim=image_channels,
            model_dim=model_dim,
            latent_dim=latent_dim,
            patch_size=patch_size,
            enc_blocks=enc_blocks,
            dec_blocks=dec_blocks,
            num_heads=num_heads,
        )
        # The pretrained LAM is a Lightning module whose inner autoencoder is stored
        # under the "lam." prefix; load that subtree directly (so we never import
        # `lightning`). We only need the encoder path, so strict=False tolerates the
        # decoder-side keys we drop next.
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        inner = {k[len("lam."):]: v for k, v in sd.items() if k.startswith("lam.")}
        missing, unexpected = model.load_state_dict(inner, strict=False)
        enc_missing = [
            m for m in missing
            if m.startswith("encoder.") or m.startswith("fc.") or m == "action_prompt"
        ]
        assert not enc_missing, f"LAM encoder weights missing from ckpt: {enc_missing[:8]}"
        print(
            f"[lam] loaded {len(inner)} tensors from ckpt; encoder/fc/action_prompt "
            f"complete (other_missing={len(missing) - len(enc_missing)}, unexpected={len(unexpected)})"
        )
        self.model = model

        # Drop the decoder-side modules — only the encoder path produces z_rep.
        self.model.decoder = None
        self.model.patch_up = None
        self.model.action_up = None

        # Freeze + permanent eval.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True) -> "LAMFeatureExtractor":
        """Override train to always keep the model in eval mode."""
        return super().train(False)

    def _resize(self, videos: torch.Tensor) -> torch.Tensor:
        """[B,T,H,W,C] → [B,T,image_h,image_w,C] (bilinear) if needed."""
        B, T, H, W, C = videos.shape
        if H == self.image_h and W == self.image_w:
            return videos
        x = videos.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)
        x = F.interpolate(x, size=(self.image_h, self.image_w), mode="bilinear", align_corners=False)
        x = x.reshape(B, T, C, self.image_h, self.image_w).permute(0, 1, 3, 4, 2)
        return x.contiguous()

    @torch.inference_mode()
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """videos ``[B, T, H, W, C]`` in ``[0,1]`` → z_rep ``[B, T-1, latent_dim]`` (fp32).

        For the V5 tokenizer T=2 (a single ``(frame0, frame1)`` pair) → ``[B, 1, latent_dim]``.
        """
        videos = videos.float()
        videos = self._resize(videos)
        B, T = videos.shape[:2]
        outputs = self.model({"videos": videos})
        z_rep = outputs["z_rep"]                 # [B, T-1, 1, latent_dim]
        return z_rep.reshape(B, T - 1, self.latent_dim).float()

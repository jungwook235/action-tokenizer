"""Action Latent Tokenizer V4 — RLA-DINO hybrid (self-contained).

V4 fuses the V3 action autoencoder with ``rla-wm``'s DINO inverse-dynamics
autoencoder. The V3 action encoder produces per-timestep action latents; these
**replace RLA's learnable query tokens** and are concatenated with DINO
feature-difference tokens (``x1 - x0``) inside a shared RLA-style fusion encoder
(``SimpleTokenTransformer``). The fusion encoder's ``out_layer`` already
bottlenecks the latent to ``token_dim`` (default 64). The single 64-dim latent is
then fed to BOTH:

  * an action reconstruction decoder (``ReconDecoderV4``, V3-style), and
  * a DINO future-feature decoder (``SimpleTokenTransformer``) that predicts the
    future-frame DINO features from the current-frame features + latent.

Two losses are applied with independent weights ``lambda_recon`` / ``lambda_dino``
(default 1.0 each). The DINO loss type is selectable (``l1`` / ``mse`` /
``cosine`` and ``+``-joined combinations).

**Self-contained by design:** this file does NOT import V2/V3 tokenizer classes
(to avoid cross-version coupling). The needed building blocks (``Transformer``,
``PositionalEmbeddingAdder``, ``LinearHead``, ...) are copied here from V2. Only
the RLA modules are imported from :mod:`gr00t.model.rla_modules` (themselves a
verbatim copy of the rla-wm modules).

The tokenizer keeps the V2/V3 interface so the wrapper / VLA code is reused:
``encode(...) -> (global, time, hand)`` and ``decode(global, time, hand) ->
actions``. Unlike V2/V3, ``encode`` additionally consumes DINO features
(``x0_feat`` / ``x1_feat``) because the latent is DINO-dependent (R2 design).

DINO feature extraction lives OUTSIDE this module (trainer- or wrapper-owned,
frozen) so the V4 state_dict stays clean and wrapper-loadable.

**Optional segment (SAM3 cutout) DINO stream** (``use_seg_stream``, default off): the
same frozen extractor is additionally run on the cutout video's frames at the SAME two
steps, and that stream's feature difference (``s1 - s0``) is concatenated side-by-side
with the RGB difference along the token axis before the fusion encoder
([B,Lp,C] → [B,2Lp,C]). Both halves land in the discarded (visual) part of the fusion
output, so the latent shape and every downstream consumer are unchanged. A twin
``seg_dino_decoder`` (identical to ``dino_decoder``, but predicting the cutout stream's
future features from its own current features + the latent) can be added with its own
loss weight ``lambda_dino_seg``. Both are additive and flag-gated: with the flags off no
parameters, buffers, losses or code paths change.

``seg_dino_decoder_input`` selects what the seg decoder conditions on: ``"seg"``
(default, unchanged — the cutout stream's own ``s0_feat``) or ``"rgb"`` (the RAW image
``x0_feat``, i.e. both decoders read the SAME visual context and differ only in their
target: ``dino_decoder`` predicts ``x1_feat``, ``seg_dino_decoder`` predicts
``s1_feat``). Shapes are identical either way, so this changes no parameters — only
which tensor is fed in. The encoder / seg stream concat is untouched.

Shape trace (B, T=action_horizon, D=action_dim, Lp=DINO patches, action width
``emb_dim``=256, fusion width ``dino_dim``=1024, ``token_dim``=64, Ng=Nh=0):
  actions [B,T,D]; dino_diff = x1-x0 [B,Lp,1024]
  action_encoder -> t256 [B,T,256]
  fusion(x=dino_diff, tokens=t256): input_layer 1024->1024, token_proj 256->1024,
    cat [B, T+Lp, 1024] -> transformer -> norm_out -> out_layer 1024->64
    -> tokens_out = out[:, :T] = t64 [B,T,64]
  action decode: input_up_proj 64->256 -> transformer -> head -> [B,T,D]
  dino decode:  x0_feat [B,Lp,1024] + token_proj(t64) -> [B,Lp,1024] = pred x1
"""

from functools import partial
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from gr00t.model.rla_modules import SimpleTokenTransformer


def _str_to_byte_tensor(s: str) -> torch.Tensor:
    """Encode a string as a uint8 buffer (same pattern as ``masking._mask_mode_bytes``)."""
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.uint8)


def byte_tensor_to_str(t: torch.Tensor) -> str:
    """Decode a uint8 buffer produced by :func:`_str_to_byte_tensor`."""
    return bytes(t.tolist()).decode("utf-8")


# =====================================================================
# Building blocks copied verbatim from action_latent_tokenizer_v2.py
# (kept local so V4 does not depend on V2/V3)
# =====================================================================


class Fp32LayerNorm(nn.LayerNorm):
    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class GatedMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(2 * (hidden_features or in_features) / 3)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x)) * self.fc3(x)))


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=False, proj_drop=0.0,
                 qk_norm=True, norm_layer=partial(Fp32LayerNorm, bias=False, elementwise_affine=False)):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        if qk_norm:
            self.q_norm = norm_layer(self.head_dim)
            self.k_norm = norm_layer(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x, block_mask=None, **kwargs):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = einops.rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = einops.rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = einops.rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask)
        x = einops.rearrange(x, "b h n d -> b n (h d)")
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads=None, head_dim=None, mlp_ratio=4.0,
                 qkv_bias=False, proj_bias=False, mlp_bias=False, drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=partial(Fp32LayerNorm, bias=False, elementwise_affine=False),
                 gated_mlp=False, qk_norm=False, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        num_heads = num_heads or dim // head_dim
        self.attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
                                  proj_drop=drop, qk_norm=qk_norm, norm_layer=norm_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer = GatedMlp if gated_mlp else Mlp
        self.mlp = mlp_layer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, bias=mlp_bias, drop=drop)

    def forward(self, x, block_mask=None, **kwargs):
        x = x + self.drop_path(self.attn(self.norm1(x), block_mask=block_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, dim=768, depth=12, head_dim=64, mlp_ratio=4.0, qkv_bias=False,
                 proj_bias=False, mlp_bias=False, drop=0.0, drop_path_rate=0.0,
                 act_layer=nn.SiLU, norm_layer=partial(Fp32LayerNorm, bias=False, elementwise_affine=False),
                 gated_mlp=True, qk_norm=True, weight_init_style="xavier"):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=dim, head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  proj_bias=proj_bias, mlp_bias=mlp_bias, drop=drop, drop_path=dpr[i],
                  act_layer=act_layer, norm_layer=norm_layer, gated_mlp=gated_mlp, qk_norm=qk_norm)
            for i in range(depth)
        ])
        self.weight_init_style = weight_init_style
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.weight_init_style == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, block_mask=None):
        for block in self.blocks:
            x = block(x, block_mask=block_mask)
        return x


def build_1d_sincos_posemb(L, embed_dim=1024, temperature=10000.0):
    assert embed_dim % 2 == 0
    pos = torch.arange(L, dtype=torch.float32)
    dim_t = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (temperature ** (dim_t / (embed_dim // 2)))
    out = torch.einsum("n,d->nd", pos, omega)
    pos_emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return pos_emb.transpose(0, 1).unsqueeze(0).contiguous()


class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_sizes: list[int]):
        super().__init__()
        assert len(max_sizes) == 1
        self.posembs = nn.Parameter(build_1d_sincos_posemb(max_sizes[0], embed_dim=dim), requires_grad=False)

    def forward(self, shape: list[int]) -> torch.Tensor:
        return self.posembs[:, :, :shape[0]]


class PositionalEmbeddingAdder(nn.Module):
    def __init__(self, dim: int, max_sizes: list[int]):
        super().__init__()
        assert len(max_sizes) == 1
        self.posembs = nn.Parameter(build_1d_sincos_posemb(max_sizes[0], embed_dim=dim), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        pe = einops.rearrange(self.posembs[:, :, :L], "1 D L -> 1 L D")
        return x + pe


class LinearHead(nn.Module):
    def __init__(self, dim: int, dim_out: int, weight_init_style: str = "zero"):
        super().__init__()
        self.norm = Fp32LayerNorm(dim, bias=False, elementwise_affine=False)
        self.proj = nn.Linear(dim, dim_out, bias=True)
        if weight_init_style == "zero":
            nn.init.constant_(self.proj.weight, 0)
        elif weight_init_style == "xavier":
            nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class LinearLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, weight_init_style: str = "xavier"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=True)
        if weight_init_style == "xavier":
            nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# =====================================================================
# V4 Action encoder (= V2 TimeWiseEncoder, copied; no internal bottleneck)
# =====================================================================


class ActionEncoderV4(nn.Module):
    """Action chunk → per-timestep latent tokens (+ optional global/hand tokens).

    Identical to V2 ``TimeWiseEncoder`` (no bottleneck): the V4 bottleneck lives in
    the fusion encoder's ``out_layer``. Output width is ``emb_dim`` (256).
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        emb_dim: int,
        head_dim: int,
        depth: int,
        pdropout: float,
        num_global_tokens: int = 0,
        num_hand_tokens: int = 0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.emb_dim = emb_dim
        self.num_global_tokens = num_global_tokens
        self.num_hand_tokens = num_hand_tokens

        self.action_proj = nn.Linear(action_dim, emb_dim)
        self.time_pos_emb = PositionalEmbeddingAdder(emb_dim, max_sizes=[action_horizon])

        self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, emb_dim)) if num_global_tokens > 0 else None
        self.hand_tokens = nn.Parameter(torch.randn(num_hand_tokens, emb_dim)) if num_hand_tokens > 0 else None

        self.transformer = Transformer(dim=emb_dim, depth=depth, head_dim=head_dim, drop=pdropout)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        Ng = self.num_global_tokens
        Nh = self.num_hand_tokens

        x = self.action_proj(x)
        x = self.time_pos_emb(x)

        parts = []
        if Ng > 0:
            parts.append(self.global_tokens.unsqueeze(0).expand(B, -1, -1))
        parts.append(x)
        if Nh > 0:
            parts.append(self.hand_tokens.unsqueeze(0).expand(B, -1, -1))
        x = torch.cat(parts, dim=1)

        x = self.transformer(x)

        global_out = x[:, :Ng]
        time_out = x[:, Ng:Ng + T]
        hand_out = x[:, Ng + T:]
        return global_out, time_out, hand_out


# =====================================================================
# V4 fusion encoder: action latents (queries) + DINO-diff → 64-dim latent
# =====================================================================


class TimeWiseEncoderV4(nn.Module):
    """Joint (fusion) encoder.

    Pipeline: ``ActionEncoderV4`` produces action latents @ ``emb_dim`` (256).
    These act as the RLA query tokens (replacing learnable tokens) and are fed,
    together with the DINO feature-difference sequence, into an RLA
    ``SimpleTokenTransformer`` whose ``out_layer`` maps the fusion width
    (``dino_dim``=1024) down to ``token_dim`` (64) — i.e. the bottleneck. The
    readout takes the action-token positions only.

    Exposes the attributes the wrapper expects: ``action_dim``, ``action_horizon``,
    ``emb_dim`` (transformer working width, 256), ``token_dim`` (output latent
    width, 64), ``num_global_tokens``, ``num_hand_tokens``, and an ``action_proj``
    property aliasing the action encoder's projection (used for dtype probing).
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        emb_dim: int = 256,
        head_dim: int = 64,
        encoder_depth: int = 4,
        pdropout: float = 0.0,
        num_global_tokens: int = 0,
        num_hand_tokens: int = 0,
        dino_dim: int = 1024,
        fusion_width: int = 1024,
        fusion_depth: int = 12,
        fusion_heads: int = 16,
        token_dim: int = 64,
        use_vae: bool = False,
        vae_sample: bool = True,
        kl_free_bits: float = 0.0,
        action_proj_mlp: bool = False,
        action_proj_hidden: Optional[int] = None,
        use_embodiment_class_token: bool = False,
        use_seg_stream: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.emb_dim = emb_dim
        self.token_dim = token_dim
        self.dino_dim = dino_dim
        self.num_global_tokens = num_global_tokens
        self.num_hand_tokens = num_hand_tokens

        # ---- segment (SAM3 cutout) DINO stream (opt-in) ----
        # When enabled, ``forward`` additionally receives the cutout stream's DINO
        # feature difference (s1 - s0) and concatenates it side-by-side with the RGB
        # ``dino_diff`` along the token axis before the fusion transformer:
        # [B, Lp, dino_dim] → [B, 2*Lp, dino_dim]. Both streams land in the discarded
        # (visual) half of the fusion output, so the kept action-token positions — and
        # therefore the latent shape / every downstream consumer — are unchanged.
        # This adds NO parameters or buffers (the fusion ``input_layer`` is shared by
        # both streams), so the state_dict stays byte-identical to standard V4; the
        # opt-in is recorded on the tokenizer as a ``_use_seg_stream`` marker so the
        # inference wrapper knows ``encode`` requires the seg features.
        self.use_seg_stream = bool(use_seg_stream)

        # ---- per-embodiment (data-type) class token (opt-in; Stage-2 side) ----
        # Set for tokenizers trained (jointly) with per-embodiment class tokens. At
        # Stage-2 this holds THIS embodiment's single [dino_dim] token (loaded from the
        # remapped checkpoint as a frozen buffer) and is prepended to ``dino_diff``
        # before the fusion encoder, matching Stage-1. Default False registers no
        # buffer, so the state_dict stays byte-identical to standard V4 checkpoints.
        self.use_embodiment_class_token = bool(use_embodiment_class_token)
        if self.use_embodiment_class_token:
            self.register_buffer("embodiment_class_token", torch.zeros(dino_dim))

        # ---- SD-style VAE bottleneck (opt-in) ----
        # When ``use_vae`` is False this branch adds NO parameters/buffers and the
        # forward path is byte-identical to the deterministic V4. When True, the
        # fusion output is treated as the posterior mean μ and a single linear head
        # predicts logσ²; ``forward`` reparameterizes z = μ + σ·ε and stashes the
        # KL(N(0,I)) term in ``self._last_kl`` for the tokenizer loss. The fusion
        # ``out_layer`` is unchanged (still → token_dim), so latent dim / decoder /
        # downstream shapes are all unaffected.
        self.use_vae = bool(use_vae)
        # Sampling toggle (only meaningful when use_vae). True (default) → the
        # encoder reparameterizes z = μ + σ·ε (existing behavior, byte-identical
        # path). False → the encoder returns the posterior mean μ directly
        # (deterministic latent) while STILL computing KL, so the logvar head keeps
        # training and stays in the graph (DDP-safe). Stored as a plain attribute
        # (not a buffer) so the ON default leaves the state_dict unchanged.
        self.vae_sample = bool(vae_sample)
        self.kl_free_bits = float(kl_free_bits)
        self.kl_logvar_min = -8.0
        self.kl_logvar_max = 8.0
        self._last_kl = None
        if self.use_vae:
            self.logvar_head = nn.Linear(token_dim, token_dim)
            # Start near-deterministic (σ≈0.08) so VAE training begins from ~the same
            # point as the AE; KL/recon then move logvar to equilibrium.
            nn.init.zeros_(self.logvar_head.weight)
            nn.init.constant_(self.logvar_head.bias, -5.0)

        self.action_encoder = ActionEncoderV4(
            action_dim=action_dim,
            action_horizon=action_horizon,
            emb_dim=emb_dim,
            head_dim=head_dim,
            depth=encoder_depth,
            pdropout=pdropout,
            num_global_tokens=num_global_tokens,
            num_hand_tokens=num_hand_tokens,
        )

        # RLA fusion encoder. num_tokens=0 → no internal learnable tokens; the
        # action latents are injected as external tokens (token_channels=emb_dim).
        # out_channels=token_dim → the out_layer IS the bottleneck (norm_out is
        # the pre-bottleneck LayerNorm).
        # ``action_proj_mlp`` (opt-in) turns the action-token projection (emb_dim →
        # fusion_width, applied just before the DINO-feature concat) from a single
        # Linear into a 2-layer MLP (Linear → GELU → Linear); ``action_proj_hidden``
        # sets the hidden width (defaults to fusion_width). Default (False) is
        # byte-identical to the original single-Linear fusion. Only the fusion
        # projection is affected — the DINO decoder keeps its Linear token_proj.
        self.joint = SimpleTokenTransformer(
            in_channels=dino_dim,
            model_channels=fusion_width,
            out_channels=token_dim,
            num_blocks=fusion_depth,
            num_heads=fusion_heads,
            num_tokens=0,
            token_channels=emb_dim,
            use_fp16=False,
            token_proj_mlp=action_proj_mlp,
            token_proj_hidden=action_proj_hidden,
        )

    @property
    def action_proj(self) -> nn.Module:
        # Wrapper/trainer probe ``encoder.action_proj.weight.dtype``.
        return self.action_encoder.action_proj

    def forward(
        self,
        actions: torch.Tensor,
        dino_diff: torch.Tensor,
        seg_diff: Optional[torch.Tensor] = None,
    ):
        B, T, _ = actions.shape
        Ng = self.num_global_tokens
        Nh = self.num_hand_tokens

        g256, t256, h256 = self.action_encoder(actions)          # [B,*,256]
        act_tokens = torch.cat([g256, t256, h256], dim=1)         # [B, Ng+T+Nh, 256]

        if self.use_seg_stream:
            assert seg_diff is not None, (
                "use_seg_stream=True but no seg_diff was passed to the fusion encoder "
                "(pass s0_feat/s1_feat to encode())."
            )
            # Side-by-side concat along the token axis: [B, Lp + Lp_seg, dino_dim].
            dino_diff = torch.cat([dino_diff, seg_diff.to(dtype=dino_diff.dtype)], dim=1)
        else:
            assert seg_diff is None, (
                "seg_diff was passed but this encoder was built with use_seg_stream=False."
            )

        if self.use_embodiment_class_token:
            # Prepend the data-type class token as an extra dino_dim patch. It lands in
            # the discarded (visual) half of the fusion output, so the kept action-token
            # positions below are unchanged in shape.
            ct = self.embodiment_class_token.to(dtype=dino_diff.dtype)  # [dino_dim]
            ct = ct.view(1, 1, -1).expand(dino_diff.shape[0], 1, -1)    # [B,1,dino_dim]
            dino_diff = torch.cat([ct, dino_diff], dim=1)               # [B,1+Lp,dino_dim]

        tokens_out, _ = self.joint(x=dino_diff, tokens=act_tokens)  # [B, Ng+T+Nh, 64]

        if self.use_vae:
            # SD-style VAE: fusion output = posterior mean μ. When ``vae_sample`` is
            # True (default) reparameterize z = μ + σ·ε (runs in train/frozen/eval
            # alike, so the VLA target is a sample z, matching SD latent-diffusion
            # practice). When False, return μ directly (deterministic latent). Either
            # way KL is still computed below, so the logvar head keeps training and
            # stays in the graph (DDP-safe). The choice is recorded as a checkpoint
            # marker so Stage-2 / inference inherit it identically.
            mu = tokens_out
            logvar = self.logvar_head(mu).clamp(self.kl_logvar_min, self.kl_logvar_max)
            if self.vae_sample:
                std = torch.exp(0.5 * logvar)
                tokens_out = mu + torch.randn_like(std) * std
            else:
                tokens_out = mu
            # Per-dim KL to N(0,I), averaged over batch+tokens; optional free-bits
            # floors each dim so it cannot collapse below the budget.
            kl_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # [B,N,token_dim]
            kl_dim = kl_dim.mean(dim=(0, 1))                           # [token_dim]
            if self.kl_free_bits > 0:
                kl_dim = torch.clamp(kl_dim, min=self.kl_free_bits)
            self._last_kl = kl_dim.sum()
        else:
            self._last_kl = None

        global_out = tokens_out[:, :Ng]
        time_out = tokens_out[:, Ng:Ng + T]
        hand_out = tokens_out[:, Ng + T:]
        return global_out, time_out, hand_out


# =====================================================================
# V4 action reconstruction decoder (= V3 ReconDecoder + bottleneck up-proj)
# =====================================================================


class ReconDecoderV4(nn.Module):
    """Reconstruction decoder with input up-projection (token_dim → emb_dim).

    Identical to V2 ``ReconDecoder`` plus a single ``input_up_proj`` applied to
    each latent stream before the V2 pipeline (which runs at ``emb_dim``=256).
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        emb_dim: int,
        head_dim: int,
        depth: int,
        pdropout: float,
        decoder_mode: str = "self_attention",
        num_global_tokens: int = 0,
        num_hand_tokens: int = 0,
        token_dim: int = 64,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.emb_dim = emb_dim
        self.token_dim = token_dim
        self.decoder_mode = decoder_mode
        self.num_global_tokens = num_global_tokens
        self.num_hand_tokens = num_hand_tokens

        if token_dim != emb_dim:
            self.input_up_proj = nn.Linear(token_dim, emb_dim)
        else:
            self.input_up_proj = nn.Identity()

        self.time_proj = LinearLayer(emb_dim, emb_dim)
        self.time_pos_emb = PositionalEmbeddingAdder(emb_dim, max_sizes=[action_horizon])
        self.head = LinearHead(emb_dim, action_dim)

        if num_global_tokens > 0:
            self.global_proj = LinearLayer(emb_dim, emb_dim)
            self.global_pos_emb = PositionalEmbeddingAdder(emb_dim, max_sizes=[num_global_tokens])
        else:
            self.global_proj = None
            self.global_pos_emb = None

        if num_hand_tokens > 0:
            self.hand_proj = LinearLayer(emb_dim, emb_dim)
            self.hand_pos_emb = PositionalEmbeddingAdder(emb_dim, max_sizes=[num_hand_tokens])
        else:
            self.hand_proj = None
            self.hand_pos_emb = None

        if decoder_mode == "cross_attention":
            self.query_pos_emb = PositionalEmbedding(emb_dim, max_sizes=[action_horizon])
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=emb_dim, nhead=emb_dim // head_dim,
                    dim_feedforward=4 * emb_dim, dropout=pdropout,
                    activation="gelu", batch_first=True, norm_first=True,
                ),
                num_layers=depth,
            )
        elif decoder_mode == "self_attention":
            self.transformer = Transformer(dim=emb_dim, depth=depth, head_dim=head_dim, drop=pdropout)
        else:
            raise ValueError(f"Unknown decoder_mode: {decoder_mode}")

    def forward(
        self,
        time_tokens: torch.Tensor,
        global_tokens: Optional[torch.Tensor] = None,
        hand_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        time_tokens = self.input_up_proj(time_tokens)
        if global_tokens is not None and global_tokens.numel() > 0:
            global_tokens = self.input_up_proj(global_tokens)
        if hand_tokens is not None and hand_tokens.numel() > 0:
            hand_tokens = self.input_up_proj(hand_tokens)

        B, T, E = time_tokens.shape

        parts = []
        if global_tokens is not None and self.global_proj is not None:
            g = self.global_pos_emb(self.global_proj(global_tokens))
            parts.append(g)

        latents = self.time_pos_emb(self.time_proj(time_tokens))
        parts.append(latents)

        if hand_tokens is not None and self.hand_proj is not None:
            h = self.hand_pos_emb(self.hand_proj(hand_tokens))
            parts.append(h)

        memory = torch.cat(parts, dim=1)
        Ng = global_tokens.shape[1] if global_tokens is not None and self.global_proj is not None else 0

        if self.decoder_mode == "cross_attention":
            query = self.query_pos_emb(shape=[T])
            query = einops.rearrange(query, "1 E T -> 1 T E").expand(B, -1, -1)
            out = self.decoder(query, memory)
        else:
            out = self.transformer(memory)
            out = out[:, Ng:Ng + T]

        return self.head(out)


# =====================================================================
# Unified decoders (opt-in via --decoder-arch; "separate" = everything above,
# untouched — no new modules, buffers or code paths when the flag is off)
# =====================================================================


class MoTBlockV4(nn.Module):
    """Mixture-of-Transformers block over the combined [latent, patch] sequence.

    Attention is fully SHARED (one set of QKV/output projections, global softmax
    under the caller-supplied asymmetric mask); the FFN and pre-norms exist in TWO
    copies and each token is routed to its group's copy (latent group = the first
    ``n_lat`` positions, patch group = the rest). The pre-norms here are non-affine
    (like ``Block``'s), so duplicating them adds no parameters — the per-group
    capacity lives in the twin GatedMlps.
    """

    def __init__(
        self,
        dim,
        head_dim=64,
        mlp_ratio=4.0,
        drop=0.0,
        norm_layer=partial(Fp32LayerNorm, bias=False, elementwise_affine=False),
    ):
        super().__init__()
        self.norm1_lat = norm_layer(dim)
        self.norm1_pat = norm_layer(dim)
        self.attn = SelfAttention(
            dim, num_heads=dim // head_dim, qkv_bias=False, proj_bias=False,
            proj_drop=drop, qk_norm=True, norm_layer=norm_layer,
        )
        self.norm2_lat = norm_layer(dim)
        self.norm2_pat = norm_layer(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp_lat = GatedMlp(dim, hidden, act_layer=nn.SiLU, bias=False, drop=drop)
        self.mlp_pat = GatedMlp(dim, hidden, act_layer=nn.SiLU, bias=False, drop=drop)

    @staticmethod
    def _per_group(x, n_lat, fn_lat, fn_pat):
        if x.shape[1] == n_lat:  # latent-only decode: no patch group present
            return fn_lat(x)
        return torch.cat([fn_lat(x[:, :n_lat]), fn_pat(x[:, n_lat:])], dim=1)

    def forward(self, x, n_lat, block_mask=None):
        h = self._per_group(x, n_lat, self.norm1_lat, self.norm1_pat)
        x = x + self.attn(h, block_mask=block_mask)
        h = self._per_group(
            x, n_lat,
            lambda t: self.mlp_lat(self.norm2_lat(t)),
            lambda t: self.mlp_pat(self.norm2_pat(t)),
        )
        return x + h


class UnifiedDecoderV4(nn.Module):
    """Unified action + DINO-feature decoder over one combined sequence X=[L, P].

    L = latent tokens (global+time+hand @ token_dim → ``width`` up-projection +
    1D sincos positional embedding); P = current-frame DINO patches
    (dino_dim → ``width`` projection, no pos-emb — mirroring the separate
    ``dino_decoder``'s convention). No learnable query tokens.

    Asymmetric attention mask (unless ``recon_sees_vision``): L rows attend ONLY
    L columns (vision never leaks into the latent positions, so the action
    reconstruction — read from L's time positions — is a function of the latent
    alone and ``decode`` works WITHOUT images); P rows attend everything (the
    future-feature prediction is action-conditioned).

    arch="shared_trunk": ``trunk_depth`` masked shared layers → a
      ``branch_depth``-layer recon branch on L′ only (action head at the time
      positions) and a ``branch_depth``-layer dino branch on the full [L′, P′]
      (bidirectional, no mask; feature head at the P positions).
    arch="mot": ``mot_depth`` MoTBlockV4 layers (shared attention with the same
      mask, per-group FFN/norms), heads read the final sequence directly.
    arch="shared_trunk_vis": shared_trunk with the recon branch and the action head
      REMOVED — the trunk carries the VISUAL branches only (dino [+ segpix]). The
      action path is not part of this module at all: the tokenizer keeps its
      ordinary separate ``ReconDecoderV4`` and decodes actions from the latent
      there. Everything else (latent embedding, patch projection, the asymmetric
      mask, trunk/branch depths) is identical to "shared_trunk", so the only
      difference against it is the missing recon branch (EXP-0011).

    Both heads are zero-init (``LinearHead`` "zero"), matching the separate
    decoders' output-layer convention.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        token_dim: int = 64,
        dino_dim: int = 1024,
        width: int = 1024,
        head_dim: int = 64,
        arch: str = "shared_trunk",
        trunk_depth: int = 4,
        branch_depth: int = 2,
        mot_depth: int = 6,
        pdropout: float = 0.0,
        num_global_tokens: int = 0,
        num_hand_tokens: int = 0,
        recon_sees_vision: bool = False,
        use_segpix_branch: bool = False,
    ):
        super().__init__()
        assert arch in ("shared_trunk", "mot", "shared_trunk_vis"), (
            f"unknown decoder arch {arch!r}"
        )
        assert not (use_segpix_branch and arch not in ("shared_trunk", "shared_trunk_vis")), (
            "the segpix branch is only supported with --decoder-arch shared_trunk "
            "/ shared_trunk_vis (mot + seg-pixel is explicitly unsupported for now)"
        )
        self.arch = arch
        # Visual-only trunk (EXP-0011): no recon branch, no action head.
        self.visual_only = arch == "shared_trunk_vis"
        assert not (self.visual_only and recon_sees_vision), (
            "recon_sees_vision is meaningless for arch='shared_trunk_vis' (the "
            "action path does not go through this decoder at all)"
        )
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.token_dim = token_dim
        self.dino_dim = dino_dim
        self.width = width
        self.num_global_tokens = num_global_tokens
        self.num_hand_tokens = num_hand_tokens
        self.recon_sees_vision = bool(recon_sees_vision)

        n_lat_max = num_global_tokens + action_horizon + num_hand_tokens
        self.latent_up_proj = nn.Linear(token_dim, width)
        self.latent_pos_emb = PositionalEmbeddingAdder(width, max_sizes=[n_lat_max])
        self.patch_proj = nn.Linear(dino_dim, width)

        if arch in ("shared_trunk", "shared_trunk_vis"):
            self.trunk = Transformer(dim=width, depth=trunk_depth, head_dim=head_dim, drop=pdropout)
            if not self.visual_only:
                self.recon_branch = Transformer(dim=width, depth=branch_depth, head_dim=head_dim, drop=pdropout)
            self.dino_branch = Transformer(dim=width, depth=branch_depth, head_dim=head_dim, drop=pdropout)
        else:
            self.layers = nn.ModuleList(
                [MoTBlockV4(width, head_dim=head_dim, drop=pdropout) for _ in range(mot_depth)]
            )
            # xavier init (MoT blocks are not Transformer instances, so they miss
            # its _init_weights — replicate it here)
            for m in self.layers.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # visual_only: no action head at all (the separate ReconDecoderV4 owns the
        # action path), so DDP never sees an unused parameter here.
        if not self.visual_only:
            self.action_head = LinearHead(width, action_dim, weight_init_style="zero")
        self.feat_head = LinearHead(width, dino_dim, weight_init_style="zero")
        for m in (self.latent_up_proj, self.patch_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

        # Third (seg-pixel) branch — built LAST on purpose so that, under the same
        # seed, every module above gets identical init with the branch on or off
        # (lets the recon/dino paths be verified unchanged at the forward level).
        # Same structure as the dino branch: branch_depth layers over the full
        # [L', P'], no mask; the pixel-block head lives on the tokenizer
        # (seg_pixel_head), mirroring the separate seg_pixel_decoder split.
        self.use_segpix_branch = bool(use_segpix_branch)
        if self.use_segpix_branch:
            self.segpix_branch = Transformer(
                dim=width, depth=branch_depth, head_dim=head_dim, drop=pdropout
            )

    def _attn_mask(self, n_lat: int, n_all: int, device) -> Optional[torch.Tensor]:
        """[n_all, n_all] bool mask (True = may attend). None when unmasked."""
        if self.recon_sees_vision or n_all == n_lat:
            return None
        m = torch.ones(n_all, n_all, dtype=torch.bool, device=device)
        m[:n_lat, n_lat:] = False  # latent rows must not see patch columns
        return m

    def _embed_latent(self, global_tok, time_tok, hand_tok) -> torch.Tensor:
        parts = [
            t for t in (global_tok, time_tok, hand_tok)
            if t is not None and t.shape[1] > 0
        ]
        lat = torch.cat(parts, dim=1)  # [B, Ng+T+Nh, token_dim]
        return self.latent_pos_emb(self.latent_up_proj(lat))

    def forward(self, global_tok, time_tok, hand_tok, x0_feat=None):
        """→ (action_pred [B,T,action_dim], x1_feat_pred [B,Lp,dino_dim] | None,
             segpix_visuals [B,Lp,width] | None).

        ``x0_feat=None`` is the latent-only decode path (valid because of the
        asymmetric mask; invalid with ``recon_sees_vision``): returns
        (action_pred, None, None). segpix_visuals is the RAW segpix-branch output
        at the patch positions (the pixel-block head lives on the tokenizer);
        None unless the branch was built and x0_feat given.

        arch="shared_trunk_vis" has no action path: action_pred is always None and
        x0_feat is REQUIRED (the caller decodes actions from the tokenizer's
        separate recon_decoder instead).
        """
        L = self._embed_latent(global_tok, time_tok, hand_tok)
        n_lat = L.shape[1]
        Ng = global_tok.shape[1] if global_tok is not None else 0
        T = time_tok.shape[1]

        if x0_feat is None:
            assert not self.visual_only, (
                "arch='shared_trunk_vis' produces no action — forward() requires "
                "x0_feat (actions come from the tokenizer's separate recon_decoder)."
            )
            assert not self.recon_sees_vision, (
                "decoder was built with recon_sees_vision=True: decode() requires "
                "x0_feat (the latent positions attend to vision)."
            )
            x = L
        else:
            x = torch.cat([L, self.patch_proj(x0_feat.to(dtype=L.dtype))], dim=1)
        mask = self._attn_mask(n_lat, x.shape[1], x.device)

        if self.visual_only:
            # Visual-only trunk (EXP-0011): same trunk + same asymmetric mask as
            # "shared_trunk" (so the ONLY difference is the removed recon branch),
            # then the visual branches. No action is produced here.
            h = self.trunk(x, block_mask=mask)
            hd = self.dino_branch(h)  # bidirectional refinement, no mask
            feat = self.feat_head(hd[:, n_lat:])
            seg_vis = None
            if self.use_segpix_branch:
                seg_vis = self.segpix_branch(h)[:, n_lat:]
            return None, feat, seg_vis

        if self.arch == "shared_trunk":
            h = self.trunk(x, block_mask=mask)
            hr = self.recon_branch(h[:, :n_lat])
            action = self.action_head(hr[:, Ng:Ng + T])
            if x0_feat is None:
                return action, None, None
            hd = self.dino_branch(h)  # bidirectional refinement, no mask
            feat = self.feat_head(hd[:, n_lat:])
            seg_vis = None
            if self.use_segpix_branch:
                hs = self.segpix_branch(h)  # same shape/mechanics as the dino branch
                seg_vis = hs[:, n_lat:]
            return action, feat, seg_vis

        h = x
        for layer in self.layers:
            h = layer(h, n_lat, block_mask=mask)
        action = self.action_head(h[:, Ng:Ng + T])
        if x0_feat is None:
            return action, None, None
        return action, self.feat_head(h[:, n_lat:]), None


# =====================================================================
# V4 tokenizer
# =====================================================================


_VALID_DINO_TERMS = ("l1", "mse", "cosine")


class ActionLatentTokenizerV4(nn.Module):
    """RLA-DINO hybrid action latent tokenizer (continuous latent).

    Losses: ``loss = lambda_recon * action_recon + lambda_dino * dino_recon``.
    DINO recon supports selectable terms (l1 / mse / cosine, ``+``-joinable) with
    per-term weights.

    Note: ``encode`` requires DINO features (``x0_feat`` / ``x1_feat``); the DINO
    extractor is external (trainer/wrapper-owned, frozen) and not part of this
    module's state_dict.
    """

    def __init__(
        self,
        encoder: TimeWiseEncoderV4,
        recon_decoder: Optional[ReconDecoderV4],
        dino_decoder: Optional[SimpleTokenTransformer] = None,
        unified_decoder: Optional[UnifiedDecoderV4] = None,
        decoder_arch: str = "separate",
        seg_dino_decoder: Optional[SimpleTokenTransformer] = None,
        seg_dino_decoder_input: str = "seg",
        seg_pixel_decoder: Optional[SimpleTokenTransformer] = None,
        seg_pixel_head: Optional[nn.Module] = None,
        seg_pixel_patch: int = 14,
        lambda_seg_pixel: float = 0.0,
        use_mask_weighted_dino_loss: bool = False,
        mask_patch_weight: float = 2.0,
        lambda_recon: float = 1.0,
        lambda_dino: float = 1.0,
        lambda_dino_seg: float = 0.0,
        lambda_kl: float = 0.0,
        recon_loss_type: str = "mse",
        dino_loss_type: str = "l1",
        dino_loss_weights: Optional[dict] = None,
        feature_source: str = "dino",
        vggt_token_source: Optional[str] = None,
        vggt_image_size: Optional[int] = None,
        vggt_model: Optional[str] = None,
        vggt_final_norm: str = "none",
        dino_final_norm: str = "affine",
    ):
        super().__init__()
        self.encoder = encoder
        self.recon_decoder = recon_decoder
        self.dino_decoder = dino_decoder
        self.seg_dino_decoder = seg_dino_decoder

        # ---- unified decoder (opt-in via decoder_arch; default "separate") ----
        # "separate" (default): recon_decoder + dino_decoder exactly as before —
        # unified_decoder is None, NOTHING extra is registered, and every code path
        # below is untouched (byte-identical state_dict and losses).
        # "shared_trunk" / "mot": ONE UnifiedDecoderV4 replaces both decoders; the
        # ``_decoder_arch`` marker lets the inference wrapper rebuild it. The
        # separate-only extras (seg decoders) are out of scope for unified.
        # "shared_trunk_vis" (EXP-0011): the unified decoder is VISUAL-ONLY — it
        # replaces dino_decoder only, and the ordinary ReconDecoderV4 keeps the
        # action path (so recon is bit-for-bit the "separate"/base configuration).
        assert decoder_arch in ("separate", "shared_trunk", "mot", "shared_trunk_vis"), (
            f"unknown decoder_arch {decoder_arch!r}"
        )
        self.decoder_arch = decoder_arch
        self.unified_decoder = unified_decoder
        self._unified_visual_only = decoder_arch == "shared_trunk_vis"
        if decoder_arch == "separate":
            assert unified_decoder is None, "decoder_arch='separate' forbids unified_decoder"
            assert recon_decoder is not None, "decoder_arch='separate' requires recon_decoder"
        elif self._unified_visual_only:
            assert unified_decoder is not None and unified_decoder.arch == decoder_arch, (
                "decoder_arch='shared_trunk_vis' requires a matching UnifiedDecoderV4"
            )
            assert recon_decoder is not None, (
                "decoder_arch='shared_trunk_vis' keeps the SEPARATE action decoder — "
                "pass recon_decoder (only dino_decoder is replaced)"
            )
            assert dino_decoder is None, (
                "the visual-only trunk replaces dino_decoder — pass dino_decoder=None"
            )
            assert seg_dino_decoder is None and seg_pixel_decoder is None, (
                "separate seg decoder modules cannot be combined with a unified "
                "decoder (segpix rides as a branch of the trunk instead)"
            )
            assert encoder.num_global_tokens == 0 and encoder.num_hand_tokens == 0, (
                "unified decoder currently supports Ng=Nh=0 only (the V4 default)"
            )
            self.register_buffer("_decoder_arch", _str_to_byte_tensor(decoder_arch))
        else:
            assert unified_decoder is not None and unified_decoder.arch == decoder_arch, (
                f"decoder_arch={decoder_arch!r} requires a matching UnifiedDecoderV4"
            )
            assert recon_decoder is None and dino_decoder is None, (
                "unified decoder replaces recon_decoder AND dino_decoder — pass both as None"
            )
            assert seg_dino_decoder is None and seg_pixel_decoder is None, (
                "separate seg decoder modules cannot be combined with a unified "
                "decoder (segpix rides as a branch of shared_trunk instead)"
            )
            if decoder_arch == "mot" and seg_pixel_head is not None:
                raise ValueError(
                    "--decoder-arch mot + --use-seg-pixel-decoder is not supported "
                    "(the segpix branch exists only for shared_trunk)."
                )
            assert encoder.num_global_tokens == 0 and encoder.num_hand_tokens == 0, (
                "unified decoder currently supports Ng=Nh=0 only (the V4 default)"
            )
            self.register_buffer("_decoder_arch", _str_to_byte_tensor(decoder_arch))
            if unified_decoder.recon_sees_vision:
                self.register_buffer("_decoder_recon_sees_vision", torch.tensor(True))

        # ---- seg-mask pixel decoder (opt-in, training-only) ----
        # A twin of ``dino_decoder`` (same SimpleTokenTransformer mechanics) that
        # predicts the FUTURE (x1-step) SAM3 union mask at pixel resolution from the
        # current-frame features x0_feat + the latent. ``seg_pixel_head`` maps each
        # patch token to its ``seg_pixel_patch``² pixel-block logits; the blocks tile
        # back to the full [grid*patch, grid*patch] logit map (a linear "pixel
        # shuffle" head — the lightweight upsampler). BCE against the binary mask,
        # weighted by ``lambda_seg_pixel``; samples with no mask npz (mask_valid=0)
        # are excluded from the loss. Default (None) registers NO modules/buffers —
        # state_dict and forward stay byte-identical. Like dino_decoder /
        # seg_dino_decoder this is training-only: the inference wrapper builds it as
        # None and ``from_checkpoint`` filters its keys generically.
        # Unified segpix: with a shared_trunk decoder carrying the segpix branch,
        # only the head is passed (the "decoder" half IS the branch). Everywhere
        # else the original pairing invariant holds unchanged.
        self._unified_segpix = bool(
            unified_decoder is not None and getattr(unified_decoder, "use_segpix_branch", False)
        )
        if self._unified_segpix:
            assert seg_pixel_decoder is None and seg_pixel_head is not None, (
                "unified segpix branch requires seg_pixel_head (and no separate "
                "seg_pixel_decoder)."
            )
        else:
            assert (seg_pixel_decoder is None) == (seg_pixel_head is None), (
                "seg_pixel_decoder and seg_pixel_head must be passed together."
            )
        self.seg_pixel_decoder = seg_pixel_decoder
        self.seg_pixel_head = seg_pixel_head
        self.seg_pixel_patch = int(seg_pixel_patch)
        self.lambda_seg_pixel = float(lambda_seg_pixel)

        # ---- mask-weighted DINO loss (opt-in, loss-only — adds NO params) ----
        # Per-patch weights on the (RGB) DINO future-feature loss: the x1-step SAM3
        # union mask is average-pooled to the DINO patch grid (per-patch coverage
        # c ∈ [0,1]) → w = 1 + (mask_patch_weight − 1)·c → normalized by the batch
        # mean so the total loss scale is unchanged. Samples with no mask npz get
        # w = 1 (before normalization). Default False leaves the loss math
        # byte-identical (the weighted branch in _dino_loss is never taken).
        self.use_mask_weighted_dino_loss = bool(use_mask_weighted_dino_loss)
        self.mask_patch_weight = float(mask_patch_weight)

        # What the seg DINO decoder conditions on: "seg" (default, the cutout stream's
        # own s0_feat — unchanged behavior) or "rgb" (the raw image x0_feat, so both
        # decoders share the same visual context and differ only in their target).
        assert seg_dino_decoder_input in ("seg", "rgb"), (
            f"seg_dino_decoder_input must be 'seg' or 'rgb'; got {seg_dino_decoder_input!r}"
        )
        self.seg_dino_decoder_input = seg_dino_decoder_input
        # Marker only for the non-default mode AND only when the decoder exists, so
        # default checkpoints stay byte-identical and the inference wrapper (which
        # builds seg_dino_decoder=None) never expects this buffer.
        if seg_dino_decoder is not None and seg_dino_decoder_input == "rgb":
            self.register_buffer("_seg_dino_decoder_input", _str_to_byte_tensor("rgb"))

        self.lambda_recon = float(lambda_recon)
        self.lambda_dino = float(lambda_dino)
        self.lambda_dino_seg = float(lambda_dino_seg)
        self.lambda_kl = float(lambda_kl)
        self.recon_loss_type = recon_loss_type

        # Segment-stream flag: the encoder is the single source of truth (it owns the
        # fusion-input concat). When set, record a detection marker so the inference
        # wrapper rebuilds an encoder that expects the seg features. When unset NO
        # buffer is registered → the state_dict stays byte-identical to standard V4.
        self.use_seg_stream = bool(getattr(encoder, "use_seg_stream", False))
        if self.use_seg_stream:
            self.register_buffer("_use_seg_stream", torch.tensor(True))

        # VAE flag is the single source of truth on the encoder. When set, record a
        # detection marker so the inference wrapper rebuilds the matching encoder
        # (with logvar_head). When unset, NO buffer is registered → state_dict stays
        # byte-identical to the deterministic V4.
        self.use_vae = bool(getattr(encoder, "use_vae", False))
        # Whether the VAE encoder reparameterizes (True, default) or returns μ
        # (False). Mirrored from the encoder; recorded below only when disabled so
        # the ON default keeps the state_dict byte-identical to existing VAE ckpts.
        self.vae_sample = bool(getattr(encoder, "vae_sample", True))

        self.dino_terms = self._parse_dino_loss_type(dino_loss_type)
        w = {"l1": 1.0, "mse": 1.0, "cosine": 1.0}
        if dino_loss_weights:
            w.update(dino_loss_weights)
        self.dino_loss_weights = w

        # convenience attrs mirrored from encoder (some wrapper paths read these)
        self.num_global_tokens = encoder.num_global_tokens
        self.num_hand_tokens = encoder.num_hand_tokens
        self.hand_in_recon = False  # V4 minimal scope: no hand tokens

        # V4 detection marker.
        self.register_buffer("_is_v4", torch.tensor(True))

        # VAE detection marker (only when enabled → off-path state_dict unchanged).
        if self.use_vae:
            self.register_buffer("_is_vae", torch.tensor(True))
            # Sampling-off marker. Registered ONLY when a VAE tokenizer disables
            # sampling (returns μ). Absence ⇒ sampling ON (the default), so ordinary
            # VAE checkpoints stay byte-identical and load strict without injection.
            # The inference wrapper reads this to rebuild the matching encoder, so
            # Stage-2 latent targets and any tokenizer inference use μ consistently.
            if not self.vae_sample:
                self.register_buffer("_vae_no_sample", torch.tensor(True))

        # Visual feature source. Default "dino" registers NO extra buffers, so the
        # state_dict / forward of DINO-trained models is byte-identical to before.
        # Only the "vggt" path records markers so the inference wrapper can rebuild
        # the matching frozen extractor (the feature extractor itself lives outside
        # this module — trainer/wrapper-owned, frozen, not in this state_dict).
        self.feature_source = feature_source
        if feature_source == "vggt":
            assert vggt_token_source in ("aggregator", "dpt_out2"), (
                f"vggt_token_source must be 'aggregator' or 'dpt_out2'; got {vggt_token_source!r}"
            )
            self.register_buffer("_feature_source", _str_to_byte_tensor("vggt"))
            self.register_buffer("_vggt_token_source", _str_to_byte_tensor(str(vggt_token_source)))
            self.register_buffer("_vggt_image_size", torch.tensor(int(vggt_image_size or 224)))
            self.register_buffer("_vggt_model", _str_to_byte_tensor(str(vggt_model or "facebook/VGGT-1B")))
            # VGGT final-norm mode. "none" (default) registers NO buffer, so existing
            # VGGT checkpoints stay byte-identical. "naive" (extra non-affine LN on the
            # final tokens) records a marker so the inference wrapper rebuilds a
            # matching VGGT extractor. Only meaningful when feature_source == "vggt".
            if vggt_final_norm == "naive":
                self.register_buffer("_vggt_final_norm", _str_to_byte_tensor("naive"))
        self.vggt_final_norm = vggt_final_norm

        # DINO final-LayerNorm mode. "affine" (default) registers NO buffer, so the
        # state_dict stays byte-identical to before. "naive" (drop the final LN's
        # learned affine) records a marker so the inference wrapper rebuilds the
        # matching DINO extractor. Only meaningful when feature_source == "dino".
        self.dino_final_norm = dino_final_norm
        if feature_source == "dino" and dino_final_norm == "naive":
            self.register_buffer("_dino_final_norm", _str_to_byte_tensor("naive"))

    @staticmethod
    def _parse_dino_loss_type(s: str):
        terms = [t.strip().lower() for t in str(s).split("+") if t.strip()]
        if not terms:
            raise ValueError(f"Empty dino_loss_type: {s!r}")
        for t in terms:
            if t not in _VALID_DINO_TERMS:
                raise ValueError(
                    f"Unknown dino loss term {t!r}; valid: {_VALID_DINO_TERMS}"
                )
        return terms

    @staticmethod
    def _zero(device):
        return torch.zeros((), device=device)

    def _recon_loss_fn(self, pred, target):
        if self.recon_loss_type == "l1":
            return F.l1_loss(pred, target)
        return F.mse_loss(pred, target)

    def _dino_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        patch_weights: Optional[torch.Tensor] = None,
    ):
        """Return (total_dino_loss, {sub_term_name: value}).

        ``patch_weights`` ([B, Lp], mean == 1) enables the mask-weighted variant:
        each per-patch loss is scaled by its weight before averaging, so patches
        under the SAM3 mask count more while the TOTAL scale is unchanged. None
        (default) keeps the original reduction — the same F.*_loss calls as before,
        so the off-path stays byte-identical.
        """
        sub = {}
        total = pred.new_zeros(())
        target = target.to(dtype=pred.dtype)
        if patch_weights is not None:
            w = patch_weights.to(dtype=pred.dtype).unsqueeze(-1)  # [B, Lp, 1]
        for t in self.dino_terms:
            if patch_weights is None:
                if t == "l1":
                    v = F.l1_loss(pred, target)
                elif t == "mse":
                    v = F.mse_loss(pred, target)
                else:  # cosine
                    v = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
            else:
                if t == "l1":
                    v = ((pred - target).abs() * w).mean()
                elif t == "mse":
                    v = ((pred - target).pow(2) * w).mean()
                else:  # cosine — per-patch distance [B, Lp], weight then average
                    v = ((1.0 - F.cosine_similarity(pred, target, dim=-1)) * w.squeeze(-1)).mean()
            sub[f"loss_dino_{t}"] = v
            total = total + self.dino_loss_weights[t] * v
        return total, sub

    def _mask_patch_weights(
        self, mask_x1: torch.Tensor, mask_valid: torch.Tensor, num_patches: int
    ) -> torch.Tensor:
        """x1-step union mask [B, S, S] → per-patch weights [B, Lp], batch-mean 1.

        Coverage c = average of the binary mask inside each DINO patch cell
        (adaptive avg-pool to the sqrt(Lp)×sqrt(Lp) grid). w = 1 + (W−1)·c, forced
        to 1 for samples whose mask npz is missing, then divided by the (local)
        batch mean so the overall loss scale is invariant to W and mask density.
        """
        grid = int(round(num_patches ** 0.5))
        assert grid * grid == num_patches, (
            f"DINO token count {num_patches} is not a square grid — mask-weighted "
            "dino loss expects square patch grids."
        )
        cov = F.adaptive_avg_pool2d(mask_x1.unsqueeze(1).float(), (grid, grid))
        cov = cov.flatten(1)  # [B, Lp] ∈ [0, 1]
        w = 1.0 + (self.mask_patch_weight - 1.0) * cov
        w = torch.where(mask_valid.view(-1, 1).bool(), w, torch.ones_like(w))
        return w / w.mean()

    # ---- interface (matches V2/V3 plus DINO feats) ----

    def encode(
        self,
        actions: torch.Tensor,
        x0_feat: torch.Tensor,
        x1_feat: torch.Tensor,
        s0_feat: Optional[torch.Tensor] = None,
        s1_feat: Optional[torch.Tensor] = None,
    ):
        """[B,T,D] actions + DINO feats [B,Lp,C] → (global, time, hand) @ token_dim.

        With the segment stream enabled, ``s0_feat``/``s1_feat`` are the cutout frames'
        DINO features; their difference is concatenated side-by-side with the RGB
        difference inside the fusion encoder.
        """
        dino_diff = x1_feat.to(dtype=actions.dtype) - x0_feat.to(dtype=actions.dtype)
        seg_diff = None
        if self.use_seg_stream:
            assert s0_feat is not None and s1_feat is not None, (
                "this tokenizer was built with use_seg_stream=True; encode() requires "
                "s0_feat/s1_feat (segment-stream DINO features)."
            )
            seg_diff = s1_feat.to(dtype=actions.dtype) - s0_feat.to(dtype=actions.dtype)
        return self.encoder(actions, dino_diff, seg_diff)

    def decode(self, global_tok, time_tok, hand_tok, x0_feat=None) -> torch.Tensor:
        """Action reconstruction from latent tokens (V2/V3-compatible signature).

        ``x0_feat`` is only consumed by a unified decoder built with
        ``recon_sees_vision`` (exploration mode); every other configuration
        decodes from the latent alone and ignores it.

        With a VISUAL-ONLY unified decoder (``shared_trunk_vis``) the action path is
        the ordinary ``recon_decoder`` below — identical to the "separate" arch.
        """
        if self.unified_decoder is not None and not self._unified_visual_only:
            ctx = x0_feat if self.unified_decoder.recon_sees_vision else None
            action, _, _ = self.unified_decoder(global_tok, time_tok, hand_tok, ctx)
            return action
        g = global_tok if (global_tok is not None and global_tok.shape[1] > 0) else None
        h = hand_tok if (hand_tok is not None and hand_tok.shape[1] > 0 and self.hand_in_recon) else None
        return self.recon_decoder(time_tok, global_tokens=g, hand_tokens=h)

    def decode_dino(self, time_tok: torch.Tensor, x0_feat: torch.Tensor) -> torch.Tensor:
        """Predict future-frame DINO features from current-frame feats + latent."""
        if self.unified_decoder is not None:
            _, visuals, _ = self.unified_decoder(None, time_tok, None, x0_feat)
            return visuals
        _, visuals = self.dino_decoder(x=x0_feat, tokens=time_tok)
        return visuals

    def decode_dino_seg(
        self,
        time_tok: torch.Tensor,
        s0_feat: torch.Tensor,
        x0_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Segment-stream twin of :meth:`decode_dino`.

        Identical mechanics — a separate ``SimpleTokenTransformer`` conditioned on the
        latent — and the TARGET is always the cutout stream's future features
        (``s1_feat``). The visual context depends on ``seg_dino_decoder_input``:
        ``"seg"`` (default) uses the cutout ``s0_feat``; ``"rgb"`` uses the raw image
        ``x0_feat`` (which must then be passed), so both decoders see the same input.
        """
        assert self.seg_dino_decoder is not None, "seg_dino_decoder was not built"
        if self.seg_dino_decoder_input == "rgb":
            assert x0_feat is not None, (
                "seg_dino_decoder_input='rgb' requires x0_feat (the raw image features) "
                "to be passed to decode_dino_seg()."
            )
            ctx = x0_feat
        else:
            ctx = s0_feat
        _, visuals = self.seg_dino_decoder(x=ctx, tokens=time_tok)
        return visuals

    def decode_seg_pixel(self, time_tok: torch.Tensor, x0_feat: torch.Tensor) -> torch.Tensor:
        """Predict the FUTURE (x1-step) union-mask logits from x0 feats + latent.

        Same mechanics as :meth:`decode_dino` (a separate ``SimpleTokenTransformer``
        conditioned on the latent, visual context = the raw image ``x0_feat``), then
        the linear head maps each patch token to its patch² pixel-block logits and
        the blocks are tiled to the full map. Returns [B, grid·patch, grid·patch].
        """
        if self._unified_segpix:
            _, _, seg_vis = self.unified_decoder(None, time_tok, None, x0_feat)
            visuals = seg_vis  # [B, Lp, width] — raw segpix-branch output
        else:
            assert self.seg_pixel_decoder is not None, "seg_pixel_decoder was not built"
            _, visuals = self.seg_pixel_decoder(x=x0_feat, tokens=time_tok)  # [B, Lp, C]
        return self._segpix_logits(visuals)

    def _segpix_logits(self, visuals: torch.Tensor) -> torch.Tensor:
        """Patch visuals [B, Lp, C] → tiled pixel-logit map [B, grid·patch, grid·patch]."""
        blocks = self.seg_pixel_head(visuals)  # [B, Lp, patch²]
        Lp = blocks.shape[1]
        grid = int(round(Lp ** 0.5))
        assert grid * grid == Lp, (
            f"DINO token count {Lp} is not a square grid — the seg-pixel head "
            "expects square patch grids."
        )
        return einops.rearrange(
            blocks,
            "b (h w) (p q) -> b (h p) (w q)",
            h=grid, w=grid, p=self.seg_pixel_patch, q=self.seg_pixel_patch,
        )

    def forward(self, batch: dict = None, **kwargs) -> dict:
        if batch is None:
            batch = kwargs
        actions = batch["action"]
        actions = actions.to(dtype=self.encoder.action_proj.weight.dtype)
        x0_feat = batch["x0_feat"].to(dtype=actions.dtype)
        x1_feat = batch["x1_feat"].to(dtype=actions.dtype)
        device = actions.device

        s0_feat = batch.get("s0_feat")
        s1_feat = batch.get("s1_feat")
        if s0_feat is not None:
            s0_feat = s0_feat.to(dtype=actions.dtype)
            s1_feat = s1_feat.to(dtype=actions.dtype)

        global_tok, time_tok, hand_tok = self.encode(
            actions, x0_feat, x1_feat, s0_feat, s1_feat
        )

        # Loss 1: action reconstruction (unified decoder computes losses 1+2 in one
        # combined masked pass — see below, after the patch weights are prepared).
        # A VISUAL-ONLY unified decoder ("shared_trunk_vis") has no action path, so
        # loss 1 is computed here from the separate recon_decoder exactly as in the
        # "separate" configuration.
        if self.unified_decoder is None or self._unified_visual_only:
            if self.lambda_recon > 0:
                recon = self.decode(global_tok, time_tok, hand_tok)
                loss_recon = self._recon_loss_fn(recon, actions)
            else:
                loss_recon = self._zero(device)

        # Optional per-patch weights for loss 2 (mask-weighted DINO loss). None when
        # the flag is off → _dino_loss takes its original (byte-identical) branch.
        patch_w = None
        if self.use_mask_weighted_dino_loss:
            assert "mask_x1" in batch and "mask_valid" in batch, (
                "use_mask_weighted_dino_loss=True but the batch has no mask_x1/"
                "mask_valid (set --mask-dataset-root so the dataset attaches them)."
            )
            patch_w = self._mask_patch_weights(
                batch["mask_x1"].to(device), batch["mask_valid"].to(device),
                num_patches=x1_feat.shape[1],
            )

        # Loss 2: DINO future-feature reconstruction (conditioned on time latent)
        if self.unified_decoder is None:
            if self.lambda_dino > 0:
                pred_x1 = self.decode_dino(time_tok, x0_feat)
                loss_dino, dino_sub = self._dino_loss(pred_x1, x1_feat, patch_weights=patch_w)
            else:
                loss_dino = self._zero(device)
                dino_sub = {}
        else:
            # Unified decoder: ONE combined masked pass yields the action recon (from
            # the latent positions, which the mask keeps vision-free) AND the future
            # feature prediction (from the patch positions). Loss weighting/combination
            # below is identical to the separate path.
            # "shared_trunk_vis": the pass is visual-only (recon is None) and loss 1
            # was already computed above from the separate recon_decoder.
            recon, pred_x1, seg_vis = self.unified_decoder(global_tok, time_tok, hand_tok, x0_feat)
            if not self._unified_visual_only:
                loss_recon = (
                    self._recon_loss_fn(recon, actions) if self.lambda_recon > 0 else self._zero(device)
                )
            if self.lambda_dino > 0:
                loss_dino, dino_sub = self._dino_loss(pred_x1, x1_feat, patch_weights=patch_w)
            else:
                loss_dino = self._zero(device)
                dino_sub = {}

        loss = self.lambda_recon * loss_recon + self.lambda_dino * loss_dino

        # Loss 2b: segment-stream DINO future-feature reconstruction. Same mechanics as
        # loss 2, on the cutout stream's features. Only when the seg DINO decoder was
        # built (default: absent → this block is skipped and `loss` is unchanged).
        out_seg = {}
        if self.seg_dino_decoder is not None:
            assert s0_feat is not None and s1_feat is not None, (
                "seg_dino_decoder is present but the batch has no s0_feat/s1_feat."
            )
            pred_s1 = self.decode_dino_seg(time_tok, s0_feat, x0_feat)
            loss_dino_seg, seg_sub = self._dino_loss(pred_s1, s1_feat)
            loss = loss + self.lambda_dino_seg * loss_dino_seg
            out_seg["loss_dino_seg"] = loss_dino_seg
            out_seg.update({f"{k}_seg": v for k, v in seg_sub.items()})

        # Loss 2c: seg-mask pixel prediction (future union mask, BCE). Only when the
        # seg-pixel decoder was built (default: absent → skipped, `loss` unchanged).
        # Samples whose mask npz is missing (mask_valid=0) are excluded from the
        # average; with zero valid samples the term is 0 but the decoder/head still
        # ran, so DDP sees no unused parameters.
        if self.seg_pixel_decoder is not None or self._unified_segpix:
            assert "mask_x1" in batch and "mask_valid" in batch, (
                "seg_pixel_decoder is present but the batch has no mask_x1/mask_valid "
                "(set --mask-dataset-root so the dataset attaches them)."
            )
            if self._unified_segpix:
                # segpix visuals come from the SAME combined pass as losses 1+2
                logits = self._segpix_logits(seg_vis)  # [B, S, S]
            else:
                logits = self.decode_seg_pixel(time_tok, x0_feat)  # [B, S, S]
            target = batch["mask_x1"].to(device=logits.device).float()
            assert logits.shape == target.shape, (
                f"seg-pixel logits {tuple(logits.shape)} != mask {tuple(target.shape)} "
                "— image_size must equal grid × seg_pixel_patch."
            )
            bce = F.binary_cross_entropy_with_logits(
                logits.float(), target, reduction="none"
            ).mean(dim=(1, 2))  # [B]
            validf = batch["mask_valid"].to(device=logits.device).float()
            loss_seg_pixel = (bce * validf).sum() / validf.sum().clamp(min=1.0)
            loss = loss + self.lambda_seg_pixel * loss_seg_pixel
            out_seg["loss_seg_pixel"] = loss_seg_pixel

        # Loss 3: VAE KL (only when the encoder is a VAE). The encoder stashes the
        # KL during encode; here we weight and add it. logvar_head is in the z graph
        # regardless of lambda_kl, so DDP sees no unused params even at lambda_kl=0.
        out = {
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_dino": loss_dino,
        }
        out.update(out_seg)
        if self.use_vae and self.encoder._last_kl is not None:
            loss_kl = self.encoder._last_kl
            if self.lambda_kl > 0:
                loss = loss + self.lambda_kl * loss_kl
            out["loss"] = loss
            out["loss_kl"] = loss_kl
        out.update(dino_sub)
        return out

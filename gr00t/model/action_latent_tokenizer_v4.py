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
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.emb_dim = emb_dim
        self.token_dim = token_dim
        self.dino_dim = dino_dim
        self.num_global_tokens = num_global_tokens
        self.num_hand_tokens = num_hand_tokens

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
        self.joint = SimpleTokenTransformer(
            in_channels=dino_dim,
            model_channels=fusion_width,
            out_channels=token_dim,
            num_blocks=fusion_depth,
            num_heads=fusion_heads,
            num_tokens=0,
            token_channels=emb_dim,
            use_fp16=False,
        )

    @property
    def action_proj(self) -> nn.Module:
        # Wrapper/trainer probe ``encoder.action_proj.weight.dtype``.
        return self.action_encoder.action_proj

    def forward(self, actions: torch.Tensor, dino_diff: torch.Tensor):
        B, T, _ = actions.shape
        Ng = self.num_global_tokens
        Nh = self.num_hand_tokens

        g256, t256, h256 = self.action_encoder(actions)          # [B,*,256]
        act_tokens = torch.cat([g256, t256, h256], dim=1)         # [B, Ng+T+Nh, 256]

        tokens_out, _ = self.joint(x=dino_diff, tokens=act_tokens)  # [B, Ng+T+Nh, 64]

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
        recon_decoder: ReconDecoderV4,
        dino_decoder: Optional[SimpleTokenTransformer] = None,
        lambda_recon: float = 1.0,
        lambda_dino: float = 1.0,
        recon_loss_type: str = "mse",
        dino_loss_type: str = "l1",
        dino_loss_weights: Optional[dict] = None,
        feature_source: str = "dino",
        vggt_token_source: Optional[str] = None,
        vggt_image_size: Optional[int] = None,
        vggt_model: Optional[str] = None,
        dino_final_norm: str = "affine",
    ):
        super().__init__()
        self.encoder = encoder
        self.recon_decoder = recon_decoder
        self.dino_decoder = dino_decoder

        self.lambda_recon = float(lambda_recon)
        self.lambda_dino = float(lambda_dino)
        self.recon_loss_type = recon_loss_type

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

    def _dino_loss(self, pred: torch.Tensor, target: torch.Tensor):
        """Return (total_dino_loss, {sub_term_name: value})."""
        sub = {}
        total = pred.new_zeros(())
        target = target.to(dtype=pred.dtype)
        for t in self.dino_terms:
            if t == "l1":
                v = F.l1_loss(pred, target)
            elif t == "mse":
                v = F.mse_loss(pred, target)
            else:  # cosine
                v = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
            sub[f"loss_dino_{t}"] = v
            total = total + self.dino_loss_weights[t] * v
        return total, sub

    # ---- interface (matches V2/V3 plus DINO feats) ----

    def encode(self, actions: torch.Tensor, x0_feat: torch.Tensor, x1_feat: torch.Tensor):
        """[B,T,D] actions + DINO feats [B,Lp,C] → (global, time, hand) @ token_dim."""
        dino_diff = x1_feat.to(dtype=actions.dtype) - x0_feat.to(dtype=actions.dtype)
        return self.encoder(actions, dino_diff)

    def decode(self, global_tok, time_tok, hand_tok) -> torch.Tensor:
        """Action reconstruction from latent tokens (V2/V3-compatible signature)."""
        g = global_tok if (global_tok is not None and global_tok.shape[1] > 0) else None
        h = hand_tok if (hand_tok is not None and hand_tok.shape[1] > 0 and self.hand_in_recon) else None
        return self.recon_decoder(time_tok, global_tokens=g, hand_tokens=h)

    def decode_dino(self, time_tok: torch.Tensor, x0_feat: torch.Tensor) -> torch.Tensor:
        """Predict future-frame DINO features from current-frame feats + latent."""
        _, visuals = self.dino_decoder(x=x0_feat, tokens=time_tok)
        return visuals

    def forward(self, batch: dict = None, **kwargs) -> dict:
        if batch is None:
            batch = kwargs
        actions = batch["action"]
        actions = actions.to(dtype=self.encoder.action_proj.weight.dtype)
        x0_feat = batch["x0_feat"].to(dtype=actions.dtype)
        x1_feat = batch["x1_feat"].to(dtype=actions.dtype)
        device = actions.device

        global_tok, time_tok, hand_tok = self.encode(actions, x0_feat, x1_feat)

        # Loss 1: action reconstruction
        if self.lambda_recon > 0:
            recon = self.decode(global_tok, time_tok, hand_tok)
            loss_recon = self._recon_loss_fn(recon, actions)
        else:
            loss_recon = self._zero(device)

        # Loss 2: DINO future-feature reconstruction (conditioned on time latent)
        if self.lambda_dino > 0:
            pred_x1 = self.decode_dino(time_tok, x0_feat)
            loss_dino, dino_sub = self._dino_loss(pred_x1, x1_feat)
        else:
            loss_dino = self._zero(device)
            dino_sub = {}

        loss = self.lambda_recon * loss_recon + self.lambda_dino * loss_dino

        out = {
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_dino": loss_dino,
        }
        out.update(dino_sub)
        return out

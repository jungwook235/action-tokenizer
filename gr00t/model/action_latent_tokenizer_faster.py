"""
DimensionWise Action Latent Tokenizer.

TimeWiseEncoder와의 차이:
- Encoder: time-dim 축을 transpose하여 각 action dimension이 하나의 토큰.
  [B, T, D] → transpose → [B, D, T] → Linear(T, E) → PE(D) → Transformer
- 각 latent 토큰 = 하나의 action dimension의 전체 time step 정보를 포함.

지원하는 loss:
  1. Recon path 1 (dim tokens only)
  2. Recon path 2 (dim + hand tokens, hand-weighted) — Nh > 0일 때
  3. Masked recon (global + masked dim tokens) — Ng > 0일 때
  4. Frequency domain loss (optional)
  + recon loss type: mse / l1 선택

Building blocks는 로컬에 포함 (cross-repo import 회피).
"""

from functools import partial
from typing import List, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Building blocks (shared with action_latent_tokenizer.py)
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
# DimensionWise Encoder
# =====================================================================


class DimensionWiseEncoder(nn.Module):
    """Action chunk → per-dimension latent tokens (+ optional global/hand tokens).

    TimeWiseEncoder와 달리 time-dim 축을 transpose:
    - [B, T, D] → transpose → [B, D, T] → Linear(T, E) → PE(D) → [global | dim | hand] → Transformer
    - Output: global[B,Ng,E], dim_tokens[B,D,E], hand[B,Nh,E]
    - 각 token은 해당 action dimension의 전체 time step 정보를 포함.
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

        # Projects each dimension's time series (T) → embedding (E)
        self.action_proj = nn.Linear(action_horizon, emb_dim)
        # PE over dimension positions (D)
        self.dim_pos_emb = PositionalEmbeddingAdder(emb_dim, max_sizes=[action_dim])

        self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, emb_dim)) if num_global_tokens > 0 else None
        self.hand_tokens = nn.Parameter(torch.randn(num_hand_tokens, emb_dim)) if num_hand_tokens > 0 else None

        self.transformer = Transformer(dim=emb_dim, depth=depth, head_dim=head_dim, drop=pdropout)

        # Type marker for auto-detection in ActionLatentTokenizerWrapper
        self.register_buffer("_is_dimension_wise", torch.tensor(True))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, D] normalized actions
        Returns:
            global_tokens: [B, Ng, E]
            dim_tokens:    [B, D, E]  — each token = one action dimension's info
            hand_tokens:   [B, Nh, E]
        """
        B, T, D = x.shape
        Ng = self.num_global_tokens
        Nh = self.num_hand_tokens

        x = x.transpose(1, 2)      # [B, D, T]
        x = self.action_proj(x)    # [B, D, E], Linear(T→E)
        x = self.dim_pos_emb(x)    # [B, D, E]

        parts = []
        if Ng > 0:
            parts.append(self.global_tokens.unsqueeze(0).expand(B, -1, -1))
        parts.append(x)
        if Nh > 0:
            parts.append(self.hand_tokens.unsqueeze(0).expand(B, -1, -1))
        x = torch.cat(parts, dim=1)  # [B, Ng+D+Nh, E]

        x = self.transformer(x)  # [B, Ng+D+Nh, E]

        global_out = x[:, :Ng]           # [B, Ng, E]
        dim_out = x[:, Ng:Ng + D]        # [B, D, E]
        hand_out = x[:, Ng + D:]         # [B, Nh, E]

        return global_out, dim_out, hand_out


# =====================================================================
# DimensionWise Decoder: Recon path 1 & 2
# =====================================================================


class DimensionWiseReconDecoder(nn.Module):
    """Reconstruction decoder for DimensionWiseEncoder.

    Path 1: forward(dim_tokens, hand_tokens=None) — dim tokens만으로 재구성
    Path 2: forward(dim_tokens, hand_tokens=hand) — dim + hand tokens로 재구성

    Input:  dim_tokens [B, D, E], hand_tokens [B, Nh, E] (optional)
    Output: [B, T, D]

    각 dim token (E) → time series (T) 복원 후 transpose.
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
        num_hand_tokens: int = 0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.emb_dim = emb_dim
        self.decoder_mode = decoder_mode
        self.num_hand_tokens = num_hand_tokens

        self.dim_proj = LinearLayer(emb_dim, emb_dim)
        self.hand_proj = LinearLayer(emb_dim, emb_dim) if num_hand_tokens > 0 else None
        self.dim_pos_emb = PositionalEmbeddingAdder(emb_dim, max_sizes=[action_dim])
        self.hand_pos_emb = PositionalEmbeddingAdder(emb_dim, max_sizes=[num_hand_tokens]) if num_hand_tokens > 0 else None
        # E → T: each dim token reconstructs its full time series
        self.head = LinearHead(emb_dim, action_horizon)

        if decoder_mode == "cross_attention":
            # Query: D positions (one per action dimension)
            self.query_pos_emb = PositionalEmbedding(emb_dim, max_sizes=[action_dim])
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

    def forward(self, dim_tokens: torch.Tensor, hand_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, D, E = dim_tokens.shape

        latents = self.dim_proj(dim_tokens)
        latents = self.dim_pos_emb(latents)

        if hand_tokens is not None:
            hand_lat = self.hand_proj(hand_tokens)
            hand_lat = self.hand_pos_emb(hand_lat)
            latents = torch.cat([latents, hand_lat], dim=1)  # [B, D+Nh, E]

        if self.decoder_mode == "cross_attention":
            query = self.query_pos_emb(shape=[D])
            query = einops.rearrange(query, "1 E D -> 1 D E").expand(B, -1, -1)
            out = self.decoder(query, latents)  # [B, D, E]
        else:
            out = self.transformer(latents)
            out = out[:, :D]  # extract dim positions only

        out = self.head(out)        # [B, D, T]
        return out.transpose(1, 2)  # [B, T, D]


# =====================================================================
# DimensionWise Decoder: Masked recon
# =====================================================================


class DimensionWiseMaskedReconDecoder(nn.Module):
    """Masked reconstruction decoder for DimensionWiseEncoder.

    global tokens + masked dim tokens → reconstruct full actions.
    Masked dim tokens are replaced with a learnable mask token.

    Input:  global[B,Ng,E], dim[B,D,E], mask[B,D] (True=masked)
    Output: [B, T, D]
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        emb_dim: int,
        head_dim: int,
        depth: int,
        pdropout: float,
        decoder_mode: str = "cross_attention",
        num_global_tokens: int = 1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.emb_dim = emb_dim
        self.decoder_mode = decoder_mode
        self.num_global_tokens = num_global_tokens

        self.global_proj = LinearLayer(emb_dim, emb_dim)
        self.dim_proj = LinearLayer(emb_dim, emb_dim)
        self.mask_token = nn.Parameter(torch.randn(emb_dim))
        self.global_pos_emb = PositionalEmbeddingAdder(emb_dim, max_sizes=[num_global_tokens])
        self.dim_pos_emb = PositionalEmbeddingAdder(emb_dim, max_sizes=[action_dim])
        self.head = LinearHead(emb_dim, action_horizon)  # E → T

        if decoder_mode == "cross_attention":
            self.query_pos_emb = PositionalEmbedding(emb_dim, max_sizes=[action_dim])
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

    def forward(self, global_tokens: torch.Tensor, dim_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, D, E = dim_tokens.shape
        Ng = self.num_global_tokens

        global_lat = self.global_pos_emb(self.global_proj(global_tokens))
        dim_lat = self.dim_pos_emb(self.dim_proj(dim_tokens))

        # Replace masked dim tokens with learnable mask token
        mask_expanded = mask.unsqueeze(-1).expand_as(dim_lat)
        mask_tok = self.mask_token.unsqueeze(0).unsqueeze(0).expand_as(dim_lat)
        dim_lat = torch.where(mask_expanded, mask_tok, dim_lat)

        latents = torch.cat([global_lat, dim_lat], dim=1)  # [B, Ng+D, E]

        if self.decoder_mode == "cross_attention":
            query = self.query_pos_emb(shape=[D])
            query = einops.rearrange(query, "1 E D -> 1 D E").expand(B, -1, -1)
            out = self.decoder(query, latents)  # [B, D, E]
        else:
            out = self.transformer(latents)
            out = out[:, Ng:]  # skip global positions → [B, D, E]

        out = self.head(out)        # [B, D, T]
        return out.transpose(1, 2)  # [B, T, D]


# =====================================================================
# Dimension Masking (masks over action_dim axis, analogous to TimestepMasking)
# =====================================================================


class DimensionMasking(nn.Module):
    """Generates masks for dim tokens. [B, D] bool where True=masked."""

    def __init__(self, mask_ratio: float = 0.5, mask_mode: str = "random",
                 min_mask_ratio: Optional[float] = None, max_mask_ratio: Optional[float] = None):
        super().__init__()
        self.mask_mode = mask_mode
        if min_mask_ratio is not None and max_mask_ratio is not None:
            self.min_mask_ratio = min_mask_ratio
            self.max_mask_ratio = max_mask_ratio
        else:
            self.min_mask_ratio = mask_ratio
            self.max_mask_ratio = mask_ratio
        self.register_buffer(
            "_mask_mode_bytes",
            torch.tensor(list(self.mask_mode.encode()), dtype=torch.uint8),
        )
        self.register_buffer("_min_mask_ratio_buf", torch.tensor(self.min_mask_ratio, dtype=torch.float32))
        self.register_buffer("_max_mask_ratio_buf", torch.tensor(self.max_mask_ratio, dtype=torch.float32))

    def _sync_from_buffers(self):
        self.mask_mode = bytes(self._mask_mode_bytes.tolist()).decode()
        self.min_mask_ratio = self._min_mask_ratio_buf.item()
        self.max_mask_ratio = self._max_mask_ratio_buf.item()

    def load_state_dict(self, state_dict, strict=True):
        result = super().load_state_dict(state_dict, strict=strict)
        self._sync_from_buffers()
        return result

    def _sample_num_mask(self, seq_len: int) -> int:
        if self.min_mask_ratio == self.max_mask_ratio:
            ratio = self.min_mask_ratio
        else:
            ratio = torch.empty(1).uniform_(self.min_mask_ratio, self.max_mask_ratio).item()
        return max(1, int(seq_len * ratio))

    def forward(self, batch_size: int, seq_len: int, device) -> torch.Tensor:
        """Returns [B, D] bool mask (True=masked) over action dimensions."""
        num_mask = self._sample_num_mask(seq_len)
        if self.mask_mode == "random":
            rand = torch.rand(batch_size, seq_len, device=device)
            _, indices = rand.topk(num_mask, dim=1)
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            mask.scatter_(1, indices, True)
        elif self.mask_mode == "block":
            max_start = seq_len - num_mask
            start = torch.randint(0, max_start + 1, (batch_size,), device=device)
            arange = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = (arange >= start.unsqueeze(1)) & (arange < (start + num_mask).unsqueeze(1))
        else:
            raise ValueError(f"Unknown mask_mode: {self.mask_mode}")
        return mask


# =====================================================================
# Frequency domain loss
# =====================================================================


def frequency_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str = "mse") -> torch.Tensor:
    """FFT loss over time axis (dim=1). pred/target: [B, T, D]."""
    pred_fft = torch.fft.rfft(pred, dim=1)
    target_fft = torch.fft.rfft(target, dim=1)
    if loss_type == "mse":
        return F.mse_loss(pred_fft.abs(), target_fft.abs())
    elif loss_type == "l1":
        return F.l1_loss(pred_fft.abs(), target_fft.abs())
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# =====================================================================
# Full DimensionWise Tokenizer Model
# =====================================================================


class DimensionWiseActionLatentTokenizer(nn.Module):
    """DimensionWise Action Latent Tokenizer.

    TimeWiseActionLatentTokenizer와 동일한 3가지 loss + frequency loss를 지원하지만
    latent 토큰이 action dimension 단위 (각 차원의 시계열 전체 표현).

    각 latent 토큰: 특정 action dimension이 전체 horizon에 걸쳐 어떻게 변하는지의 정보.
    """

    def __init__(
        self,
        encoder: DimensionWiseEncoder,
        recon_decoder: DimensionWiseReconDecoder,
        masked_recon_decoder: Optional[DimensionWiseMaskedReconDecoder] = None,
        masking: Optional[DimensionMasking] = None,
        # loss weights
        lambda_recon: float = 1.0,
        lambda_masked: float = 1.0,
        # hand config
        hand_action_dims: Optional[List[int]] = None,
        hand_loss_weight: float = 1.0,
        # recon loss config
        recon_loss_type: str = "mse",
        freq_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.recon_decoder = recon_decoder
        self.masked_recon_decoder = masked_recon_decoder
        self.masking = masking

        self.lambda_recon = lambda_recon
        self.lambda_masked = lambda_masked
        self.hand_action_dims = hand_action_dims if hand_action_dims is not None else []
        self.hand_loss_weight = hand_loss_weight
        self.recon_loss_type = recon_loss_type
        self.freq_loss_weight = freq_loss_weight

        self.num_global_tokens = encoder.num_global_tokens
        self.num_hand_tokens = encoder.num_hand_tokens

    def _zero(self, device):
        return torch.tensor(0.0, device=device, requires_grad=False)

    def _recon_loss_fn(self, pred, target):
        return F.mse_loss(pred, target) if self.recon_loss_type == "mse" else F.l1_loss(pred, target)

    def _can_masked(self):
        return (self.num_global_tokens > 0 and self.masked_recon_decoder is not None
                and self.masking is not None and self.lambda_masked > 0)

    def _can_recon2(self):
        return self.num_hand_tokens > 0 and self.lambda_recon > 0

    def forward(self, batch: dict = None, **kwargs) -> dict:
        """
        Args:
            batch: {"action": [B, T, D]}
        Returns:
            dict with "loss" and per-component losses
        """
        if batch is None:
            batch = kwargs
        actions = batch["action"]  # [B, T, D]
        actions = actions.to(dtype=self.encoder.action_proj.weight.dtype)
        device = actions.device
        B, T, D = actions.shape

        # Encode: each dim token encodes one action dimension's full time series
        global_tok, dim_tok, hand_tok = self.encoder(actions)

        # --- Loss 1: Recon path 1 (dim tokens only) ---
        if self.lambda_recon > 0:
            recons_1 = self.recon_decoder(dim_tok, hand_tokens=None)
            loss_recon1 = self._recon_loss_fn(recons_1, actions)
        else:
            recons_1 = None
            loss_recon1 = self._zero(device)

        # --- Loss 2: Recon path 2 (dim + hand tokens) ---
        if self._can_recon2():
            recons_2 = self.recon_decoder(dim_tok, hand_tokens=hand_tok)
            if len(self.hand_action_dims) > 0:
                sq_err = (recons_2 - actions) ** 2
                dim_weight = torch.ones(D, device=device, dtype=sq_err.dtype)
                dim_weight[self.hand_action_dims] = self.hand_loss_weight
                loss_recon2 = (sq_err * dim_weight.view(1, 1, -1)).mean()
                loss_recon2_hand = sq_err[..., self.hand_action_dims].mean()
                non_hand = [i for i in range(D) if i not in self.hand_action_dims]
                loss_recon2_non_hand = sq_err[..., non_hand].mean() if non_hand else self._zero(device)
            else:
                loss_recon2 = self._recon_loss_fn(recons_2, actions)
                loss_recon2_hand = self._zero(device)
                loss_recon2_non_hand = loss_recon2
        else:
            loss_recon2 = self._zero(device)
            loss_recon2_hand = self._zero(device)
            loss_recon2_non_hand = self._zero(device)

        # --- Loss 3: Masked recon (over action dimensions) ---
        if self._can_masked():
            mask = self.masking(B, D, device)  # [B, D] — mask over action dims
            recons_m = self.masked_recon_decoder(global_tok, dim_tok, mask)
            # Compute loss only on masked dimensions: expand mask [B,D] → [B,T,D]
            mask_exp = mask.unsqueeze(1).expand(B, T, D).float()
            sq_err_m = (recons_m - actions) ** 2
            loss_masked = (sq_err_m * mask_exp).sum() / mask_exp.sum().clamp(min=1.0)
        else:
            loss_masked = self._zero(device)

        # --- Frequency loss (on recon 1, over time axis) ---
        if self.freq_loss_weight > 0 and recons_1 is not None:
            loss_freq = frequency_loss(recons_1, actions, loss_type=self.recon_loss_type)
        else:
            loss_freq = self._zero(device)

        # --- Total ---
        loss = (
            self.lambda_recon * (loss_recon1 + loss_recon2)
            + self.lambda_masked * loss_masked
            + self.freq_loss_weight * loss_freq
        )

        return {
            "loss": loss,
            "loss_recon1": loss_recon1,
            "loss_recon2": loss_recon2,
            "loss_recon2_hand": loss_recon2_hand,
            "loss_recon2_non_hand": loss_recon2_non_hand,
            "loss_masked": loss_masked,
            "loss_freq": loss_freq,
        }

    def encode(self, actions: torch.Tensor):
        """Inference encode: [B, T, D] → (global, dim, hand)"""
        return self.encoder(actions)

    def decode(self, global_tok, dim_tok, hand_tok):
        """Inference decode using best path."""
        if self.num_hand_tokens > 0:
            return self.recon_decoder(dim_tok, hand_tokens=hand_tok)
        return self.recon_decoder(dim_tok, hand_tokens=None)

    def autoencode(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode → Decode: [B, T, D] → [B, T, D]"""
        g, d, h = self.encode(actions)
        return self.decode(g, d, h)

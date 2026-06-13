"""LAM decoder blocks ported from the DreamDojo ``external/lam`` project.

This file is a **focused, structural 1:1 copy** of the modules the V5 action
tokenizer needs for its *pixel* reconstruction decoder, so V5 can reuse the
DreamDojo Latent Action Model (LAM) decoder without importing the in-tree
``external.lam`` package (whose import path and a hardcoded ``.cuda()`` make it
awkward to use for an in-state_dict, device-agnostic, trainable module).

Copied with identical structure / submodule names from:
  - ``DreamDojo/external/lam/modules/blocks.py`` :
      ``patchify``, ``unpatchify``, ``PositionalEncoding``, ``SelfAttention``,
      ``SpatioBlock``, ``SpatioTransformer``

The V5 pixel decoder uses **only** the ``SpatioTransformer`` path (spatial
attention, no temporal / rotary), plus ``patchify`` / ``unpatchify``. The
``SpatioTemporalTransformer`` (encoder side) is NOT copied — that lives in the
*frozen* LAM extractor (``gr00t.utils.lam_feature``), loaded from the pretrained
checkpoint and never part of this state_dict.

Only change vs. the original:
  * ``PositionalEncoding`` stores its table as a **non-persistent buffer** instead
    of a plain attribute with a hardcoded ``.cuda()``. This (a) makes the module
    device-agnostic (CPU smoke tests work, ``.to(device)`` moves it) and (b) keeps
    it OUT of the state_dict — byte-identical to LAM, where ``pos_enc`` is a plain
    attribute and never serialized. So loading the pretrained LAM ``decoder.*`` /
    ``patch_up.*`` weights matches key-for-key.
"""

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


def patchify(videos: Tensor, size: int) -> Tensor:
    """[B,T,H,W,C] video → [B,T,N,(size*size*C)] flattened pixel patches."""
    B, T, H, W, C = videos.shape
    videos = videos[:, :, :H - (H % size), :W - (W % size), :]
    x = rearrange(videos, "b t (hn hp) (wn wp) c -> b t (hn wn) (hp wp c)", hp=size, wp=size)
    return x


def unpatchify(patches: Tensor, size: int, h_out: int, w_out: int) -> Tensor:
    """[B,T,N,(size*size*C)] patches → [B,T,H,W,C] video."""
    h_pad = -h_out % size
    hn = (h_out + h_pad) // size
    x = rearrange(patches, "b t (hn wn) (hp wp c) -> b t (hn hp) (wn wp) c", hp=size, wp=size, hn=hn)
    return x[:, :, :h_out, :w_out]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding over the spatial (token) axis.

    Differs from the DreamDojo original only in storing ``pos_enc`` as a
    non-persistent buffer (device-agnostic, not serialized) instead of a plain
    attribute with a hardcoded ``.cuda()``.
    """

    def __init__(self, model_dim: int, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        exponent = torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim)
        div_term = torch.exp(exponent)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # non-persistent: moves with .to(device) but stays out of state_dict
        # (matches LAM, where pos_enc is a plain attribute and never saved).
        self.register_buffer("pos_enc", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_enc[:x.shape[2]]


class SelfAttention(nn.Module):
    """Multi-head self-attention (LAM variant, copied verbatim; rot_emb unused here)."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(SelfAttention, self).__init__()
        inner_dim = model_dim // num_heads
        self.scale = inner_dim ** -0.5
        self.heads = num_heads

        self.to_q = nn.Linear(model_dim, model_dim, bias=False)
        self.to_k = nn.Linear(model_dim, model_dim, bias=False)
        self.to_v = nn.Linear(model_dim, model_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout)
        )

    def scaled_dot_product_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        attn_weight = query @ key.transpose(-2, -1) * self.scale
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value

    def forward(self, x: Tensor) -> Tensor:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
        out = self.scaled_dot_product_attention(q, k, v)
        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class SpatioBlock(nn.Module):
    """Per-frame spatial self-attention + FFN (copied verbatim)."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super(SpatioBlock, self).__init__()
        self.spatial_attn = SelfAttention(model_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor) -> Tensor:
        t_len = x.shape[1]

        # Spatial attention
        x = rearrange(x, "b t s e -> (b t) s e")
        x_ = self.norm1(x)
        x_ = self.spatial_attn(x_)
        x = x + x_
        x = rearrange(x, "(b t) s e -> b t s e", t=t_len)

        # Feedforward
        x_ = self.norm2(x)
        x_ = self.ffn(x_)
        x = x + x_
        return x


class SpatioTransformer(nn.Module):
    """Spatial transformer used as the LAM video decoder (copied verbatim).

    Input/output ``(B, T, S, E)``. ``ffn`` projects ``in_dim → model_dim`` and
    ``out`` projects ``model_dim → out_dim``. Submodule names match the pretrained
    LAM ``decoder.*`` keys so the checkpoint loads key-for-key.
    """

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        out_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.0
    ) -> None:
        super(SpatioTransformer, self).__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        self.pos_enc = PositionalEncoding(model_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                SpatioBlock(model_dim, num_heads, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.out = nn.Linear(model_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ffn(x)
        x = self.pos_enc(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.out(x)
        return x  # (B, T, S, out_dim)

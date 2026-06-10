"""RLA modules ported verbatim from the ``rla-wm`` project.

This file is a **1:1 structural copy** of the dense (non-sparse) modules used by
RLA's DINO inverse-dynamics autoencoder, so the V4 action tokenizer can reuse the
exact same architecture without importing ``rla-wm`` (whose
``src.models.attention_block`` pulls in ``src.modules.sparse`` → CUDA/sparse deps
that are not available in this package).

Copied with identical ``forward``/structure from:
  - ``rla-wm/src/models/simple_token_transformer.py`` : ``SimpleTokenTransformer``
  - ``rla-wm/src/models/attention_block.py``         : ``AttentionBlock`` (dense)
  - ``rla-wm/src/modules/norm.py``                   : ``LayerNorm32``
  - ``rla-wm/src/modules/utils.py``                  : ``convert_module_to_f16/f32``

Only change vs. the originals: the ``FP16_MODULES`` tuple drops the ``sparse.*``
entries (we never run the sparse path here). The V4 tokenizer instantiates these
with ``use_fp16=False`` so the fp16 conversion is a no-op and the HF Trainer's
bf16 autocast manages dtypes; the module structure and forward are unchanged.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# norm.py (copied)
# =====================================================================


class LayerNorm32(nn.LayerNorm):
    """LayerNorm that always computes in fp32 then casts back to input dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(x.float()).type(x.dtype)


# =====================================================================
# utils.py (copied; sparse modules removed from FP16_MODULES)
# =====================================================================

FP16_MODULES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
    nn.MultiheadAttention,
    nn.GroupNorm,
)


def convert_module_to_f16(l):
    """Convert primitive modules to float16."""
    if isinstance(l, FP16_MODULES):
        for p in l.parameters():
            p.data = p.data.half()


def convert_module_to_f32(l):
    """Convert primitive modules to float32, undoing convert_module_to_f16()."""
    if isinstance(l, FP16_MODULES):
        for p in l.parameters():
            p.data = p.data.float()


# =====================================================================
# attention_block.py :: AttentionBlock (dense path, copied verbatim)
# =====================================================================


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float,
        use_fp16: bool = False,
        cond_channels: int = -1,
        use_condition: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.norm1 = LayerNorm32(channels, elementwise_affine=False)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False)

        self.attn = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

        if use_condition:
            self.cond_proj = nn.Linear(cond_channels, channels)
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels)
            )
        else:
            self.cond_proj = None
            self.modulation = None
        self.use_condition = use_condition

        if use_fp16:
            self.convert_to_fp16()
        else:
            self.convert_to_fp32()

    def convert_to_fp16(self) -> None:
        """Convert attention/MLP torso to fp16."""
        self.dtype = torch.float16
        self.attn.apply(convert_module_to_f16)
        self.mlp.apply(convert_module_to_f16)
        if self.cond_proj is not None:
            self.cond_proj.apply(convert_module_to_f16)
        if self.modulation is not None:
            self.modulation.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """Convert attention/MLP torso to fp32."""
        self.dtype = torch.float32
        self.attn.apply(convert_module_to_f32)
        self.mlp.apply(convert_module_to_f32)
        if self.cond_proj is not None:
            self.cond_proj.apply(convert_module_to_f32)
        if self.modulation is not None:
            self.modulation.apply(convert_module_to_f32)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, lp, ch = x.shape
        if self.use_condition:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.modulation(self.cond_proj(cond)).chunk(6, dim=-1)
            )
            if scale_msa.ndim == 2:
                scale_msa = scale_msa[:, None, :]
                shift_msa = shift_msa[:, None, :]
                gate_msa = gate_msa[:, None, :]
                scale_mlp = scale_mlp[:, None, :]
                shift_mlp = shift_mlp[:, None, :]
                gate_mlp = gate_mlp[:, None, :]

            h = self.norm1(x)
            h = h * (1.0 + scale_msa) + shift_msa
            h, _ = self.attn(h, h, h, need_weights=False)
            x = x + h * gate_msa
            h2 = self.norm2(x)
            h2 = h2 * (1.0 + scale_mlp) + shift_mlp
            h2 = self.mlp(h2)
            x = x + h2 * gate_mlp
        else:
            h = self.norm1(x)
            h, _ = self.attn(h, h, h, need_weights=False)
            x = x + h
            h2 = self.norm2(x)
            h2 = self.mlp(h2)
            x = x + h2
        return x


# =====================================================================
# simple_token_transformer.py :: SimpleTokenTransformer (copied verbatim)
# =====================================================================


class SimpleTokenTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        use_fp16: bool = False,
        num_tokens: int = 16,
        zero_init: bool = False,
        token_channels: Optional[int] = None,
        norm_output_tokens: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_tokens = num_tokens
        self.zero_init = zero_init
        self.norm_output_tokens = norm_output_tokens

        if num_tokens > 0:  # for internal tokens
            self.tokens = nn.Parameter(torch.randn(num_tokens, model_channels) * 0.2)
        else:
            self.tokens = None
        if token_channels is not None and token_channels != model_channels:  # for input tokens
            self.token_proj = nn.Linear(token_channels, model_channels)
        else:
            self.token_proj = None
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    channels=model_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_fp16=use_fp16,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm_out = nn.LayerNorm(model_channels)
        self.out_layer = nn.Linear(model_channels, out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
        else:
            self.convert_to_fp32()

    def initialize_weights(self) -> None:
        def _init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init)
        if self.zero_init:
            nn.init.zeros_(self.out_layer.weight)
            nn.init.zeros_(self.out_layer.bias)

    def convert_to_fp16(self) -> None:
        """Convert model torso to fp16."""
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """Convert model torso to fp32."""
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)

    def forward(
        self,
        x: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, Lp, C)
        h = self.input_layer(x.float())

        if tokens is not None:
            tokens = tokens.float()
            if self.token_proj is not None:
                tokens = self.token_proj(tokens)
            if self.tokens is not None:
                tokens = tokens + self.tokens.unsqueeze(0)
        else:
            if self.num_tokens > 0:
                tokens = self.tokens.unsqueeze(0).repeat(h.shape[0], 1, 1)

        if tokens is not None:
            num_tokens = tokens.shape[1]
            h = torch.cat([tokens, h], dim=1)
        else:
            num_tokens = 0
        h = h.type(self.dtype)
        for block in self.blocks:
            h = block(h)

        h = h.float()
        out = self.out_layer(self.norm_out(h))
        tokens, visuals = out[:, :num_tokens], out[:, num_tokens:]
        if self.norm_output_tokens:
            tokens = F.normalize(tokens, dim=-1)
        return tokens, visuals

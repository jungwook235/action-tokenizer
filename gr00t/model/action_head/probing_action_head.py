# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ProbingActionHead: VLM + Stage1-pretrained QFormer freeze, train only a small decoder.

QFormer 클래스는 기존 flow_matching_action_head_flare_qformer_action_dit.py의 것을
그대로 import해서 동일한 weight 로드 형식을 유지한다. Decoder는 3가지 (mlp / cnn /
attention) 중 하나를 config로 선택한다.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.flow_matching_action_head_flare_qformer_action_dit import QFormer


@dataclass
class ProbingActionHeadConfig(PretrainedConfig):
    """Configuration for the QFormer probing action head."""

    backbone_embedding_dim: int = field(default=1536)
    hidden_size: int = field(default=1024)
    action_dim: int = field(default=32)
    action_horizon: int = field(default=16)
    max_num_embodiments: int = field(default=32)

    # QFormer (must match stage1 ckpt shape so weights load cleanly).
    num_qformer_queries: int = field(default=64)
    qformer_num_layers: int = field(default=4)
    qformer_num_heads: int = field(default=8)
    qformer_dropout: float = field(default=0.0)
    qformer_mlp_ratio: float = field(default=4.0)
    qformer_use_self_attn: bool = field(default=True)
    qformer_use_cross_attn: bool = field(default=True)
    qformer_use_mlp: bool = field(default=True)
    qformer_use_norm: bool = field(default=True)
    qformer_use_residual: bool = field(default=True)

    # Probing decoder.
    decoder_type: str = field(default="mlp")  # "mlp" | "cnn" | "attention"
    decoder_hidden_dim: int = field(default=512)
    decoder_num_layers: int = field(default=2)
    decoder_num_heads: int = field(default=8)
    decoder_dropout: float = field(default=0.0)

    freeze_qformer: bool = field(default=True)

    # Logging.
    log_l1_denorm: bool = field(default=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# --------------------------------------------------------------------------- #
# Decoders
# --------------------------------------------------------------------------- #


class MLPProbingDecoder(nn.Module):
    """Mean-pool over QFormer tokens → MLP → (T, A)."""

    def __init__(self, in_dim, hidden_dim, num_layers, action_horizon, action_dim, dropout=0.0):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        layers = []
        cur = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(cur, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            cur = hidden_dim
        layers += [nn.Linear(cur, action_horizon * action_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, K, D)
        h = x.mean(dim=1)
        out = self.net(h)
        return out.view(-1, self.action_horizon, self.action_dim)


class CNNProbingDecoder(nn.Module):
    """1D-conv stack over QFormer tokens, adaptive pool to T, per-step linear → (T, A)."""

    def __init__(self, in_dim, hidden_dim, num_layers, action_horizon, action_dim, dropout=0.0):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        blocks = []
        cur = in_dim
        for _ in range(num_layers):
            blocks += [
                nn.Conv1d(cur, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            cur = hidden_dim
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(action_horizon)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, K, D)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        return self.head(x)


class AttentionProbingDecoder(nn.Module):
    """T learnable queries cross-attend to QFormer output, per-step linear."""

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers,
        action_horizon,
        action_dim,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.ctx_proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.queries = nn.Parameter(torch.randn(action_horizon, hidden_dim) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "ln_q": nn.LayerNorm(hidden_dim),
                        "cross": nn.MultiheadAttention(
                            hidden_dim, num_heads, dropout=dropout, batch_first=True
                        ),
                        "ln_f": nn.LayerNorm(hidden_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim * 4, hidden_dim),
                            nn.Dropout(dropout),
                        ),
                    }
                )
            )
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, K, D)
        ctx = self.ctx_proj(x)
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1).contiguous()
        for layer in self.layers:
            qn = layer["ln_q"](q)
            attn_out, _ = layer["cross"](qn, ctx, ctx)
            q = q + attn_out
            qn = layer["ln_f"](q)
            q = q + layer["ffn"](qn)
        return self.head(q)


def _build_decoder(config: ProbingActionHeadConfig) -> nn.Module:
    common = dict(
        in_dim=config.backbone_embedding_dim,
        hidden_dim=config.decoder_hidden_dim,
        num_layers=config.decoder_num_layers,
        action_horizon=config.action_horizon,
        action_dim=config.action_dim,
        dropout=config.decoder_dropout,
    )
    if config.decoder_type == "mlp":
        return MLPProbingDecoder(**common)
    if config.decoder_type == "cnn":
        return CNNProbingDecoder(**common)
    if config.decoder_type == "attention":
        return AttentionProbingDecoder(num_heads=config.decoder_num_heads, **common)
    raise ValueError(f"Unknown decoder_type: {config.decoder_type}")


# --------------------------------------------------------------------------- #
# Action head
# --------------------------------------------------------------------------- #


class ProbingActionHead(nn.Module):
    config_class = ProbingActionHeadConfig

    def __init__(self, config: ProbingActionHeadConfig):
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon

        self.qformer = QFormer(
            num_queries=config.num_qformer_queries,
            hidden_dim=config.backbone_embedding_dim,
            num_layers=config.qformer_num_layers,
            num_heads=config.qformer_num_heads,
            dropout=config.qformer_dropout,
            mlp_ratio=config.qformer_mlp_ratio,
            use_self_attn=config.qformer_use_self_attn,
            use_cross_attn=config.qformer_use_cross_attn,
            use_mlp=config.qformer_use_mlp,
            use_norm=config.qformer_use_norm,
            use_residual=config.qformer_use_residual,
        )

        self.decoder = _build_decoder(config)

        # Per-dim half-range for denormalized L1 logging
        # (training script overwrites via set_action_scale).
        self.register_buffer(
            "action_scale", torch.ones(config.action_dim, dtype=torch.float32)
        )

        # Attributes that DualBrainTrainer.log() auto-picks up. action_loss is the
        # normalized L1 (= eval_loss); l1_denorm is logged via the probing-specific
        # trainer subclass in the training script.
        self.action_loss = None
        self.l1_denorm = None
        self.flare_loss = None
        self.trm_action_loss = None
        self.trm_reasoning_loss = None

        if config.freeze_qformer:
            self.freeze_qformer_parameters()

        # First-batch diagnostic flag: print activation stats once to help debug NaN.
        self._diag_printed = False

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def freeze_qformer_parameters(self):
        for p in self.qformer.parameters():
            p.requires_grad = False
        print("[ProbingActionHead] QFormer frozen")

    def unfreeze_qformer_parameters(self):
        for p in self.qformer.parameters():
            p.requires_grad = True

    def set_action_scale(self, scale: torch.Tensor):
        """Store per-dim half-range so l1_denorm = (l1_norm_per_dim * scale).mean().

        Args:
            scale: shape (action_dim,). For min-max normalization to [-1, 1],
                   scale = (max - min) / 2.
        """
        assert scale.numel() == self.config.action_dim, (
            f"scale must have {self.config.action_dim} entries, got {scale.numel()}"
        )
        with torch.no_grad():
            self.action_scale.copy_(
                scale.to(self.action_scale.dtype).to(self.action_scale.device)
            )

    @staticmethod
    def _tstats(name: str, t):
        """One-line tensor stats including NaN/Inf count. Safe on NaN tensors."""
        if t is None:
            return f"{name}=None"
        if not isinstance(t, torch.Tensor):
            return f"{name}={t}"
        n_nan = int(torch.isnan(t).sum().item())
        n_inf = int(torch.isinf(t).sum().item())
        if t.numel() == 0:
            return f"{name}=shape{tuple(t.shape)} dtype={t.dtype} empty"
        if n_nan == t.numel():
            return f"{name}=shape{tuple(t.shape)} dtype={t.dtype} ALL-NaN"
        # finite-only stats so a single NaN doesn't poison everything
        mask = torch.isfinite(t)
        if mask.any():
            tf = t[mask].float()
            return (
                f"{name}=shape{tuple(t.shape)} dtype={t.dtype} "
                f"mean={tf.mean().item():.4g} std={tf.std().item():.4g} "
                f"min={tf.min().item():.4g} max={tf.max().item():.4g} "
                f"nan={n_nan} inf={n_inf}"
            )
        return f"{name}=shape{tuple(t.shape)} dtype={t.dtype} no-finite nan={n_nan} inf={n_inf}"

    def _scan_module_params(self, module: nn.Module, tag: str):
        """Scan all parameters of a module for NaN/Inf; print summary."""
        n_total, n_with_nan, n_with_inf = 0, 0, 0
        worst = None
        for n, p in module.named_parameters():
            n_total += 1
            nn_ = int(torch.isnan(p).sum().item())
            ni_ = int(torch.isinf(p).sum().item())
            if nn_ > 0:
                n_with_nan += 1
                if worst is None:
                    worst = (n, "nan", nn_, tuple(p.shape))
            if ni_ > 0:
                n_with_inf += 1
                if worst is None or worst[1] == "nan":
                    if worst is None:
                        worst = (n, "inf", ni_, tuple(p.shape))
        print(
            f"[DIAG/{tag}] params: total={n_total} with_nan={n_with_nan} "
            f"with_inf={n_with_inf} worst={worst}",
            flush=True,
        )

    def forward(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        future_backbone_output: Optional[BatchFeature] = None,
    ) -> BatchFeature:
        do_diag = not self._diag_printed

        if do_diag:
            print("[DIAG/probing_head] === first forward diagnostics ===", flush=True)
            self._scan_module_params(self.qformer, "qformer")
            self._scan_module_params(self.decoder, "decoder")
            bf = backbone_output["backbone_features"]
            bm = backbone_output["backbone_attention_mask"]
            print("[DIAG/probing_head] " + self._tstats("backbone_features", bf), flush=True)
            print("[DIAG/probing_head] " + self._tstats("backbone_attn_mask", bm), flush=True)

        qformer_out = self.qformer(
            context=backbone_output["backbone_features"],
            context_mask=backbone_output["backbone_attention_mask"],
        )

        if do_diag:
            print("[DIAG/probing_head] " + self._tstats("qformer_out", qformer_out), flush=True)

        pred_norm = self.decoder(qformer_out)  # (B, T, A)

        if do_diag:
            print("[DIAG/probing_head] " + self._tstats("pred_norm", pred_norm), flush=True)

        action = action_input["action"]
        action_mask = action_input["action_mask"]

        if do_diag:
            print("[DIAG/probing_head] " + self._tstats("action", action), flush=True)
            print("[DIAG/probing_head] " + self._tstats("action_mask", action_mask), flush=True)
            print("[DIAG/probing_head] " + self._tstats("action_scale", self.action_scale), flush=True)

        diff_abs = (pred_norm - action).abs() * action_mask
        denom = action_mask.sum().clamp_min(1.0)
        l1_norm = diff_abs.sum() / denom

        if self.config.log_l1_denorm:
            scale = self.action_scale.view(1, 1, -1).to(dtype=pred_norm.dtype)
            l1_denorm = (diff_abs * scale).sum() / denom
            l1_denorm_detached = l1_denorm.detach()
        else:
            l1_denorm = None
            l1_denorm_detached = None

        if do_diag:
            denorm_str = (
                f"{l1_denorm.item():.4g}" if l1_denorm is not None else "disabled"
            )
            print(
                f"[DIAG/probing_head] denom={denom.item():.4g} "
                f"l1_norm={l1_norm.item():.4g} l1_denorm={denorm_str}",
                flush=True,
            )
            print("[DIAG/probing_head] === end first forward diagnostics ===", flush=True)
            self._diag_printed = True

        self.action_loss = l1_norm
        self.l1_denorm = l1_denorm_detached
        self.flare_loss = None
        self.trm_action_loss = None
        self.trm_reasoning_loss = None

        data = {"loss": l1_norm}
        if l1_denorm_detached is not None:
            data["l1_denorm"] = l1_denorm_detached
        return BatchFeature(data=data)

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        **kwargs,
    ) -> BatchFeature:
        qformer_out = self.qformer(
            context=backbone_output["backbone_features"],
            context_mask=backbone_output["backbone_attention_mask"],
        )
        pred_norm = self.decoder(qformer_out)
        return BatchFeature(data={"action_pred": pred_norm})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

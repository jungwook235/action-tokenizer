# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T_N1_5_Probing: backbone + stage1-pretrained QFormer (both frozen) + small probing decoder.

The backbone is loaded from the base GR00T-N1.5-3B checkpoint and frozen. The QFormer
weights inside the action head are loaded from a stage1 fine-tuned checkpoint
(``qformer_checkpoint_path``) and frozen. Only the probing decoder
(``action_head.decoder.*``) is trainable.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Tuple

import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.probing_action_head import (
    ProbingActionHead,
    ProbingActionHeadConfig,
    _build_decoder,
)
from .backbone import EagleBackbone

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"


@dataclass
class GR00T_N1_5_ProbingConfig(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})
    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})
    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})
    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


def _load_qformer_state_dict(qformer_checkpoint_path: str) -> dict:
    """Filter QFormer parameters out of a stage1 checkpoint.

    Mirrors the logic in ``gr00t/model/gr00t_n1_flare_qformer_action_dit.py``'s
    from_pretrained (sharded safetensors index + single-file fallback).
    Returns a state dict whose keys are ``qformer.*`` (ready for
    ``action_head.qformer.load_state_dict(..., strict=False)``).
    """
    from safetensors.torch import load_file

    qformer_state_dict: dict[str, torch.Tensor] = {}

    def _add_key(raw_key, tensor):
        if raw_key.startswith("module.action_head.qformer."):
            new_key = raw_key.replace("module.action_head.", "")
        elif raw_key.startswith("action_head.qformer."):
            new_key = raw_key.replace("action_head.", "")
        elif raw_key.startswith("qformer."):
            new_key = raw_key
        else:
            return
        qformer_state_dict[new_key] = tensor

    if os.path.isdir(qformer_checkpoint_path):
        index_file = os.path.join(qformer_checkpoint_path, "model.safetensors.index.json")

        if os.path.exists(index_file):
            print(f"[probing] Loading QFormer from sharded safetensors: {index_file}")
            with open(index_file, "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})

            shard_to_keys: dict[str, list[str]] = {}
            for k in weight_map:
                if "qformer." in k:
                    shard_to_keys.setdefault(weight_map[k], []).append(k)

            for shard_file, keys in shard_to_keys.items():
                shard_path = os.path.join(qformer_checkpoint_path, shard_file)
                print(f"  Loading {len(keys)} QFormer keys from {shard_file}")
                shard_state = load_file(shard_path)
                for k in keys:
                    _add_key(k, shard_state[k])
        else:
            safetensor_path = os.path.join(qformer_checkpoint_path, "model.safetensors")
            bin_path = os.path.join(qformer_checkpoint_path, "pytorch_model.bin")
            if os.path.exists(safetensor_path):
                print(f"[probing] Loading QFormer from single safetensors: {safetensor_path}")
                state = load_file(safetensor_path)
            elif os.path.exists(bin_path):
                print(f"[probing] Loading QFormer from pytorch_model.bin: {bin_path}")
                state = torch.load(bin_path, map_location="cpu")
            else:
                raise FileNotFoundError(
                    f"No checkpoint file found in {qformer_checkpoint_path}"
                )
            for k, v in state.items():
                if "qformer." in k:
                    _add_key(k, v)
    else:
        if qformer_checkpoint_path.endswith(".safetensors"):
            state = load_file(qformer_checkpoint_path)
        else:
            state = torch.load(qformer_checkpoint_path, map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
        for k, v in state.items():
            if "qformer." in k:
                _add_key(k, v)

    return qformer_state_dict


class GR00T_N1_5_Probing(PreTrainedModel):
    """Backbone + frozen Stage1 QFormer + trainable probing decoder."""

    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_ProbingConfig

    def __init__(
        self,
        config: GR00T_N1_5_ProbingConfig,
        local_model_path: str,
        action_head_update: dict | None = None,
    ):
        assert isinstance(config.backbone_cfg, dict)
        super().__init__(config)
        self.local_model_path = local_model_path

        self.backbone = EagleBackbone(**config.backbone_cfg)

        # Start from ProbingActionHeadConfig defaults, then inherit dimension fields
        # from the base model's saved action_head_cfg so the QFormer / decoder shapes
        # automatically match the actual VLM output (e.g. backbone_embedding_dim=2048
        # for GR00T-N1.5-3B even though our default is 1536). Caller-provided
        # action_head_update is applied last so explicit overrides still win.
        action_head_cfg = ProbingActionHeadConfig()
        base_ah = getattr(config, "action_head_cfg", None) or {}
        for k in (
            "backbone_embedding_dim",
            "input_embedding_dim",
            "hidden_size",
            "max_num_embodiments",
        ):
            v = base_ah.get(k) if isinstance(base_ah, dict) else None
            if v is not None:
                setattr(action_head_cfg, k, v)

        if action_head_update is not None:
            for key, value in action_head_update.items():
                setattr(action_head_cfg, key, value)

        print(
            f"[probing] ProbingActionHeadConfig dims: "
            f"backbone_embedding_dim={action_head_cfg.backbone_embedding_dim}, "
            f"input_embedding_dim={action_head_cfg.input_embedding_dim}, "
            f"hidden_size={action_head_cfg.hidden_size}, "
            f"num_qformer_queries={action_head_cfg.num_qformer_queries}, "
            f"qformer_num_layers={action_head_cfg.qformer_num_layers}"
        )

        self.action_head = ProbingActionHead(action_head_cfg)

        self.action_horizon = action_head_cfg.action_horizon
        self.action_dim = action_head_cfg.action_dim
        self.compute_dtype = config.compute_dtype

        # Keep the saved config in sync with the actual action head used.
        self.config.action_head_cfg = self.action_head.config.to_dict()
        self.config.action_horizon = self.action_horizon
        self.config.action_dim = self.action_dim

        # First-batch diagnostic flag (set False to re-run after recovery).
        self._diag_printed = False

    @staticmethod
    def _tstats(name, t):
        if t is None:
            return f"{name}=None"
        if not isinstance(t, torch.Tensor):
            return f"{name}={t}"
        n_nan = int(torch.isnan(t).sum().item())
        n_inf = int(torch.isinf(t).sum().item())
        if t.numel() == 0:
            return f"{name}=shape{tuple(t.shape)} empty"
        if n_nan == t.numel():
            return f"{name}=shape{tuple(t.shape)} dtype={t.dtype} ALL-NaN"
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

    def _scan_backbone_params(self):
        n_total, n_with_nan, n_with_inf = 0, 0, 0
        worst = None
        for n, p in self.backbone.named_parameters():
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
            f"[DIAG/probing_model] backbone params: total={n_total} "
            f"with_nan={n_with_nan} with_inf={n_with_inf} worst={worst}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # I/O plumbing (mirrors gr00t_n1_flare_qformer_action_dit.py)
    # ------------------------------------------------------------------ #

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        # Only current-step (input_1). No future backbone pass for probing.
        input_1 = {
            "state": inputs["state"][:, 0:1, :],
            "state_mask": inputs["state_mask"][:, 0:1, :],
            "segmentation_target": inputs.get("segmentation_target"),
            "segmentation_target_mask": inputs.get("segmentation_target_mask"),
            "has_real_action": inputs.get("has_real_action"),
            "action": inputs["action"],
            "action_mask": inputs["action_mask"],
            "eagle_input_ids": inputs["eagle_input_ids"],
            "eagle_attention_mask": inputs["eagle_attention_mask"],
            "eagle_pixel_values": inputs["eagle_pixel_values"],
            "eagle_image_sizes": inputs["eagle_image_sizes"],
            "embodiment_id": inputs["embodiment_id"],
        }
        # Drop None values so backbone/action_head don't see them.
        input_1 = {k: v for k, v in input_1.items() if v is not None}

        backbone_inputs, action_inputs = self.prepare_input(input_1)

        do_diag = not self._diag_printed
        if do_diag:
            print("[DIAG/probing_model] === first forward diagnostics ===", flush=True)
            self._scan_backbone_params()
            for k in ("eagle_input_ids", "eagle_pixel_values", "eagle_attention_mask"):
                if k in backbone_inputs:
                    print("[DIAG/probing_model] " + self._tstats(k, backbone_inputs[k]), flush=True)

        backbone_outputs = self.backbone(backbone_inputs)

        if do_diag:
            for k in ("backbone_features", "backbone_attention_mask"):
                if k in backbone_outputs:
                    print(
                        "[DIAG/probing_model] " + self._tstats(k, backbone_outputs[k]),
                        flush=True,
                    )
            print("[DIAG/probing_model] === end first forward diagnostics ===", flush=True)
            self._diag_printed = True

        action_head_outputs = self.action_head(backbone_outputs, action_inputs)

        if LOSS_KEY not in action_head_outputs:
            raise ValueError(
                f"{ERROR_MSG}: action head did not return '{LOSS_KEY}'."
            )
        return action_head_outputs

    def get_action(self, inputs: dict) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        return self.action_head.get_action(backbone_outputs, action_inputs)

    # ------------------------------------------------------------------ #
    # Pretrained loading: base VLM backbone + stage1 QFormer
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        # Pull out probing-specific kwargs (not consumed by HF).
        qformer_checkpoint_path: str = kwargs.pop("qformer_checkpoint_path", None)
        if qformer_checkpoint_path is None:
            raise ValueError(
                "GR00T_N1_5_Probing.from_pretrained requires qformer_checkpoint_path "
                "(path to a stage1 checkpoint)."
            )

        action_horizon = kwargs.pop("action_horizon", 16)
        action_dim = kwargs.pop("action_dim", 32)

        # QFormer hyperparams (must match stage1 ckpt shape)
        num_qformer_queries = kwargs.pop("num_qformer_queries", 64)
        qformer_num_layers = kwargs.pop("qformer_num_layers", 4)
        qformer_num_heads = kwargs.pop("qformer_num_heads", 8)
        qformer_dropout = kwargs.pop("qformer_dropout", 0.0)
        qformer_mlp_ratio = kwargs.pop("qformer_mlp_ratio", 4.0)

        # Probing decoder hyperparams
        decoder_type = kwargs.pop("decoder_type", "mlp")
        decoder_hidden_dim = kwargs.pop("decoder_hidden_dim", 512)
        decoder_num_layers = kwargs.pop("decoder_num_layers", 2)
        decoder_num_heads = kwargs.pop("decoder_num_heads", 8)
        decoder_dropout = kwargs.pop("decoder_dropout", 0.0)

        action_scale = kwargs.pop("action_scale", None)
        log_l1_denorm = kwargs.pop("log_l1_denorm", True)

        # Backbone freeze flags (defaults: freeze all)
        tune_visual = kwargs.pop("tune_visual", False)
        tune_llm = kwargs.pop("tune_llm", False)

        # ------------------------------------------------------------------ #
        # Step 1: download / resolve base model snapshot
        # ------------------------------------------------------------------ #
        print(f"[probing] Loading base model from {pretrained_model_name_or_path}")
        try:
            local_model_path = snapshot_download(
                pretrained_model_name_or_path, repo_type="model"
            )
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"[probing] Treating as local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        # ------------------------------------------------------------------ #
        # Step 2: build action_head_update from probing args
        # ------------------------------------------------------------------ #
        update_action_head_cfg = {
            "action_horizon": action_horizon,
            "action_dim": action_dim,
            "num_qformer_queries": num_qformer_queries,
            "qformer_num_layers": qformer_num_layers,
            "qformer_num_heads": qformer_num_heads,
            "qformer_dropout": qformer_dropout,
            "qformer_mlp_ratio": qformer_mlp_ratio,
            "decoder_type": decoder_type,
            "decoder_hidden_dim": decoder_hidden_dim,
            "decoder_num_layers": decoder_num_layers,
            "decoder_num_heads": decoder_num_heads,
            "decoder_dropout": decoder_dropout,
            "freeze_qformer": True,
            "log_l1_denorm": log_l1_denorm,
        }

        # ------------------------------------------------------------------ #
        # Step 3: load base model. The base ckpt's action_head.* keys are
        # unexpected (different class) — HF reports them in loading_info but
        # does NOT raise. Our action_head starts random except for QFormer
        # which we load separately below.
        # ------------------------------------------------------------------ #
        pretrained_model, loading_info = super().from_pretrained(
            local_model_path,
            local_model_path=local_model_path,
            action_head_update=update_action_head_cfg,
            output_loading_info=True,
            **kwargs,
        )

        print(
            f"[probing] super().from_pretrained loaded. unexpected action_head keys "
            f"(expected — from base FlowmatchingActionHead): "
            f"{sum(1 for k in loading_info.get('unexpected_keys', []) if k.startswith('action_head.'))}"
        )

        # ------------------------------------------------------------------ #
        # Step 3.5: rebuild decoder from scratch.
        # HF from_pretrained can leave params with no checkpoint counterpart
        # (= our probing decoder) in a partially uninitialized state when meta
        # device / low_cpu_mem_usage paths are used, surfacing as a handful of
        # NaN entries in LayerNorm weights. Rebuilding guarantees fresh init.
        # ------------------------------------------------------------------ #
        old_decoder = pretrained_model.action_head.decoder
        ref_param = next(old_decoder.parameters(), None)
        ref_device = ref_param.device if ref_param is not None else pretrained_model.device
        ref_dtype = ref_param.dtype if ref_param is not None else torch.float32
        new_decoder = _build_decoder(pretrained_model.action_head.config).to(
            device=ref_device, dtype=ref_dtype
        )
        pretrained_model.action_head.decoder = new_decoder
        # Sanity check the rebuild produced finite params.
        n_total, n_nan = 0, 0
        for _, p in new_decoder.named_parameters():
            n_total += 1
            n_nan += int(torch.isnan(p).sum().item() > 0)
        print(
            f"[probing] decoder rebuilt fresh. params: total={n_total} with_nan={n_nan} "
            f"(device={ref_device}, dtype={ref_dtype})"
        )

        # ------------------------------------------------------------------ #
        # Step 4: load QFormer weights from the stage1 checkpoint
        # ------------------------------------------------------------------ #
        qformer_state = _load_qformer_state_dict(qformer_checkpoint_path)
        if len(qformer_state) == 0:
            raise RuntimeError(
                f"No QFormer parameters found in {qformer_checkpoint_path}"
            )
        missing, unexpected = pretrained_model.action_head.load_state_dict(
            qformer_state, strict=False
        )
        missing_q = [k for k in missing if "qformer." in k]
        unexpected_q = [k for k in unexpected if "qformer." in k]
        print(
            f"[probing] QFormer loaded: {len(qformer_state)} tensors, "
            f"missing_qformer={len(missing_q)}, unexpected_qformer={len(unexpected_q)}"
        )
        if missing_q:
            raise RuntimeError(
                f"[probing] QFormer missing keys after load: {missing_q[:5]} ..."
            )

        # ------------------------------------------------------------------ #
        # Step 5: freeze backbone + QFormer; only decoder is trainable
        # ------------------------------------------------------------------ #
        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.freeze_qformer_parameters()

        # Explicit decoder requires_grad=True (defensive — already True at init)
        for p in pretrained_model.action_head.decoder.parameters():
            p.requires_grad = True
        # action_scale buffer never needs grad
        pretrained_model.action_head.action_scale.requires_grad_(False)

        # ------------------------------------------------------------------ #
        # Step 6: store action scale buffer if provided
        # ------------------------------------------------------------------ #
        if action_scale is not None:
            scale_t = torch.as_tensor(action_scale, dtype=torch.float32)
            pretrained_model.action_head.set_action_scale(scale_t)

        # ------------------------------------------------------------------ #
        # Step 7: trainable param summary
        # ------------------------------------------------------------------ #
        n_total = sum(p.numel() for p in pretrained_model.parameters())
        n_train = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
        print(
            f"[probing] trainable params: {n_train:,} / total: {n_total:,} "
            f"({100.0 * n_train / max(1, n_total):.3f}%)"
        )
        trainable_modules = sorted(
            {n.split(".decoder.", 1)[0] + ".decoder" if ".decoder." in n else n
             for n, p in pretrained_model.named_parameters() if p.requires_grad}
        )
        print(f"[probing] trainable modules: {trainable_modules}")

        pretrained_model.action_horizon = action_horizon
        pretrained_model.action_dim = action_dim
        pretrained_model.config.action_horizon = action_horizon
        pretrained_model.config.action_dim = action_dim

        return pretrained_model


# Register so HF AutoModel works for our config_class as well.
# Note: existing GR00T_N1_5 model files also register "gr00t_n1_5" — last-write wins,
# which is fine because we always use the explicit class via from_pretrained.
try:
    AutoConfig.register("gr00t_n1_5", GR00T_N1_5_ProbingConfig, exist_ok=True)
except TypeError:
    # Older transformers without exist_ok parameter
    try:
        AutoConfig.register("gr00t_n1_5", GR00T_N1_5_ProbingConfig)
    except ValueError:
        pass
try:
    AutoModel.register(GR00T_N1_5_ProbingConfig, GR00T_N1_5_Probing, exist_ok=True)
except TypeError:
    try:
        AutoModel.register(GR00T_N1_5_ProbingConfig, GR00T_N1_5_Probing)
    except ValueError:
        pass

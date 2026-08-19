"""Multi-embodiment Action Latent Tokenizer V4 training (joint Stage-1).

Trains ONE V4 tokenizer jointly across several embodiments. Per-embodiment action
encoders/decoders, a SINGLE shared fusion encoder and a SINGLE shared DINO decoder
(see ``gr00t.model.action_latent_tokenizer_v4_multiemb``).

Embodiment groups are defined in a JSON config (``--embodiments-config``)::

  {"embodiments": [
    {"name": "gr1", "data_config": "fourier_gr1_arms_waist",
     "embodiment_tag": "new_embodiment", "weight": 1.0,
     "dataset_path": ["/.../gr1_unified.*"]},
    {"name": "dexjoco", "data_config": "dexjoco_dual_arm_front",
     "embodiment_tag": "new_embodiment", "weight": 1.0,
     "dataset_path": ["/.../bimanual_assembly", ...]}
  ]}

``weight`` is optional. If NO group sets a weight, sampling is plain
size-proportional (a shuffled ConcatDataset). If any weight is set, a weighted
sampler is used so each embodiment's expected sample mass equals its weight.

Cloned from ``train_action_latent_tokenizer_v4.py``. Model hyper-params are shared
across all embodiments; only each embodiment's ``action_dim`` differs.
"""

import glob
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
import transformers
import tyro
from transformers import TrainingArguments

import gr00t.experiment.data_config_v3  # noqa: F401  (register extra configs)
from gr00t.data.dataset_action_frames_v4 import ActionFramesDatasetV4
from gr00t.data.dataset_dino_cache_v4 import CachedActionFramesDatasetV4
from gr00t.data.dataset_egodex_frames_v4 import EgoDexActionFramesDataset
from gr00t.data.dataset_action_frames_v4_multiemb import (
    EmbodimentTaggedDataset,
    MultiEmbActionFramesCollator,
    WeightedEmbodimentSampler,
    build_per_index_weights,
)
from gr00t.data.merge_norm_stats import apply_merged_normalization_metadata
from gr00t.experiment.trainer import BaseSampler
from gr00t.model.action_latent_tokenizer_v4_multiemb import (
    HUMAN_DECODER_SUFFIX,
    MultiEmbActionLatentTokenizerV4,
)
from gr00t.utils.dino import DINOv3FeatureExtractor


# =====================================================================
# Trainer
# =====================================================================


class MultiEmbActionLatentV4Trainer(transformers.Trainer):
    """V4 multi-embodiment trainer: owns a frozen DINO/VGGT extractor, extracts
    feats on-the-fly per embodiment group, runs ONE model forward per step."""

    def __init__(
        self,
        *args,
        dino_model: str,
        dino_channels: int,
        feature_source: str = "dino",
        vggt_token_source: str = "dpt_out2",
        vggt_model: str = "facebook/VGGT-1B",
        vggt_image_size: int = 224,
        vggt_final_norm: str = "none",
        dino_final_norm: str = "affine",
        train_sampler_weights=None,
        embodiment_names: Optional[list] = None,
        use_seg_stream: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.feature_source = feature_source
        # Segment (cutout) stream: embedded by the SAME frozen extractor as the RGB
        # stream (no second extractor), so only two extra forward passes per group.
        self.use_seg_stream = bool(use_seg_stream)
        self._train_sampler_weights = train_sampler_weights
        self._embodiment_names = embodiment_names or []
        if feature_source == "vggt":
            from gr00t.utils.vggt_feature import VGGTFeatureExtractor

            self.dino = VGGTFeatureExtractor(
                model_name=vggt_model,
                token_source=vggt_token_source,
                image_size=vggt_image_size,
                use_compile=False,
                final_norm=vggt_final_norm,
            )
        else:
            self.dino = DINOv3FeatureExtractor(
                model_name=dino_model, use_compile=False, final_norm=dino_final_norm
            )
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False
        assert self.dino.embed_dim == dino_channels, (
            f"[{feature_source}] extractor embed_dim={self.dino.embed_dim} != "
            f"dino_channels={dino_channels}."
        )
        self._dino_on_device = False

    def _get_train_sampler(self, *args, **kwargs):
        if self._train_sampler_weights is not None:
            return WeightedEmbodimentSampler(
                self.train_dataset, self._train_sampler_weights, seed=self.args.seed
            )
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def get_eval_dataloader(self, eval_dataset=None):
        if not hasattr(self, "_cached_eval_dataloader"):
            self._cached_eval_dataloader = super().get_eval_dataloader(eval_dataset)
        return self._cached_eval_dataloader

    def _device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        return next(self.model.parameters()).device

    @torch.no_grad()
    def _extract_feats_frames(self, frame_x0, frame_x1):
        """frames (uint8 [B,3,H,W]) → DINO patch features [B, Lp, C] (fp32)."""
        device = self._device()
        if not self._dino_on_device:
            self.dino.to(device)
            self._dino_on_device = True
        f0 = frame_x0.to(device).float() / 255.0
        f1 = frame_x1.to(device).float() / 255.0
        return self._frames_to_feats(f0), self._frames_to_feats(f1)

    def _frames_to_feats(self, frames):
        """Normalized frames [B,3,H,W] in [0,1] → patch features [B, Lp, C] (fp32)."""
        if self.feature_source == "vggt":
            tok, _ = self.dino(frames)
            return tok.float()
        _, grid = self.dino(frames, return_spatial_grid=True)
        return grid.flatten(2).transpose(1, 2).float()

    def _group_feats(self, g):
        """Return (x0_feat, x1_feat) [B, Lp, C] fp32 for one embodiment group.

        Cached groups (collated by ``CachedActionFramesCollatorV4``) already carry
        ``x0_feat``/``x1_feat``; we just move + cast them to fp32 — value-identical
        to the live path's trailing ``.float()`` (the precompute stores the exact
        tensor the trainer would feed the model). Live groups carry frames, so we
        run DINO on them."""
        if "x0_feat" in g:
            device = self._device()
            return g["x0_feat"].to(device).float(), g["x1_feat"].to(device).float()
        return self._extract_feats_frames(g["frame_x0"], g["frame_x1"])

    @torch.no_grad()
    def _group_seg_feats(self, g):
        """(s0_feat, s1_feat) for one group, or (None, None) when the seg stream is off.

        The cutout frames go through the SAME frozen extractor as the RGB stream. There
        is no cached variant (the DINO cache holds only the RGB stream), so seg groups
        are always live — enforced by an assert in ``main``."""
        if not self.use_seg_stream:
            return None, None
        assert "seg_x0" in g, (
            "--use-seg-stream is on but this group's batch has no seg_x0; every "
            "embodiment group must declare a seg_dataset_root."
        )
        device = self._device()
        if not self._dino_on_device:
            self.dino.to(device)
            self._dino_on_device = True
        s0 = g["seg_x0"].to(device).float() / 255.0
        s1 = g["seg_x1"].to(device).float() / 255.0
        return self._frames_to_feats(s0), self._frames_to_feats(s1)

    def _build_groups_with_feats(self, inputs):
        """Turn the collated {embodiment_order, groups} batch into
        {embodiment_order, groups:{name:{action,x0_feat,x1_feat[,s0_feat,s1_feat]}}}."""
        order = inputs["embodiment_order"]
        if isinstance(order, (list, tuple)) and len(order) and isinstance(order[0], (list, tuple)):
            order = order[0]  # defensive: some collate paths wrap scalars
        groups = {}
        for name, g in inputs["groups"].items():
            x0_feat, x1_feat = self._group_feats(g)
            entry = {"action": g["action"], "x0_feat": x0_feat, "x1_feat": x1_feat}
            if "is_human" in g:  # [EXP-0010] per-sample domain label
                entry["is_human"] = g["is_human"]
            s0_feat, s1_feat = self._group_seg_feats(g)
            if s0_feat is not None:
                entry["s0_feat"] = s0_feat
                entry["s1_feat"] = s1_feat
            groups[name] = entry
        return {"embodiment_order": list(inputs["groups"].keys()), "groups": groups}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        batch = self._build_groups_with_feats(inputs)
        outputs = model(batch)
        loss = outputs["loss"]
        if model.training:
            if not hasattr(self, "_train_loss_buffer"):
                self._train_loss_buffer = {}
            for key, val in outputs.items():
                if key == "loss" or not isinstance(val, torch.Tensor):
                    continue
                self._train_loss_buffer.setdefault(key, []).append(val.item())
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            batch = self._build_groups_with_feats(inputs)
            outputs = model(batch)
            loss = outputs["loss"].detach()
        return (loss, None, None)

    def log(self, logs: dict, start_time=None) -> None:
        buf = getattr(self, "_train_loss_buffer", {})
        if buf and "loss" in logs:
            for key, values in buf.items():
                logs[key] = sum(values) / len(values)
            self._train_loss_buffer = {}
        try:
            super().log(logs, start_time=start_time)
        except TypeError:
            super().log(logs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model
        core = model.module if hasattr(model, "module") else model
        model.eval()

        # per-embodiment accumulators
        def _new_acc():
            return {"mse": 0.0, "l1": 0.0, "dino_l1": 0.0, "dino_cos": 0.0,
                    "seg_l1": 0.0, "seg_cos": 0.0, "n": 0}

        acc = {name: _new_acc() for name in self._embodiment_names}
        has_seg_decoder = getattr(core, "seg_dino_decoder", None) is not None
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = self._prepare_inputs(batch)
                for name, g in batch["groups"].items():
                    x0_feat, x1_feat = self._group_feats(g)
                    s0_feat, s1_feat = self._group_seg_feats(g)
                    dtype = core.action_encoders[name].action_proj.weight.dtype
                    actions = g["action"].to(device=x0_feat.device, dtype=dtype)
                    time_tok, _ = core.encode(
                        name, actions, x0_feat, x1_feat, s0_feat, s1_feat
                    )
                    preds = core.decode(name, time_tok)
                    pred_x1 = core.decode_dino(time_tok, x0_feat.to(dtype=time_tok.dtype), name=name)

                    B = actions.shape[0]
                    a = acc.setdefault(name, _new_acc())
                    a["mse"] += F.mse_loss(preds, actions).item() * B
                    a["l1"] += F.l1_loss(preds, actions).item() * B
                    a["dino_l1"] += F.l1_loss(pred_x1, x1_feat.to(dtype=pred_x1.dtype)).item() * B
                    a["dino_cos"] += (
                        1.0 - F.cosine_similarity(pred_x1, x1_feat.to(dtype=pred_x1.dtype), dim=-1).mean()
                    ).item() * B
                    if has_seg_decoder:
                        pred_s1 = core.decode_dino_seg(
                            time_tok, s0_feat.to(dtype=time_tok.dtype), name=name
                        )
                        tgt_s1 = s1_feat.to(dtype=pred_s1.dtype)
                        a["seg_l1"] += F.l1_loss(pred_s1, tgt_s1).item() * B
                        a["seg_cos"] += (
                            1.0 - F.cosine_similarity(pred_s1, tgt_s1, dim=-1).mean()
                        ).item() * B
                    a["n"] += B

        extra = {}
        for name, a in acc.items():
            if a["n"] == 0:
                continue
            extra[f"{metric_key_prefix}_{name}_recon_mse"] = a["mse"] / a["n"]
            extra[f"{metric_key_prefix}_{name}_recon_l1"] = a["l1"] / a["n"]
            extra[f"{metric_key_prefix}_{name}_dino_l1"] = a["dino_l1"] / a["n"]
            extra[f"{metric_key_prefix}_{name}_dino_cos_dist"] = a["dino_cos"] / a["n"]
            if has_seg_decoder:
                extra[f"{metric_key_prefix}_{name}_dino_seg_l1"] = a["seg_l1"] / a["n"]
                extra[f"{metric_key_prefix}_{name}_dino_seg_cos_dist"] = a["seg_cos"] / a["n"]
        if extra:
            self.log(extra)
            metrics.update(extra)
        return metrics


# =====================================================================
# Config
# =====================================================================


@dataclass
class ArgsConfig:
    """Multi-embodiment Action Latent Tokenizer V4 training config."""

    # ── Dataset (groups defined in JSON) ──
    embodiments_config: str
    normalization_mode: str = "min_max"

    # ── Action encoder (V3-style) ──
    emb_dim: int = 256
    head_dim: int = 64
    encoder_depth: int = 4
    decoder_depth: int = 2
    decoder_mode: Literal["self_attention", "cross_attention"] = "self_attention"
    pdropout: float = 0.0
    token_dim: int = 64

    # ── Fusion (RLA SimpleTokenTransformer) ──
    dino_model: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    dino_channels: int = 1024
    fusion_width: int = 1024
    fusion_depth: int = 12
    fusion_heads: int = 16

    # ── Visual feature source ──
    feature_source: Literal["dino", "vggt"] = "dino"
    vggt_token_source: Literal["aggregator", "dpt_out2"] = "dpt_out2"
    vggt_model: str = "facebook/VGGT-1B"
    vggt_image_size: int = 224
    vggt_final_norm: Literal["none", "naive"] = "none"
    dino_final_norm: Literal["affine", "naive"] = "affine"

    # ── DINO decoder ──
    dino_decoder_depth: int = 12

    # ── Segment (SAM3 cutout) DINO stream ──
    # All default-off → byte-identical to the pre-seg behavior. See the single-embodiment
    # script for the full description. Here the cutout mirror root may be set globally
    # (--seg-dataset-root) and/or per group ("seg_dataset_root" in the embodiments JSON,
    # which wins); with --use-seg-stream EVERY group must resolve to a root, so the
    # shared fusion encoder always sees the same two-stream input.
    use_seg_stream: bool = False
    seg_dataset_root: Optional[str] = None
    seg_video_subdir: str = "cutout"
    use_seg_dino_decoder: bool = False
    seg_dino_decoder_depth: Optional[int] = None
    lambda_dino_seg: float = 1.0

    # ── Per-embodiment (data-type) class token ──
    # When True, a learnable [dino_dim] class token per JSON ``class_token_id`` is
    # prepended to the DINO features entering the shared fusion + DINO decoder, so the
    # shared modules can condition on the data type. Requires ``class_token_id`` in every
    # embodiment group. Default False = byte-identical to the original behavior.
    use_embodiment_class_token: bool = False

    # ── Tokenizer finetuning (add a NEW embodiment to a pretrained joint tokenizer) ──
    # All default-off → byte-identical to a normal joint-training run. When
    # ``tokenizer_finetuning_mode`` is True, the model is built with the (new) embodiment(s)
    # in ``embodiments_config``, then the pretrained checkpoint at
    # ``finetuning_pretrained_path`` is loaded with strict=False: shared fusion + DINO
    # decoder + any shared embodiments load; the new embodiment's action encoder/decoder
    # stay randomly-init'd. ``new_class_token`` (>0) adds that many NEW learnable embodiment
    # class tokens (see model). ``finetuning_freeze_mode`` freezes every param that WAS in
    # the pretrained checkpoint, leaving only the newly-added params (new enc/dec + new class
    # tokens) trainable.
    tokenizer_finetuning_mode: bool = False
    finetuning_freeze_mode: bool = False
    new_class_token: int = 0
    finetuning_pretrained_path: Optional[str] = None

    # ── Loss ──
    lambda_recon: float = 1.0
    lambda_dino: float = 1.0
    recon_loss_type: Literal["mse", "l1"] = "mse"
    dino_loss_type: str = "l1+mse"
    dino_w_l1: float = 1.0
    dino_w_mse: float = 1.0
    dino_w_cosine: float = 1.0

    # ── VAE bottleneck ──
    use_vae: bool = False
    # Sampling toggle (only meaningful with --use-vae). True (default) → encoder
    # reparameterizes z = μ + σ·ε (existing behavior). --no-vae-sample → encoder returns μ
    # (deterministic latent) while still computing KL. Recorded as a checkpoint marker only
    # when disabled, so the ON default keeps VAE checkpoints byte-identical; Stage-2 inherits.
    vae_sample: bool = True
    lambda_kl: float = 1e-6
    kl_free_bits: float = 0.0

    # ── Frames / DINO input ──
    image_size: int = 224
    video_backend: str = "decord"

    # ── Training ──
    output_dir: str = "/tmp/action_latent_tokenizer_v4_multiemb"
    batch_size: int = 64
    max_steps: int = 100000
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    lr_scheduler_type: Literal["cosine", "constant", "constant_with_warmup"] = "constant"
    num_gpus: int = 1
    save_steps: int = 10000
    save_total_limit: int = 3
    eval_steps: Optional[int] = None
    dataloader_num_workers: int = 16
    report_to: Literal["wandb", "tensorboard"] = "wandb"
    run_name: Optional[str] = None
    wandb_project: str = "action-latent-tokenizer-v4-multiemb"
    resume: bool = False

    # ── [EXP-0010] Change A: embodiment-agnostic regularizer on the action latent ──
    # Default None/0 -> the module is never built: no params, no forward change.
    # vicreg is the recommended mode; meanshift alone collapsed in the reference study
    # (its apparent gain was a small-batch variance-shrinkage artifact), and the variance
    # hinge in vicreg is exactly the term that prevents that.
    embod_reg_mode: Optional[str] = None  # vicreg | coral | meanshift | dann
    embod_reg_weight: float = 0.0
    embod_reg_gather: bool = True  # all-gather before contrasting; effectively mandatory
    embod_reg_pool: Literal["mean", "tokens"] = "mean"
    embod_reg_vic_var: float = 1.0
    embod_reg_vic_cov: float = 0.04
    # Per-dim std floor of the variance hinge. 1.0 is the reference constant, calibrated
    # for its feature scale, not ours (measured ~1.8 pooled / ~2.2 per-token here).
    embod_reg_vic_std: float = 1.0
    embod_reg_lambda: float = 1.0  # GRL strength (dann only)
    # Fallback labelling for configs where a whole embodiment group is one domain
    # (comma-separated group names). Unused when the data carries per-sample is_human,
    # which the egopi_prq loader does.
    embod_reg_human_embodiments: Optional[str] = None

    # ── [EXP-0010] Change B: per-domain recon decoder split (encoder stays shared) ──
    split_recon_decoder: bool = False
    split_recon_decoder_init: Literal["copy", "random"] = "copy"

    # ── Validation ──
    val_ratio: float = 0.003
    val_seed: int = 42
    use_fixed_val: bool = True
    fixed_val_path: Optional[str] = None


# =====================================================================
# Group building
# =====================================================================


def _load_embodiment_groups(config: ArgsConfig):
    """Parse the JSON config, build per-embodiment train/val datasets (with
    within-group normalization merge), and return everything the trainer needs.

    Returns: (train_dataset, val_dataset, embodiment_specs, train_group_sizes,
              group_weights, any_weight_set)
    """
    with open(config.embodiments_config, "r") as f:
        spec = json.load(f)
    groups = spec["embodiments"]
    assert len(groups) >= 1, "embodiments_config must list >=1 embodiment"

    def make_dataset(path, data_config, embodiment_tag, split, use_cache=False,
                     seg_root=None):
        # use_cache: look up precomputed DINO feats from <dataset>/dino_feature_cache
        # (built by scripts/precompute_dino_features.py) instead of decoding video +
        # running DINO live. The cache identity (model / final-norm / image-size /
        # camera) MUST match these args, or the reader asserts on the meta. Mixed
        # runs are fine: only the cached embodiment(s) read the cache; others stay
        # live. Items become {action, x0_feat, x1_feat} (no frames).
        if use_cache:
            return CachedActionFramesDatasetV4(
                dataset_path=path,
                data_config_name=data_config,
                embodiment_tag=embodiment_tag,
                split=split,
                val_ratio=config.val_ratio,
                val_seed=config.val_seed,
                normalization_mode=config.normalization_mode,
                image_size=config.image_size,
                feature_source=config.feature_source,
                dino_model=config.dino_model,
                dino_final_norm=config.dino_final_norm,
                use_fixed_val=config.use_fixed_val,
                fixed_val_path=config.fixed_val_path,
                video_backend=config.video_backend,
            )
        return ActionFramesDatasetV4(
            dataset_path=path,
            data_config_name=data_config,
            embodiment_tag=embodiment_tag,
            split=split,
            val_ratio=config.val_ratio,
            val_seed=config.val_seed,
            normalization_mode=config.normalization_mode,
            image_size=config.image_size,
            video_backend=config.video_backend,
            use_fixed_val=config.use_fixed_val,
            fixed_val_path=config.fixed_val_path,
            seg_dataset_root=seg_root,
            seg_video_subdir=config.seg_video_subdir,
        )

    embodiment_specs = []
    train_wrapped, val_wrapped = [], []
    train_group_sizes, group_weights = [], []
    any_weight_set = any("weight" in g for g in groups)

    for g in groups:
        name = str(g["name"])
        embodiment_tag = g.get("embodiment_tag", "new_embodiment")
        weight = float(g.get("weight", 1.0))
        loader = str(g.get("loader", "lerobot")).lower()

        # Segment-stream root for this group: per-group JSON value wins, else the global
        # CLI flag. None when the seg stream is off → the datasets built below are
        # constructed exactly as before.
        seg_root = None
        if config.use_seg_stream:
            seg_root = g.get("seg_dataset_root") or config.seg_dataset_root
            assert seg_root, (
                f"[{name}] --use-seg-stream is set but no segment root resolved: set "
                f"\"seg_dataset_root\" in this embodiment group or pass "
                f"--seg-dataset-root."
            )
            assert loader == "lerobot", (
                f"[{name}] the segment stream is only implemented for the default "
                f"'lerobot' loader (got loader={loader!r}); that reader is the one that "
                f"mirrors the cutout directory layout."
            )

        # glob-expand paths (GR1 uses a glob; EgoDex lists task folders directly;
        # egopi_prq lists per-source paths inside "sources" instead — see below);
        # preserve order, dedup via extend.
        raw_paths = g.get("dataset_path", [])
        if isinstance(raw_paths, str):
            raw_paths = [raw_paths]
        paths = []
        for p in raw_paths:
            matched = sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
            assert matched, f"[{name}] no path matched: {p}"
            paths.extend(matched)
        for p in paths:
            assert os.path.exists(p), f"[{name}] dataset path does not exist: {p}"

        if loader == "egodex":
            # Non-LeRobot EgoDex reader: ONE dataset instance owns all task
            # folders and self-normalizes (min_max over `action_key`), so there is
            # no per-path ConcatDataset / cross-path stats merge. It emits the same
            # {action, frame_x0, frame_x1} items as ActionFramesDatasetV4, with
            # byte-identical frame preprocessing (decord RGB + Resize(linear)).
            ek = dict(
                dataset_paths=paths,
                action_horizon=int(g.get("action_horizon", 16)),
                action_key=str(g.get("action_key", "gr1_state")),
                action_offset=int(g.get("action_offset", 0)),
                stride=int(g["stride"]) if "stride" in g else None,
                video_suffix=str(g.get("video_suffix", ".mp4")),
                stats_max_episodes=g.get("stats_max_episodes", 3000),
                image_size=config.image_size,
                val_ratio=config.val_ratio,
                val_seed=config.val_seed,
                video_backend=config.video_backend,
            )
            train_ds = EgoDexActionFramesDataset(split="train", **ek)
            val_ds = EgoDexActionFramesDataset(split="val", **ek)
            desc = f"egodex:{ek['action_key']}"
            feats_label = "live"
        elif loader == "egopi_prq":
            # EgoPi shared {p,r,q} 15D action space (gr00t/data/dataset_egopi_prq_v4.py):
            # robot (openarm FK cache) + human (eef→prq mapping) sources land in ONE
            # embodiment group → a single action encoder/decoder serves both datasets.
            # Actions are EgoPi-normalized (merged robot∪human min-max from prq_stats),
            # so there is NO within-group LeRobot stats merge here (the sources have
            # different raw action keys anyway). DINO feature caches are REQUIRED.
            from gr00t.data.dataset_egopi_prq_v4 import EgoPiPrqCachedDatasetV4

            prq_stats = g["prq_stats"]
            filter_json = g.get("filter")
            sources = g["sources"]
            assert sources, f"[{name}] egopi_prq needs a non-empty 'sources' list"
            paths = [s["dataset_path"] for s in sources]
            for p in paths:
                assert os.path.exists(p), f"[{name}] dataset path does not exist: {p}"
            assert os.path.exists(prq_stats), f"[{name}] prq_stats not found: {prq_stats}"

            def make_prq(src, split):
                return EgoPiPrqCachedDatasetV4(
                    prq_mode=src["mode"],
                    prq_stats_path=prq_stats,
                    fk_cache_h5=src.get("fk_cache"),
                    filter_json=filter_json,
                    filter_tag=src.get("filter_tag"),
                    dataset_path=src["dataset_path"],
                    data_config_name=src["data_config"],
                    embodiment_tag=embodiment_tag,
                    split=split,
                    val_ratio=config.val_ratio,
                    val_seed=config.val_seed,
                    normalization_mode=config.normalization_mode,
                    image_size=config.image_size,
                    feature_source=config.feature_source,
                    dino_model=config.dino_model,
                    dino_final_norm=config.dino_final_norm,
                    use_fixed_val=config.use_fixed_val,
                    fixed_val_path=config.fixed_val_path,
                    video_backend=config.video_backend,
                )

            g_train = [make_prq(s, "train") for s in sources]
            g_val = [make_prq(s, "val") for s in sources]
            train_ds = g_train[0] if len(g_train) == 1 else torch.utils.data.ConcatDataset(g_train)
            val_ds = g_val[0] if len(g_val) == 1 else torch.utils.data.ConcatDataset(g_val)
            n_robot = sum(1 for s in sources if s["mode"] == "robot")
            desc = f"egopi_prq (robot={n_robot}, human={len(sources) - n_robot})"
            feats_label = "CACHED"
        else:
            data_config = g["data_config"]
            use_cache = bool(g.get("use_dino_cache", False))
            assert not (use_cache and seg_root is not None), (
                f"[{name}] use_dino_cache is incompatible with the segment stream: the "
                f"precomputed cache holds only the RGB stream's features."
            )
            g_train = [make_dataset(p, data_config, embodiment_tag, "train", use_cache, seg_root)
                       for p in paths]
            g_val = [make_dataset(p, data_config, embodiment_tag, "val", use_cache, seg_root)
                     for p in paths]

            # Merge normalization stats WITHIN this embodiment only (different
            # embodiments have different action keys/dims → cross-merge is invalid).
            apply_merged_normalization_metadata(g_train, g_train + g_val)

            train_ds = g_train[0] if len(g_train) == 1 else torch.utils.data.ConcatDataset(g_train)
            val_ds = g_val[0] if len(g_val) == 1 else torch.utils.data.ConcatDataset(g_val)
            desc = f"data_config={data_config}"
            feats_label = "CACHED" if use_cache else "live"

        # action_dim / action_horizon from a sample
        sample = train_ds[0]
        action_horizon, action_dim = sample["action"].shape
        spec_entry = {"name": name, "action_dim": int(action_dim),
                      "action_horizon": int(action_horizon)}
        if config.use_embodiment_class_token:
            assert "class_token_id" in g, (
                f"[{name}] class_token_id is required in the embodiments JSON when "
                f"--use-embodiment-class-token is set."
            )
            spec_entry["class_token_id"] = int(g["class_token_id"])
        embodiment_specs.append(spec_entry)

        train_wrapped.append(EmbodimentTaggedDataset(train_ds, name))
        val_wrapped.append(EmbodimentTaggedDataset(val_ds, name))
        train_group_sizes.append(len(train_ds))
        group_weights.append(weight)
        ct_label = (f" class_token_id={spec_entry['class_token_id']}"
                    if config.use_embodiment_class_token else "")
        seg_label = f" seg_root={seg_root}" if seg_root is not None else ""
        print(f"[group:{name}] {desc} action_dim={action_dim} "
              f"action_horizon={action_horizon} train={len(train_ds)} val={len(val_ds)} "
              f"weight={weight} ({len(paths)} path(s)) dino_feats={feats_label}"
              f"{ct_label}{seg_label}")

    # all embodiments must share action_horizon
    horizons = {s["action_horizon"] for s in embodiment_specs}
    assert len(horizons) == 1, f"embodiments must share action_horizon; got {horizons}"

    train_dataset = (train_wrapped[0] if len(train_wrapped) == 1
                     else torch.utils.data.ConcatDataset(train_wrapped))
    val_dataset = (val_wrapped[0] if len(val_wrapped) == 1
                   else torch.utils.data.ConcatDataset(val_wrapped))
    return (train_dataset, val_dataset, embodiment_specs, train_group_sizes,
            group_weights, any_weight_set)


# =====================================================================
# Model builder
# =====================================================================


def _load_pretrained_state_dict(path: str, device: str = "cpu") -> dict:
    """Load a pretrained joint-tokenizer state_dict for finetuning.

    Accepts an HF Trainer checkpoint dir (model.safetensors / pytorch_model.bin) or a raw
    .pt file (``model_state_dict`` / ``state_dict`` / bare dict). Mirrors the resolution in
    ``ActionLatentTokenizerWrapper.from_checkpoint``.
    """
    if os.path.isdir(path):
        safetensors_path = os.path.join(path, "model.safetensors")
        pt_path = os.path.join(path, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file

            return load_file(safetensors_path, device=device)
        if os.path.exists(pt_path):
            return torch.load(pt_path, map_location=device, weights_only=False)
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin found in {path}"
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def _build_model(config: ArgsConfig, embodiment_specs, action_horizon,
                 num_pretrain_class_tokens: int = 0):
    return MultiEmbActionLatentTokenizerV4(
        embodiment_specs=embodiment_specs,
        action_horizon=action_horizon,
        emb_dim=config.emb_dim,
        head_dim=config.head_dim,
        encoder_depth=config.encoder_depth,
        decoder_depth=config.decoder_depth,
        decoder_mode=config.decoder_mode,
        pdropout=config.pdropout,
        token_dim=config.token_dim,
        dino_dim=config.dino_channels,
        fusion_width=config.fusion_width,
        fusion_depth=config.fusion_depth,
        fusion_heads=config.fusion_heads,
        dino_decoder_depth=config.dino_decoder_depth,
        seg_dino_decoder_depth=config.seg_dino_decoder_depth,
        use_seg_stream=config.use_seg_stream,
        use_seg_dino_decoder=config.use_seg_dino_decoder,
        lambda_dino_seg=config.lambda_dino_seg,
        use_vae=config.use_vae,
        vae_sample=config.vae_sample,
        kl_free_bits=config.kl_free_bits,
        lambda_recon=config.lambda_recon,
        lambda_dino=config.lambda_dino,
        lambda_kl=config.lambda_kl,
        recon_loss_type=config.recon_loss_type,
        dino_loss_type=config.dino_loss_type,
        dino_loss_weights={
            "l1": config.dino_w_l1,
            "mse": config.dino_w_mse,
            "cosine": config.dino_w_cosine,
        },
        feature_source=config.feature_source,
        vggt_token_source=config.vggt_token_source,
        vggt_image_size=config.vggt_image_size,
        vggt_model=config.vggt_model,
        vggt_final_norm=config.vggt_final_norm,
        dino_final_norm=config.dino_final_norm,
        use_embodiment_class_token=config.use_embodiment_class_token,
        tokenizer_finetuning_mode=config.tokenizer_finetuning_mode,
        new_class_token=config.new_class_token,
        num_pretrain_class_tokens=num_pretrain_class_tokens,
        embod_reg_mode=config.embod_reg_mode or "",
        embod_reg_weight=config.embod_reg_weight,
        embod_reg_gather=config.embod_reg_gather,
        embod_reg_pool=config.embod_reg_pool,
        embod_reg_vic_var=config.embod_reg_vic_var,
        embod_reg_vic_cov=config.embod_reg_vic_cov,
        embod_reg_vic_std=config.embod_reg_vic_std,
        embod_reg_lambda=config.embod_reg_lambda,
        embod_reg_human_names=[
            s.strip() for s in (config.embod_reg_human_embodiments or "").split(",") if s.strip()
        ],
        split_recon_decoder=config.split_recon_decoder,
    )


# =====================================================================
# Main
# =====================================================================


def main(config: ArgsConfig):
    # Segment-stream arg validation (all no-ops when the flags are off). The per-group
    # root resolution / cache-conflict checks live in _load_embodiment_groups.
    if config.use_seg_dino_decoder:
        assert config.use_seg_stream, (
            "--use-seg-dino-decoder requires --use-seg-stream."
        )
    if config.use_seg_stream:
        assert config.seg_dataset_root is None or os.path.isdir(config.seg_dataset_root), (
            f"--seg-dataset-root does not exist: {config.seg_dataset_root}"
        )
        print(
            f"[seg-stream] ON  root={config.seg_dataset_root} "
            f"subdir={config.seg_video_subdir} "
            f"seg_dino_decoder={config.use_seg_dino_decoder} "
            f"lambda_dino_seg={config.lambda_dino_seg}"
        )

    # Finetuning-mode arg validation (no-op off the finetuning path).
    if config.tokenizer_finetuning_mode:
        assert config.finetuning_pretrained_path, (
            "tokenizer_finetuning_mode requires --finetuning-pretrained-path "
            "(the pretrained joint tokenizer checkpoint to adapt)."
        )
    if config.finetuning_freeze_mode:
        assert config.tokenizer_finetuning_mode, (
            "finetuning_freeze_mode requires tokenizer_finetuning_mode."
        )

    # [EXP-0010] arg validation (no-ops when both features are off).
    if config.embod_reg_mode:
        assert config.embod_reg_weight > 0, (
            f"--embod-reg-mode {config.embod_reg_mode} is set but --embod-reg-weight is "
            f"{config.embod_reg_weight}; the regularizer would be a no-op."
        )
        print(f"[embod-reg] ON mode={config.embod_reg_mode} weight={config.embod_reg_weight} "
              f"pool={config.embod_reg_pool} gather={config.embod_reg_gather} "
              f"vic_var={config.embod_reg_vic_var} vic_cov={config.embod_reg_vic_cov} "
              f"vic_std={config.embod_reg_vic_std}")
    if config.split_recon_decoder:
        print(f"[split-decoder] ON init={config.split_recon_decoder_init} "
              f"(shared action encoder + per-domain recon decoders)")

    (train_dataset, val_dataset, embodiment_specs, train_group_sizes,
     group_weights, any_weight_set) = _load_embodiment_groups(config)

    action_horizon = embodiment_specs[0]["action_horizon"]

    # Finetuning: read the pretrained checkpoint up front so the model's base class-token
    # parameter is sized to match (loads strict), and to compute the new-param set below.
    pretrained_sd = None
    num_pretrain_class_tokens = 0
    if config.tokenizer_finetuning_mode:
        pretrained_sd = _load_pretrained_state_dict(config.finetuning_pretrained_path)
        if config.use_embodiment_class_token:
            # Base class-token count from the pretrained checkpoint. If the pretrained
            # tokenizer had NO class tokens (base=0), every class token is newly added via
            # finetuning_class_token — a prompt-tuning-style adaptation where the frozen
            # fusion learns to attend to a brand-new learnable token. That requires
            # --new-class-token > 0.
            num_pretrain_class_tokens = int(
                pretrained_sd["embodiment_class_token"].shape[0]
                if "embodiment_class_token" in pretrained_sd else 0
            )
            if num_pretrain_class_tokens == 0:
                assert config.new_class_token > 0, (
                    "pretrained checkpoint has no class tokens ('embodiment_class_token' "
                    "absent); forcing class tokens in finetuning requires --new-class-token > 0."
                )

    model = _build_model(config, embodiment_specs, action_horizon, num_pretrain_class_tokens)

    # Whether HF will resume from an existing finetuning checkpoint in output_dir (in which
    # case that checkpoint — already carrying the new params — is loaded by trainer.train,
    # so we must NOT overwrite it with the pretrained weights here).
    is_resuming = bool(
        config.resume
        and os.path.isdir(config.output_dir)
        and transformers.trainer_utils.get_last_checkpoint(config.output_dir) is not None
    )

    if config.tokenizer_finetuning_mode:
        # New params = model params absent from the pretrained checkpoint (new embodiment's
        # action encoder/decoder + finetuning_class_token). Everything else must be shared /
        # loadable; a shared param going "missing" signals a config mismatch → fail loud.
        # ``seg_dino_decoder.*`` is also legitimately new: enabling the segment DINO
        # decoder on top of a tokenizer pretrained without it adds a fresh module (the
        # seg-stream fusion concat itself adds no params, so it never shows up here).
        new_param_names = [n for n, _ in model.named_parameters() if n not in pretrained_sd]
        for n in new_param_names:
            assert n.startswith(("action_encoders.", "recon_decoders.", "seg_dino_decoder.",
                                 "embod_reg.")) or n == "finetuning_class_token", (
                f"[finetune] unexpected new (missing-from-checkpoint) param {n!r}; the "
                f"pretrained checkpoint likely has a different config (fusion/decoder/etc.)."
            )
        if not is_resuming:
            missing, unexpected = model.load_state_dict(pretrained_sd, strict=False)
            print(f"[finetune] loaded pretrained weights from {config.finetuning_pretrained_path}")
            print(f"[finetune] new trainable params ({len(new_param_names)} tensors): {new_param_names}")
            print(f"[finetune] skipped {len(unexpected)} unexpected (other-embodiment) checkpoint keys")
        else:
            print(f"[finetune] resuming finetuning checkpoint in {config.output_dir}; "
                  "skipping pretrained load (HF resume restores all params).")
        if config.finetuning_freeze_mode:
            trainable = set(new_param_names)
            for n, p in model.named_parameters():
                p.requires_grad = n in trainable
            print(f"[finetune] freeze mode ON: only {len(trainable)} newly-added param "
                  "tensors train; all pretrained/shared modules frozen.")

    # [EXP-0010] Change B init: start the human recon decoder as an exact copy of the
    # (already-loaded, pretrained) robot decoder, so enabling the split does not perturb
    # the loss at step 0 -- the twins diverge only as their own domain's gradients arrive.
    # --split-recon-decoder-init random keeps the fresh init instead.
    if (config.split_recon_decoder and config.split_recon_decoder_init == "copy"
            and not is_resuming):
        n_copied = 0
        for nm in [k for k in model.recon_decoders.keys()
                   if not k.endswith(HUMAN_DECODER_SUFFIX)]:
            twin = nm + HUMAN_DECODER_SUFFIX
            if twin in model.recon_decoders:
                model.recon_decoders[twin].load_state_dict(
                    model.recon_decoders[nm].state_dict()
                )
                n_copied += 1
        print(f"[split-decoder] human recon decoder copy-initialized from the robot "
              f"decoder ({n_copied} pair(s))")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable:,}")

    # Sampler weights: only when at least one group set a weight.
    sampler_weights = None
    if any_weight_set:
        sampler_weights = build_per_index_weights(train_group_sizes, group_weights)
        print(f"[sampler] weighted (group weights={group_weights})")
    else:
        print("[sampler] size-proportional (no weights set)")

    eval_steps = config.eval_steps if config.eval_steps is not None else config.save_steps

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.run_name,
        remove_unused_columns=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=10,
        num_train_epochs=9999,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit if config.save_total_limit > 0 else None,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        report_to=config.report_to,
        seed=42,
        # per-step a rank's batch may miss an embodiment → its encoder/decoder
        # params are unused that step. Guard against DDP unused-param errors.
        ddp_find_unused_parameters=True,
    )

    world_size = max(1, config.num_gpus)
    micro_batch_global = config.batch_size * world_size
    steps_per_epoch = math.ceil(len(train_dataset) / micro_batch_global)
    print(
        f"[TrainInfo] train={len(train_dataset):,} val={len(val_dataset):,} "
        f"micro_batch(global)={micro_batch_global:,} steps/epoch={steps_per_epoch:,}"
    )

    # [EXP-0010] the per-sample domain label is only stacked when something needs it.
    collator = MultiEmbActionFramesCollator(
        pass_is_human=bool(config.embod_reg_mode) or config.split_recon_decoder
    )

    trainer = MultiEmbActionLatentV4Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        dino_model=config.dino_model,
        dino_channels=config.dino_channels,
        feature_source=config.feature_source,
        vggt_token_source=config.vggt_token_source,
        vggt_model=config.vggt_model,
        vggt_image_size=config.vggt_image_size,
        vggt_final_norm=config.vggt_final_norm,
        dino_final_norm=config.dino_final_norm,
        train_sampler_weights=sampler_weights,
        embodiment_names=[s["name"] for s in embodiment_specs],
        use_seg_stream=config.use_seg_stream,
    )

    if config.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = config.wandb_project
        os.environ["WANDB_DIR"] = config.output_dir

    # --resume should be a no-op when there is no checkpoint yet: HF Trainer
    # raises if resume_from_checkpoint=True but output_dir has no checkpoint.
    resume_from_checkpoint = config.resume
    if config.resume:
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(
            config.output_dir
        ) if os.path.isdir(config.output_dir) else None
        if last_checkpoint is None:
            print(
                f"[resume] no checkpoint found in {config.output_dir}; "
                "starting from scratch."
            )
            resume_from_checkpoint = False
        else:
            print(f"[resume] resuming from {last_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        save_path = os.path.join(config.output_dir, "multiemb_full.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "tokenizer_version": "v4_multiemb",
                    "embodiments": embodiment_specs,
                    "action_horizon": action_horizon,
                    "emb_dim": config.emb_dim,
                    "head_dim": config.head_dim,
                    "encoder_depth": config.encoder_depth,
                    "decoder_depth": config.decoder_depth,
                    "decoder_mode": config.decoder_mode,
                    "token_dim": config.token_dim,
                    "dino_model": config.dino_model,
                    "dino_channels": config.dino_channels,
                    "fusion_width": config.fusion_width,
                    "fusion_depth": config.fusion_depth,
                    "fusion_heads": config.fusion_heads,
                    "dino_decoder_depth": config.dino_decoder_depth,
                    "use_seg_stream": config.use_seg_stream,
                    "seg_dataset_root": config.seg_dataset_root,
                    "seg_video_subdir": config.seg_video_subdir,
                    "use_seg_dino_decoder": config.use_seg_dino_decoder,
                    "seg_dino_decoder_depth": config.seg_dino_decoder_depth,
                    "lambda_dino_seg": config.lambda_dino_seg,
                    "feature_source": config.feature_source,
                    "vggt_token_source": config.vggt_token_source,
                    "vggt_model": config.vggt_model,
                    "vggt_image_size": config.vggt_image_size,
                    "vggt_final_norm": config.vggt_final_norm,
                    "dino_final_norm": config.dino_final_norm,
                    "lambda_recon": config.lambda_recon,
                    "lambda_dino": config.lambda_dino,
                    "lambda_kl": config.lambda_kl,
                    "use_vae": config.use_vae,
                    "vae_sample": config.vae_sample,
                    "kl_free_bits": config.kl_free_bits,
                    "recon_loss_type": config.recon_loss_type,
                    "dino_loss_type": config.dino_loss_type,
                    "image_size": config.image_size,
                    "use_embodiment_class_token": config.use_embodiment_class_token,
                    "tokenizer_finetuning_mode": config.tokenizer_finetuning_mode,
                    "finetuning_freeze_mode": config.finetuning_freeze_mode,
                    "new_class_token": config.new_class_token,
                    "num_pretrain_class_tokens": num_pretrain_class_tokens,
                    "finetuning_pretrained_path": config.finetuning_pretrained_path,
                },
            },
            save_path,
        )
        print(f"Final multi-embodiment model saved to {save_path}")
        print(f"  embodiments: {[s['name'] for s in embodiment_specs]}")
        print("  Stage-2: --actlat-tokenizer-path "
              f"{save_path} --embodiment-id <name>")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)

    print("\n" + "=" * 60)
    print("MULTI-EMBODIMENT ACTION LATENT TOKENIZER V4 TRAINING:")
    print("=" * 60)
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert config.num_gpus <= available_gpus, (
        f"Requested GPUs ({config.num_gpus}) > available ({available_gpus})."
    )
    assert config.num_gpus > 0

    if config.num_gpus == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            script_path = Path(__file__).absolute()
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",
                str(script_path),
            ]
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    cmd.append(
                        f"--{key.replace('_', '-')}" if value else f"--no-{key.replace('_', '-')}"
                    )
                elif value is None:
                    continue
                else:
                    cmd.append(f"--{key.replace('_', '-')}")
                    if isinstance(value, list):
                        for v in value:
                            cmd.append(str(v))
                    else:
                        cmd.append(str(value))

            print("Running torchrun:", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)

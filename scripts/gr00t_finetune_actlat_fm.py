# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training script for action latent flow matching VLA and baseline VLA comparison.

All training conditions (dataset, optimizer, hyperparameters, trainer) are identical.
Only the model differs based on --mode:
  --mode actlat_fm  (default): flow matching in latent action space (gr00t_n1_actlat_fm.py)
  --mode vla:                  standard GR00T VLA baseline (gr00t_n1.py)
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import torch
import tyro
from transformers import TrainingArguments, set_seed

import numpy as np

from gr00t.data.dataset import LeRobotMixtureDataset
from gr00t.data.dataset_actlat_fm import (
    ActlatFMDataCollator,
    LeRobotSingleDatasetActlatFM,
)
from gr00t.data.dataset_actlat_fm_v4 import LeRobotSingleDatasetActlatFMV4
from gr00t.data.dataset_actlat_fm_v4_cached import LeRobotSingleDatasetActlatFMV4Cached
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.trainer_actlat_fm import ActlatFMTrainer
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.experiment import CheckpointFormatCallback
from gr00t.utils.peft import get_lora_model


@dataclass
class ArgsConfig:
    """Configuration for actlat-FM and baseline VLA fine-tuning."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories."""

    output_dir: str = "/tmp/gr00t_actlat_fm"
    """Directory to save model checkpoints."""

    mode: Literal["actlat_fm", "vla"] = "actlat_fm"
    """Training mode. 'actlat_fm': latent-space flow matching. 'vla': standard GR00T baseline.
    All other training conditions (dataset, optimizer, hyperparameters) are identical."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_waist"
    """Data configuration name from DATA_CONFIG_MAP."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag for the dataset."""

    video_backend: str = "decord"
    """Video backend to use."""

    # Training parameters
    batch_size: int = 32
    """Per-device training batch size."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Save checkpoint every N steps."""

    save_total_limit: int = 8
    """Max number of checkpoints to keep; oldest are deleted first. Set <=0 to keep all."""

    learning_rate: float = 1e-4
    """Learning rate."""

    weight_decay: float = 1e-5
    """Weight decay."""

    warmup_ratio: float = 0.05
    """Warmup ratio."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path to the base pretrained model."""

    tune_llm: bool = False
    """Whether to fine-tune the LLM backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the visual backbone."""

    tune_projector: bool = True
    """Whether to fine-tune the action head projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the DiT."""

    resume: bool = False
    """Whether to resume from checkpoint."""

    load_action_head: bool = True
    """Whether to load pretrained action head."""

    # Action latent tokenizer (actlat_fm mode only)
    actlat_tokenizer_path: str = ""
    """Path to pretrained action latent tokenizer checkpoint. (actlat_fm mode only)"""

    actlat_target_tokens: str = "all"
    """Which tokens to use as target: 'time', 'global_time', 'time_hand', 'all'. (actlat_fm mode only)"""

    actlat_vae_no_sample: bool = False
    """If True, force the VAE tokenizer to produce a DETERMINISTIC posterior-mean (μ)
    latent target, regardless of whether the tokenizer checkpoint was trained with
    sampling. Default False = use the tokenizer checkpoint's own setting. Only affects
    VAE tokenizers. (actlat_fm mode only)"""

    embodiment_id: str = ""
    """Which embodiment to select from a multi-embodiment joint V4 tokenizer
    checkpoint (matches the JSON 'name', e.g. 'gr1' / 'dexjoco'). Leave empty for
    ordinary single-embodiment tokenizers. (actlat_fm mode only)"""

    actlat_frames: bool = False
    """If True, use the V4 dataset that also yields (frame_x0, frame_x1) so the
    V4 (RLA-DINO) tokenizer can compute DINO-dependent latent targets. Required
    when the tokenizer is V4; harmless to leave False for v2/v3."""

    frame_image_size: int = 224
    """Square resize for the V4 frame pair (must match V4 tokenizer training)."""

    actlat_frame_video_key: str = ""
    """Camera key fed to the V4 tokenizer for its (frame_x0, frame_x1) latent
    target. Empty -> use the data-config's `tokenizer_frame_video_key`, else its
    first video key. Lets VLA training keep the tokenizer single-camera (matching
    its training) even when the backbone consumes multiple cameras."""

    # ── Latent z-norm (port of the WAM DiT4DiT actlat_latent_norm) ──
    actlat_latent_norm: bool = False
    """If True, per-dim z-normalize the tokenizer's latent FM target with
    PRECOMPUTED dataset-wide stats (like VLA action normalization); inference
    de-normalizes the predicted latent before the tokenizer decoder. Requires
    --actlat-latent-stats-path (written by a prior dump pass). (actlat_fm only)"""

    actlat_latent_stats_path: str = ""
    """JSON with per-dim {"mean": [D], "std": [D]} of the RAW latent target,
    written by --actlat-dump-latent-stats-path. Required with --actlat-latent-norm."""

    actlat_dump_latent_stats_path: str = ""
    """Dump-ONLY mode: stream every training sample once through the SAME
    dataset/collator/tokenizer path used at train time, write per-dim latent
    mean/std JSON here, and exit WITHOUT training. Must run with --num-gpus 1
    and WITHOUT --actlat-latent-norm (the stats must be RAW-latent moments)."""

    actlat_dump_max_samples: int = 0
    """Early-stop the stats dump after N samples (0 = full pass)."""

    # ── Segment (SAM3 cutout) DINO stream (V4 + actlat_frames only) ──
    actlat_seg_dataset_root: str = ""
    """Root of the SAM3 cutout mirror (e.g.
    .../GR00T-X-Embodiment-Sim_sam3_robot_task). REQUIRED when the Stage-1 tokenizer was
    trained with --use-seg-stream: its encoder consumes the cutout stream's DINO features
    too, so the latent target cannot be computed without them. The dataset attaches
    (seg_x0, seg_x1) read at the SAME two steps as (frame_x0, frame_x1); the frozen
    tokenizer embeds them with the same extractor. Empty (default) → unchanged."""

    actlat_seg_video_subdir: str = "cutout"
    """Subdir inside the seg mirror holding the videos ("cutout")."""

    # ── Precomputed DINO cache (V4 + actlat_frames only) ──
    use_dino_cache: bool = False
    """If True, read precomputed DINO feats (x0_feat/x1_feat) from
    <dataset>/dino_feature_cache/<key> instead of decoding the frame pair and
    running DINO. Reuses the SAME cache built for Stage-1
    (scripts/precompute_dino_features.py). Default False → unchanged frame path."""
    dino_cache_model: str = "facebook/dinov2-large"
    """DINO model id the cache was built with (cache-key component). Must match
    the Stage-1 tokenizer / precompute run."""
    dino_cache_final_norm: str = "naive"
    """DINO final-norm the cache was built with ('naive' or 'affine')."""
    dino_cache_feature_source: str = "dino"
    """Feature source the cache was built with ('dino')."""

    # ── EgoPi prq action override (openarm_prq tokenizer embodiment only) ──
    actlat_prq_stats: str = ""
    """Path to egopi_prq_stats.json. When set (with --actlat-frames and
    --use-dino-cache), the dataset replaces the LeRobot joint action with the
    FK-converted EgoPi 15D {p,rot6d,q} chunk, normalized with the merged
    robot∪human min-max stats — exactly matching the Stage-1 training of the
    tokenizer's openarm_prq embodiment. Empty → unchanged action path."""
    actlat_prq_fk_cache_dir: str = ""
    """Directory of per-object FK cache h5 files (<dir>/<dataset_dir_name>.h5),
    built by RLDX-1-egopi scripts/build_egopi_cache.py. Required with
    --actlat-prq-stats."""
    actlat_prq_filter_json: str = ""
    """egopi_filter.json (per-episode left-arm gate; tag = dataset dir name).
    Applied AFTER the fixed-val split, matching Stage-1. Required with
    --actlat-prq-stats."""

    # Validation
    val_ratio: float = 0.003
    """Fraction of episodes for validation."""

    val_seed: int = 42
    """Seed used to derive the train/val split when the fixed-val file is created."""

    use_fixed_val: bool = True
    """If True, load/persist the train/val split as JSON so it matches the Stage-1
    tokenizer split. The file is created on first use if missing."""

    fixed_val_path: Optional[str] = None
    """Explicit path to the fixed-val JSON. If None, defaults to
    <dataset>/meta/fixed_val_split.json (one file per dataset)."""

    eval_steps: int = 1000
    """Evaluate every N steps."""

    # LoRA
    lora_rank: int = 0
    """LoRA rank. 0 disables LoRA."""

    lora_alpha: int = 16
    """LoRA alpha."""

    lora_dropout: float = 0.1
    """LoRA dropout."""

    lora_full_model: bool = False
    """Apply LoRA to full model (vs action head only)."""

    # Other
    dataloader_num_workers: int = 8
    """Number of dataloader workers."""
  
    balance_dataset_weights: bool = True
    """Balance dataset weights by trajectory length."""

    balance_trajectory_weights: bool = True
    """Balance trajectory weights."""

    report_to: Literal["wandb", "tensorboard", "none"] = "wandb"
    """Reporting backend."""

    run_name: str = None
    """WandB run name."""

    # Kept for compatibility but not used
    vision_token_num: int = 0
    """Number of vision tokens (kept for compat, not used)."""


def _load_model(config: ArgsConfig, data_action_horizon: int, data_action_dim: int):
    """Load model based on --mode. Only this part differs between modes."""
    if config.mode == "vla":
        from gr00t.model.gr00t_n1 import GR00T_N1_5

        model = GR00T_N1_5.from_pretrained(
            pretrained_model_name_or_path=config.base_model_path,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
            load_action_head=config.load_action_head,
        )

        # Adjust action_horizon / action_dim / num_target_vision_tokens if data differs from pretrained
        need_recreate = (
            data_action_horizon != model.action_head.config.action_horizon
            or data_action_dim != model.action_head.config.action_dim
            or config.vision_token_num != model.action_head.config.num_target_vision_tokens
        )
        if need_recreate:
            print(
                f"Recreating action head: "
                f"action_horizon {model.action_head.config.action_horizon}→{data_action_horizon}, "
                f"action_dim {model.action_head.config.action_dim}→{data_action_dim}, "
                f"num_target_vision_tokens {model.action_head.config.num_target_vision_tokens}→{config.vision_token_num}"
            )
            from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead

            old_num_vision_tokens = model.action_head.config.num_target_vision_tokens

            if config.load_action_head:
                # Save shape-safe weights before recreation
                dit_state = model.action_head.model.state_dict()
                state_enc_state = model.action_head.state_encoder.state_dict()
                vlln_state = model.action_head.vlln.state_dict()
                vl_sa_state = model.action_head.vl_self_attention.state_dict()
                future_tok_state = None
                if hasattr(model.action_head, "future_tokens"):
                    future_tok_state = model.action_head.future_tokens.state_dict()
                pos_state = None
                if hasattr(model.action_head, "position_embedding"):
                    pos_state = model.action_head.position_embedding.state_dict()

            new_cfg = model.action_head.config
            new_cfg.action_horizon = data_action_horizon
            new_cfg.action_dim = data_action_dim
            new_cfg.num_target_vision_tokens = config.vision_token_num
            model.action_head = FlowmatchingActionHead(new_cfg)

            if config.load_action_head:
                # Restore shape-safe weights; action_encoder/action_decoder stay randomly initialized
                model.action_head.model.load_state_dict(dit_state)
                model.action_head.state_encoder.load_state_dict(state_enc_state)
                model.action_head.vlln.load_state_dict(vlln_state)
                model.action_head.vl_self_attention.load_state_dict(vl_sa_state)
                # Only restore future_tokens if shape matches (num_target_vision_tokens unchanged)
                if (
                    future_tok_state is not None
                    and hasattr(model.action_head, "future_tokens")
                    and old_num_vision_tokens == config.vision_token_num
                ):
                    model.action_head.future_tokens.load_state_dict(future_tok_state)
                if pos_state is not None and hasattr(model.action_head, "position_embedding"):
                    model.action_head.position_embedding.load_state_dict(pos_state)
                print("[VLA] Pretrained weights preserved, action_encoder/decoder reinitialized")
            else:
                print("[VLA] All action head weights randomly initialized")

            model.config.action_horizon = data_action_horizon
            model.config.action_dim = data_action_dim
            model.action_horizon = data_action_horizon
            model.action_dim = data_action_dim
            model.config.action_head_cfg["action_horizon"] = data_action_horizon
            model.config.action_head_cfg["action_dim"] = data_action_dim
            model.config.action_head_cfg["num_target_vision_tokens"] = config.vision_token_num
            model.action_head.set_trainable_parameters(
                tune_projector=config.tune_projector,
                tune_diffusion_model=config.tune_diffusion_model,
            )

    else:  # actlat_fm
        from gr00t.model.gr00t_n1_actlat_fm import GR00T_N1_5

        model = GR00T_N1_5.from_pretrained(
            pretrained_model_name_or_path=config.base_model_path,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
            load_action_head=config.load_action_head,
            vision_token_num=config.vision_token_num,
            resume=config.resume,
            actlat_tokenizer_path=config.actlat_tokenizer_path if config.actlat_tokenizer_path else None,
            actlat_target_tokens=config.actlat_target_tokens,
            actlat_embodiment_id=config.embodiment_id if config.embodiment_id else None,
            actlat_vae_no_sample=config.actlat_vae_no_sample,
            actlat_latent_norm=config.actlat_latent_norm,
            actlat_latent_stats_path=config.actlat_latent_stats_path,
        )

    return model


def dump_actlat_latent_stats(model, train_dataset, config: ArgsConfig):
    """One-shot preprocessing: per-dim mean/std of the frozen tokenizer's latent target.

    Port of WAM's dump_actlat_latent_stats (Isaac-GR00T-AlinVLA
    gr00t/model/wam_dit4dit/setup.py). Streams every training step of every dataset
    exactly once (sequential pass over each child dataset's all_steps — the mixture's
    random __getitem__ is bypassed) through the SAME collator/tokenizer path used in
    GR00T_N1_5.forward, accumulates per-latent-dim moments in float64 over all
    (sample, token) pairs, and writes them as JSON. The result feeds
    --actlat-latent-norm / --actlat-latent-stats-path.
    """
    tok = model.action_latent_tokenizer
    assert tok is not None, (
        "latent-stats dump requires a loaded actlat tokenizer (--actlat-tokenizer-path)"
    )
    assert getattr(model, "_actlat_latent_mean", None) is None, (
        "latent-stats dump must run with actlat_latent_norm OFF — the stats must be "
        "moments of the RAW latent, not of an already-normalized one."
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Only the tokenizer is needed — keep the 3B backbone on CPU.
    tok.to(device)

    children = (
        train_dataset.datasets
        if isinstance(train_dataset, LeRobotMixtureDataset)
        else [train_dataset]
    )
    data_collator = ActlatFMDataCollator()

    latent_dim = int(tok.emb_dim)
    n_entries = 0   # (sample, token) pairs — the per-dim moment population
    n_samples = 0   # dataset samples seen
    num_tokens = None
    acc_sum = torch.zeros(latent_dim, dtype=torch.float64, device=device)
    acc_sumsq = torch.zeros(latent_dim, dtype=torch.float64, device=device)
    acc_min = torch.full((latent_dim,), float("inf"), dtype=torch.float64, device=device)
    acc_max = torch.full((latent_dim,), float("-inf"), dtype=torch.float64, device=device)
    max_samples = int(config.actlat_dump_max_samples or 0)

    print(
        f"[actlat-stats] streaming {len(children)} dataset(s) once: "
        f"latent_dim={latent_dim}, batch_size={config.batch_size}, "
        f"max_samples={max_samples or 'ALL'}"
    )

    def _to_dev(x):
        return x.to(device) if torch.is_tensor(x) else x

    stop = False
    with torch.inference_mode():
        for child_idx, child in enumerate(children):
            if stop:
                break
            loader = torch.utils.data.DataLoader(
                child,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.dataloader_num_workers,
                collate_fn=data_collator,
                pin_memory=(device.type == "cuda"),
            )
            for batch_idx, batch in enumerate(loader):
                # Same fields GR00T_N1_5.forward hands to get_latent_target.
                latent = tok.get_latent_target(
                    batch["action"].to(device=device, dtype=torch.float32),
                    target_tokens=config.actlat_target_tokens,
                    x0=_to_dev(batch.get("frame_x0")),
                    x1=_to_dev(batch.get("frame_x1")),
                    x0_feat=_to_dev(batch.get("x0_feat")),
                    x1_feat=_to_dev(batch.get("x1_feat")),
                    s0=_to_dev(batch.get("seg_x0")),
                    s1=_to_dev(batch.get("seg_x1")),
                    s0_feat=_to_dev(batch.get("s0_feat")),
                    s1_feat=_to_dev(batch.get("s1_feat")),
                )  # [B, N, D] raw latent
                lat = latent.double()
                num_tokens = int(lat.shape[1])
                flat = lat.reshape(-1, latent_dim)  # [(B*N), D]
                acc_sum += flat.sum(dim=0)
                acc_sumsq += (flat * flat).sum(dim=0)
                acc_min = torch.minimum(acc_min, flat.min(dim=0).values)
                acc_max = torch.maximum(acc_max, flat.max(dim=0).values)
                n_entries += flat.shape[0]
                n_samples += int(lat.shape[0])
                if batch_idx % 50 == 0:
                    print(
                        f"[actlat-stats] dataset {child_idx + 1}/{len(children)}: "
                        f"{n_samples} samples ({n_entries} latent tokens) ..."
                    )
                if max_samples and n_samples >= max_samples:
                    print(f"[actlat-stats] max_samples={max_samples} reached — stopping early.")
                    stop = True
                    break

    if n_entries == 0:
        raise RuntimeError("actlat latent-stats dump saw 0 samples — dataset empty?")

    mean = acc_sum / n_entries
    # Population variance; guard fp roundoff (E[x^2] - E[x]^2 can dip below 0).
    var = (acc_sumsq / n_entries - mean * mean).clamp_min(0.0)
    std = var.sqrt()

    payload = {
        "latent_dim": latent_dim,
        "num_tokens": num_tokens,
        "num_samples": n_samples,
        "num_entries": n_entries,
        "target_tokens": config.actlat_target_tokens,
        "tokenizer_path": config.actlat_tokenizer_path,
        "embodiment_id": config.embodiment_id,
        "vae_no_sample": config.actlat_vae_no_sample,
        "dataset_path": config.dataset_path,
        "mean": mean.cpu().tolist(),
        "std": std.cpu().tolist(),
        "var": var.cpu().tolist(),
        "min": acc_min.cpu().tolist(),
        "max": acc_max.cpu().tolist(),
    }
    out = Path(config.actlat_dump_latent_stats_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(
        f"[actlat-stats] wrote {out} — samples={n_samples}, tokens/sample={num_tokens}, "
        f"mean range=[{mean.min().item():.4f}, {mean.max().item():.4f}], "
        f"std range=[{std.min().item():.4f}, {std.max().item():.4f}]"
    )


def main(config: ArgsConfig):
    """Main training function. Dataset, optimizer, trainer are identical across modes."""
    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # ------------ step 1: dataset (identical for all modes) ------------
    # V4 (RLA-DINO) tokenizer needs the chunk start/end frames for its latent
    # target → swap in the frame-pair dataset and pass the frame options.
    if config.actlat_frames:
        DatasetCls = (
            LeRobotSingleDatasetActlatFMV4Cached
            if config.use_dino_cache
            else LeRobotSingleDatasetActlatFMV4
        )
        # The V4 tokenizer was trained on a single camera, so its latent-target
        # frame pair must come from that same camera even though the VLA backbone
        # consumes all of `video_keys`. Priority: CLI override ->
        # data-config's `tokenizer_frame_video_key` -> first video key (legacy).
        frame_video_key = (
            config.actlat_frame_video_key
            or getattr(data_config_cls, "tokenizer_frame_video_key", None)
            or data_config_cls.video_keys[0]
        )
        assert frame_video_key.startswith("video."), (
            f"actlat frame_video_key must start with 'video.': {frame_video_key!r}"
        )
        print(
            f"[actlat] tokenizer frame_video_key = {frame_video_key} "
            f"(backbone video_keys = {data_config_cls.video_keys}) "
            f"use_dino_cache={config.use_dino_cache}"
        )
        frame_kwargs = dict(
            frame_video_key=frame_video_key,
            frame_image_size=config.frame_image_size,
            frame_action_horizon=len(data_config_cls.action_indices),
        )
        if config.actlat_seg_dataset_root:
            # Segment (cutout) stream for seg-trained V4 tokenizers. The cached-DINO
            # dataset skips the frame decode entirely and its cache holds only the RGB
            # stream, so the two are mutually exclusive.
            assert not config.use_dino_cache, (
                "--actlat-seg-dataset-root is incompatible with --use-dino-cache: the "
                "precomputed cache holds only the RGB stream's DINO features."
            )
            assert os.path.isdir(config.actlat_seg_dataset_root), (
                f"--actlat-seg-dataset-root does not exist: "
                f"{config.actlat_seg_dataset_root}"
            )
            frame_kwargs.update(
                seg_dataset_root=config.actlat_seg_dataset_root,
                seg_video_subdir=config.actlat_seg_video_subdir,
            )
            print(
                f"[actlat] segment stream ON  root={config.actlat_seg_dataset_root} "
                f"subdir={config.actlat_seg_video_subdir}"
            )
        if config.use_dino_cache:
            # Cache-key components — must match the precompute / Stage-1 config so
            # the reader resolves to the existing cache directory.
            frame_kwargs.update(
                feature_source=config.dino_cache_feature_source,
                dino_model=config.dino_cache_model,
                dino_final_norm=config.dino_cache_final_norm,
            )
        if config.actlat_prq_stats:
            # EgoPi prq action override: the openarm_prq tokenizer embodiment was
            # trained on FK-converted 15D {p,rot6d,q} actions (merged min-max
            # normalization), so the action handed to the frozen tokenizer must be
            # rebuilt the same way — the LeRobot joint action is replaced per sample.
            assert config.use_dino_cache, (
                "--actlat-prq-stats requires --use-dino-cache (prq dataset extends "
                "the cached V4 dataset)"
            )
            assert config.actlat_prq_fk_cache_dir and config.actlat_prq_filter_json, (
                "--actlat-prq-stats requires --actlat-prq-fk-cache-dir and "
                "--actlat-prq-filter-json"
            )
            from gr00t.data.dataset_actlat_fm_v4_cached_prq import (
                LeRobotSingleDatasetActlatFMV4CachedPrq,
            )

            DatasetCls = LeRobotSingleDatasetActlatFMV4CachedPrq
            frame_kwargs.update(
                prq_stats_path=config.actlat_prq_stats,
                fk_cache_dir=config.actlat_prq_fk_cache_dir,
                filter_json=config.actlat_prq_filter_json,
            )
            print(
                f"[actlat] EgoPi prq action override ON "
                f"(stats={config.actlat_prq_stats}, "
                f"fk_cache_dir={config.actlat_prq_fk_cache_dir})"
            )
    else:
        DatasetCls = LeRobotSingleDatasetActlatFM
        frame_kwargs = {}

    if len(config.dataset_path) == 1:
        train_dataset = DatasetCls(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
            split="train",
            val_ratio=config.val_ratio,
            val_seed=config.val_seed,
            use_fixed_val=config.use_fixed_val,
            fixed_val_path=config.fixed_val_path,
            **frame_kwargs,
        )
    else:
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            single_datasets.append(
                DatasetCls(
                    dataset_path=p,
                    modality_configs=modality_configs,
                    transforms=transforms,
                    embodiment_tag=embodiment_tag,
                    video_backend=config.video_backend,
                    split="train",
                    val_ratio=config.val_ratio,
                    val_seed=config.val_seed,
                    use_fixed_val=config.use_fixed_val,
                    fixed_val_path=config.fixed_val_path,
                    **frame_kwargs,
                )
            )
        train_dataset = LeRobotMixtureDataset(
            data_mixture=[(d, 1.0) for d in single_datasets],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
        )

    if len(config.dataset_path) == 1:
        eval_dataset = DatasetCls(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
            split="val",
            val_ratio=config.val_ratio,
            val_seed=config.val_seed,
            use_fixed_val=config.use_fixed_val,
            fixed_val_path=config.fixed_val_path,
            **frame_kwargs,
        )
    else:
        # Build the eval set as a mixture over the val split of ALL datasets,
        # using the same weighting as train (weight 1.0 each + balance flags),
        # so validation is sampled proportionally to the train mixture.
        eval_single_datasets = []
        for p in config.dataset_path:
            d = DatasetCls(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
                split="val",
                val_ratio=config.val_ratio,
                val_seed=config.val_seed,
                use_fixed_val=config.use_fixed_val,
                fixed_val_path=config.fixed_val_path,
                **frame_kwargs,
            )
            # A dataset can end up with 0 val episodes (e.g. the prq left-arm
            # gate drops the single fixed-val episode of cup/doll). An empty
            # dataset breaks the mixture weighting → skip it, loudly.
            if len(d) == 0:
                print(f"[eval] SKIPPING {p}: 0 val steps after split/filter")
                continue
            eval_single_datasets.append(d)
        assert eval_single_datasets, "all eval datasets are empty"
        eval_dataset = LeRobotMixtureDataset(
            data_mixture=[(d, 1.0) for d in eval_single_datasets],
            mode="val",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
        )

    # All datasets (train children + eval children) share the SAME `transforms`
    # object, mutated in place; each __init__ overwrites the normalization stats.
    # Re-apply the train mixture's MERGED stats LAST so both train and eval
    # normalize with the intended merged statistics (not whichever single
    # dataset was constructed last).
    if isinstance(train_dataset, LeRobotMixtureDataset):
        merged_metadata = next(iter(train_dataset.merged_metadata.values()))
        transforms.set_metadata(merged_metadata)

    print(f"Train dataset: {len(train_dataset)} steps")
    print(f"Eval dataset: {len(eval_dataset)} steps")

    # ---- DEBUG: print normalization stats ACTUALLY applied at train time ----
    # The transforms object is SHARED by reference across all datasets and is
    # mutated in place by set_transforms_metadata. The mixture's __getitem__
    # runs `child.transforms(...)`, i.e. this same shared object, so the ground
    # truth of what normalization is applied lives in its Normalizers, not in
    # any separate metadata dict. We read the live Normalizers here.
    if int(os.environ.get("RANK", 0)) == 0:
        # The dataset the train dataloader actually pulls from.
        if isinstance(train_dataset, LeRobotMixtureDataset):
            live_tf = train_dataset.datasets[0].transforms
            print("[norm] reading LIVE transforms from mixture child[0]")
        else:
            live_tf = train_dataset.transforms
            print("[norm] reading LIVE transforms from single dataset")

        # State and action are normalized by SEPARATE transforms in the GR00T
        # pipeline, each carrying its own `_normalizers`. Collect ALL of them so
        # the APPLIED log covers both state.* and action.* (a single `next()`
        # would grab only the first, i.e. state).
        sa_trs = [
            t for t in getattr(live_tf, "transforms", []) if hasattr(t, "_normalizers")
        ]
        # Helper: print both min/max and q01/q99 so the APPLIED-vs-MERGED
        # comparison works regardless of the active mode (min_max OR q99).
        def _fmt(st):
            def g(k):
                # Rotation keys (e.g. rotation_6d) carry only min/max overrides,
                # so q01/q99 may be absent — print NA instead of crashing.
                if k not in st:
                    return "NA"
                v = st[k]
                return v.tolist() if hasattr(v, "tolist") else list(v)
            return (
                f"min={g('min')} max={g('max')} q01={g('q01')} q99={g('q99')}"
            )

        if not sa_trs:
            print("[norm] WARNING: no StateActionTransform with _normalizers found")
        else:
            for sa_tr in sa_trs:
                for key, normd in sa_tr._normalizers.items():
                    # normd.statistics holds the FULL stat dict (model_dump) used.
                    print(f"[norm] APPLIED {key} mode={normd.mode} {_fmt(normd.statistics)}")

        # For comparison: the MERGED metadata the mixture computed (intended).
        # If MERGED differs from APPLIED for the mode's stat (min/max when
        # min_max, q01/q99 when q99), the shared transform was overwritten
        # (e.g. by eval_dataset built from dataset_path[0]).
        if isinstance(train_dataset, LeRobotMixtureDataset):
            merged = next(iter(train_dataset.merged_metadata.values()))
            for subkey, v in merged.statistics.action.items():
                print(
                    f"[norm] MERGED action.{subkey} "
                    f"min={np.asarray(v.min).tolist()} max={np.asarray(v.max).tolist()} "
                    f"q01={np.asarray(v.q01).tolist()} q99={np.asarray(v.q99).tolist()}"
                )
    # ----------------------------------------------------------------------

    # ------------ step 2: model (only part that differs by mode) ------------
    # Get actual action shape from data to match model config
    data_action_horizon = len(data_config_cls.action_indices)
    sample_action = train_dataset[0]["action"]  # [T, D]
    data_action_dim = sample_action.shape[-1]
    print(f"Data action shape: horizon={data_action_horizon}, dim={data_action_dim}")

    model = _load_model(config, data_action_horizon, data_action_dim)

    # ------------ latent-stats dump mode: write the JSON and exit (no training) --
    if config.actlat_dump_latent_stats_path:
        dump_actlat_latent_stats(model, train_dataset, config)
        return

    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    # ------------ step 3: training args (identical for all modes) ------------
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.run_name,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit if config.save_total_limit > 0 else None,
        report_to=config.report_to,
        seed=42,
        do_eval=True,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # ------------ step 4: trainer & run (identical for all modes) ------------
    data_collator = ActlatFMDataCollator()
    compute_dtype = torch.float16 if training_args.bf16 else torch.float32
    set_seed(training_args.seed)

    training_args.run_name = (
        training_args.output_dir.split("/")[-1]
        if training_args.run_name is None
        else training_args.run_name
    )

    if config.report_to == "wandb":
        if "WANDB_PROJECT" not in os.environ:
            os.environ["WANDB_PROJECT"] = "gr00t-actlat-fm"
        os.environ["WANDB_DIR"] = config.output_dir
        training_args.report_to = ["wandb"]
    elif config.report_to == "none":
        training_args.report_to = []
    else:
        training_args.report_to = ["tensorboard"]

    trainer = ActlatFMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_dtype=compute_dtype,
    )

    ckpt_format_callback = CheckpointFormatCallback(
        run_name=training_args.run_name,
        exp_cfg_dir=Path(training_args.output_dir) / "experiment_cfg",
    )
    trainer.add_callback(ckpt_format_callback)

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        exp_cfg_dir = Path(training_args.output_dir) / "experiment_cfg"
        exp_cfg_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "mode": config.mode,
            "dataset_path": config.dataset_path,
            "data_config": config.data_config,
            "embodiment_tag": config.embodiment_tag,
            "actlat_tokenizer_path": config.actlat_tokenizer_path,
            "actlat_target_tokens": config.actlat_target_tokens,
            "actlat_vae_no_sample": config.actlat_vae_no_sample,
            "embodiment_id": config.embodiment_id,
            "actlat_latent_norm": config.actlat_latent_norm,
            "actlat_latent_stats_path": config.actlat_latent_stats_path,
        }
        # Persist the normalization statistics actually applied (whole-mixture
        # merged stats for a LeRobotMixtureDataset) so inference can reuse them
        # directly from metadata.json instead of re-reading and re-merging every
        # source dataset's meta/stats.json. CheckpointFormatCallback copies this
        # experiment_cfg dir into each checkpoint, so the stats travel with them.
        if isinstance(train_dataset, LeRobotMixtureDataset):
            stats_meta = next(iter(train_dataset.merged_metadata.values()))
        else:
            stats_meta = getattr(train_dataset, "metadata", None)
        if stats_meta is not None:
            metadata["statistics"] = stats_meta.statistics.model_dump(mode="json")
        with open(exp_cfg_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

    print(f"Train dataloader length: {len(trainer.get_train_dataloader())}")
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Eval dataset length: {len(eval_dataset)}")
    print(
        f"GPU memory before training: "
        f"{torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB"
    )

    trainer.train(resume_from_checkpoint=config.resume if config.resume else None)


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)

    print("\n" + "=" * 50)
    print(f"GR00T FINE-TUNING CONFIGURATION (mode={config.mode}):")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    # ── Latent z-norm / stats-dump validation ──
    if config.actlat_dump_latent_stats_path:
        assert config.mode == "actlat_fm" and config.actlat_tokenizer_path, (
            "--actlat-dump-latent-stats-path requires --mode actlat_fm and "
            "--actlat-tokenizer-path"
        )
        assert config.num_gpus == 1, (
            "latent-stats dump must run single-process — relaunch with --num-gpus 1"
        )
        assert not config.actlat_latent_norm, (
            "latent-stats dump must run WITHOUT --actlat-latent-norm (the stats must "
            "be RAW-latent moments)"
        )
    if config.actlat_latent_norm:
        assert config.mode == "actlat_fm" and config.actlat_tokenizer_path, (
            "--actlat-latent-norm requires --mode actlat_fm and --actlat-tokenizer-path"
        )
        assert config.actlat_latent_stats_path or config.resume, (
            "--actlat-latent-norm requires --actlat-latent-stats-path (run the dump "
            "pass first), unless resuming a checkpoint with embedded stats"
        )

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert config.num_gpus <= available_gpus, (
        f"Requested {config.num_gpus} GPUs but only {available_gpus} available"
    )
    assert config.num_gpus > 0
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--no-{key.replace('_', '-')}")
                else:
                    cmd.append(f"--{key.replace('_', '-')}")
                    if isinstance(value, list):
                        for v in value:
                            cmd.append(str(v))
                    else:
                        cmd.append(str(value))

            print("Running torchrun command:", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)

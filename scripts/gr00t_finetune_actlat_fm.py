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

    actlat_frames: bool = False
    """If True, use the V4 dataset that also yields (frame_x0, frame_x1) so the
    V4 (RLA-DINO) tokenizer can compute DINO-dependent latent targets. Required
    when the tokenizer is V4; harmless to leave False for v2/v3."""

    frame_image_size: int = 224
    """Square resize for the V4 frame pair (must match V4 tokenizer training)."""

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
        )

    return model


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
        DatasetCls = LeRobotSingleDatasetActlatFMV4
        frame_kwargs = dict(
            frame_video_key=data_config_cls.video_keys[0],
            frame_image_size=config.frame_image_size,
            frame_action_horizon=len(data_config_cls.action_indices),
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
            eval_single_datasets.append(
                DatasetCls(
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
            )
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

        sa_tr = next(
            (t for t in getattr(live_tf, "transforms", []) if hasattr(t, "_normalizers")),
            None,
        )
        # Helper: print both min/max and q01/q99 so the APPLIED-vs-MERGED
        # comparison works regardless of the active mode (min_max OR q99).
        def _fmt(st):
            def g(k):
                v = st[k]
                return v.tolist() if hasattr(v, "tolist") else list(v)
            return (
                f"min={g('min')} max={g('max')} q01={g('q01')} q99={g('q99')}"
            )

        if sa_tr is None:
            print("[norm] WARNING: no StateActionTransform with _normalizers found")
        else:
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
        }
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

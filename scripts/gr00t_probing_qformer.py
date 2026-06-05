# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Train a small probing decoder on top of a frozen Stage1 QFormer.

Pipeline:
  - Load base GR00T-N1.5-3B (Eagle backbone), freeze.
  - Init ProbingActionHead with chosen decoder type (mlp / cnn / attention).
  - Load Stage1 QFormer weights from --qformer-checkpoint-path, freeze.
  - Train only the decoder. Compute L1 (normalized) as the training/val loss.
  - Log L1 (denormalized using dataset stats) to wandb as a side metric.

Train/val split via gr00t.data.dataset_val.LeRobotSingleDatasetWithSplit.
Validation runs every --eval-steps and reports eval_loss (mean L1 norm) +
eval_l1_denorm (mean L1 denorm) to wandb.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset
from gr00t.data.dataset_val import LeRobotSingleDatasetWithSplit
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner_val import TrainRunnerWithVal
from gr00t.experiment.trainer import DualBrainTrainer
from gr00t.model.gr00t_n1_qformer_probing import GR00T_N1_5_Probing
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class ArgsConfig:
    """Configuration for QFormer probing fine-tuning."""

    # Dataset
    dataset_path: List[str]
    output_dir: str = "/tmp/gr00t_probing"
    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_waist_flare"
    val_ratio: float = 0.003
    val_seed: int = 42
    eval_steps: int = None

    # Model
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    qformer_checkpoint_path: str = ""  # REQUIRED at runtime
    action_horizon: int = 16
    action_dim: int = 32

    # QFormer (must match stage1 ckpt)
    num_qformer_queries: int = 64
    qformer_num_layers: int = 4
    qformer_num_heads: int = 8
    qformer_dropout: float = 0.0
    qformer_mlp_ratio: float = 4.0

    # Probing decoder
    decoder_type: Literal["mlp", "cnn", "attention"] = "mlp"
    decoder_hidden_dim: int = 512
    decoder_num_layers: int = 2
    decoder_num_heads: int = 8
    decoder_dropout: float = 0.0

    # Training
    batch_size: int = 64
    max_steps: int = 20000
    save_steps: int = 5000
    num_gpus: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.02
    lr_scheduler_type: Literal["cosine", "constant", "constant_with_warmup"] = "cosine"
    seed: int = 42
    dataloader_num_workers: int = 24
    # Eval dataloader uses fewer non-persistent workers to avoid RAM accumulation
    # (decord opens leak across long-lived workers when eval-steps is small).
    eval_dataloader_num_workers: int = 2
    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    run_name: str = None

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    video_backend: Literal["decord", "torchvision_av", "opencv"] = "decord"

    # Logging: denormalized L1 (computed from dataset min/max stats). Default ON.
    log_l1_denorm: bool = True

    resume: bool = False


# --------------------------------------------------------------------------- #
# Action scale extraction (min_max → half-range per dim, padded to action_dim)
# --------------------------------------------------------------------------- #


def _extract_action_scale(dataset, action_keys: List[str], action_dim: int) -> torch.Tensor:
    """Build per-dim half-range = (max - min) / 2 for min_max normalization.

    Concatenates action_keys in order (matches ConcatTransform's action_concat_order),
    pads the rest of action_dim with zeros (those positions are masked out by action_mask
    anyway).
    """
    stats = dataset.metadata.statistics.action  # dict[str, DatasetStatisticalValues]
    scale = torch.zeros(action_dim, dtype=torch.float32)
    offset = 0
    for full_key in action_keys:
        subkey = full_key.split(".", 1)[1] if "." in full_key else full_key
        if subkey not in stats:
            raise KeyError(
                f"[probing] action subkey '{subkey}' not found in dataset stats "
                f"(available: {list(stats.keys())})"
            )
        mn = np.asarray(stats[subkey].min, dtype=np.float32)
        mx = np.asarray(stats[subkey].max, dtype=np.float32)
        dim = mn.shape[0]
        if offset + dim > action_dim:
            raise ValueError(
                f"[probing] action_dim overflow: {action_keys} requires "
                f">{offset + dim} dims, but action_dim={action_dim}"
            )
        # Min-max normalization maps x to [-1, 1] via 2*(x-min)/(max-min) - 1.
        # So |x - y| in original space = |x_norm - y_norm| * (max - min) / 2.
        half_range = (mx - mn) / 2.0
        half_range = np.where(mx > mn, half_range, 0.0)
        scale[offset : offset + dim] = torch.from_numpy(half_range.astype(np.float32))
        offset += dim
    print(
        f"[probing] action_scale built ({offset}/{action_dim} dims filled): "
        f"min={float(scale[:offset].min()):.4f}, "
        f"max={float(scale[:offset].max()):.4f}, "
        f"mean={float(scale[:offset].mean()):.4f}"
    )
    return scale


# --------------------------------------------------------------------------- #
# Probing trainer + runner (adds l1_denorm logging on top of DualBrainTrainer)
# --------------------------------------------------------------------------- #


class ProbingTrainer(DualBrainTrainer):
    """DualBrainTrainer + train-time and eval-time l1_denorm logging."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._eval_l1_denorm_sum = 0.0
        self._eval_l1_denorm_count = 0

    def _get_action_head(self):
        return self.model.module.action_head if hasattr(self.model, "module") else self.model.action_head

    def get_eval_dataloader(self, eval_dataset=None):
        """Build eval dataloader with fewer non-persistent workers.

        With persistent_workers=True for train, HF caches the eval dataloader
        too. Combined with very frequent eval (e.g. eval_steps=100), the eval
        worker pool accumulates decord/libavformat state and leaks RAM until
        SLURM OOM-kills the job. Forcing persistent_workers=False during eval
        construction makes HF skip the cache so each eval spawns fresh workers
        and releases them on completion.
        """
        orig_workers = self.args.dataloader_num_workers
        orig_persistent = self.args.dataloader_persistent_workers
        try:
            self.args.dataloader_num_workers = min(
                getattr(self, "_eval_dataloader_num_workers", 2), orig_workers
            )
            self.args.dataloader_persistent_workers = False
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.args.dataloader_num_workers = orig_workers
            self.args.dataloader_persistent_workers = orig_persistent

    def log(self, logs, start_time=None):
        action_head = self._get_action_head()
        # Pick up the most recent batch's denormalized L1 — for training this gives
        # per-logging-step value (HF averages train loss internally, but the head
        # attribute is single-batch).
        if "eval_l1_denorm" not in logs and getattr(action_head, "l1_denorm", None) is not None:
            v = action_head.l1_denorm
            logs["loss/l1_denorm"] = v.item() if torch.is_tensor(v) else float(v)
        super().log(logs, start_time)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # HF Trainer's default prediction_step calls `model(**inputs)`, which
        # unpacks the input dict as kwargs. Our GR00T_N1_5_Probing.forward
        # signature is `forward(self, inputs: dict)` — single positional dict.
        # Replicate the dict-passing pattern used by DualBrainTrainer.compute_loss.
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(inputs)
            loss = outputs["loss"].detach()

        action_head = self._get_action_head()
        denorm = getattr(action_head, "l1_denorm", None)
        if denorm is not None:
            try:
                bs = inputs["action"].shape[0]
            except (KeyError, AttributeError):
                bs = 1
            self._eval_l1_denorm_sum += (
                denorm.item() if torch.is_tensor(denorm) else float(denorm)
            ) * bs
            self._eval_l1_denorm_count += bs

        # HF eval loop expects (loss, logits, labels). We don't need logits/labels
        # for L1 metric — None is acceptable when prediction_loss_only is True
        # (and harmless otherwise since we report only loss/eval_l1_denorm).
        return (loss, None, None)

    def evaluation_loop(self, *args, **kwargs):
        self._eval_l1_denorm_sum = 0.0
        self._eval_l1_denorm_count = 0
        output = super().evaluation_loop(*args, **kwargs)
        if self._eval_l1_denorm_count > 0:
            mean_denorm = self._eval_l1_denorm_sum / self._eval_l1_denorm_count
            # Use HF's metric_key_prefix convention. evaluation_loop is called with
            # metric_key_prefix kwarg; we can't see it directly here so use 'eval'.
            output.metrics["eval_l1_denorm"] = mean_denorm
        return output


class ProbingTrainRunner(TrainRunnerWithVal):
    """TrainRunnerWithVal but instantiates ProbingTrainer instead of DualBrainTrainer."""

    def create_trainer(
        self,
        model,
        training_args,
        train_dataset,
        data_collator,
        compute_dtype,
        global_batch_size=None,
    ):
        if global_batch_size is not None:
            bs = training_args.per_device_train_batch_size
            num_gpus = torch.cuda.device_count()
            grad_acc = max(1, global_batch_size // (bs * num_gpus))
            training_args.gradient_accumulation_steps = grad_acc
            print(
                f"Set global batch size to {global_batch_size}, "
                f"grad accumulation steps to {grad_acc}"
            )

        trainer = ProbingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=self._eval_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
        )
        trainer._eval_dataloader_num_workers = getattr(
            self, "_eval_dataloader_num_workers", 2
        )

        from gr00t.utils.experiment import CheckpointFormatCallback

        run_name = training_args.run_name
        trainer.add_callback(
            CheckpointFormatCallback(run_name=run_name, exp_cfg_dir=self.exp_cfg_dir)
        )

        train_dl_len = len(trainer.get_train_dataloader())
        eval_info = (
            f"eval dataset length: {len(self._eval_dataset)}\n"
            if self._eval_dataset is not None
            else "eval dataset: None\n"
        )
        print(
            f"train dataloader length: {train_dl_len}\n"
            f"train dataset length: {len(trainer.train_dataset)}\n"
            + eval_info
            + f"GPU memory before training: "
            f"{torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB",
            flush=True,
        )
        return trainer


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(config: ArgsConfig):
    assert config.qformer_checkpoint_path, "--qformer-checkpoint-path is required"

    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # ------------------------------------------------------------------ #
    # 1. Dataset (train / val episode-level split)
    # ------------------------------------------------------------------ #
    dataset_kwargs = dict(
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=config.video_backend,
        val_ratio=config.val_ratio,
        val_seed=config.val_seed,
    )

    if len(config.dataset_path) == 1:
        train_dataset = LeRobotSingleDatasetWithSplit(
            dataset_path=config.dataset_path[0], split="train", **dataset_kwargs
        )
        val_dataset = LeRobotSingleDatasetWithSplit(
            dataset_path=config.dataset_path[0], split="val", **dataset_kwargs
        )
        first_single = train_dataset
    else:
        train_singles, val_singles = [], []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            train_singles.append(
                LeRobotSingleDatasetWithSplit(dataset_path=p, split="train", **dataset_kwargs)
            )
            val_singles.append(
                LeRobotSingleDatasetWithSplit(dataset_path=p, split="val", **dataset_kwargs)
            )
        train_dataset = LeRobotMixtureDataset(
            data_mixture=[(ds, 1.0) for ds in train_singles],
            mode="train",
            balance_dataset_weights=True,
            balance_trajectory_weights=True,
            seed=config.seed,
            metadata_config={"percentile_mixing_method": "weighted_average"},
        )
        val_dataset = LeRobotMixtureDataset(
            data_mixture=[(ds, 1.0) for ds in val_singles],
            mode="train",
            balance_dataset_weights=False,
            balance_trajectory_weights=False,
            seed=config.seed,
            metadata_config={"percentile_mixing_method": "weighted_average"},
        )
        first_single = train_singles[0]
        print(f"Loaded {len(train_singles)} datasets: {config.dataset_path}")

    # ------------------------------------------------------------------ #
    # 2. Build per-dim action_scale from first dataset's stats (only if
    #    denormalized L1 logging is enabled).
    # ------------------------------------------------------------------ #
    if config.log_l1_denorm:
        action_keys = list(data_config_cls.action_keys)
        action_scale = _extract_action_scale(first_single, action_keys, config.action_dim)
    else:
        action_scale = None
        print("[probing] log_l1_denorm=False — skipping action_scale extraction")

    # ------------------------------------------------------------------ #
    # 3. Load model: frozen backbone + frozen stage1 QFormer + trainable decoder
    # ------------------------------------------------------------------ #
    model = GR00T_N1_5_Probing.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        qformer_checkpoint_path=config.qformer_checkpoint_path,
        action_horizon=config.action_horizon,
        action_dim=config.action_dim,
        num_qformer_queries=config.num_qformer_queries,
        qformer_num_layers=config.qformer_num_layers,
        qformer_num_heads=config.qformer_num_heads,
        qformer_dropout=config.qformer_dropout,
        qformer_mlp_ratio=config.qformer_mlp_ratio,
        decoder_type=config.decoder_type,
        decoder_hidden_dim=config.decoder_hidden_dim,
        decoder_num_layers=config.decoder_num_layers,
        decoder_num_heads=config.decoder_num_heads,
        decoder_dropout=config.decoder_dropout,
        action_scale=action_scale,
        log_l1_denorm=config.log_l1_denorm,
        tune_visual=False,
        tune_llm=False,
    )
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    # ------------------------------------------------------------------ #
    # 4. Training args
    # ------------------------------------------------------------------ #
    eval_steps = config.eval_steps if config.eval_steps is not None else config.save_steps

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
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=10,
        report_to=config.report_to,
        seed=config.seed,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # ------------------------------------------------------------------ #
    # 5. Train
    # ------------------------------------------------------------------ #
    experiment = ProbingTrainRunner(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )
    experiment._eval_dataloader_num_workers = config.eval_dataloader_num_workers
    experiment.train()


# --------------------------------------------------------------------------- #
# Entrypoint (handles torchrun multi-GPU like the existing scripts)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)

    print("\n" + "=" * 50)
    print("GR00T QFORMER PROBING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert config.num_gpus <= available_gpus, (
        f"Number of GPUs requested ({config.num_gpus}) is greater than available ({available_gpus})"
    )
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
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
                    cmd.append(f"--{key.replace('_', '-')}" if value else f"--no-{key.replace('_', '-')}")
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

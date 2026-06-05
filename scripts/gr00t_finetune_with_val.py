"""
gr00t_finetune_with_val.py

gr00t_finetune_flare_qformer_action_dit_discrete_ctf.py와 동일하지만
train/val split을 지원합니다.

변경 사항:
- ArgsConfig에 val_ratio (기본값 0.003 = 0.3%) 추가
- LeRobotSingleDatasetWithSplit으로 train/val dataset 분리
- TrainRunnerWithVal을 사용하여 eval_dataset 전달
- TrainingArguments에서 do_eval=True, eval_strategy="steps" 활성화
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset
from gr00t.data.dataset_val import LeRobotSingleDatasetWithSplit
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner_val import TrainRunnerWithVal
from gr00t.model.gr00t_n1_flare_qformer_action_dit_discrete_ctf import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model


@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning with validation split."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories"""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_only"
    """Data configuration name from DATA_CONFIG_MAP, we assume all datasets have the same data config"""

    lr_scheduler_type: Literal["cosine", "constant", "constant_with_warmup"] = "cosine"
    """Learning rate scheduler type."""

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    action_horizon: int = 16
    """Action horizon for model/action-head config."""

    action_dim: int = 32
    """Action dimension for model/action-head config."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    dataloader_num_workers: int = 24
    """Number of workers for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    """Where to report training metrics."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: Literal["decord", "torchvision_av"] = "torchvision_av"
    """Video backend to use for training."""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset."""

    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset."""

    run_name: str = None
    """Name of the run."""

    load_action_head: bool = True
    """Whether to load the action head."""

    keep_pretrained_action_head_when_no_load: bool = False
    """Only used when load_action_head=False."""

    debug_pretrained_loading: bool = False
    """Print missing/unexpected/mismatched key summary after from_pretrained."""

    vision_token_num: int = 32
    """Number of vision tokens to use for training."""

    flare_loss_lambda: float = 1.0
    """Lambda for the flare loss."""

    flare_align_layers: int = 12
    """Layer number from DiT layers to align (Max 16)."""

    dit_action_loss_lambda: float = 1.0
    """Lambda for the DiT action loss."""

    image_count: int = 1
    """Number of images to use for training."""

    video_only: bool = False
    """Whether to only use video for training."""

    # QFormer parameters
    use_qformer: bool = True
    """Whether to use QFormer for reasoning."""

    qformer_loss_lambda: float = 0.2
    """Lambda for the QFormer loss."""

    num_qformer_queries: int = 64
    """Number of QFormer query tokens."""

    num_reasoning_tokens: int = 64
    """Number of reasoning tokens (DiT input, 0 in stage1)."""

    qformer_align_layers: int = 12
    """Layer number from DiT layers to align for QFormer (Max 16)."""

    qformer_num_layers: int = 4
    """Number of QFormer layers."""

    qformer_num_heads: int = 8
    """Number of attention heads in QFormer."""

    qformer_dropout: float = 0.0
    """Dropout rate in QFormer."""

    qformer_mlp_ratio: float = 4.0
    """MLP expansion ratio in QFormer."""

    # Two-stage training parameters
    training_stage: Literal["stage1", "stage2", "eval"] = "stage2"
    """Training stage."""

    stage1_load_qformer_weights: bool = False
    """Whether to load QFormer weights when doing Stage1 training."""

    qformer_checkpoint_path: str = None
    """Path to Stage1 checkpoint for loading QFormer weights in Stage2."""

    freeze_qformer_stage2: bool = False
    """Whether to freeze QFormer in stage2."""

    # OpenVLA discrete action loss parameters
    openvla_discrete_action_loss_lambda: float = 0.0
    """Lambda for the OpenVLA-style discrete action loss."""

    openvla_num_bins: int = 256
    """Number of bins for continuous action discretization."""

    openvla_discrete_head_hidden_ratio: float = 1.0
    """Hidden size ratio for discrete action head MLPs."""

    openvla_modality_json_path: str = "/sjw_alinlab1/home/jungwook/robocasa_mg_gr00t_100/meta/modality.json"
    """Path to modality.json file for auto-configuration."""

    openvla_discrete_action_dim: int = None
    """Action dimension for discrete action heads."""

    openvla_action_keys: List[str] = None
    """List of action keys to use from modality.json."""

    openvla_per_step_action_min: List[float] = None
    """Per-step min values for normalization."""

    openvla_per_step_action_max: List[float] = None
    """Per-step max values for normalization."""

    discrete_action_indices: List[int] = None
    """Indices of natively discrete action dimensions."""

    # Coarse-to-Fine (CTF) parameters
    ctf_num_levels: int = 3
    """Number of coarse-to-fine levels for continuous actions."""

    ctf_num_bins: int = 8
    """Number of bins per level for continuous actions in CTF."""

    enable_queued_config_override: bool = False
    """If True, apply in-code config overrides."""

    config_override_profile: Literal["none", "franka_default"] = "none"
    """Override profile key."""

    # ── Validation split (이 스크립트에서 추가된 파라미터) ──
    val_ratio: float = 0.003
    """Train/val split 비율. 전체 에피소드 중 val에 사용할 비율 (기본값: 0.003 = 0.3%)"""

    val_seed: int = 42
    """Train/val split에 사용할 random seed."""

    eval_steps: int = None
    """Validation을 수행할 step 간격. None이면 save_steps와 동일하게 설정."""


def apply_queued_config_overrides(config: ArgsConfig) -> ArgsConfig:
    if not config.enable_queued_config_override:
        return config

    override_profiles = {
        "none": {},
        "franka_default": {
            "data_config": "real_franka_joint",
            "max_steps": 60000,
            "save_steps": 10000,
            "batch_size": 32,
            "learning_rate": 1e-4,
        },
    }

    overrides = override_profiles[config.config_override_profile]
    if not overrides:
        print(f"[QUEUE_CONFIG_OVERRIDE] profile='{config.config_override_profile}' has no changes.")
        return config

    print(f"[QUEUE_CONFIG_OVERRIDE] Applying profile='{config.config_override_profile}'")
    for key, new_value in overrides.items():
        old_value = getattr(config, key)
        setattr(config, key, new_value)
        print(f"[QUEUE_CONFIG_OVERRIDE] {key}: {old_value} -> {new_value}")
    return config


#####################################################################################
# main training function
#####################################################################################


def main(config: ArgsConfig):
    """Main training function with validation split."""
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # 1.2 dataset 공통 kwargs
    dataset_kwargs = dict(
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=config.video_backend,
        val_ratio=config.val_ratio,
        val_seed=config.val_seed,
    )

    # 1.3 단일 / 복수 데이터셋 처리
    if len(config.dataset_path) == 1:
        train_dataset = LeRobotSingleDatasetWithSplit(
            dataset_path=config.dataset_path[0],
            split="train",
            **dataset_kwargs,
        )
        val_dataset = LeRobotSingleDatasetWithSplit(
            dataset_path=config.dataset_path[0],
            split="val",
            **dataset_kwargs,
        )
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
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={"percentile_mixing_method": "weighted_average"},
        )
        val_dataset = LeRobotMixtureDataset(
            data_mixture=[(ds, 1.0) for ds in val_singles],
            mode="train",
            balance_dataset_weights=False,
            balance_trajectory_weights=False,
            seed=42,
            metadata_config={"percentile_mixing_method": "weighted_average"},
        )
        print(f"Loaded {len(train_singles)} datasets: {config.dataset_path}")

    # ------------ step 2: load model ------------
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        action_horizon=config.action_horizon,
        action_dim=config.action_dim,
        tune_llm=config.tune_llm,
        tune_visual=config.tune_visual,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
        load_action_head=config.load_action_head,
        keep_pretrained_action_head_when_no_load=config.keep_pretrained_action_head_when_no_load,
        debug_pretrained_loading=config.debug_pretrained_loading,
        vision_token_num=config.vision_token_num,
        flare_loss_lambda=config.flare_loss_lambda,
        flare_align_layers=config.flare_align_layers,
        dit_action_loss_lambda=config.dit_action_loss_lambda,
        image_count=config.image_count,
        resume=config.resume,
        video_only=config.video_only,
        use_qformer=config.use_qformer,
        qformer_loss_lambda=config.qformer_loss_lambda,
        num_qformer_queries=config.num_qformer_queries,
        num_reasoning_tokens=config.num_reasoning_tokens,
        qformer_align_layers=config.qformer_align_layers,
        qformer_num_layers=config.qformer_num_layers,
        qformer_num_heads=config.qformer_num_heads,
        qformer_dropout=config.qformer_dropout,
        qformer_mlp_ratio=config.qformer_mlp_ratio,
        training_stage=config.training_stage,
        stage1_load_qformer_weights=config.stage1_load_qformer_weights,
        qformer_checkpoint_path=config.qformer_checkpoint_path,
        freeze_qformer_stage2=config.freeze_qformer_stage2,
        openvla_discrete_action_loss_lambda=config.openvla_discrete_action_loss_lambda,
        openvla_num_bins=config.openvla_num_bins,
        openvla_discrete_head_hidden_ratio=config.openvla_discrete_head_hidden_ratio,
        openvla_modality_json_path=config.openvla_modality_json_path,
        openvla_discrete_action_dim=config.openvla_discrete_action_dim,
        openvla_action_keys=config.openvla_action_keys,
        openvla_per_step_action_min=config.openvla_per_step_action_min,
        openvla_per_step_action_max=config.openvla_per_step_action_max,
        discrete_action_indices=config.discrete_action_indices,
        ctf_num_levels=config.ctf_num_levels,
        ctf_num_bins=config.ctf_num_bins,
    )

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
        seed=42,
        # ── Validation 활성화 ──
        do_eval=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        # ──────────────────────
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    experiment = TrainRunnerWithVal(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    experiment.train()


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    config = apply_queued_config_overrides(config)

    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING WITH VALIDATION CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than available ({available_gpus})"
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

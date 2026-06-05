"""
Action Latent Tokenizer 학습 스크립트.

GR00T의 데이터 파이프라인으로 action을 로드하고,
TimeWiseEncoder → latent → ReconDecoder → reconstruct 파이프라인을 학습합니다.

사용 예:
    # 단일 GPU
    python scripts/train_action_latent_tokenizer.py \
        --dataset-path /path/to/dataset \
        --data-config fourier_gr1_arms_only \
        --embodiment-tag new_embodiment \
        --max-steps 50000 --batch-size 256

    # 멀티 GPU
    python scripts/train_action_latent_tokenizer.py \
        --dataset-path /path/to/dataset \
        --data-config fourier_gr1_arms_only \
        --num-gpus 2 --batch-size 256
"""

import os
import subprocess
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import torch
import torch.nn.functional as F
import transformers
import tyro
from transformers import TrainingArguments

#from gr00t.data.dataset_action_only import ActionOnlyCollator, ActionOnlyDataset
from gr00t.data.dataset_action_only_pretransform import PreTransformedActionOnlyDataset as ActionOnlyDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.trainer import BaseSampler
from gr00t.model.action_latent_tokenizer import (
    ActionLatentTokenizer,
    MaskedReconDecoder,
    ReconDecoder,
    TimeWiseEncoder,
    TimestepMasking,
)
from gr00t.model.action_latent_tokenizer_faster import (
    DimensionMasking,
    DimensionWiseActionLatentTokenizer,
    DimensionWiseMaskedReconDecoder,
    DimensionWiseReconDecoder,
    DimensionWiseEncoder,
)


# =====================================================================
# Trainer
# =====================================================================


class ActionLatentTrainer(transformers.Trainer):
    """ActionLatentTokenizer 전용 Trainer.

    DualBrainTrainer와 달리 model.action_head에 접근하지 않음.
    """

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def get_eval_dataloader(self, eval_dataset=None):
        # Cache the eval dataloader so we never accumulate persistent workers across eval steps.
        # Without caching, super().evaluate() + our custom evaluate() each call
        # get_eval_dataloader() → 2 new DataLoaders × 16 workers per eval step → OOM at ~35K steps.
        if not hasattr(self, "_cached_eval_dataloader"):
            self._cached_eval_dataloader = super().get_eval_dataloader(eval_dataset)
        return self._cached_eval_dataloader

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs)
        loss = outputs["loss"]
        # Buffer per-component losses for logging
        if model.training:
            if not hasattr(self, "_train_loss_buffer"):
                self._train_loss_buffer = {}
            for key in ("loss_recon1", "loss_recon2", "loss_masked", "loss_freq"):
                val = outputs.get(key)
                if val is not None:
                    v = val.item() if isinstance(val, torch.Tensor) else float(val)
                    self._train_loss_buffer.setdefault(key, []).append(v)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, start_time=None) -> None:
        # Flush buffered per-component train losses
        buf = getattr(self, "_train_loss_buffer", {})
        if buf and "loss" in logs:  # only during train steps
            for key, values in buf.items():
                logs[key] = sum(values) / len(values)
            self._train_loss_buffer = {}
        super().log(logs, start_time=start_time)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Standard eval + compute both MSE and L1 recon metrics for wandb logging."""
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Compute both MSE and L1 on the best reconstruction path
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model
        model.eval()

        total_mse, total_l1, n_samples = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = self._prepare_inputs(batch)
                actions = batch["action"]
                preds = model.autoencode(actions)
                B = actions.shape[0]
                total_mse += F.mse_loss(preds, actions).item() * B
                total_l1 += F.l1_loss(preds, actions).item() * B
                n_samples += B

        if n_samples > 0:
            extra = {
                f"{metric_key_prefix}_recon_mse": total_mse / n_samples,
                f"{metric_key_prefix}_recon_l1": total_l1 / n_samples,
            }
            self.log(extra)
            metrics.update(extra)

        return metrics


# =====================================================================
# Config
# =====================================================================


@dataclass
class ArgsConfig:
    """Action Latent Tokenizer 학습 설정."""

    # ── Dataset ──
    dataset_path: List[str]
    """데이터셋 경로 (하나 이상)"""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_only"
    """DATA_CONFIG_MAP의 키"""

    embodiment_tag: str = "new_embodiment"
    """로봇 태그"""

    normalization_mode: str = "min_max"
    """Action 정규화 방식"""

    # ── Model Architecture ──
    emb_dim: int = 256
    """Transformer embedding dimension"""

    head_dim: int = 64
    """Attention head dimension"""

    encoder_depth: int = 4
    """Encoder transformer depth"""

    decoder_depth: int = 2
    """Decoder transformer depth"""

    decoder_mode: Literal["self_attention", "cross_attention"] = "self_attention"
    """Decoder mode"""

    pdropout: float = 0.0
    """Dropout rate"""

    # ── Global / Hand tokens ──
    num_global_tokens: int = 0
    """Global token 수 (0이면 masked recon 비활성화)"""

    num_hand_tokens: int = 0
    """Hand token 수 (0이면 recon path 2 비활성화)"""

    # ── Masked Recon ──
    masked_decoder_depth: int = 2
    """Masked recon decoder depth"""

    masked_decoder_mode: Literal["self_attention", "cross_attention"] = "self_attention"
    """Masked recon decoder mode"""

    mask_ratio: float = 0.5
    """Timestep masking 비율 (mask_ratio_min/max 미설정 시 고정값으로 사용)"""

    mask_ratio_min: float = None
    """Masking ratio 하한 (설정 시 매 배치마다 [min, max] 에서 균등 샘플링)"""

    mask_ratio_max: float = None
    """Masking ratio 상한"""

    mask_mode: Literal["random", "block"] = "random"
    """Masking 방식"""

    # ── Loss ──
    lambda_recon: float = 1.0
    """Recon loss (path 1 + path 2) 가중치"""

    lambda_masked: float = 1.0
    """Masked recon loss 가중치 (num_global_tokens > 0일 때만 유효)"""

    recon_loss_type: Literal["mse", "l1"] = "mse"
    """Reconstruction loss 종류"""

    freq_loss_weight: float = 0.0
    """Frequency domain loss 가중치 (0이면 비활성화)"""

    # ── Tokenizer Type ──
    tokenizer_type: Literal["timewise", "dimwise"] = "timewise"
    """토크나이저 종류.
    - timewise: 각 latent 토큰 = 하나의 timestep (TimeWiseEncoder, action_latent_tokenizer.py)
    - dimwise:  각 latent 토큰 = 하나의 action dimension (DimensionWiseEncoder, action_latent_tokenizer_faster.py)
    """

    # ── Hand action config ──
    hand_action_dims: List[int] = None
    """Hand action dimension indices (e.g., 14 15 16 ...). None이면 hand-weighted loss 비활성화."""

    hand_loss_weight: float = 1.0
    """Recon path 2에서 hand dimension에 적용할 loss 가중치"""

    # ── Training ──
    output_dir: str = "/tmp/action_latent_tokenizer"
    """체크포인트 저장 경로"""

    batch_size: int = 256
    """GPU당 배치 크기"""

    max_steps: int = 50000
    """최대 학습 step"""

    learning_rate: float = 5e-5
    """학습률"""

    weight_decay: float = 1e-5
    """Weight decay"""

    warmup_ratio: float = 0.05
    """Warmup 비율"""

    lr_scheduler_type: Literal["cosine", "constant", "constant_with_warmup"] = "constant"
    """LR scheduler"""

    num_gpus: int = 1
    """사용할 GPU 수"""

    save_steps: int = 5000
    """체크포인트 저장 간격"""

    eval_steps: int = None
    """Validation 간격 (None이면 save_steps와 동일)"""

    dataloader_num_workers: int = 16
    """DataLoader worker 수"""

    report_to: Literal["wandb", "tensorboard"] = "wandb"
    """로깅 대상"""

    run_name: str = None
    """Run 이름"""

    wandb_project: str = "action-latent-tokenizer"
    """Wandb 프로젝트 이름"""

    resume: bool = False
    """체크포인트에서 재개"""

    # ── Validation ──
    val_ratio: float = 0.003
    """Validation 에피소드 비율 (기본값: 0.3%)"""

    val_seed: int = 42
    """Validation split seed"""


# =====================================================================
# Tokenizer build helpers — 새 tokenizer 타입 추가 시 여기에 함수 추가
# =====================================================================


def _build_timewise(config: "ArgsConfig", action_dim: int, action_horizon: int):
    """TimeWiseEncoder 기반 ActionLatentTokenizer 생성."""
    encoder = TimeWiseEncoder(
        action_dim=action_dim,
        action_horizon=action_horizon,
        emb_dim=config.emb_dim,
        head_dim=config.head_dim,
        depth=config.encoder_depth,
        pdropout=config.pdropout,
        num_global_tokens=config.num_global_tokens,
        num_hand_tokens=config.num_hand_tokens,
    )
    recon_decoder = ReconDecoder(
        action_dim=action_dim,
        action_horizon=action_horizon,
        emb_dim=config.emb_dim,
        head_dim=config.head_dim,
        depth=config.decoder_depth,
        pdropout=config.pdropout,
        decoder_mode=config.decoder_mode,
        num_hand_tokens=config.num_hand_tokens,
    )
    masked_recon_decoder = None
    masking = None
    if config.num_global_tokens > 0 and config.lambda_masked > 0:
        masked_recon_decoder = MaskedReconDecoder(
            action_dim=action_dim,
            action_horizon=action_horizon,
            emb_dim=config.emb_dim,
            head_dim=config.head_dim,
            depth=config.masked_decoder_depth,
            pdropout=config.pdropout,
            decoder_mode=config.masked_decoder_mode,
            num_global_tokens=config.num_global_tokens,
        )
        masking = TimestepMasking(
            mask_ratio=config.mask_ratio,
            mask_mode=config.mask_mode,
            min_mask_ratio=config.mask_ratio_min,
            max_mask_ratio=config.mask_ratio_max,
        )
    return ActionLatentTokenizer(
        encoder=encoder,
        recon_decoder=recon_decoder,
        masked_recon_decoder=masked_recon_decoder,
        masking=masking,
        lambda_recon=config.lambda_recon,
        lambda_masked=config.lambda_masked,
        hand_action_dims=config.hand_action_dims,
        hand_loss_weight=config.hand_loss_weight,
        recon_loss_type=config.recon_loss_type,
        freq_loss_weight=config.freq_loss_weight,
    )


def _build_dimwise(config: "ArgsConfig", action_dim: int, action_horizon: int):
    """DimensionWiseEncoder 기반 DimensionWiseActionLatentTokenizer 생성."""
    encoder = DimensionWiseEncoder(
        action_dim=action_dim,
        action_horizon=action_horizon,
        emb_dim=config.emb_dim,
        head_dim=config.head_dim,
        depth=config.encoder_depth,
        pdropout=config.pdropout,
        num_global_tokens=config.num_global_tokens,
        num_hand_tokens=config.num_hand_tokens,
    )
    recon_decoder = DimensionWiseReconDecoder(
        action_dim=action_dim,
        action_horizon=action_horizon,
        emb_dim=config.emb_dim,
        head_dim=config.head_dim,
        depth=config.decoder_depth,
        pdropout=config.pdropout,
        decoder_mode=config.decoder_mode,
        num_hand_tokens=config.num_hand_tokens,
    )
    masked_recon_decoder = None
    masking = None
    if config.num_global_tokens > 0 and config.lambda_masked > 0:
        masked_recon_decoder = DimensionWiseMaskedReconDecoder(
            action_dim=action_dim,
            action_horizon=action_horizon,
            emb_dim=config.emb_dim,
            head_dim=config.head_dim,
            depth=config.masked_decoder_depth,
            pdropout=config.pdropout,
            decoder_mode=config.masked_decoder_mode,
            num_global_tokens=config.num_global_tokens,
        )
        masking = DimensionMasking(
            mask_ratio=config.mask_ratio,
            mask_mode=config.mask_mode,
            min_mask_ratio=config.mask_ratio_min,
            max_mask_ratio=config.mask_ratio_max,
        )
    return DimensionWiseActionLatentTokenizer(
        encoder=encoder,
        recon_decoder=recon_decoder,
        masked_recon_decoder=masked_recon_decoder,
        masking=masking,
        lambda_recon=config.lambda_recon,
        lambda_masked=config.lambda_masked,
        hand_action_dims=config.hand_action_dims,
        hand_loss_weight=config.hand_loss_weight,
        recon_loss_type=config.recon_loss_type,
        freq_loss_weight=config.freq_loss_weight,
    )


# 새 tokenizer 타입 추가 시: 위에 _build_XXX 함수를 만들고 여기에 등록
_TOKENIZER_BUILDERS = {
    "timewise": _build_timewise,
    "dimwise": _build_dimwise,
}


def _build_tokenizer_model(config: "ArgsConfig", action_dim: int, action_horizon: int):
    """tokenizer_type에 맞는 모델을 생성해 반환."""
    if config.tokenizer_type not in _TOKENIZER_BUILDERS:
        raise ValueError(
            f"Unknown tokenizer_type: {config.tokenizer_type!r}. "
            f"Available: {list(_TOKENIZER_BUILDERS.keys())}"
        )
    return _TOKENIZER_BUILDERS[config.tokenizer_type](config, action_dim, action_horizon)


# =====================================================================
# Main
# =====================================================================


def main(config: ArgsConfig):
    """학습 메인 함수."""

    # ── 1. Dataset 생성 ──
    datasets_train = []
    datasets_val = []

    for path in config.dataset_path:
        assert os.path.exists(path), f"Dataset path가 존재하지 않습니다: {path}"

        ds_train = ActionOnlyDataset(
            dataset_path=path,
            data_config_name=config.data_config,
            embodiment_tag=config.embodiment_tag,
            split="train",
            val_ratio=config.val_ratio,
            val_seed=config.val_seed,
            normalization_mode=config.normalization_mode,
        )
        ds_val = ActionOnlyDataset(
            dataset_path=path,
            data_config_name=config.data_config,
            embodiment_tag=config.embodiment_tag,
            split="val",
            val_ratio=config.val_ratio,
            val_seed=config.val_seed,
            normalization_mode=config.normalization_mode,
        )
        datasets_train.append(ds_train)
        datasets_val.append(ds_val)

    # 단일 데이터셋이면 그대로 사용, 복수이면 ConcatDataset
    if len(datasets_train) == 1:
        train_dataset = datasets_train[0]
        val_dataset = datasets_val[0]
    else:
        train_dataset = torch.utils.data.ConcatDataset(datasets_train)
        val_dataset = torch.utils.data.ConcatDataset(datasets_val)

    # ── 2. action_dim과 action_horizon 자동 추출 ──
    sample = datasets_train[0][0]  # {"action": Tensor[T, D]}
    action_horizon, action_dim = sample["action"].shape
    print(f"action_horizon={action_horizon}, action_dim={action_dim}")

    # ── 3. Model 생성 ──
    model = _build_tokenizer_model(config, action_dim, action_horizon)
    print(f"[TokenizerType] {config.tokenizer_type}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # ── 4. TrainingArguments ──
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
        save_total_limit=20,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        report_to=config.report_to,
        seed=42,
        ddp_find_unused_parameters=False,
    )

    # ── Epoch 단위 샘플/스텝 정보 출력 ──
    train_samples_per_epoch = len(train_dataset)
    val_samples_per_epoch = len(val_dataset)
    world_size = max(1, config.num_gpus)
    micro_batch_global = config.batch_size * world_size
    train_dataloader_steps_per_epoch = math.ceil(train_samples_per_epoch / micro_batch_global)
    train_optimizer_steps_per_epoch = math.ceil(
        train_dataloader_steps_per_epoch / training_args.gradient_accumulation_steps
    )
    val_dataloader_steps_per_epoch = math.ceil(val_samples_per_epoch / micro_batch_global)

    print(
        "[TrainInfo] train_samples/epoch="
        f"{train_samples_per_epoch:,} | val_samples/epoch={val_samples_per_epoch:,} "
        f"| micro_batch(global)={micro_batch_global:,} "
        f"| train_dataloader_steps/epoch={train_dataloader_steps_per_epoch:,} "
        f"| train_optimizer_steps/epoch={train_optimizer_steps_per_epoch:,} "
        f"| val_dataloader_steps/epoch={val_dataloader_steps_per_epoch:,}"
    )

    # ── 5. Trainer 생성 및 학습 ──
    collator = ActionOnlyCollator()

    trainer = ActionLatentTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # wandb 설정
    if config.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = config.wandb_project
        os.environ["WANDB_DIR"] = config.output_dir

    trainer.train(resume_from_checkpoint=config.resume)

    # ── 6. 최종 모델 저장 ──
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        save_path = os.path.join(config.output_dir, "action_latent_tokenizer_final.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "tokenizer_type": config.tokenizer_type,
                    "action_dim": action_dim,
                    "action_horizon": action_horizon,
                    "emb_dim": config.emb_dim,
                    "head_dim": config.head_dim,
                    "encoder_depth": config.encoder_depth,
                    "decoder_depth": config.decoder_depth,
                    "decoder_mode": config.decoder_mode,
                    "pdropout": config.pdropout,
                    "num_global_tokens": config.num_global_tokens,
                    "num_hand_tokens": config.num_hand_tokens,
                    "masked_decoder_depth": config.masked_decoder_depth,
                    "masked_decoder_mode": config.masked_decoder_mode,
                    "mask_ratio": config.mask_ratio,
                    "mask_mode": config.mask_mode,
                    "lambda_recon": config.lambda_recon,
                    "lambda_masked": config.lambda_masked,
                    "hand_action_dims": config.hand_action_dims,
                    "hand_loss_weight": config.hand_loss_weight,
                    "recon_loss_type": config.recon_loss_type,
                    "freq_loss_weight": config.freq_loss_weight,
                },
            },
            save_path,
        )
        print(f"Final model saved to {save_path}")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)

    print("\n" + "=" * 50)
    print("ACTION LATENT TOKENIZER TRAINING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert config.num_gpus <= available_gpus, (
        f"요청한 GPU 수({config.num_gpus})가 가용 GPU 수({available_gpus})보다 큽니다."
    )
    assert config.num_gpus > 0

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

            print("Running torchrun:", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)

"""
Action Latent Tokenizer V2 학습 스크립트.

v1 스크립트와 분리된 독립 스크립트.
ActionLatentTokenizerV2를 사용하며, 다양한 auxiliary loss를 지원:
  - Recon: 기본 action reconstruction
  - Hand state prediction: hand token으로 미래 hand state 예측
  - Masked latent recon: encoder output 마스킹 후 동일 decoder로 reconstruct
  - Global token learning: FAST tokenizer + text encoder contrastive/regression
  - Frequency loss

사용 예:
    # 기본 (recon만)
    python scripts/train_action_latent_tokenizer_v2.py \
        --dataset-path /path/to/dataset \
        --data-config fourier_gr1_arms_waist \
        --embodiment-tag new_embodiment \
        --max-steps 100000 --batch-size 1024

    # 전체 loss 활성화
    python scripts/train_action_latent_tokenizer_v2.py \
        --dataset-path /path/to/dataset \
        --data-config fourier_gr1_arms_waist \
        --num-global-tokens 2 --num-hand-tokens 2 \
        --lambda-hand-pred 0.1 --hand-pred-future-steps 8 16 \
        --hand-state-dims 14 15 16 17 18 19 20 21 22 23 24 25 \
        --lambda-mask-recon 0.1 \
        --lambda-global 0.1 --global-loss-mode contrastive
"""

import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
import transformers
import tyro
from transformers import TrainingArguments

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.trainer import BaseSampler
from gr00t.model.action_latent_tokenizer_v2 import (
    ActionLatentTokenizerV2,
    ActionTextEncoder,
    GlobalTokenLossModule,
    HandStatePredDecoder,
    ReconDecoder,
    TimeWiseEncoder,
)


# =====================================================================
# Trainer
# =====================================================================


class ActionLatentV2Trainer(transformers.Trainer):
    """ActionLatentTokenizerV2 전용 Trainer."""

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def get_eval_dataloader(self, eval_dataset=None):
        if not hasattr(self, "_cached_eval_dataloader"):
            self._cached_eval_dataloader = super().get_eval_dataloader(eval_dataset)
        return self._cached_eval_dataloader

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs)
        loss = outputs["loss"]
        if model.training:
            if not hasattr(self, "_train_loss_buffer"):
                self._train_loss_buffer = {}
            for key in ("loss_recon", "loss_hand_pred", "loss_mask_recon", "loss_global", "loss_freq"):
                val = outputs.get(key)
                if val is not None:
                    v = val.item() if isinstance(val, torch.Tensor) else float(val)
                    self._train_loss_buffer.setdefault(key, []).append(v)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, start_time=None) -> None:
        buf = getattr(self, "_train_loss_buffer", {})
        if buf and "loss" in logs:
            for key, values in buf.items():
                logs[key] = sum(values) / len(values)
            self._train_loss_buffer = {}
        super().log(logs, start_time=start_time)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Standard eval + MSE/L1 recon metrics."""
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

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
    """Action Latent Tokenizer V2 학습 설정."""

    # ── Dataset ──
    dataset_path: List[str]
    """데이터셋 경로 (하나 이상)"""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_waist"
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
    """Global token 수"""

    num_hand_tokens: int = 0
    """Hand token 수"""

    # ── Loss Weights ──
    lambda_recon: float = 1.0
    """Recon loss 가중치"""

    lambda_hand_pred: float = 0.0
    """Hand state prediction loss 가중치 (0이면 비활성화)"""

    lambda_mask_recon: float = 0.0
    """Masked latent recon loss 가중치 (0이면 비활성화)"""

    lambda_mask_hand_pred: float = 0.0
    """Masked path의 state prediction loss 가중치 (masked latent → future state) (0이면 비활성화)."""

    lambda_global: float = 0.0
    """Global token loss 가중치 (0이면 비활성화)"""

    freq_loss_weight: float = 0.0
    """Frequency domain loss 가중치 (0이면 비활성화)"""

    recon_loss_type: Literal["mse", "l1"] = "mse"
    """Reconstruction loss 종류"""

    # ── Hand State Prediction ──
    hand_state_dims: Optional[List[int]] = None
    """State에서 hand에 해당하는 dim indices (e.g., 14 15 16 ...)"""

    hand_pred_future_steps: Optional[List[int]] = None
    """미래 예측할 step 간격 리스트 (e.g., 8 16 → 8step/16step 후 예측)"""

    hand_pred_decoder_depth: int = 2
    """Hand state prediction decoder depth"""

    hand_in_recon: bool = True
    """Whether hand tokens are passed to the recon decoder. False for Experiment A."""

    state_pred_kv_source: Literal["hand", "time"] = "hand"
    """KV source for state prediction decoder: 'hand' (hand_tok) or 'time' (time_tok)."""

    # ── Masked Latent Recon ──
    mask_ratio: float = 0.5
    """마스킹 비율 (mask_ratio_min/max 미설정 시 고정값으로 사용)"""

    mask_ratio_min: Optional[float] = None
    """Masking ratio 하한 (설정 시 매 배치마다 [min, max]에서 균등 샘플링)"""

    mask_ratio_max: Optional[float] = None
    """Masking ratio 상한"""

    mask_mode: Literal["random", "block"] = "random"
    """Masking 방식: random (랜덤 timestep) 또는 block (연속 블록)"""

    mask_batch_ratio: float = 0.5
    """배치 중 마스킹을 적용할 비율"""

    # ── Global Token Learning ──
    global_loss_mode: Literal["contrastive", "regression"] = "contrastive"
    """Global token loss 방식"""

    global_pool_type: Literal["mean", "max", "attn", "linear"] = "mean"
    """Global token pooling 방식 (mean, max, attn=attention pooling, linear=concat+linear)"""

    text_encoder_width: int = 256
    """Text encoder hidden dimension"""

    text_encoder_layers: int = 4
    """Text encoder transformer depth"""

    text_encoder_heads: int = 4
    """Text encoder attention heads"""

    text_encoder_pretrained_path: Optional[str] = None
    """Text encoder pretrained weight 경로 (None이면 from scratch)"""

    fast_tokenizer_path: str = "physical-intelligence/fast"
    """FAST tokenizer 경로"""

    fast_vocab_size: int = 2048
    """FAST tokenizer vocab size"""

    # ── Training ──
    output_dir: str = "/tmp/action_latent_tokenizer_v2"
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

    eval_steps: Optional[int] = None
    """Validation 간격 (None이면 save_steps와 동일)"""

    dataloader_num_workers: int = 16
    """DataLoader worker 수"""

    report_to: Literal["wandb", "tensorboard"] = "wandb"
    """로깅 대상"""

    run_name: Optional[str] = None
    """Run 이름"""

    wandb_project: str = "action-latent-tokenizer-v2"
    """Wandb 프로젝트 이름"""

    resume: bool = False
    """체크포인트에서 재개"""

    # ── Validation ──
    val_ratio: float = 0.003
    """Validation 에피소드 비율"""

    val_seed: int = 42
    """Validation split seed"""


# =====================================================================
# Model builder
# =====================================================================


def _build_v2_tokenizer(config: ArgsConfig, action_dim: int, action_horizon: int):
    """ActionLatentTokenizerV2 생성. lambda=0인 모듈은 None."""

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

    # hand_in_recon=False이면 decoder에 hand tokens를 사용하지 않음
    decoder_num_hand = config.num_hand_tokens if config.hand_in_recon else 0

    recon_decoder = ReconDecoder(
        action_dim=action_dim,
        action_horizon=action_horizon,
        emb_dim=config.emb_dim,
        head_dim=config.head_dim,
        depth=config.decoder_depth,
        pdropout=config.pdropout,
        decoder_mode=config.decoder_mode,
        num_global_tokens=config.num_global_tokens,
        num_hand_tokens=decoder_num_hand,
    )

    # Hand state prediction decoder
    hand_pred_decoder = None
    if config.lambda_hand_pred > 0 and config.hand_state_dims and config.hand_pred_future_steps:
        hand_state_dim = len(config.hand_state_dims)
        num_future_steps = len(config.hand_pred_future_steps)
        # KV source에 따라 num_kv_tokens 결정
        if config.state_pred_kv_source == "time":
            num_kv_tokens = action_horizon
        else:
            num_kv_tokens = config.num_hand_tokens
        hand_pred_decoder = HandStatePredDecoder(
            hand_state_dim=hand_state_dim,
            emb_dim=config.emb_dim,
            head_dim=config.head_dim,
            depth=config.hand_pred_decoder_depth,
            pdropout=config.pdropout,
            num_future_steps=num_future_steps,
            num_kv_tokens=num_kv_tokens,
        )

    # Global token learning (text encoder + loss module)
    action_text_encoder = None
    global_loss_module = None
    if config.lambda_global > 0 and config.num_global_tokens > 0:
        text_output_dim = config.emb_dim

        if config.text_encoder_pretrained_path:
            action_text_encoder = ActionTextEncoder.from_pretrained(
                config.text_encoder_pretrained_path,
                vocab_size=config.fast_vocab_size,
                context_length=256,
                width=config.text_encoder_width,
                heads=config.text_encoder_heads,
                layers=config.text_encoder_layers,
                output_dim=text_output_dim,
                pad_token_id=config.fast_vocab_size,
            )
        else:
            action_text_encoder = ActionTextEncoder(
                vocab_size=config.fast_vocab_size,
                context_length=256,
                width=config.text_encoder_width,
                heads=config.text_encoder_heads,
                layers=config.text_encoder_layers,
                output_dim=text_output_dim,
                pad_token_id=config.fast_vocab_size,
            )

        global_loss_module = GlobalTokenLossModule(
            mode=config.global_loss_mode,
            emb_dim=config.emb_dim,
            text_feat_dim=text_output_dim,
            pool_type=config.global_pool_type,
            num_global_tokens=config.num_global_tokens,
        )

    return ActionLatentTokenizerV2(
        encoder=encoder,
        recon_decoder=recon_decoder,
        hand_pred_decoder=hand_pred_decoder,
        action_text_encoder=action_text_encoder,
        global_loss_module=global_loss_module,
        lambda_recon=config.lambda_recon,
        lambda_hand_pred=config.lambda_hand_pred,
        lambda_mask_recon=config.lambda_mask_recon,
        lambda_mask_hand_pred=config.lambda_mask_hand_pred,
        lambda_global=config.lambda_global,
        freq_loss_weight=config.freq_loss_weight,
        mask_ratio=config.mask_ratio,
        mask_ratio_min=config.mask_ratio_min,
        mask_ratio_max=config.mask_ratio_max,
        mask_mode=config.mask_mode,
        mask_batch_ratio=config.mask_batch_ratio,
        recon_loss_type=config.recon_loss_type,
        hand_in_recon=config.hand_in_recon,
        state_pred_kv_source=config.state_pred_kv_source,
    )


# =====================================================================
# Main
# =====================================================================


def main(config: ArgsConfig):
    """학습 메인 함수."""

    # ── 1. Dataset 선택 및 생성 ──
    need_state = (
        (config.lambda_hand_pred > 0 or config.lambda_mask_hand_pred > 0)
        and config.hand_state_dims is not None
        and len(config.hand_state_dims) > 0
    )
    need_fast = config.lambda_global > 0

    if need_state or need_fast:
        from gr00t.data.dataset_action_state_pretransform import (
            ActionStateCollator,
            PreTransformedActionStateDataset,
        )
        DatasetCls = PreTransformedActionStateDataset
        CollatorCls = ActionStateCollator

        def make_dataset(path, split):
            return DatasetCls(
                dataset_path=path,
                data_config_name=config.data_config,
                embodiment_tag=config.embodiment_tag,
                split=split,
                val_ratio=config.val_ratio,
                val_seed=config.val_seed,
                normalization_mode=config.normalization_mode,
                hand_state_dims=config.hand_state_dims,
                hand_pred_future_steps=config.hand_pred_future_steps,
                cache_fast_tokens=need_fast,
                fast_tokenizer_path=config.fast_tokenizer_path,
                fast_vocab_size=config.fast_vocab_size,
            )
    else:
        from gr00t.data.dataset_action_only_pretransform import (
            PreTransformedActionOnlyDataset,
        )
        from gr00t.data.dataset_action_only import ActionOnlyCollator

        DatasetCls = PreTransformedActionOnlyDataset
        CollatorCls = ActionOnlyCollator

        def make_dataset(path, split):
            return DatasetCls(
                dataset_path=path,
                data_config_name=config.data_config,
                embodiment_tag=config.embodiment_tag,
                split=split,
                val_ratio=config.val_ratio,
                val_seed=config.val_seed,
                normalization_mode=config.normalization_mode,
            )

    datasets_train = []
    datasets_val = []
    for path in config.dataset_path:
        assert os.path.exists(path), f"Dataset path가 존재하지 않습니다: {path}"
        datasets_train.append(make_dataset(path, "train"))
        datasets_val.append(make_dataset(path, "val"))

    if len(datasets_train) == 1:
        train_dataset = datasets_train[0]
        val_dataset = datasets_val[0]
    else:
        train_dataset = torch.utils.data.ConcatDataset(datasets_train)
        val_dataset = torch.utils.data.ConcatDataset(datasets_val)

    # ── 2. action_dim과 action_horizon 자동 추출 ──
    sample = datasets_train[0][0]
    action_horizon, action_dim = sample["action"].shape
    print(f"action_horizon={action_horizon}, action_dim={action_dim}")

    # ── 3. Model 생성 ──
    model = _build_v2_tokenizer(config, action_dim, action_horizon)

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

    # ── Info 출력 ──
    train_samples_per_epoch = len(train_dataset)
    val_samples_per_epoch = len(val_dataset)
    world_size = max(1, config.num_gpus)
    micro_batch_global = config.batch_size * world_size
    train_steps_per_epoch = math.ceil(train_samples_per_epoch / micro_batch_global)

    print(
        f"[TrainInfo] train_samples={train_samples_per_epoch:,} | val_samples={val_samples_per_epoch:,} "
        f"| micro_batch(global)={micro_batch_global:,} "
        f"| train_steps/epoch={train_steps_per_epoch:,}"
    )

    # ── 5. Trainer 생성 및 학습 ──
    collator = CollatorCls()

    trainer = ActionLatentV2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    if config.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = config.wandb_project
        os.environ["WANDB_DIR"] = config.output_dir

    trainer.train(resume_from_checkpoint=config.resume)

    # ── 6. 최종 모델 저장 ──
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        save_path = os.path.join(config.output_dir, "action_latent_tokenizer_v2_final.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "tokenizer_version": "v2",
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
                    "lambda_recon": config.lambda_recon,
                    "lambda_hand_pred": config.lambda_hand_pred,
                    "lambda_mask_recon": config.lambda_mask_recon,
                    "lambda_mask_hand_pred": config.lambda_mask_hand_pred,
                    "lambda_global": config.lambda_global,
                    "freq_loss_weight": config.freq_loss_weight,
                    "mask_ratio": config.mask_ratio,
                    "mask_ratio_min": config.mask_ratio_min,
                    "mask_ratio_max": config.mask_ratio_max,
                    "mask_mode": config.mask_mode,
                    "mask_batch_ratio": config.mask_batch_ratio,
                    "recon_loss_type": config.recon_loss_type,
                    "hand_state_dims": config.hand_state_dims,
                    "hand_pred_future_steps": config.hand_pred_future_steps,
                    "hand_pred_decoder_depth": config.hand_pred_decoder_depth,
                    "hand_in_recon": config.hand_in_recon,
                    "state_pred_kv_source": config.state_pred_kv_source,
                    "global_loss_mode": config.global_loss_mode,
                    "text_encoder_width": config.text_encoder_width,
                    "text_encoder_layers": config.text_encoder_layers,
                    "text_encoder_heads": config.text_encoder_heads,
                    "fast_vocab_size": config.fast_vocab_size,
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

    print("\n" + "=" * 60)
    print("ACTION LATENT TOKENIZER V2 TRAINING CONFIGURATION:")
    print("=" * 60)
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")

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

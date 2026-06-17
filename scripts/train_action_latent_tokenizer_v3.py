"""
Action Latent Tokenizer V3 학습 스크립트.

V3 = V2 + 다음 옵션:
  - encoder_output_layernorm: encoder transformer 출력에 LayerNorm 적용
  - latent_noise_std:         encode → decode 사이에 Gaussian noise 추가 (training only)
  - use_bottleneck / token_dim: VTP 스타일 bottleneck. 켜지면 encoder 출력 latent
                               차원이 emb_dim → token_dim 으로 축소되고 decoder 에서
                               token_dim → emb_dim 로 복원. 기본 token_dim=64.
                               이후 VLA 학습은 wrapper.emb_dim (= token_dim) 을 읽음.
  - use_fixed_val:            train/val split 을 디스크에 저장/로드
  - fixed_val_path:           split JSON의 명시적 절대 경로 (None 이면 dataset/meta/ 기본)
  - data_config_v3:           q99 정규화 데이터 컨피그 (fourier_gr1_arms_waist_q99 등)

기본값은 모두 V2 와 동치 (output_layernorm=False, latent_noise_std=0.0,
use_bottleneck=False, use_fixed_val=True). 즉 모든 옵션을 끄면 V2와 동일하게
동작하면서 state_dict 에 ``_is_v3`` 버퍼가 추가된 형태.
"""

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

# Side-effect import: register V3 q99 data configs in DATA_CONFIG_MAP before
# we evaluate ``Literal[tuple(DATA_CONFIG_MAP.keys())]`` below.
import gr00t.experiment.data_config_v3  # noqa: F401
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.trainer import BaseSampler
from gr00t.model.action_latent_tokenizer_v2 import (
    ActionTextEncoder,
    GlobalTokenLossModule,
)
from gr00t.model.action_latent_tokenizer_v3 import (
    ActionLatentTokenizerV3,
    HandStatePredDecoderV3,
    ReconDecoderV3,
    TimeWiseEncoderV3,
)


# =====================================================================
# Trainer
# =====================================================================


class ActionLatentV3Trainer(transformers.Trainer):
    """ActionLatentTokenizerV3 전용 Trainer (V2 trainer 와 동일한 logging)."""

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
            for key in (
                "loss_recon",
                "loss_hand_pred",
                "loss_mask_recon",
                "loss_mask_hand_pred",
                "loss_global",
                "loss_freq",
                "loss_kl",
            ):
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
        """Standard eval + MSE/L1 recon metrics (computed without latent noise)."""
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model
        model.eval()

        total_mse, total_l1, n_samples = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = self._prepare_inputs(batch)
                actions = batch["action"]
                # autoencode (encode + decode) — eval mode → noise inactive.
                g, t, h = model.encode(actions.to(dtype=model.encoder.action_proj.weight.dtype))
                preds = model.decode(g, t, h)
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
    """Action Latent Tokenizer V3 학습 설정."""

    # ── Dataset ──
    dataset_path: List[str]
    """데이터셋 경로 (하나 이상)"""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_waist_q99"
    """DATA_CONFIG_MAP의 키 (V3 기본은 q99 변형)"""

    embodiment_tag: str = "new_embodiment"
    """로봇 태그"""

    normalization_mode: str = "min_max"
    """data_config 가 action_normalization_modes 를 정의하지 않은 키에 대한 fallback"""

    # ── Model Architecture ──
    emb_dim: int = 256
    head_dim: int = 64
    encoder_depth: int = 4
    decoder_depth: int = 2
    decoder_mode: Literal["self_attention", "cross_attention"] = "self_attention"
    pdropout: float = 0.0

    # ── V3 additions ──
    encoder_output_layernorm: bool = False
    """Encoder transformer 출력에 LayerNorm 적용"""

    latent_noise_std: float = 0.0
    """Encoded latent 에 더할 Gaussian noise std (training only). 0 이면 비활성"""

    use_bottleneck: bool = False
    """VTP 스타일 bottleneck 사용. 켜지면 encoder 출력 latent 차원이
    emb_dim → token_dim 으로 축소되고 decoder 에서 다시 복원됨.
    이후 VLA 학습 시 wrapper.emb_dim 은 자동으로 token_dim 을 반환."""

    token_dim: int = 64
    """Bottleneck 사용 시 출력 latent 차원. use_bottleneck=False 이면 무시되고
    실제 출력 차원은 emb_dim 이 됨."""

    compress_token: int = 1
    """Time 축 토큰 압축 배율. 1 이면 압축 비활성 (기본, V3 와 동치). >1 이면
    encoder 입력단 Conv1d(kernel=stride=compress_token)로 time 토큰 수를
    action_horizon → action_horizon // compress_token 로 줄이고, decoder 의
    sub-pixel head 가 마지막에 다시 action_horizon 으로 복원함. action_horizon 은
    compress_token 으로 나누어 떨어져야 함. VLA 학습 시 예측 토큰 수는 wrapper 가
    압축된 토큰 수를 자동으로 노출하여 그에 맞게 설정됨."""

    use_vae: bool = False
    """SD 스타일 VAE bottleneck 사용 (V4 와 동일). 켜지면 bottleneck 출력이
    posterior mean μ 로 취급되고 logvar_head 가 추가되어 z = μ + σ·ε 로
    reparameterize 됨. lambda_kl 로 KL(N(0,I)) 가중. 기본 off (deterministic V3,
    state_dict 가 기존 v3 와 byte-identical)."""

    lambda_kl: float = 1e-6
    """VAE KL loss 가중치 (SD regime, 기본 1e-6). use_vae=False 이면 무시."""

    kl_free_bits: float = 0.0
    """per-dim KL 하한 (free-bits). 0 이면 비활성."""

    # ── Global / Hand tokens ──
    num_global_tokens: int = 0
    num_hand_tokens: int = 0

    # ── Loss Weights ──
    lambda_recon: float = 1.0
    lambda_hand_pred: float = 0.0
    lambda_mask_recon: float = 0.0
    lambda_mask_hand_pred: float = 0.0
    lambda_global: float = 0.0
    freq_loss_weight: float = 0.0

    recon_loss_type: Literal["mse", "l1"] = "l1"
    """Recon loss 종류 (V3 기본은 l1)"""

    # ── Hand State Prediction ──
    hand_state_dims: Optional[List[int]] = None
    hand_pred_future_steps: Optional[List[int]] = None
    hand_pred_decoder_depth: int = 2
    hand_in_recon: bool = True
    state_pred_kv_source: Literal["hand", "time"] = "hand"

    # ── Masked Latent Recon ──
    mask_ratio: float = 0.5
    mask_ratio_min: Optional[float] = None
    mask_ratio_max: Optional[float] = None
    mask_mode: Literal["random", "block"] = "random"
    mask_batch_ratio: float = 0.5

    # ── Global Token Learning ──
    global_loss_mode: Literal["contrastive", "regression"] = "contrastive"
    global_pool_type: Literal["mean", "max", "attn", "linear"] = "mean"
    text_encoder_width: int = 256
    text_encoder_layers: int = 4
    text_encoder_heads: int = 4
    text_encoder_pretrained_path: Optional[str] = None
    fast_tokenizer_path: str = "physical-intelligence/fast"
    fast_vocab_size: int = 2048

    # ── Training ──
    output_dir: str = "/tmp/action_latent_tokenizer_v3"
    batch_size: int = 256
    max_steps: int = 50000
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    lr_scheduler_type: Literal["cosine", "constant", "constant_with_warmup"] = "constant"
    num_gpus: int = 1
    save_steps: int = 5000
    save_total_limit: int = 20
    """Max number of checkpoints to keep; oldest are deleted first. Set <=0 to keep all."""
    eval_steps: Optional[int] = None
    dataloader_num_workers: int = 16
    cache_dataset: bool = True
    """If True (default), pre-cache all actions into memory before training (fast
    steps, but GPUs idle during the caching phase). If False, read samples
    on-the-fly each step like Stage-2 via dataloader workers — keeps GPU
    utilization up from the start. Only supported for the action-only path
    (no state-pred / global losses)."""
    report_to: Literal["wandb", "tensorboard"] = "wandb"
    run_name: Optional[str] = None
    wandb_project: str = "action-latent-tokenizer-v3"
    resume: bool = False

    # ── Validation (with fixed-val support) ──
    val_ratio: float = 0.003
    val_seed: int = 42
    use_fixed_val: bool = True
    """True 이면 train/val split 을 JSON 으로 저장/로드 → 같은 dataset/같은 fixed_val_path 인 모든 v3 실험이 동일 val set"""

    fixed_val_path: Optional[str] = None
    """Fixed-val JSON 의 절대 경로. None 이면 <dataset>/meta/fixed_val_split.json"""


# =====================================================================
# Model builder
# =====================================================================


def _build_v3_tokenizer(config: ArgsConfig, action_dim: int, action_horizon: int):
    """ActionLatentTokenizerV3 생성. lambda=0 인 모듈은 None."""

    # Bottleneck 사용 시 출력 latent 차원. 그 외에는 emb_dim 과 동일.
    # global / hand_pred / text encoder 의 입출력 차원은 모두 이 값으로 정렬해야
    # encoder 출력 (token_dim) 과 일관됨.
    effective_token_dim = config.token_dim if config.use_bottleneck else config.emb_dim

    encoder = TimeWiseEncoderV3(
        action_dim=action_dim,
        action_horizon=action_horizon,
        emb_dim=config.emb_dim,
        head_dim=config.head_dim,
        depth=config.encoder_depth,
        pdropout=config.pdropout,
        num_global_tokens=config.num_global_tokens,
        num_hand_tokens=config.num_hand_tokens,
        output_layernorm=config.encoder_output_layernorm,
        use_bottleneck=config.use_bottleneck,
        token_dim=config.token_dim,
        compress_token=config.compress_token,
        use_vae=config.use_vae,
        kl_free_bits=config.kl_free_bits,
    )

    decoder_num_hand = config.num_hand_tokens if config.hand_in_recon else 0

    recon_decoder = ReconDecoderV3(
        action_dim=action_dim,
        action_horizon=action_horizon,
        emb_dim=config.emb_dim,
        head_dim=config.head_dim,
        depth=config.decoder_depth,
        pdropout=config.pdropout,
        decoder_mode=config.decoder_mode,
        num_global_tokens=config.num_global_tokens,
        num_hand_tokens=decoder_num_hand,
        use_bottleneck=config.use_bottleneck,
        token_dim=config.token_dim,
        compress_token=config.compress_token,
    )

    hand_pred_decoder = None
    if config.lambda_hand_pred > 0 and config.hand_state_dims and config.hand_pred_future_steps:
        hand_state_dim = len(config.hand_state_dims)
        num_future_steps = len(config.hand_pred_future_steps)
        if config.state_pred_kv_source == "time":
            # time tokens are compressed by compress_token (1 → no change)
            num_kv_tokens = action_horizon // config.compress_token
        else:
            num_kv_tokens = config.num_hand_tokens
        hand_pred_decoder = HandStatePredDecoderV3(
            hand_state_dim=hand_state_dim,
            emb_dim=config.emb_dim,
            head_dim=config.head_dim,
            depth=config.hand_pred_decoder_depth,
            pdropout=config.pdropout,
            num_future_steps=num_future_steps,
            num_kv_tokens=num_kv_tokens,
            use_bottleneck=config.use_bottleneck,
            token_dim=config.token_dim,
        )

    action_text_encoder = None
    global_loss_module = None
    if config.lambda_global > 0 and config.num_global_tokens > 0:
        # Global token 은 encoder 출력이므로 차원이 effective_token_dim 임.
        # Contrastive/regression 모두 latent 차원과 일치해야 하므로
        # text encoder 의 output_dim 도 effective_token_dim 으로 맞춤.
        text_output_dim = effective_token_dim

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
            emb_dim=effective_token_dim,
            text_feat_dim=text_output_dim,
            pool_type=config.global_pool_type,
            num_global_tokens=config.num_global_tokens,
        )

    return ActionLatentTokenizerV3(
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
        latent_noise_std=config.latent_noise_std,
        lambda_kl=config.lambda_kl,
    )


# =====================================================================
# Main
# =====================================================================


def main(config: ArgsConfig):
    """학습 메인 함수."""

    # ── 1. Dataset 선택 ──
    need_state = (
        (config.lambda_hand_pred > 0 or config.lambda_mask_hand_pred > 0)
        and config.hand_state_dims is not None
        and len(config.hand_state_dims) > 0
    )
    need_fast = config.lambda_global > 0

    if need_state or need_fast:
        assert config.cache_dataset, (
            "cache_dataset=False (on-the-fly loading) is only supported for the "
            "action-only path. The action+state / FAST-token path pre-computes "
            "hand_state / future_hand_states / fast_tokens during caching, which "
            "the on-the-fly source does not produce. Enable caching, or disable "
            "state-pred / global losses to use --no-cache-dataset."
        )
        from gr00t.data.dataset_action_state_pretransform_v3 import (
            ActionStateCollator,
            PreTransformedActionStateDatasetV3,
        )
        DatasetCls = PreTransformedActionStateDatasetV3
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
                use_fixed_val=config.use_fixed_val,
                fixed_val_path=config.fixed_val_path,
            )
    else:
        from gr00t.data.dataset_action_only import ActionOnlyCollator

        CollatorCls = ActionOnlyCollator

        if config.cache_dataset:
            from gr00t.data.dataset_action_only_pretransform_v3 import (
                PreTransformedActionOnlyDatasetV3,
            )
            DatasetCls = PreTransformedActionOnlyDatasetV3
        else:
            # On-the-fly: same item format ({"action": ...}) and collator as the
            # cached wrapper, so it is a drop-in. No pre-caching → GPUs are fed
            # from step 0 (relies on dataloader_num_workers > 0).
            from gr00t.data.dataset_action_only_v3 import ActionOnlyDatasetV3

            DatasetCls = ActionOnlyDatasetV3
            print(
                "[dataset] cache_dataset=False → action-only on-the-fly loading "
                f"(no pre-caching; dataloader_num_workers={config.dataloader_num_workers})."
            )

        def make_dataset(path, split):
            return DatasetCls(
                dataset_path=path,
                data_config_name=config.data_config,
                embodiment_tag=config.embodiment_tag,
                split=split,
                val_ratio=config.val_ratio,
                val_seed=config.val_seed,
                normalization_mode=config.normalization_mode,
                use_fixed_val=config.use_fixed_val,
                fixed_val_path=config.fixed_val_path,
            )

    datasets_train = []
    datasets_val = []
    for path in config.dataset_path:
        assert os.path.exists(path), f"Dataset path가 존재하지 않습니다: {path}"
        datasets_train.append(make_dataset(path, "train"))
        datasets_val.append(make_dataset(path, "val"))

    # ConcatDataset does NOT merge normalization stats — without this each
    # dataset would normalize actions with its own single-dataset min/max.
    # Merge across all datasets (matching LeRobotMixtureDataset / the VLA) and
    # apply to train+val so the whole-mixture statistics are used. No-op for 1.
    from gr00t.data.merge_norm_stats import apply_merged_normalization_metadata

    apply_merged_normalization_metadata(datasets_train, datasets_train + datasets_val)

    if len(datasets_train) == 1:
        train_dataset = datasets_train[0]
        val_dataset = datasets_val[0]
    else:
        train_dataset = torch.utils.data.ConcatDataset(datasets_train)
        val_dataset = torch.utils.data.ConcatDataset(datasets_val)

    # ── 2. action_dim/horizon 추출 ──
    sample = datasets_train[0][0]
    action_horizon, action_dim = sample["action"].shape
    print(f"action_horizon={action_horizon}, action_dim={action_dim}")

    # ── 3. Model 생성 ──
    model = _build_v3_tokenizer(config, action_dim, action_horizon)

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
        save_total_limit=config.save_total_limit if config.save_total_limit > 0 else None,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        report_to=config.report_to,
        seed=42,
        ddp_find_unused_parameters=False,
    )

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

    # ── 5. Trainer ──
    collator = CollatorCls()

    trainer = ActionLatentV3Trainer(
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
        save_path = os.path.join(config.output_dir, "action_latent_tokenizer_v3_final.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "tokenizer_version": "v3",
                    "action_dim": action_dim,
                    "action_horizon": action_horizon,
                    "emb_dim": config.emb_dim,
                    "head_dim": config.head_dim,
                    "encoder_depth": config.encoder_depth,
                    "decoder_depth": config.decoder_depth,
                    "decoder_mode": config.decoder_mode,
                    "pdropout": config.pdropout,
                    "encoder_output_layernorm": config.encoder_output_layernorm,
                    "latent_noise_std": config.latent_noise_std,
                    "use_bottleneck": config.use_bottleneck,
                    "token_dim": config.token_dim,
                    "compress_token": config.compress_token,
                    "use_vae": config.use_vae,
                    "lambda_kl": config.lambda_kl,
                    "kl_free_bits": config.kl_free_bits,
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
                    "use_fixed_val": config.use_fixed_val,
                    "fixed_val_path": config.fixed_val_path,
                    "val_ratio": config.val_ratio,
                    "val_seed": config.val_seed,
                    "data_config": config.data_config,
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
    print("ACTION LATENT TOKENIZER V3 TRAINING CONFIGURATION:")
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

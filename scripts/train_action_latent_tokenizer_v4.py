"""Action Latent Tokenizer V4 (RLA-DINO hybrid) training script.

V4 fuses the V3 action autoencoder with rla-wm's DINO inverse-dynamics
autoencoder. Per step:
  1. The dataset yields an action chunk + two RGB frames (chunk start / end).
  2. A frozen DINOv3 extractor (trainer-owned, on-the-fly) turns the frames into
     patch-token features ``x0_feat`` / ``x1_feat`` ([B, Lp, dino_channels]).
  3. The V4 tokenizer encodes (action latents as RLA queries + DINO-diff) → a
     64-dim latent, then decodes it BOTH to actions (recon) and to future DINO
     features (dino recon). Loss = lambda_recon * recon + lambda_dino * dino.

Cloned from ``train_action_latent_tokenizer_v3.py`` (logging / launcher / fixed-val
conventions preserved). Action-recon + DINO-recon only (no mask / global / hand).
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

# Side-effect import to register any extra data configs (kept for parity with v3).
import gr00t.experiment.data_config_v3  # noqa: F401
from gr00t.data.dataset_action_frames_v4 import (
    ActionFramesCollatorV4,
    ActionFramesDatasetV4,
)
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.trainer import BaseSampler
from gr00t.model.action_latent_tokenizer_v4 import (
    ActionLatentTokenizerV4,
    ReconDecoderV4,
    TimeWiseEncoderV4,
)
from gr00t.model.rla_modules import SimpleTokenTransformer
from gr00t.utils.dino import DINOv3FeatureExtractor


# =====================================================================
# Trainer
# =====================================================================


class ActionLatentV4Trainer(transformers.Trainer):
    """V4 trainer: owns a frozen DINO extractor, extracts feats on-the-fly."""

    def __init__(
        self,
        *args,
        dino_model: str,
        dino_channels: int,
        feature_source: str = "dino",
        vggt_token_source: str = "dpt_out2",
        vggt_model: str = "facebook/VGGT-1B",
        vggt_image_size: int = 224,
        dino_final_norm: str = "affine",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.feature_source = feature_source
        if feature_source == "vggt":
            from gr00t.utils.vggt_feature import VGGTFeatureExtractor

            self.dino = VGGTFeatureExtractor(
                model_name=vggt_model,
                token_source=vggt_token_source,
                image_size=vggt_image_size,
                use_compile=False,
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
            f"dino_channels={dino_channels}. For dino, check --dino-model (a silent "
            "fallback to dinov2-small gives 384). For vggt, set --dino-channels to "
            "match --vggt-token-source (dpt_out2→1024, aggregator→2048)."
        )
        self._dino_on_device = False

    def _get_train_sampler(self, *args, **kwargs):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def get_eval_dataloader(self, eval_dataset=None):
        if not hasattr(self, "_cached_eval_dataloader"):
            self._cached_eval_dataloader = super().get_eval_dataloader(eval_dataset)
        return self._cached_eval_dataloader

    @torch.no_grad()
    def _extract_feats(self, inputs):
        """frames (uint8 [B,3,H,W]) → DINO patch features [B, Lp, C] (fp32)."""
        device = self.model.device if hasattr(self.model, "device") else next(self.model.parameters()).device
        if not self._dino_on_device:
            self.dino.to(device)
            self._dino_on_device = True

        f0 = inputs["frame_x0"].to(device).float() / 255.0
        f1 = inputs["frame_x1"].to(device).float() / 255.0
        if self.feature_source == "vggt":
            # VGGT extractor returns patch tokens [B, Lp, C] directly.
            x0, _ = self.dino(f0)
            x1, _ = self.dino(f1)
            return x0.float(), x1.float()
        _, g0 = self.dino(f0, return_spatial_grid=True)  # [B, C, h, w] fp16
        _, g1 = self.dino(f1, return_spatial_grid=True)
        x0 = g0.flatten(2).transpose(1, 2).float()       # [B, h*w, C]
        x1 = g1.flatten(2).transpose(1, 2).float()
        return x0, x1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        x0_feat, x1_feat = self._extract_feats(inputs)
        batch = {"action": inputs["action"], "x0_feat": x0_feat, "x1_feat": x1_feat}
        outputs = model(batch)
        loss = outputs["loss"]
        if model.training:
            if not hasattr(self, "_train_loss_buffer"):
                self._train_loss_buffer = {}
            for key in ("loss_recon", "loss_dino", "loss_dino_l1", "loss_dino_mse", "loss_dino_cosine"):
                val = outputs.get(key)
                if val is not None:
                    v = val.item() if isinstance(val, torch.Tensor) else float(val)
                    self._train_loss_buffer.setdefault(key, []).append(v)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Eval forward. HF's default eval loop calls ``model(**inputs)`` with the
        RAW batch (no x0_feat/x1_feat) → KeyError. Override to extract DINO feats
        from the frames first, mirroring compute_loss. Returns (loss, None, None)."""
        with torch.no_grad():
            x0_feat, x1_feat = self._extract_feats(inputs)
            batch = {"action": inputs["action"], "x0_feat": x0_feat, "x1_feat": x1_feat}
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
        """Standard eval + action recon (MSE/L1) and DINO recon (L1/cosine) metrics."""
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model
        model.eval()

        total_mse = total_l1 = total_dino_l1 = total_dino_cos = 0.0
        n_samples = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = self._prepare_inputs(batch)
                x0_feat, x1_feat = self._extract_feats(batch)
                actions = batch["action"].to(dtype=model.encoder.action_proj.weight.dtype)

                g, t, h = model.encode(actions, x0_feat, x1_feat)
                preds = model.decode(g, t, h)
                pred_x1 = model.decode_dino(t, x0_feat)

                B = actions.shape[0]
                total_mse += F.mse_loss(preds, actions).item() * B
                total_l1 += F.l1_loss(preds, actions).item() * B
                total_dino_l1 += F.l1_loss(pred_x1, x1_feat.to(dtype=pred_x1.dtype)).item() * B
                total_dino_cos += (
                    1.0 - F.cosine_similarity(pred_x1, x1_feat.to(dtype=pred_x1.dtype), dim=-1).mean()
                ).item() * B
                n_samples += B

        if n_samples > 0:
            extra = {
                f"{metric_key_prefix}_recon_mse": total_mse / n_samples,
                f"{metric_key_prefix}_recon_l1": total_l1 / n_samples,
                f"{metric_key_prefix}_dino_l1": total_dino_l1 / n_samples,
                f"{metric_key_prefix}_dino_cos_dist": total_dino_cos / n_samples,
            }
            self.log(extra)
            metrics.update(extra)

        return metrics


# =====================================================================
# Config
# =====================================================================


@dataclass
class ArgsConfig:
    """Action Latent Tokenizer V4 training config."""

    # ── Dataset ──
    dataset_path: List[str]
    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_waist"
    embodiment_tag: str = "new_embodiment"
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
    # "dino" (default): unchanged DINO path. "vggt": replace DINO feats with VGGT
    # patch tokens. ``dino_channels`` must match the chosen VGGT source width
    # (dpt_out2 → 1024, aggregator → 2048).
    feature_source: Literal["dino", "vggt"] = "dino"
    vggt_token_source: Literal["aggregator", "dpt_out2"] = "dpt_out2"
    vggt_model: str = "facebook/VGGT-1B"
    vggt_image_size: int = 224
    # DINO final LayerNorm: "affine" (default, standard last_hidden_state) or
    # "naive" (drop the final LN's learned γ/β, normalize only). dino source only.
    dino_final_norm: Literal["affine", "naive"] = "affine"

    # ── DINO decoder ──
    dino_decoder_depth: int = 12

    # ── Loss ──
    lambda_recon: float = 1.0
    lambda_dino: float = 1.0
    recon_loss_type: Literal["mse", "l1"] = "mse"
    dino_loss_type: str = "l1+mse"  # RLA default (L1 + MSE). Also: "cosine", "l1+cosine" ...
    dino_w_l1: float = 1.0
    dino_w_mse: float = 1.0
    dino_w_cosine: float = 1.0

    # ── Frames / DINO input ──
    image_size: int = 224
    video_backend: str = "decord"
    """Video backend for frame loading. Use 'decord' (nearest-frame mapping);
    'torchvision_av' requires exact pts match and can drop frames."""

    # ── Training ──
    output_dir: str = "/tmp/action_latent_tokenizer_v4"
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
    wandb_project: str = "action-latent-tokenizer-v4"
    resume: bool = False

    # ── Validation ──
    val_ratio: float = 0.003
    val_seed: int = 42
    use_fixed_val: bool = True
    fixed_val_path: Optional[str] = None


# =====================================================================
# Model builder
# =====================================================================


def _build_v4_tokenizer(config: ArgsConfig, action_dim: int, action_horizon: int):
    encoder = TimeWiseEncoderV4(
        action_dim=action_dim,
        action_horizon=action_horizon,
        emb_dim=config.emb_dim,
        head_dim=config.head_dim,
        encoder_depth=config.encoder_depth,
        pdropout=config.pdropout,
        num_global_tokens=0,
        num_hand_tokens=0,
        dino_dim=config.dino_channels,
        fusion_width=config.fusion_width,
        fusion_depth=config.fusion_depth,
        fusion_heads=config.fusion_heads,
        token_dim=config.token_dim,
    )

    recon_decoder = ReconDecoderV4(
        action_dim=action_dim,
        action_horizon=action_horizon,
        emb_dim=config.emb_dim,
        head_dim=config.head_dim,
        depth=config.decoder_depth,
        pdropout=config.pdropout,
        decoder_mode=config.decoder_mode,
        num_global_tokens=0,
        num_hand_tokens=0,
        token_dim=config.token_dim,
    )

    # DINO decoder: x0 feats + latent (token_channels=token_dim) → predict x1 feats.
    # num_tokens = action_horizon (internal learnable tokens added to the external
    # latent), mirroring rla-wm's decoder which uses internal tokens too.
    dino_decoder = SimpleTokenTransformer(
        in_channels=config.dino_channels,
        model_channels=config.fusion_width,
        out_channels=config.dino_channels,
        num_blocks=config.dino_decoder_depth,
        num_heads=config.fusion_heads,
        num_tokens=action_horizon,
        token_channels=config.token_dim,
        zero_init=True,
        use_fp16=False,
    )

    return ActionLatentTokenizerV4(
        encoder=encoder,
        recon_decoder=recon_decoder,
        dino_decoder=dino_decoder,
        lambda_recon=config.lambda_recon,
        lambda_dino=config.lambda_dino,
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
        dino_final_norm=config.dino_final_norm,
    )


# =====================================================================
# Main
# =====================================================================


def main(config: ArgsConfig):
    def make_dataset(path, split):
        return ActionFramesDatasetV4(
            dataset_path=path,
            data_config_name=config.data_config,
            embodiment_tag=config.embodiment_tag,
            split=split,
            val_ratio=config.val_ratio,
            val_seed=config.val_seed,
            normalization_mode=config.normalization_mode,
            image_size=config.image_size,
            video_backend=config.video_backend,
            use_fixed_val=config.use_fixed_val,
            fixed_val_path=config.fixed_val_path,
        )

    datasets_train, datasets_val = [], []
    for path in config.dataset_path:
        assert os.path.exists(path), f"Dataset path does not exist: {path}"
        datasets_train.append(make_dataset(path, "train"))
        datasets_val.append(make_dataset(path, "val"))

    if len(datasets_train) == 1:
        train_dataset, val_dataset = datasets_train[0], datasets_val[0]
    else:
        train_dataset = torch.utils.data.ConcatDataset(datasets_train)
        val_dataset = torch.utils.data.ConcatDataset(datasets_val)

    sample = datasets_train[0][0]
    action_horizon, action_dim = sample["action"].shape
    print(f"action_horizon={action_horizon}, action_dim={action_dim}")

    model = _build_v4_tokenizer(config, action_dim, action_horizon)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable:,}")

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
        # V4: the fusion transformer computes outputs at the DINO-token positions
        # that the latent readout discards → guard against DDP unused-param errors.
        ddp_find_unused_parameters=True,
    )

    world_size = max(1, config.num_gpus)
    micro_batch_global = config.batch_size * world_size
    steps_per_epoch = math.ceil(len(train_dataset) / micro_batch_global)
    print(
        f"[TrainInfo] train={len(train_dataset):,} val={len(val_dataset):,} "
        f"micro_batch(global)={micro_batch_global:,} steps/epoch={steps_per_epoch:,}"
    )

    collator = ActionFramesCollatorV4()

    trainer = ActionLatentV4Trainer(
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
        dino_final_norm=config.dino_final_norm,
    )

    if config.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = config.wandb_project
        os.environ["WANDB_DIR"] = config.output_dir

    trainer.train(resume_from_checkpoint=config.resume)

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        save_path = os.path.join(config.output_dir, "action_latent_tokenizer_v4_final.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "tokenizer_version": "v4",
                    "action_dim": action_dim,
                    "action_horizon": action_horizon,
                    "emb_dim": config.emb_dim,
                    "head_dim": config.head_dim,
                    "encoder_depth": config.encoder_depth,
                    "decoder_depth": config.decoder_depth,
                    "decoder_mode": config.decoder_mode,
                    "pdropout": config.pdropout,
                    "token_dim": config.token_dim,
                    "dino_model": config.dino_model,
                    "dino_channels": config.dino_channels,
                    "fusion_width": config.fusion_width,
                    "fusion_depth": config.fusion_depth,
                    "fusion_heads": config.fusion_heads,
                    "dino_decoder_depth": config.dino_decoder_depth,
                    "feature_source": config.feature_source,
                    "vggt_token_source": config.vggt_token_source,
                    "vggt_model": config.vggt_model,
                    "vggt_image_size": config.vggt_image_size,
                    "dino_final_norm": config.dino_final_norm,
                    "lambda_recon": config.lambda_recon,
                    "lambda_dino": config.lambda_dino,
                    "recon_loss_type": config.recon_loss_type,
                    "dino_loss_type": config.dino_loss_type,
                    "image_size": config.image_size,
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
    print("ACTION LATENT TOKENIZER V4 TRAINING CONFIGURATION:")
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

"""Action Latent Tokenizer V5 (RLA-LAM hybrid) training script.

V5 is V4 with the visual element swapped from DINO/VGGT to DreamDojo's Latent
Action Model (LAM):

  1. A frozen LAM encoder (trainer-owned, on-the-fly) turns the (frame0, frame1)
     pair into a single latent-action token ``z_rep`` ([B, 1, lam_latent_dim]).
  2. The V5 tokenizer encodes (action latents as RLA queries + z_rep) → a
     token_dim latent, then decodes it BOTH to actions (recon) and to the
     future-frame pixels (LAM SpatioTransformer pixel decoder). The per-timestep
     latents are merged into one (for the pixel decoder) by a learnable softmax
     weighted sum. Loss = lambda_recon * recon + lambda_pixel * pixel.

The training / eval / save / DDP structure is identical to V4 (same Trainer,
sampler, eval-metric harness, torchrun relaunch). Only the feature extractor
(DINO/VGGT → LAM), the second decoder (DINO-feature → pixel), and the associated
loss/metric wiring change.
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

# Side-effect import to register any extra data configs (kept for parity with v4).
import gr00t.experiment.data_config_v3  # noqa: F401
from gr00t.data.dataset_action_frames_v4 import (
    ActionFramesCollatorV4,
    ActionFramesDatasetV4,
)
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.trainer import BaseSampler
from gr00t.model.action_latent_tokenizer_v4 import ReconDecoderV4, TimeWiseEncoderV4
from gr00t.model.action_latent_tokenizer_v5 import (
    ActionLatentTokenizerV5,
    PixelDecoderV5,
)
from gr00t.utils.lam_feature import LAMFeatureExtractor, resolve_lam_ckpt


# =====================================================================
# LAM pixel-decoder checkpoint init
# =====================================================================


def _init_pixel_decoder_from_lam(pixel_decoder: PixelDecoderV5, ckpt_path: str) -> None:
    """Initialize the V5 pixel decoder's ``patch_up`` / ``decoder`` from LAM weights.

    The pretrained LAM Lightning checkpoint stores keys under ``lam.*`` (e.g.
    ``lam.decoder.ffn.0.weight``, ``lam.patch_up.weight``). We copy only the
    ``decoder.*`` and ``patch_up.*`` subtrees (submodule names match by design).
    ``action_up`` is left fresh-initialized (LAM is 32→model_dim, ours is
    token_dim→model_dim), and ``pool_logits`` is fresh (zeros → uniform mean).
    """
    # Resolve to a local path (auto-downloads from HF nvidia/DreamDojo if absent).
    ckpt_path = resolve_lam_ckpt(ckpt_path)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    want = {}
    for k, v in sd.items():
        if k.startswith("lam.decoder.") or k.startswith("lam.patch_up."):
            want[k[len("lam."):]] = v  # strip "lam." → "decoder.*" / "patch_up.*"
    missing, unexpected = pixel_decoder.load_state_dict(want, strict=False)
    # Expected-missing: pool_logits + action_up.* (fresh). Unexpected should be empty.
    fresh = [m for m in missing if not (m.startswith("action_up") or m == "pool_logits")]
    print(
        f"[v5] LAM pixel-decoder init from {ckpt_path}: loaded {len(want)} tensors "
        f"(decoder/patch_up). fresh={['pool_logits', 'action_up']} "
        f"unexpected={list(unexpected)[:5]} other_missing={fresh[:5]}"
    )


# =====================================================================
# Trainer
# =====================================================================


class ActionLatentV5Trainer(transformers.Trainer):
    """V5 trainer: owns a frozen LAM extractor, extracts z_rep on-the-fly."""

    def __init__(
        self,
        *args,
        lam_ckpt: str,
        lam_model_dim: int = 1024,
        lam_latent_dim: int = 32,
        lam_patch_size: int = 16,
        lam_enc_blocks: int = 24,
        lam_dec_blocks: int = 24,
        lam_num_heads: int = 16,
        lam_image_h: int = 240,
        lam_image_w: int = 320,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lam_image_h = lam_image_h
        self.lam_image_w = lam_image_w
        self.lam = LAMFeatureExtractor(
            ckpt_path=lam_ckpt,
            model_dim=lam_model_dim,
            latent_dim=lam_latent_dim,
            patch_size=lam_patch_size,
            enc_blocks=lam_enc_blocks,
            dec_blocks=lam_dec_blocks,
            num_heads=lam_num_heads,
            image_h=lam_image_h,
            image_w=lam_image_w,
        )
        self.lam.eval()
        for p in self.lam.parameters():
            p.requires_grad = False
        assert self.lam.latent_dim == lam_latent_dim
        self._lam_on_device = False

    def _get_train_sampler(self, *args, **kwargs):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def get_eval_dataloader(self, eval_dataset=None):
        if not hasattr(self, "_cached_eval_dataloader"):
            self._cached_eval_dataloader = super().get_eval_dataloader(eval_dataset)
        return self._cached_eval_dataloader

    @torch.no_grad()
    def _extract(self, inputs):
        """frames (uint8 [B,3,H,W]) → (z_rep [B,1,latent], frame0/frame1 [B,1,Hl,Wl,3]).

        Frames are resized to the LAM training resolution (Hl×Wl) and arranged as
        ``[B, 2, Hl, Wl, 3]`` videos; the frozen LAM encoder produces z_rep, and the
        same videos provide the pixel-decoder input (frame0) and target (frame1).
        """
        device = self.model.device if hasattr(self.model, "device") else next(self.model.parameters()).device
        if not self._lam_on_device:
            self.lam.to(device)
            self._lam_on_device = True

        def to_frame(key):
            f = inputs[key].to(device).float() / 255.0  # [B,3,H,W]
            if f.shape[-2:] != (self.lam_image_h, self.lam_image_w):
                f = F.interpolate(
                    f, size=(self.lam_image_h, self.lam_image_w),
                    mode="bilinear", align_corners=False,
                )
            return f.permute(0, 2, 3, 1)  # [B,Hl,Wl,3]

        f0 = to_frame("frame_x0")
        f1 = to_frame("frame_x1")
        videos = torch.stack([f0, f1], dim=1)  # [B,2,Hl,Wl,3]
        z_rep = self.lam(videos).float()       # [B,1,latent]
        frame0 = videos[:, :1]                 # [B,1,Hl,Wl,3]
        frame1 = videos[:, 1:]                 # [B,1,Hl,Wl,3]
        return z_rep, frame0, frame1

    def _build_batch(self, inputs):
        z_rep, frame0, frame1 = self._extract(inputs)
        return {"action": inputs["action"], "z_rep": z_rep, "frame0": frame0, "frame1": frame1}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        batch = self._build_batch(inputs)
        outputs = model(batch)
        loss = outputs["loss"]
        if model.training:
            if not hasattr(self, "_train_loss_buffer"):
                self._train_loss_buffer = {}
            for key in ("loss_recon", "loss_pixel", "loss_kl"):
                val = outputs.get(key)
                if val is not None:
                    v = val.item() if isinstance(val, torch.Tensor) else float(val)
                    self._train_loss_buffer.setdefault(key, []).append(v)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Eval forward. HF's default eval loop calls ``model(**inputs)`` with the
        RAW batch (no z_rep/frames extracted) → KeyError. Override to extract LAM
        z_rep + frames first, mirroring compute_loss. Returns (loss, None, None)."""
        with torch.no_grad():
            batch = self._build_batch(inputs)
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
        """Standard eval + action recon (MSE/L1) and pixel recon (MSE) metrics."""
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model
        model.eval()

        total_mse = total_l1 = total_pixel_mse = 0.0
        n_samples = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = self._prepare_inputs(batch)
                z_rep, frame0, frame1 = self._extract(batch)
                actions = batch["action"].to(dtype=model.encoder.action_proj.weight.dtype)

                g, t, h = model.encode(actions, z_rep)
                preds = model.decode(g, t, h)
                frame1_hat = model.decode_pixel(t, frame0.to(dtype=preds.dtype))

                B = actions.shape[0]
                total_mse += F.mse_loss(preds, actions).item() * B
                total_l1 += F.l1_loss(preds, actions).item() * B
                total_pixel_mse += F.mse_loss(
                    frame1_hat, frame1.to(dtype=frame1_hat.dtype)
                ).item() * B
                n_samples += B

        if n_samples > 0:
            extra = {
                f"{metric_key_prefix}_recon_mse": total_mse / n_samples,
                f"{metric_key_prefix}_recon_l1": total_l1 / n_samples,
                f"{metric_key_prefix}_pixel_mse": total_pixel_mse / n_samples,
            }
            self.log(extra)
            metrics.update(extra)

        return metrics


# =====================================================================
# Config
# =====================================================================


@dataclass
class ArgsConfig:
    """Action Latent Tokenizer V5 training config."""

    # ── Dataset ──
    dataset_path: List[str]
    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_waist"
    embodiment_tag: str = "new_embodiment"
    normalization_mode: str = "min_max"

    # ── Action encoder (V4-style) ──
    emb_dim: int = 256
    head_dim: int = 64
    encoder_depth: int = 4
    decoder_depth: int = 2
    decoder_mode: Literal["self_attention", "cross_attention"] = "self_attention"
    pdropout: float = 0.0
    token_dim: int = 64

    # ── Fusion (RLA SimpleTokenTransformer) ──
    fusion_width: int = 1024
    fusion_depth: int = 12
    fusion_heads: int = 16

    # ── LAM visual source (frozen extractor + pixel decoder init) ──
    lam_ckpt: str = "DreamDojo/checkpoints/DreamDojo/LAM_400k.ckpt"
    lam_model_dim: int = 1024
    lam_latent_dim: int = 32  # = fusion visual in_channels (z_rep width)
    lam_patch_size: int = 16
    lam_enc_blocks: int = 24
    lam_dec_blocks: int = 24
    lam_num_heads: int = 16
    lam_image_h: int = 240
    lam_image_w: int = 320

    # ── Loss ──
    lambda_recon: float = 1.0
    lambda_pixel: float = 1.0
    recon_loss_type: Literal["mse", "l1"] = "mse"
    pixel_loss_type: Literal["mse", "l1"] = "mse"

    # ── VAE bottleneck (SD-style, opt-in; identical semantics to V4) ──
    use_vae: bool = False
    lambda_kl: float = 1e-6
    kl_free_bits: float = 0.0

    # ── Frames ──
    image_size: int = 256
    video_backend: str = "decord"

    # ── Training ──
    output_dir: str = "/tmp/action_latent_tokenizer_v5"
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
    wandb_project: str = "action-latent-tokenizer-v5"
    resume: bool = False

    # ── Validation ──
    val_ratio: float = 0.003
    val_seed: int = 42
    use_fixed_val: bool = True
    fixed_val_path: Optional[str] = None


# =====================================================================
# Model builder
# =====================================================================


def _build_v5_tokenizer(config: ArgsConfig, action_dim: int, action_horizon: int):
    encoder = TimeWiseEncoderV4(
        action_dim=action_dim,
        action_horizon=action_horizon,
        emb_dim=config.emb_dim,
        head_dim=config.head_dim,
        encoder_depth=config.encoder_depth,
        pdropout=config.pdropout,
        num_global_tokens=0,
        num_hand_tokens=0,
        dino_dim=config.lam_latent_dim,  # fusion visual context = z_rep width
        fusion_width=config.fusion_width,
        fusion_depth=config.fusion_depth,
        fusion_heads=config.fusion_heads,
        token_dim=config.token_dim,
        use_vae=config.use_vae,
        kl_free_bits=config.kl_free_bits,
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

    pixel_decoder = PixelDecoderV5(
        action_horizon=action_horizon,
        token_dim=config.token_dim,
        image_channels=3,
        patch_size=config.lam_patch_size,
        model_dim=config.lam_model_dim,
        dec_blocks=config.lam_dec_blocks,
        num_heads=config.lam_num_heads,
    )
    # Initialize patch_up + decoder from the pretrained LAM checkpoint (trainable).
    _init_pixel_decoder_from_lam(pixel_decoder, config.lam_ckpt)

    return ActionLatentTokenizerV5(
        encoder=encoder,
        recon_decoder=recon_decoder,
        pixel_decoder=pixel_decoder,
        lambda_recon=config.lambda_recon,
        lambda_pixel=config.lambda_pixel,
        lambda_kl=config.lambda_kl,
        recon_loss_type=config.recon_loss_type,
        pixel_loss_type=config.pixel_loss_type,
        lam_ckpt=config.lam_ckpt,
        lam_model_dim=config.lam_model_dim,
        lam_latent_dim=config.lam_latent_dim,
        lam_patch_size=config.lam_patch_size,
        lam_enc_blocks=config.lam_enc_blocks,
        lam_dec_blocks=config.lam_dec_blocks,
        lam_num_heads=config.lam_num_heads,
        lam_image_h=config.lam_image_h,
        lam_image_w=config.lam_image_w,
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

    # ConcatDataset does NOT merge normalization stats — without this each
    # dataset would normalize actions with its own single-dataset min/max.
    # Merge across all datasets (matching LeRobotMixtureDataset / the VLA) and
    # apply to train+val so the whole-mixture statistics are used. No-op for 1.
    from gr00t.data.merge_norm_stats import apply_merged_normalization_metadata

    apply_merged_normalization_metadata(datasets_train, datasets_train + datasets_val)

    if len(datasets_train) == 1:
        train_dataset, val_dataset = datasets_train[0], datasets_val[0]
    else:
        train_dataset = torch.utils.data.ConcatDataset(datasets_train)
        val_dataset = torch.utils.data.ConcatDataset(datasets_val)

    sample = datasets_train[0][0]
    action_horizon, action_dim = sample["action"].shape
    print(f"action_horizon={action_horizon}, action_dim={action_dim}")

    model = _build_v5_tokenizer(config, action_dim, action_horizon)
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
        # V5: the fusion transformer computes outputs at the z_rep-token position
        # that the latent readout discards; pixel decoder params are unused when
        # lambda_pixel=0 → guard against DDP unused-param errors.
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

    trainer = ActionLatentV5Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        lam_ckpt=config.lam_ckpt,
        lam_model_dim=config.lam_model_dim,
        lam_latent_dim=config.lam_latent_dim,
        lam_patch_size=config.lam_patch_size,
        lam_enc_blocks=config.lam_enc_blocks,
        lam_dec_blocks=config.lam_dec_blocks,
        lam_num_heads=config.lam_num_heads,
        lam_image_h=config.lam_image_h,
        lam_image_w=config.lam_image_w,
    )

    if config.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = config.wandb_project
        os.environ["WANDB_DIR"] = config.output_dir

    trainer.train(resume_from_checkpoint=config.resume)

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        save_path = os.path.join(config.output_dir, "action_latent_tokenizer_v5_final.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "tokenizer_version": "v5",
                    "action_dim": action_dim,
                    "action_horizon": action_horizon,
                    "emb_dim": config.emb_dim,
                    "head_dim": config.head_dim,
                    "encoder_depth": config.encoder_depth,
                    "decoder_depth": config.decoder_depth,
                    "decoder_mode": config.decoder_mode,
                    "pdropout": config.pdropout,
                    "token_dim": config.token_dim,
                    "fusion_width": config.fusion_width,
                    "fusion_depth": config.fusion_depth,
                    "fusion_heads": config.fusion_heads,
                    "lam_ckpt": config.lam_ckpt,
                    "lam_model_dim": config.lam_model_dim,
                    "lam_latent_dim": config.lam_latent_dim,
                    "lam_patch_size": config.lam_patch_size,
                    "lam_enc_blocks": config.lam_enc_blocks,
                    "lam_dec_blocks": config.lam_dec_blocks,
                    "lam_num_heads": config.lam_num_heads,
                    "lam_image_h": config.lam_image_h,
                    "lam_image_w": config.lam_image_w,
                    "lambda_recon": config.lambda_recon,
                    "lambda_pixel": config.lambda_pixel,
                    "lambda_kl": config.lambda_kl,
                    "use_vae": config.use_vae,
                    "kl_free_bits": config.kl_free_bits,
                    "recon_loss_type": config.recon_loss_type,
                    "pixel_loss_type": config.pixel_loss_type,
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
    print("ACTION LATENT TOKENIZER V5 TRAINING CONFIGURATION:")
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

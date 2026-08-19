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
from gr00t.data.dataset_dino_cache_v4 import (
    CachedActionFramesCollatorV4,
    CachedActionFramesDatasetV4,
)
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.trainer import BaseSampler, S3CompatCheckpointStaging
from gr00t.model.action_latent_tokenizer_v4 import (
    ActionLatentTokenizerV4,
    ReconDecoderV4,
    TimeWiseEncoderV4,
    UnifiedDecoderV4,
)
from gr00t.model.rla_modules import SimpleTokenTransformer
from gr00t.utils.dino import DINOv3FeatureExtractor


# =====================================================================
# Trainer
# =====================================================================


class ActionLatentV4Trainer(S3CompatCheckpointStaging, transformers.Trainer):
    """V4 trainer: owns a frozen DINO extractor, extracts feats on-the-fly.

    S3CompatCheckpointStaging is inert unless GR00T_S3_COMPAT=1 (gpu26).
    """

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
        use_dino_cache: bool = False,
        use_seg_stream: bool = False,
        seg_feats_only: bool = False,
        pass_masks: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.feature_source = feature_source
        self.use_dino_cache = use_dino_cache
        # Extract s0/s1 cutout features even though the encoder has no seg stream —
        # they feed ONLY the seg DINO decoder's ctx/target (EXP-0004 decoder-only
        # mode). False (default) leaves _extract_seg_feats gating byte-identical.
        self.seg_feats_only = bool(seg_feats_only)
        # Forward the dataset's mask_x1/mask_valid to the model batch (mask-weighted
        # dino loss / seg-pixel decoder). False (default) leaves the batch dict
        # byte-identical to before.
        self.pass_masks = bool(pass_masks)
        # Segment (cutout) stream: the SAME frozen extractor embeds the seg frames, so
        # no second extractor is built — only two extra forward passes per step.
        self.use_seg_stream = bool(use_seg_stream)
        if use_dino_cache:
            # Features come precomputed from the dataset (x0_feat/x1_feat) → no
            # extractor is built at all (saves GPU memory + the DINO forward).
            self.dino = None
            self._dino_on_device = True
            return
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
        """frames (uint8 [B,3,H,W]) → DINO patch features [B, Lp, C] (fp32).

        With ``use_dino_cache`` the dataset already supplies precomputed feats as
        ``x0_feat``/``x1_feat``; we just move + cast them (mirroring the live
        path's trailing ``.float()``) so cached training is value-identical."""
        device = self.model.device if hasattr(self.model, "device") else next(self.model.parameters()).device
        if self.use_dino_cache:
            return (
                inputs["x0_feat"].to(device).float(),
                inputs["x1_feat"].to(device).float(),
            )
        if not self._dino_on_device:
            self.dino.to(device)
            self._dino_on_device = True

        f0 = inputs["frame_x0"].to(device).float() / 255.0
        f1 = inputs["frame_x1"].to(device).float() / 255.0
        return self._frames_to_feats(f0), self._frames_to_feats(f1)

    def _frames_to_feats(self, frames):
        """Normalized frames [B,3,H,W] in [0,1] → patch features [B, Lp, C] (fp32)."""
        if self.feature_source == "vggt":
            # VGGT extractor returns patch tokens [B, Lp, C] directly.
            tok, _ = self.dino(frames)
            return tok.float()
        _, grid = self.dino(frames, return_spatial_grid=True)  # [B, C, h, w] fp16
        return grid.flatten(2).transpose(1, 2).float()         # [B, h*w, C]

    @torch.no_grad()
    def _extract_seg_feats(self, inputs):
        """Segment (cutout) frames → features [B, Lp, C], via the SAME frozen extractor.

        Returns (None, None) when the seg stream is off, so the batch built below stays
        byte-identical to the pre-seg trainer. ``seg_feats_only`` also enables the
        extraction (decoder-only mode — the model routes s0/s1 to the seg decoder
        while the encoder ignores them).
        """
        if not (self.use_seg_stream or self.seg_feats_only):
            return None, None
        device = self.model.device if hasattr(self.model, "device") else next(self.model.parameters()).device
        if not self._dino_on_device:
            self.dino.to(device)
            self._dino_on_device = True
        s0 = inputs["seg_x0"].to(device).float() / 255.0
        s1 = inputs["seg_x1"].to(device).float() / 255.0
        return self._frames_to_feats(s0), self._frames_to_feats(s1)

    def _build_batch(self, inputs):
        """{action, x0_feat, x1_feat} (+ {s0_feat, s1_feat} when the seg stream is on)."""
        x0_feat, x1_feat = self._extract_feats(inputs)
        batch = {"action": inputs["action"], "x0_feat": x0_feat, "x1_feat": x1_feat}
        s0_feat, s1_feat = self._extract_seg_feats(inputs)
        if s0_feat is not None:
            batch["s0_feat"] = s0_feat
            batch["s1_feat"] = s1_feat
        if self.pass_masks:
            batch["mask_x1"] = inputs["mask_x1"]
            batch["mask_valid"] = inputs["mask_valid"]
        return batch

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        batch = self._build_batch(inputs)
        outputs = model(batch)
        loss = outputs["loss"]
        if model.training:
            if not hasattr(self, "_train_loss_buffer"):
                self._train_loss_buffer = {}
            for key in ("loss_recon", "loss_dino", "loss_kl",
                        "loss_dino_l1", "loss_dino_mse", "loss_dino_cosine",
                        "loss_dino_seg", "loss_dino_l1_seg", "loss_dino_mse_seg",
                        "loss_dino_cosine_seg", "loss_seg_pixel"):
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
        """Standard eval + action recon (MSE/L1) and DINO recon (L1/cosine) metrics."""
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model
        model.eval()

        total_mse = total_l1 = total_dino_l1 = total_dino_cos = 0.0
        total_seg_l1 = total_seg_cos = 0.0
        total_pix_bce = total_pix_iou = 0.0
        n_pix_samples = 0
        n_samples = 0
        has_seg_decoder = getattr(model, "seg_dino_decoder", None) is not None
        has_pix_decoder = (
            getattr(model, "seg_pixel_decoder", None) is not None
            or getattr(model, "_unified_segpix", False)
        )
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = self._prepare_inputs(batch)
                x0_feat, x1_feat = self._extract_feats(batch)
                s0_feat, s1_feat = self._extract_seg_feats(batch)
                actions = batch["action"].to(dtype=model.encoder.action_proj.weight.dtype)

                g, t, h = model.encode(actions, x0_feat, x1_feat, s0_feat, s1_feat)
                ud = getattr(model, "unified_decoder", None)
                if ud is not None and ud.recon_sees_vision:
                    preds = model.decode(g, t, h, x0_feat=x0_feat)
                else:
                    preds = model.decode(g, t, h)
                pred_x1 = model.decode_dino(t, x0_feat)

                B = actions.shape[0]
                total_mse += F.mse_loss(preds, actions).item() * B
                total_l1 += F.l1_loss(preds, actions).item() * B
                total_dino_l1 += F.l1_loss(pred_x1, x1_feat.to(dtype=pred_x1.dtype)).item() * B
                total_dino_cos += (
                    1.0 - F.cosine_similarity(pred_x1, x1_feat.to(dtype=pred_x1.dtype), dim=-1).mean()
                ).item() * B
                if has_seg_decoder:
                    pred_s1 = model.decode_dino_seg(t, s0_feat, x0_feat)
                    tgt_s1 = s1_feat.to(dtype=pred_s1.dtype)
                    total_seg_l1 += F.l1_loss(pred_s1, tgt_s1).item() * B
                    total_seg_cos += (
                        1.0 - F.cosine_similarity(pred_s1, tgt_s1, dim=-1).mean()
                    ).item() * B
                if has_pix_decoder and "mask_x1" in batch:
                    # Per-sample BCE + IoU@0.5, valid-mask-weighted (mask_valid=0
                    # samples excluded, mirroring the training loss).
                    logits = model.decode_seg_pixel(t, x0_feat)
                    tgt = batch["mask_x1"].to(device=logits.device).float()
                    validf = batch["mask_valid"].to(device=logits.device).float()
                    bce = F.binary_cross_entropy_with_logits(
                        logits.float(), tgt, reduction="none"
                    ).mean(dim=(1, 2))
                    pred_bin = (logits > 0).float()
                    inter = (pred_bin * tgt).sum(dim=(1, 2))
                    union = ((pred_bin + tgt) > 0).float().sum(dim=(1, 2))
                    iou = inter / union.clamp(min=1.0)
                    total_pix_bce += (bce * validf).sum().item()
                    total_pix_iou += (iou * validf).sum().item()
                    n_pix_samples += validf.sum().item()
                n_samples += B

        if n_samples > 0:
            extra = {
                f"{metric_key_prefix}_recon_mse": total_mse / n_samples,
                f"{metric_key_prefix}_recon_l1": total_l1 / n_samples,
                f"{metric_key_prefix}_dino_l1": total_dino_l1 / n_samples,
                f"{metric_key_prefix}_dino_cos_dist": total_dino_cos / n_samples,
            }
            if has_seg_decoder:
                extra[f"{metric_key_prefix}_dino_seg_l1"] = total_seg_l1 / n_samples
                extra[f"{metric_key_prefix}_dino_seg_cos_dist"] = total_seg_cos / n_samples
            if has_pix_decoder and n_pix_samples > 0:
                extra[f"{metric_key_prefix}_seg_pixel_bce"] = total_pix_bce / n_pix_samples
                extra[f"{metric_key_prefix}_seg_pixel_iou"] = total_pix_iou / n_pix_samples
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
    # VGGT final LayerNorm: "none" (default, raw token features) or "naive" (apply
    # an extra non-affine LayerNorm to the final VGGT tokens). vggt source only.
    vggt_final_norm: Literal["none", "naive"] = "none"
    # DINO final LayerNorm: "affine" (default, standard last_hidden_state) or
    # "naive" (drop the final LN's learned γ/β, normalize only). dino source only.
    dino_final_norm: Literal["affine", "naive"] = "affine"

    # ── DINO decoder ──
    dino_decoder_depth: int = 12

    # ── Unified decoder (opt-in; default "separate" = byte-identical) ──
    # "separate" (default): the existing ReconDecoderV4 + dino_decoder pair —
    #   nothing changes (no new modules/buffers/paths).
    # "shared_trunk": ONE decoder over the combined [latent, x0-patch] sequence —
    #   decoder_trunk_depth masked shared layers, then a decoder_branch_depth-layer
    #   recon branch (latent only → action head) and dino branch (full sequence →
    #   future-feature head). Latent rows attend only latent columns, so decode()
    #   still works from the latent alone.
    # "mot": mot_depth Mixture-of-Transformers layers — shared attention (same
    #   asymmetric mask), per-group (latent/patch) FFN + norms, no branches.
    # "shared_trunk_vis" (EXP-0011): shared_trunk WITHOUT the recon branch — the
    #   trunk serves the visual branches only (dino [+ segpix]) and the action is
    #   decoded by the ordinary ReconDecoderV4 (--decoder-depth / --emb-dim), i.e.
    #   the action path never touches the trunk. Everything else (trunk/branch
    #   depths, asymmetric mask, heads) matches "shared_trunk" exactly.
    decoder_arch: Literal["separate", "shared_trunk", "mot", "shared_trunk_vis"] = "separate"
    decoder_trunk_depth: int = 4
    decoder_branch_depth: int = 2
    mot_depth: int = 6
    # Exploration only: lift the mask so latent rows also attend the patches.
    # decode() then REQUIRES x0_feat (the checkpoint records a marker).
    decoder_recon_sees_vision: bool = False

    # ── Segment (SAM3 cutout) DINO stream ──
    # All default-off → byte-identical to the pre-seg behavior (no extra data loading,
    # no extra params/buffers, no extra losses).
    #
    # use_seg_stream=True: the dataset additionally reads the cutout video's frames at
    #   the SAME two steps as (frame_x0, frame_x1) from
    #   <seg_dataset_root>/<dataset_dir_name>/<seg_video_subdir>/chunk-XXX/<video_key>/
    #   episode_XXXXXX.mp4 (same fps/resolution/frame count as the source video, so the
    #   same timestamp lookup lands on the same step). The SAME frozen extractor embeds
    #   them, and their feature difference (s1 - s0) is concatenated side-by-side with
    #   the RGB difference along the token axis before the fusion transformer.
    # use_seg_dino_decoder=True (requires use_seg_stream): adds a twin of the DINO
    #   decoder that predicts the CUTOUT stream's future features from its own current
    #   features + the latent, weighted by lambda_dino_seg.
    use_seg_stream: bool = False
    seg_dataset_root: Optional[str] = None
    seg_video_subdir: str = "cutout"
    use_seg_dino_decoder: bool = False
    # Depth of the segment DINO decoder; None → same as dino_decoder_depth.
    seg_dino_decoder_depth: Optional[int] = None
    # What the segment DINO decoder conditions on (only when use_seg_dino_decoder):
    #   "seg" (default, unchanged): the cutout stream's own current features s0_feat.
    #   "rgb": the RAW image current features x0_feat — i.e. BOTH decoders get the same
    #     visual input and differ only in their target (dino_decoder → x1_feat,
    #     seg_dino_decoder → s1_feat). Same shapes ⇒ no parameter/architecture change.
    # The encoder-side seg stream (feature-difference concat) is unaffected either way.
    seg_dino_decoder_input: Literal["seg", "rgb"] = "seg"

    # ── SAM3 mask npz stream (shared by the two mask features below) ──
    # Root of the SAM3 mask mirror; the dataset reads the x1-step union mask from
    # <mask_dataset_root>/<dataset_dir_name>/<mask_subdir>/chunk-XXX/<video_key>/
    # episode_XXXXXX.npz (key 'mask': (T,H,W) uint8 {0,1}, frame t == parquet step t)
    # and attaches mask_x1/mask_valid to each sample. Only read when one of the two
    # features below is on; episodes with no npz are tolerated (mask_valid=0).
    mask_dataset_root: Optional[str] = None
    mask_subdir: str = "masks"

    # ── Mask-weighted DINO loss (EXP-0002; default off = byte-identical) ──
    # use_mask_weighted_dino_loss=True: per-patch weights on the (RGB) DINO
    #   future-feature loss. The x1-step mask is avg-pooled to the DINO patch grid
    #   (coverage c) → w = 1 + (mask_patch_weight − 1)·c → normalized by the batch
    #   mean (total loss scale invariant) → multiplies the per-patch loss. Missing
    #   npz → w = 1. Loss-only: adds NO parameters/buffers, so Stage-2 loading is
    #   unaffected either way.
    use_mask_weighted_dino_loss: bool = False
    mask_patch_weight: float = 2.0

    # ── Seg-mask pixel decoder (EXP-0003; default off = byte-identical) ──
    # use_seg_pixel_decoder=True: adds a twin of the DINO decoder that predicts the
    #   FUTURE (x1-step) union mask at pixel resolution from x0_feat + the latent,
    #   through a linear per-patch pixel-block head (patch² logits per token, tiled
    #   to [grid·patch, grid·patch]). BCE loss weighted by lambda_seg_pixel; samples
    #   with mask_valid=0 are excluded. Training-only module — the Stage-2 wrapper
    #   builds it as None and filters its checkpoint keys generically.
    use_seg_pixel_decoder: bool = False
    # BCE weight. Default 0.1 mirrors lambda_dino in the v4 gr1 recipes (both are
    # auxiliary future-prediction losses; BCE starts at ln2≈0.69, the same order as
    # the dino terms, so 0.1 keeps the auxiliary term ~an order below recon).
    lambda_seg_pixel: float = 0.1
    # Depth of the seg-pixel decoder; None → same as dino_decoder_depth.
    seg_pixel_decoder_depth: Optional[int] = None
    # Pixel-block side per patch token (dinov2-large: patch 14 → 16×16 grid @ 224).
    # image_size must equal (image_size // seg_pixel_patch) · seg_pixel_patch.
    seg_pixel_patch: int = 14

    # ── Loss ──
    lambda_recon: float = 1.0
    lambda_dino: float = 1.0
    # Weight of the segment-stream DINO recon loss (only when use_seg_dino_decoder).
    lambda_dino_seg: float = 1.0
    recon_loss_type: Literal["mse", "l1"] = "mse"
    dino_loss_type: str = "l1+mse"  # RLA default (L1 + MSE). Also: "cosine", "l1+cosine" ...
    dino_w_l1: float = 1.0
    dino_w_mse: float = 1.0
    dino_w_cosine: float = 1.0

    # ── VAE bottleneck (SD-style, opt-in) ──
    # use_vae=False (default): deterministic V4 — byte-identical to before (no extra
    #   params/buffers, no sampling).
    # use_vae=True: the fusion output is the posterior mean μ; a logvar head is added
    #   and ``encode`` returns a reparameterized sample z (so the VLA target is z,
    #   matching SD latent-diffusion). lambda_kl weights KL(N(0,I)); SD uses a tiny
    #   KL (~1e-6). kl_free_bits floors per-dim KL (0 = off).
    use_vae: bool = False
    # Sampling toggle for the VAE bottleneck (only meaningful with --use-vae).
    # True (default) → reparameterize z = μ + σ·ε (existing behavior). Pass
    # --no-vae-sample to keep the full VAE (logvar head + KL) but return the
    # posterior mean μ instead of a sample. The choice is baked into the checkpoint
    # (a _vae_no_sample marker) so Stage-2 latent targets / inference inherit it.
    vae_sample: bool = True
    # KL(N(0,I)) weight (SD regime ~1e-6); ignored when use_vae=False.
    lambda_kl: float = 1e-6
    # Per-dim KL free-bits floor (0 = off); ignored when use_vae=False.
    kl_free_bits: float = 0.0

    # ── Action-token projection (fusion) ──
    # The action latents are projected emb_dim → fusion_width right before the DINO
    # feats are concatenated in the fusion transformer. Default (False) uses a single
    # Linear (byte-identical to before). action_proj_mlp=True swaps it for a 2-layer
    # MLP (Linear → GELU → Linear); action_proj_hidden sets the hidden width (None →
    # defaults to fusion_width). The structure is shape-detectable at reload, so no
    # checkpoint marker is needed and off-path checkpoints are unaffected.
    action_proj_mlp: bool = False
    action_proj_hidden: Optional[int] = None

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

    # ── Precomputed DINO cache ──
    # If True, read precomputed DINO feats from
    # <dataset>/dino_feature_cache/<key> (built by
    # scripts/precompute_dino_features.py) instead of decoding video + running
    # DINO on the fly. The cache <key> is derived from feature_source / dino_model
    # / dino_final_norm / image_size / camera, so it MUST already exist for this
    # exact config. Default False → unchanged on-the-fly behavior.
    use_dino_cache: bool = False

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
        use_vae=config.use_vae,
        vae_sample=config.vae_sample,
        kl_free_bits=config.kl_free_bits,
        action_proj_mlp=config.action_proj_mlp,
        action_proj_hidden=config.action_proj_hidden,
        use_seg_stream=config.use_seg_stream,
    )

    # ---- unified decoder (opt-in; early-return keeps the separate path below
    # byte-identical) ----
    if config.decoder_arch != "separate":
        assert not config.use_seg_dino_decoder, (
            "--decoder-arch shared_trunk/mot does not support --use-seg-dino-decoder "
            "(separate-arch-only feature)."
        )
        if config.use_seg_pixel_decoder and config.decoder_arch not in (
            "shared_trunk", "shared_trunk_vis"
        ):
            raise ValueError(
                "--use-seg-pixel-decoder with --decoder-arch mot is not supported "
                "(the segpix branch exists only for shared_trunk/shared_trunk_vis)."
            )
        visual_only = config.decoder_arch == "shared_trunk_vis"
        unified_decoder = UnifiedDecoderV4(
            action_dim=action_dim,
            action_horizon=action_horizon,
            token_dim=config.token_dim,
            dino_dim=config.dino_channels,
            width=config.fusion_width,
            head_dim=config.head_dim,
            arch=config.decoder_arch,
            trunk_depth=config.decoder_trunk_depth,
            branch_depth=config.decoder_branch_depth,
            mot_depth=config.mot_depth,
            pdropout=config.pdropout,
            num_global_tokens=0,
            num_hand_tokens=0,
            recon_sees_vision=config.decoder_recon_sees_vision,
            use_segpix_branch=config.use_seg_pixel_decoder,
        )
        # shared_trunk_vis: the action path stays SEPARATE — build the ordinary
        # ReconDecoderV4 (same construction as the separate path below, i.e. base's
        # emb_dim/decoder_depth decoder) and hand it to the tokenizer alongside the
        # visual-only trunk. Built after the unified decoder so the trunk's init
        # under a fixed seed does not depend on it.
        vis_recon_decoder = None
        if visual_only:
            vis_recon_decoder = ReconDecoderV4(
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

        # segpix branch head (built AFTER the decoder so shared-module init under a
        # fixed seed is identical with the branch on or off). Zero-init → initial
        # logits 0 → BCE starts at ln2, mirroring the separate seg_pixel_head.
        seg_pixel_head = None
        if config.use_seg_pixel_decoder:
            from gr00t.model.action_latent_tokenizer_v4 import LinearHead

            seg_pixel_head = LinearHead(
                config.fusion_width, config.seg_pixel_patch ** 2, weight_init_style="zero"
            )
        return ActionLatentTokenizerV4(
            encoder=encoder,
            recon_decoder=vis_recon_decoder,
            dino_decoder=None,
            unified_decoder=unified_decoder,
            decoder_arch=config.decoder_arch,
            seg_pixel_head=seg_pixel_head,
            seg_pixel_patch=config.seg_pixel_patch,
            lambda_seg_pixel=config.lambda_seg_pixel,
            use_mask_weighted_dino_loss=config.use_mask_weighted_dino_loss,
            mask_patch_weight=config.mask_patch_weight,
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

    # Segment-stream DINO decoder: same architecture/config as the RGB one, but trained
    # against the cutout stream's features. None when the flag is off, in which case
    # ActionLatentTokenizerV4 registers nothing extra.
    seg_dino_decoder = None
    if config.use_seg_dino_decoder:
        seg_dino_decoder = SimpleTokenTransformer(
            in_channels=config.dino_channels,
            model_channels=config.fusion_width,
            out_channels=config.dino_channels,
            num_blocks=int(config.seg_dino_decoder_depth or config.dino_decoder_depth),
            num_heads=config.fusion_heads,
            num_tokens=action_horizon,
            token_channels=config.token_dim,
            zero_init=True,
            use_fp16=False,
        )

    # Seg-mask pixel decoder (EXP-0003): same SimpleTokenTransformer mechanics as the
    # DINO decoder (visual context x0_feat + latent tokens), plus a zero-init linear
    # head mapping each patch token to its patch² pixel-block logits. None when the
    # flag is off → ActionLatentTokenizerV4 registers nothing extra.
    seg_pixel_decoder = None
    seg_pixel_head = None
    if config.use_seg_pixel_decoder:
        from gr00t.model.action_latent_tokenizer_v4 import LinearHead

        seg_pixel_decoder = SimpleTokenTransformer(
            in_channels=config.dino_channels,
            model_channels=config.fusion_width,
            out_channels=config.dino_channels,
            num_blocks=int(config.seg_pixel_decoder_depth or config.dino_decoder_depth),
            num_heads=config.fusion_heads,
            num_tokens=action_horizon,
            token_channels=config.token_dim,
            zero_init=True,
            use_fp16=False,
        )
        # Zero-init head → initial logits 0 → p=0.5 everywhere → BCE starts at ln2.
        seg_pixel_head = LinearHead(
            config.dino_channels, config.seg_pixel_patch ** 2, weight_init_style="zero"
        )

    return ActionLatentTokenizerV4(
        encoder=encoder,
        recon_decoder=recon_decoder,
        dino_decoder=dino_decoder,
        seg_dino_decoder=seg_dino_decoder,
        seg_dino_decoder_input=config.seg_dino_decoder_input,
        seg_pixel_decoder=seg_pixel_decoder,
        seg_pixel_head=seg_pixel_head,
        seg_pixel_patch=config.seg_pixel_patch,
        lambda_seg_pixel=config.lambda_seg_pixel,
        use_mask_weighted_dino_loss=config.use_mask_weighted_dino_loss,
        mask_patch_weight=config.mask_patch_weight,
        lambda_recon=config.lambda_recon,
        lambda_dino=config.lambda_dino,
        lambda_dino_seg=config.lambda_dino_seg,
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
    )


# =====================================================================
# Main
# =====================================================================


def main(config: ArgsConfig):
    # ---- segment-stream arg validation (all no-ops when the flags are off) ----
    # --use-seg-dino-decoder WITHOUT --use-seg-stream (EXP-0004): the seg decoder's
    # auxiliary loss is kept but the ENCODER never sees the cutout — the dataset
    # still loads seg frames and the trainer still extracts s0/s1 feats, which feed
    # ONLY decode_dino_seg (ctx s0, target s1). The encoder is built with
    # use_seg_stream=False, so its fusion input is byte-identical to the plain base
    # (model-side: encode() leaves seg_diff=None and the encoder asserts it).
    if config.use_seg_dino_decoder and not config.use_seg_stream:
        assert config.seg_dataset_root, (
            "--use-seg-dino-decoder without --use-seg-stream still needs "
            "--seg-dataset-root (the decoder's ctx/target come from the cutout mirror)."
        )
        assert os.path.isdir(config.seg_dataset_root), (
            f"--seg-dataset-root does not exist: {config.seg_dataset_root}"
        )
        assert not config.use_dino_cache, (
            "the seg decoder needs live cutout features; incompatible with --use-dino-cache."
        )
        print(
            f"[seg-decoder-only] ON  root={config.seg_dataset_root} "
            f"subdir={config.seg_video_subdir} "
            f"seg_dino_decoder_input={config.seg_dino_decoder_input} "
            f"lambda_dino_seg={config.lambda_dino_seg} (encoder gets NO seg stream)"
        )
    if config.seg_dino_decoder_input == "rgb":
        assert config.use_seg_dino_decoder, (
            "--seg-dino-decoder-input rgb only means anything with --use-seg-dino-decoder."
        )
    if config.use_seg_stream:
        assert config.seg_dataset_root, (
            "--use-seg-stream requires --seg-dataset-root (root of the SAM3 cutout "
            "mirror, e.g. .../GR00T-X-Embodiment-Sim_sam3_robot_task)."
        )
        assert os.path.isdir(config.seg_dataset_root), (
            f"--seg-dataset-root does not exist: {config.seg_dataset_root}"
        )
        assert not config.use_dino_cache, (
            "--use-seg-stream is incompatible with --use-dino-cache: the precomputed "
            "DINO cache holds only the RGB stream's features (no cutout features), so "
            "the seg stream must decode video live."
        )
        print(
            f"[seg-stream] ON  root={config.seg_dataset_root} "
            f"subdir={config.seg_video_subdir} "
            f"seg_dino_decoder={config.use_seg_dino_decoder} "
            f"seg_dino_decoder_input={config.seg_dino_decoder_input} "
            f"lambda_dino_seg={config.lambda_dino_seg}"
        )

    # ---- decoder-arch banner (silent for the default "separate") ----
    if config.decoder_arch != "separate":
        print(
            f"[decoder-arch] {config.decoder_arch}  trunk_depth={config.decoder_trunk_depth} "
            f"branch_depth={config.decoder_branch_depth} mot_depth={config.mot_depth} "
            f"width={config.fusion_width} recon_sees_vision={config.decoder_recon_sees_vision} "
            f"segpix_branch={config.use_seg_pixel_decoder}"
        )
        if config.decoder_arch == "shared_trunk_vis":
            print(
                "[decoder-arch] VISUAL-ONLY trunk: no recon branch / action head — the "
                f"action path is the separate ReconDecoderV4 (emb_dim={config.emb_dim}, "
                f"depth={config.decoder_depth}, mode={config.decoder_mode}), so the trunk "
                "carries the visual tasks only (EXP-0011)."
            )

    # ---- mask-stream arg validation (all no-ops when the flags are off) ----
    use_masks = config.use_mask_weighted_dino_loss or config.use_seg_pixel_decoder
    if use_masks:
        assert config.mask_dataset_root, (
            "--use-mask-weighted-dino-loss / --use-seg-pixel-decoder require "
            "--mask-dataset-root (root of the SAM3 mask mirror, e.g. "
            ".../PhysicalAI-Robotics-GR00T-X-Embodiment-Sim_sam3_D_parts_nouns_norobot)."
        )
        assert os.path.isdir(config.mask_dataset_root), (
            f"--mask-dataset-root does not exist: {config.mask_dataset_root}"
        )
        assert not config.use_dino_cache, (
            "the mask features require the live-video dataset (ActionFramesDatasetV4); "
            "they are not implemented for --use-dino-cache."
        )
    if config.use_mask_weighted_dino_loss:
        assert config.mask_patch_weight > 0, (
            f"--mask-patch-weight must be > 0; got {config.mask_patch_weight}"
        )
        assert config.lambda_dino > 0, (
            "--use-mask-weighted-dino-loss weights the DINO loss, but --lambda-dino "
            "is 0 — the weighted loss would never be computed."
        )
        print(
            f"[mask-weight] ON  root={config.mask_dataset_root} "
            f"subdir={config.mask_subdir} W={config.mask_patch_weight}"
        )
    if config.use_seg_pixel_decoder and config.decoder_arch == "mot":
        raise ValueError(
            "--use-seg-pixel-decoder with --decoder-arch mot is not supported "
            "(the segpix branch exists only for shared_trunk; separate keeps the "
            "original EXP-0003 module)."
        )
    if config.use_seg_pixel_decoder:
        assert config.lambda_seg_pixel > 0, (
            f"--lambda-seg-pixel must be > 0 with --use-seg-pixel-decoder; got "
            f"{config.lambda_seg_pixel}"
        )
        grid = config.image_size // config.seg_pixel_patch
        assert grid * config.seg_pixel_patch == config.image_size, (
            f"--image-size ({config.image_size}) must be a multiple of "
            f"--seg-pixel-patch ({config.seg_pixel_patch})."
        )
        print(
            f"[seg-pixel] ON  root={config.mask_dataset_root} "
            f"subdir={config.mask_subdir} lambda={config.lambda_seg_pixel} "
            f"depth={config.seg_pixel_decoder_depth or config.dino_decoder_depth} "
            f"patch={config.seg_pixel_patch} (grid {grid}x{grid} @ {config.image_size})"
        )

    def make_dataset(path, split):
        if config.use_dino_cache:
            return CachedActionFramesDatasetV4(
                dataset_path=path,
                data_config_name=config.data_config,
                embodiment_tag=config.embodiment_tag,
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
            seg_dataset_root=(
                config.seg_dataset_root
                if (config.use_seg_stream or config.use_seg_dino_decoder)
                else None
            ),
            seg_video_subdir=config.seg_video_subdir,
            mask_dataset_root=config.mask_dataset_root if use_masks else None,
            mask_subdir=config.mask_subdir,
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
    from gr00t.data.merge_norm_stats import (
        apply_merged_normalization_metadata,
        save_normalization_stats,
    )

    merged_metadata = apply_merged_normalization_metadata(
        datasets_train, datasets_train + datasets_val
    )

    # Persist the merged (whole-mixture) normalization statistics next to the
    # checkpoints so Stage-2 / inference can reuse the exact stats without
    # re-reading and re-merging every source dataset's meta/stats.json. For a
    # single dataset the merge is a no-op (merged_metadata is None), so fall
    # back to that dataset's own metadata (already the whole-dataset stats).
    if int(os.environ.get("RANK", 0)) == 0:
        stats_meta = merged_metadata or datasets_train[0].metadata
        stats_path = save_normalization_stats(
            stats_meta, os.path.join(config.output_dir, "norm_stats.json")
        )
        print(f"[merge-stats] wrote normalization stats -> {stats_path}")

    # ---- DEBUG: print normalization stats ACTUALLY applied at train time ----
    # Mirror of the Stage-2 VLA's [norm] logging (gr00t_finetune_actlat_fm.py) so
    # Stage-1/Stage-2 normalization consistency can be checked by eye. APPLIED =
    # the live Normalizers on the dataset's shared transform (ground truth of
    # what gets applied at __getitem__ time); MERGED = the merged whole-mixture
    # stats computed above (None for a single dataset, whose own stats already
    # equal the mixture stats — nothing to merge).
    if int(os.environ.get("RANK", 0)) == 0:
        live_tf = datasets_train[0].transforms

        def _aslist(v):
            return v.tolist() if hasattr(v, "tolist") else list(v)

        def _fmt(st):
            def g(k):
                # Rotation keys carry only min/max overrides → q01/q99 may be
                # absent; print NA instead of crashing.
                return _aslist(st[k]) if k in st else "NA"
            return f"min={g('min')} max={g('max')} q01={g('q01')} q99={g('q99')}"

        sa_tr = next(
            (t for t in getattr(live_tf, "transforms", []) if hasattr(t, "_normalizers")),
            None,
        )
        if sa_tr is None:
            print("[norm] WARNING: no StateActionTransform with _normalizers found")
        else:
            for key, normd in sa_tr._normalizers.items():
                print(f"[norm] APPLIED {key} mode={normd.mode} {_fmt(normd.statistics)}")

        if merged_metadata is not None:
            for subkey, v in merged_metadata.statistics.action.items():
                print(
                    f"[norm] MERGED action.{subkey} "
                    f"min={_aslist(v.min)} max={_aslist(v.max)} "
                    f"q01={_aslist(v.q01)} q99={_aslist(v.q99)}"
                )
    # ----------------------------------------------------------------------

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

    collator = CachedActionFramesCollatorV4() if config.use_dino_cache else ActionFramesCollatorV4()

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
        vggt_final_norm=config.vggt_final_norm,
        dino_final_norm=config.dino_final_norm,
        use_dino_cache=config.use_dino_cache,
        use_seg_stream=config.use_seg_stream,
        seg_feats_only=(config.use_seg_dino_decoder and not config.use_seg_stream),
        pass_masks=use_masks,
    )

    if config.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = config.wandb_project
        if os.environ.get("GR00T_S3_COMPAT") == "1":
            # gpu26/AWS: output_dir is an S3 mount that rejects wandb's append
            # writes — respect the WANDB_DIR the sbatch script pre-set (home).
            os.environ.setdefault("WANDB_DIR", config.output_dir)
        else:
            os.environ["WANDB_DIR"] = config.output_dir

    # Resolve --resume gracefully: only resume if a checkpoint actually exists.
    # transformers.Trainer raises ValueError when resume_from_checkpoint=True but
    # output_dir has no checkpoint (e.g. the very first run), so fall back to
    # training from scratch in that case.
    resume_from_checkpoint = False
    if config.resume:
        from transformers.trainer_utils import get_last_checkpoint

        last_ckpt = None
        if os.path.isdir(config.output_dir):
            last_ckpt = get_last_checkpoint(config.output_dir)
        if last_ckpt is not None:
            print(f"[resume] Resuming from checkpoint: {last_ckpt}")
            resume_from_checkpoint = last_ckpt
        else:
            print(
                f"[resume] --resume set but no checkpoint found in "
                f"'{config.output_dir}'; starting training from scratch."
            )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

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
                    "use_seg_stream": config.use_seg_stream,
                    "seg_dataset_root": config.seg_dataset_root,
                    "seg_video_subdir": config.seg_video_subdir,
                    "use_seg_dino_decoder": config.use_seg_dino_decoder,
                    "seg_dino_decoder_depth": config.seg_dino_decoder_depth,
                    "seg_dino_decoder_input": config.seg_dino_decoder_input,
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
                    "action_proj_mlp": config.action_proj_mlp,
                    "action_proj_hidden": config.action_proj_hidden,
                    "recon_loss_type": config.recon_loss_type,
                    "dino_loss_type": config.dino_loss_type,
                    "image_size": config.image_size,
                    "use_fixed_val": config.use_fixed_val,
                    "fixed_val_path": config.fixed_val_path,
                    "val_ratio": config.val_ratio,
                    "val_seed": config.val_seed,
                    "data_config": config.data_config,
                    # Unified-decoder keys are recorded ONLY when the arch is not
                    # "separate", so default final .pt files stay byte-identical.
                    **(
                        {
                            "decoder_arch": config.decoder_arch,
                            "decoder_trunk_depth": config.decoder_trunk_depth,
                            "decoder_branch_depth": config.decoder_branch_depth,
                            "mot_depth": config.mot_depth,
                            "decoder_recon_sees_vision": config.decoder_recon_sees_vision,
                        }
                        if config.decoder_arch != "separate"
                        else {}
                    ),
                    # Mask-feature keys are recorded ONLY when the flags are on, so
                    # flag-off final .pt files stay byte-identical to before.
                    **(
                        {
                            "use_mask_weighted_dino_loss": True,
                            "mask_patch_weight": config.mask_patch_weight,
                            "mask_dataset_root": config.mask_dataset_root,
                            "mask_subdir": config.mask_subdir,
                        }
                        if config.use_mask_weighted_dino_loss
                        else {}
                    ),
                    **(
                        {
                            "use_seg_pixel_decoder": True,
                            "lambda_seg_pixel": config.lambda_seg_pixel,
                            "seg_pixel_decoder_depth": config.seg_pixel_decoder_depth,
                            "seg_pixel_patch": config.seg_pixel_patch,
                            "mask_dataset_root": config.mask_dataset_root,
                            "mask_subdir": config.mask_subdir,
                        }
                        if config.use_seg_pixel_decoder
                        else {}
                    ),
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

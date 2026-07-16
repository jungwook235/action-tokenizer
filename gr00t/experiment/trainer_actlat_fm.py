"""Trainer for action latent flow matching VLA.

Extends DualBrainTrainer with:
- Logging of actlat_fm_loss (MSE) and actlat_fm_l1
- Custom evaluation with full denoising + tokenizer decode
"""

import math

import torch
import torch.nn.functional as F

from gr00t.experiment.trainer import BaseSampler, DualBrainTrainer


class ActlatFMTrainer(DualBrainTrainer):

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def get_eval_dataloader(self, eval_dataset=None):
        if not hasattr(self, "_cached_eval_dataloader"):
            self._cached_eval_dataloader = super().get_eval_dataloader(eval_dataset)
        return self._cached_eval_dataloader

    def log(self, logs: dict[str, float], start_time=None) -> None:
        if hasattr(self.model, "module"):
            action_head = self.model.module.action_head
        else:
            action_head = self.model.action_head

        if hasattr(action_head, "actlat_fm_loss"):
            val = action_head.actlat_fm_loss
            logs["loss/actlat_fm_mse"] = val.item() if torch.is_tensor(val) else val

        if hasattr(action_head, "actlat_fm_l1"):
            val = action_head.actlat_fm_l1
            logs["loss/actlat_fm_l1"] = val.item() if torch.is_tensor(val) else val

        super().log(logs, start_time)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Custom evaluation: prepare_input → backbone → action_head.get_action (full denoising).

        Common metrics (both baseline VLA and actlat_fm):
        - eval/action_mse: MSE between predicted action and real action
        - eval/action_l1:  L1  between predicted action and real action

        Additional metrics for actlat_fm (when action_latent_tokenizer is set):
        - eval/latent_mse: MSE between predicted latent and target latent
        - eval/latent_l1:  L1  between predicted latent and target latent
        """
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            return {}

        model = self.model
        model.eval()

        unwrapped = model.module if hasattr(model, "module") else model
        tokenizer = getattr(unwrapped, "action_latent_tokenizer", None)
        is_actlat = tokenizer is not None

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # ── Fixed evaluation budget ────────────────────────────────────────
        # Evaluate a CONSTANT number of samples per eval, independent of GPU
        # count and (per-device) batch size. The old code capped by *batches*
        # (50), so the sample count scaled with batch size, and accelerate
        # shards the eval dataloader across ranks without any cross-rank
        # reduction — both made the effective sample count depend on the
        # training config. Here we instead:
        #   1. cap by SAMPLES (not batches),
        #   2. give each rank an equal share of the global budget, and
        #   3. pool element-wise error SUMS + COUNTS across ranks at the end,
        # so the logged metric is computed over exactly MAX_EVAL_SAMPLES
        # samples total (or the whole val set if it is smaller).
        MAX_EVAL_SAMPLES = 3200
        world_size = max(1, getattr(self.accelerator, "num_processes", 1))
        per_rank_samples = math.ceil(MAX_EVAL_SAMPLES / world_size)

        # Element-wise sums of squared / absolute errors and element counts.
        # Pooling sums+counts (rather than averaging per-batch means) makes the
        # cross-rank aggregate exact even when the final batch is trimmed.
        action_se = 0.0
        action_ae = 0.0
        action_elems = 0
        latent_se = 0.0
        latent_ae = 0.0
        latent_elems = 0
        n_samples = 0  # samples processed on THIS rank

        with torch.no_grad():
            for inputs in eval_dataloader:
                if n_samples >= per_rank_samples:
                    break

                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(unwrapped.device)

                real_actions = inputs["action"].float()  # [B, T, D]
                # Trim the final batch so this rank stops at exactly its share:
                # keeps the total sample count independent of batch size.
                take = min(real_actions.shape[0], per_rank_samples - n_samples)

                # Single-observation input for inference (no future index)
                infer_input = {
                    "state": inputs["state"][:, 0:1, :],
                    "state_mask": inputs["state_mask"][:, 0:1, :],
                    "eagle_input_ids": inputs["eagle_input_ids"],
                    "eagle_attention_mask": inputs["eagle_attention_mask"],
                    "eagle_pixel_values": inputs["eagle_pixel_values"],
                    "eagle_image_sizes": inputs["eagle_image_sizes"],
                    "embodiment_id": inputs["embodiment_id"],
                }

                # prepare_input → VLM backbone → action_head denoising
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    backbone_inputs, action_inputs = unwrapped.prepare_input(infer_input)
                    backbone_outputs = unwrapped.backbone(backbone_inputs)
                    action_head_outputs = unwrapped.action_head.get_action(
                        backbone_outputs, action_inputs
                    )

                # action_head.get_action output is action_pred
                # - baseline VLA: directly predicted actions [B, T, D]
                # - actlat_fm:    predicted latent tokens [B, N, latent_dim]
                raw_pred = action_head_outputs["action_pred"].float()

                if is_actlat:
                    # Latent metrics
                    target_tokens = unwrapped.actlat_target_tokens
                    # V4 (RLA-DINO) tokenizer needs chunk start/end frames; pass
                    # them when present (None-safe for v2/v3 tokenizers).
                    latent_target = tokenizer.get_latent_target(
                        real_actions.to(device=unwrapped.device),
                        target_tokens=target_tokens,
                        x0=inputs.get("frame_x0"),
                        x1=inputs.get("frame_x1"),
                    )
                    latent_target_dev = latent_target.to(
                        device=raw_pred.device, dtype=raw_pred.dtype
                    )
                    pred_l = raw_pred[:take]
                    tgt_l = latent_target_dev[:take]
                    latent_se += F.mse_loss(pred_l, tgt_l, reduction="sum").item()
                    latent_ae += F.l1_loss(pred_l, tgt_l, reduction="sum").item()
                    latent_elems += pred_l.numel()

                    # Decode latent → action for action-space comparison
                    predicted_action = tokenizer.decode_latent(
                        raw_pred, target_tokens=target_tokens
                    ).float()
                else:
                    # Baseline VLA: raw_pred is already the action
                    predicted_action = raw_pred

                real_actions_dev = real_actions.to(device=predicted_action.device)
                pred_a = predicted_action[:take]
                real_a = real_actions_dev[:take]
                action_se += F.mse_loss(pred_a, real_a, reduction="sum").item()
                action_ae += F.l1_loss(pred_a, real_a, reduction="sum").item()
                action_elems += pred_a.numel()

                n_samples += take

        # Pool sums + counts across ranks so the metric covers the whole budget
        # (a single collective after the loop → no deadlock on uneven shards).
        if (
            world_size > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            agg = torch.tensor(
                [
                    action_se,
                    action_ae,
                    float(action_elems),
                    latent_se,
                    latent_ae,
                    float(latent_elems),
                ],
                device=unwrapped.device,
                dtype=torch.float64,
            )
            torch.distributed.all_reduce(agg, op=torch.distributed.ReduceOp.SUM)
            (
                action_se,
                action_ae,
                action_elems,
                latent_se,
                latent_ae,
                latent_elems,
            ) = agg.tolist()

        if action_elems == 0:
            model.train()
            return {}

        metrics = {
            f"{metric_key_prefix}/action_mse": action_se / action_elems,
            f"{metric_key_prefix}/action_l1": action_ae / action_elems,
        }
        if is_actlat and latent_elems > 0:
            metrics[f"{metric_key_prefix}/latent_mse"] = latent_se / latent_elems
            metrics[f"{metric_key_prefix}/latent_l1"] = latent_ae / latent_elems

        self.log(metrics)
        model.train()
        return metrics

"""Trainer for action latent flow matching VLA.

Extends DualBrainTrainer with:
- Logging of actlat_fm_loss (MSE) and actlat_fm_l1
- Custom evaluation with full denoising + tokenizer decode
"""

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

        action_mse_sum = 0.0
        action_l1_sum = 0.0
        latent_mse_sum = 0.0
        latent_l1_sum = 0.0
        n_batches = 0
        max_eval_batches = 50

        with torch.no_grad():
            for step, inputs in enumerate(eval_dataloader):
                if step >= max_eval_batches:
                    break

                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(unwrapped.device)

                real_actions = inputs["action"].float()  # [B, T, D]

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
                    latent_target = tokenizer.get_latent_target(
                        real_actions.to(device=unwrapped.device),
                        target_tokens=target_tokens,
                    )
                    latent_target_dev = latent_target.to(
                        device=raw_pred.device, dtype=raw_pred.dtype
                    )
                    latent_mse_sum += F.mse_loss(raw_pred, latent_target_dev).item()
                    latent_l1_sum += F.l1_loss(raw_pred, latent_target_dev).item()

                    # Decode latent → action for action-space comparison
                    predicted_action = tokenizer.decode_latent(
                        raw_pred, target_tokens=target_tokens
                    ).float()
                else:
                    # Baseline VLA: raw_pred is already the action
                    predicted_action = raw_pred

                real_actions_dev = real_actions.to(device=predicted_action.device)
                action_mse_sum += F.mse_loss(predicted_action, real_actions_dev).item()
                action_l1_sum += F.l1_loss(predicted_action, real_actions_dev).item()

                n_batches += 1

        if n_batches == 0:
            return {}

        metrics = {
            f"{metric_key_prefix}/action_mse": action_mse_sum / n_batches,
            f"{metric_key_prefix}/action_l1": action_l1_sum / n_batches,
        }
        if is_actlat:
            metrics[f"{metric_key_prefix}/latent_mse"] = latent_mse_sum / n_batches
            metrics[f"{metric_key_prefix}/latent_l1"] = latent_l1_sum / n_batches

        self.log(metrics)
        model.train()
        return metrics

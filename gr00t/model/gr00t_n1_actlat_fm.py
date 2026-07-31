# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GR00T N1.5 model variant for action latent flow matching.

Flow matching operates in latent space: action_dim = latent_dim, action_horizon = num_tokens.
Uses a frozen ActionLatentTokenizerWrapper to encode actions to latent targets during training
and decode predicted latents back to actions during inference.
"""

import json
from dataclasses import dataclass, field
from typing import Tuple
import time
import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head_actlat_fm import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper
from .backbone import EagleBackbone

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


@dataclass
class GR00T_N1_5_Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})
    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})
    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})
    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class GR00T_N1_5(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_Config

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str,
        action_head_update: dict = None,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        self.backbone = EagleBackbone(**config.backbone_cfg)
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)

        if action_head_update is not None:
            for key, value in action_head_update.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        tmp_copy = getattr(action_head_cfg, key)
                        tmp_copy[sub_key] = sub_value
                        setattr(action_head_cfg, key, tmp_copy)
                else:
                    setattr(action_head_cfg, key, value)

        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

        # Will be set in from_pretrained / from_same_trained
        self.action_latent_tokenizer = None
        self.actlat_target_tokens = "all"
        # Store original action dimensions for decode validation
        self.original_action_dim = None
        self.original_action_horizon = None
        # Optional per-dim latent z-normalization (actlat_latent_norm — port of the
        # WAM DiT4DiT norm variant). None = OFF, byte-identical to before. Enabled
        # via setup_latent_norm(); stored as plain fp32 tensors (NOT params/buffers)
        # so the state_dict and any later dtype cast never touch them.
        self._actlat_latent_mean = None
        self._actlat_latent_std = None

    def validate_inputs(self, inputs):
        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            # During training, action is original [B, T, D], not latent yet
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True
        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )
        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            raise ValueError(error_msg)

        if is_training:
            if LOSS_KEY not in action_head_outputs:
                raise ValueError(f"Missing '{LOSS_KEY}' in action_head_outputs during training")
        else:
            if ACTION_KEY not in action_head_outputs:
                raise ValueError(f"Missing '{ACTION_KEY}' in action_head_outputs during inference")

    def forward(self, inputs: dict) -> BatchFeature:
        # Single observation — no dual input / future processing
        input_1 = {}
        input_1["state"] = inputs["state"][:, 0:1, :]
        input_1["state_mask"] = inputs["state_mask"][:, 0:1, :]
        input_1["segmentation_target"] = inputs["segmentation_target"]
        input_1["segmentation_target_mask"] = inputs["segmentation_target_mask"]
        input_1["has_real_action"] = inputs["has_real_action"]
        input_1["eagle_input_ids"] = inputs["eagle_input_ids"]
        input_1["eagle_attention_mask"] = inputs["eagle_attention_mask"]
        input_1["eagle_pixel_values"] = inputs["eagle_pixel_values"]
        input_1["eagle_image_sizes"] = inputs["eagle_image_sizes"]
        input_1["embodiment_id"] = inputs["embodiment_id"]

        # Encode actions → latent target
        actions = inputs["action"]  # [B, T, D] original actions, normalized to [-1, 1]
        # V4 (RLA-DINO) tokenizers produce DINO-dependent latents and need the
        # chunk start/end frames. The dataset (LeRobotSingleDatasetActlatFMV4)
        # supplies frame_x0/frame_x1; for v2/v3 tokenizers these are ignored, and
        # if absent .get(...) returns None (the wrapper raises only for v4).
        latent_target = self.action_latent_tokenizer.get_latent_target(
            actions.to(dtype=torch.float32),
            target_tokens=self.actlat_target_tokens,
            x0=inputs.get("frame_x0"),
            x1=inputs.get("frame_x1"),
            # Precomputed DINO feats (V4 cache). Absent unless the cached dataset
            # supplies them → None → the raw-frame path above is used unchanged.
            x0_feat=inputs.get("x0_feat"),
            x1_feat=inputs.get("x1_feat"),
            # Segment (SAM3 cutout) pair — present only when the dataset was built with
            # a seg root. Required by tokenizers trained with the seg DINO stream;
            # ignored (None) by every other tokenizer.
            s0=inputs.get("seg_x0"),
            s1=inputs.get("seg_x1"),
            s0_feat=inputs.get("s0_feat"),
            s1_feat=inputs.get("s1_feat"),
        )
        # Optional per-dim z-norm of the FM target (actlat_latent_norm). Applied in
        # fp32 BEFORE the dtype cast below; get_action inverts it before the decoder.
        if self._actlat_latent_mean is not None:
            latent_target = (
                latent_target.float() - self._actlat_latent_mean.to(latent_target.device)
            ) / self._actlat_latent_std.to(latent_target.device)
        # Set latent as the "action" for the action head
        input_1["action"] = latent_target.to(
            device=actions.device, dtype=actions.dtype
        )
        # All-ones mask for latent dimensions
        input_1["action_mask"] = torch.ones_like(input_1["action"])

        backbone_inputs, action_inputs = self.prepare_input(input_1)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)

        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(self, inputs: dict) -> BatchFeature:
        """Inference: predict latent → decode to real actions.

        Returns:
            action_pred: decoded real actions [B, T, D]
            latent_pred: raw predicted latent tokens [B, N, latent_dim]
        """
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        # get_action returns predicted latent [B, N, latent_dim]
        latent_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        predicted_latent = latent_outputs["action_pred"]

        # Decode latent → real actions [B, T, D]
        if self.action_latent_tokenizer is not None:
            latent_for_decode = predicted_latent.to(dtype=torch.float32)
            # actlat_latent_norm: the head was trained on z-normalized latents —
            # invert the normalization (z * std + mean) BEFORE the tokenizer decoder.
            if self._actlat_latent_mean is not None:
                latent_for_decode = (
                    latent_for_decode * self._actlat_latent_std.to(latent_for_decode.device)
                    + self._actlat_latent_mean.to(latent_for_decode.device)
                )
            decoded_actions = self.action_latent_tokenizer.decode_latent(
                latent_for_decode,
                target_tokens=self.actlat_target_tokens,
            )
            decoded_actions = decoded_actions.to(dtype=predicted_latent.dtype)
        else:
            # nactlat_baseline: no tokenizer, model predicts actions directly
            decoded_actions = predicted_latent

        return BatchFeature(data={"action_pred": decoded_actions, "latent_pred": predicted_latent})

    def get_action_with_time_check(self, inputs: dict) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        start_time = time.perf_counter()
        backbone_outputs = self.backbone(backbone_inputs)
        latent_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        predicted_latent = latent_outputs["action_pred"]

        if self.action_latent_tokenizer is not None:
            latent_for_decode = predicted_latent.to(dtype=torch.float32)
            # actlat_latent_norm: invert the z-normalization before the decoder.
            if self._actlat_latent_mean is not None:
                latent_for_decode = (
                    latent_for_decode * self._actlat_latent_std.to(latent_for_decode.device)
                    + self._actlat_latent_mean.to(latent_for_decode.device)
                )
            decoded_actions = self.action_latent_tokenizer.decode_latent(
                latent_for_decode,
                target_tokens=self.actlat_target_tokens,
            )
            decoded_actions = decoded_actions.to(dtype=predicted_latent.dtype)
        else:
            # nactlat_baseline: no tokenizer, model predicts actions directly
            decoded_actions = predicted_latent
        end_time = time.perf_counter()

        return BatchFeature(data={"action_pred": decoded_actions, "latent_pred": predicted_latent}), end_time - start_time

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    def setup_latent_norm(self, stats: dict = None, stats_path: str = ""):
        """Enable per-dim z-normalization of the tokenizer latent FM target.

        Port of the WAM DiT4DiT `actlat_latent_norm`: the latent target is
        z-normalized in forward() with PRECOMPUTED dataset-wide per-dim stats
        (mirrors VLA action normalization) and get_action() de-normalizes the
        predicted latent before the tokenizer decoder. Stats resolution order:
        embedded `stats` dict (checkpoint reload from config.json) > `stats_path`
        JSON (fresh training launch, written by the --actlat-dump-latent-stats-path
        pass). The stats are embedded into self.config so save_pretrained writes
        them into config.json — eval hosts reload without the stats file.
        """
        if self.action_latent_tokenizer is None:
            raise ValueError(
                "actlat_latent_norm requires a loaded actlat tokenizer "
                "(actlat_tokenizer_path) — there is no latent to normalize."
            )
        if not stats:
            if not stats_path:
                raise ValueError(
                    "actlat_latent_norm=True but neither embedded latent stats "
                    "(config.actlat_latent_stats) nor actlat_latent_stats_path is set."
                )
            with open(stats_path) as f:
                loaded = json.load(f)
            stats = {
                "mean": [float(v) for v in loaded["mean"]],
                "std": [float(v) for v in loaded["std"]],
            }
        latent_dim = int(self.action_latent_tokenizer.emb_dim)
        if len(stats["mean"]) != latent_dim or len(stats["std"]) != latent_dim:
            raise ValueError(
                f"actlat latent stats dim mismatch: mean/std have "
                f"{len(stats['mean'])}/{len(stats['std'])} dims but the tokenizer "
                f"latent dim is {latent_dim}."
            )
        mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 1, -1)
        std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 1, -1)
        tiny = (std < 1e-6).sum().item()
        if tiny:
            print(
                f"[ActlatFM] WARNING: latent stats have {tiny}/{latent_dim} std dims "
                f"< 1e-6 — clamping to 1e-6 to avoid divide-by-zero blowup."
            )
        self._actlat_latent_mean = mean
        self._actlat_latent_std = std.clamp_min(1e-6)
        # Persist flag + stats into the config for checkpoint portability.
        self.config.actlat_latent_norm = True
        self.config.actlat_latent_stats = {"mean": stats["mean"], "std": stats["std"]}
        print(
            f"[ActlatFM] latent z-norm ENABLED: dim={latent_dim}, "
            f"mean range=[{mean.min().item():.4f}, {mean.max().item():.4f}], "
            f"std range=[{self._actlat_latent_std.min().item():.4f}, "
            f"{self._actlat_latent_std.max().item():.4f}]"
        )

    @classmethod
    def _load_tokenizer(cls, actlat_tokenizer_path, actlat_target_tokens="all", embodiment_id=None,
                        vae_sample_override=None):
        """Load tokenizer and query dimensions.

        ``embodiment_id`` selects one embodiment when the checkpoint is a
        multi-embodiment joint V4 tokenizer; None for ordinary checkpoints.
        ``vae_sample_override`` (None/True/False) forces the VAE sampling behavior of
        the latent target regardless of the checkpoint marker; None = use the
        checkpoint's setting.
        """
        tokenizer = ActionLatentTokenizerWrapper.from_checkpoint(
            actlat_tokenizer_path, embodiment_id=embodiment_id,
            vae_sample_override=vae_sample_override,
        )
        latent_dim = tokenizer.emb_dim
        num_tokens = tokenizer.get_num_tokens(actlat_target_tokens)
        print(
            f"[ActlatFM] Tokenizer: latent_dim={latent_dim}, "
            f"num_tokens={num_tokens}, target={actlat_target_tokens}, "
            f"embodiment_id={embodiment_id}"
        )
        return tokenizer, latent_dim, num_tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, load_action_head: bool = True, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        # Pop flare params for compat (not used but may be in kwargs)
        vision_token_num = kwargs.pop("vision_token_num", 0)
        flare_loss_lambda = kwargs.pop("flare_loss_lambda", 0.0)
        flare_align_layers = kwargs.pop("flare_align_layers", 12)
        image_count = kwargs.pop("image_count", 1)
        resume = kwargs.pop("resume", False)
        video_only = kwargs.pop("video_only", False)
        flare_image_time = kwargs.pop("flare_image_time", "future")

        # Action latent tokenizer config
        actlat_tokenizer_path = kwargs.pop("actlat_tokenizer_path", None)
        actlat_target_tokens = kwargs.pop("actlat_target_tokens", "all")
        actlat_embodiment_id = kwargs.pop("actlat_embodiment_id", None)
        # Force deterministic-mu latent target regardless of the tokenizer's checkpoint
        # marker (None/False here → use the checkpoint setting).
        actlat_vae_no_sample = kwargs.pop("actlat_vae_no_sample", False)
        # Latent z-norm (see setup_latent_norm).
        actlat_latent_norm = kwargs.pop("actlat_latent_norm", False)
        actlat_latent_stats_path = kwargs.pop("actlat_latent_stats_path", "")

        # Resolve model path
        try:
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            print(f"Model not found in HF hub. Loading from local path: {pretrained_model_name_or_path}")
            local_model_path = pretrained_model_name_or_path

        # Load tokenizer to get latent dimensions
        tokenizer = None
        latent_dim = 256
        num_tokens = 16
        if actlat_tokenizer_path is not None:
            tokenizer, latent_dim, num_tokens = cls._load_tokenizer(
                actlat_tokenizer_path, actlat_target_tokens, embodiment_id=actlat_embodiment_id,
                vae_sample_override=(False if actlat_vae_no_sample else None),
            )

        # Build action head config updates
        update_action_head_cfg = {
            "num_target_vision_tokens": vision_token_num,
            "flare_loss_lambda": 0.0,
            "flare_align_layers": flare_align_layers,
            "image_count": image_count,
            "flare_image_time": flare_image_time,
        }
        key_mapping = {}
        if load_action_head and not resume and "nvidia/GR00T-N1.5-3B" in pretrained_model_name_or_path:
            key_mapping = {
                "action_head.future_tokens.weight": "action_head.future_tokens.weight_l",
            }

        pretrained_model, loading_info = super().from_pretrained(
            local_model_path,
            local_model_path=local_model_path,
            output_loading_info=True,
            action_head_update=update_action_head_cfg,
            key_mapping=key_mapping,
            **kwargs,
        )

        # Store original dimensions before overriding
        orig_action_dim = pretrained_model.config.action_dim
        orig_action_horizon = pretrained_model.config.action_horizon

        # Reinitialize action head with latent dimensions
        if tokenizer is not None:
            action_head_cfg = FlowmatchingActionHeadConfig(**pretrained_model.config.action_head_cfg)
            action_head_cfg.action_dim = latent_dim
            action_head_cfg.action_horizon = num_tokens
            action_head_cfg.num_target_vision_tokens = vision_token_num
            action_head_cfg.flare_loss_lambda = 0.0

            if load_action_head:
                # Save shape-safe weights before recreating action head
                dit_state = pretrained_model.action_head.model.state_dict()
                state_enc_state = pretrained_model.action_head.state_encoder.state_dict()
                vlln_state = pretrained_model.action_head.vlln.state_dict()
                vl_sa_state = pretrained_model.action_head.vl_self_attention.state_dict()
                future_tok_state = None
                if hasattr(pretrained_model.action_head, "future_tokens"):
                    future_tok_state = pretrained_model.action_head.future_tokens.state_dict()
                pos_state = None
                if hasattr(pretrained_model.action_head, "position_embedding"):
                    pos_state = pretrained_model.action_head.position_embedding.state_dict()

            pretrained_model.action_head = FlowmatchingActionHead(action_head_cfg)

            if load_action_head:
                # Restore shape-safe weights; action_encoder/action_decoder stay randomly initialized
                pretrained_model.action_head.model.load_state_dict(dit_state)
                pretrained_model.action_head.state_encoder.load_state_dict(state_enc_state)
                pretrained_model.action_head.vlln.load_state_dict(vlln_state)
                pretrained_model.action_head.vl_self_attention.load_state_dict(vl_sa_state)
                if future_tok_state is not None and hasattr(pretrained_model.action_head, "future_tokens"):
                    pretrained_model.action_head.future_tokens.load_state_dict(future_tok_state)
                if pos_state is not None and hasattr(pretrained_model.action_head, "position_embedding"):
                    pretrained_model.action_head.position_embedding.load_state_dict(pos_state)
                print(f"[ActlatFM] Reinitialized action head: action_dim={latent_dim}, action_horizon={num_tokens}")
                print(f"[ActlatFM] DiT + state_encoder + future_tokens preserved, action_encoder/decoder reinitialized")
            else:
                print(f"[ActlatFM] Initialized action head from scratch: action_dim={latent_dim}, action_horizon={num_tokens}")
                print(f"[ActlatFM] All action head weights randomly initialized")

            # Update model config
            pretrained_model.action_horizon = num_tokens
            pretrained_model.action_dim = latent_dim

            # Attach tokenizer
            pretrained_model.action_latent_tokenizer = tokenizer.to(pretrained_model.device)
            pretrained_model.actlat_target_tokens = actlat_target_tokens
            pretrained_model.original_action_dim = orig_action_dim
            pretrained_model.original_action_horizon = orig_action_horizon

            # Latent z-norm: explicit kwarg OR a checkpoint that was trained with it
            # (its config.json carries the flag + the embedded stats).
            if actlat_latent_norm or bool(
                getattr(pretrained_model.config, "actlat_latent_norm", False)
            ):
                pretrained_model.setup_latent_norm(
                    stats=getattr(pretrained_model.config, "actlat_latent_stats", None),
                    stats_path=actlat_latent_stats_path,
                )

        elif not load_action_head:
            print("Initializing action head from scratch. Only loading backbone.")
            action_head_cfg = FlowmatchingActionHeadConfig(**pretrained_model.config.action_head_cfg)
            pretrained_model.action_head = FlowmatchingActionHead(action_head_cfg)

        elif not resume and "nvidia/GR00T-N1.5-3B" in pretrained_model_name_or_path:
            for name, param in pretrained_model.action_head.named_parameters():
                if "flare" in name or "future_tokens" in name:
                    pretrained_model.action_head._init_weights(param, name)
                    print(f"Reinitialized {name}")

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model

    @classmethod
    def from_same_trained(cls, pretrained_model_name_or_path: str, load_action_head: bool = True, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        # Pop flare params for compat
        vision_token_num = kwargs.pop("vision_token_num", 0)
        flare_loss_lambda = kwargs.pop("flare_loss_lambda", 0.0)
        flare_align_layers = kwargs.pop("flare_align_layers", 12)
        image_count = kwargs.pop("image_count", 1)
        resume = kwargs.pop("resume", False)
        video_only = kwargs.pop("video_only", False)
        flare_image_time = kwargs.pop("flare_image_time", "future")

        # Action latent tokenizer config
        actlat_tokenizer_path = kwargs.pop("actlat_tokenizer_path", None)
        actlat_target_tokens = kwargs.pop("actlat_target_tokens", "all")
        actlat_embodiment_id = kwargs.pop("actlat_embodiment_id", None)
        # Force deterministic-mu latent target regardless of the tokenizer's checkpoint
        # marker (None/False here → use the checkpoint setting).
        actlat_vae_no_sample = kwargs.pop("actlat_vae_no_sample", False)
        # Latent z-norm (see setup_latent_norm).
        actlat_latent_norm = kwargs.pop("actlat_latent_norm", False)
        actlat_latent_stats_path = kwargs.pop("actlat_latent_stats_path", "")

        try:
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            print(f"Model not found in HF hub. Loading from local path: {pretrained_model_name_or_path}")
            local_model_path = pretrained_model_name_or_path

        update_action_head_cfg = {
            "num_target_vision_tokens": vision_token_num,
            "flare_loss_lambda": 0.0,
            "flare_align_layers": flare_align_layers,
            "image_count": image_count,
            "flare_image_time": flare_image_time,
        }

        # Load tokenizer FIRST to get latent dimensions, so the model is
        # created with the correct action_dim/action_horizon before loading
        # checkpoint weights (which were saved with latent dimensions).
        tokenizer = None
        if actlat_tokenizer_path is not None:
            tokenizer, latent_dim, num_tokens = cls._load_tokenizer(
                actlat_tokenizer_path, actlat_target_tokens, embodiment_id=actlat_embodiment_id,
                vae_sample_override=(False if actlat_vae_no_sample else None),
            )
            update_action_head_cfg["action_dim"] = latent_dim
            update_action_head_cfg["action_horizon"] = num_tokens

        pretrained_model, loading_info = super().from_pretrained(
            local_model_path,
            local_model_path=local_model_path,
            output_loading_info=True,
            action_head_update=update_action_head_cfg,
            **kwargs,
        )

        print("Loading Info:")
        print(loading_info)

        # Attach tokenizer for inference decode
        if tokenizer is not None:
            pretrained_model.action_latent_tokenizer = tokenizer.to(pretrained_model.device)
            pretrained_model.actlat_target_tokens = actlat_target_tokens

            # Latent z-norm: explicit kwarg OR a checkpoint trained with it (the
            # checkpoint's config.json carries the flag + the embedded stats, so
            # eval hosts de-normalize without needing the stats file).
            if actlat_latent_norm or bool(
                getattr(pretrained_model.config, "actlat_latent_norm", False)
            ):
                pretrained_model.setup_latent_norm(
                    stats=getattr(pretrained_model.config, "actlat_latent_stats", None),
                    stats_path=actlat_latent_stats_path,
                )

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model


# register
AutoConfig.register("gr00t_n1_5", GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)

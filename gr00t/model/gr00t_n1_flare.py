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

from .action_head.flow_matching_action_head_flare import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .backbone import EagleBackbone
#from .diff_sit import SiT

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
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


# real model
class GR00T_N1_5(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

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

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
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
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:

        input_1 = {}
        input_1["state"] = inputs["state"][:, 0:1, :]
        input_1["state_mask"] = inputs["state_mask"][:, 0:1, :]
        input_1["segmentation_target"] = inputs["segmentation_target"]
        input_1["segmentation_target_mask"] = inputs["segmentation_target_mask"]
        input_1["has_real_action"] = inputs["has_real_action"]
        input_1["action"] = inputs["action"]
        input_1["action_mask"] = inputs["action_mask"]
        input_1["eagle_input_ids"] = inputs["eagle_input_ids"]
        input_1["eagle_attention_mask"] = inputs["eagle_attention_mask"]
        input_1["eagle_pixel_values"] = inputs["eagle_pixel_values"]
        input_1["eagle_image_sizes"] = inputs["eagle_image_sizes"]
        input_1["embodiment_id"] = inputs["embodiment_id"]

        input_2 = {}
        input_2["state"] = inputs["state"][:, 1:2, :]
        input_2["state_mask"] = inputs["state_mask"][:, 1:2, :]
        input_2["segmentation_target"] = inputs["segmentation_target"]
        input_2["segmentation_target_mask"] = inputs["segmentation_target_mask"]
        input_2["has_real_action"] = inputs["has_real_action"]
        input_2["action"] = inputs["action"]
        input_2["action_mask"] = inputs["action_mask"]
        input_2["eagle_input_ids"] = inputs["eagle2_input_ids"]
        input_2["eagle_attention_mask"] = inputs["eagle2_attention_mask"]
        input_2["eagle_pixel_values"] = inputs["eagle2_pixel_values"]
        input_2["eagle_image_sizes"] = inputs["eagle2_image_sizes"]
        input_2["embodiment_id"] = inputs["embodiment_id"]

        backbone_inputs, action_inputs = self.prepare_input(input_1)
        backbone_inputs_2, action_inputs_2 = self.prepare_input(input_2)
        backbone_outputs = self.backbone(backbone_inputs)
        if self.action_head.config.flare_image_time == "current":
            backbone_outputs_2 = None
        else:
            backbone_outputs_2 = self.backbone(backbone_inputs_2)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs, backbone_outputs_2)

        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def get_action_with_time_check(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        start_time = time.perf_counter()
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        end_time = time.perf_counter()
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        time_taken = end_time - start_time
        return action_head_outputs, time_taken

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, load_action_head: bool=True, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        vision_token_num = kwargs.pop("vision_token_num", 64)
        flare_loss_lambda = kwargs.pop("flare_loss_lambda", 0.2)
        flare_align_layers = kwargs.pop("flare_align_layers", 12)
        image_count = kwargs.pop("image_count", 1)
        resume = kwargs.pop("resume", False)
        video_only = kwargs.pop("video_only", False)
        flare_image_time = kwargs.pop("flare_image_time", "future")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        update_action_head_cfg = {}
        key_mapping = {}

        if load_action_head and not resume:   #load_action_head=False
            update_action_head_cfg = {
                "num_target_vision_tokens": vision_token_num,
                "flare_loss_lambda": flare_loss_lambda,
                "flare_align_layers": flare_align_layers,
                "image_count": image_count,
                "flare_image_time": flare_image_time,
            }

            if "nvidia/GR00T-N1.5-3B" in pretrained_model_name_or_path:
                key_mapping = {
                    "action_head.future_tokens.weight": "action_head.future_tokens.weight_l",
                }

        pretrained_model, loading_info = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, output_loading_info=True, \
                action_head_update=update_action_head_cfg, key_mapping=key_mapping, **kwargs
        )

        if not load_action_head:
            print("TY: Initializing action head from scratch. Only loading backbone.")
            action_head_cfg = FlowmatchingActionHeadConfig(**pretrained_model.config.action_head_cfg)
            action_head_cfg.num_target_vision_tokens = vision_token_num
            action_head_cfg.flare_loss_lambda = flare_loss_lambda
            action_head_cfg.flare_align_layers = flare_align_layers
            action_head_cfg.image_count = image_count
            action_head_cfg.video_only = video_only
            action_head_cfg.flare_image_time = flare_image_time

            pretrained_model.action_head = FlowmatchingActionHead(action_head_cfg)
        elif not resume and "nvidia/GR00T-N1.5-3B" in pretrained_model_name_or_path:
            for name, param in pretrained_model.action_head.named_parameters():
                # print(name, type(name), end=" / ")
                if "flare" in name or "future_tokens" in name:
                    pretrained_model.action_head._init_weights(param, name)
                    print(f"Reinitialized {name}, {type(param)}")

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model

    @classmethod
    def from_same_trained(cls, pretrained_model_name_or_path: str, load_action_head: bool=True, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        vision_token_num = kwargs.pop("vision_token_num", 64)
        flare_loss_lambda = kwargs.pop("flare_loss_lambda", 0)
        flare_align_layers = kwargs.pop("flare_align_layers", 12)
        image_count = kwargs.pop("image_count", 1)
        resume = kwargs.pop("resume", False)
        video_only = kwargs.pop("video_only", False)
        flare_image_time = kwargs.pop("flare_image_time", "future")

        update_action_head_cfg = {
                "num_target_vision_tokens": vision_token_num,
                "flare_loss_lambda": flare_loss_lambda,
                "flare_align_layers": flare_align_layers,
                "image_count": image_count,
                "flare_image_time": flare_image_time,
            }
        
        pretrained_model, loading_info = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, output_loading_info=True, action_head_update=update_action_head_cfg, **kwargs
        )

        print("Loading Info:")
        print(loading_info)

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

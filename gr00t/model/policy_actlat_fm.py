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

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.gr00t_n1_actlat_fm import GR00T_N1_5

COMPUTE_DTYPE = torch.bfloat16


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        raise NotImplementedError


class Gr00tPolicy(BasePolicy):
    """Policy wrapper for actlat_fm and nactlat_baseline (VLA) models."""

    def __init__(
        self,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        denoising_steps: Optional[int] = None,
        device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
        actlat_tokenizer_path: Optional[str] = None,
        actlat_target_tokens: str = "all",
    ):
        try:
            model_path = snapshot_download(model_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {model_path}"
            )

        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self._modality_transform.eval()
        self.model_path = Path(model_path)
        self.device = device
        self.actlat_tokenizer_path = actlat_tokenizer_path
        self.actlat_target_tokens = actlat_target_tokens

        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        self._load_model(model_path)
        self._load_metadata(self.model_path / "experiment_cfg")
        self._load_horizons()

        if denoising_steps is not None:
            if hasattr(self.model, "action_head") and hasattr(
                self.model.action_head, "num_inference_timesteps"
            ):
                self.model.action_head.num_inference_timesteps = denoising_steps
                print(f"Set action denoising steps to {denoising_steps}")

    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return self._modality_transform(obs)

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return self._modality_transform.unapply(action)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        is_batch = self._check_state_is_batched(observations)
        if not is_batch:
            observations = unsqueeze_dict_values(observations)

        for k, v in observations.items():
            if not isinstance(v, np.ndarray):
                observations[k] = np.array(v)

        normalized_input = self.apply_transforms(observations)
        normalized_action = self._get_action_from_normalized_input(normalized_input)
        unnormalized_action = self._get_unnormalized_action(normalized_action)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    def _get_action_from_normalized_input(self, normalized_input: Dict[str, Any]) -> torch.Tensor:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = self.model.get_action(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def _get_unnormalized_action(self, normalized_action: torch.Tensor) -> Dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()})

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        return self._modality_config

    @property
    def modality_config(self) -> Dict[str, ModalityConfig]:
        return self._modality_config

    @property
    def modality_transform(self) -> ComposedModalityTransform:
        return self._modality_transform

    @property
    def video_delta_indices(self) -> np.ndarray:
        return self._video_delta_indices

    @property
    def state_delta_indices(self) -> np.ndarray | None:
        return self._state_delta_indices

    @property
    def denoising_steps(self) -> int:
        return self.model.action_head.num_inference_timesteps

    @denoising_steps.setter
    def denoising_steps(self, value: int):
        self.model.action_head.num_inference_timesteps = value

    def _check_state_is_batched(self, obs: Dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:
                return False
        return True

    def _load_model(self, model_path):
        model = GR00T_N1_5.from_same_trained(
            model_path,
            torch_dtype=COMPUTE_DTYPE,
            actlat_tokenizer_path=self.actlat_tokenizer_path,
            actlat_target_tokens=self.actlat_target_tokens,
        )
        model.eval()
        model.to(device=self.device)
        self.model = model

    def _load_metadata(self, exp_cfg_dir: Path):
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)
        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

    def _load_horizons(self):
        self._video_delta_indices = np.array(self._modality_config["video"].delta_indices)
        self._assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        if "state" in self._modality_config:
            self._state_delta_indices = np.array(self._modality_config["state"].delta_indices)
            self._assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None

    def _assert_delta_indices(self, delta_indices: np.ndarray):
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"


def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.array(v)
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v)
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze()
        else:
            squeezed_data[k] = v
    return squeezed_data

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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform, AnyResolutionConcatTransform
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
    VideoCropDroid,
)
from gr00t.model.transforms import GR00TTransform, GR00TInferTransform, GR00TTactileTransform, GR00TAnyResolutionTransform


class BaseDataConfig(ABC):
    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


#####################################################################################
# helper functions
#####################################################################################


def import_external_data_config(data_config_str: str) -> Optional[BaseDataConfig]:
    """
    Import and instantiate an external data configuration class.

    Format: "module_path:ClassName" (e.g., "my_configs:RobotConfig")
    Supports nested modules like "package.submodule:ClassName"
    """
    if ":" not in data_config_str:
        return None

    import importlib
    import os
    import sys
    from pathlib import Path

    # Add current working directory to Python path
    current_dir = str(Path(os.getcwd()).absolute())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    try:
        module_path, class_name = data_config_str.split(":", 1)
        if not module_path or not class_name:
            raise ValueError(f"Invalid format: '{data_config_str}'. Use 'module:ClassName'")

        print(f"Loading external config: {module_path}.{class_name}")

        module = importlib.import_module(module_path)
        if not hasattr(module, class_name):
            available = [
                n
                for n in dir(module)
                if not n.startswith("_") and isinstance(getattr(module, n), type)
            ]
            raise AttributeError(
                f"Class '{class_name}' not found in '{module_path}'. Available: {available}"
            )

        # assert if the class has 'transform' and 'modality_config' methods
        if not hasattr(getattr(module, class_name), "transform"):
            raise AttributeError(f"Class '{class_name}' does not have a 'transform' method")
        if not hasattr(getattr(module, class_name), "modality_config"):
            raise AttributeError(f"Class '{class_name}' does not have a 'modality_config' method")

        return getattr(module, class_name)()

    except (ModuleNotFoundError, AttributeError, ValueError) as e:
        print(f"Config loading failed: {e}")
        print("Example: my_configs:MyConfig, package.submodule:ClassName")
        raise


def load_data_config(data_config_str: str, from_oxe=False) -> BaseDataConfig:
    """
    Get a data config class from a string.
    >>> load_data_config("so100")
    >>> get_data_config("dir.subdir.my_configs:RobotConfig")
    """
    if data_config_str in DATA_CONFIG_MAP:
        return DATA_CONFIG_MAP[data_config_str]
    elif from_oxe:
        return OXERLDSDataConfig(data_config_str)
    data_config_cls = import_external_data_config(data_config_str)
    if data_config_cls is not None:
        return data_config_cls
    # Yellow warning color
    yellow = "\033[93m"
    reset = "\033[0m"
    raise ValueError(
        f"{yellow}Invalid data_config '{data_config_str}'. "
        f"Available options: {list(DATA_CONFIG_MAP.keys())}, "
        f"or use 'module:ClassName' for external configs{reset}"
    )


###########################################################################################

###########################################################################################


class FourierGr1ArmsOnlyDataConfig(BaseDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0, 15]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class So100DataConfig(BaseDataConfig):
    video_keys = ["video.webcam"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class So100DualCamDataConfig(So100DataConfig):
    video_keys = ["video.front", "video.wrist"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))


###########################################################################################


class UnitreeG1DataConfig(BaseDataConfig):
    video_keys = ["video.rs_view"]
    state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class UnitreeG1FullBodyDataConfig(UnitreeG1DataConfig):
    video_keys = ["video.rs_view"]
    state_keys = [
        "state.left_leg",
        "state.right_leg",
        "state.waist",
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))


###########################################################################################


class FourierGr1FullUpperBodyDataConfig(BaseDataConfig):
    video_keys = ["video.front_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
        "state.neck",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
        "action.neck",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class BimanualPandaGripperDataConfig(BaseDataConfig):
    video_keys = [
        "video.right_wrist_view",
        "video.left_wrist_view",
        "video.front_view",
    ]
    state_keys = [
        "state.right_arm_eef_pos",
        "state.right_arm_eef_quat",
        "state.right_gripper_qpos",
        "state.left_arm_eef_pos",
        "state.left_arm_eef_quat",
        "state.left_gripper_qpos",
    ]
    action_keys = [
        "action.right_arm_eef_pos",
        "action.right_arm_eef_rot",
        "action.right_gripper_close",
        "action.left_arm_eef_pos",
        "action.left_arm_eef_rot",
        "action.left_gripper_close",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.right_arm_eef_pos": "min_max",
        "state.right_gripper_qpos": "min_max",
        "state.left_arm_eef_pos": "min_max",
        "state.left_gripper_qpos": "min_max",
    }
    state_target_rotations = {
        "state.right_arm_eef_quat": "rotation_6d",
        "state.left_arm_eef_quat": "rotation_6d",
    }
    action_normalization_modes = {
        "action.right_gripper_close": "binary",
        "action.left_gripper_close": "binary",
    }

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class BimanualPandaHandDataConfig(BimanualPandaGripperDataConfig):
    video_keys = [
        "video.right_wrist_view",
        "video.left_wrist_view",
        "video.ego_view",
    ]
    state_keys = [
        "state.right_arm_eef_pos",
        "state.right_arm_eef_quat",
        "state.right_hand",
        "state.left_arm_eef_pos",
        "state.left_arm_eef_quat",
        "state.left_hand",
    ]
    action_keys = [
        "action.right_arm_eef_pos",
        "action.right_arm_eef_rot",
        "action.right_hand",
        "action.left_arm_eef_pos",
        "action.left_arm_eef_rot",
        "action.left_hand",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.right_arm_eef_pos": "min_max",
        "state.right_hand": "min_max",
        "state.left_arm_eef_pos": "min_max",
        "state.left_hand": "min_max",
    }
    action_normalization_modes = {
        "action.right_hand": "min_max",
        "action.left_hand": "min_max",
    }
    state_target_rotations = {
        "state.right_arm_eef_quat": "rotation_6d",
        "state.left_arm_eef_quat": "rotation_6d",
    }


###########################################################################################


class SinglePandaGripperDataConfig(BimanualPandaGripperDataConfig):
    video_keys = [
        "video.left_view",
        "video.right_view",
        "video.wrist_view",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.end_effector_position_relative": "min_max",
        "state.end_effector_rotation_relative": "min_max",
        "state.gripper_qpos": "min_max",
        "state.base_position": "min_max",
        "state.base_rotation": "min_max",
    }
    state_target_rotations = {
        "state.end_effector_rotation_relative": "rotation_6d",
        "state.base_rotation": "rotation_6d",
    }
    action_normalization_modes = {
        "action.end_effector_position": "min_max",
        "action.end_effector_rotation": "min_max",
        "action.gripper_close": "binary",
        "action.base_motion": "min_max",
        "action.control_mode": "binary",
    }

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TInferTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

class SinglePandaGripperFrontDataConfig(SinglePandaGripperDataConfig):
    """Single-camera (front) variant of dexjoco_dual_arm for the V4 action-latent
    tokenizer, which is hard-pinned to one camera (dataset_action_frames_v4.py
    asserts len(video_keys) == 1). State/action/language keys are inherited
    unchanged; only the video modality is narrowed to video.front. The Stage-2
    VLA training keeps using the full 3-camera dexjoco_dual_arm config."""
    video_keys = ["video.left_view"]


class SinglePandaGripperFLAREDataConfig(BimanualPandaGripperDataConfig):
    video_keys = [
        "video.left_view",
        "video.right_view",
        "video.wrist_view",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0, 15]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.end_effector_position_relative": "min_max",
        "state.end_effector_rotation_relative": "min_max",
        "state.gripper_qpos": "min_max",
        "state.base_position": "min_max",
        "state.base_rotation": "min_max",
    }
    state_target_rotations = {
        "state.end_effector_rotation_relative": "rotation_6d",
        "state.base_rotation": "rotation_6d",
    }
    action_normalization_modes = {
        "action.end_effector_position": "min_max",
        "action.end_effector_rotation": "min_max",
        "action.gripper_close": "binary",
        "action.base_motion": "min_max",
        "action.control_mode": "binary",
    }


###########################################################################################
class SinglePandaGripperActlatFMDataConfig(SinglePandaGripperDataConfig):
    """Actlat-FM variant of SinglePandaGripperDataConfig.

    `SinglePandaGripperDataConfig` 와의 차이는 `transform()` 의
    `GR00TInferTransform(max_action_dim=12)` (부모: 32) 하나뿐.
    12 는 실제 concat action dim (EE_pos 3 + EE_rot 3 + gripper 1 + base 4 + ctrl_mode 1)
    과 일치시켜 토크나이저 ↔ VLA encode/decode 정합성을 확보.

    action_normalization_modes (gripper_close / control_mode 의 binary 포함),
    state_normalization_modes, state_target_rotations (quat → rotation_6d) 등
    나머지 속성은 모두 부모 상속. Tokenizer pretransform 이 이 속성들을 respect 하므로
    VLA 와 동일한 action/state 표현으로 aux loss 학습.
    """
    tokenizer_frame_video_key = "video.left_view"
    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TInferTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=12,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################
class RoboTwinActlatFMDataConfig(BaseDataConfig):
    """Actlat-FM config for RoboTwin2.0 (bimanual Franka).

    modality.json (RoboTwin2.0_easy_merged):
      state  : left_endpose(7) + left_gripper(1) + right_endpose(7) + right_gripper(1) = 16
      action : left_arm(7)     + left_gripper(1) + right_arm(7)     + right_gripper(1) = 16
      video  : head_camera, left_camera, right_camera, third_view
      lang   : annotation.human.action.task_description

    정규화는 AlinVLAv0 (g0_v1d3d1) 컨벤션을 따라 state/action 전부 q99 (q01/q99 기반).
    min_max 대비 outlier robust. gripper 도 binary 가 아닌 연속 q99 (AlinVLAv0 동일).
    endpose 는 modality.json 에서 단일 7-dim key (3 pos + 4 quat) 로 묶여 있어 sub-key 분할 안 함.

    observation_indices=[0] / action_indices=range(16) — actlat_fm 표준.
    max_action_dim=16 (RoboTwin 실제 action_dim) → 토크나이저 ↔ VLA 정합.
    """

    video_keys = [
        "video.head_camera",
        # "video.left_camera",
        # "video.right_camera",
        # "video.third_view",
    ]
    state_keys = [
        "state.left_endpose",
        "state.left_gripper",
        "state.right_endpose",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_arm",
        "action.left_gripper",
        "action.right_arm",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    state_normalization_modes = {
        "state.left_endpose": "q99",
        "state.left_gripper": "q99",
        "state.right_endpose": "q99",
        "state.right_gripper": "q99",
    }
    action_normalization_modes = {
        "action.left_arm": "q99",
        "action.left_gripper": "q99",
        "action.right_arm": "q99",
        "action.right_gripper": "q99",
    }

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TInferTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=16,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)



class FourierGr1ArmsWaistActlatFMDataConfig(FourierGr1ArmsOnlyDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return super().modality_config()

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TInferTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=29,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class FourierGr1ArmsWaistActlatFM1000DemosDataConfig(FourierGr1ArmsWaistActlatFMDataConfig):
    """Same as FourierGr1ArmsWaistActlatFMDataConfig but actions are normalized
    with q01/q99 (``q99`` mode) instead of ``min_max``. State is unchanged
    (still sin/cos via StateActionSinCosTransform)."""

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms (unchanged: sin/cos)
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms (q01/q99 normalization instead of min_max)
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TInferTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=29,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class FourierGr1ArmsWaistDataConfig(FourierGr1ArmsOnlyDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return super().modality_config()

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TInferTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class FourierGr1ArmsWaistFLAREDataConfig(FourierGr1ArmsOnlyDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0, 15]
    action_indices = list(range(16))

    def modality_config(self):
        return super().modality_config()

    def transform(self):
        return super().transform()


###########################################################################################


class OxeDroidDataConfig:
    video_keys = [
        "video.exterior_image_1",
        "video.exterior_image_2",
        "video.wrist_image",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_position",
    ]
    action_keys = [
        "action.eef_position_delta",
        "action.eef_rotation_delta",
        "action.gripper_position",
    ]
    language_keys = ["annotation.language.language_instruction"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.gripper_position": "min_max",
                },
                target_rotations={
                    "state.eef_rotation": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.gripper_position": "binary",
                },
                target_rotations={"action.eef_rotation_delta": "axis_angle"},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class AgibotGenie1DataConfig:
    video_keys = [
        "video.top_head",
        "video.hand_left",
        "video.hand_right",
    ]
    state_keys = [
        "state.left_arm_joint_position",
        "state.right_arm_joint_position",
        "state.left_effector_position",
        "state.right_effector_position",
        "state.head_position",
        "state.waist_position",
    ]
    action_keys = [
        "action.left_arm_joint_position",
        "action.right_arm_joint_position",
        "action.left_effector_position",
        "action.right_effector_position",
        "action.head_position",
        "action.waist_position",
        "action.robot_velocity",
    ]
    language_keys = ["annotation.language.action_text"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_arm_joint_position": "min_max",
                    "state.right_arm_joint_position": "min_max",
                    "state.left_effector_position": "min_max",
                    "state.right_effector_position": "min_max",
                    "state.head_position": "min_max",
                    "state.waist_position": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_arm_joint_position": "min_max",
                    "action.right_arm_joint_position": "min_max",
                    "action.left_effector_position": "min_max",
                    "action.right_effector_position": "min_max",
                    "action.head_position": "min_max",
                    "action.waist_position": "min_max",
                    "action.robot_velocity": "min_max",
                },
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class LiberoDataConfig(BaseDataConfig):
    video_keys = ["video.front_view", "video.left_wrist_view"]
    state_keys = [
        "state.eef_pos_absolute",
        "state.eef_rot_absolute",
        "state.gripper_close"
    ]
    action_keys = [
        "action.eef_pos_delta",
        "action.eef_rot_delta",
        "action.gripper_close"
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self, num_frames=1):
        video_modality = ModalityConfig(
            delta_indices=[-1 * num_frames + 1 + i for i in range(num_frames)],
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, backbone_model_type="eagle"):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_pos_absolute": "min_max",
                    "state.eef_rot_absolute": "min_max",
                    "state.gripper_close": "min_max",
                },
                target_rotations={
                    "state.eef_rot_absolute": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.eef_pos_delta": "min_max",
                    "action.eef_rot_delta": "min_max",
                    "action.gripper_close": "min_max",
                },
                # target_rotations={
                #    "action.eef_rot_delta": "axis_angle" # Relative ???
                # }
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TInferTransform(
                backbone_model_type=backbone_model_type,
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
    

class LiberoDataConfigFinetune(BaseDataConfig):
    video_keys = ["video.front_view", "video.left_wrist_view"]
    state_keys = [
        "state.eef_pos_absolute",
        "state.eef_rot_absolute",
        "state.gripper_close"
    ]
    action_keys = [
        "action.eef_pos_delta",
        "action.eef_rot_delta",
        "action.gripper_close"
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self, num_frames=1):
        video_modality = ModalityConfig(
            delta_indices=[-1 * num_frames + 1 + i for i in range(num_frames)],
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, backbone_model_type="eagle"):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_pos_absolute": "min_max",
                    "state.eef_rot_absolute": "min_max",
                    "state.gripper_close": "min_max",
                },
                target_rotations={
                    "state.eef_rot_absolute": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.eef_pos_delta": "min_max",
                    "action.eef_rot_delta": "min_max",
                    "action.gripper_close": "min_max",
                },
                # target_rotations={
                #    "action.eef_rot_delta": "axis_angle" # Relative ???
                # }
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                backbone_model_type=backbone_model_type,
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class LiberoFLAREDataConfig(BaseDataConfig):
    video_keys = ["video.front_view", "video.left_wrist_view"]
    state_keys = [
        "state.eef_pos_absolute",
        "state.eef_rot_absolute",
        "state.gripper_close"
    ]
    action_keys = [
        "action.eef_pos_delta",
        "action.eef_rot_delta",
        "action.gripper_close"
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0, 15]
    action_indices = list(range(16))

    def modality_config(self, num_frames=1):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, backbone_model_type="eagle"):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_pos_absolute": "min_max",
                    "state.eef_rot_absolute": "min_max",
                    "state.gripper_close": "min_max",
                },
                target_rotations={
                    "state.eef_rot_absolute": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.eef_pos_delta": "min_max",
                    "action.eef_rot_delta": "min_max",
                    "action.gripper_close": "min_max",
                },
                # target_rotations={
                #    "action.eef_rot_delta": "axis_angle" # Relative ???
                # }
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                backbone_model_type=backbone_model_type,
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

# class BridgeDataConfig(BaseDataConfig):
#     video_keys = [
#         "video.image_0", 
#         # "video.image_1", 
#         # "video.image_2", 
#         # "video.image_3"
#     ]
#     state_keys = [
#         "state.eef_position",
#         "state.eef_rotation",
#         "state.gripper_position",
#     ]
#     action_keys = [
#         "action.eef_position_delta",
#         "action.eef_rotation_delta",
#         "action.gripper_position",
#     ]
#     language_keys = ["annotation.human.action.task_description"]
#     observation_indices = [0]
#     action_indices = list(range(16))

#     def modality_config(self):
#         video_modality = ModalityConfig(
#             delta_indices=self.observation_indices,
#             modality_keys=self.video_keys,
#         )
#         state_modality = ModalityConfig(
#             delta_indices=self.observation_indices,
#             modality_keys=self.state_keys,
#         )
#         action_modality = ModalityConfig(
#             delta_indices=self.action_indices,
#             modality_keys=self.action_keys,
#         )
#         language_modality = ModalityConfig(
#             delta_indices=self.observation_indices,
#             modality_keys=self.language_keys,
#         )
#         modality_configs = {
#             "video": video_modality,
#             "state": state_modality,
#             "action": action_modality,
#             "language": language_modality,
#         }
#         return modality_configs

#     def transform(self, backbone_model_type="eagle"):
#         transforms = [
#             # video transforms
#             VideoToTensor(apply_to=self.video_keys),
#             VideoCrop(apply_to=self.video_keys, scale=0.95),
#             VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
#             VideoColorJitter(
#                 apply_to=self.video_keys,
#                 brightness=0.3,
#                 contrast=0.4,
#                 saturation=0.5,
#                 hue=0.08,
#             ),
#             VideoToNumpy(apply_to=self.video_keys),
#             # state transforms
#             StateActionToTensor(apply_to=self.state_keys),
#             StateActionTransform(
#                 apply_to=self.state_keys,
#                 normalization_modes={
#                     "state.eef_position": "min_max",
#                     "state.eef_rotation": "min_max",
#                     "state.gripper_position": "min_max",
#                 },
#                 # target_rotations={
#                 #     "state.eef_rotation": "rotation_6d",
#                 # },
#             ),
#             # action transforms
#             StateActionToTensor(apply_to=self.action_keys),
#             StateActionTransform(
#                 apply_to=self.action_keys,
#                 normalization_modes={
#                     "action.eef_position_delta": "min_max",
#                     "action.eef_rotation_delta": "min_max",
#                     "action.gripper_position": "min_max",
#                 },
#                 # target_rotations={"action.eef_rotation_delta": "axis_angle"},
#             ),
#             # concat transforms
#             ConcatTransform(
#                 video_concat_order=self.video_keys,
#                 state_concat_order=self.state_keys,
#                 action_concat_order=self.action_keys,
#             ),
#             GR00TTransform(
#                 backbone_model_type=backbone_model_type,
#                 state_horizon=len(self.observation_indices),
#                 action_horizon=len(self.action_indices),
#                 max_state_dim=64,
#                 max_action_dim=32,
#             ),
#         ]

#         return ComposedModalityTransform(transforms=transforms)

# class BridgeNewDataConfig(BaseDataConfig):
#     video_keys = [
#         "video.image_0", 
#         "video.image_1", 
#         "video.image_2", 
#         "video.image_3"
#     ]
#     state_keys = [
#         "state.eef_position",
#         "state.eef_rotation",
#         "state.gripper_position",
#     ]
#     action_keys = [
#         "action.eef_position_delta",
#         "action.eef_rotation_delta",
#         "action.gripper_position",
#     ]
#     language_keys = ["annotation.human.action.task_description"]
#     observation_indices = [0]
#     action_indices = list(range(16))

#     def modality_config(self):
#         video_modality = ModalityConfig(
#             delta_indices=self.observation_indices,
#             modality_keys=self.video_keys,
#         )
#         state_modality = ModalityConfig(
#             delta_indices=self.observation_indices,
#             modality_keys=self.state_keys,
#         )
#         action_modality = ModalityConfig(
#             delta_indices=self.action_indices,
#             modality_keys=self.action_keys,
#         )
#         language_modality = ModalityConfig(
#             delta_indices=self.observation_indices,
#             modality_keys=self.language_keys,
#         )
#         modality_configs = {
#             "video": video_modality,
#             "state": state_modality,
#             "action": action_modality,
#             "language": language_modality,
#         }
#         return modality_configs

#     def transform(self, backbone_model_type="eagle"):
#         transforms = [
#             # video transforms
#             VideoToTensor(apply_to=self.video_keys),
#             VideoCrop(apply_to=self.video_keys, scale=0.95),
#             VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
#             VideoColorJitter(
#                 apply_to=self.video_keys,
#                 brightness=0.3,
#                 contrast=0.4,
#                 saturation=0.5,
#                 hue=0.08,
#             ),
#             VideoToNumpy(apply_to=self.video_keys),
#             # state transforms
#             StateActionToTensor(apply_to=self.state_keys),
#             StateActionTransform(
#                 apply_to=self.state_keys,
#                 normalization_modes={
#                     "state.eef_position": "min_max",
#                     "state.eef_rotation": "min_max",
#                     # "state.gripper_position": "min_max",
#                 },
#                 # target_rotations={
#                 #     "state.eef_rotation": "rotation_6d",
#                 # },
#             ),
#             # action transforms
#             StateActionToTensor(apply_to=self.action_keys),
#             StateActionTransform(
#                 apply_to=self.action_keys,
#                 normalization_modes={
#                     # "action.eef_position_delta": "min_max",
#                     # "action.eef_rotation_delta": "min_max",
#                     "action.gripper_position": "binary",
#                 },
#                 # target_rotations={"action.eef_rotation_delta": "axis_angle"},
#             ),
#             # concat transforms
#             ConcatTransform(
#                 video_concat_order=self.video_keys,
#                 state_concat_order=self.state_keys,
#                 action_concat_order=self.action_keys,
#             ),
#             GR00TTransform(
#                 backbone_model_type=backbone_model_type,
#                 state_horizon=len(self.observation_indices),
#                 action_horizon=len(self.action_indices),
#                 max_state_dim=64,
#                 max_action_dim=32,
#             ),
#         ]

#         return ComposedModalityTransform(transforms=transforms)

# class BridgeNew5DataConfig(BaseDataConfig):
#     video_keys = [
#         "video.image_0", 
#         "video.image_1", 
#         "video.image_2", 
#         "video.image_3"
#     ]
#     state_keys = [
#         "state.eef_position",
#         "state.eef_rotation",
#         "state.gripper_position",
#     ]
#     action_keys = [
#         "action.eef_position_delta",
#         "action.eef_rotation_delta",
#         "action.gripper_position",
#     ]
#     language_keys = ["annotation.human.action.task_description"]
#     observation_indices = [0]
#     action_indices = list(range(5))

#     def modality_config(self):
#         video_modality = ModalityConfig(
#             delta_indices=self.observation_indices,
#             modality_keys=self.video_keys,
#         )
#         state_modality = ModalityConfig(
#             delta_indices=self.observation_indices,
#             modality_keys=self.state_keys,
#         )
#         action_modality = ModalityConfig(
#             delta_indices=self.action_indices,
#             modality_keys=self.action_keys,
#         )
#         language_modality = ModalityConfig(
#             delta_indices=self.observation_indices,
#             modality_keys=self.language_keys,
#         )
#         modality_configs = {
#             "video": video_modality,
#             "state": state_modality,
#             "action": action_modality,
#             "language": language_modality,
#         }
#         return modality_configs

#     def transform(self, backbone_model_type="eagle"):
#         transforms = [
#             # video transforms
#             VideoToTensor(apply_to=self.video_keys),
#             VideoCrop(apply_to=self.video_keys, scale=0.95),
#             VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
#             VideoColorJitter(
#                 apply_to=self.video_keys,
#                 brightness=0.3,
#                 contrast=0.4,
#                 saturation=0.5,
#                 hue=0.08,
#             ),
#             VideoToNumpy(apply_to=self.video_keys),
#             # state transforms
#             StateActionToTensor(apply_to=self.state_keys),
#             StateActionTransform(
#                 apply_to=self.state_keys,
#                 normalization_modes={
#                     "state.eef_position": "min_max",
#                     "state.eef_rotation": "min_max",
#                     # "state.gripper_position": "min_max",
#                 },
#                 # target_rotations={
#                 #     "state.eef_rotation": "rotation_6d",
#                 # },
#             ),
#             # action transforms
#             StateActionToTensor(apply_to=self.action_keys),
#             StateActionTransform(
#                 apply_to=self.action_keys,
#                 normalization_modes={
#                     # "action.eef_position_delta": "min_max",
#                     # "action.eef_rotation_delta": "min_max",
#                     "action.gripper_position": "binary",
#                 },
#                 # target_rotations={"action.eef_rotation_delta": "axis_angle"},
#             ),
#             # concat transforms
#             ConcatTransform(
#                 video_concat_order=self.video_keys,
#                 state_concat_order=self.state_keys,
#                 action_concat_order=self.action_keys,
#             ),
#             GR00TTransform(
#                 backbone_model_type=backbone_model_type,
#                 state_horizon=len(self.observation_indices),
#                 action_horizon=len(self.action_indices),
#                 max_state_dim=64,
#                 max_action_dim=32,
#             ),
#         ]

#         return ComposedModalityTransform(transforms=transforms)


class FractalDataConfig(So100DataConfig):
    video_keys = ["video.image", ]
    state_keys = ["state.x", "state.y", "state.z", "state.rx", "state.ry", "state.rz", "state.rw",  "state.gripper"]
    action_keys = ["action.x", "action.y", "action.z", "action.roll", "action.pitch", "action.yaw", "action.gripper"]
    language_keys = ["annotation.human.action.task_description"]

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class BridgeDataConfig(FractalDataConfig):
    video_keys = ["video.image_0", ]
    state_keys = ["state.x", "state.y", "state.z", "state.roll", "state.pitch", "state.yaw", "state.pad",  "state.gripper"]
    action_keys = ["action.x", "action.y", "action.z", "action.roll", "action.pitch", "action.yaw", "action.gripper"]
    language_keys = ["annotation.human.action.task_description"]

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TInferTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class BridgeFLAREDataConfig(FractalDataConfig):
    video_keys = ["video.image_0", ]
    state_keys = ["state.x", "state.y", "state.z", "state.roll", "state.pitch", "state.yaw", "state.pad",  "state.gripper"]
    action_keys = ["action.x", "action.y", "action.z", "action.roll", "action.pitch", "action.yaw", "action.gripper"]
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0, 15]



class RealDroidJointFLAREDataConfig:
    video_keys = [
        "video.exterior_image_1_left",
        "video.wrist_image_left",
    ]
    state_keys = [
        "state.joint_pos_abs",
        "state.gripper_close",
    ]
    action_keys = [
        "action.joint_pos_abs",
        "action.gripper_close",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0, 15]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, backbone_model_type="eagle"):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCropDroid(apply_to=self.video_keys),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.joint_pos_abs": "min_max",
                    "state.gripper_close": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.joint_pos_abs": "min_max",
                    "action.gripper_close": "binary",
                },
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                backbone_model_type=backbone_model_type,
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class RealDroidCartesianFLAREDataConfig:
    video_keys = [
        "video.exterior_image_1_left",
        "video.wrist_image_left",
    ]
    state_keys = [
        "state.end_effector_position",
        "state.end_effector_rotation",
        "state.gripper_close",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0, 15]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, backbone_model_type="eagle"):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCropDroid(apply_to=self.video_keys),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.end_effector_position": "min_max",
                    "state.end_effector_rotation": "min_max",
                    "state.gripper_close": "min_max",
                },
                target_rotations={
                    "state.end_effector_rotation": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.end_effector_position": "min_max",
                    "action.end_effector_rotation": "min_max",
                    "action.gripper_close": "binary",
                },
                target_rotations={"action.end_effector_rotation": "axis_angle"},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                backbone_model_type=backbone_model_type,
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class RealTactileDroidJointDataConfig:
    video_keys = [
        "video.exterior_image_1_left",
        "video.wrist_image_left",
    ]
    state_keys = [
        "state.joint_pos_abs",
        "state.gripper_close",
    ]
    action_keys = [
        "action.joint_pos_abs",
        "action.gripper_close",
    ]
    language_keys = ["annotation.human.action.task_description"]
    tactile_keys = [
        "tactile.left",
        "tactile.right",
    ]
    
    observation_indices = [0]
    action_indices = list(range(16))


    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        tactile_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.tactile_keys,
        )
        
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
            "tactile": tactile_modality,
        }
        return modality_configs

    def transform(self, backbone_model_type="eagle"):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.joint_pos_abs": "min_max",
                    "state.gripper_close": "min_max", # binary
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.joint_pos_abs": "min_max",
                    "action.gripper_close": "min_max", # binary
                },
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTactileTransform(
                backbone_model_type=backbone_model_type,
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class BridgeFlareKTYDataConfig(BaseDataConfig):
    video_keys = [
        "video.image_0", 
        # "video.image_1", 
        # "video.image_2", 
        # "video.image_3"
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_position",
    ]
    action_keys = [
        "action.eef_position_delta",
        "action.eef_rotation_delta",
        "action.gripper_position",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0, 15]
    action_indices = list(range(16))  

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, backbone_model_type="eagle"):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.gripper_position": "min_max",
                },
                target_rotations={
                    "state.eef_rotation": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.gripper_position": "binary",
                    "action.eef_position_delta": "min_max", ### add this
                },
                target_rotations={"action.eef_rotation_delta": "axis_angle"},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                #backbone_model_type=backbone_model_type,  ### delete this
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class BridgeFlareKTYActlatFMDataConfig(BridgeFlareKTYDataConfig):
    """Actlat-FM variant of BridgeFlareKTYDataConfig.

    Robocasa의 SinglePandaGripperActlatFMDataConfig 와 동일한 패턴.
    - max_action_dim=7 로 하향(부모 32) → 토크나이저(action_dim=7) ↔ VLA encode/decode 정합 확보.
    - state_target_rotations / action_target_rotations / state_normalization_modes /
      action_normalization_modes 를 class-level로 명시 → tokenizer pretransform 도
      rotation_6d / axis_angle / binary 정규화 모두 respect → 토크나이저 ↔ VLA
      action·state 표현 완전 일관:
        state_dim=10 (3 pos + 6 rot_6d + 1 grip)
        action_dim=7 (3 pos_delta + 3 rot_delta_axis_angle + 1 grip)

    observation_indices=[0] (single obs step) — robocasa/gr1 actlat_fm 패턴과 동일.
    부모 BridgeFlareKTYDataConfig 는 FLARE 용 [0, 15] 이지만 actlat_fm 은 미래 align 안 함.
    """

    observation_indices = [0]

    state_normalization_modes = {
        "state.eef_position": "min_max",
        "state.gripper_position": "min_max",
    }
    state_target_rotations = {
        "state.eef_rotation": "rotation_6d",
    }
    action_normalization_modes = {
        "action.eef_position_delta": "min_max",
        "action.gripper_position": "binary",
    }
    action_target_rotations = {
        "action.eef_rotation_delta": "axis_angle",
    }

    def transform(self, backbone_model_type="eagle"):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
                target_rotations=self.action_target_rotations,
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=7,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class FrankaRealEEFDataConfig(BaseDataConfig):
    video_keys = [
        "video.exterior_image_1_left", 
        "video.wrist_image_left",
        # "video.image_1", 
        # "video.image_2", 
        # "video.image_3"
    ]
    state_keys = [
        "state.end_effector_position",
        "state.end_effector_rotation",
        "state.gripper_position",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0, 15]
    action_indices = list(range(16))  

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self, backbone_model_type="eagle"):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.end_effector_position": "min_max",
                    "state.gripper_position": "min_max",
                    "state.end_effector_rotation": "min_max",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.gripper_close": "binary",
                    "action.end_effector_position": "min_max", ### add this
                    "action.end_effector_rotation": "min_max",
                },
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                #backbone_model_type=backbone_model_type,  ### delete this
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

class CalvinDataConfig(BaseDataConfig):
    video_keys = [
        "video.image", 
        "video.wrist_image",
    ]
    state_keys = [
        "state.state",
    ]
    action_keys = [
        "action.eef_pos_delta",
        "action.eef_rot_delta",
        "action.gripper_close"
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0, 15]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs


    def transform(self, backbone_model_type="eagle"):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.state": "min_max"
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.gripper_close": "binary",
                    "action.eef_pos_delta": "min_max", ### add this
                },
                target_rotations={"action.eef_rot_delta": "axis_angle"},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                #backbone_model_type=backbone_model_type,  ### delete this
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
###########################################################################################
###########################################################################################
# OXE
###########################################################################################


try:
    from gr00t.data.rlds.oxe.configs import StateEncoding, ActionEncoding, OXE_DATASET_CONFIGS

    class OXERLDSDataConfig(BaseDataConfig):
        def __init__(self, dataset_name: str):
            self.dataset_name = dataset_name
            self.video_keys = ["video.image_primary", "video.image_secondary", "video.image_wrist"]
            dataset_config = OXE_DATASET_CONFIGS[dataset_name]
            self.state_keys = self._state_to_keys(dataset_config["state_encoding"])
            self.action_keys = self._action_to_keys(dataset_config["action_encoding"])
            self.observation_indices = [0]
            self.action_indices = list(range(16))

            # Used in StateActionTransform for normalization and target rotations
            self.state_normalization_modes = dict()
            for key in self.state_keys:
                if "gripper_close" in key:
                    self.state_normalization_modes[key] = "binary"
                else:
                    self.state_normalization_modes[key] = "min_max"

            self.state_target_rotations = {
                "state.eef_rotation": "rotation_6d",
            } if "state.eef_rotation" in self.state_keys else dict()

            self.action_normalization_modes = dict()

            for key in self.action_keys:
                if "gripper_close" in key:
                    self.action_normalization_modes[key] = "binary"
                else:
                    self.action_normalization_modes[key] = "min_max"

            self.language_keys = ["annotation.task_description"]

        def modality_config(self) -> dict[str, ModalityConfig]:
            """Return modality config for OXE/RLDS datasets. Required by BaseDataConfig interface."""
            video_modality = ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            )
            state_modality = ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            )
            action_modality = ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            )
            language_modality = ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            )
            return {
                "video": video_modality,
                "state": state_modality,
                "action": action_modality,
                "language": language_modality,
            }

        def transform(self, backbone_model_type="eagle", backbone_path=None) -> ModalityTransform:
            # video_transforms = [# video transforms
            #     VideoToTensor(apply_to=self.video_keys),
            #     VideoCrop(apply_to=self.video_keys, scale=0.95),
            #     VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            #     VideoColorJitter(
            #         apply_to=self.video_keys,
            #         brightness=0.3,
            #         contrast=0.4,
            #         saturation=0.5,
            #         hue=0.08,
            #     ),
            #     VideoToNumpy(apply_to=self.video_keys),
            # ]
            state_transforms = [# state transforms
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes=self.state_normalization_modes,
                    target_rotations=self.state_target_rotations,
                ),
            ]
            action_transforms = [# action transforms
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=self.action_normalization_modes,
                    # target_rotations={"action.eef_delta_rotation": "axis_angle"},
                ),
            ]
            concat_transforms = [# concat transforms
                AnyResolutionConcatTransform(
                    video_concat_order=self.video_keys,
                    state_concat_order=self.state_keys,
                    action_concat_order=self.action_keys,
                ),
            ]
            model_transforms = [
                GR00TAnyResolutionTransform(
                    state_horizon=len(self.observation_indices),
                    action_horizon=len(self.action_indices),
                    backbone_model_type=backbone_model_type,
                    backbone_path=backbone_path,
                    max_state_dim=64,
                    max_action_dim=64,
                ),
            ]
            # transforms = video_transforms + state_transforms + action_transforms + concat_transforms + model_transforms
            transforms = state_transforms + action_transforms + concat_transforms + model_transforms

            return ComposedModalityTransform(transforms=transforms)

        def _state_to_keys(self, state: StateEncoding) -> list[str]:
            match state:
                case StateEncoding.POS_EULER:
                    return [
                        "state.eef_position",
                        "state.eef_rotation",
                        "state.gripper_close"
                    ]
                case StateEncoding.POS_QUAT:
                    return [
                        "state.eef_position",
                        "state.eef_rotation",
                        "state.gripper_close"
                    ]
                case StateEncoding.JOINT:
                    return [
                        "state.joint_positions",
                        "state.gripper_close"
                    ]
                case StateEncoding.JOINT_BIMANUAL:
                    return [
                        "state.left_joint_positions",
                        "state.left_gripper_close",
                        "state.right_joint_positions",
                        "state.right_gripper_close"
                    ]
                case StateEncoding.AGIBOT_DEXHAND:
                    return [
                        "state.agibot_dexhand"
                    ]
                case StateEncoding.AGIBOT_GRIPPER:
                    return [
                        "state.agibot_gripper"
                    ]
                case StateEncoding.GALAXEA:
                    return [
                        "state.galaxea"
                    ]
                case StateEncoding.HUMANOID_EVERYDAY_G1:
                    return [
                        "state.humanoid_everyday_g1"
                    ]
                case StateEncoding.HUMANOID_EVERYDAY_H1:
                    return [
                        "state.humanoid_everyday_h1"
                    ]
                case StateEncoding.ACTION_NET:
                    return [
                        "state.action_net"
                    ]
                case StateEncoding.NEURAL_GR1:
                    return [
                        "state.neural_gr1"
                    ]
                case StateEncoding.NONE:
                    return []
                case _:
                    raise ValueError(f"Unknown StateEncoding: {state}")

        def _action_to_keys(self, action: ActionEncoding) -> list[str]:
            match action:
                case ActionEncoding.EEF_POS:
                    return [
                        "action.eef_delta_position",
                        "action.eef_delta_rotation",
                        "action.gripper_close"
                    ]
                case ActionEncoding.JOINT_POS:
                    return [
                        "action.joint_delta_positions",
                        "action.gripper_close",
                    ]
                case ActionEncoding.JOINT_POS_BIMANUAL:
                    return [
                        "action.left_joint_delta_positions",
                        "action.left_gripper_close",
                        "action.right_joint_delta_positions",
                        "action.right_gripper_close"
                    ]
                case ActionEncoding.EEF_R6:
                    return [
                        "action.eef_delta_position",
                        "action.eef_rotation_6d",
                        "action.gripper_close"
                    ]
                case ActionEncoding.AGIBOT_DEXHAND:
                    return [
                        "action.agibot_dexhand"
                    ]
                case ActionEncoding.AGIBOT_GRIPPER:
                    return [
                        "action.agibot_gripper"
                    ]
                case ActionEncoding.GALAXEA:
                    return [
                        "action.galaxea"
                    ]
                case ActionEncoding.HUMANOID_EVERYDAY_G1:
                    return [
                        "action.humanoid_everyday_g1"
                    ]
                case ActionEncoding.HUMANOID_EVERYDAY_H1:
                    return [
                        "action.humanoid_everyday_h1"
                    ]
                case ActionEncoding.ACTION_NET:
                    return [
                        "action.action_net"
                    ]
                case ActionEncoding.NEURAL_GR1:
                    return [
                        "action.neural_gr1"
                    ]
                case _:
                    raise ValueError(f"Unknown ActionEncoding: {action}")
except ImportError:
    print("OXE RLDS configs not found. Make sure gr00t is installed with the 'oxe' extra.")
    pass

###########################################################################################

class RobocasaV1FLAREDataConfig(BaseDataConfig):
    video_keys = [
        "video.left_view",
        "video.right_view",
        "video.wrist_view",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0, 15]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.end_effector_position_relative": "min_max",
        "state.end_effector_rotation_relative": "min_max",
        "state.gripper_qpos": "min_max",
        "state.base_position": "min_max",
        "state.base_rotation": "min_max",
    }
    state_target_rotations = {
        "state.end_effector_rotation_relative": "rotation_6d",
        "state.base_rotation": "rotation_6d",
    }
    action_normalization_modes = {
        "action.end_effector_position": "q99",
        "action.end_effector_rotation": "q99",
        "action.gripper_close": "binary",
    }

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

class RobocasaV2FLAREDataConfig(BaseDataConfig):
    video_keys = [
        "video.left_view",
        "video.right_view",
        "video.wrist_view",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0, 15]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.end_effector_position_relative": "min_max",
        "state.end_effector_rotation_relative": "min_max",
        "state.gripper_qpos": "min_max",
        "state.base_position": "min_max",
        "state.base_rotation": "min_max",
    }
    state_target_rotations = {
        "state.end_effector_rotation_relative": "rotation_6d",
        "state.base_rotation": "rotation_6d",
    }
    #action_normalization_modes = {
    #    "action.end_effector_position": "min_max",
    #    "action.end_effector_rotation": "min_max",
    #    "action.gripper_close": "binary",
    #}

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.gripper_close": "binary",
                    "action.end_effector_position": "min_max",
                },
                target_rotations={"action.end_effector_rotation": "axis_angle"},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

class RobocasaV1FLAREEvalDataConfig(BaseDataConfig):
    video_keys = [
        "video.left_view",
        "video.right_view",
        "video.wrist_view",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.end_effector_position_relative": "min_max",
        "state.end_effector_rotation_relative": "min_max",
        "state.gripper_qpos": "min_max",
        "state.base_position": "min_max",
        "state.base_rotation": "min_max",
    }
    state_target_rotations = {
        "state.end_effector_rotation_relative": "rotation_6d",
        "state.base_rotation": "rotation_6d",
    }
    action_normalization_modes = {
        "action.end_effector_position": "q99",
        "action.end_effector_rotation": "q99",
        "action.gripper_close": "binary",
    }

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TInferTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

class RobocasaV2FLAREEvalDataConfig(BaseDataConfig):
    video_keys = [
        "video.left_view",
        "video.right_view",
        "video.wrist_view",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.end_effector_position_relative": "min_max",
        "state.end_effector_rotation_relative": "min_max",
        "state.gripper_qpos": "min_max",
        "state.base_position": "min_max",
        "state.base_rotation": "min_max",
    }
    state_target_rotations = {
        "state.end_effector_rotation_relative": "rotation_6d",
        "state.base_rotation": "rotation_6d",
    }
    #action_normalization_modes = {
    #    "action.end_effector_position": "min_max",
    #    "action.end_effector_rotation": "min_max",
    #    "action.gripper_close": "binary",
    #}

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.gripper_close": "binary",
                    "action.end_effector_position": "min_max",
                    "action.base_motion": "min_max",
                    "action.control_mode": "binary",
                },
                target_rotations={"action.end_effector_rotation": "axis_angle"},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TInferTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)



class RealFrankaJointDataConfig(BimanualPandaGripperDataConfig):
    video_keys = [
        "video.exterior_image_1_left",
        "video.wrist_image_left",
    ]
    state_keys = [
        "state.joint_pos_abs",
        "state.gripper_close",
    ]
    action_keys = [
        "action.joint_pos_abs",
        "action.gripper_close",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0, 15]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.joint_pos_abs": "min_max",
        "state.gripper_close": "binary",
    }
    action_normalization_modes = {
        "action.joint_pos_abs": "min_max",
        "action.gripper_close": "binary",
    }

    def transform(self, backbone_model_type="eagle", backbone_path=None):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                backbone_model_type=backbone_model_type,
                backbone_path=backbone_path,
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class RealOpenARMJointDataConfig(BimanualPandaGripperDataConfig):
    video_keys = [
        "video.camera_ego_left",
    ]
    state_keys = [
        "state.neck_joints",
        "state.left_arm_joints",
        "state.right_arm_joints",
        "state.left_hand_joints",
        "state.right_hand_joints",
    ]
    action_keys = [
        "action.neck_joints",
        "action.left_arm_joints",
        "action.right_arm_joints",
        "action.left_hand_joints",
        "action.right_hand_joints",
    ]

    language_keys = ["annotation.human.task_description"]
    observation_indices = [0, 15]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.neck_joints": "min_max",
        "state.left_arm_joints": "min_max",
        "state.right_arm_joints": "min_max",
        "state.left_hand_joints": "min_max",
        "state.right_hand_joints": "min_max",
    }
    action_normalization_modes = {
        "action.neck_joints": "min_max",
        "action.left_arm_joints": "min_max",
        "action.right_arm_joints": "min_max",
        "action.left_hand_joints": "min_max",
        "action.right_hand_joints": "min_max",
    }

    def transform(self, backbone_model_type="qwen3_vl_8b", backbone_path=None):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                backbone_model_type=backbone_model_type,
                backbone_path=backbone_path,
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class OpenARMTeleopJointRightDataConfig(BaseDataConfig):
    """openarm_teleop_v3 (openarm_rh56f1 joints) with the RIGHT ego camera, for the
    actlat tokenizer/VLA pipeline. Action/state: 28-dim joints — neck(2) +
    left/right arm(7) + left/right hand(6). min_max normalization.
    """
    video_keys = [
        "video.camera_ego_right",
    ]
    state_keys = [
        "state.neck_joints",
        "state.left_arm_joints",
        "state.right_arm_joints",
        "state.left_hand_joints",
        "state.right_hand_joints",
    ]
    action_keys = [
        "action.neck_joints",
        "action.left_arm_joints",
        "action.right_arm_joints",
        "action.left_hand_joints",
        "action.right_hand_joints",
    ]

    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
    action_dim = 28

    state_normalization_modes = {key: "min_max" for key in state_keys}
    action_normalization_modes = {key: "min_max" for key in action_keys}

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self, backbone_model_type="qwen3_vl_8b", backbone_path=None):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                backbone_model_type=backbone_model_type,
                backbone_path=backbone_path,
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=self.action_dim,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class OpenARMPnpEefRightDataConfig(BaseDataConfig):
    """RLWRLD openarm_inspire eef_inspire pick&place (e.g. pnp_clean_260506).

    Action/state: 30-dim eef format — left/right wrist_trans(3) + wrist_rot6d(6)
    + inspire6(6). Single RIGHT ego camera. min_max normalization.
    """
    video_keys = [
        "video.camera_ego_right",
    ]
    state_keys = [
        "state.left_wrist_trans",
        "state.left_wrist_rot6d",
        "state.left_inspire6",
        "state.right_wrist_trans",
        "state.right_wrist_rot6d",
        "state.right_inspire6",
    ]
    action_keys = [
        "action.left_wrist_trans",
        "action.left_wrist_rot6d",
        "action.left_inspire6",
        "action.right_wrist_trans",
        "action.right_wrist_rot6d",
        "action.right_inspire6",
    ]

    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
    action_dim = 30

    state_normalization_modes = {key: "min_max" for key in state_keys}
    action_normalization_modes = {key: "min_max" for key in action_keys}

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self, backbone_model_type="qwen3_vl_8b", backbone_path=None):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                backbone_model_type=backbone_model_type,
                backbone_path=backbone_path,
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=self.action_dim,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class DexmgSingleViewArmsHandsDataConfig(BaseDataConfig):
    """
    DexMimicGen 통합 DataConfig.
    
    데이터 형식:
    - State: EEF absolute pose (pos + quat) - NOT joint angles
    - Action: EEF pose (delta or absolute)
    
    → SinCosTransform 대신 min_max 사용
    """
    
    video_keys = ["video.agentview"]
    state_keys = [
        "state.left_arm",      # EEF pos(3) + quat(4) = 7 dims
        "state.right_arm",     # EEF pos(3) + quat(4) = 7 dims
        "state.left_hand",     # gripper qpos (12 dims: dual_panda=12, gr1=11→zero-padded to 12)
        "state.right_hand",    # gripper qpos (12 dims: dual_panda=12, gr1=11→zero-padded to 12)
    ]
    action_keys = [
        "action.left_arm",     # pos(3) + rot(3) = 6 dims
        "action.right_arm",    # pos(3) + rot(3) = 6 dims
        "action.left_hand",    # gripper action (6 dims)
        "action.right_hand",   # gripper action (6 dims)
    ]
    language_keys = ["annotation.task"]
    observation_indices = [0, 15]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

        
    def transform(self, backbone_model_type="eagle", backbone_path=None):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            
            # ⚠️ state transforms - min_max (NOT SinCos!)
            # EEF pose에는 SinCosTransform이 부적합
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            
            # action transforms - min_max (동일)
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                backbone_model_type=backbone_model_type,
                backbone_path=backbone_path,
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class AllexDataConfig(BaseDataConfig):
    video_keys = ["video.camera_ego_left"]
    state_keys = [
        "state.right_arm_joints",
        "state.left_arm_joints",
        "state.right_hand_joints",
        "state.left_hand_joints",
        "state.neck_joints",
        "state.waist_joints",
    ]
    action_keys = [
        "action.right_arm_joints",
        "action.left_arm_joints",
        "action.right_hand_joints",
        "action.left_hand_joints",
        "action.neck_joints",
        "action.waist_joints",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0, 15]
    action_indices = list(range(40))
    action_dim = 48

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "q99" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=self.action_dim,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class AllexRLWRldDataConfig(BaseDataConfig):
    video_keys = ["video.camera_ego_left", "video.camera_ego_right"]
    state_keys = [
        "state.right_arm_joints",
        "state.left_arm_joints",
        "state.right_hand_joints",
        "state.left_hand_joints",
        "state.neck_joints",
        "state.waist_joints",
    ]
    action_keys = [
        "action.right_arm_joints",
        "action.left_arm_joints",
        "action.right_hand_joints",
        "action.left_hand_joints",
        "action.neck_joints",
        "action.waist_joints",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0, 15]
    action_indices = list(range(40))
    action_dim = 48

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "q99" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=self.action_dim,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class DebugG0FrankaTeleopDataConfig(BimanualPandaGripperDataConfig):
    video_keys = [
        "video.exterior_image_1_left",
        "video.wrist_image_left",
    ]
    state_keys = [
        "state.joint_pos_abs",
        "state.gripper_close",
    ]
    action_keys = [
        "action.joint_pos_abs",
        "action.gripper_close",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0, 15]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.joint_pos_abs": "min_max",
        "state.gripper_close": "binary",
    }
    action_normalization_modes = {
        "action.joint_pos_abs": "min_max",
        "action.gripper_close": "binary",
    }

    def transform(self, backbone_model_type="eagle", backbone_path=None):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                backbone_model_type=backbone_model_type,
                backbone_path=backbone_path,
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class DexJoCoSingleArmDataConfig:
    """DexJoCo single-arm dexterous-hand config (e.g. hammer_nail, water_plant).

    LeRobot layout: observation.state (23) = [pos3, quat4, hand16];
    action (22) = [pos3, rotvec3, hand16]; cameras = front, wrist. No gripper /
    discrete action dims (the 16-DoF hand is continuous). Fetches a 64-step action
    sequence (multi-horizon) so the MoE action head can compute compressed losses;
    the baseline head simply uses the first 16. State quaternion -> rotation_6d.
    """
    video_keys = ["video.front", "video.wrist"]
    state_keys = ["state.arm_pos", "state.arm_rot", "state.hand"]
    action_keys = ["action.arm_pos", "action.arm_rot", "action.hand"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Camera fed to the V4 action-latent tokenizer for its (frame_x0, frame_x1)
    # latent target during VLA training. The tokenizer is trained single-camera
    # (dexjoco_single_arm_front -> video.front), so its latent-target input must
    # stay that one camera even though the VLA backbone consumes all of
    # `video_keys`. gr00t_finetune_actlat_fm.py reads this; if unset it falls
    # back to video_keys[0].
    tokenizer_frame_video_key = "video.front"

    state_normalization_modes = {
        "state.arm_pos": "min_max",
        "state.arm_rot": "min_max",
        "state.hand": "min_max",
    }
    state_target_rotations = {"state.arm_rot": "rotation_6d"}
    action_normalization_modes = {
        "action.arm_pos": "min_max",
        "action.arm_rot": "min_max",
        "action.hand": "min_max",
    }

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            
            # model-specific transform
            # max_action_dim=22 (single-arm 실제 action_dim: pos3+rotvec3+hand16)
            # → 토크나이저(action_dim=22) ↔ VLA encode/decode 정합. 기본 32로 두면
            # VLA 파이프라인이 22→32 패딩하여 tokenizer.action_proj(22→256)와 mismatch.
            GR00TInferTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=22,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class DexJoCoSingleArmFrontDataConfig(DexJoCoSingleArmDataConfig):
    """Single-camera (front) variant of dexjoco_single_arm for the V4 action-latent
    tokenizer, which is hard-pinned to one camera (dataset_action_frames_v4.py
    asserts len(video_keys) == 1). State/action/language keys are inherited
    unchanged; only the video modality is narrowed to video.front. The Stage-2
    VLA training keeps using the full 2-camera dexjoco_single_arm config."""
    video_keys = ["video.front"]


class DexJoCoDualArmDataConfig:
    """DexJoCo dual-arm dexterous-hand config (e.g. hammer_nail, water_plant).

    LeRobot layout: observation.state (23) = [pos3, quat4, hand16];
    action (22) = [pos3, rotvec3, hand16]; cameras = front, wrist. No gripper /
    discrete action dims (the 16-DoF hand is continuous). Fetches a 64-step action
    sequence (multi-horizon) so the MoE action head can compute compressed losses;
    the baseline head simply uses the first 16. State quaternion -> rotation_6d.
    """
    video_keys = ["video.front", "video.wrist_left", "video.wrist_right"]
    state_keys = ["state.right_arm_pos", "state.right_arm_rot", "state.right_hand", "state.left_arm_pos", "state.left_arm_rot", "state.left_hand"]
    action_keys = ["action.right_arm_pos", "action.right_arm_rot", "action.right_hand", "action.left_arm_pos", "action.left_arm_rot", "action.left_hand"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Camera fed to the V4 action-latent tokenizer for its (frame_x0, frame_x1)
    # latent target during VLA training. The tokenizer is trained single-camera
    # (dexjoco_dual_arm_front -> video.front), so its latent-target input must
    # stay that one camera even though the VLA backbone consumes all of
    # `video_keys`. gr00t_finetune_actlat_fm.py reads this; if unset it falls
    # back to video_keys[0].
    tokenizer_frame_video_key = "video.front"

    state_normalization_modes = {
        "state.right_arm_pos": "min_max",
        "state.right_arm_rot": "min_max",
        "state.right_hand": "min_max",
        "state.left_arm_pos": "min_max",
        "state.left_arm_rot": "min_max",
        "state.left_hand": "min_max",
    }
    state_target_rotations = {"state.right_arm_rot": "rotation_6d", "state.left_arm_rot": "rotation_6d"}
    action_normalization_modes = {
        "action.right_arm_pos": "min_max",
        "action.right_arm_rot": "min_max",
        "action.right_hand": "min_max",
        "action.left_arm_pos": "min_max",
        "action.left_arm_rot": "min_max",
        "action.left_hand": "min_max",
    }

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TInferTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=44,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class DexJoCoDualArmFrontDataConfig(DexJoCoDualArmDataConfig):
    """Single-camera (front) variant of dexjoco_dual_arm for the V4 action-latent
    tokenizer, which is hard-pinned to one camera (dataset_action_frames_v4.py
    asserts len(video_keys) == 1). State/action/language keys are inherited
    unchanged; only the video modality is narrowed to video.front. The Stage-2
    VLA training keeps using the full 3-camera dexjoco_dual_arm config."""
    video_keys = ["video.front"]


# === DexJoCo single-arm, action_horizon=24 ===
# Identical to DexJoCoSingleArmDataConfig except the action chunk is 24 steps
# instead of 16. action_horizon is fully data-driven (len(action_indices)):
#   - Stage-1 tokenizer reads it from the action sample shape and builds 24 tokens.
#   - Stage-2 VLA rebuilds the tokenizer from ckpt shapes and recreates the action
#     head with len(action_indices) — so both stages stay consistent at 24.
# Keep this as a separate class so existing 16-step dexjoco checkpoints/runs are
# untouched (a 16-trained tokenizer cannot be mixed with a 24-horizon VLA).
class DexJoCoSingleArmH24DataConfig(DexJoCoSingleArmDataConfig):
    action_indices = list(range(24))


class DexJoCoSingleArmFrontH24DataConfig(DexJoCoSingleArmH24DataConfig):
    """Single-camera (front) variant of dexjoco_single_arm_h24 for the V4
    action-latent tokenizer (asserts len(video_keys) == 1). Inherits the 24-step
    action_indices from DexJoCoSingleArmH24DataConfig; only narrows the video
    modality to video.front."""
    video_keys = ["video.front"]

class Gr1ActionnetDataConfig(BaseDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=29,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class EgoDexCameraHandUnitDataConfig:
    """DexJoCo dual-arm dexterous-hand config (e.g. hammer_nail, water_plant).

    LeRobot layout: observation.state (23) = [pos3, quat4, hand16];
    action (22) = [pos3, rotvec3, hand16]; cameras = front, wrist. No gripper /
    discrete action dims (the 16-DoF hand is continuous). Fetches a 64-step action
    sequence (multi-horizon) so the MoE action head can compute compressed losses;
    the baseline head simply uses the first 16. State quaternion -> rotation_6d.
    """
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.camera_pos",
        "state.camera_rot",
        "state.rightHand_pos",
        "state.rightHand_rot",
        "state.rightThumbTip_pos",
        "state.rightIndexFingerTip_pos",
        "state.rightMiddleFingerTip_pos",
        "state.rightRingFingerTip_pos",
        "state.rightLittleFingerTip_pos",
        "state.leftHand_pos",
        "state.leftHand_rot",
        "state.leftThumbTip_pos",
        "state.leftIndexFingerTip_pos",
        "state.leftMiddleFingerTip_pos",
        "state.leftRingFingerTip_pos",
        "state.leftLittleFingerTip_pos",
    ]
    action_keys = [
        "action.camera_pos",
        "action.camera_rot",
        "action.rightHand_pos",
        "action.rightHand_rot",
        "action.rightThumbTip_pos",
        "action.rightIndexFingerTip_pos",
        "action.rightMiddleFingerTip_pos",
        "action.rightRingFingerTip_pos",
        "action.rightLittleFingerTip_pos",
        "action.leftHand_pos",
        "action.leftHand_rot",
        "action.leftThumbTip_pos",
        "action.leftIndexFingerTip_pos",
        "action.leftMiddleFingerTip_pos",
        "action.leftRingFingerTip_pos",
        "action.leftLittleFingerTip_pos",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Camera fed to the V4 action-latent tokenizer for its (frame_x0, frame_x1)
    # latent target during VLA training. The tokenizer is trained single-camera
    # (dexjoco_dual_arm_front -> video.front), so its latent-target input must
    # stay that one camera even though the VLA backbone consumes all of
    # `video_keys`. gr00t_finetune_actlat_fm.py reads this; if unset it falls
    # back to video_keys[0].
    tokenizer_frame_video_key = "video.ego_view"

    # Every state/action key is normalized with min_max. Built from the actual
    # key lists so all dims are covered (the leftmost iterable of a class-body
    # comprehension is evaluated in the enclosing scope, so state_keys /
    # action_keys are visible here).
    state_normalization_modes = {k: "min_max" for k in state_keys}
    action_normalization_modes = {k: "min_max" for k in action_keys}

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TInferTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=57,
                max_action_dim=57,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class HumanoidEverydayG1DataConfig:

    video_keys = ["video.egocentric_resized"]
    state_keys = [
        "state.left_arm",
        "state.left_hand",
        "state.right_arm",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.left_hand",
        "action.right_arm",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Camera fed to the V4 action-latent tokenizer for its (frame_x0, frame_x1)
    # latent target during VLA training. The tokenizer is trained single-camera
    # (dexjoco_dual_arm_front -> video.front), so its latent-target input must
    # stay that one camera even though the VLA backbone consumes all of
    # `video_keys`. gr00t_finetune_actlat_fm.py reads this; if unset it falls
    # back to video_keys[0].
    tokenizer_frame_video_key = "video.egocentric_resized"

    # Every state/action key is normalized with min_max. Built from the actual
    # key lists so all dims are covered (the leftmost iterable of a class-body
    # comprehension is evaluated in the enclosing scope, so state_keys /
    # action_keys are visible here).
    state_normalization_modes = {k: "min_max" for k in state_keys}
    action_normalization_modes = {k: "min_max" for k in action_keys}

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TInferTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=28,
                max_action_dim=28,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

class HumanoidEverydayH1DataConfig:

    video_keys = ["video.egocentric_resized"]
    state_keys = [
        "state.left_arm",
        "state.left_hand",
        "state.right_arm",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.left_hand",
        "action.right_arm",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    # Camera fed to the V4 action-latent tokenizer for its (frame_x0, frame_x1)
    # latent target during VLA training. The tokenizer is trained single-camera
    # (dexjoco_dual_arm_front -> video.front), so its latent-target input must
    # stay that one camera even though the VLA backbone consumes all of
    # `video_keys`. gr00t_finetune_actlat_fm.py reads this; if unset it falls
    # back to video_keys[0].
    tokenizer_frame_video_key = "video.egocentric_resized"

    # Every state/action key is normalized with min_max. Built from the actual
    # key lists so all dims are covered (the leftmost iterable of a class-body
    # comprehension is evaluated in the enclosing scope, so state_keys /
    # action_keys are visible here).
    state_normalization_modes = {k: "min_max" for k in state_keys}
    action_normalization_modes = {k: "min_max" for k in action_keys}

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TInferTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=26,
                max_action_dim=26,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class HumanoidEverydayG1EgocentricDataConfig(HumanoidEverydayG1DataConfig):
    """G1 humanoid_everyday variant for datasets whose only egocentric video
    stream is named ``egocentric`` (not the precomputed ``egocentric_resized``
    downscale present on the origin cluster). The V4 tokenizer resizes to 224
    regardless, so reading the full-res ``egocentric`` is functionally
    identical. Only the video key names change."""

    video_keys = ["video.egocentric"]
    tokenizer_frame_video_key = "video.egocentric"


class HumanoidEverydayH1EgocentricDataConfig(HumanoidEverydayH1DataConfig):
    """H1 counterpart of HumanoidEverydayG1EgocentricDataConfig."""

    video_keys = ["video.egocentric"]
    tokenizer_frame_video_key = "video.egocentric"


DATA_CONFIG_MAP = {
    "fourier_gr1_arms_waist_actlat_fm": FourierGr1ArmsWaistActlatFMDataConfig(),
    "fourier_gr1_arms_waist_actlat_fm_1000demos": FourierGr1ArmsWaistActlatFM1000DemosDataConfig(),
    "fourier_gr1_arms_waist": FourierGr1ArmsWaistDataConfig(),
    "fourier_gr1_arms_only": FourierGr1ArmsOnlyDataConfig(),
    "fourier_gr1_full_upper_body": FourierGr1FullUpperBodyDataConfig(),
    "bimanual_panda_gripper": BimanualPandaGripperDataConfig(),
    "bimanual_panda_hand": BimanualPandaHandDataConfig(),
    "single_panda_gripper": SinglePandaGripperDataConfig(),
    "single_panda_gripper_front": SinglePandaGripperFrontDataConfig(),
    "single_panda_gripper_actlat_fm": SinglePandaGripperActlatFMDataConfig(),
    "so100": So100DataConfig(),
    "so100_dualcam": So100DualCamDataConfig(),
    "unitree_g1": UnitreeG1DataConfig(),
    "unitree_g1_full_body": UnitreeG1FullBodyDataConfig(),
    "oxe_droid": OxeDroidDataConfig(),
    "agibot_genie1": AgibotGenie1DataConfig(),
    "libero": LiberoDataConfig(),
    "libero_finetune": LiberoDataConfigFinetune(),
    "libero_flare": LiberoFLAREDataConfig(),
    "single_panda_gripper_flare": SinglePandaGripperFLAREDataConfig(),
    "fourier_gr1_arms_waist_flare": FourierGr1ArmsWaistFLAREDataConfig(),
    "bridge": BridgeDataConfig(),
    "bridge_flare": BridgeFLAREDataConfig(),
    "real_droid_joint_flare": RealDroidJointFLAREDataConfig(),
    "real_droid_cartesian_flare": RealDroidCartesianFLAREDataConfig(),
    "real_tactile_droid_joint": RealTactileDroidJointDataConfig(),
    "bridge_flare_kty": BridgeFlareKTYDataConfig(),
    "bridge_flare_kty_actlat_fm": BridgeFlareKTYActlatFMDataConfig(),
    "robotwin_actlat_fm": RoboTwinActlatFMDataConfig(),
    "calvin": CalvinDataConfig(),
    "robocasa_v1_flare": RobocasaV1FLAREDataConfig(),
    "robocasa_v2_flare": RobocasaV2FLAREDataConfig(),
    "robocasa_v1_flare_eval": RobocasaV1FLAREEvalDataConfig(),
    "robocasa_v2_flare_eval": RobocasaV2FLAREEvalDataConfig(),
    "real_franka_joint": RealFrankaJointDataConfig(),
    "real_franka_eef": FrankaRealEEFDataConfig(),
    "real_open_arm_joint": RealOpenARMJointDataConfig(),
    "openarm_teleop_joint_right": OpenARMTeleopJointRightDataConfig(),
    "openarm_pnp_eef_right": OpenARMPnpEefRightDataConfig(),
    "dexmg_single_view_arms_hands": DexmgSingleViewArmsHandsDataConfig(),
    "allex": AllexDataConfig(),
    "allex_rlwrld": AllexRLWRldDataConfig(),
    "debug_G0_franka_teleop": DebugG0FrankaTeleopDataConfig(),
    "dexjoco_single_arm": DexJoCoSingleArmDataConfig(),
    "dexjoco_single_arm_front": DexJoCoSingleArmFrontDataConfig(),
    "dexjoco_single_arm_h24": DexJoCoSingleArmH24DataConfig(),
    "dexjoco_single_arm_front_h24": DexJoCoSingleArmFrontH24DataConfig(),
    "dexjoco_dual_arm": DexJoCoDualArmDataConfig(),
    "dexjoco_dual_arm_front": DexJoCoDualArmFrontDataConfig(),
    "gr1_actionnet": Gr1ActionnetDataConfig(),
    "human_egodex_camera_hand_unit": EgoDexCameraHandUnitDataConfig(),
    "humanoid_everyday_g1": HumanoidEverydayG1DataConfig(),
    "humanoid_everyday_h1": HumanoidEverydayH1DataConfig(),
    "humanoid_everyday_g1_egocentric": HumanoidEverydayG1EgocentricDataConfig(),
    "humanoid_everyday_h1_egocentric": HumanoidEverydayH1EgocentricDataConfig(),
}

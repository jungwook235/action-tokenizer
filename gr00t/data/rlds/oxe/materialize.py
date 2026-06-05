"""
materialize.py

Factory class for initializing Open-X Embodiment dataset kwargs and other parameters; provides and exports functions for
clear control flow.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from gr00t.data.rlds.oxe.configs import OXE_DATASET_CONFIGS, ActionEncoding
from gr00t.data.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS

import logging


def make_oxe_dataset_kwargs(
    dataset_name: str,
    data_root_dir: Path,
    load_camera_views: Tuple[str] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    # action_proprio_normalization_type = ACTION_PROPRIO_NORMALIZATION_TYPE, # do at ComposedModalityTransform
) -> Dict[str, Any]:
    """Generates config (kwargs) for given dataset from Open-X Embodiment."""
    configs_name = dataset_name
    if "agibot_gripper" in configs_name:
        configs_name = "agibot_gripper"
    elif "galaxea" in configs_name:
        configs_name = "galaxea"
    elif "neural_gr1" in configs_name:
        configs_name = "neural_gr1"
    dataset_kwargs = deepcopy(OXE_DATASET_CONFIGS[configs_name])

    available_encodings = [
        ActionEncoding.EEF_POS, 
        ActionEncoding.EEF_R6, 
        ActionEncoding.JOINT_POS_BIMANUAL, 
        ActionEncoding.AGIBOT_DEXHAND,
        ActionEncoding.AGIBOT_GRIPPER,
        ActionEncoding.GALAXEA,
        ActionEncoding.HUMANOID_EVERYDAY_G1,
        ActionEncoding.HUMANOID_EVERYDAY_H1,
        ActionEncoding.ACTION_NET,
        ActionEncoding.NEURAL_GR1,
    ]

    if dataset_kwargs["action_encoding"] not in available_encodings:
        raise ValueError(f"Cannot load `{dataset_name}`; ActionEncoding not supported!")

    # [Contract] For EEF_POS & EEF_R6 actions, only the last action dimension (gripper) is absolute!
    # Normalize all action dimensions *except* the gripper
    if dataset_kwargs["action_encoding"] is ActionEncoding.EEF_POS:
        dataset_kwargs["absolute_action_mask"] = [False] * 6 + [True]
        dataset_kwargs["action_normalization_mask"] = [True] * 6 + [False]
    elif dataset_kwargs["action_encoding"] is ActionEncoding.EEF_R6:
        dataset_kwargs["absolute_action_mask"] = [False] * 9 + [True]
        dataset_kwargs["action_normalization_mask"] = [True] * 9 + [False]
    elif dataset_kwargs["action_encoding"] is ActionEncoding.JOINT_POS_BIMANUAL:
        dataset_kwargs["absolute_action_mask"] = [True] * 14
        dataset_kwargs["action_normalization_mask"] = [True] * 14
    elif dataset_kwargs["action_encoding"] is ActionEncoding.AGIBOT_DEXHAND:
        dataset_kwargs["absolute_action_mask"] = [True] * 44
        dataset_kwargs["action_normalization_mask"] = [True] * 44
    elif dataset_kwargs["action_encoding"] is ActionEncoding.AGIBOT_GRIPPER:
        dataset_kwargs["absolute_action_mask"] = [True] * 34
        dataset_kwargs["action_normalization_mask"] = [True] * 34
    elif dataset_kwargs["action_encoding"] is ActionEncoding.GALAXEA:
        dataset_kwargs["absolute_action_mask"] = [True] * 26
        dataset_kwargs["action_normalization_mask"] = [True] * 26
    elif dataset_kwargs["action_encoding"] is ActionEncoding.HUMANOID_EVERYDAY_G1:
        dataset_kwargs["absolute_action_mask"] = [True] * 28
        dataset_kwargs["action_normalization_mask"] = [True] * 28
    elif dataset_kwargs["action_encoding"] is ActionEncoding.HUMANOID_EVERYDAY_H1:
        dataset_kwargs["absolute_action_mask"] = [True] * 26
        dataset_kwargs["action_normalization_mask"] = [True] * 26
    elif dataset_kwargs["action_encoding"] is ActionEncoding.ACTION_NET:
        dataset_kwargs["absolute_action_mask"] = [True] * 44
        dataset_kwargs["action_normalization_mask"] = [True] * 44
    elif dataset_kwargs["action_encoding"] is ActionEncoding.NEURAL_GR1:
        dataset_kwargs["absolute_action_mask"] = [True] * 44
        dataset_kwargs["action_normalization_mask"] = [True] * 44
    # dataset_kwargs["action_proprio_normalization_type"] = action_proprio_normalization_type

    # Adjust Loaded Camera Views
    if len(missing_keys := (set(load_camera_views) - set(dataset_kwargs["image_obs_keys"]))) > 0:
        raise ValueError(f"Cannot load `{dataset_name}`; missing camera views `{missing_keys}`")

    # Filter
    dataset_kwargs["image_obs_keys"] = {
        k: v for k, v in dataset_kwargs["image_obs_keys"].items() if k in load_camera_views
    }
    dataset_kwargs["depth_obs_keys"] = {
        k: v for k, v in dataset_kwargs["depth_obs_keys"].items() if k in load_camera_views
    }

    # Eliminate Unnecessary Keys
    dataset_kwargs.pop("state_encoding")
    dataset_kwargs.pop("action_encoding")
    dataset_kwargs.pop("control_frequency", None)
    if not load_depth:
        dataset_kwargs.pop("depth_obs_keys")
    if not load_proprio:
        dataset_kwargs.pop("state_obs_keys")

    # Load Language
    if load_language:
        dataset_kwargs["language_key"] = "language_instruction"

    # Specify Standardization Transform
    standardize_fn_name = dataset_name
    if "agibot_gripper" in standardize_fn_name:
        standardize_fn_name = "agibot_gripper"
    elif "galaxea" in standardize_fn_name:
        standardize_fn_name = "galaxea"
    elif "neural_gr1" in standardize_fn_name:
        standardize_fn_name = "neural_gr1"
    dataset_kwargs["standardize_fn"] = OXE_STANDARDIZATION_TRANSFORMS[standardize_fn_name]

    # Add any aux arguments
    if "aux_kwargs" in dataset_kwargs:
        dataset_kwargs.update(dataset_kwargs.pop("aux_kwargs"))

    return {"name": dataset_name, "data_dir": str(data_root_dir), **dataset_kwargs}


def get_oxe_dataset_kwargs_and_weights(
    data_root_dir: Path,
    mixture_spec: List[Tuple[str, float]],
    load_camera_views: Tuple[str] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    # action_proprio_normalization_type = ACTION_PROPRIO_NORMALIZATION_TYPE,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Generates dataset kwargs for a given dataset mix from the Open X-Embodiment dataset. The returned kwargs
    (per-dataset configs) and weights can be passed directly to `make_interleaved_dataset`.

    :param data_root_dir: Base directory containing RLDS/TFDS-formatted datasets (from Open-X)
    :param mixture_spec: List of (dataset_name, sampling_weight) from `oxe.mixtures.OXE_NAMED_MIXTURES`
    :param load_camera_views: Camera views to load; see `oxe.dataset_configs.py` for available views.
    :param load_depth: Load depth information in addition to camera RGB.
    :param load_proprio: Load proprioceptive state.
    :param load_language: Load language instructions.
    :param action_proprio_normalization_type: Normalization scheme to use for proprioceptive actions.

    return: Tuple of (per_dataset_kwargs, sampling_weights)
    """
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight in mixture_spec:
        if d_name in included_datasets:
            logging.warning(f"Skipping Duplicate Dataset: `{(d_name, d_weight)}`")
            continue

        included_datasets.add(d_name)
        filtered_mixture_spec.append((d_name, d_weight))

    # Assemble Dataset Config (kwargs) and Weights
    per_dataset_kwargs, sampling_weights = [], []
    for d_name, d_weight in filtered_mixture_spec:
        try:
            per_dataset_kwargs.append(
                make_oxe_dataset_kwargs(
                    d_name,
                    data_root_dir,
                    load_camera_views,
                    load_depth,
                    load_proprio,
                    load_language,
                    # action_proprio_normalization_type,
                )
            )
            sampling_weights.append(d_weight)

        except ValueError as e:
            logging.warning(f"Skipping `{d_name}` due to Error: {e}")

    return per_dataset_kwargs, sampling_weights

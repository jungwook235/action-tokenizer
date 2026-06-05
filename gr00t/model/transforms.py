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

import os
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from tree import map_structure_with_path

import numpy as np
import torch
import tree
from einops import rearrange
from PIL import Image
from pydantic import Field, PrivateAttr
from transformers import AutoProcessor, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import InvertibleModalityTransform
from gr00t.utils.dist import rank_zero_print as _print
#from gr00t.utils.qwen_vision_process import process_vision_info as qwen_process_vision_info

from .backbone.eagle_backbone import DEFAULT_EAGLE_PATH
#from .backbone.qwen_3_vl_backbone import DEFAULT_QWEN_3_VL_PATH


def formalize_language(language: str) -> str:
    """
    1. Force lowercase
    2. Remove all punctuations
    """
    language = language.lower()
    language = re.sub(r"[^\w\s]", "", language)
    return language


def build_eagle_processor(eagle_path: str) -> ProcessorMixin:
    eagle_processor = AutoProcessor.from_pretrained(
        eagle_path, trust_remote_code=True, use_fast=True
    )
    eagle_processor.tokenizer.padding_side = "left"
    return eagle_processor


def build_qwen_processor(qwen_path: str) -> ProcessorMixin:
    qwen_processor = AutoProcessor.from_pretrained(
        qwen_path, trust_remote_code=True, use_fast=True, min_pixels=256 * 28 * 28, max_pixels=672 * 28 * 28
    )
    qwen_processor.tokenizer.padding_side = "left"
    return qwen_processor


def collate_infer(features: List[dict], eagle_processor) -> dict:
    batch = {}
    keys = features[0].keys()

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "eagle_content":
            text_list = []
            image_inputs = []
            for v in values:
                curr_text_list = v["text_list"]
                curr_image_inputs = v["image_inputs"]
                text_list += curr_text_list
                image_inputs += curr_image_inputs

            eagle_inputs_1 = eagle_processor(
                text=text_list, images=image_inputs, return_tensors="pt", padding=True
            )

            for k, v in eagle_inputs_1.items():
                k = "eagle_" + k
                batch[k] = v

        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
    return batch



def qwen_3_vl_collate(features: List[dict], qwen_processor) -> dict:
    batch = {}
    keys = features[0].keys()

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "qwen_content":
            text_list = []
            image_inputs = []
            video_inputs = []
            for v in values:
                curr_text_list = v["text_list"]
                curr_image_inputs = v["image_inputs"]
                curr_video_inputs = v["video_inputs"]
                text_list += curr_text_list
                if curr_image_inputs is not None:
                    image_inputs += curr_image_inputs
                if curr_video_inputs is not None:
                    video_inputs += curr_video_inputs

            # Pass both images and videos to qwen processor
            qwen_inputs = qwen_processor(
                text=text_list,
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                return_tensors="pt",
                padding=True,
            )

            for k, v in qwen_inputs.items():
                if k == "attention_mask" or k == "input_ids":
                    batch[k] = v
                else:
                    k = "qwen_" + k
                    batch[k] = v

        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


def collate_oxe(features: List[dict], eagle_processor) -> dict:
    """
    Collate for OXE/RLDS format: 3 images per sample (num_views=3).
    Passes all images to eagle_processor once, then duplicates for eagle2_* to maintain format.
    """
    batch = {}
    keys = features[0].keys()
    if not getattr(collate_oxe, "_debug_keys_logged", False):
        _print(f"[collate_oxe] keys to process: {list(keys)}")
        collate_oxe._debug_keys_logged = True

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "eagle_content":
            text_list = []
            image_inputs = []
            for v in values:
                curr_text_list = v["text_list"]
                curr_image_inputs = v["image_inputs"]
                text_list += curr_text_list
                image_inputs += curr_image_inputs

            # OXE: 3 images per sample, text has <image-1><image-2><image-3>
            # Single eagle call with all images, then duplicate for eagle2
            eagle_inputs_1 = eagle_processor(
                text=text_list, images=image_inputs, return_tensors="pt", padding=True
            )
            eagle_inputs_2 = eagle_inputs_1  # Duplicate for format compatibility (eagle2 unused)

            for k, v in eagle_inputs_1.items():
                batch["eagle_" + k] = v
            for k, v in eagle_inputs_2.items():
                batch["eagle2_" + k] = v

            batch["raw_eagle_images_1"] = image_inputs
            batch["raw_eagle_images_2"] = image_inputs  # Same for format compatibility

        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            batch[key] = torch.cat(values)
        elif key == "dataset_name":
            # dataset_name is list of strings per sample, cannot np.stack; flatten to list
            batch[key] = [v[0] if isinstance(v, list) else v for v in values]
        else:
            try:
                batch[key] = torch.from_numpy(np.stack(values))
            except ValueError as e:
                shapes = [getattr(v, "shape", None) for v in values]
                _print(f"[collate_oxe] np.stack failed for key={key!r}: {e}")
                _print(f"[collate_oxe] len(values)={len(values)}, dtype={[getattr(v, 'dtype', type(v).__name__) for v in values]}")
                for i, (v, s) in enumerate(zip(values, shapes)):
                    _print(f"[collate_oxe]   sample[{i}]: shape={s}")
                raise
    return batch


def collate(features: List[dict], eagle_processor) -> dict:
    # Delegate to collate_oxe when explicit OXE flag is set (RLDS format)
    if "eagle_content" in features[0]:
        ec0 = features[0]["eagle_content"]
        if ec0.get("use_oxe_collate"):
            return collate_oxe(features, eagle_processor)

    batch = {}
    keys = features[0].keys()

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "eagle_content":
            text_list = []
            image_inputs = []
            for v in values:
                curr_text_list = v["text_list"]
                curr_image_inputs = v["image_inputs"]
                text_list += curr_text_list
                image_inputs += curr_image_inputs

            #########################################################
            if '<image-6>' in text_list[0]: ###True
                #print("image-6 found")
                text_list = [s.replace('<image-4><image-5><image-6>', '') for s in text_list]
                image_inputs_1 = [x for i, x in enumerate(image_inputs) if i % 6 in (0, 1, 2)]
                image_inputs_2 = [x for i, x in enumerate(image_inputs) if i % 6 in (3, 4, 5)]
                image_count = 3
                """
                # Debug: Save first image from image_inputs_1
                if len(image_inputs_1) > 0:
                    debug_dir = "/sjw_alinlab1/home/jungwook/Isaac-GR00T/vggt_debug"
                    os.makedirs(debug_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img = image_inputs_1[0]
                    if isinstance(img, Image.Image):
                        img.save(os.path.join(debug_dir, f"{timestamp}_collate_image.png"))
                        print(f"[collate] Saved image to {debug_dir}/{timestamp}_collate_image.png")
                print("--------------------------------")
                print("--------------------------------")
                exit()
                """
            elif '<image-4>' in text_list[0]:
                #print("image-4 found")
                text_list = [s.replace('<image-3><image-4>', '') for s in text_list]
                image_inputs_1 = [x for i, x in enumerate(image_inputs) if i % 4 in (0, 1)]
                image_inputs_2 = [x for i, x in enumerate(image_inputs) if i % 4 in (2, 3)]
                image_count = 2
            elif '<image-3>' in text_list[0]:
                #print("image-3 found")
                image_inputs_1 = image_inputs
                image_inputs_2 = None

                image_count = 3
            else:
                text_list = [s.replace('<image-2>', '') for s in text_list]
                image_inputs_1 = [x for i, x in enumerate(image_inputs) if i % 2 == 0]
                image_inputs_2 = [x for i, x in enumerate(image_inputs) if i % 2 == 1]
                #image_inputs_1 = image_inputs
                #image_inputs_2 = None
                image_count = 2

            #########################################################

            eagle_inputs_1 = eagle_processor(
                text=text_list, images=image_inputs_1, return_tensors="pt", padding=True
            )

            #########################################################
            try:
                if image_inputs_2 is not None:
                    eagle_inputs_2 = eagle_processor(
                        text=text_list, images=image_inputs_2, return_tensors="pt", padding=True
                    )
            except Exception as e:
                print("Exception accessed {}".format(e))
                if len(image_inputs_2) != len(text_list) * image_count:
                    text_list = text_list[:len(image_inputs_2) // image_count]
                eagle_inputs_2 = eagle_processor(
                    text=text_list, images=image_inputs_2, return_tensors="pt", padding=True
                )
            #########################################################

            for k, v in eagle_inputs_1.items():
                k = "eagle_" + k
                batch[k] = v

            #########################################################
            if image_inputs_2 is not None:
                for k, v in eagle_inputs_2.items():
                    k = "eagle2_" + k
                    batch[k] = v
            #########################################################
            
            # Add raw (unprocessed) PIL Images to batch for VGGT
            # Pass PIL Image lists directly (no tensor conversion)
            if len(image_inputs_1) > 0:
                batch["raw_eagle_images_1"] = image_inputs_1  # List[PIL.Image]
            if image_inputs_2 is not None and len(image_inputs_2) > 0:
                batch["raw_eagle_images_2"] = image_inputs_2  # List[PIL.Image]
            #print(f"transformers.py can change!!!!!!!!  {len(image_inputs_1)} {len(image_inputs_2)} {image_count}")
            #########################################################

        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


class DefaultDataCollator(DataCollatorMixin):
    def __init__(self, eagle_path: str = DEFAULT_EAGLE_PATH):
        super().__init__()
        self.eagle_path = eagle_path
        self.eagle_processor = build_eagle_processor(eagle_path)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = collate(features, self.eagle_processor)

        return result


class GR00TTransform(InvertibleModalityTransform):

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    formalize_language: bool = Field(default=False, description="Formalize language if True.")
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[list[str]] = PrivateAttr(default=None)

    eagle_processor: ProcessorMixin = Field(default=build_eagle_processor(DEFAULT_EAGLE_PATH))

    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            if "annotation" in key:
                modality = "language"
            else:
                try:
                    modality, _ = key.split(".")
                except:  # noqa: E722
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            assert len(language_keys) == 1, f"{language_keys=}"
            self._language_key = language_keys[0]
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                video: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        # TODO(YL, FH): check if this is correct
        images = batch["images"]  # [V, T, C, H, W]
        images.shape[0]

        np_images = rearrange(images, "v t c h w -> (t v) c h w")
        text_content = []

        # handle language
        lang = batch["language"]
        if isinstance(lang, list):
            lang = lang[0]
        text_content.append({"type": "text", "text": lang})

        eagle_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
        eagle_image = [{"type": "image", "image": img} for img in eagle_images]
        eagle_conversation = [
            {
                "role": "user",
                "content": eagle_image + text_content,
            }
        ]

        text_list = [
            self.eagle_processor.apply_chat_template(
                eagle_conversation, tokenize=False, add_generation_prompt=True
            )
        ]
        image_inputs, video_inputs = self.eagle_processor.process_vision_info(eagle_conversation)
        eagle_content = {
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "text_list": text_list,
        }
        inputs = {}
        inputs["eagle_content"] = eagle_content
        return inputs

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        ## TODO(YL, FH): check if this is correct
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        return images

    def _prepare_language(self, data: dict):
        """Tokenize data['language'] (or default_instruction if missing)."""
        if self._language_key is not None:
            raw_language = data[self._language_key]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]

            # Language dropout
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction
        return raw_language

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        assert state.shape[0] == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]

        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        # Create mask for real state dims
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def apply_single(self, data: dict) -> dict:
        transformed_data = {}

        # 1) Prepare video and language with vlm processing.
        images = self._prepare_video(data)
        images = images.astype(np.uint8)
        language = self._prepare_language(data)
        batch_data = {"images": images, "language": language}
        vlm_outputs = self._apply_vlm_processing(batch_data)

        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # 3) Prepare actions
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"
        # transformed_data["trajectory_id"] = np.array(data["trajectory_id"])

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate(data_split_processed, self.eagle_processor)

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)



class GR00TInferTransform(InvertibleModalityTransform):

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    formalize_language: bool = Field(default=False, description="Formalize language if True.")
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[list[str]] = PrivateAttr(default=None)

    eagle_processor: ProcessorMixin = Field(default=build_eagle_processor(DEFAULT_EAGLE_PATH))

    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            if "annotation" in key:
                modality = "language"
            else:
                try:
                    modality, _ = key.split(".")
                except:  # noqa: E722
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            assert len(language_keys) == 1, f"{language_keys=}"
            self._language_key = language_keys[0]
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                video: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        # TODO(YL, FH): check if this is correct
        images = batch["images"]  # [V, T, C, H, W]
        images.shape[0]

        np_images = rearrange(images, "v t c h w -> (t v) c h w")
        text_content = []

        # handle language
        lang = batch["language"]
        if isinstance(lang, list):
            lang = lang[0]
        text_content.append({"type": "text", "text": lang})

        eagle_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
        eagle_image = [{"type": "image", "image": img} for img in eagle_images]
        eagle_conversation = [
            {
                "role": "user",
                "content": eagle_image + text_content,
            }
        ]

        text_list = [
            self.eagle_processor.apply_chat_template(
                eagle_conversation, tokenize=False, add_generation_prompt=True
            )
        ]
        image_inputs, video_inputs = self.eagle_processor.process_vision_info(eagle_conversation)
        eagle_content = {
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "text_list": text_list,
        }
        inputs = {}
        inputs["eagle_content"] = eagle_content
        return inputs

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        ## TODO(YL, FH): check if this is correct
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        return images

    def _prepare_language(self, data: dict):
        """Tokenize data['language'] (or default_instruction if missing)."""
        if self._language_key is not None:
            raw_language = data[self._language_key]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]

            # Language dropout
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction
        return raw_language

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        assert state.shape[0] == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]

        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        # Create mask for real state dims
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def apply_single(self, data: dict) -> dict:
        transformed_data = {}

        # 1) Prepare video and language with vlm processing.
        images = self._prepare_video(data)
        images = images.astype(np.uint8)
        language = self._prepare_language(data)
        batch_data = {"images": images, "language": language}
        vlm_outputs = self._apply_vlm_processing(batch_data)

        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # 3) Prepare actions
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data
    
    def _index_or_keep(self, x, i, B):
        # numpy / torch with batch dim
        if isinstance(x, (np.ndarray, torch.Tensor)):
            if x.ndim >= 1 and x.shape[0] == B:
                return x[i]
            else:
                return x  # scalar or wrong shape: keep as-is
        # python list that matches batch size
        if isinstance(x, list) and len(x) == B:
            return x[i]
        # everything else (bool / str / numbers / dicts already handled by tree): keep
        return x

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        data_split = [tree.map_structure(lambda x, i=i: self._index_or_keep(x, i, batch_size), data) for i in range(batch_size)]

        # def debug_take_i(path, x, i):
        #     try:
        #         return x if (isinstance(x, (np.ndarray, torch.Tensor)) and x.ndim == 0) else x[i]
        #     except Exception as e:
        #         print("Index error at path:", "/".join(map(str, path)), "type:", type(x), "repr:", repr(x))
        #         raise

        # _ = [map_structure_with_path(lambda p, x, i=i: debug_take_i(p, x, i), data)
        #     for i in range(batch_size)]

        # for k, v in data.items():
        #     if isinstance(v, np.ndarray):
        #         print(k, v.shape, v)
        #     else:
        #         print(k, v, type(v))
        # exit()

        # Split on batch dimension.
        # data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate_infer(data_split_processed, self.eagle_processor)

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)


class GR00TTactileTransform(InvertibleModalityTransform):

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    formalize_language: bool = Field(default=False, description="Formalize language if True.")
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[list[str]] = PrivateAttr(default=None)

    eagle_processor: ProcessorMixin = Field(default=build_eagle_processor(DEFAULT_EAGLE_PATH))

    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            if "annotation" in key:
                modality = "language"
            else:
                try:
                    modality, _ = key.split(".")
                except:  # noqa: E722
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            assert len(language_keys) == 1, f"{language_keys=}"
            self._language_key = language_keys[0]
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                video: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        # TODO(YL, FH): check if this is correct
        images = batch["images"]  # [V, T, C, H, W]
        images.shape[0]

        np_images = rearrange(images, "v t c h w -> (t v) c h w")
        text_content = []

        # handle language
        lang = batch["language"]
        if isinstance(lang, list):
            lang = lang[0]
        text_content.append({"type": "text", "text": lang})

        eagle_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
        eagle_image = [{"type": "image", "image": img} for img in eagle_images]
        eagle_conversation = [
            {
                "role": "user",
                "content": eagle_image + text_content,
            }
        ]

        text_list = [
            self.eagle_processor.apply_chat_template(
                eagle_conversation, tokenize=False, add_generation_prompt=True
            )
        ]
        image_inputs, video_inputs = self.eagle_processor.process_vision_info(eagle_conversation)
        eagle_content = {
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "text_list": text_list,
        }
        inputs = {}
        inputs["eagle_content"] = eagle_content
        return inputs

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        ## TODO(YL, FH): check if this is correct
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        return images

    def _prepare_language(self, data: dict):
        """Tokenize data['language'] (or default_instruction if missing)."""
        if self._language_key is not None:
            raw_language = data[self._language_key]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]

            # Language dropout
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction
        return raw_language

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        assert state.shape[0] == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]

        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        # Create mask for real state dims
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def _prepare_tactile(self, data: dict):
        """
        Gather tactile arrays (e.g., 'tactile.left', 'tactile.right'), keep native values,
        no padding. We only flatten the last two dims: (T, 5, 3) -> (T, 15).
        If both left/right exist, we concat along channel: (T, 30).
        Returns: tactile (T, D), tactile_mask (T, D), n_tactile_tokens (=T)
        """
        # 어떤 키들이 들어왔는지 스캔 (예: 'tactile.left', 'tactile.right')
        tactile_keys = [k for k in data.keys() if k.startswith("tactile.")]
        if len(tactile_keys) == 0:
            # 없는 경우: 패딩/더미 없이 깔끔하게 비어있는 텐서를 반환하고 마스크도 비움
            # (원한다면 여기서 0-dim 텐서 대신 None을 반환하고 호출부에서 분기해도 OK)
            print("[DEBUG] No tactile data found in input.")
            tactile = np.zeros((self.state_horizon, 0), dtype=np.float32)
            tactile_mask = np.zeros_like(tactile, dtype=bool)
            return tactile, tactile_mask, self.state_horizon

        parts = []
        T_expected = self.state_horizon

        for k in sorted(tactile_keys):  # 일정한 순서 보장: 'tactile.left', 'tactile.right'
            arr = data[k]
            # 허용 형태: (T, 5, 3) 또는 (T, 15)
            if not isinstance(arr, np.ndarray):
                arr = np.asarray(arr)

            # 시간 길이 검증
            assert arr.shape[0] == T_expected, f"{k} time dim mismatch: {arr.shape=} vs {T_expected=}"

            # (T, 5, 3) -> (T, 15)
            if arr.ndim == 3:
                T, A, B = arr.shape
                assert (A, B) == (5, 3), f"{k} expected shape (T,5,3), got {arr.shape}"
                arr = arr.reshape(T, A * B)
            elif arr.ndim == 2:
                # 이미 (T, D) 형태인 경우. D가 15인지 체크(원형 보장)
                assert arr.shape[1] in (15, 30), f"{k} unexpected channel dim: {arr.shape}"
            else:
                raise ValueError(f"{k} invalid ndim {arr.ndim}, expected 2 or 3")

            # dtype 정규화(값 변경 없음)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)

            parts.append(arr)

        # 여러 개면 채널 방향으로 concat (예: left(15) + right(15) = 30)
        tactile = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]

        # 마스크: 패딩이 없으므로 전부 True
        tactile_mask = np.ones_like(tactile, dtype=bool)

        n_tactile_tokens = tactile.shape[0]  # = T
        return tactile, tactile_mask, n_tactile_tokens

    def apply_single(self, data: dict) -> dict:
        transformed_data = {}

        # 1) Prepare video and language with vlm processing.
        images = self._prepare_video(data)
        images = images.astype(np.uint8)
        language = self._prepare_language(data)
        batch_data = {"images": images, "language": language}
        vlm_outputs = self._apply_vlm_processing(batch_data)

        tactile = self._prepare_tactile(data)
        transformed_data["tactile"] = tactile

        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # 3) Prepare actions
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"
        # transformed_data["trajectory_id"] = np.array(data["trajectory_id"])

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate(data_split_processed, self.eagle_processor)

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)
# Temporary file to store the GR00TAnyResolutionTransform class

class GR00TAnyResolutionTransform(InvertibleModalityTransform):

    backbone_model_type: str = "eagle"
    backbone_path: Optional[str] = None

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    formalize_language: bool = Field(default=False, description="Formalize language if True.")
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[list[str]] = PrivateAttr(default=None)

    eagle_processor: Optional[ProcessorMixin] = Field(default=None)
    qwen_3_vl_processor: Optional[ProcessorMixin] = Field(default=None)

    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None

    use_contextvla_chat_template: bool = Field(
        default=False,
        description="If True, use ContextVLA's custom chat template format (system message, <embodiment_tag> and <state> placeholders, role='assistant' for assistant)."
    )

    data_config: Optional[str] = Field(
        default=None,
        description="Data config name (for LeRobot datasets). Used to look up ContextVLA embodiment tag string."
    )

    concat_frames: bool = Field(
        default=False,
        description="If True, concatenate multiple views horizontally (along width axis) before passing to VLM backbone, matching the reference implementation."
    )

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def _discretize_state_for_contextvla(self, state: np.ndarray) -> str:
        """
        Discretize state for ContextVLA template.
        Args:
            state: [state_horizon, max_state_dim] or [max_state_dim] array
        Returns:
            Discretized state string: "123 456 789 ..."
        """
        # Take the first timestep if state has time dimension
        if state.ndim == 2:
            state = state[0]  # [max_state_dim]
        
        # Discretize: [-1, 1] -> [0, 999] using 1000 bins
        state = np.clip(state, -1, 1)
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 1000 + 1)[:-1]) - 1
        # Convert to string with space-separated values
        state_str = " ".join(map(str, discretized_state))
        return state_str

    def _get_embodiment_tag_string(self, dataset_name: Optional[str] = None) -> str:
        """
        Get embodiment tag string for ContextVLA template.
        
        Args:
            dataset_name: Optional dataset name (for RLDS datasets). If provided, uses ContextVLA mapping.
        
        Returns:
            Embodiment tag string for ContextVLA template.
            - For RLDS datasets: Returns full string like "Embodiment Tag: Bridge Dataset, Robot: WidowX, ..."
            - For LeRobot datasets: Returns full string like "Embodiment Tag: GR1 Dataset, Robot: GR1, ..."
        
        Note:
            In the reference code (tmp_AlinVLA-VLM/qwenvl/data/data_utils.py line 130),
            item['embodiment_tag'] is already a string (e.g., "Embodiment Tag: Bridge Dataset, Robot: WidowX, Morphology: Single Arm, Gripper: Default").
            For RLDS datasets, we use the mapping from gr00t/data/rlds/contextvla_embodiment_tags.py.
            For LeRobot datasets, we use the mapping from LEROBOT_EMBODIMENT_TAG_MAPPING based on self.data_config.
        """
        # For RLDS datasets: use ContextVLA mapping if dataset_name is provided
        if dataset_name is not None:
            from gr00t.data.rlds.contextvla_embodiment_tags import get_contextvla_embodiment_tag_string
            return get_contextvla_embodiment_tag_string(dataset_name)

        # For LeRobot datasets: use ContextVLA mapping based on data_config
        if self.data_config is not None:
            from gr00t.data.rlds.contextvla_embodiment_tags import LEROBOT_EMBODIMENT_TAG_MAPPING as LEROBOT_EMBODIMENT_TAG_STR_MAPPING
            from gr00t.data.rlds.contextvla_embodiment_tags import EMBODIMENT_TAG_MAPPING as EMBODIMENT_TAG_STR_MAPPING
            embodiment_tag_str = LEROBOT_EMBODIMENT_TAG_STR_MAPPING.get(
                self.data_config,
                EMBODIMENT_TAG_STR_MAPPING.get("default", "Embodiment Tag: Unknown Dataset, Robot: Unknown, Morphology: Unknown, Gripper: Unknown")
            )
            # _print(f"[DEBUG: use_contextvla_chat_template=True] data_config='{self.data_config}' -> embodiment_tag='{embodiment_tag_str}'")
            return embodiment_tag_str

        if self.embodiment_tag is None:
            embodiment_tag_value = "new_embodiment"
        else:
            embodiment_tag_value = self.embodiment_tag.value

        from gr00t.data.rlds.contextvla_embodiment_tags import get_contextvla_embodiment_tag_string_from_enum
        return get_contextvla_embodiment_tag_string_from_enum(embodiment_tag_value)

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            if "annotation" in key:
                modality = "language"
            else:
                try:
                    modality, _ = key.split(".")
                except:  # noqa: E722
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        # _print(f"[DEBUG] check_keys_and_batch_size data.keys(): {data.keys()}")
        # _print(f"[DEBUG] check_keys_and_batch_size grouped_keys: {grouped_keys}")
        is_batched = False
        batch_size = 1

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            assert len(language_keys) == 1, f"{language_keys=}"
            self._language_key = language_keys[0]
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                images: List[np.ndarray(T, H, W, C)] - each element is for one view (RLDS format)
                     or [V, T, C, H, W] (LeRobot format for Eagle)
        Returns: required input with the format `BatchFeature`
        """

        if self.eagle_processor is None:
            self.eagle_processor = build_eagle_processor(DEFAULT_EAGLE_PATH if self.backbone_path is None else self.backbone_path)
        
        # RLDS format: images is List[np.ndarray(T, H, W, C)] - each element is one view
        images_list = batch["images"]  # List of [T, H, W, C]
        # Debug: print shape for RLDS format (once per transform instance)
        #if not getattr(self, "_rlds_shape_logged", False):
        #    shapes = [f"view{i}:{arr.shape}" for i, arr in enumerate(images_list)]
        #    _print(f"[RLDS Eagle] images_list: num_views={len(images_list)}, shapes={shapes}")
        #    self._rlds_shape_logged = True
        
        num_views = len(images_list)
        T = images_list[0].shape[0]
        
        # Interleave timesteps and views: T0_V0, T0_V1, ..., T1_V0, T1_V1, ...
        eagle_images = []
        for t in range(T):
            for v_idx in range(num_views):
                img_arr = images_list[v_idx][t]  # (H, W, C)
                eagle_images.append(Image.fromarray(img_arr.astype(np.uint8)))
        
        eagle_image = [{"type": "image", "image": img} for img in eagle_images]
        
        # handle language
        lang = batch["language"]
        if isinstance(lang, list):
            lang = lang[0]
        text_content = [{"type": "text", "text": lang}]

        eagle_conversation = [
            {
                "role": "user",
                "content": eagle_image + text_content,
            }
        ]

        text_list = [
            self.eagle_processor.apply_chat_template(
                eagle_conversation, tokenize=False, add_generation_prompt=True
            )
        ]
        image_inputs, video_inputs = self.eagle_processor.process_vision_info(eagle_conversation)
        eagle_content = {
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "text_list": text_list,
            "use_oxe_collate": True,  # RLDS format: 3 views per sample
        }
        inputs = {}
        inputs["eagle_content"] = eagle_content
        return inputs
        
        """
        elif self.qwen_3_vl_processor is None:
            # Check if backbone_path points to a ContextVLA checkpoint (which doesn't have processor config)
            is_contextvla_checkpoint = self.backbone_path is not None and "contextvla" in self.backbone_path.lower()

            if self.backbone_model_type == "contextvla_qwen3_vl_8b" or is_contextvla_checkpoint:
                from gr00t.model.backbone.roboalign_contextvla.modeling_contextvla import get_contextvla_processor
                self.qwen_3_vl_processor = get_contextvla_processor(
                    "Qwen/Qwen3-VL-8B-Instruct",
                    contextvla_checkpoint_path=self.backbone_path,
                    verbose=False,
                )
            else:
                processor_path = DEFAULT_QWEN_3_VL_PATH if self.backbone_path is None else self.backbone_path
                self.qwen_3_vl_processor = build_qwen_processor(processor_path)

        if self.concat_frames:
            # If 'concat_frames=True', images is [T, H, W*V, C] (already concatenated in _prepare_video)
            images = batch["images"]  # [T, H, W*V, C]
            T, H, W_total, C = images.shape
            
            # Convert to numpy array for video format: [T, H, W*V, C]
            video_array = images.astype(np.uint8)  # [T, H, W*V, C]
            
            # Convert each timestep frame to PIL Image
            video_frames = [
                Image.fromarray(video_array[t]) for t in range(T)
            ]  # List of PIL Images, each is [H, W*V, C]
            
            qwen_video = [{"type": "video", "video": video_frames}]
            
            text_content = []
            lang = batch["language"]
            if isinstance(lang, list):
                lang = lang[0]

            # Build conversation based on template type
            if self.use_contextvla_chat_template:
                state_str = ""
                if "state" in batch:
                    state = batch["state"]
                    state_str = self._discretize_state_for_contextvla(state)
                    state_str = f"Robot state is {state_str}."

                dataset_name = batch.get("dataset_name", None)
                embodiment_tag_str = self._get_embodiment_tag_string(dataset_name=dataset_name)
                if embodiment_tag_str:
                    if not embodiment_tag_str.startswith("Embodiment Tag:"):
                        embodiment_tag_str = f"Embodiment Tag: {embodiment_tag_str}"
                    embodiment_tag_str = f"{embodiment_tag_str}."

                # Build user message text with hardcoded ContextVLA format
                user_text = f"Current task is {lang}. {embodiment_tag_str} {state_str} Output the robot's actions to perform this task through FAST tokens."
                if not hasattr(self, "_debug_logged_chat_template"):
                    _print(f"[DEBUG: use_contextvla_chat_template=True] User text (first 200 chars): {user_text[:200]}...")
                    _print(f"[DEBUG: use_contextvla_chat_template=True] Embodiment tag: {embodiment_tag_str}")
                    self._debug_logged_chat_template = True
                text_content.append({"type": "text", "text": user_text})

                qwen_conversation = [
                    {
                        "role": "system",
                        "content": "You are an embodied vision-language robotic assistant for multi-object manipulation."
                    },
                    {
                        "role": "user",
                        "content": qwen_video + text_content,
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": ""}]
                    }
                ]
                add_generation_prompt = False  # Already included in conversation
            else:
                # Standard template
                text_content.append({"type": "text", "text": lang})
                qwen_conversation = [
                    {
                        "role": "user",
                        "content": qwen_video + text_content,
                    }
                ]
                add_generation_prompt = True

            text_list = [
                self.qwen_3_vl_processor.apply_chat_template(
                    qwen_conversation, tokenize=False, add_generation_prompt=add_generation_prompt
                )
            ]
            # Process vision info
            image_inputs, video_inputs = qwen_process_vision_info(qwen_conversation, image_patch_size=16)
            if not hasattr(self, "_debug_logged_qwen_process"):
                _print(f"[DEBUG: qwen_process_vision_info] After processing:")
                _print(f"  image_inputs type: {type(image_inputs)}, length: {len(image_inputs) if image_inputs is not None else 'None'}")
                _print(f"  video_inputs type: {type(video_inputs)}, length: {len(video_inputs) if video_inputs is not None else 'None'}")
                if video_inputs is not None and len(video_inputs) > 0:
                    for i, vid in enumerate(video_inputs):
                        if isinstance(vid, list):
                            _print(f"  video_inputs[{i}]: list with {len(vid)} frames")
                            if len(vid) > 0:
                                if hasattr(vid[0], 'size'):
                                    _print(f"    First frame size: {vid[0].size} (W, H)")
                                elif isinstance(vid[0], torch.Tensor):
                                    _print(f"    First frame tensor shape: {vid[0].shape}")
                        elif isinstance(vid, torch.Tensor):
                            _print(f"  video_inputs[{i}]: tensor shape {vid.shape}")
                        else:
                            _print(f"  video_inputs[{i}]: type {type(vid)}")
                self._debug_logged_qwen_process = True

            qwen_content = {
                "image_inputs": image_inputs,
                "video_inputs": video_inputs,
                "text_list": text_list,
            }
        else:
            # concat_frames=False: images is List[np.ndarray(T, H, W, C)] - each element is one view
            images_list = batch["images"]  # List of [T, H, W, C]

            num_views = len(images_list)
            T = images_list[0].shape[0]

            qwen_images = []
            # We want to maintain order: T0_V0, T0_V1, T1_V0, T1_V1 ... (to match standard interleaving)
            for t in range(T):
                for v_idx in range(num_views):
                    img_arr = images_list[v_idx][t] # (H, W, C)
                    qwen_images.append(Image.fromarray(img_arr.astype(np.uint8)))

            qwen_image_dicts = [{"type": "image", "image": img} for img in qwen_images]

            # handle language
            lang = batch["language"]
            if isinstance(lang, list):
                lang = lang[0]
            # Build conversation based on template type
            if self.use_contextvla_chat_template:
                state_str = ""
                if "state" in batch:
                    state = batch["state"]
                    state_str = self._discretize_state_for_contextvla(state)
                    state_str = f"Robot state is {state_str}."

                dataset_name = batch.get("dataset_name", None)
                embodiment_tag_str = self._get_embodiment_tag_string(dataset_name=dataset_name)
                if embodiment_tag_str:
                    if not embodiment_tag_str.startswith("Embodiment Tag:"):
                        embodiment_tag_str = f"Embodiment Tag: {embodiment_tag_str}"
                    embodiment_tag_str = f"{embodiment_tag_str}."

                # Build user message text with hardcoded ContextVLA format
                user_text = f"Current task is {lang}. {embodiment_tag_str} {state_str} Output the robot's actions to perform this task through FAST tokens."
                # _print(f"[DEBUG: use_contextvla_chat_template=True] User text (first 200 chars): {user_text[:200]}...")
                text_content = [{"type": "text", "text": user_text}]

                qwen_conversation = [
                    {
                        "role": "system",
                        "content": "You are an embodied vision-language robotic assistant for multi-object manipulation."
                    },
                    {
                        "role": "user",
                        "content": qwen_image_dicts + text_content,
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": ""}]
                    }
                ]
                add_generation_prompt = False  # Already included in conversation
            else:
                # Standard template
                text_content = [{"type": "text", "text": lang}]
                qwen_conversation = [
                    {
                        "role": "user",
                        "content": qwen_image_dicts + text_content,
                    }
                ]
                add_generation_prompt = True

            text_list = [
                self.qwen_3_vl_processor.apply_chat_template(
                    qwen_conversation, tokenize=False, add_generation_prompt=add_generation_prompt
                )
            ]


            # Process vision info
            # qwen_process_vision_info will prepare the pixel_values and image_grid_thw
            image_inputs, video_inputs = qwen_process_vision_info(qwen_conversation, image_patch_size=16)

            qwen_content = {
                "image_inputs": image_inputs,
                "video_inputs": video_inputs,
                "text_list": text_list,
            }

        inputs = {}
        inputs["qwen_content"] = qwen_content
        return inputs"""

    def _prepare_video(self, data: dict):
        """Process video data. If concat_frames=True, concatenate multiple views horizontally."""
        video_list = data["video"]  # List[np.ndarray], each is [T, H, W, C] for one view

        if self.concat_frames:
            # Concatenate multiple views horizontally (along width axis) for each frame
            num_views = len(video_list)
            if num_views == 0:
                raise ValueError("No video views found in data")

            # Use first view's height as reference for resizing other views
            T, target_H, _, C = video_list[0].shape

            concatenated_frames = []
            for t in range(T):
                frame_views = []
                for v_idx in range(num_views):
                    frame = video_list[v_idx][t]  # [H, W, C]
                    curr_H, curr_W = frame.shape[:2]

                    # Resize to target height while maintaining aspect ratio if needed
                    if curr_H != target_H:
                        # Calculate new width maintaining aspect ratio
                        aspect_ratio = curr_W / curr_H
                        target_W = int(target_H * aspect_ratio)

                        # Convert to PIL Image, resize, and convert back
                        pil_image = Image.fromarray(frame.astype(np.uint8))
                        resized_pil = pil_image.resize((target_W, target_H), Image.BILINEAR)
                        frame = np.array(resized_pil)  # [target_H, target_W, C]

                    frame_views.append(frame)  # [target_H, W, C] (W may vary)

                # Concatenate views along width axis (axis=1)
                concat_frame = np.concatenate(frame_views, axis=1)  # [target_H, W_total, C]
                concatenated_frames.append(concat_frame)

            # Stack along time axis: [T, target_H, W_total, C]
            images = np.stack(concatenated_frames, axis=0)
            return images
        else:
            # Return as list (original behavior)
            return video_list  # List[np.ndarray]

    def _prepare_language(self, data: dict):
        """Tokenize data['language'] (or default_instruction if missing)."""
        if self._language_key is not None:
            raw_language = data[self._language_key]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]

            # Language dropout
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction
        return raw_language

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        if state.ndim == 1:
            state = state[None, :]  # Add time dimension
        assert state.shape[0] == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]

        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        # Create mask for real state dims
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def apply_single(self, data: dict) -> dict:
        transformed_data = {}

        # 1) Prepare video and language with vlm processing.
        images = self._prepare_video(data)
        language = self._prepare_language(data)
        batch_data = {"images": images, "language": language}

        

        vlm_outputs = self._apply_vlm_processing(batch_data)

        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # 3) Prepare actions
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        # Get embodiment_id: use dataset_name if available (for RLDS datasets), otherwise use self.embodiment_tag
        if "dataset_name" in data:
            dataset_name = data["dataset_name"]
            if isinstance(dataset_name, list):
                dataset_name = dataset_name[0]
            # Normalize dataset_name (handle special cases like agibot_gripper, galaxea)
            normalized_name = dataset_name.lower()
            if "agibot_gripper" in normalized_name:
                dataset_name = "agibot_gripper"
            elif "agibot_dexhand" in normalized_name or ("agibot" in normalized_name and "gripper" not in normalized_name):
                dataset_name = "agibot_dexhand"
            elif "galaxea" in normalized_name:
                dataset_name = "galaxea"
            # elif "egodex" in normalized_name or ("gr1" in normalized_name and "egodex" in normalized_name):
            #     dataset_name = "egodex_gr1"

            try:
                # Try to find matching EmbodimentTag enum
                embodiment_tag_enum = None
                for tag in EmbodimentTag:
                    if tag.value == dataset_name:
                        embodiment_tag_enum = tag
                        break

                if embodiment_tag_enum is not None:
                    transformed_data["embodiment_id"] = self.embodiment_tag_mapping[embodiment_tag_enum.value]
                else:
                    # Fallback to default if dataset_name doesn't match any enum
                    _print(f"[w] Dataset name '{dataset_name}' not found in EmbodimentTag enum, using default")
                    transformed_data["embodiment_id"] = self.embodiment_tag_mapping[EmbodimentTag.NEW_EMBODIMENT.value]
            except Exception as e:
                _print(f"[w] Error getting embodiment_id from dataset_name '{dataset_name}': {e}, using default")
                transformed_data["embodiment_id"] = self.embodiment_tag_mapping[EmbodimentTag.NEW_EMBODIMENT.value]
        else:
            transformed_data["embodiment_id"] = self.get_embodiment_tag()

        # Pass dataset_name if present (for per-dataset loss logging)
        if "dataset_name" in data:
            transformed_data["dataset_name"] = data["dataset_name"]

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        
        # Select appropriate collate function based on backbone
        if self.backbone_model_type == "eagle":
            return collate_oxe(data_split_processed, self.eagle_processor)
        else:
            # Qwen3 VL and other models
            return qwen_3_vl_collate(data_split_processed, self.qwen_3_vl_processor)

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched: # TY: Need for eval
            return self.apply_batch(data, batch_size)
        else: # TY: At training, processes single sample at GR00TTransform
            return self.apply_single(data)
        # return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)


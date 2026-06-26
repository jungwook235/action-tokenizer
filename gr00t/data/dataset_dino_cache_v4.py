"""Cached-DINO variant of the Stage-1 V4 tokenizer dataset.

Drop-in replacement for ``ActionFramesDatasetV4`` when a precomputed DINO feature
cache exists (see ``scripts/precompute_dino_features.py``). Instead of decoding
two video frames per sample and running DINO at train time, this dataset:

  * loads the action chunk through the SAME action pipeline as the live V4
    dataset (it reuses ``ActionOnlyDataset``'s ToTensor → StateActionTransform →
    Concat, so normalized actions are byte-identical), and
  * reads ``x0_feat`` / ``x1_feat`` straight from the cache by
    ``(episode_id, base_index)`` — no video decode, no DINO forward.

The train/val episode partition uses the SAME persistent fixed-val split as
``ActionFramesDatasetV4`` so train and val select identical episodes to a live
run; the cache itself is split-agnostic (it stores every row).

Returned per sample: ``{"action": [T,D], "x0_feat": [Lp,C], "x1_feat": [Lp,C]}``
with the feats in the cache's stored dtype (float32 by default). The trainer
casts them to the model dtype, exactly like ``_extract_feats``'s ``.float()``.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from gr00t.data.dataset import LE_ROBOT_EPISODE_FILENAME
from gr00t.data.dataset_action_only import ActionOnlyDataset
from gr00t.data.dino_feature_cache import DinoFeatureCacheReader, make_cache_key
from gr00t.data.fixed_val_split import get_fixed_split_for_split
from gr00t.experiment.data_config import DATA_CONFIG_MAP


class CachedActionFramesDatasetV4(ActionOnlyDataset):
    """Action chunk + cached (x0_feat, x1_feat) for V4 tokenizer training."""

    def __init__(
        self,
        dataset_path: str | Path,
        data_config_name: str,
        embodiment_tag: str,
        split: str = "all",
        val_ratio: float = 0.003,
        val_seed: int = 42,
        normalization_mode: str = "min_max",
        image_size: int = 224,
        # Cache identity — MUST match what precompute_dino_features.py used.
        feature_source: str = "dino",
        dino_model: str = "facebook/dinov2-large",
        dino_final_norm: str = "naive",
        # Fixed-val split (matches ActionFramesDatasetV4).
        use_fixed_val: bool = True,
        fixed_val_path: Optional[str] = None,
        # Unused; accepted for call-site parity with ActionFramesDatasetV4.
        video_backend: str = "decord",
    ):
        # Set BEFORE super().__init__ — LeRobotSingleDataset.__init__ calls
        # self._get_trajectories(), which reads these.
        self._use_fixed_val = use_fixed_val
        self._fixed_val_path = fixed_val_path

        data_config_cls = DATA_CONFIG_MAP[data_config_name]
        full_modality_configs = data_config_cls.modality_config()
        assert "video" in full_modality_configs, (
            f"data_config '{data_config_name}' has no video modality."
        )
        video_keys = list(full_modality_configs["video"].modality_keys)
        assert len(video_keys) == 1, (
            f"V4 expects a single camera; got video_keys={video_keys}"
        )
        video_key = video_keys[0]
        action_indices = list(full_modality_configs["action"].delta_indices)
        action_horizon = len(action_indices)

        super().__init__(
            dataset_path=dataset_path,
            data_config_name=data_config_name,
            embodiment_tag=embodiment_tag,
            split=split,
            val_ratio=val_ratio,
            val_seed=val_seed,
            normalization_mode=normalization_mode,
            video_backend=video_backend,
        )

        self._action_horizon = action_horizon
        key = make_cache_key(
            feature_source=feature_source,
            model_name=dino_model,
            final_norm=dino_final_norm,
            image_size=image_size,
            video_key=video_key,
        )
        self._reader = DinoFeatureCacheReader(
            dataset_path,
            key,
            action_horizon=action_horizon,
            expect={
                "feature_source": feature_source,
                "model_name": dino_model,
                "final_norm": dino_final_norm,
                "image_size": image_size,
                "video_key": video_key,
            },
        )

    def _get_trajectories(self):
        """Episode-level split using the persistent fixed-val file (matches V4)."""
        episode_path = self._dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]

        all_ids = np.array([e["episode_index"] for e in episode_metadata])
        all_lengths = np.array([e["length"] for e in episode_metadata])

        if self._split == "all":
            return all_ids, all_lengths

        if self._use_fixed_val:
            ids, lengths = get_fixed_split_for_split(
                dataset_path=self._dataset_path,
                all_ids=all_ids,
                all_lengths=all_lengths,
                split=self._split,
                val_seed=self._val_seed,
                val_ratio=self._val_ratio,
                fixed_val_path=self._fixed_val_path,
            )
            print(
                f"[CachedActionFramesDatasetV4][{self._split}] "
                f"{Path(self._dataset_path).name}: {len(all_ids)} episodes → "
                f"{len(ids)} used (fixed-val)"
            )
            return ids, lengths

        n_total = len(all_ids)
        n_val = max(1, int(n_total * self._val_ratio))
        rng = np.random.default_rng(self._val_seed)
        shuffled = rng.permutation(n_total)
        selected = np.sort(shuffled[:n_val]) if self._split == "val" else np.sort(shuffled[n_val:])
        return all_ids[selected], all_lengths[selected]

    def __getitem__(self, index: int) -> dict:
        trajectory_id, base_index = self.all_steps[index]
        data = self.get_step_data(trajectory_id, base_index)
        data = self.transforms(data)  # ConcatTransform → only "action" remains
        x0_feat, x1_feat = self._reader.get_pair(int(trajectory_id), int(base_index))
        return {
            "action": data["action"],
            "x0_feat": x0_feat,  # [Lp, C] (cache dtype)
            "x1_feat": x1_feat,
        }


class CachedActionFramesCollatorV4:
    """Collate {action, x0_feat, x1_feat} into batched tensors (no frames)."""

    def __call__(self, features: list[dict]) -> dict:
        return {
            "action": torch.stack([f["action"] for f in features]),  # [B, T, D]
            "x0_feat": torch.from_numpy(np.stack([f["x0_feat"] for f in features])),  # [B, Lp, C]
            "x1_feat": torch.from_numpy(np.stack([f["x1_feat"] for f in features])),
        }

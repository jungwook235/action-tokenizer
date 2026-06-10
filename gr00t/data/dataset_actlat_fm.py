"""
Dataset and collator for action latent flow matching VLA training.

- LeRobotSingleDatasetActlatFM: LeRobotSingleDataset with train/val split.
  Data processing is identical to base — only split is added.
- ActlatFMDataCollator: Collator for single-timestep observations (1–3 cameras).
  No future index / FLARE dual-timestep support.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from gr00t.data.dataset import LE_ROBOT_EPISODE_FILENAME, LeRobotSingleDataset
from gr00t.data.fixed_val_split import get_fixed_split_for_split
from gr00t.data.transform import ComposedModalityTransform
from gr00t.model.transforms import build_eagle_processor, DEFAULT_EAGLE_PATH


class LeRobotSingleDatasetActlatFM(LeRobotSingleDataset):
    """LeRobotSingleDataset with train/val split support.

    Episode-level split prevents data leakage. Same val_seed produces
    identical splits across runs.

    When ``use_fixed_val=True`` (default), the split is loaded from / persisted
    to a JSON file (``<dataset>/meta/fixed_val_split.json`` by default) so that
    this Stage-2 VLA training shares the exact same val episodes as the Stage-1
    tokenizer training. The file is created on first use if missing. Set
    ``use_fixed_val=False`` to fall back to the purely seed-deterministic split.
    """

    def __init__(
        self,
        dataset_path,
        modality_configs,
        embodiment_tag,
        video_backend: str = "torchvision_av",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        split: str = "train",
        val_ratio: float = 0.003,
        val_seed: int = 42,
        use_fixed_val: bool = True,
        fixed_val_path: Optional[str] = None,
    ):
        assert split in ("train", "val"), f"split must be 'train' or 'val', got: {split}"
        assert 0.0 < val_ratio < 1.0, f"val_ratio must be in (0, 1), got: {val_ratio}"

        self.split = split
        self.val_ratio = val_ratio
        self.val_seed = val_seed
        self.use_fixed_val = use_fixed_val
        self.fixed_val_path = fixed_val_path

        super().__init__(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            transforms=transforms,
        )

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Episode-level train/val split.

        Uses the persistent fixed-val JSON (shared with Stage-1) when
        ``use_fixed_val`` is set; otherwise falls back to the seed-deterministic
        split computed inline.
        """
        episode_path = self._dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]

        all_ids = np.array([e["episode_index"] for e in episode_metadata])
        all_lengths = np.array([e["length"] for e in episode_metadata])

        if self.use_fixed_val:
            ids, lengths = get_fixed_split_for_split(
                dataset_path=self._dataset_path,
                all_ids=all_ids,
                all_lengths=all_lengths,
                split=self.split,
                val_seed=self.val_seed,
                val_ratio=self.val_ratio,
                fixed_val_path=self.fixed_val_path,
            )
            print(
                f"[{self.split}] {self._dataset_path.name}: "
                f"{len(ids)} / {len(all_ids)} episodes (fixed-val, "
                f"path={self.fixed_val_path or '<dataset>/meta/fixed_val_split.json'})"
            )
            return ids, lengths

        n_total = len(all_ids)
        n_val = max(1, int(n_total * self.val_ratio))

        rng = np.random.default_rng(self.val_seed)
        shuffled = rng.permutation(n_total)

        if self.split == "val":
            selected = np.sort(shuffled[:n_val])
        else:
            selected = np.sort(shuffled[n_val:])

        print(
            f"[{self.split}] {self._dataset_path.name}: "
            f"{len(selected)} / {n_total} episodes "
            f"(val_ratio={self.val_ratio}, val_seed={self.val_seed})"
        )

        return all_ids[selected], all_lengths[selected]


def _collate_actlat_fm(features: List[dict], eagle_processor) -> dict:
    """Collate function for single-timestep observations with 1–3 cameras.

    Always assumes a single observation timestep (no future index / FLARE dual-timestep).
    Supports 1, 2, or 3 camera images per sample, detected from the text tags.
    """
    batch = {}
    keys = features[0].keys()

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "eagle_content":
            text_list = []
            image_inputs = []
            for v in values:
                text_list += v["text_list"]
                image_inputs += v["image_inputs"]

            eagle_inputs = eagle_processor(
                text=text_list, images=image_inputs, return_tensors="pt", padding=True
            )
            for k, v in eagle_inputs.items():
                batch["eagle_" + k] = v

        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            batch[key] = torch.cat(values)
        else:
            batch[key] = torch.from_numpy(np.stack(values))

    return batch


class ActlatFMDataCollator:
    """Data collator with fixed single-observation handling."""

    def __init__(self, eagle_path: str = DEFAULT_EAGLE_PATH):
        self.eagle_processor = build_eagle_processor(eagle_path)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return _collate_actlat_fm(features, self.eagle_processor)

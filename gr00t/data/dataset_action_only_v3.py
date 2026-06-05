"""ActionOnlyDatasetV3: extends ActionOnlyDataset with persistent fixed-val split.

V3 difference: when ``use_fixed_val=True`` (default), the train/val episode
split is loaded from / persisted to a JSON file so multiple experiments
sharing a path use the exact same val episodes. Falls back to v2's
deterministic-by-seed behavior when ``use_fixed_val=False``.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from gr00t.data.dataset import LE_ROBOT_EPISODE_FILENAME
from gr00t.data.dataset_action_only import ActionOnlyDataset
from gr00t.data.fixed_val_split import get_fixed_split_for_split


class ActionOnlyDatasetV3(ActionOnlyDataset):
    """ActionOnlyDataset with optional persistent fixed-val split.

    Args (in addition to ActionOnlyDataset args):
        use_fixed_val: if True (default), load/save fixed val split.
        fixed_val_path: explicit absolute path for the split JSON. If None,
            defaults to ``<dataset>/meta/fixed_val_split.json``.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        data_config_name: str,
        embodiment_tag: str,
        split: str = "all",
        val_ratio: float = 0.003,
        val_seed: int = 42,
        normalization_mode: str = "min_max",
        video_backend: str = "torchvision_av",
        use_fixed_val: bool = True,
        fixed_val_path: Optional[str] = None,
    ):
        # Stash before super().__init__ — _get_trajectories will look these up.
        self._use_fixed_val = use_fixed_val
        self._fixed_val_path = fixed_val_path
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

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Episode-level split using the persistent file when enabled."""
        episode_path = self._dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]

        all_ids = np.array([e["episode_index"] for e in episode_metadata])
        all_lengths = np.array([e["length"] for e in episode_metadata])

        if self._split == "all":
            return all_ids, all_lengths

        if not self._use_fixed_val:
            return super()._get_trajectories()

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
            f"[ActionOnlyDatasetV3][{self._split}] {Path(self._dataset_path).name}: "
            f"전체 {len(all_ids)}개 에피소드 중 {len(ids)}개 사용 (fixed-val)"
        )
        return ids, lengths

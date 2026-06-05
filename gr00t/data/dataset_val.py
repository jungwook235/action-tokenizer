"""
LeRobotSingleDatasetWithSplit: LeRobotSingleDataset에 train/val split 기능을 추가한 서브클래스.

에피소드(trajectory) 단위로 split하여 data leakage를 방지합니다.
"""

import json

import numpy as np

from gr00t.data.dataset import LE_ROBOT_EPISODE_FILENAME, LeRobotSingleDataset
from gr00t.data.transform import ComposedModalityTransform


class LeRobotSingleDatasetWithSplit(LeRobotSingleDataset):
    """
    Train/val split을 지원하는 LeRobotSingleDataset 서브클래스.

    에피소드(trajectory) 단위로 split하므로 data leakage가 없습니다.
    같은 val_seed를 사용하면 train/val 간 에피소드가 겹치지 않습니다.

    Args:
        split: "train" 또는 "val"
        val_ratio: validation에 사용할 에피소드 비율 (기본값: 0.003 = 0.3%)
        val_seed: split 결정에 사용할 random seed (기본값: 42)
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
    ):
        assert split in ("train", "val"), f"split은 'train' 또는 'val'이어야 합니다. 입력값: {split}"
        assert 0.0 < val_ratio < 1.0, f"val_ratio는 0과 1 사이여야 합니다. 입력값: {val_ratio}"

        # _get_trajectories가 super().__init__ 안에서 호출되므로 먼저 설정
        self.split = split
        self.val_ratio = val_ratio
        self.val_seed = val_seed

        super().__init__(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            transforms=transforms,
        )

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """에피소드 단위로 train/val split 후 해당 split의 trajectory만 반환."""
        episode_path = self._dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]

        all_ids = np.array([e["episode_index"] for e in episode_metadata])
        all_lengths = np.array([e["length"] for e in episode_metadata])

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
            f"전체 {n_total}개 에피소드 중 {len(selected)}개 사용 "
            f"(val_ratio={self.val_ratio}, val_seed={self.val_seed})"
        )

        return all_ids[selected], all_lengths[selected]

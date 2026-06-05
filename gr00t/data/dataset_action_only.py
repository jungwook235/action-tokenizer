"""
ActionOnlyDataset: action 데이터만 로드하는 Dataset 클래스.

LeRobotSingleDataset를 상속하되, action modality만 로드하고
action 전용 transform 파이프라인을 내부에서 구성합니다.
비디오/상태/언어 데이터는 전혀 로드하지 않아 훨씬 빠릅니다.

train/val split 기능도 내장되어 있습니다 (에피소드 단위 split).
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from gr00t.data.dataset import LE_ROBOT_EPISODE_FILENAME, LeRobotSingleDataset
from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform import ComposedModalityTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.experiment.data_config import DATA_CONFIG_MAP


class ActionOnlyDataset(LeRobotSingleDataset):
    """Action 데이터만 로드하는 LeRobotSingleDataset 서브클래스.

    - 기존 data_config에서 action modality 정보만 추출
    - 비디오/상태/언어 로딩 없이 parquet에서 action만 읽음
    - action 전용 transform (ToTensor → Normalize → Concat) 자동 구성
    - 에피소드 단위 train/val split 지원

    Args:
        dataset_path: 데이터셋 경로
        data_config_name: DATA_CONFIG_MAP의 키 (e.g., "fourier_gr1_arms_only")
        embodiment_tag: 로봇 태그 (e.g., "new_embodiment")
        split: "train", "val", 또는 "all" (split 없이 전체 사용)
        val_ratio: validation에 사용할 에피소드 비율 (기본값: 0.003 = 0.3%)
        val_seed: split 결정에 사용할 random seed
        normalization_mode: action 정규화 방식 (기본값: "min_max")
        video_backend: 비디오 백엔드 (사용하지 않지만 부모 클래스 호환성을 위해)
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
    ):
        assert split in ("train", "val", "all"), f"split은 'train', 'val', 'all' 중 하나여야 합니다: {split}"

        self._split = split
        self._val_ratio = val_ratio
        self._val_seed = val_seed

        # data_config에서 action 정보 추출
        data_config_cls = DATA_CONFIG_MAP[data_config_name]
        full_modality_configs = data_config_cls.modality_config()
        assert "action" in full_modality_configs, f"data_config '{data_config_name}'에 action modality가 없습니다."

        # action-only modality config
        action_modality_config = {"action": full_modality_configs["action"]}
        action_keys = full_modality_configs["action"].modality_keys

        # Per-key normalization modes from data_config (respect binary, rotation-related 등).
        # 정의 안 되어 있으면 uniform fallback — 기존 동작과 동일.
        action_normalization_modes = getattr(data_config_cls, "action_normalization_modes", None)
        if not action_normalization_modes:
            action_normalization_modes = {key: normalization_mode for key in action_keys}

        # Per-key target rotations from data_config (e.g. axis_angle for eef_rotation_delta).
        # 정의 안 되어 있으면 빈 dict — 회전 변환 없음 (기존 동작과 동일).
        # PreTransformedActionStateDataset 와 동일하게 data_config 를 respect 해서
        # tokenizer 학습 시 VLA inference 와 입력 분포가 정확히 일치하도록 한다.
        action_target_rotations = getattr(data_config_cls, "action_target_rotations", {}) or {}

        # action 전용 transform 파이프라인 구성
        transforms = self._build_action_transforms(
            action_keys, action_normalization_modes, action_target_rotations
        )

        super().__init__(
            dataset_path=dataset_path,
            modality_configs=action_modality_config,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            transforms=transforms,
        )

        self._action_keys = action_keys

    @staticmethod
    def _build_action_transforms(
        action_keys: list[str],
        action_normalization_modes: dict[str, str],
        action_target_rotations: dict[str, str] | None = None,
    ) -> ComposedModalityTransform:
        """Action 전용 transform 파이프라인 생성.

        action_normalization_modes / action_target_rotations 는 data_config 의 per-key
        설정을 그대로 전달받으므로 tokenizer 학습 시 VLA inference (= bridge_flare_kty_actlat_fm
        의 ConcatTransform 직전까지의 동작) 와 정확히 동일한 값 분포로 학습한다.

        action_target_rotations 미지정 시 빈 dict — 회전 변환 없음 (backwards compat).
        """
        if action_target_rotations is None:
            action_target_rotations = {}
        return ComposedModalityTransform(
            transforms=[
                # 1. numpy → tensor
                StateActionToTensor(
                    apply_to=action_keys,
                    output_dtypes={key: torch.float32 for key in action_keys},
                ),
                # 2. 회전 변환 (per-key: axis_angle / rotation_6d / ...) + 정규화 (min_max / binary / ...)
                StateActionTransform(
                    apply_to=action_keys,
                    normalization_modes=action_normalization_modes,
                    target_rotations=action_target_rotations,
                ),
                # 3. 개별 action key들을 하나의 "action" tensor로 합침
                ConcatTransform(
                    video_concat_order=[],
                    state_concat_order=None,
                    action_concat_order=action_keys,
                ),
            ]
        )

    def _get_needed_parquet_columns(self) -> list[str]:
        """Parquet 에서 로드할 최소 컬럼 집합 (action 만).

        `self.modality_keys["action"]` 는 이미 `"action.xxx"` full key 이므로 그대로
        `get_key_meta` 에 넘긴다. robocasa 의 경우 image dict 컬럼 (observation.images.*)
        은 로드 대상에서 제외 → parquet I/O 가 수배 빨라진다.
        """
        needed: set[str] = set()
        for full_key in self.modality_keys.get("action", []):
            meta = self._lerobot_modality_meta.get_key_meta(full_key)
            if getattr(meta, "original_key", None):
                needed.add(meta.original_key)
        return sorted(needed)

    def get_trajectory_data(self, trajectory_id: int):
        """LeRobotSingleDataset.get_trajectory_data override.

        두 가지 문제를 동시에 해결한다 (PreTransformedActionStateDataset 와 동일 패턴):
          1) column projection — 원본은 `pd.read_parquet(path)` 를 불러 image dict 컬럼
             (robocasa 의 observation.images.*) 까지 매번 로드/파싱. 필요한 컬럼만 projection.
          2) cache key 갱신 — 원본은 `self.curr_traj_id` 를 절대 재할당하지 않아
             같은 trajectory 연속 접근에도 cache miss 가 반복된다. 여기서 제대로 세팅.

        동치성: action 컬럼은 그대로 전부 포함되며 downstream `get_step_data` /
        `self.transforms` 는 `self.modality_keys["action"]` 만 사용하므로 byte-identical.
        """
        import pandas as pd

        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data

        chunk_index = self.get_episode_chunk(trajectory_id)
        parquet_path = self.dataset_path / self.data_path_pattern.format(
            episode_chunk=chunk_index, episode_index=trajectory_id
        )
        assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"

        needed_cols = self._get_needed_parquet_columns()
        df = (
            pd.read_parquet(parquet_path, columns=needed_cols)
            if needed_cols
            else pd.read_parquet(parquet_path)
        )

        self.curr_traj_id = trajectory_id
        self.curr_traj_data = df
        return df

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """에피소드 단위 train/val split 후 해당 split의 trajectory만 반환."""
        episode_path = self._dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]

        all_ids = np.array([e["episode_index"] for e in episode_metadata])
        all_lengths = np.array([e["length"] for e in episode_metadata])

        if self._split == "all":
            return all_ids, all_lengths

        n_total = len(all_ids)
        n_val = max(1, int(n_total * self._val_ratio))

        rng = np.random.default_rng(self._val_seed)
        shuffled = rng.permutation(n_total)

        if self._split == "val":
            selected = np.sort(shuffled[:n_val])
        else:  # "train"
            selected = np.sort(shuffled[n_val:])

        print(
            f"[ActionOnlyDataset][{self._split}] {Path(self._dataset_path).name}: "
            f"전체 {n_total}개 에피소드 중 {len(selected)}개 사용 "
            f"(val_ratio={self._val_ratio})"
        )

        return all_ids[selected], all_lengths[selected]

    def __getitem__(self, index: int) -> dict:
        """action tensor만 반환하도록 오버라이드.

        Returns:
            dict: {"action": Tensor[T, D]} (정규화된 action)
        """
        trajectory_id, base_index = self.all_steps[index]
        data = self.get_step_data(trajectory_id, base_index)
        data = self.transforms(data)
        # ConcatTransform 이후 "action" key만 존재
        return {"action": data["action"]}


class ActionOnlyCollator:
    """Action 텐서만 batch로 묶는 간단한 collator."""

    def __call__(self, features: list[dict]) -> dict:
        actions = torch.stack([f["action"] for f in features])
        return {"action": actions}  # [B, T, D]

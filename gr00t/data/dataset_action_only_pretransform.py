"""
PreTransformedActionOnlyDataset: 초기화 시 전체 데이터를 미리 transform하여 메모리에 캐싱.

ActionOnlyDataset과 동일한 출력을 보장하면서, __getitem__은 단순 텐서 인덱싱만 수행.
parquet I/O 및 transform 오버헤드를 완전히 제거하여 학습 속도를 크게 향상시킵니다.

ActionOnlyCollator와 호환됩니다 (출력: {"action": Tensor[T, D]}).
"""

from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from gr00t.data.dataset_action_only import ActionOnlyDataset


class PreTransformedActionOnlyDataset(Dataset):
    """초기화 시 전체 데이터를 미리 transform하여 메모리에 캐싱하는 Dataset.

    ActionOnlyDataset을 내부적으로 생성하여 모든 인덱스를 순회하며 결과를 하나의
    큰 텐서에 저장합니다. __getitem__은 단순 인덱싱만 수행합니다.

    Args:
        dataset_path: 데이터셋 경로
        data_config_name: DATA_CONFIG_MAP의 키 (e.g., "fourier_gr1_arms_waist")
        embodiment_tag: 로봇 태그 (e.g., "new_embodiment")
        split: "train", "val", 또는 "all"
        val_ratio: validation에 사용할 에피소드 비율
        val_seed: split 결정에 사용할 random seed
        normalization_mode: action 정규화 방식
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
    ):
        source = ActionOnlyDataset(
            dataset_path=dataset_path,
            data_config_name=data_config_name,
            embodiment_tag=embodiment_tag,
            split=split,
            val_ratio=val_ratio,
            val_seed=val_seed,
            normalization_mode=normalization_mode,
        )

        n = len(source)
        assert n > 0, f"Dataset is empty: {dataset_path}"

        # 첫 샘플로 shape 확인
        first = source[0]["action"]
        T, D = first.shape

        # 전체 데이터를 미리 transform하여 하나의 텐서에 저장
        cache = torch.empty(n, T, D, dtype=torch.float32)
        cache[0] = first
        for i in tqdm(range(1, n), desc=f"[PreTransform][{split}] Caching actions"):
            cache[i] = source[i]["action"]

        self._cache = cache
        print(
            f"[PreTransformedActionOnlyDataset][{split}] "
            f"Cached {n:,} samples, shape=({T}, {D}), "
            f"memory={cache.nbytes / 1024**2:.1f} MB"
        )

    def __len__(self) -> int:
        return self._cache.shape[0]

    def __getitem__(self, index: int) -> dict:
        return {"action": self._cache[index]}

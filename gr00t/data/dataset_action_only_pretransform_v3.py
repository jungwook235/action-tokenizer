"""PreTransformedActionOnlyDatasetV3: V3 variant of PreTransformedActionOnlyDataset.

Identical caching behavior to v2's :class:`PreTransformedActionOnlyDataset` but
constructs an :class:`ActionOnlyDatasetV3` underneath, so the train/val split
honors the persistent fixed-val file.
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from gr00t.data.dataset_action_only_v3 import ActionOnlyDatasetV3


class PreTransformedActionOnlyDatasetV3(Dataset):
    """V3 pre-transformed action-only dataset with fixed-val support."""

    def __init__(
        self,
        dataset_path: str | Path,
        data_config_name: str,
        embodiment_tag: str,
        split: str = "all",
        val_ratio: float = 0.003,
        val_seed: int = 42,
        normalization_mode: str = "min_max",
        use_fixed_val: bool = True,
        fixed_val_path: Optional[str] = None,
    ):
        source = ActionOnlyDatasetV3(
            dataset_path=dataset_path,
            data_config_name=data_config_name,
            embodiment_tag=embodiment_tag,
            split=split,
            val_ratio=val_ratio,
            val_seed=val_seed,
            normalization_mode=normalization_mode,
            use_fixed_val=use_fixed_val,
            fixed_val_path=fixed_val_path,
        )

        n = len(source)
        assert n > 0, f"Dataset is empty: {dataset_path}"

        first = source[0]["action"]
        T, D = first.shape

        cache = torch.empty(n, T, D, dtype=torch.float32)
        cache[0] = first
        for i in tqdm(range(1, n), desc=f"[PreTransform v3][{split}] Caching actions"):
            cache[i] = source[i]["action"]

        self._cache = cache
        print(
            f"[PreTransformedActionOnlyDatasetV3][{split}] "
            f"Cached {n:,} samples, shape=({T}, {D}), "
            f"memory={cache.nbytes / 1024**2:.1f} MB"
        )

    def __len__(self) -> int:
        return self._cache.shape[0]

    def __getitem__(self, index: int) -> dict:
        return {"action": self._cache[index]}

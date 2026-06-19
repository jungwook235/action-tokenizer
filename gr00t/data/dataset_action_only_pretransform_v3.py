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
        # Retain the source dataset (not just its cached actions) so merged
        # normalization stats can be re-applied after construction — see
        # ``set_transforms_metadata``. The source is a lightweight lazy dataset
        # (no big tensors held), so keeping it around is cheap.
        self._source = ActionOnlyDatasetV3(
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
        self._split = split
        self._build_cache()

    def _build_cache(self) -> None:
        """Cache every (already-normalized) action from the source dataset.

        Called once at construction, and again by ``set_transforms_metadata``
        when merged cross-dataset normalization is applied (so the cache holds
        the merged-normalized values, matching the on-the-fly / VLA paths).
        """
        source = self._source
        n = len(source)
        assert n > 0, f"Dataset is empty: {source._dataset_path}"

        first = source[0]["action"]
        T, D = first.shape

        cache = torch.empty(n, T, D, dtype=torch.float32)
        cache[0] = first
        for i in tqdm(range(1, n), desc=f"[PreTransform v3][{self._split}] Caching actions"):
            cache[i] = source[i]["action"]

        self._cache = cache
        print(
            f"[PreTransformedActionOnlyDatasetV3][{self._split}] "
            f"Cached {n:,} samples, shape=({T}, {D}), "
            f"memory={cache.nbytes / 1024**2:.1f} MB"
        )

    @property
    def metadata(self):
        """Delegate to the source so merge_norm_stats can read per-dataset stats."""
        return self._source.metadata

    def set_transforms_metadata(self, metadata) -> None:
        """Apply (merged) normalization metadata, then rebuild the cache so the
        cached actions reflect the updated statistics."""
        self._source.set_transforms_metadata(metadata)
        self._build_cache()

    def __len__(self) -> int:
        return self._cache.shape[0]

    def __getitem__(self, index: int) -> dict:
        return {"action": self._cache[index]}

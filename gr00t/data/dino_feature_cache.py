"""Shared on-disk cache for frozen visual (DINO / VGGT) features.

Both the Stage-1 action-latent tokenizer trainer and the Stage-2 VLA trainer feed
the SAME frozen extractor the SAME (frame_x0, frame_x1) pair for each action
chunk. The features are deterministic (frozen model, frozen preprocessing), so we
can precompute them once and look them up by ``(episode_id, base_index)`` at
train time — skipping BOTH video decoding and the DINO forward.

Granularity: one feature map per *video frame row* of each episode (NOT per
sample). Adjacent action chunks share frames, so per-row storage avoids massive
duplication. For a chunk starting at ``base_index`` with horizon ``H``:

    x0 = row  clip(base_index,         0, L-1)   ( == base_index in practice )
    x1 = row  clip(base_index + H - 1, 0, L-1)

mirroring ``LeRobotSingleDataset.get_video`` (dataset.py) and
``ActionFramesDatasetV4._video_indices = [0, H-1]``.

Layout (per dataset root):

    <dataset>/dino_feature_cache/<key>/
        meta.json            # everything the feature VALUES depend on
        ep_000000.npy        # [L, Lp, C] float16   (L = episode length)
        ep_000001.npy
        ...

``<key>`` encodes every value-affecting setting so configs never collide, e.g.::

    dino__dinov2-large__naive__img224__front

Precision: the live trainer feeds the model the DINO grid as **float32** (with the
naive/affine final-norm, autocast keeps LayerNorm in fp32, so ``_extract_feats``'s
trailing ``.float()`` is a no-op). The cache therefore stores float32 by default,
which is bit-identical to live training. A float16 cache (``--store-dtype
float16`` in the precompute script) halves the size but adds ~2e-4 relative
rounding to the features — NOT identical. ``meta.json["dtype"]`` records which was
used; the reader returns that dtype and the dataset casts to the model dtype.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

CACHE_ROOT_NAME = "dino_feature_cache"


def _slug(s: str) -> str:
    """Filesystem-safe slug: drop a leading 'video.' / org prefix, replace '/'."""
    s = s.replace("video.", "")
    s = s.split("/")[-1]  # 'facebook/dinov2-large' -> 'dinov2-large'
    return s.replace("/", "_").replace(" ", "_")


def make_cache_key(
    *,
    feature_source: str,
    model_name: str,
    final_norm: str,
    image_size: int,
    video_key: str,
) -> str:
    """Stable directory key for one (extractor, preprocessing, camera) combo.

    Any change that alters the stored feature values must change the key so two
    different settings can never share a cache directory.
    """
    return (
        f"{feature_source}__{_slug(model_name)}__{final_norm}"
        f"__img{int(image_size)}__{_slug(video_key)}"
    )


def get_cache_dir(dataset_path: str | Path, key: str) -> Path:
    return Path(dataset_path) / CACHE_ROOT_NAME / key


def episode_file(cache_dir: str | Path, episode_id: int) -> Path:
    return Path(cache_dir) / f"ep_{int(episode_id):06d}.npy"


# ---------------------------------------------------------------------------
# Writer (used by scripts/precompute_dino_features.py)
# ---------------------------------------------------------------------------


def write_meta(cache_dir: str | Path, meta: dict) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def read_meta(cache_dir: str | Path) -> dict:
    with open(Path(cache_dir) / "meta.json", "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reader (used by the cached datasets at train time)
# ---------------------------------------------------------------------------


class DinoFeatureCacheReader:
    """Read precomputed per-row features for one dataset.

    Opens each episode's ``.npy`` lazily with ``mmap_mode='r'`` so DataLoader
    workers only fault in the rows they touch. Handles are cached per process,
    so each forked worker keeps its own (fork-safe — nothing is opened until the
    first ``get_pair`` inside the worker).

    Args:
        dataset_path: dataset root (the cache lives under
            ``<dataset_path>/dino_feature_cache/<key>``).
        key: directory key from :func:`make_cache_key`.
        action_horizon: chunk length H (x1 row = clip(base+H-1, 0, L-1)).
        expect: optional dict of value-affecting settings to assert against the
            cache's ``meta.json`` (feature_source / model_name / final_norm /
            image_size / video_key) so a stale/mismatched cache fails loudly.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        key: str,
        action_horizon: int,
        expect: Optional[dict] = None,
    ):
        self.cache_dir = get_cache_dir(dataset_path, key)
        assert self.cache_dir.is_dir(), (
            f"DINO feature cache not found: {self.cache_dir}. "
            f"Run scripts/precompute_dino_features.py first."
        )
        self.action_horizon = int(action_horizon)
        self.meta = read_meta(self.cache_dir)

        if expect is not None:
            for k, v in expect.items():
                got = self.meta.get(k)
                # image_size compared as int; others as str.
                if k == "image_size":
                    ok = int(got) == int(v)
                else:
                    ok = str(got) == str(v)
                assert ok, (
                    f"DINO cache mismatch at {self.cache_dir}: meta[{k!r}]={got!r} "
                    f"but training expects {v!r}. Rebuild the cache for this config."
                )

        self.embed_dim = int(self.meta["embed_dim"])
        self.num_patches = int(self.meta["num_patches"])
        self._handles: dict[int, np.ndarray] = {}

    def _arr(self, episode_id: int) -> np.ndarray:
        h = self._handles.get(episode_id)
        if h is None:
            path = episode_file(self.cache_dir, episode_id)
            assert path.is_file(), f"Missing cache file: {path}"
            h = np.load(path, mmap_mode="r")  # [L, Lp, C] float16
            self._handles[episode_id] = h
        return h

    def get_pair(self, episode_id: int, base_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (x0_feat, x1_feat) as contiguous float16 ``[Lp, C]`` copies.

        Copies out of the mmap so downstream ``torch.from_numpy``/pin-memory does
        not alias a read-only memory map.
        """
        arr = self._arr(episode_id)
        L = arr.shape[0]
        i0 = min(max(base_index, 0), L - 1)
        i1 = min(max(base_index + self.action_horizon - 1, 0), L - 1)
        x0 = np.ascontiguousarray(arr[i0])
        x1 = np.ascontiguousarray(arr[i1])
        return x0, x1

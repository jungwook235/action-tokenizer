"""EgoDexActionFramesDataset: a non-LeRobot V4 dataset for the EgoDex corpus.

It yields the SAME per-item dict as :class:`ActionFramesDatasetV4`::

    {"action": FloatTensor [H, D], "frame_x0": uint8 [S, S, 3], "frame_x1": uint8 [S, S, 3]}

so it drops straight into the multi-embodiment V4 tokenizer pipeline
(``EmbodimentTaggedDataset`` → ``ActionFramesCollatorV4`` → trainer → model) with
no other code changes. The tokenizer Stage-1 objective is reconstruction + DINO
feature prediction and does NOT consume any language instruction, so EgoDex task
metadata is intentionally ignored here.

Per episode the EgoDex layout is (see EGODEX_DATASET_NOTES.md)::

    <task_folder>/<N>.mp4          egocentric RGB (1920x1080, 30fps), frame t ↔ state t
    <task_folder>/<N>_mano.hdf5    contains ``gr1_state`` (T, 44): GR1 retargeted joints

For an observation window at frame ``start`` (horizon ``H``):
  * frame_x0 = video frame ``start``
  * frame_x1 = video frame ``start+H-1``
  * action   = ``gr1_state[start+offset : start+offset+H]``
``action_offset`` (default 0) is the frame→action gap: 0 aligns ``action[0]`` with
``frame_x0`` (matching ``ActionFramesDatasetV4`` / dexjoco ``action_indices=range(16)``);
1 makes the action the NEXT-step state chunk while the frames stay anchored.

Image preprocessing is byte-identical to ``ActionFramesDatasetV4``:
decord RGB uint8 → ``float/255`` → torchvision ``Resize(S, BILINEAR, antialias)``
→ uint8. The frozen DINO extractor (run by the trainer) then applies its own
ImageNet normalization uniformly across embodiments, so EgoDex frames reach DINO
exactly like dexjoco/gr1 frames.

Action normalization is ``min_max`` to [-1, 1], computed exactly like
``StateActionTransform`` (``2*(x-min)/(max-min)-1``; dims with ``min==max`` → 0;
no clipping). Stats and the (expensive) episode scan are cached to disk and built
rank-safely under DDP.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import decord
import numpy as np
import torch
import torchvision.transforms.v2 as TV

try:
    import h5py
except ImportError as e:  # pragma: no cover - h5py is required for EgoDex
    raise ImportError("EgoDexActionFramesDataset requires h5py") from e


_CACHE_VERSION = 1


def _local_rank() -> int:
    for k in ("LOCAL_RANK", "RANK"):
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return 0


class EgoDexActionFramesDataset(torch.utils.data.Dataset):
    """Action chunk + (x0, x1) frames sampled from EgoDex task folders.

    Args:
        dataset_paths: list of EgoDex *task* folders (each holds ``N.mp4`` +
            ``N_mano.hdf5`` pairs). One dataset instance owns ALL folders and
            normalizes over their union — pick the tasks you want via this list.
        action_horizon: chunk length H (must match the other embodiments in a
            joint run, e.g. 16 for dexjoco_dual_arm_front).
        action_key: hdf5 dataset key used as the action (default ``gr1_state``).
        action_offset: window start offset (default 0 → chunk begins at frame_x0).
        stride: window stride within an episode (default = action_horizon →
            non-overlapping chunks). Controls dataset size.
        image_size: square resize target fed to DINO (default 224).
        split / val_ratio / val_seed: episode-level train/val split. Stats are
            computed over the FULL kept-episode union (split-independent) so the
            train and val instances normalize identically.
        video_suffix: video filename suffix (default ``.mp4``; pass
            ``_resized.mp4`` to read the lighter resized clips — both are resized
            to ``image_size`` anyway).
        stats_max_episodes: cap on episodes scanned for min/max stats (deterministic
            even sampling). ``None`` → use all kept episodes. Default 3000.
        cache_dir: where to cache the episode scan + stats (default
            ``~/.cache/egodex_actlat_v4``).
        video_backend: kept for API symmetry; decord is always used (it returns
            RGB, matching the LeRobot path — opencv would be BGR).
    """

    def __init__(
        self,
        dataset_paths: list[str] | str,
        action_horizon: int = 16,
        action_key: str = "gr1_state",
        action_offset: int = 0,
        stride: Optional[int] = None,
        image_size: int = 224,
        split: str = "all",
        val_ratio: float = 0.003,
        val_seed: int = 42,
        video_suffix: str = ".mp4",
        stats_max_episodes: Optional[int] = 3000,
        cache_dir: Optional[str] = None,
        video_backend: str = "decord",
    ):
        assert split in ("train", "val", "all"), f"split must be train/val/all: {split}"
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]
        self._paths = [str(Path(p).resolve()) for p in dataset_paths]
        for p in self._paths:
            assert os.path.isdir(p), f"[egodex] not a directory: {p}"

        self.action_horizon = int(action_horizon)
        self.action_key = str(action_key)
        self.action_offset = int(action_offset)
        self.stride = int(stride) if stride is not None else self.action_horizon
        assert self.stride >= 1, "stride must be >= 1"
        self.image_size = int(image_size)
        self._split = split
        self._val_ratio = float(val_ratio)
        self._val_seed = int(val_seed)
        self._video_suffix = str(video_suffix)
        self._stats_max_episodes = stats_max_episodes
        self._cache_dir = Path(
            cache_dir or os.path.expanduser("~/.cache/egodex_actlat_v4")
        )

        # Resize identical to ActionFramesDatasetV4: VideoResize(linear→BILINEAR,
        # antialias=True) to a square. Runs on float [N,C,H,W] in [0,1].
        self._resize = TV.Resize(
            (self.image_size, self.image_size),
            interpolation=TV.InterpolationMode.BILINEAR,
            antialias=True,
        )

        # ── episode scan + stats (cached, rank-safe) ──
        cache = self._load_or_build_cache()
        # episodes: list of (mano_path, mp4_path, T)
        self._episodes = [(m, v, int(T)) for (m, v, T) in cache["episodes"]]
        self._min = torch.tensor(cache["stats"]["min"], dtype=torch.float32)
        self._max = torch.tensor(cache["stats"]["max"], dtype=torch.float32)
        self.action_dim = int(self._min.numel())

        # ── episode-level split + window index ──
        ep_ids = self._split_episode_ids(len(self._episodes))
        self._windows = self._build_windows(ep_ids)
        assert len(self._windows) > 0, (
            f"[egodex][{self._split}] no windows built — episodes too short for "
            f"horizon={self.action_horizon} (offset={self.action_offset})?"
        )
        print(
            f"[EgoDexActionFramesDataset][{self._split}] folders={len(self._paths)} "
            f"episodes(kept)={len(self._episodes)} used_eps={len(ep_ids)} "
            f"windows={len(self._windows)} action_dim={self.action_dim} "
            f"horizon={self.action_horizon} stride={self.stride}"
        )

    # ------------------------------------------------------------------ cache
    def _cache_key(self) -> str:
        payload = {
            "v": _CACHE_VERSION,
            "folders": sorted(self._paths),
            "action_key": self.action_key,
            "video_suffix": self._video_suffix,
            "stats_max_episodes": self._stats_max_episodes,
        }
        h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return h[:16]

    def _load_or_build_cache(self) -> dict:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_dir / f"egodex_{self._cache_key()}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass  # corrupt/partial → rebuild

        rank = _local_rank()
        if rank != 0:
            # Non-primary rank: wait for rank 0 to publish the cache; fall back to
            # building it ourselves if it never appears (deterministic → same result).
            for _ in range(900):  # up to ~30 min
                if cache_file.exists():
                    try:
                        with open(cache_file, "r") as f:
                            return json.load(f)
                    except (json.JSONDecodeError, OSError):
                        break
                time.sleep(2)

        episodes = self._scan_episodes()
        assert episodes, f"[egodex] no (N{self._video_suffix} + N_mano.hdf5) pairs found in {self._paths}"
        stats = self._compute_stats(episodes)
        cache = {"episodes": episodes, "stats": stats}
        # Atomic publish (write tmp then replace) so concurrent readers never see
        # a half-written file. Unique tmp per rank/pid to avoid clobbering.
        tmp = cache_file.with_suffix(f".tmp.{rank}.{os.getpid()}")
        try:
            with open(tmp, "w") as f:
                json.dump(cache, f)
            os.replace(tmp, cache_file)
        except OSError:
            if tmp.exists():
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        return cache

    def _scan_episodes(self) -> list:
        """Find ``N.mp4`` + ``N_mano.hdf5`` pairs; record (mano, mp4, T) where
        T = len(action_key). Episodes with T < horizon+offset are dropped."""
        min_len = self.action_horizon + self.action_offset
        episodes = []
        for folder in sorted(self._paths):
            fp = Path(folder)
            mano_files = sorted(
                fp.glob("*_mano.hdf5"),
                key=lambda p: self._episode_num(p.name),
            )
            for mano in mano_files:
                stem = mano.name[: -len("_mano.hdf5")]  # "N"
                mp4 = fp / f"{stem}{self._video_suffix}"
                if not mp4.exists():
                    continue
                try:
                    with h5py.File(mano.as_posix(), "r") as f:
                        if self.action_key not in f:
                            continue
                        T = int(f[self.action_key].shape[0])
                except OSError:
                    continue
                if T >= min_len:
                    episodes.append([mano.as_posix(), mp4.as_posix(), T])
        return episodes

    @staticmethod
    def _episode_num(name: str) -> tuple:
        # sort "12_mano.hdf5" numerically when possible, else lexicographically
        stem = name.split("_mano.hdf5")[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    def _compute_stats(self, episodes: list) -> dict:
        """Global per-dim min/max of action_key over (a deterministic sample of)
        episodes. Loads each chosen episode's full array and reduces."""
        n = len(episodes)
        idxs = np.arange(n)
        if self._stats_max_episodes is not None and n > int(self._stats_max_episodes):
            idxs = np.linspace(0, n - 1, int(self._stats_max_episodes)).astype(int)
            idxs = np.unique(idxs)
        gmin = gmax = None
        for i in idxs:
            mano = episodes[int(i)][0]
            try:
                with h5py.File(mano, "r") as f:
                    a = np.asarray(f[self.action_key][:], dtype=np.float64)  # [T, D]
            except OSError:
                continue
            if a.ndim != 2 or a.shape[0] == 0:
                continue
            mn, mx = a.min(axis=0), a.max(axis=0)
            gmin = mn if gmin is None else np.minimum(gmin, mn)
            gmax = mx if gmax is None else np.maximum(gmax, mx)
        assert gmin is not None, "[egodex] failed to compute action stats (no readable episodes)"
        return {"min": gmin.astype(np.float64).tolist(),
                "max": gmax.astype(np.float64).tolist()}

    # ------------------------------------------------------------ split/index
    def _split_episode_ids(self, n_total: int) -> np.ndarray:
        ids = np.arange(n_total)
        if self._split == "all" or n_total == 0:
            return ids
        n_val = max(1, int(n_total * self._val_ratio))
        rng = np.random.default_rng(self._val_seed)
        shuffled = rng.permutation(n_total)
        if self._split == "val":
            return np.sort(shuffled[:n_val])
        return np.sort(shuffled[n_val:])

    def _build_windows(self, ep_ids: np.ndarray) -> np.ndarray:
        """[(episode_index, start), ...] as int64 [N, 2]. start spans
        [0, T - H - offset] with the configured stride; T-H-offset always
        included so each episode contributes its final aligned chunk."""
        H, off, stride = self.action_horizon, self.action_offset, self.stride
        rows = []
        for ep in ep_ids:
            T = self._episodes[int(ep)][2]
            last = T - H - off  # inclusive max start
            if last < 0:
                continue
            starts = list(range(0, last + 1, stride))
            if starts[-1] != last:
                starts.append(last)
            for s in starts:
                rows.append((int(ep), int(s)))
        if not rows:
            return np.zeros((0, 2), dtype=np.int64)
        return np.asarray(rows, dtype=np.int64)

    # --------------------------------------------------------------- read ops
    def _normalize_action(self, arr: np.ndarray) -> torch.Tensor:
        """min_max → [-1, 1]; dims with min==max → 0; no clipping (mirrors
        StateActionTransform)."""
        x = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)  # [H, D]
        denom = self._max - self._min
        mask = denom != 0
        out = torch.zeros_like(x)
        out[..., mask] = (x[..., mask] - self._min[mask]) / denom[mask]
        out[..., mask] = 2 * out[..., mask] - 1
        return out

    def _read_two_frames(self, mp4_path: str, i0: int, i1: int) -> np.ndarray:
        """decord RGB read of two frame indices (clamped to video length).
        Returns uint8 [2, H, W, 3]."""
        vr = decord.VideoReader(mp4_path)
        n = len(vr)
        i0 = min(max(i0, 0), n - 1)
        i1 = min(max(i1, 0), n - 1)
        frames = vr.get_batch([i0, i1]).asnumpy()  # [2, H, W, 3] uint8 RGB
        del vr
        return frames

    def _preprocess_frames(self, frames: np.ndarray) -> np.ndarray:
        """Identical to VideoToTensor → VideoResize → VideoToNumpy."""
        t = torch.from_numpy(frames).to(torch.float32) / 255.0  # [2,H,W,C]
        t = t.permute(0, 3, 1, 2)  # [2,C,H,W]
        t = self._resize(t)  # [2,C,S,S]
        out = (t.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()  # [2,S,S,C]
        return out

    # ------------------------------------------------------------------ torch
    def __len__(self) -> int:
        return int(self._windows.shape[0])

    def __getitem__(self, index: int) -> dict:
        ep_idx, start = self._windows[index]
        mano_path, mp4_path, T = self._episodes[int(ep_idx)]
        H = self.action_horizon
        fb = int(start)                   # frame (observation) base index
        ab = fb + self.action_offset      # action base; offset=1 → next-step action

        with h5py.File(mano_path, "r") as f:
            action_np = np.asarray(f[self.action_key][ab : ab + H], dtype=np.float32)  # [<=H, D]
        if action_np.shape[0] < H:  # defensive (index filter should prevent this)
            pad = np.repeat(action_np[-1:], H - action_np.shape[0], axis=0)
            action_np = np.concatenate([action_np, pad], axis=0)
        action = self._normalize_action(action_np)  # [H, D] float32

        # frames stay anchored to the observation window [fb, fb+H-1]; only the
        # action chunk shifts by action_offset (so offset=1 = "next-step state").
        frames = self._read_two_frames(mp4_path, fb, fb + H - 1)
        frames = self._preprocess_frames(frames)  # [2, S, S, 3] uint8

        return {
            "action": action,
            "frame_x0": np.ascontiguousarray(frames[0]),
            "frame_x1": np.ascontiguousarray(frames[1]),
        }

    def set_epoch(self, epoch):  # parity with ActionFramesDatasetV4 wrappers
        self._epoch = epoch

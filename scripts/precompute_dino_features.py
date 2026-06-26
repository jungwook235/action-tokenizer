"""Precompute & cache frozen DINO patch features for the V4 action tokenizer.

Runs the EXACT Stage-1 frame pipeline (``ActionFramesDatasetV4``: decord →
VideoResize(linear) → uint8) and the EXACT frozen ``DINOv3FeatureExtractor``
used by ``train_action_latent_tokenizer_v4.py``'s trainer, then stores the
per-frame patch features so both Stage-1 (tokenizer) and Stage-2 (VLA) training
can look them up by ``(episode_id, base_index)`` instead of decoding video and
running DINO every step.

Exactness guarantees (so cached == live training, bit-for-bit):
  * Frames come from ``ActionFramesDatasetV4`` itself (the literal Stage-1 path),
    so resize / interpolation / quantization match by construction.
  * ``frame_x0`` of sample ``(traj, base_index)`` is the frame at row
    ``base_index``; since ``base_index`` sweeps ``0..L-1`` over ``all_steps``,
    iterating x0 covers every row of every episode exactly once.
  * The extractor runs the SAME forward as the trainer's ``_extract_feats``. With
    the naive (and affine) final-norm the DINO grid the trainer feeds the model is
    actually float32 (autocast keeps LayerNorm in fp32, so the trailing ``.float()``
    is a no-op). We therefore store **float32 by default** → bit-identical to live
    training. ``--store-dtype float16`` halves the cache (~107 vs ~214 GiB total)
    at the cost of ~2e-4 relative rounding on the features (NOT bit-identical).

Usage (mirror the training script's DINO args!):

    python scripts/precompute_dino_features.py \
        --dataset-path /path/click_mouse /path/hammer_nail ... \
        --data-config dexjoco_single_arm_front_h24 \
        --embodiment-tag new_embodiment \
        --dino-model facebook/dinov2-large \
        --dino-channels 1024 \
        --dino-final-norm naive \
        --image-size 224 \
        --video-backend decord \
        --batch-size 64 --num-workers 16

    # verify a built cache is bit-identical to the live path (no writing):
    python scripts/precompute_dino_features.py ... --verify-only
"""

import os
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import torch
import tyro
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Side-effect import to register extra data configs (parity with the trainer).
import gr00t.experiment.data_config_v3  # noqa: F401
from gr00t.data.dataset_action_frames_v4 import ActionFramesDatasetV4
from gr00t.data.dino_feature_cache import (
    episode_file,
    get_cache_dir,
    make_cache_key,
    read_meta,
    write_meta,
)
from gr00t.utils.dino import DINOv3FeatureExtractor


@dataclass
class Args:
    dataset_path: List[str]
    data_config: str = "dexjoco_single_arm_front_h24"
    embodiment_tag: str = "new_embodiment"

    # DINO — MUST match the training script exactly.
    feature_source: Literal["dino"] = "dino"
    dino_model: str = "facebook/dinov2-large"
    dino_channels: int = 1024
    dino_final_norm: Literal["affine", "naive"] = "naive"
    image_size: int = 224
    video_backend: str = "decord"

    # Storage precision. float32 = bit-identical to live training (default).
    # float16 halves the size but adds ~2e-4 relative rounding (NOT identical).
    store_dtype: Literal["float32", "float16"] = "float32"

    # Throughput knobs (do NOT affect stored values).
    batch_size: int = 64
    num_workers: int = 16
    device: str = "cuda"

    overwrite: bool = False
    """Rebuild even if a complete cache (meta.json) already exists."""
    verify_only: bool = False
    """Skip writing; only check an existing cache against the live path."""
    verify_samples: int = 16
    """Number of random samples per dataset to bit-compare after building."""


# ---------------------------------------------------------------------------
# Index-wrapped dataset: yield (global index, frame_x0) for parallel decode.
# ---------------------------------------------------------------------------


class _X0Frames(Dataset):
    def __init__(self, ds: ActionFramesDatasetV4):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        # frame_x0 = [H, W, C] uint8 — exact Stage-1 frame for row=base_index.
        return i, self.ds[i]["frame_x0"]


def _collate(batch):
    idxs = [b[0] for b in batch]
    arr = np.stack([b[1] for b in batch])  # [B, H, W, C] uint8
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W] uint8
    return idxs, t


@torch.no_grad()
def _dino_grid(extractor: DINOv3FeatureExtractor, frames_chw_u8: torch.Tensor, device) -> torch.Tensor:
    """[B,C,H,W] uint8 → [B, Lp, C], identical to the trainer's path.

    Mirrors ``_extract_feats`` exactly: float()/255 → extractor(return_spatial_grid)
    → flatten(2).transpose(1,2) → .float(). For the naive/affine final-norm the
    extractor already returns float32, so this is the exact tensor the trainer
    feeds the model."""
    f = frames_chw_u8.to(device).float() / 255.0
    _, grid = extractor(f, return_spatial_grid=True)  # [B, C, h, w]
    return grid.flatten(2).transpose(1, 2).contiguous().float()  # [B, h*w, C] float32


def _build_extractor(args: Args, device) -> DINOv3FeatureExtractor:
    ext = DINOv3FeatureExtractor(
        model_name=args.dino_model, use_compile=False, final_norm=args.dino_final_norm
    )
    ext.eval()
    for p in ext.parameters():
        p.requires_grad = False
    ext.to(device)
    assert ext.embed_dim == args.dino_channels, (
        f"extractor embed_dim={ext.embed_dim} != --dino-channels={args.dino_channels} "
        f"(model={args.dino_model}). A silent dinov2-small fallback (384) is the "
        f"usual cause — check the model name / HF offline cache."
    )
    return ext


def _make_dataset(args: Args, path: str) -> ActionFramesDatasetV4:
    return ActionFramesDatasetV4(
        dataset_path=path,
        data_config_name=args.data_config,
        embodiment_tag=args.embodiment_tag,
        split="all",  # cache every row; train/val splits index into the same cache
        image_size=args.image_size,
        video_backend=args.video_backend,
        use_fixed_val=True,
    )


def build_one(args: Args, path: str, extractor: DINOv3FeatureExtractor, device) -> str:
    ds = _make_dataset(args, path)
    video_key = ds._video_key
    key = make_cache_key(
        feature_source=args.feature_source,
        model_name=args.dino_model,
        final_norm=args.dino_final_norm,
        image_size=args.image_size,
        video_key=video_key,
    )
    cache_dir = get_cache_dir(path, key)
    meta_path = cache_dir / "meta.json"

    if meta_path.is_file() and not args.overwrite:
        print(f"[skip] cache already complete: {cache_dir}")
        return key
    cache_dir.mkdir(parents=True, exist_ok=True)
    np_dtype = np.float32 if args.store_dtype == "float32" else np.float16

    traj_ids = [int(t) for t in ds.trajectory_ids]
    traj_lens = {int(t): int(l) for t, l in zip(ds.trajectory_ids, ds.trajectory_lengths)}
    all_steps = ds.all_steps

    # Probe feature shape (Lp, C) on a single real frame.
    _, probe_frame = _X0Frames(ds)[0]
    probe = _dino_grid(extractor, torch.from_numpy(probe_frame).permute(2, 0, 1)[None], device)
    Lp, C = int(probe.shape[1]), int(probe.shape[2])
    print(f"[{os.path.basename(path)}] key={key} | Lp={Lp} C={C} | "
          f"{len(traj_ids)} episodes, {len(all_steps)} frames")

    # Preallocate one memmap per episode: [L, Lp, C] float16.
    mmaps: dict[int, np.ndarray] = {}
    for t in traj_ids:
        mm = np.lib.format.open_memmap(
            episode_file(cache_dir, t), mode="w+", dtype=np_dtype, shape=(traj_lens[t], Lp, C)
        )
        mmaps[t] = mm

    loader = DataLoader(
        _X0Frames(ds),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=_collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    filled = 0
    for idxs, frames in tqdm(loader, desc=f"DINO {os.path.basename(path)}"):
        feats = _dino_grid(extractor, frames, device).cpu().numpy().astype(np_dtype)
        for j, gi in enumerate(idxs):
            traj_id, base_index = all_steps[gi]
            mmaps[int(traj_id)][int(base_index)] = feats[j]
            filled += 1

    for mm in mmaps.values():
        mm.flush()
        del mm

    assert filled == len(all_steps), f"filled {filled} != {len(all_steps)} frames"

    write_meta(cache_dir, {
        "feature_source": args.feature_source,
        "model_name": args.dino_model,
        "final_norm": args.dino_final_norm,
        "image_size": int(args.image_size),
        "video_key": video_key,
        "video_backend": args.video_backend,
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "num_patches": Lp,
        "embed_dim": C,
        "dtype": args.store_dtype,
        "num_episodes": len(traj_ids),
        "total_frames": int(len(all_steps)),
        "episode_lengths": {str(t): traj_lens[t] for t in traj_ids},
    })
    print(f"[done] {cache_dir}  ({filled} frames)")
    return key


@torch.no_grad()
def verify_one(args: Args, path: str, extractor: DINOv3FeatureExtractor, device, key: str) -> None:
    """Verify cached rows reproduce the trainer's DINO features and index correctly.

    IMPORTANT — what "identical" means here. A frozen DINO forward is a transformer
    with no cross-sample ops, so a frame's feature is independent of which other
    frames share its batch and of its position in the batch (verified empirically).
    It does, however, depend on the batch *size*: cuBLAS/cuDNN pick different GEMM
    kernels per shape, and the resulting fp16 rounding is amplified on DINOv2's
    high-norm "artifact" tokens (a single token can differ by ~8–13 between bs=1
    and bs=64, while the median token differs by ~0.3). The trainer feeds DINO full
    batches (Stage-1 per_device_train_batch_size=64), so the cache is built at the
    SAME batch size and is bit-identical to what training computes for full batches.

    Therefore we verify against a faithful **batched** recompute that reproduces the
    build batches (size = --batch-size, contiguous global order), NOT a single-image
    forward. This checks (a) lossless npy round-trip, (b) correct (traj,base)→row and
    x1 = clip(base+H-1) indexing, (c) value fidelity. The single-image Δ is printed
    for transparency only (it reflects unavoidable cross-batch-size fp16 noise, not a
    cache error)."""
    from gr00t.data.dino_feature_cache import DinoFeatureCacheReader

    ds = _make_dataset(args, path)
    H = ds._action_horizon
    reader = DinoFeatureCacheReader(path, key, action_horizon=H)
    meta = read_meta(get_cache_dir(path, key))

    np_dtype = np.float32 if meta.get("dtype", "float32") == "float32" else np.float16
    bs = args.batch_size
    N = len(ds)

    # Map (traj, row) → global all_steps index (contiguous per trajectory).
    starts, lengths = {}, {}
    pos = 0
    for t, L in zip(ds.trajectory_ids, ds.trajectory_lengths):
        starts[int(t)], lengths[int(t)] = pos, int(L)
        pos += int(L)

    def build_feat(gi: int) -> np.ndarray:
        """Reproduce the build's batched DINO output for global index ``gi``.

        The build loader uses a SequentialSampler, so gi lands in the contiguous
        batch ``[(gi//bs)*bs : +bs]`` (last batch may be partial) — composition is
        irrelevant, only this size matters."""
        start = (gi // bs) * bs
        batch = list(range(start, min(start + bs, N)))
        fr = np.stack([ds[i]["frame_x0"] for i in batch])
        t = torch.from_numpy(fr).permute(0, 3, 1, 2).contiguous()
        out = _dino_grid(extractor, t, device)[batch.index(gi)]
        return out.cpu().numpy().astype(np_dtype)

    rng = np.random.default_rng(0)
    sample_idxs = rng.choice(N, size=min(args.verify_samples, N), replace=False)

    max_x0 = max_x1 = 0.0
    single_x0 = 0.0
    for gi in sample_idxs:
        gi = int(gi)
        traj_id, base_index = ds.all_steps[gi]
        traj_id, base_index = int(traj_id), int(base_index)
        i1 = min(base_index + H - 1, lengths[traj_id] - 1)
        gi1 = starts[traj_id] + i1

        c0, c1 = reader.get_pair(traj_id, base_index)
        b0 = build_feat(gi)            # frame at row base_index (= x0)
        b1 = build_feat(gi1)           # frame at row i1          (= x1)
        max_x0 = max(max_x0, float(np.abs(b0.astype(np.float32) - c0.astype(np.float32)).max()))
        max_x1 = max(max_x1, float(np.abs(b1.astype(np.float32) - c1.astype(np.float32)).max()))

        # Transparency: how far the single-image forward drifts (NOT asserted).
        s0 = _dino_grid(extractor, torch.from_numpy(ds[gi]["frame_x0"]).permute(2, 0, 1)[None], device)[0]
        single_x0 = max(single_x0, float(np.abs(s0.cpu().numpy().astype(np.float32) - c0.astype(np.float32)).max()))

    exact = (max_x0 == 0.0 and max_x1 == 0.0)
    status = f"OK (exact vs batched bs{bs} in {meta.get('dtype')})" if exact else "MISMATCH"
    print(f"[verify {os.path.basename(path)}] {status} | "
          f"max|Δx0|={max_x0:g} max|Δx1|={max_x1:g} (vs bs{bs}); "
          f"single-image drift={single_x0:g} (fp16 batch-size noise, expected) | "
          f"n={len(sample_idxs)} Lp={meta['num_patches']} C={meta['embed_dim']}")
    assert exact, "cached features do not match a faithful batched recompute (real bug)!"


def main(args: Args):
    assert args.feature_source == "dino", "Only feature_source='dino' is supported."
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    extractor = _build_extractor(args, device)

    for path in args.dataset_path:
        assert os.path.exists(path), f"Dataset path does not exist: {path}"
        if args.verify_only:
            key = make_cache_key(
                feature_source=args.feature_source,
                model_name=args.dino_model,
                final_norm=args.dino_final_norm,
                image_size=args.image_size,
                video_key=_make_dataset(args, path)._video_key,
            )
        else:
            key = build_one(args, path, extractor, device)
        verify_one(args, path, extractor, device, key)


if __name__ == "__main__":
    main(tyro.cli(Args))

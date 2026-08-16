"""Verify the per-frame mask conversion (correctness / completeness / speed).

  1. Correctness: >=6 datasets x 5 episodes x 10 random steps — the frame produced
     by the legacy path (z['mask'][t]) and the per-frame path
     (unpackbits(z[f'f{t}'])[:H*W].reshape(H,W)) must be np.array_equal.
  2. Completeness: masks/ and masks_pf/ hold the same COUNT and the same SET of
     relative episode paths.
  3. Speed: 100 random (episode, step) loads per path; per-frame must be >=5x faster.

Exits nonzero on any failure. Prints a summary table.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np


def load_legacy(path, t):
    with np.load(path) as z:
        arr = z["mask"]
        return arr[min(t, arr.shape[0] - 1)]


def load_pf(path, t):
    with np.load(path) as z:
        T, H, W = (int(v) for v in z["meta"])
        row = z[f"f{min(t, T - 1)}"]
        return np.unpackbits(row)[: H * W].reshape(H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--src-subdir", default="masks")
    ap.add_argument("--dst-subdir", default="masks_pf")
    ap.add_argument("--datasets", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--speed-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.root)
    results = {}

    # ---- 2. completeness (do first: cheap, and correctness sampling relies on pairs)
    src_set = {p.relative_to(root).as_posix().replace(f"/{args.src_subdir}/", "/", 1)
               for p in root.glob(f"*/{args.src_subdir}/chunk-*/*/episode_*.npz")}
    dst_set = {p.relative_to(root).as_posix().replace(f"/{args.dst_subdir}/", "/", 1)
               for p in root.glob(f"*/{args.dst_subdir}/chunk-*/*/episode_*.npz")}
    comp_ok = src_set == dst_set and len(src_set) > 0
    results["completeness"] = (
        f"src={len(src_set)} dst={len(dst_set)} set_equal={src_set == dst_set}",
        comp_ok,
    )

    def pair(rel):
        ds, rest = rel.split("/", 1)
        return (root / ds / args.src_subdir / rest, root / ds / args.dst_subdir / rest)

    # ---- 1. correctness
    by_ds = {}
    for rel in src_set & dst_set:
        by_ds.setdefault(rel.split("/", 1)[0], []).append(rel)
    ds_pick = rng.sample(sorted(by_ds), min(args.datasets, len(by_ds)))
    n_checked, mismatches = 0, []
    for ds in ds_pick:
        for rel in rng.sample(sorted(by_ds[ds]), min(args.episodes, len(by_ds[ds]))):
            src, dst = pair(rel)
            with np.load(src) as z:
                T = z["mask"].shape[0]
            for t in rng.sample(range(T), min(args.steps, T)):
                if not np.array_equal(load_legacy(src, t), load_pf(dst, t)):
                    mismatches.append(f"{rel} t={t}")
                n_checked += 1
    corr_ok = not mismatches
    results["correctness"] = (
        f"{len(ds_pick)} datasets, {n_checked} frames checked, mismatches={len(mismatches)}",
        corr_ok,
    )

    # ---- 3. speed
    sample_rels = rng.sample(sorted(src_set & dst_set), min(args.speed_samples, len(src_set)))
    samples = []
    for rel in sample_rels:
        src, dst = pair(rel)
        with np.load(dst) as z:
            T = int(z["meta"][0])
        samples.append((src, dst, rng.randrange(T)))
    t0 = time.perf_counter()
    for src, _, t in samples:
        load_legacy(src, t)
    legacy_ms = (time.perf_counter() - t0) * 1000 / len(samples)
    t0 = time.perf_counter()
    for _, dst, t in samples:
        load_pf(dst, t)
    pf_ms = (time.perf_counter() - t0) * 1000 / len(samples)
    speedup = legacy_ms / pf_ms
    speed_ok = speedup >= 5.0
    results["speed"] = (
        f"legacy {legacy_ms:.1f} ms/sample vs per-frame {pf_ms:.1f} ms/sample "
        f"({speedup:.1f}x, n={len(samples)})",
        speed_ok,
    )

    print("\n| check        | result | pass |")
    print("|--------------|--------|------|")
    for k in ("correctness", "completeness", "speed"):
        msg, ok = results[k]
        print(f"| {k:12s} | {msg} | {'PASS' if ok else 'FAIL'} |")
    if mismatches:
        print("mismatched frames:", mismatches[:10])
    if not all(ok for _, ok in results.values()):
        sys.exit(1)
    print("ALL_CHECKS_PASSED")


if __name__ == "__main__":
    main()

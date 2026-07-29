"""Manifold-geometry analysis of raw actions, per embodiment and DoF group.

Reads only the dataset's raw `action` parquet column -- no model inference, no
train/val split (the whole episode pool, matching the M5 convention), and by
default ALL episodes rather than the 400-episode subsample the original M5
scripts used.

For every (embodiment, group, granularity) it reports:

  spectral   PR_corr / PR_minmax, n_pc90/95/99          (linear, upper bound on d)
  nn         TwoNN + Levina-Bickel MLE, bootstrapped     (nonlinear)
  occupancy  eps-occupancy CDF, codimension slope        (volume collapse)

Granularity:
  single  sample = one timestep,          D = |group|
  chunk   sample = T consecutive steps,   D = |group| * T   (non-overlapping)

Usage
    python mg_run.py --embodiment gr1_tabletop
    python mg_run.py --embodiment all --granularity single chunk
    python mg_run.py --embodiment bridge --max-episodes 4000     # cap if huge

Writes results/<embodiment>.json (one file per embodiment, both granularities).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mg_embodiments import EMBODIMENTS, EMBODIMENT_ORDER, episode_files  # noqa: E402
from mg_metrics import (  # noqa: E402
    correlation_dimension, eps_occupancy, nn_id_bootstrap, nondegenerate_mask,
    spectral_metrics, zscore,
)

RESULT_DIR = os.path.join(_HERE, "results")


# ------------------------------------------------------------------ loading
def load_episodes(paths, max_episodes=None, seed=0, verbose=True):
    """Return (list of per-episode [L, raw_dim] arrays, stats dict).

    max_episodes caps the count PER dataset root (per task), mirroring how the
    dual-arm M5 script interprets the flag.
    """
    rng = np.random.default_rng(seed)
    eps, per_root, n_files_total = [], {}, 0
    for root, files in episode_files(paths):
        n_files_total += len(files)
        if max_episodes is not None and max_episodes < len(files):
            idx = np.sort(rng.choice(len(files), size=max_episodes, replace=False))
            files = [files[i] for i in idx]
        n_steps = 0
        for fp in files:
            a = np.stack(pd.read_parquet(fp, columns=["action"])["action"].values)
            a = np.asarray(a, dtype=np.float32)
            if a.ndim != 2:
                a = a.reshape(a.shape[0], -1)
            eps.append(a)
            n_steps += a.shape[0]
        per_root[os.path.basename(root)] = {"episodes": len(files), "timesteps": n_steps}
        if verbose:
            print(f"    {os.path.basename(root):<28} {len(files):>6} eps  "
                  f"{n_steps:>9,} steps", flush=True)
    stats = {"n_episodes": sum(v["episodes"] for v in per_root.values()),
             "n_episodes_available": n_files_total,
             "n_timesteps": sum(v["timesteps"] for v in per_root.values()),
             "per_task": per_root}
    return eps, stats


def build_single(eps, cols):
    return np.concatenate([e[:, cols] for e in eps], axis=0).astype(np.float64)


def build_chunks(eps, cols, T):
    """Non-overlapping T-step windows, flattened to D = |cols| * T."""
    out = []
    for e in eps:
        n = e.shape[0] // T
        if n == 0:
            continue
        out.append(e[: n * T, cols].reshape(n, -1))
    if not out:
        return np.zeros((0, len(cols) * T), dtype=np.float64)
    return np.concatenate(out, axis=0).astype(np.float64)


def subsample_rows(X, cap, seed=0):
    if cap is None or X.shape[0] <= cap:
        return X, X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(X.shape[0], size=cap, replace=False))
    return np.ascontiguousarray(X[idx]), X.shape[0]


# ----------------------------------------------------------------- per group
def analyze_group(X, args, seed=0, nn_stride=1):
    """X: [n_samples, D] raw (unscaled) feature matrix for one group.

    `nn_stride` temporally decimates the rows fed to the nearest-neighbour and
    occupancy estimators. Rows arrive in acquisition order, so without this the
    nearest neighbour of a point is almost always its own next timestep and
    TwoNN/corr-dim measure the local dimension of the *trajectory curve* (~1)
    rather than of the action manifold. PCA is unaffected and keeps every row.
    """
    t0 = time.time()
    res = spectral_metrics(X)

    # NN / occupancy estimators run on the non-degenerate subspace: constant
    # columns create exact duplicates that make r1 = 0 and break TwoNN.
    mask = nondegenerate_mask(X)
    res["n_degenerate_dims"] = int((~mask).sum())
    Xe = X[:, mask]
    res["D_effective_ambient"] = int(Xe.shape[1])

    if nn_stride > 1:
        Xe = np.ascontiguousarray(Xe[::nn_stride])
    res["nn_stride"] = int(nn_stride)
    res["n_samples_after_stride"] = int(Xe.shape[0])

    if Xe.shape[1] >= 2 and Xe.shape[0] >= 200:
        Xn, n_full = subsample_rows(Xe, args.nn_pool_cap, seed=seed)
        res["nn"] = nn_id_bootstrap(
            Xn, n_sub=args.nn_subsample, n_boot=args.nn_boot,
            seed=seed, with_mle=not args.no_mle, n_jobs=args.n_jobs)
        res["nn"]["pool_size"] = int(Xn.shape[0])
        res["nn"]["pool_size_full"] = int(n_full)

        res["corrdim"] = correlation_dimension(
            zscore(Xn), n_sub=args.corrdim_sub, seed=seed)

        res["occupancy"] = {}
        for amb in args.ambient:
            res["occupancy"][amb] = eps_occupancy(
                Xn, n_ambient=args.n_ambient, n_ref=args.occ_ref,
                ambient=amb, seed=seed, n_jobs=args.n_jobs)
    else:
        res["nn"] = {"note": "skipped (D<2 or too few samples)"}
        res["corrdim"] = {"note": "skipped (D<2 or too few samples)"}
        res["occupancy"] = {"note": "skipped (D<2 or too few samples)"}

    res["seconds"] = round(time.time() - t0, 1)
    return res


def run_embodiment(key, args):
    spec = EMBODIMENTS[key]
    print(f"\n{'=' * 78}\n{key}  ({spec['label']})\n{'=' * 78}", flush=True)
    print("  loading episodes ...", flush=True)
    eps, stats = load_episodes(spec["paths"], max_episodes=args.max_episodes,
                               seed=args.seed)
    print(f"  total: {stats['n_episodes']:,} episodes "
          f"({stats['n_episodes_available']:,} available), "
          f"{stats['n_timesteps']:,} timesteps", flush=True)

    out = {"embodiment": key, "label": spec["label"], "paths": spec["paths"],
           "raw_dim": spec["raw_dim"], "split": "all (no train/val split)",
           "max_episodes_per_task": args.max_episodes, "seed": args.seed,
           "chunk_T": args.chunk_T, "dataset_stats": stats,
           "config": {"nn_subsample": args.nn_subsample, "nn_boot": args.nn_boot,
                      "nn_pool_cap": args.nn_pool_cap, "n_ambient": args.n_ambient,
                      "occ_ref": args.occ_ref, "ambient": args.ambient,
                      "corrdim_sub": args.corrdim_sub, "nn_stride": args.nn_stride},
           "granularity": {}}

    for gran in args.granularity:
        out["granularity"][gran] = {}
        for gname, cols in spec["groups"].items():
            X = (build_single(eps, cols) if gran == "single"
                 else build_chunks(eps, cols, args.chunk_T))
            if X.shape[0] < 200:
                print(f"  [{gran}] {gname}: too few samples ({X.shape[0]}), skip",
                      flush=True)
                continue
            # PCA can use everything; cap only for memory sanity on huge pools
            Xp, n_full = subsample_rows(X, args.pca_pool_cap, seed=args.seed)
            # a chunk already spans chunk_T steps, so it needs proportionally
            # less extra decimation to break temporal adjacency
            stride = (args.nn_stride if gran == "single"
                      else max(1, int(round(args.nn_stride / args.chunk_T))))
            r = analyze_group(Xp, args, seed=args.seed, nn_stride=stride)
            r["n_samples_full"] = int(n_full)
            out["granularity"][gran][gname] = r
            nn = r.get("nn", {})
            cd = r.get("corrdim", {})
            occ = r.get("occupancy", {}).get(args.ambient[0], {})
            o2 = occ.get("occ_at_2x_rmed", float("nan"))
            print(f"  [{gran:<6}] {gname:<13} D={r['nominal_dim']:>4}  "
                  f"PR={r['PR_corr']:>7.2f}  pc95={r['n_pc95_corr']:>4}  "
                  f"TwoNN={nn.get('twonn_mean', float('nan')):>6.2f}"
                  f"±{nn.get('twonn_std', float('nan')):<5.2f}  "
                  f"Cdim={cd.get('id_mid', float('nan')):>6.2f}  "
                  f"occ2x={o2:.1e}  "
                  f"tube={occ.get('tube_slope', float('nan')):>6.2f}  "
                  f"tail={occ.get('tail_slope', float('nan')):>6.2f}  "
                  f"({r['seconds']}s)", flush=True)

    os.makedirs(RESULT_DIR, exist_ok=True)
    fp = os.path.join(RESULT_DIR, f"{key}.json")
    with open(fp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> wrote {fp}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embodiment", nargs="+", default=["all"],
                    help="one or more of " + ", ".join(EMBODIMENT_ORDER) + ", or 'all'")
    ap.add_argument("--granularity", nargs="+", default=["single", "chunk"],
                    choices=["single", "chunk"])
    ap.add_argument("--chunk-T", type=int, default=16)
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="cap episodes PER task (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    # estimator budgets
    ap.add_argument("--nn-subsample", type=int, default=10000,
                    help="points per TwoNN/MLE bootstrap replicate")
    ap.add_argument("--nn-boot", type=int, default=5)
    ap.add_argument("--nn-pool-cap", type=int, default=200000,
                    help="cap on the pool the bootstrap draws from")
    ap.add_argument("--pca-pool-cap", type=int, default=1000000)
    ap.add_argument("--nn-stride", type=int, default=25,
                    help="temporal decimation (in timesteps) before the NN and "
                         "occupancy estimators; breaks trajectory adjacency")
    ap.add_argument("--corrdim-sub", type=int, default=5000,
                    help="points for the Grassberger-Procaccia pair-distance fit")
    ap.add_argument("--n-ambient", type=int, default=100000)
    ap.add_argument("--occ-ref", type=int, default=20000,
                    help="data points indexed for the ambient NN query")
    ap.add_argument("--ambient", nargs="+", default=["gauss", "uniform"],
                    choices=["gauss", "uniform"])
    ap.add_argument("--no-mle", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    keys = EMBODIMENT_ORDER if "all" in args.embodiment else args.embodiment
    for k in keys:
        if k not in EMBODIMENTS:
            raise SystemExit(f"unknown embodiment {k!r}")

    t0 = time.time()
    for k in keys:
        run_embodiment(k, args)
    print(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min")
    print("#### MG RUN DONE ####")


if __name__ == "__main__":
    main()

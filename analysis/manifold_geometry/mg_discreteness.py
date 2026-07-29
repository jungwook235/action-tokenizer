"""Discreteness diagnostic: is a DoF group a continuous manifold at all?

Motivation. The GR-1 "12-DoF dexterous hand" action is in fact a binary
open/close command: every hand column takes 2-3 distinct values and the whole
12-vector visits only a handful of distinct states across the entire dataset.
A participation ratio of 2.30/12 there is NOT evidence of postural synergy on a
low-dimensional continuous manifold -- there is no manifold, just a few points.
Nearest-neighbour estimators correctly return NaN (all NN distances are zero),
and any intrinsic-dimension or volume statement about such a group is void.

This script quantifies that per (embodiment, group) so the main tables can flag
it instead of silently reporting a meaningless PR:

    n_unique_rows          distinct action vectors observed
    unique_row_frac        n_unique_rows / n_rows
    per_dim_unique         distinct values per column
    median_dim_unique      summary of the above
    dup_pair_frac          fraction of sampled point pairs at distance exactly 0
    verdict                'continuous' | 'quantized' | 'discrete'

Cheap: subsamples episodes, reads only the `action` column, no model, no NN.

    python mg_discreteness.py                       # all embodiments
    python mg_discreteness.py --embodiment gr1_tabletop --max-episodes 300
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mg_embodiments import EMBODIMENTS, EMBODIMENT_ORDER  # noqa: E402
from mg_run import build_chunks, build_single, load_episodes  # noqa: E402

RESULT_DIR = os.path.join(_HERE, "results")

# A group is called 'discrete' when the whole vector visits very few states, and
# 'quantized' when individual columns are coarsely valued but the vector is not.
DISCRETE_MAX_STATES = 64
QUANTIZED_MAX_DIM_VALUES = 32


def group_stats(X, rng, n_pair_sub=3000):
    n, D = X.shape
    uniq = np.unique(X, axis=0)
    per_dim = [int(len(np.unique(X[:, j]))) for j in range(D)]

    m = min(n_pair_sub, n)
    idx = rng.choice(n, size=m, replace=False) if m < n else np.arange(n)
    Y = X[idx]
    sq = (Y * Y).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Y @ Y.T)
    iu = np.triu_indices(m, k=1)
    dd = np.sqrt(np.clip(d2[iu], 0.0, None))
    dup_pair_frac = float((dd <= 0).mean()) if dd.size else float("nan")

    n_states = int(uniq.shape[0])
    med_dim_u = float(np.median(per_dim))
    if n_states <= DISCRETE_MAX_STATES:
        verdict = "discrete"
    elif med_dim_u <= QUANTIZED_MAX_DIM_VALUES:
        verdict = "quantized"
    else:
        verdict = "continuous"
    return {
        "n_rows": int(n), "D": int(D),
        "n_unique_rows": n_states,
        "unique_row_frac": float(n_states / n) if n else float("nan"),
        "per_dim_unique": per_dim,
        "median_dim_unique": med_dim_u,
        "min_dim_unique": int(min(per_dim)) if per_dim else 0,
        "n_constant_dims": int(sum(1 for v in per_dim if v <= 1)),
        "dup_pair_frac": dup_pair_frac,
        "verdict": verdict,
        "id_estimates_valid": verdict == "continuous",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embodiment", nargs="+", default=["all"])
    ap.add_argument("--granularity", nargs="+", default=["single", "chunk"])
    ap.add_argument("--chunk-T", type=int, default=16)
    ap.add_argument("--max-episodes", type=int, default=200,
                    help="episodes per task; discreteness saturates fast, so a "
                         "subsample is enough and keeps this script cheap")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    keys = EMBODIMENT_ORDER if "all" in args.embodiment else args.embodiment
    rng = np.random.default_rng(args.seed)
    out = {}
    for key in keys:
        spec = EMBODIMENTS[key]
        print(f"\n=== {key} ({spec['label']}) ===", flush=True)
        eps, stats = load_episodes(spec["paths"], max_episodes=args.max_episodes,
                                   seed=args.seed, verbose=False)
        out[key] = {"label": spec["label"], "dataset_stats": stats,
                    "max_episodes_per_task": args.max_episodes, "granularity": {}}
        for gran in args.granularity:
            out[key]["granularity"][gran] = {}
            for gname, cols in spec["groups"].items():
                X = (build_single(eps, cols) if gran == "single"
                     else build_chunks(eps, cols, args.chunk_T))
                if X.shape[0] < 50:
                    continue
                s = group_stats(X, rng)
                out[key]["granularity"][gran][gname] = s
                print(f"  [{gran:<6}] {gname:<13} D={s['D']:>4} "
                      f"states={s['n_unique_rows']:>8,} "
                      f"({s['unique_row_frac']:.4f})  "
                      f"median dim-values={s['median_dim_unique']:>9,.0f}  "
                      f"const dims={s['n_constant_dims']:>3}  "
                      f"dup pairs={s['dup_pair_frac']:.3f}  -> {s['verdict']}",
                      flush=True)

    os.makedirs(RESULT_DIR, exist_ok=True)
    fp = os.path.join(RESULT_DIR, "discreteness.json")
    with open(fp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {fp}")
    print("#### MG DISCRETENESS DONE ####")


if __name__ == "__main__":
    main()

"""S2 effective degrees-of-freedom & redundancy of raw actions vs v3/v4 latents.

For each representation compute, from its covariance eigenspectrum:
  nominal_dim  = feature dimensionality
  PR           = participation ratio (sum l)^2 / sum(l^2)   = effective dim
  n_pc90/95/99 = # PCs to reach 90/95/99% variance
on z-scored features (primary) and raw features.

Representations:
  raw_perstep : pool steps (N*T, D)  -> instantaneous joint effective DoF (D=29/44)
  raw_chunk   : (N, T*D)             -> chunk effective DoF (comparable to flattened latents)
  v3          : (N, 16*16=256)
  v4          : (N, 16*64=1024) mu

Intent-variance decomposition vs PCA rank (on z-scored raw_chunk):
  For increasing PCA rank r, project raw onto top-r PCs and report the fraction of
  variance that is between-task (task-predictive): between/total in that subspace,
  cumulative. Shows how quickly task-discriminative variance accumulates with rank.

Runs on gr1 and dexjoco. CPU-only from cache.

Usage:
  python effdim_dof.py --cache output/visual_sep_gr1/cache.npz --tag gr1 --out <dir>/results_gr1.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np


def eig_desc(X):
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    w = np.linalg.eigvalsh(cov)
    return np.clip(w[::-1], 0.0, None)


def pr(eigs):
    s1, s2 = float(eigs.sum()), float((eigs ** 2).sum())
    return (s1 * s1) / s2 if s2 > 0 else 0.0


def n_pc(eigs, frac):
    tot = eigs.sum()
    return int(np.searchsorted(np.cumsum(eigs) / tot, frac) + 1) if tot > 0 else 0


def zscore(X):
    sd = X.std(0, keepdims=True); sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - X.mean(0, keepdims=True)) / sd


def dof_block(X):
    D = X.shape[1]
    out = {"nominal_dim": int(D), "n_samples": int(X.shape[0])}
    for tag, Xt in (("corr", zscore(X)), ("raw", X.astype(np.float64))):
        e = eig_desc(Xt)
        out[f"PR_{tag}"] = float(pr(e))
        out[f"redundancy_{tag}"] = float(D / pr(e)) if pr(e) > 0 else float("inf")
        out[f"n_pc90_{tag}"] = n_pc(e, 0.90)
        out[f"n_pc95_{tag}"] = n_pc(e, 0.95)
        out[f"n_pc99_{tag}"] = n_pc(e, 0.99)
    return out


def between_total_ratio(X, y):
    """trace(between-task cov) / trace(total cov) on the given feature block."""
    gmean = X.mean(0)
    total = float(((X - gmean) ** 2).sum())
    between = 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        d = Xc.mean(0) - gmean
        between += Xc.shape[0] * float((d * d).sum())
    return between / total if total > 0 else 0.0


def intent_var_vs_rank(raw, y, ranks, seed):
    from sklearn.decomposition import PCA
    Xz = zscore(raw)
    maxr = min(max(ranks), raw.shape[1])
    scores = PCA(n_components=maxr, random_state=seed).fit_transform(Xz)
    overall = between_total_ratio(Xz, y)
    curve = []
    for r in ranks:
        rr = min(r, raw.shape[1])
        eta = between_total_ratio(scores[:, :rr], y)
        curve.append({"rank": int(rr), "between_total": float(eta)})
    return {"overall_between_total": float(overall), "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ranks", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]

    d = np.load(args.cache, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    A, Z3, Z4, y = d["A"], d["Z3"], d["Z4"], d["task"].astype(int)
    N, T, D = A.shape

    reps = {
        "raw_perstep": A.reshape(N * T, D).astype(np.float64),
        "raw_chunk": A.reshape(N, T * D).astype(np.float64),
        "v3": Z3.reshape(N, -1).astype(np.float64),
        "v4": Z4.reshape(N, -1).astype(np.float64),
    }
    res = {"tag": args.tag, "cache": args.cache, "N": int(N), "T": int(T), "Dact": int(D),
           "n_tasks": int(len(np.unique(y))), "seed": args.seed,
           "v3_ckpt": meta.get("v3_ckpt"), "v4_ckpt": meta.get("v4_ckpt"), "reps": {}}
    for name, X in reps.items():
        t0 = time.time()
        res["reps"][name] = dof_block(X)
        res["reps"][name]["seconds"] = round(time.time() - t0, 1)
        r = res["reps"][name]
        print(f"[{args.tag}] {name}: nominal={r['nominal_dim']} PR_corr={r['PR_corr']:.1f} "
              f"n_pc95={r['n_pc95_corr']} redund={r['redundancy_corr']:.1f}", flush=True)

    res["intent_var_vs_rank_raw_chunk"] = intent_var_vs_rank(
        A.reshape(N, T * D).astype(np.float64), y, ranks, args.seed)
    print(f"[{args.tag}] intent between/total (raw_chunk) overall="
          f"{res['intent_var_vs_rank_raw_chunk']['overall_between_total']:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("WROTE", args.out)
    print("#### EFFDIM DOF DONE ####")


if __name__ == "__main__":
    main()

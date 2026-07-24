"""S4 semantic geometry: do {raw, PCA-k, v3, v4} action representations place
same-TASK chunks near each other?

Metrics per representation (features z-scored):
  (a) NN retrieval precision@k by task label. For each query chunk, retrieve its
      k nearest neighbors (L2 on z-scored feats); P@k = fraction of neighbors with
      the same task. Reported k=1,5,10.
      RIGOR: also report P@k EXCLUDING same-episode neighbors ("cross_ep") — nearest
      neighbors are often the query's own temporally-adjacent chunks (same episode,
      same task), which inflates P@k via adjacency rather than semantics. The
      cross-episode variant is the leak-free semantic-retrieval number.
  (b) KMeans (n_clusters = #tasks, n_init=10) -> NMI, ARI vs task labels.

Runs on gr1 (24 shared-primitive tasks) and dexjoco (5 disjoint-action tasks).
Requires the episode sidecar (recover_episode_ids.py) for the cross-episode variant.

Usage:
  python semantic_geometry.py --cache output/visual_sep_gr1/cache.npz \
     --episode-ids output/visual_sep_gr1/episode_ids.npz --tag gr1 --out <dir>/results_gr1.json
"""
import argparse, json, time
from pathlib import Path
import numpy as np


def zscore(X):
    sd = X.std(0, keepdims=True); sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - X.mean(0, keepdims=True)) / sd


def build_reps(A, Z3, Z4, pca_dim, seed):
    """Return reps each standardized ONCE for a fair L2/cosine geometry.
    raw/v3/v4 -> per-dim z-score. pca -> top-k PCA of z-scored raw, used AS-IS
    (a rotation+truncation of raw_z, preserving its L2 structure). Re-standardizing
    the PCA scores would WHITEN them (equalize noise PCs) and artificially wreck NN
    retrieval, so we deliberately do not z-score pca a second time."""
    from sklearn.decomposition import PCA
    N = A.shape[0]
    raw_z = zscore(A.reshape(N, -1).astype(np.float64))
    v3_z = zscore(Z3.reshape(N, -1).astype(np.float64))
    v4_z = zscore(Z4.reshape(N, -1).astype(np.float64))
    k = min(pca_dim, raw_z.shape[1])
    pca = PCA(n_components=k, random_state=seed).fit_transform(raw_z)
    return {"raw": raw_z, "pca": pca, "v3": v3_z, "v4": v4_z}


def nn_precision(Xz, y, ep, ks=(1, 5, 10), kmax_pool=60):
    """P@k over all neighbors, and P@k excluding same-episode neighbors."""
    from sklearn.neighbors import NearestNeighbors
    N = len(y)
    kq = min(kmax_pool, N - 1)
    nn = NearestNeighbors(n_neighbors=kq + 1, n_jobs=6).fit(Xz)
    _, idx = nn.kneighbors(Xz)
    idx = idx[:, 1:]                    # drop self  [N, kq]
    neigh_lab = y[idx]
    neigh_ep = ep[idx]
    same = (neigh_lab == y[:, None])    # [N, kq]
    cross_mask = neigh_ep != ep[:, None]  # exclude same-episode neighbors
    out = {}
    for k in ks:
        out[f"P@{k}"] = float(same[:, :k].mean())
    # cross-episode: for each query take first-k neighbors that are cross-episode
    for k in ks:
        vals = []
        for i in range(N):
            csel = np.where(cross_mask[i])[0]
            if len(csel) == 0:
                continue
            take = csel[:k]
            vals.append(same[i, take].mean())
        out[f"P@{k}_cross_ep"] = float(np.mean(vals)) if vals else None
    return out


def clustering(Xz, y, seed):
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    nc = len(np.unique(y))
    km = KMeans(n_clusters=nc, n_init=10, random_state=seed).fit(Xz)
    return {"n_clusters": int(nc),
            "NMI": float(normalized_mutual_info_score(y, km.labels_)),
            "ARI": float(adjusted_rand_score(y, km.labels_))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--episode-ids", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca-dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    A, Z3, Z4, y = d["A"], d["Z3"], d["Z4"], d["task"].astype(int)
    ep = np.load(args.episode_ids, allow_pickle=True)["episode_id"].astype(np.int64)
    reps = build_reps(A, Z3, Z4, args.pca_dim, args.seed)

    res = {"tag": args.tag, "cache": args.cache, "N": int(A.shape[0]),
           "n_tasks": int(len(np.unique(y))), "pca_dim": args.pca_dim, "seed": args.seed,
           "v3_ckpt": meta.get("v3_ckpt"), "v4_ckpt": meta.get("v4_ckpt"), "reps": {}}
    for name, X in reps.items():
        t0 = time.time()
        Xz = X  # already standardized once in build_reps (see docstring)
        entry = {"dim": int(X.shape[1])}
        entry["nn"] = nn_precision(Xz, y, ep)
        entry["cluster"] = clustering(Xz, y, args.seed)
        entry["seconds"] = round(time.time() - t0, 1)
        res["reps"][name] = entry
        print(f"[{args.tag}] {name}: P@1={entry['nn']['P@1']:.3f} "
              f"P@10={entry['nn']['P@10']:.3f} P@10x={entry['nn']['P@10_cross_ep']} "
              f"ARI={entry['cluster']['ARI']:.3f} NMI={entry['cluster']['NMI']:.3f} "
              f"({entry['seconds']}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print("WROTE", args.out)
    print("#### SEMANTIC GEOMETRY DONE ####")


if __name__ == "__main__":
    main()

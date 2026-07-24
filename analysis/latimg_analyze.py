"""Latent distance vs image (DINO) distance: do close-latent pairs have close images?

v3 (action-only) vs v4 (DINO-fused).  Each point = one action chunk with TWO frames
(x0=chunk start, x1=chunk end).

  image distance  D_img(i,j) = 1/2 [ cos_dist(g0_i, g0_j) + cos_dist(g1_i, g1_j) ]
                               on patch-mean DINO features (identical for v3 & v4).
  latent distance D_v(i,j)   = euclidean on z-scored, chunk-flattened latent.

The image metric is the SAME regardless of which latent we use; the only thing that
changes between v3 and v4 is WHICH pairs each calls "latent-close". So the fair
question is: among the pairs a tokenizer thinks are closest, are the scenes actually
closer?  If v4 folds visual context into the latent, its close-latent pairs should
have smaller image distance (and higher rank-correlation) than v3's.

Reports same-episode-included ("all") and cross-episode-only ("cross") side by side,
per the request. Reads cache_stride20.npz. CPU only.

Outputs (analysis/output/visual_sep_gr1/):
    latimg_curve.png    percentile(latent dist) -> mean image dist, v3/v4/action
    latimg_knn.png      mean image dist among k latent-nearest neighbours, v3 vs v4
    latimg_scatter.png  latent-dist rank vs image dist (v3, v4), with Spearman rho
    latimg.txt / .json  numbers
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

OUTDIR = Path(__file__).resolve().parent / "output" / "visual_sep_gr1"
CACHE = OUTDIR / "cache_stride20.npz"
KNN = 10
NBIN = 25
MAX_SPEARMAN = 400_000
SEED = 0

COL = {"v3": "#c0392b", "v4": "#2f6db5", "action": "#7f8c9b"}
LAB = {"v3": "v3 latent (action-only)", "v4": "v4 latent (DINO-fused)", "action": "raw action"}


def zscore_flat(Z):
    return StandardScaler().fit_transform(Z.reshape(Z.shape[0], -1))


def l2n(X):
    X = X.astype(np.float64)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def curve(dl, dimg, nbin=NBIN):
    """mean image dist as a function of latent-distance percentile."""
    order = np.argsort(dl, kind="stable")
    di = dimg[order]
    n = len(dl)
    edges = np.linspace(0, n, nbin + 1).astype(int)
    x = np.array([100.0 * (edges[b] + edges[b + 1]) / 2 / n for b in range(nbin)])
    y = np.array([di[edges[b]:edges[b + 1]].mean() for b in range(nbin)])
    return x, y


def main():
    d = np.load(CACHE, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    A, Z3, Z4, Vf0, Vf1 = d["A"], d["Z3"], d["Z4"], d["Vf0"], d["Vf1"]
    task, traj = d["task"].astype(np.int64), d["traj_id"].astype(np.int64)
    N = A.shape[0]
    epid = task * 1_000_000 + traj

    X3, X4, Xa = zscore_flat(Z3), zscore_flat(Z4), zscore_flat(A)
    g0, g1 = l2n(Vf0), l2n(Vf1)

    # ---- all-pairs condensed vectors (pdist order == np.triu_indices(N,1) order) ----
    iu, ju = np.triu_indices(N, 1)
    same_ep = epid[iu] == epid[ju]
    d3 = pdist(X3, "euclidean")
    d4 = pdist(X4, "euclidean")
    da = pdist(Xa, "euclidean")
    dimg = 0.5 * (pdist(g0, "cosine") + pdist(g1, "cosine"))
    n_pairs = len(dimg)
    n_cross = int((~same_ep).sum())
    baseline = {"all": float(dimg.mean()), "cross": float(dimg[~same_ep].mean())}
    print(f"[latimg] N={N}  pairs={n_pairs}  cross-episode={n_cross} "
          f"({100*n_cross/max(n_pairs,1):.1f}%)  img-dist baseline all={baseline['all']:.4f}")

    dl = {"v3": d3, "v4": d4, "action": da}
    subsets = {"all": np.ones(n_pairs, bool), "cross": ~same_ep}

    # ---- curves + Spearman ----
    rng = np.random.default_rng(SEED)
    curves, spear = {}, {}
    for name, mask in subsets.items():
        curves[name] = {k: curve(dl[k][mask], dimg[mask]) for k in dl}
        idx = np.where(mask)[0]
        si = rng.choice(idx, min(MAX_SPEARMAN, len(idx)), replace=False)
        spear[name] = {k: float(spearmanr(dl[k][si], dimg[si]).correlation) for k in dl}

    # ---- kNN headline: mean image dist among k latent-nearest neighbours ----
    def knn_img(X, cross_only):
        nn = NearestNeighbors(n_neighbors=min(KNN + 1, N)).fit(X)
        _, nbr = nn.kneighbors(X)
        vals = []
        for i in range(N):
            for j in nbr[i, 1:]:
                if cross_only and epid[i] == epid[j]:
                    continue
                vals.append(0.5 * ((1.0 - g0[i] @ g0[j]) + (1.0 - g1[i] @ g1[j])))
        return float(np.mean(vals)) if vals else float("nan")

    knn = {"all": {k: knn_img({"v3": X3, "v4": X4}[k], False) for k in ("v3", "v4")},
           "cross": {k: knn_img({"v3": X3, "v4": X4}[k], True) for k in ("v3", "v4")}}

    # ------------------------------------------------------------------ figures
    # (1) curves: rows = [full 0-100%, zoom 0-20%], cols = [all, cross]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ci, sub in enumerate(("all", "cross")):
        for ri, (xmax, tag) in enumerate(((100, "full range"), (20, "close regime (0-20%)"))):
            ax = axes[ri, ci]
            for k in ("action", "v3", "v4"):
                x, y = curves[sub][k]
                sel = x <= xmax + 1e-6
                ax.plot(x[sel], y[sel], "-o", ms=3.5, lw=1.8, color=COL[k],
                        label=LAB[k] if (ri == 0 and ci == 0) else None)
            ax.axhline(baseline[sub], ls="--", lw=1.2, color="0.4",
                       label="random-pair mean" if (ri == 0 and ci == 0) else None)
            ax.set_xlim(0, xmax)
            ax.set_xlabel("latent-distance percentile (0 = closest pairs)", fontsize=9.5)
            ax.set_ylabel("mean image distance (DINO cosine)", fontsize=9.5)
            sub_lab = "all pairs (same-episode included)" if sub == "all" else "cross-episode pairs only"
            ax.set_title(f"{sub_lab}\n{tag}", fontsize=11, fontweight="bold")
            ax.grid(alpha=0.3)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.legend(loc="upper center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Do close-latent pairs have close images?  gr1 val, stride-20  "
                 "(lower curve at small percentile = latent proximity implies visual proximity)",
                 fontsize=12.5, fontweight="bold", y=1.035)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p1 = OUTDIR / "latimg_curve.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)

    # (2) kNN bars
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = ["all", "cross"]
    x = np.arange(len(groups)); w = 0.36
    b1 = ax.bar(x - w / 2, [knn[g]["v3"] for g in groups], w, color=COL["v3"], label="v3 (action-only)")
    b2 = ax.bar(x + w / 2, [knn[g]["v4"] for g in groups], w, color=COL["v4"], label="v4 (DINO-fused)")
    for g, xc in zip(groups, x):
        ax.plot([xc - 0.5, xc + 0.5], [baseline[g], baseline[g]], ls="--", lw=1.4, color="0.4")
    ax.text(0.5, baseline["all"], "  random-pair mean", va="bottom", ha="left",
            transform=ax.get_yaxis_transform() if False else ax.transData, fontsize=8, color="0.4")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["all pairs", "cross-episode only"], fontsize=10)
    ax.set_ylabel(f"mean image distance among {KNN} latent-nearest neighbours", fontsize=10)
    ax.set_title("Image distance of latent nearest-neighbours (lower = latent captures visual context)",
                 fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(alpha=0.3, axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p2 = OUTDIR / "latimg_knn.png"
    fig.savefig(p2, dpi=150); plt.close(fig)

    # (3) scatter/hexbin: latent-dist rank vs image dist (cross-episode), v3 & v4
    mask = subsets["cross"]
    idx = np.where(mask)[0]
    si = rng.choice(idx, min(120_000, len(idx)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, k in zip(axes, ("v3", "v4")):
        r = np.argsort(np.argsort(dl[k][si])) / len(si) * 100.0  # percentile rank
        hb = ax.hexbin(r, dimg[si], gridsize=45, cmap="viridis", mincnt=1, bins="log")
        ax.axhline(baseline["cross"], ls="--", lw=1.2, color="w")
        ax.set_xlabel("latent-distance percentile rank", fontsize=10)
        ax.set_title(f"{LAB[k]}\nSpearman rho(latent, image) = {spear['cross'][k]:.3f}",
                     fontsize=11, fontweight="bold", color=COL[k])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("image distance (DINO cosine)", fontsize=10)
    fig.colorbar(hb, ax=axes, label="log10 pair count", fraction=0.03, pad=0.02)
    fig.suptitle("Latent distance vs image distance (cross-episode pairs) — gr1 val stride-20",
                 fontsize=12.5, fontweight="bold")
    p3 = OUTDIR / "latimg_scatter.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight"); plt.close(fig)

    # ------------------------------------------------------------------ report
    res = dict(meta={k: meta[k] for k in ("stride", "split", "N", "n_episodes", "K3", "K4", "C")},
               n_pairs=int(n_pairs), n_cross=int(n_cross), baseline=baseline,
               spearman=spear, knn_img=knn,
               curves={s: {k: {"pct": curves[s][k][0].tolist(), "img": curves[s][k][1].tolist()}
                           for k in curves[s]} for s in curves})
    json.dump(res, open(OUTDIR / "latimg.json", "w"), indent=2)

    L = ["=" * 92,
         "Latent distance vs image (DINO) distance — do close-latent pairs have close images?",
         f"  gr1 val, stride={meta['stride']}, N={N} chunks, {meta['n_episodes']} episodes",
         f"  pairs={n_pairs}  cross-episode={n_cross} ({100*n_cross/max(n_pairs,1):.1f}%)",
         f"  image dist = 1/2[cos(x0,x0')+cos(x1,x1')] on patch-mean DINO (same for v3 & v4)",
         "=" * 92,
         "Spearman rho(latent distance, image distance)  — higher = latent proximity tracks visual proximity",
         f"{'subset':<20}{'raw action':>13}{'v3 latent':>13}{'v4 latent':>13}",
         "-" * 92]
    for s in ("all", "cross"):
        L.append(f"{('all pairs' if s=='all' else 'cross-episode'):<20}"
                 f"{spear[s]['action']:>13.3f}{spear[s]['v3']:>13.3f}{spear[s]['v4']:>13.3f}")
    L += ["-" * 92,
          f"Mean image distance among {KNN} latent-nearest neighbours (lower = closer scenes):",
          f"{'subset':<20}{'v3 latent':>13}{'v4 latent':>13}{'random base':>14}",
          "-" * 92]
    for s in ("all", "cross"):
        L.append(f"{('all pairs' if s=='all' else 'cross-episode'):<20}"
                 f"{knn[s]['v3']:>13.4f}{knn[s]['v4']:>13.4f}{baseline[s]:>14.4f}")
    L += ["-" * 92,
          "Interp: if v4 < v3 (kNN) and rho_v4 > rho_v3, the DINO-fused latent places",
          "visually-similar chunks nearer than the action-only latent does.",
          "=" * 92]
    txt = "\n".join(L)
    (OUTDIR / "latimg.txt").write_text(txt + "\n")
    print(txt)
    print(f"\nwrote {p1}\nwrote {p2}\nwrote {p3}")
    print("#### LATIMG ANALYZE DONE ####")


if __name__ == "__main__":
    main()

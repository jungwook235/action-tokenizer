"""Latent distance vs image (DINO) distance — parametrized (stride, metric, outdir).

Two image-distance definitions, each per point i having two frames (x0, x1):
  metric="avg" : D_img = 1/2 [ cos(g0_i,g0_j) + cos(g1_i,g1_j) ]      (scene similarity)
  metric="diff": per-point descriptor  gd_i = g1_i - g0_i  (DINO change x0->x1);
                 D_img = cos(gd_i, gd_j)                               (visual-change similarity)

Latent distance = euclidean on z-scored chunk-flattened latent (same for both metrics).
The image metric is identical for v3 & v4; only WHICH pairs each calls close differs.
Reports same-episode-included ("all") and cross-episode-only ("cross").

Reads {outdir}/cache_stride{S}.npz. CPU only. Runs BOTH metrics in one call.
Outputs {outdir}/s{S}_{metric}_{curve,knn,scatter}.png and s{S}_{metric}.{json}.
"""

import argparse
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

KNN = 10
NBIN = 25
MAX_SPEARMAN = 400_000
MAX_SCATTER = 120_000
SEED = 0

COL = {"v3": "#c0392b", "v4": "#2f6db5", "action": "#7f8c9b"}
LAB = {"v3": "v3 latent (action-only)", "v4": "v4 latent (DINO-fused)", "action": "raw action"}
METRIC_DESC = {
    "avg": "image distance = 1/2[cos(x0,x0')+cos(x1,x1')]  (scene similarity)",
    "diff": "image distance = cos(  (x1-x0)_i , (x1-x0)_j  )  (DINO change-vector similarity)",
}
METRIC_YLAB = {"avg": "mean image distance (DINO cosine)",
               "diff": "change-vector distance (DINO cosine of x1-x0)"}


def zscore_flat(Z):
    return StandardScaler().fit_transform(Z.reshape(Z.shape[0], -1))


def l2n(X):
    X = X.astype(np.float64)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def curve(dl, dimg, nbin=NBIN):
    order = np.argsort(dl, kind="stable")
    di = dimg[order]; n = len(dl)
    edges = np.linspace(0, n, nbin + 1).astype(int)
    x = np.array([100.0 * (edges[b] + edges[b + 1]) / 2 / n for b in range(nbin)])
    y = np.array([di[edges[b]:edges[b + 1]].mean() for b in range(nbin)])
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    outdir = Path(a.outdir)
    cache = outdir / f"cache_stride{a.stride}.npz"

    d = np.load(cache, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    A, Z3, Z4, Vf0, Vf1 = d["A"], d["Z3"], d["Z4"], d["Vf0"], d["Vf1"]
    task, traj = d["task"].astype(np.int64), d["traj_id"].astype(np.int64)
    N = A.shape[0]
    epid = task * 1_000_000 + traj

    X3, X4, Xa = zscore_flat(Z3), zscore_flat(Z4), zscore_flat(A)
    g0n, g1n = l2n(Vf0), l2n(Vf1)
    gd = (Vf1.astype(np.float64) - Vf0.astype(np.float64))
    gdn = l2n(gd)
    # points with (almost) no visual change x0->x1 (short episodes clamp x1==x0)
    # give a zero change-vector -> undefined cosine direction; drop them for the
    # "diff" metric only.
    pt_valid_diff = np.linalg.norm(gd, axis=1) > 1e-6
    n_degen = int((~pt_valid_diff).sum())

    iu, ju = np.triu_indices(N, 1)
    same_ep = epid[iu] == epid[ju]
    d3 = pdist(X3, "euclidean"); d4 = pdist(X4, "euclidean"); da = pdist(Xa, "euclidean")
    dl = {"v3": d3, "v4": d4, "action": da}
    n_pairs = len(d3); n_cross = int((~same_ep).sum())
    subsets = {"all": np.ones(n_pairs, bool), "cross": ~same_ep}
    pair_valid = {"avg": np.ones(n_pairs, bool),
                  "diff": pt_valid_diff[iu] & pt_valid_diff[ju]}
    pt_valid = {"avg": np.ones(N, bool), "diff": pt_valid_diff}
    rng = np.random.default_rng(SEED)
    print(f"[s{a.stride}] N={N}  pairs={n_pairs}  cross={n_cross} "
          f"({100*n_cross/max(n_pairs,1):.1f}%)  episodes={meta['n_episodes']}  "
          f"degenerate(x0==x1)={n_degen}")

    def img_pairs(metric):
        if metric == "avg":
            return 0.5 * (pdist(g0n, "cosine") + pdist(g1n, "cosine"))
        return pdist(gdn, "cosine")

    def img_ij(metric, i, j):
        if metric == "avg":
            return 0.5 * ((1.0 - g0n[i] @ g0n[j]) + (1.0 - g1n[i] @ g1n[j]))
        return 1.0 - gdn[i] @ gdn[j]

    def knn_img(X, metric, cross_only, pvalid):
        nn = NearestNeighbors(n_neighbors=min(KNN + 1, N)).fit(X)
        _, nbr = nn.kneighbors(X)
        vals = []
        for i in range(N):
            if not pvalid[i]:
                continue
            for j in nbr[i, 1:]:
                if cross_only and epid[i] == epid[j]:
                    continue
                if not pvalid[j]:
                    continue
                vals.append(img_ij(metric, i, j))
        return float(np.mean(vals)) if vals else float("nan")

    summary = {"meta": {k: meta[k] for k in ("stride", "split", "N", "n_episodes", "K3", "K4", "C")},
               "n_pairs": int(n_pairs), "n_cross": int(n_cross), "metrics": {}}

    for metric in ("avg", "diff"):
        dimg = img_pairs(metric)
        pv = pair_valid[metric]
        eff_masks = {"all": subsets["all"] & pv, "cross": subsets["cross"] & pv}
        baseline = {name: float(np.nanmean(dimg[m])) for name, m in eff_masks.items()}
        curves, spear = {}, {}
        for name, mask in eff_masks.items():
            curves[name] = {k: curve(dl[k][mask], dimg[mask]) for k in dl}
            idx = np.where(mask)[0]
            si = rng.choice(idx, min(MAX_SPEARMAN, len(idx)), replace=False)
            spear[name] = {k: float(spearmanr(dl[k][si], dimg[si]).correlation) for k in dl}
        knn = {"all": {k: knn_img({"v3": X3, "v4": X4}[k], metric, False, pt_valid[metric]) for k in ("v3", "v4")},
               "cross": {k: knn_img({"v3": X3, "v4": X4}[k], metric, True, pt_valid[metric]) for k in ("v3", "v4")}}

        # ---- curve figure (all/cross x full/close) ----
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ci, sub in enumerate(("all", "cross")):
            for ri, (xmax, tag) in enumerate(((100, "full range"), (20, "close regime (0-20%)"))):
                ax = axes[ri, ci]
                for k in ("action", "v3", "v4"):
                    x, y = curves[sub][k]; sel = x <= xmax + 1e-6
                    ax.plot(x[sel], y[sel], "-o", ms=3.5, lw=1.8, color=COL[k],
                            label=LAB[k] if (ri == 0 and ci == 0) else None)
                ax.axhline(baseline[sub], ls="--", lw=1.2, color="0.4",
                           label="random-pair mean" if (ri == 0 and ci == 0) else None)
                ax.set_xlim(0, xmax)
                ax.set_xlabel("latent-distance percentile (0 = closest pairs)", fontsize=9.5)
                ax.set_ylabel(METRIC_YLAB[metric], fontsize=9.5)
                sl = "all pairs (same-episode incl.)" if sub == "all" else "cross-episode only"
                ax.set_title(f"{sl}\n{tag}", fontsize=11, fontweight="bold")
                ax.grid(alpha=0.3)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
        fig.legend(loc="upper center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(f"Close-latent pairs vs image distance — gr1 val stride-{a.stride}  [metric={metric}]\n"
                     + METRIC_DESC[metric], fontsize=12, fontweight="bold", y=1.05)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        p1 = outdir / f"s{a.stride}_{metric}_curve.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)

        # ---- kNN bars ----
        fig, ax = plt.subplots(figsize=(7, 5))
        groups = ["all", "cross"]; x = np.arange(2); w = 0.36
        b1 = ax.bar(x - w / 2, [knn[g]["v3"] for g in groups], w, color=COL["v3"], label="v3 (action-only)")
        b2 = ax.bar(x + w / 2, [knn[g]["v4"] for g in groups], w, color=COL["v4"], label="v4 (DINO-fused)")
        for g, xc in zip(groups, x):
            ax.plot([xc - 0.5, xc + 0.5], [baseline[g], baseline[g]], ls="--", lw=1.4, color="0.4")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.3f}",
                        ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(["all pairs", "cross-episode only"], fontsize=10)
        ax.set_ylabel(f"mean image dist among {KNN} latent-NN", fontsize=10)
        ax.set_title(f"Image dist of latent nearest-neighbours  [stride-{a.stride}, {metric}]\n"
                     "(dashed = random-pair mean; lower = latent captures visual)", fontsize=10, fontweight="bold")
        ax.legend(frameon=False, fontsize=9.5); ax.grid(alpha=0.3, axis="y")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        p2 = outdir / f"s{a.stride}_{metric}_knn.png"
        fig.savefig(p2, dpi=150); plt.close(fig)

        # ---- scatter (cross-episode) ----
        idx = np.where(eff_masks["cross"])[0]
        si = rng.choice(idx, min(MAX_SCATTER, len(idx)), replace=False)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, k in zip(axes, ("v3", "v4")):
            r = np.argsort(np.argsort(dl[k][si])) / len(si) * 100.0
            hb = ax.hexbin(r, dimg[si], gridsize=45, cmap="viridis", mincnt=1, bins="log")
            ax.axhline(baseline["cross"], ls="--", lw=1.2, color="w")
            ax.set_xlabel("latent-distance percentile rank", fontsize=10)
            ax.set_title(f"{LAB[k]}\nSpearman rho = {spear['cross'][k]:.3f}",
                         fontsize=11, fontweight="bold", color=COL[k])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0].set_ylabel(METRIC_YLAB[metric], fontsize=10)
        fig.colorbar(hb, ax=axes, label="log10 pair count", fraction=0.03, pad=0.02)
        fig.suptitle(f"Latent vs image distance (cross-episode) — stride-{a.stride}, metric={metric}",
                     fontsize=12.5, fontweight="bold")
        p3 = outdir / f"s{a.stride}_{metric}_scatter.png"
        fig.savefig(p3, dpi=150, bbox_inches="tight"); plt.close(fig)

        summary["metrics"][metric] = dict(baseline=baseline, spearman=spear, knn_img=knn)
        json.dump(summary["metrics"][metric], open(outdir / f"s{a.stride}_{metric}.json", "w"), indent=2)
        print(f"  [{metric}] baseline(cross)={baseline['cross']:.4f}  "
              f"rho_cross v3={spear['cross']['v3']:.3f} v4={spear['cross']['v4']:.3f}  "
              f"knn_cross v3={knn['cross']['v3']:.4f} v4={knn['cross']['v4']:.4f}")

    # combined text table
    L = ["=" * 96, f"Latent vs image distance — gr1 val stride-{a.stride}, N={N}, {meta['n_episodes']} episodes",
         f"pairs={n_pairs} cross={n_cross} ({100*n_cross/max(n_pairs,1):.1f}%)", "=" * 96]
    for metric in ("avg", "diff"):
        s = summary["metrics"][metric]
        L += [f"[metric={metric}]  {METRIC_DESC[metric]}",
              f"  Spearman rho(latent,image):",
              f"    {'subset':<14}{'action':>10}{'v3':>10}{'v4':>10}"]
        for sub in ("all", "cross"):
            L.append(f"    {sub:<14}{s['spearman'][sub]['action']:>10.3f}"
                     f"{s['spearman'][sub]['v3']:>10.3f}{s['spearman'][sub]['v4']:>10.3f}")
        L += [f"  kNN({KNN}) mean image dist  (base all={s['baseline']['all']:.4f} cross={s['baseline']['cross']:.4f}):",
              f"    {'subset':<14}{'v3':>10}{'v4':>10}"]
        for sub in ("all", "cross"):
            L.append(f"    {sub:<14}{s['knn_img'][sub]['v3']:>10.4f}{s['knn_img'][sub]['v4']:>10.4f}")
        L.append("-" * 96)
    txt = "\n".join(L)
    (outdir / f"s{a.stride}_summary.txt").write_text(txt + "\n")
    json.dump(summary, open(outdir / f"s{a.stride}_summary.json", "w"), indent=2)
    print(txt)
    print(f"#### ANALYZE stride {a.stride} DONE ####")


if __name__ == "__main__":
    main()

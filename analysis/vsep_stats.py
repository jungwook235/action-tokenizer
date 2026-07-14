"""Quantify & visualize how the DINO-fused (v4) tokenizer separates action chunks
that are IDENTICAL in action space but DIFFERENT in visual dynamics — vs the
action-only (v3) tokenizer, which cannot.

Reads the shared cache from vsep_collect.py and produces three analyses:

  ① RESIDUAL VARIANCE DECOMPOSITION  (analysis/output/visual_sep/residual_r2.txt)
     Fit  latent ≈ f(action)  (ridge, 5-fold CV) → R²(action→latent).
     residual r = latent − f_cv(action)  is the part NOT explained by the action.
     Fit  r ≈ g(visual)  → R²(visual→residual).
       v3: residual≈0 and visual explains ~nothing  (latent is a pure action code)
       v4: residual is large AND visual-predictable  ⇒ latent encodes visual info
           beyond the action.  This is the core quantitative claim.

  ③ ACTION-DIST vs LATENT-DIST  (analysis/output/visual_sep/dist_vs_dist.png/.txt)
     For random chunk pairs, plot (relative) latent distance against action
     distance. v3 → straight line through origin (Δlatent ∝ Δaction). v4 keeps a
     non-zero latent-distance *floor* as Δaction→0 = the visual signal size.

  ④ NN-OVERLAP + LATENT t-SNE  (analysis/output/visual_sep/nn_overlap.txt,
     analysis/output/visual_sep/latent_tsne.png)
     kNN overlap of each latent with the action-NN and the visual-NN graphs
     (v3 latent tracks action; v4 latent drifts toward visual), plus a t-SNE of
     each latent colored by action-cluster and by task, and a "controlled-for-
     action" task-ARI (within each action cluster, does the latent still recover
     the task/visual grouping?).

Run from the action_tokenizer repo root; needs scikit-learn (+ umap for --umap).
CPU-only is fine (no GPU / no model load here).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_viz import tsne2d, scatter  # noqa: E402

from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402
from sklearn.metrics import r2_score, adjusted_rand_score  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402


# --------------------------------------------------------------- helpers
def _flat_std(X):
    """Flatten [N,...] to [N,-1] and z-score each column."""
    X = X.reshape(X.shape[0], -1)
    return StandardScaler().fit_transform(X)


def _pca(X, n, seed=0):
    n = min(n, X.shape[1], X.shape[0] - 1)
    return PCA(n_components=n, random_state=seed).fit_transform(X) if X.shape[1] > n else X


def cv_r2(X, Y, seed=0, alphas=(1.0, 10.0, 100.0, 1000.0, 1e4)):
    """5-fold cross-validated R² of a ridge map X→Y (variance-weighted, held-out).
    Returns (r2, residual = Y - cv_pred)."""
    Xs = _pca(StandardScaler().fit_transform(X), 256, seed)
    Ys = StandardScaler().fit_transform(Y.reshape(Y.shape[0], -1))
    model = RidgeCV(alphas=list(alphas))
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, Xs, Ys, cv=kf)
    r2 = r2_score(Ys, pred, multioutput="variance_weighted")
    return float(r2), (Ys - pred)


# --------------------------------------------------------------- ① residual R²
def analysis_residual(A, Z3, Z4, Vc, outdir, seed):
    lat = {"v3 (action-only, K={})".format(Z3.shape[-1]): Z3,
           "v4 (DINO-fused, K={})".format(Z4.shape[-1]): Z4}
    Aflat = A.reshape(A.shape[0], -1)
    Vflat = Vc.reshape(Vc.shape[0], -1)

    rows = []
    for name, Z in lat.items():
        Zf = Z.reshape(Z.shape[0], -1)
        r2_a, res_a = cv_r2(Aflat, Zf, seed)              # action → latent
        r2_v, res_v = cv_r2(Vflat, Zf, seed)              # visual → latent
        r2_av, _ = cv_r2(np.concatenate([Aflat, Vflat], 1), Zf, seed)  # both → latent
        # part of latent NOT explained by action, then explained by visual:
        r2_v_on_resA, _ = cv_r2(Vflat, res_a, seed)       # visual → residual(action)
        r2_a_on_resV, _ = cv_r2(Aflat, res_v, seed)       # action → residual(visual)
        rows.append((name, r2_a, r2_v, r2_av, r2_v_on_resA, r2_a_on_resV))

    L = ["=" * 92, "①  RESIDUAL VARIANCE DECOMPOSITION  (5-fold CV R², variance-weighted)", "=" * 92,
         "Latent is z-scored per-dim; action=[T*D], visual=mean-pooled DINO(f0)‖DINO(f1).", ""]
    hdr = f"{'tokenizer':<26}{'R2(act→z)':>11}{'R2(vis→z)':>11}{'R2(a+v→z)':>11}" \
          f"{'R2(vis→resAct)':>16}{'R2(act→resVis)':>16}"
    L.append(hdr); L.append("─" * len(hdr))
    for name, ra, rv, rav, rvra, rarv in rows:
        L.append(f"{name:<26}{ra:>11.4f}{rv:>11.4f}{rav:>11.4f}{rvra:>16.4f}{rarv:>16.4f}")
    L += ["",
          "Read (interpretation; compare the two rows, do not assume a direction):",
          "  • R2(act→z): how fully the action LINEARLY determines the latent. Near 1 ⇒ the",
          "    latent is essentially an action code. A lower value ⇒ the latent is not a pure",
          "    function of the action.",
          "  • R2(vis→resAct): of the latent variance the action canNOT linearly explain, how",
          "    much the VISUAL dynamics explains. NOTE: this is confounded when action and",
          "    visual are themselves correlated (few episodes / disjoint per-task actions) —",
          "    visual can then proxy the nonlinear action encoding. Read it together with the",
          "    frame-swap intervention (vsep_swap.py), which holds the action byte-fixed.",
          "  • R2(act→resVis): symmetric check — the visual-residual still carries action info.",
          "=" * 92]
    txt = outdir / "residual_r2.txt"
    txt.write_text("\n".join(L))
    print("\n".join(L)); print(f"[①] wrote {txt}")


# --------------------------------------------------------------- ③ dist vs dist
def analysis_dist(A, Z3, Z4, outdir, seed, n_pairs=60000, n_bins=20, scatter_n=4000):
    rng = np.random.default_rng(seed)
    Xa = _flat_std(A); X3 = _flat_std(Z3); X4 = _flat_std(Z4)
    N = Xa.shape[0]
    i = rng.integers(0, N, n_pairs); j = rng.integers(0, N, n_pairs)
    ok = i != j; i, j = i[ok], j[ok]

    da = np.linalg.norm(Xa[i] - Xa[j], axis=1)
    d3 = np.linalg.norm(X3[i] - X3[j], axis=1)
    d4 = np.linalg.norm(X4[i] - X4[j], axis=1)
    # normalize each latent distance by its own median → cross-tokenizer comparable
    d3r = d3 / np.median(d3); d4r = d4 / np.median(d4); dar = da / np.median(da)

    # bin by action-distance quantile
    qs = np.quantile(dar, np.linspace(0, 1, n_bins + 1))
    qs[-1] += 1e-9
    binid = np.clip(np.digitize(dar, qs) - 1, 0, n_bins - 1)
    xc, m3, m4, s3, s4 = [], [], [], [], []
    for b in range(n_bins):
        m = binid == b
        if m.sum() < 5:
            continue
        xc.append(dar[m].mean())
        m3.append(d3r[m].mean()); s3.append(d3r[m].std())
        m4.append(d4r[m].mean()); s4.append(d4r[m].std())
    xc, m3, m4, s3, s4 = map(np.asarray, (xc, m3, m4, s3, s4))

    # floor = mean relative latent-dist in the SMALLEST action-dist bin
    lo = dar <= np.quantile(dar, 0.05)
    floor3, floor4 = d3r[lo].mean(), d4r[lo].mean()

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    sub = rng.choice(len(da), min(scatter_n, len(da)), replace=False)
    for a, dr, tag, col in ((ax[0], d3r, f"v3 (K={Z3.shape[-1]})", "#1f77b4"),
                            (ax[1], d4r, f"v4 (K={Z4.shape[-1]})", "#d62728")):
        a.scatter(dar[sub], dr[sub], s=4, alpha=0.25, color=col, linewidths=0)
        a.set_title(f"{tag}\nΔaction vs Δlatent (relative)"); a.set_xlabel("Δaction / median")
        a.set_ylabel("Δlatent / median"); a.grid(alpha=0.2)
        a.axhline(1.0, color="k", lw=0.6, ls=":")
    ax[2].plot(xc, m3, "-o", ms=4, color="#1f77b4", label=f"v3 latent (floor={floor3:.2f})")
    ax[2].fill_between(xc, m3 - s3, m3 + s3, color="#1f77b4", alpha=0.12)
    ax[2].plot(xc, m4, "-o", ms=4, color="#d62728", label=f"v4 latent (floor={floor4:.2f})")
    ax[2].fill_between(xc, m4 - s4, m4 + s4, color="#d62728", alpha=0.12)
    ax[2].set_title("binned mean Δlatent vs Δaction\n(floor = value as Δaction→0)")
    ax[2].set_xlabel("Δaction / median"); ax[2].set_ylabel("Δlatent / median")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.2)
    fig.suptitle("③  Same action, different latent?  Latent-distance floor at Δaction→0 = visual signal",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = outdir / "dist_vs_dist.png"
    fig.savefig(png, dpi=130); plt.close(fig)

    L = ["=" * 78, "③  ACTION-DIST vs LATENT-DIST", "=" * 78,
         f"pairs={len(da)}  bins={n_bins}", "",
         "Relative latent-distance FLOOR (mean over smallest-5% action-distance pairs):",
         f"  v3 : {floor3:.4f}   (→ near 0 as Δaction→0 : latent is an action code)",
         f"  v4 : {floor4:.4f}   (→ stays high : same action can map to different latents)",
         f"  ratio v4/v3 = {floor4 / max(floor3, 1e-9):.2f}×", "",
         f"figure -> {png}", "=" * 78]
    txt = outdir / "dist_vs_dist.txt"
    txt.write_text("\n".join(L))
    print("\n".join(L)); print(f"[③] wrote {png}")


# --------------------------------------------------------------- ④ NN + t-SNE
def _knn_sets(X, k):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    idx = nn.kneighbors(X, return_distance=False)[:, 1:]  # drop self
    return idx


def _overlap(a_idx, b_idx):
    k = a_idx.shape[1]
    return np.mean([len(set(a) & set(b)) for a, b in zip(a_idx, b_idx)]) / k


def analysis_nn(A, Z3, Z4, Vc, task, task_names, outdir, seed, k=10, kk=10, do_umap=False):
    Xa = _pca(_flat_std(A), 50, seed)
    X3 = _pca(_flat_std(Z3), 50, seed)
    X4 = _pca(_flat_std(Z4), 50, seed)
    Xv = _pca(_flat_std(Vc), 50, seed)

    na, n3, n4, nv = (_knn_sets(X, kk) for X in (Xa, X3, X4, Xv))
    ov = {
        "action ↔ v3-latent": _overlap(na, n3), "action ↔ v4-latent": _overlap(na, n4),
        "visual ↔ v3-latent": _overlap(nv, n3), "visual ↔ v4-latent": _overlap(nv, n4),
        "action ↔ visual (ref)": _overlap(na, nv),
    }

    # action clusters (for coloring + controlled-for-action task ARI)
    ca = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xa)
    n_task = len(task_names)

    def task_ari_global(Xz):
        return adjusted_rand_score(task, KMeans(n_task, n_init=10, random_state=0).fit_predict(Xz))

    def task_ari_within_action(Xz):
        """Within each action cluster, does the latent recover the task grouping?
        Weighted-mean ARI over action clusters that contain ≥2 tasks & ≥20 pts."""
        vals, wts = [], []
        for c in np.unique(ca):
            m = ca == c
            if m.sum() < 20 or len(np.unique(task[m])) < 2:
                continue
            kt = len(np.unique(task[m]))
            lab = KMeans(kt, n_init=10, random_state=0).fit_predict(Xz[m])
            vals.append(adjusted_rand_score(task[m], lab)); wts.append(m.sum())
        if not vals:
            return float("nan")
        return float(np.average(vals, weights=wts))

    ari_glob = {"v3": task_ari_global(X3), "v4": task_ari_global(X4)}
    ari_ctrl = {"v3": task_ari_within_action(X3), "v4": task_ari_within_action(X4)}

    # t-SNE of each latent, colored by action-cluster and by task
    emb3, emb4 = tsne2d(X3, seed), tsne2d(X4, seed)
    fig, ax = plt.subplots(2, 2, figsize=(11, 11))
    scatter(ax[0][0], emb3, ca, f"v3 latent  (color = action-cluster k={k})")
    scatter(ax[0][1], emb3, task, "v3 latent  (color = task)")
    scatter(ax[1][0], emb4, ca, f"v4 latent  (color = action-cluster k={k})")
    scatter(ax[1][1], emb4, task, "v4 latent  (color = task)")
    fig.suptitle("④  Latent t-SNE — v3 vs v4, colored by action-cluster and by task", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = outdir / "latent_tsne.png"
    fig.savefig(png, dpi=130); plt.close(fig)

    L = ["=" * 78, "④  NEAREST-NEIGHBOR OVERLAP + TASK STRUCTURE", "=" * 78,
         f"kNN overlap (k={kk}, in 50-d PCA space) — fraction of shared neighbors:", ""]
    for kn, v in ov.items():
        L.append(f"  {kn:<26}{v:>8.3f}")
    L += ["",
          "  Hypothesis would predict: action↔v4 < action↔v3 (v4 drifts off the action graph)",
          "  and visual↔v4 > visual↔v3 (v4 moves toward the visual graph). Compare the numbers;",
          "  on near-disjoint-action data both latents can track the action graph tightly.", "",
          "Task recovery from the latent (ARI vs source task):",
          f"  {'':<20}{'v3':>10}{'v4':>10}",
          f"  {'global':<20}{ari_glob['v3']:>10.3f}{ari_glob['v4']:>10.3f}",
          f"  {'within-action-clust':<20}{ari_ctrl['v3']:>10.3f}{ari_ctrl['v4']:>10.3f}", "",
          "  'within-action-cluster' CONTROLS for the action. CAVEAT: if the tasks have",
          "  near-disjoint action spaces (few action-collisions), action alone already",
          "  separates tasks and a clean action code (v3) can score HIGHER here — this metric",
          "  only tests the hypothesis on data where the same action recurs across tasks.", "",
          "  task legend: " + ", ".join(f"{i}={n}" for i, n in enumerate(task_names)),
          f"figure -> {png}", "=" * 78]
    txt = outdir / "nn_overlap.txt"
    txt.write_text("\n".join(L))
    print("\n".join(L)); print(f"[④] wrote {png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(_REPO_ROOT / "analysis" / "output" / "visual_sep" / "cache.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=8, help="#action clusters for coloring/control")
    ap.add_argument("--only", nargs="+", default=["residual", "dist", "nn"],
                    choices=["residual", "dist", "nn"])
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    A, Z3, Z4, Vc = d["A"], d["Z3"], d["Z4"], d["Vcontext"]
    task = d["task"]; task_names = meta["task_names"]
    print(f"[stats] N={A.shape[0]}  A{A.shape} Z3{Z3.shape} Z4{Z4.shape} V{Vc.shape}  tasks={task_names}")

    outdir = Path(args.cache).parent
    outdir.mkdir(parents=True, exist_ok=True)
    if "residual" in args.only:
        analysis_residual(A, Z3, Z4, Vc, outdir, args.seed)
    if "dist" in args.only:
        analysis_dist(A, Z3, Z4, outdir, args.seed)
    if "nn" in args.only:
        analysis_nn(A, Z3, Z4, Vc, task, task_names, outdir, args.seed, k=args.k)


if __name__ == "__main__":
    main()

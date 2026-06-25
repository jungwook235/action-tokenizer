"""Clustering + 2D-embedding study of a tokenizer's action latents.

For a balanced sample of validation action chunks we collect:
    A   = normalized input action chunk      [N, T, D]
    Z   = tokenizer latent (the VLA target)  [N, n_main, K]
    Dec = decode(Z) → reconstructed action   [N, T, D]
plus a TASK label (= source dataset index).

We then CLUSTER the input action chunks two ways and color every embedding by the
resulting classes:
    • euclid : KMeans on the standardized flattened action chunk ([T*D]).
    • dtw    : tslearn TimeSeriesKMeans with DTW — treats the chunk as a short
               multivariate time series and warps over time, so it groups by
               motion SHAPE regardless of phase/speed offset.

Embeddings are computed with BOTH t-SNE and UMAP. For each reducer we draw a grid
(rows = euclid-cluster / dtw-cluster / task ; cols = action / latent / decoded),
so we can see whether the tokenizer latent (and the decoded action) preserve the
action-class structure under either clustering / either reducer.

Quantitative metrics (silhouette, ARI) are written to a .txt report.

Outputs (analysis/output/cluster/):
    <tag>_tsne_embedding.png
    <tag>_umap_embedding.png
    <tag>_clustering.txt

Run from the action_tokenizer repo root, gr00t-actlat env, on a GPU.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gr00t.experiment.data_config_v3  # noqa: F401
from gr00t.data.dataset_action_frames_v4 import (  # noqa: E402
    ActionFramesCollatorV4, ActionFramesDatasetV4,
)
from gr00t.data.merge_norm_stats import apply_merged_normalization_metadata  # noqa: E402
from gr00t.model.action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper  # noqa: E402
from analyze_latents import _encode_mu_and_sample  # noqa: E402

from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.cluster import KMeans, AgglomerativeClustering  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score, adjusted_rand_score  # noqa: E402


# ---------------------------------------------------------------- collection
@torch.no_grad()
def collect(args, device):
    wrapper = ActionLatentTokenizerWrapper.from_checkpoint(args.checkpoint, device=device)
    wrapper.eval()
    tok = wrapper.tokenizer
    encoder = tok.encoder
    is_v4 = hasattr(tok, "_is_v4")
    is_v5 = hasattr(tok, "_is_v5")
    needs_visual = is_v4 or is_v5
    if is_v5:
        raise NotImplementedError("V5 not supported here.")
    dtype = encoder.action_proj.weight.dtype

    datasets, task_names = [], []
    for p in args.dataset_path:
        datasets.append(ActionFramesDatasetV4(
            dataset_path=p, data_config_name=args.data_config,
            embodiment_tag=args.embodiment_tag, split="val",
            val_ratio=args.val_ratio, val_seed=args.val_seed,
            normalization_mode=args.normalization_mode, image_size=args.image_size,
            video_backend=args.video_backend, use_fixed_val=True,
            fixed_val_path=args.fixed_val_path))
        task_names.append(Path(p).name)
    apply_merged_normalization_metadata(datasets, datasets)

    n_ds = len(datasets)
    per_ds = max(args.min_per_dataset, -(-args.target_total // n_ds))
    rng = np.random.default_rng(args.sample_seed)
    subsets, task_labels = [], []
    for ti, d in enumerate(datasets):
        n = len(d)
        k = min(per_ds, n)
        idx = rng.choice(n, size=k, replace=False)
        subsets.append(torch.utils.data.Subset(d, idx.tolist()))
        task_labels += [ti] * k
    concat = torch.utils.data.ConcatDataset(subsets)
    task_labels = np.array(task_labels)

    loader = torch.utils.data.DataLoader(
        concat, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=ActionFramesCollatorV4())
    gen = torch.Generator(device=device).manual_seed(args.sample_seed)

    A, Z, Dec = [], [], []
    for batch in loader:
        actions = batch["action"].to(device)
        if needs_visual:
            f0, f1 = wrapper._resolve_dino_feats(batch["frame_x0"], batch["frame_x1"], None, None, device)
            mu, sigma, logvar, z = _encode_mu_and_sample(encoder, actions, f0, f1, generator=gen)
            zero = mu[:, :0]
            dec = tok.decode(zero, z.to(dtype), zero)
        else:
            g, t, h = tok.encode(actions.to(dtype))
            z = t.float()
            dec = tok.decode(g, t, h)
        A.append(actions.cpu().float()); Z.append(z.cpu().float()); Dec.append(dec.cpu().float())

    A = torch.cat(A).numpy(); Z = torch.cat(Z).numpy(); Dec = torch.cat(Dec).numpy()
    meta = dict(K=Z.shape[-1], T=A.shape[1], D=A.shape[2],
                is_vae=bool(getattr(encoder, "use_vae", False)))
    return A, Z, Dec, task_labels, task_names, meta


# ---------------------------------------------------------------- clustering
def euclid_sweep(Xa, task_labels, ks):
    rows, best = [], (None, -2.0)
    n_tasks = len(np.unique(task_labels))
    for k in ks:
        for name, fit in (
            ("kmeans", lambda k=k: KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xa)),
            ("ward", lambda k=k: AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xa)),
            ("gmm", lambda k=k: GaussianMixture(n_components=k, covariance_type="diag",
                                                random_state=0).fit_predict(Xa)),
        ):
            lab = fit()
            sil = silhouette_score(Xa, lab)
            ari = adjusted_rand_score(task_labels, lab) if n_tasks > 1 else float("nan")
            rows.append((name, k, sil, ari))
            if sil > best[1]:
                best = ((name, k), sil)
    return rows, best


def dtw_cluster(A, k, seed, dtw_max, sil_max):
    """DTW TimeSeriesKMeans on action sequences [N,T,D]. Fit on a subsample for
    tractability, then predict all. Return (labels[N], dtw_silhouette_on_subsample)."""
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    from tslearn.clustering import TimeSeriesKMeans, silhouette_score as ts_sil
    N = A.shape[0]
    As = TimeSeriesScalerMeanVariance().fit_transform(A)  # per-series, per-dim z-score
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(N, min(dtw_max, N), replace=False)
    km = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=5, n_init=1,
                          random_state=seed, n_jobs=-1)
    km.fit(As[fit_idx])
    labels = km.predict(As)
    sidx = fit_idx[:min(sil_max, len(fit_idx))]
    try:
        sil = ts_sil(As[sidx], labels[sidx], metric="dtw", n_jobs=-1)
    except Exception:
        sil = float("nan")
    return labels, float(sil)


# ---------------------------------------------------------------- embeddings
def _pca50(X, seed):
    return PCA(n_components=min(50, X.shape[1]), random_state=seed).fit_transform(X) if X.shape[1] > 50 else X


def tsne2d(X, seed):
    n = X.shape[0]
    perp = min(30, max(5, (n - 1) // 3))
    return TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed,
                max_iter=1000).fit_transform(_pca50(X, seed))


def umap2d(X, seed):
    from umap import UMAP
    return UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=seed).fit_transform(_pca50(X, seed))


def scatter(ax, emb, labels, title):
    uniq = np.unique(labels)
    big = len(uniq) > 20
    cmap = plt.get_cmap("gist_ncar" if big else "tab20")
    for i, u in enumerate(uniq):
        m = labels == u
        c = cmap(i / max(1, len(uniq) - 1)) if big else cmap(i % 20)
        ax.scatter(emb[m, 0], emb[m, 1], s=6, color=c, alpha=0.6, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def make_figure(embs, label_sets, tag, reducer, meta, N, out):
    """embs: dict name->2D for action/latent/decoded. label_sets: list of (row_title, labels)."""
    cols = [("INPUT action", embs["action"]), ("LATENT z", embs["latent"]),
            ("DECODED action", embs["decoded"])]
    nrows = len(label_sets)
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 5 * nrows), squeeze=False)
    for r, (rtitle, labels) in enumerate(label_sets):
        for c, (cname, emb) in enumerate(cols):
            scatter(axes[r][c], emb, labels, f"{cname}\ncolor = {rtitle}")
    fig.suptitle(f"{tag}  —  {reducer.upper()}  (N={N}, K={meta['K']})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=125)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--dataset-path", nargs="+", required=True)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--normalization-mode", default="min_max")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--video-backend", default="decord")
    ap.add_argument("--val-ratio", type=float, default=0.003)
    ap.add_argument("--val-seed", type=int, default=42)
    ap.add_argument("--fixed-val-path", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--target-total", type=int, default=3000)
    ap.add_argument("--min-per-dataset", type=int, default=60)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 6, 8, 10, 12, 16])
    ap.add_argument("--final-k", type=int, default=0, help="0 = best-silhouette KMeans k")
    ap.add_argument("--reducers", nargs="+", default=["tsne", "umap"])
    ap.add_argument("--no-dtw", action="store_true")
    ap.add_argument("--dtw-max", type=int, default=800, help="max series for DTW fit")
    ap.add_argument("--dtw-sil-max", type=int, default=600, help="max series for DTW silhouette")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    outdir = _REPO_ROOT / "analysis" / "output" / "cluster"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[cluster] {args.tag}: collecting val samples ...")
    A, Z, Dec, task_labels, task_names, meta = collect(args, device)
    N = A.shape[0]
    n_tasks = len(task_names)
    print(f"[cluster] N={N}  A{A.shape} Z{Z.shape} Dec{Dec.shape}  tasks={n_tasks}")

    Xa = StandardScaler().fit_transform(A.reshape(N, -1))
    Xz = StandardScaler().fit_transform(Z.reshape(N, -1))
    Xd = StandardScaler().fit_transform(Dec.reshape(N, -1))

    # ---- euclidean clustering on actions ----
    rows, best = euclid_sweep(Xa, task_labels, args.ks)
    if args.final_k > 0:
        final_k = args.final_k
    else:
        final_k = max([(k, s) for (nm, k, s, a) in rows if nm == "kmeans"], key=lambda x: x[1])[0]
    eu_labels = KMeans(n_clusters=final_k, n_init=10, random_state=0).fit_predict(Xa)
    eu = dict(
        sil_a=silhouette_score(Xa, eu_labels), sil_z=silhouette_score(Xz, eu_labels),
        sil_d=silhouette_score(Xd, eu_labels),
        ari_z=adjusted_rand_score(eu_labels, KMeans(final_k, n_init=10, random_state=0).fit_predict(Xz)),
        ari_task=(adjusted_rand_score(task_labels, eu_labels) if n_tasks > 1 else float("nan")),
    )

    # ---- DTW clustering on actions ----
    dtw_labels = None
    dt = {}
    if not args.no_dtw:
        print(f"[cluster] DTW TimeSeriesKMeans (k={final_k}) ...")
        dtw_labels, dtw_sil = dtw_cluster(A, final_k, args.sample_seed, args.dtw_max, args.dtw_sil_max)
        dt = dict(
            dtw_sil=dtw_sil,
            sil_z=silhouette_score(Xz, dtw_labels), sil_d=silhouette_score(Xd, dtw_labels),
            ari_z=adjusted_rand_score(dtw_labels, KMeans(final_k, n_init=10, random_state=0).fit_predict(Xz)),
            ari_task=(adjusted_rand_score(task_labels, dtw_labels) if n_tasks > 1 else float("nan")),
            ari_eu=adjusted_rand_score(eu_labels, dtw_labels),
        )

    # ---- embeddings (each reducer) ----
    label_sets = [("euclid-cluster", eu_labels)]
    if dtw_labels is not None:
        label_sets.append(("DTW-cluster", dtw_labels))
    if n_tasks > 1:
        label_sets.append((f"task ({n_tasks})", task_labels))

    figs = {}
    for red in args.reducers:
        print(f"[cluster] {red} embedding (action/latent/decoded) ...")
        fn = {"tsne": tsne2d, "umap": umap2d}[red]
        embs = {"action": fn(Xa, args.sample_seed), "latent": fn(Xz, args.sample_seed),
                "decoded": fn(Xd, args.sample_seed)}
        png = outdir / f"{args.tag}_{red}_embedding.png"
        make_figure(embs, label_sets, args.tag, red, meta, N, png)
        figs[red] = png

    # ---- report ----
    def fnum(v, p=4):
        return "NA" if v is None or (isinstance(v, float) and v != v) else f"{v:.{p}f}"

    L = ["=" * 86, f"CLUSTERING / EMBEDDING STUDY  —  {args.tag}", "=" * 86]
    L.append(f"checkpoint : {args.checkpoint}")
    L.append(f"N samples  : {N}  (balanced across {n_tasks} task/dataset source(s))")
    L.append(f"features   : action[{meta['T']}x{meta['D']}]  latent z[{Z.shape[1]}x{meta['K']}]  "
             f"decoded[{meta['T']}x{meta['D']}]")
    L.append(f"reducers   : {', '.join(args.reducers)}    clusterings: euclid"
             + ("" if args.no_dtw else " + dtw"))
    L.append("")
    L.append("─ Euclidean KMeans sweep on INPUT actions ─")
    L.append(f"  {'method':<10}{'k':>4}{'silhouette':>13}{'ARI vs task':>14}")
    L.append("  " + "─" * 41)
    for nm, k, sil, ari in rows:
        L.append(f"  {nm:<10}{k:>4}{sil:>13.4f}{fnum(ari):>14}")
    L.append(f"  → best silhouette: method={best[0][0]} k={best[0][1]} (sil={best[1]:.4f})")
    L.append(f"  → chosen action-cluster k (KMeans): {final_k}")
    L.append("")
    L.append("─ Cluster-preservation: silhouette under each action-clustering ─")
    L.append(f"  {'space':<16}{'EUCLID labels':>16}{'DTW labels':>16}")
    L.append("  " + "─" * 47)
    L.append(f"  {'INPUT action':<16}{eu['sil_a']:>16.4f}{'(by constr.)':>16}")
    L.append(f"  {'LATENT z':<16}{eu['sil_z']:>16.4f}{fnum(dt.get('sil_z')):>16}")
    L.append(f"  {'DECODED action':<16}{eu['sil_d']:>16.4f}{fnum(dt.get('sil_d')):>16}")
    L.append("")
    L.append("─ DTW clustering (TimeSeriesKMeans, metric=dtw) ─")
    if args.no_dtw:
        L.append("  (skipped: --no-dtw)")
    else:
        L.append(f"  DTW silhouette in DTW space (subsample) : {fnum(dt['dtw_sil'])}")
        L.append(f"  ARI( DTW-clusters , euclid-clusters )   : {fnum(dt['ari_eu'])}")
        L.append(f"  ARI( KMeans-on-latent , DTW-clusters )  : {fnum(dt['ari_z'])}")
        if n_tasks > 1:
            L.append(f"  ARI( DTW-clusters , task )              : {fnum(dt['ari_task'])}")
    L.append("")
    L.append("─ Latent organized by action? (euclid labels) ─")
    L.append(f"  ARI( KMeans-on-latent , euclid-clusters ) : {fnum(eu['ari_z'])}")
    if n_tasks > 1:
        L.append(f"  ARI( euclid-clusters , task )             : {fnum(eu['ari_task'])}")
    L.append("")
    L.append("  Read: latent silhouette ≈ input-action silhouette ⇒ latent keeps action classes")
    L.append("  separable. high ARI(latent,clusters) ⇒ clustering the latent recovers the action")
    L.append("  groups. ARI(DTW,euclid): do time-warped clusters differ from plain Euclidean ones.")
    L.append("")
    L.append("  task legend: " + ", ".join(f"{i}={n}" for i, n in enumerate(task_names))[:4000])
    L.append("")
    for red, png in figs.items():
        L.append(f"figure ({red}) -> {png}")
    L.append("=" * 86)
    rep = outdir / f"{args.tag}_clustering.txt"
    rep.write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n[cluster] wrote {rep}")


if __name__ == "__main__":
    main()

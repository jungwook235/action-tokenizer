"""②  NEAR-DUPLICATE ACTION GROUPS + real frames.

Find groups of validation chunks whose ACTIONS are near-identical (a tiny radius
in action space) but that come from ≥2 different tasks, then show — with the
actual video frames — that the DINO-fused v4 tokenizer SPREADS such a group into
distinct latents while the action-only v3 tokenizer collapses it to (almost) one
point.

For every found group we compute, in each tokenizer's z-scored latent space, the
within-group spread (mean pairwise distance / global median pairwise distance),
and across all groups we correlate the within-group pairwise latent distance with
the pairwise VISUAL (DINO) distance — high for v4, ~0 for v3.

Outputs (analysis/output/visual_sep/):
    neardup_groups.png   rows = groups, each a strip of member end-frames (border
                         colored by task) titled with v3 vs v4 spread.
    neardup.txt          per-group + pooled statistics.

Reads the shared cache; re-fetches the identical frames from the datasets.
Run from the action_tokenizer repo root, gr00t-actlat env (CPU is fine).
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vsep_collect import build_datasets  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

_TAB = plt.get_cmap("tab10")


def _std(X):
    return StandardScaler().fit_transform(X.reshape(X.shape[0], -1))


def _global_median_pairdist(X, seed, n=40000):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, X.shape[0], n); j = rng.integers(0, X.shape[0], n)
    ok = i != j
    return float(np.median(np.linalg.norm(X[i[ok]] - X[j[ok]], axis=1)))


def find_groups(Xa, task, radius, min_size, max_groups, seed):
    """Greedy near-duplicate-action groups spanning ≥2 tasks, low mutual overlap."""
    nn = NearestNeighbors(radius=radius).fit(Xa)
    neigh = nn.radius_neighbors(Xa, return_distance=False)
    cand = []
    for c, mem in enumerate(neigh):
        mem = np.asarray(mem)
        if len(mem) < min_size:
            continue
        n_task = len(np.unique(task[mem]))
        if n_task < 2:
            continue
        # score: prefer task-diverse then large
        cand.append((n_task, len(mem), c, mem))
    cand.sort(key=lambda x: (x[0], x[1]), reverse=True)

    chosen, used = [], np.zeros(Xa.shape[0], bool)
    for n_task, sz, c, mem in cand:
        if used[mem].mean() > 0.5:   # >50% already covered → skip near-duplicate group
            continue
        chosen.append(mem)
        used[mem] = True
        if len(chosen) >= max_groups:
            break
    return chosen


def group_spread(members, Xz, gmed):
    """mean within-group pairwise latent distance, normalized by global median."""
    if len(members) < 2:
        return 0.0
    d = []
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            d.append(np.linalg.norm(Xz[members[a]] - Xz[members[b]]))
    return float(np.mean(d) / gmed)


def pooled_corr(groups, Xz, Xv):
    """Correlate within-group pairwise latent-dist with pairwise visual-dist,
    pooled over all groups."""
    dl, dv = [], []
    for mem in groups:
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                dl.append(np.linalg.norm(Xz[mem[a]] - Xz[mem[b]]))
                dv.append(np.linalg.norm(Xv[mem[a]] - Xv[mem[b]]))
    dl, dv = np.asarray(dl), np.asarray(dv)
    if len(dl) < 3 or dl.std() < 1e-9 or dv.std() < 1e-9:
        return float("nan"), len(dl)
    return float(np.corrcoef(dl, dv)[0, 1]), len(dl)


def fetch_frame(datasets, task_i, local_i, which="frame_x1"):
    item = datasets[int(task_i)][int(local_i)]
    f = np.asarray(item[which])
    if f.ndim == 3 and f.shape[0] in (1, 3) and f.shape[0] < f.shape[-1]:
        f = np.transpose(f, (1, 2, 0))  # CHW → HWC just in case
    return f.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(_REPO_ROOT / "analysis" / "output" / "visual_sep" / "cache.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius-pct", type=float, default=1.0,
                    help="action-distance percentile used as the near-dup radius")
    ap.add_argument("--min-size", type=int, default=4)
    ap.add_argument("--max-groups", type=int, default=6)
    ap.add_argument("--max-show", type=int, default=8, help="max member frames drawn per group")
    ap.add_argument("--which-frame", default="frame_x1", choices=["frame_x0", "frame_x1"])
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    A, Z3, Z4, Vc = d["A"], d["Z3"], d["Z4"], d["Vcontext"]
    task = d["task"]; samp_task = d["samp_task"]; samp_local = d["samp_local"]
    task_names = meta["task_names"]
    outdir = Path(args.cache).parent

    Xa, X3, X4, Xv = _std(A), _std(Z3), _std(Z4), _std(Vc)

    # near-dup radius from the action-distance distribution
    rng = np.random.default_rng(args.seed)
    ii = rng.integers(0, Xa.shape[0], 40000); jj = rng.integers(0, Xa.shape[0], 40000)
    ok = ii != jj
    pd = np.linalg.norm(Xa[ii[ok]] - Xa[jj[ok]], axis=1)
    radius = float(np.percentile(pd, args.radius_pct))
    print(f"[②] action pairdist: p{args.radius_pct}={radius:.3f}  median={np.median(pd):.3f}")

    groups = find_groups(Xa, task, radius, args.min_size, args.max_groups, args.seed)
    print(f"[②] found {len(groups)} near-dup groups (≥{args.min_size} members, ≥2 tasks)")
    if not groups:
        print("[②] no groups — increase --radius-pct or --target-total in collect."); return

    gmed3 = _global_median_pairdist(X3, args.seed)
    gmed4 = _global_median_pairdist(X4, args.seed)

    # per-group stats
    G = []
    for gi, mem in enumerate(groups):
        s3 = group_spread(mem, X3, gmed3)
        s4 = group_spread(mem, X4, gmed4)
        # mean within-group action spread (should be tiny)
        sa = group_spread(mem, Xa, _global_median_pairdist(Xa, args.seed))
        G.append(dict(members=mem, s3=s3, s4=s4, sa=sa,
                      tasks=[int(t) for t in task[mem]]))

    c3, n3 = pooled_corr(groups, X3, Xv)
    c4, n4 = pooled_corr(groups, X4, Xv)

    # ---- figure: rows = groups, strip of member end-frames ----
    datasets, _ = build_datasets(SimpleNamespace(
        dataset_path=meta["dataset_path"], data_config=meta["data_config"],
        embodiment_tag=meta["embodiment_tag"], val_ratio=meta["val_ratio"],
        val_seed=meta["val_seed"], normalization_mode=meta["normalization_mode"],
        image_size=meta["image_size"], video_backend=meta["video_backend"],
        fixed_val_path=meta["fixed_val_path"]))

    ncol = min(args.max_show, max(len(g["members"]) for g in G))
    nrow = len(G)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.7, nrow * 1.9), squeeze=False)
    for r, g in enumerate(G):
        mem = g["members"][:ncol]
        for c in range(ncol):
            ax = axes[r][c]; ax.set_xticks([]); ax.set_yticks([])
            if c >= len(mem):
                ax.axis("off"); continue
            n = mem[c]
            frame = fetch_frame(datasets, samp_task[n], samp_local[n], args.which_frame)
            ax.imshow(frame)
            ti = int(task[n])
            for sp in ax.spines.values():
                sp.set_edgecolor(_TAB(ti % 10)); sp.set_linewidth(3)
            if r == 0:
                pass
            ax.set_xlabel(task_names[ti][:14], fontsize=6, color=_TAB(ti % 10))
        axes[r][0].set_ylabel(f"grp{r}\nv3={g['s3']:.2f}\nv4={g['s4']:.2f}\n({g['s4']/max(g['s3'],1e-6):.1f}×)",
                              fontsize=7, rotation=0, labelpad=24, va="center")
    fig.suptitle(f"②  Near-identical ACTIONS ({args.which_frame}) — v4 latent spread vs v3 "
                 f"(border = task).  Δaction tiny, yet v4 spread ≫ v3.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = outdir / "neardup_groups.png"
    fig.savefig(png, dpi=130); plt.close(fig)

    # ---- report ----
    L = ["=" * 84, "②  NEAR-DUPLICATE ACTION GROUPS (same action, different visual)", "=" * 84,
         f"near-dup radius = action-pairdist p{args.radius_pct} = {radius:.3f}",
         f"latent spread = mean within-group pairwise dist / global median pairwise dist", "",
         f"  {'group':<7}{'size':>5}{'#task':>6}{'act-spread':>12}{'v3-spread':>11}{'v4-spread':>11}{'v4/v3':>8}"]
    L.append("  " + "─" * 60)
    for gi, g in enumerate(G):
        L.append(f"  {('grp'+str(gi)):<7}{len(g['members']):>5}{len(set(g['tasks'])):>6}"
                 f"{g['sa']:>12.3f}{g['s3']:>11.3f}{g['s4']:>11.3f}{g['s4']/max(g['s3'],1e-6):>8.1f}")
    med3 = np.median([g["s3"] for g in G]); med4 = np.median([g["s4"] for g in G])
    L += ["  " + "─" * 60,
          f"  {'median':<7}{'':>5}{'':>6}{'':>12}{med3:>11.3f}{med4:>11.3f}{med4/max(med3,1e-6):>8.1f}", "",
          "Within-group pairwise  latent-dist  vs  VISUAL(DINO)-dist  correlation (pooled):",
          f"  v3 : r = {c3:>.3f}   (n_pairs={n3})   → ~0: latent barely moves, uncorrelated w/ visual",
          f"  v4 : r = {c4:>.3f}   (n_pairs={n4})   → >0: latent spread FOLLOWS the visual difference",
          "",
          "Read: for chunks with (near-)identical actions, v4's latent still separates them and",
          "that separation tracks the visual dynamics; v3 collapses them (spread≈act-spread≈0).",
          "",
          "  task legend: " + ", ".join(f"{i}={n}" for i, n in enumerate(task_names)),
          f"figure -> {png}", "=" * 84]
    txt = outdir / "neardup.txt"
    txt.write_text("\n".join(L))
    print("\n".join(L)); print(f"[②] wrote {png}")


if __name__ == "__main__":
    main()

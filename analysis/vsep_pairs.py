"""②″  SAME-ACTION / DIFFERENT-CONTEXT — point-distribution view with x0→x1 pairs.

Goal (exactly what we want to show):
  Take action chunks whose ACTIONS are near-identical (so the *motion* looks the
  same in the frames), among which the surrounding context still differs — e.g.
  one chunk is lowering the hand TO grasp an object, another is lowering an object
  that is ALREADY grasped. The arm trajectory is ~the same; the visual/semantic
  situation is not.

  Then show, as a POINT-DISTRIBUTION plot, that the tokenized action latent of
  these near-identical-action chunks:
     • v3 (action-only): collapses to (almost) one location — it cannot see the
       difference, the action is the same.
     • v4 (DINO-fused):  spreads out — it places them at different locations
       because it sees the different visual context.

For each selected near-dup group we draw:
  - left  : v3 latent of the group (within-group PCA-2, scaled to global-median
            units) → a tight blob.
  - right : v4 latent of the same group, SAME axis scale → a spread cloud.
  - bottom: x0 (first frame) over x1 (last frame) pairs for K members sampled
            across the group's visual axis; each pair's border color matches its
            point in the scatters, so you can read "this end of the v4 spread =
            this kind of scene" (e.g. empty gripper reaching vs object in gripper).

Groups are chosen to MAXIMIZE within-group visual diversity (not by task), so the
shown pairs actually differ in context. Reuses the shared cache; re-fetches the
identical x0/x1 frames from the datasets. Run from repo root, gr00t-actlat env.
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
from matplotlib import cm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vsep_collect import build_datasets  # noqa: E402
from vsep_frames import fetch_frame       # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402


def _std(X):
    return StandardScaler().fit_transform(X.reshape(X.shape[0], -1))


def _gmed(X, seed, n=40000):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, X.shape[0], n); j = rng.integers(0, X.shape[0], n); ok = i != j
    return float(np.median(np.linalg.norm(X[i[ok]] - X[j[ok]], axis=1)))


def _spread(mem, X, gmed):
    if len(mem) < 2:
        return 0.0
    P = X[mem]
    # mean pairwise distance via a capped random sample for large groups
    m = len(mem)
    if m > 120:
        rng = np.random.default_rng(0)
        a = rng.integers(0, m, 8000); b = rng.integers(0, m, 8000); ok = a != b
        d = np.linalg.norm(P[a[ok]] - P[b[ok]], axis=1)
    else:
        d = [np.linalg.norm(P[a] - P[b]) for a in range(m) for b in range(a + 1, m)]
    return float(np.mean(d) / gmed)


def find_groups(Xa, radius, min_size, max_scan=250):
    """Near-duplicate action groups (NO task constraint), low mutual overlap.
    Greedy over size with <50% overlap ⇒ groups are spread across action space
    (different motion phases)."""
    nn = NearestNeighbors(radius=radius).fit(Xa)
    neigh = nn.radius_neighbors(Xa, return_distance=False)
    cand = [np.asarray(m) for m in neigh if len(m) >= min_size]
    cand.sort(key=len, reverse=True)
    chosen, used = [], np.zeros(Xa.shape[0], bool)
    for mem in cand:
        if used[mem].mean() > 0.5:
            continue
        chosen.append(mem); used[mem] = True
        if len(chosen) >= max_scan:
            break
    return chosen


def group_pca2(Xz, mem, gmed):
    P = PCA(n_components=2, random_state=0).fit_transform(Xz[mem])
    return P / gmed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(_REPO_ROOT / "analysis" / "output" / "visual_sep_gr1" / "cache.npz"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius-pct", type=float, default=0.7,
                    help="action-distance percentile = near-dup radius (smaller = stricter same-action)")
    ap.add_argument("--min-size", type=int, default=12)
    ap.add_argument("--n-groups", type=int, default=3, help="how many groups to render")
    ap.add_argument("--rank", choices=["visual", "size"], default="visual",
                    help="visual = most scene-diverse groups; size = action-space-spread phases")
    ap.add_argument("--n-pairs", type=int, default=7, help="x0/x1 pairs shown per group")
    ap.add_argument("--clean", action="store_true",
                    help="color ONLY the selected (paired) points; rest gray; no number labels")
    ap.add_argument("--suffix", default="", help="filename suffix, e.g. _clean")
    ap.add_argument("--min-tasks", type=int, default=1, help="only keep groups spanning >= this many tasks")
    ap.add_argument("--skip", type=int, default=0, help="skip the first N qualifying groups (to get later ranks)")
    ap.add_argument("--only-group", type=int, default=-1,
                    help="render ONLY this group index (matches the gi in the filename)")
    ap.add_argument("--keep-pairs", default="",
                    help="comma-list of 1-indexed pair positions to keep colored; others gray, "
                         "bottom strip shows only these (e.g. '5,7')")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    A, Z3, Z4, Vc = d["A"], d["Z3"], d["Z4"], d["Vcontext"]
    task = d["task"]; samp_task = d["samp_task"]; samp_local = d["samp_local"]
    task_names = meta["task_names"]
    outdir = Path(args.cache).parent

    Xa, X3, X4, Xv = _std(A), _std(Z3), _std(Z4), _std(Vc)
    gmed3, gmed4 = _gmed(X3, args.seed), _gmed(X4, args.seed)

    # near-dup radius from the action-distance distribution
    rng = np.random.default_rng(args.seed)
    i = rng.integers(0, Xa.shape[0], 40000); j = rng.integers(0, Xa.shape[0], 40000); ok = i != j
    pd = np.linalg.norm(Xa[i[ok]] - Xa[j[ok]], axis=1)
    radius = float(np.percentile(pd, args.radius_pct))
    print(f"[pairs] action pairdist p{args.radius_pct}={radius:.3f} median={np.median(pd):.3f}")

    groups = find_groups(Xa, radius, args.min_size)
    if not groups:
        print("[pairs] no near-dup groups — raise --radius-pct / --min-size lower."); return
    gmedV = _gmed(Xv, args.seed)
    if args.rank == "visual":   # most scene-diverse groups first
        ranked = sorted(groups, key=lambda m: _spread(m, Xv, gmedV), reverse=True)
    else:                        # size order = spread across action space (different phases)
        ranked = groups
    ntask = lambda m: len(set(int(t) for t in task[m]))
    ranked = [m for m in ranked if ntask(m) >= args.min_tasks]
    print(f"[pairs] {len(ranked)} groups with >= {args.min_tasks} tasks; skip {args.skip}, take {args.n_groups}")
    groups = ranked[args.skip: args.skip + args.n_groups]
    if not groups:
        print("[pairs] no groups after task-filter/skip."); return

    datasets, _ = build_datasets(SimpleNamespace(
        dataset_path=meta["dataset_path"], data_config=meta["data_config"],
        embodiment_tag=meta["embodiment_tag"], val_ratio=meta["val_ratio"],
        val_seed=meta["val_seed"], normalization_mode=meta["normalization_mode"],
        image_size=meta["image_size"], video_backend=meta["video_backend"],
        fixed_val_path=meta["fixed_val_path"]))

    # GLOBAL 2D map of each latent space (fit on ALL points) → shows the group
    # sitting as a tight clump (v3) vs spread out (v4) within the full latent cloud.
    E3 = PCA(n_components=2, random_state=0).fit_transform(X3)
    E4 = PCA(n_components=2, random_state=0).fit_transform(X4)
    lim3 = np.percentile(np.abs(E3), 99.5); lim4 = np.percentile(np.abs(E4), 99.5)

    cmap = plt.get_cmap("turbo")
    report = ["=" * 80, "②″  SAME-ACTION / DIFFERENT-CONTEXT  (point distribution + x0→x1 pairs)",
              "=" * 80, f"near-dup radius = action pairdist p{args.radius_pct} = {radius:.3f}",
              "spread = within-group mean pairwise latent dist / global median", ""]

    for gi, mem in enumerate(groups):
        if args.only_group >= 0 and gi != args.only_group:
            continue
        m = len(mem)
        s3 = _spread(mem, X3, gmed3); s4 = _spread(mem, X4, gmed4)
        sa = _spread(mem, Xa, _gmed(Xa, args.seed)); sv = _spread(mem, Xv, gmedV)
        vpc = PCA(n_components=2, random_state=0).fit_transform(Xv[mem])[:, 0]
        order = np.argsort(vpc)
        rank = np.empty(m); rank[order] = np.linspace(0, 1, m)
        colors = cmap(rank)
        K = min(args.n_pairs, m)
        sel_local = order[np.linspace(0, m - 1, K).astype(int)]   # indices INTO mem (0..m-1)
        sel_glob = [int(mem[li]) for li in sel_local]             # -> global chunk indices, per pair position
        sel_colors = cmap(np.linspace(0.05, 0.95, K))             # per-position color (1..K)
        # which pairs to KEEP colored + show below (default: all)
        keep = ([p - 1 for p in map(int, args.keep_pairs.split(","))] if args.keep_pairs
                else list(range(K)))
        keep = [c for c in keep if 0 <= c < K]
        kept_members = {sel_glob[c] for c in keep}               # global indices of kept pairs

        # ---- figure: group on the GLOBAL latent map (gray = all chunks) ----
        ncol = max(len(keep), 2)
        fig = plt.figure(figsize=(max(ncol * 1.7, 9), 8.8))
        outer = fig.add_gridspec(2, 1, height_ratios=[3.2, 2.3], hspace=0.22)
        top = outer[0].subgridspec(1, 2, wspace=0.08)
        bot = outer[1].subgridspec(2, ncol, hspace=0.05, wspace=0.08)
        axv3 = fig.add_subplot(top[0, 0]); axv4 = fig.add_subplot(top[0, 1])
        for ax, E, lim, tag, sp in ((axv3, E3, lim3, "v3 (action-only)", s3),
                                    (axv4, E4, lim4, "v4 (DINO-fused)", s4)):
            ax.scatter(E[:, 0], E[:, 1], s=4, c="0.85", alpha=0.35, linewidths=0)   # all chunks
            if args.clean:
                gray_li = [li for li in range(m) if int(mem[li]) not in kept_members]  # rest of group + non-kept pairs
                ax.scatter(E[[mem[li] for li in gray_li], 0], E[[mem[li] for li in gray_li], 1],
                           c="0.5", s=30, alpha=0.9, linewidths=0)
                ax.scatter([E[sel_glob[c], 0] for c in keep], [E[sel_glob[c], 1] for c in keep],
                           c=[sel_colors[c] for c in keep], s=55, alpha=1.0,
                           edgecolors="k", linewidths=0.5, zorder=3)                # kept pairs only
            else:
                ax.scatter(E[mem, 0], E[mem, 1], c=colors, s=34, alpha=0.95, linewidths=0)
                for r_i, li in enumerate(sel_local):
                    ax.scatter(E[mem[li], 0], E[mem[li], 1], s=170, facecolors="none",
                               edgecolors="k", linewidths=1.7)
                    ax.annotate(str(r_i + 1), (E[mem[li], 0], E[mem[li], 1]), fontsize=9,
                                fontweight="bold", ha="center", va="center")
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            ax.set_title(f"{tag}   group spread = {sp:.2f}", fontsize=12)
            ax.grid(alpha=0.2); ax.set_xticklabels([]); ax.set_yticklabels([])
        axv3.set_ylabel("global latent map (PCA-2)")

        # image pairs: row0 = x0 (first), row1 = x1 (last) — only the kept pairs
        for j, c in enumerate(keep):
            n = sel_glob[c]; col = sel_colors[c] if args.clean else colors[sel_local[c]]
            x0 = fetch_frame(datasets, samp_task[n], samp_local[n], "frame_x0")
            x1 = fetch_frame(datasets, samp_task[n], samp_local[n], "frame_x1")
            for r, img in ((0, x0), (1, x1)):
                ax = fig.add_subplot(bot[r, j]); ax.imshow(img)
                ax.set_xticks([]); ax.set_yticks([])
                for sp_ in ax.spines.values():
                    sp_.set_edgecolor(col); sp_.set_linewidth(4)
                if j == 0:
                    ax.set_ylabel("x0 (first)" if r == 0 else "x1 (last)", fontsize=9)
        fig.suptitle(
            f"②″  {m} real chunks with NEAR-IDENTICAL action (Δaction spread={sa:.2f}) "
            f"but different situation (visual spread={sv:.2f})   (grp{gi})\n"
            f"v3 (action-only): they stay together — group spread={s3:.2f}   ·   "
            f"v4 (DINO-fused): they separate — group spread={s4:.2f}  ({s4/max(s3,1e-6):.1f}× wider)\n"
            "light gray = all chunks; dark gray = rest of group; COLORED points ↔ same-colored x0/x1 pairs below"
            " (motion same, scene differs)",
            fontsize=11, y=0.99)
        png = outdir / f"pairs_grp{gi}{args.suffix}.png"
        fig.savefig(png, dpi=125, bbox_inches="tight"); plt.close(fig)
        report += [f"grp{gi}: size={m}  #task={len(set(int(t) for t in task[mem]))}  "
                   f"act-spread={sa:.3f}  vis-spread={sv:.3f}  v3={s3:.3f}  v4={s4:.3f}  ({s4/max(s3,1e-6):.1f}×)",
                   f"   -> {png}"]
        print(report[-2]); print(f"[pairs] wrote {png}")

    (outdir / "pairs.txt").write_text("\n".join(report + ["=" * 80]))


if __name__ == "__main__":
    main()

"""Compose several hand-picked (group, pair-pair) selections into ONE figure.

Each row = one selection: [v3 scatter | v4 scatter | pair A (x0/x1) | pair B (x0/x1)].
No descriptive suptitle; the per-scatter titles (v3/v4 + group spread) are kept.
Reproduces each group exactly like vsep_pairs (same radius/min_size/rank/skip),
selects group index `only`, keeps only the requested 1-indexed pair positions.

Run from repo root, gr00t-actlat env (frames are re-fetched from the datasets).
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vsep_pairs import _std, _gmed, _spread, find_groups   # noqa: E402
from vsep_frames import fetch_frame                          # noqa: E402
from vsep_collect import build_datasets                      # noqa: E402

CACHE = _REPO_ROOT / "analysis" / "output" / "visual_sep_gr1" / "cache.npz"
RADIUS_PCT = 0.1
N_PAIRS = 7

# (label, min_size, min_tasks, skip, only_group, [1-indexed pairs to keep])
SELECTIONS = [
    ("grp2_clean",  8, 1, 0,  2, [5, 7]),
    ("grp6_multi",  8, 2, 3,  6, [2, 7]),
    ("grp4_multi2", 5, 2, 11, 4, [6, 7]),
    ("grp6_multi2", 5, 2, 11, 6, [3, 4]),
]


def reproduce_group(Xa, Xv, task, radius, min_size, min_tasks, skip, only, seed=0):
    groups = find_groups(Xa, radius, min_size)          # size-desc, greedy overlap<0.5
    ntask = lambda m: len(set(int(t) for t in task[m]))
    ranked = [m for m in groups if ntask(m) >= min_tasks]
    idx = skip + only
    if idx >= len(ranked):
        raise IndexError(f"only-group {only} (skip {skip}) out of range: {len(ranked)} groups")
    return ranked[idx]


def main():
    d = np.load(CACHE, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    A, Z3, Z4, Vc = d["A"], d["Z3"], d["Z4"], d["Vcontext"]
    task = d["task"]; samp_task = d["samp_task"]; samp_local = d["samp_local"]
    Xa, X3, X4, Xv = _std(A), _std(Z3), _std(Z4), _std(Vc)
    gmed3, gmed4, gmedA = _gmed(X3, 0), _gmed(X4, 0), _gmed(Xa, 0)

    rng = np.random.default_rng(0)
    i = rng.integers(0, Xa.shape[0], 40000); j = rng.integers(0, Xa.shape[0], 40000); ok = i != j
    radius = float(np.percentile(np.linalg.norm(Xa[i[ok]] - Xa[j[ok]], axis=1), RADIUS_PCT))

    E3 = PCA(n_components=2, random_state=0).fit_transform(X3)
    E4 = PCA(n_components=2, random_state=0).fit_transform(X4)
    lim3 = np.percentile(np.abs(E3), 99.5); lim4 = np.percentile(np.abs(E4), 99.5)
    cmap = plt.get_cmap("turbo")

    datasets, _ = build_datasets(SimpleNamespace(
        dataset_path=meta["dataset_path"], data_config=meta["data_config"],
        embodiment_tag=meta["embodiment_tag"], val_ratio=meta["val_ratio"],
        val_seed=meta["val_seed"], normalization_mode=meta["normalization_mode"],
        image_size=meta["image_size"], video_backend=meta["video_backend"],
        fixed_val_path=meta["fixed_val_path"]))

    nrow = len(SELECTIONS)
    fig = plt.figure(figsize=(15.5, 4.2 * nrow))
    outer = fig.add_gridspec(nrow, 1, hspace=0.32)

    for r, (label, min_size, min_tasks, skip, only, keep1) in enumerate(SELECTIONS):
        mem = reproduce_group(Xa, Xv, task, radius, min_size, min_tasks, skip, only)
        m = len(mem)
        s3, s4 = _spread(mem, X3, gmed3), _spread(mem, X4, gmed4)
        vpc = PCA(n_components=2, random_state=0).fit_transform(Xv[mem])[:, 0]
        order = np.argsort(vpc)
        K = min(N_PAIRS, m)
        sel_local = order[np.linspace(0, m - 1, K).astype(int)]
        sel_glob = [int(mem[li]) for li in sel_local]
        sel_colors = cmap(np.linspace(0.05, 0.95, K))
        keep = [p - 1 for p in keep1 if 0 <= p - 1 < K]
        kept = {sel_glob[c] for c in keep}

        row = outer[r].subgridspec(2, 4, width_ratios=[2.2, 2.2, 1, 1], wspace=0.12, hspace=0.06)
        ax3 = fig.add_subplot(row[:, 0]); ax4 = fig.add_subplot(row[:, 1])
        for ax, E, lim, tag, sp in ((ax3, E3, lim3, "v3 (action-only)", s3),
                                    (ax4, E4, lim4, "v4 (DINO-fused)", s4)):
            ax.scatter(E[:, 0], E[:, 1], s=4, c="0.85", alpha=0.35, linewidths=0)
            gray_li = [li for li in range(m) if int(mem[li]) not in kept]
            ax.scatter(E[[mem[li] for li in gray_li], 0], E[[mem[li] for li in gray_li], 1],
                       c="0.5", s=26, alpha=0.9, linewidths=0)
            ax.scatter([E[sel_glob[c], 0] for c in keep], [E[sel_glob[c], 1] for c in keep],
                       c=[sel_colors[c] for c in keep], s=70, alpha=1.0,
                       edgecolors="k", linewidths=0.6, zorder=3)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            ax.set_title(f"{tag}   group spread = {sp:.2f}", fontsize=11)
            ax.grid(alpha=0.2); ax.set_xticklabels([]); ax.set_yticklabels([])
        ax3.set_ylabel(f"{label}\n({m} chunks)", fontsize=10)

        for jj, c in enumerate(keep):
            n = sel_glob[c]; col = sel_colors[c]
            x0 = fetch_frame(datasets, samp_task[n], samp_local[n], "frame_x0")
            x1 = fetch_frame(datasets, samp_task[n], samp_local[n], "frame_x1")
            for rr, im in ((0, x0), (1, x1)):
                ax = fig.add_subplot(row[rr, 2 + jj]); ax.imshow(im)
                ax.set_xticks([]); ax.set_yticks([])
                for sp_ in ax.spines.values():
                    sp_.set_edgecolor(col); sp_.set_linewidth(4)
                if jj == 0:
                    ax.set_ylabel("x0" if rr == 0 else "x1", fontsize=8)

    out = CACHE.parent / "pairs_composite.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[compose] wrote {out}")


if __name__ == "__main__":
    main()

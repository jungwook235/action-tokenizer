"""Save one visual example of the "무작위 1개" comparison.

Picks a representative robot anchor chunk and shows the three chunks that the distance
table compares -- the anchor itself, its H-same target (paired human episode, same object,
same phase) and one randomly drawn R-diff target (robot, different object, same phase) --
as the 224x224 frame pairs that DINO actually saw, annotated with the real cosine distance
in the latent (wrist only) space.

The anchor is chosen as the MEDIAN case of d(H-same) - d(R-diff), not the best one, so the
picture is representative rather than cherry-picked. The R-diff draw uses seed 0 here; it
is an independent draw from the one inside the table's own loop, so its distance will not
match a table cell exactly -- the table cell is a mean over all anchors.

Run: python analysis/hrdexdb_egodex/pair_example_figure.py
"""

import importlib.util
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
_here = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pld = _load("pair_latent_distance")
    d = np.load(f"{OUT}/pair_latents.npz", allow_pickle=True)
    obj, pair, kind, phase, ep = d["obj"], d["pair"], d["kind"], d["phase"], d["ep"]
    X = d["lat_wrist"]
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    D = 1.0 - Xn @ Xn.T

    rng = np.random.default_rng(0)
    cand = []
    for i in np.where(kind == "robot")[0]:
        sp, so = phase == phase[i], obj == obj[i]
        hs = np.where(sp & so & (kind == "human") & (pair == pair[i]))[0]
        rd_pool = np.where(sp & ~so & (kind == "robot"))[0]
        if not len(hs) or not len(rd_pool):
            continue
        rd = int(rng.choice(rd_pool, 1)[0])
        cand.append((i, int(hs[0]), rd, D[i, hs[0]] - D[i, rd]))
    gaps = np.array([c[3] for c in cand])
    pick = cand[int(np.argsort(gaps)[len(gaps) // 2])]        # median case
    i, j, k, gap = pick
    print(f"anchor  : {ep[i]}  [{kind[i]}, {phase[i]}]")
    print(f"H-same  : {ep[j]}  [{kind[j]}, {phase[j]}]  d={D[i,j]:.4f}")
    print(f"R-diff  : {ep[k]}  [{kind[k]}, {phase[k]}]  d={D[i,k]:.4f}")
    print(f"median-case gap d(H-same)-d(R-diff) = {gap:+.4f}")

    rows = [("ANCHOR (robot)", i, None),
            ("H-same  (human, same object + paired episode)", j, D[i, j]),
            ("R-diff  (robot, different object)", k, D[i, k])]
    plt.rcParams.update({"font.size": 9, "figure.facecolor": "#fcfcfb",
                         "text.color": "#0b0b0b"})
    fig, axes = plt.subplots(3, 2, figsize=(6.4, 9.4))
    for r, (label, idx, dist) in enumerate(rows):
        start = pld.lift_onset(pld.object_traj(str(ep[idx])))
        if phase[idx] == "approach":
            start -= pld.H
        x0, x1 = pld.read_frames(str(ep[idx]), start)
        for c, (fr, tag) in enumerate([(x0, f"frame {start}  (x0)"),
                                       (x1, f"frame {start+pld.H-1}  (x1)")]):
            ax = axes[r, c]
            ax.imshow(fr[0].permute(1, 2, 0).numpy().clip(0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#b8b7b0")
            ax.set_title(tag, fontsize=8, color="#52514e", pad=3)
        head = f"{label}\n{str(ep[idx]).split('HRDexDB/')[1]}   |   phase {phase[idx]}"
        if dist is not None:
            head += f"   |   cosine distance to anchor = {dist:.4f}"
        axes[r, 0].text(0, -0.16, head, transform=axes[r, 0].transAxes, va="top",
                        fontsize=9, color="#0b0b0b")
    fig.suptitle("latent (wrist only) distance -- representative (median) case\n"
                 f"same action (H-same) {D[i,j]:.4f}   vs   different action (R-diff) {D[i,k]:.4f}"
                 f"   |   the 224x224 frames DINO actually saw, camera {pld.CAM}",
                 fontsize=10, x=0.02, ha="left", y=0.995)
    fig.subplots_adjust(top=0.90, hspace=0.42, wspace=0.06)
    p = f"{OUT}/pair_example_median.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"saved {p}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    main()

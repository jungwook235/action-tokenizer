"""Distances + figure for the paired human/robot latent analysis.

Reads out/pair_latents.npz (written by pair_latent_distance.py) and, for every ROBOT
anchor chunk, measures cosine distance to four target groups:
    H-same : paired human episode, SAME object   (same action, cross-embodiment)
    R-same : other robot episode, SAME object    (same action, same embodiment)
    R-diff : robot, DIFFERENT object             (diff action, same embodiment)
    H-diff : human, DIFFERENT object             (diff action, cross-embodiment)
Targets are always matched on phase (approach/lift), so phase never confounds a group.

Outputs: console table + out/pair_latent_distance.png
"""

import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
GROUPS = ["H-same", "H-diff", "R-same", "R-diff"]
# validated categorical slots 1-4 (dataviz reference palette), fixed order
COLORS = {"H-same": "#2a78d6", "H-diff": "#eb6834", "R-same": "#1baf7a", "R-diff": "#eda100"}
STYLES = {"H-same": "-", "H-diff": "--", "R-same": "-", "R-diff": "--"}


def cosdist(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return 1.0 - A @ B.T


def collect(X, obj, pair, kind, phase):
    """{group: distances}, plus per-anchor paired records for the decisive test."""
    D = cosdist(X, X)
    out = {g: [] for g in GROUPS}
    per_anchor = []
    for i in np.where(kind == "robot")[0]:
        same_ph = phase == phase[i]
        same_ob = obj == obj[i]
        hs = np.where(same_ph & same_ob & (kind == "human") & (pair == pair[i]))[0]
        rs = np.where(same_ph & same_ob & (kind == "robot") & (pair != pair[i]))[0]
        rd = np.where(same_ph & ~same_ob & (kind == "robot"))[0]
        hd = np.where(same_ph & ~same_ob & (kind == "human"))[0]
        if not len(hs):
            continue
        out["H-same"] += list(D[i, hs])
        out["R-same"] += list(D[i, rs])
        out["R-diff"] += list(D[i, rd])
        out["H-diff"] += list(D[i, hd])
        # retrieval: among ALL human chunks of the same phase, is the paired one nearest?
        hall = np.where(same_ph & (kind == "human"))[0]
        rank = int(np.sum(D[i, hall] < D[i, hs].min()))
        per_anchor.append(dict(i=i, obj=obj[i], phase=phase[i],
                               d_hs=float(D[i, hs].mean()), d_hd=float(D[i, hd].mean()),
                               d_rs=float(D[i, rs].mean()) if len(rs) else np.nan,
                               d_rd=float(D[i, rd].mean()) if len(rd) else np.nan,
                               top1=rank == 0, rank=rank, n_h=len(hall)))
    return {g: np.array(v) for g, v in out.items()}, per_anchor


def table(name, dist, pa):
    print(f"\n===== {name} =====")
    print(f"{'group':8s} {'n':>5s} {'mean':>7s} {'median':>7s} {'std':>7s}")
    for g in GROUPS:
        v = dist[g]
        if not len(v):
            continue
        print(f"{g:8s} {len(v):5d} {v.mean():7.4f} {np.median(v):7.4f} {v.std():7.4f}")
    dhs = np.array([p["d_hs"] for p in pa])
    dhd = np.array([p["d_hd"] for p in pa])
    delta = dhs - dhd
    win = (delta < 0).mean()
    top1 = np.mean([p["top1"] for p in pa])
    nh = int(np.mean([p["n_h"] for p in pa]))
    print(f"per-anchor (cross-embodiment): d(H-same) < d(H-diff) in {win*100:.1f}% of "
          f"{len(pa)} anchors (mean delta {delta.mean():+.4f})")
    print(f"top-1 retrieval of the paired human chunk among {nh} human chunks: "
          f"{top1*100:.1f}%  (chance {100/nh:.1f}%)")
    # within-embodiment control: same-object robot vs different-object robot
    rs = np.array([p["d_rs"] for p in pa], dtype=float)
    rd = np.array([p["d_rd"] for p in pa], dtype=float)
    m = np.isfinite(rs) & np.isfinite(rd)
    if m.sum():
        dr = rs[m] - rd[m]
        print(f"per-anchor (within-embodiment): d(R-same) < d(R-diff) in "
              f"{(dr<0).mean()*100:.1f}% of {int(m.sum())} anchors (mean delta {dr.mean():+.4f})")
    return dict(delta=delta, win=win, top1=top1, chance=1 / nh)


def distance_tables(d, out_md):
    """Markdown tables of the actual distances (not win-rates), for the record.

    Three H-diff/R-diff target definitions x three distance spaces:
      * latent (wrist only)  -- cosine distance on the flattened [16,64] latent
      * raw action (wrist only) -- cosine distance on the flattened [16,57] normalized action
                                   with the camera dims zeroed, i.e. the SAME condition as
                                   the wrist-only latent (with the camera dims left in this
                                   is not an apples-to-apples baseline)
      * physical wrist trajectory -- mean per-frame wrist position difference, in cm
    """
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "chk", os.path.join(here, "check_mano_egodex_correspondence.py"))
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)

    obj, pair, kind, phase = d["obj"], d["pair"], d["kind"], d["phase"]
    n = len(obj)
    raw_w = d["raw"].reshape(n, 16, 57).copy()
    raw_w[:, :, 0:9] = 0.0
    mn, mx = chk.egodex_minmax_57()
    pos = (d["raw"].reshape(n, 16, 57)[:, :, 9:12] + 1) / 2 * (mx[9:12] - mn[9:12]) + mn[9:12]
    D_phys = np.stack([np.linalg.norm(pos - pos[i], axis=2).mean(1) for i in range(n)]) * 100
    spaces = [("latent (wrist only), cosine", cosdist(d["lat_wrist"], d["lat_wrist"]), 4),
              ("raw action (wrist only), cosine", cosdist(raw_w.reshape(n, -1), raw_w.reshape(n, -1)), 4),
              ("physical wrist trajectory, cm", D_phys, 2)]
    modes = [("all", "다른 물체 전부"), ("pair", "pair-index 일치"), ("one", "무작위 1개")]
    rng = np.random.default_rng(0)

    lines = ["# Distance tables (mean / median)", "",
             f"chunks {n} | objects {len(set(obj))} | robot {int((kind=='robot').sum())} "
             f"human {int((kind=='human').sum())} | anchors = robot chunks", ""]
    for title, D, prec in spaces:
        lines += [f"## {title}", "",
                  "| H-diff 정의 | H-same | H-diff | R-same | R-diff | anchor별 (H-same − H-diff) |",
                  "|---|---|---|---|---|---|"]
        for mode, label in modes:
            g = {k: [] for k in GROUPS}
            delta = []
            for i in np.where(kind == "robot")[0]:
                sp, so = phase == phase[i], obj == obj[i]
                hs = np.where(sp & so & (kind == "human") & (pair == pair[i]))[0]
                if not len(hs):
                    continue
                bh, br = sp & ~so & (kind == "human"), sp & ~so & (kind == "robot")
                if mode == "pair":
                    hd, rd = np.where(bh & (pair == pair[i]))[0], np.where(br & (pair == pair[i]))[0]
                elif mode == "one":
                    hd, rd = rng.choice(np.where(bh)[0], 1), rng.choice(np.where(br)[0], 1)
                else:
                    hd, rd = np.where(bh)[0], np.where(br)[0]
                rs = np.where(sp & so & (kind == "robot") & (pair != pair[i]))[0]
                g["H-same"] += list(D[i, hs]); g["H-diff"] += list(D[i, hd])
                g["R-same"] += list(D[i, rs]); g["R-diff"] += list(D[i, rd])
                delta.append(D[i, hs].mean() - D[i, hd].mean())
            f = lambda k: f"{np.mean(g[k]):.{prec}f} / {np.median(g[k]):.{prec}f}"
            lines.append(f"| {label} | {f('H-same')} | {f('H-diff')} | {f('R-same')} | "
                         f"{f('R-diff')} | {np.mean(delta):+.{prec}f} |")
        lines.append("")
    open(out_md, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved {out_md}")


def kde(v, grid):
    v = np.asarray(v)
    h = 1.06 * v.std() * len(v) ** (-1 / 5) + 1e-9
    return np.exp(-0.5 * ((grid[:, None] - v[None]) / h) ** 2).sum(1) / (len(v) * h * np.sqrt(2 * np.pi))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(f"{OUT}/pair_latents.npz", allow_pickle=True)
    obj, pair, kind, phase = d["obj"], d["pair"], d["kind"], d["phase"]
    n = len(obj)
    raw_w = d["raw"].reshape(n, 16, 57).copy()
    raw_w[:, :, 0:9] = 0.0          # match the wrist-only latent condition
    variants = [("latent (wrist only)", d["lat_wrist"]),
                ("latent (wrist + camera)", d["lat_wristcam"]),
                ("raw action (wrist only)", raw_w.reshape(n, -1))]
    print(f"chunks: {len(obj)}  objects: {len(set(obj))}  "
          f"robot {int((kind=='robot').sum())} / human {int((kind=='human').sum())}")

    results = []
    for name, X in variants:
        dist, pa = collect(X, obj, pair, kind, phase)
        summ = table(name, dist, pa)
        results.append((name, dist, summ))

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b8b7b0",
                         "axes.labelcolor": "#52514e", "text.color": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.6),
                             gridspec_kw={"height_ratios": [1.35, 1]})
    for col, (name, dist, summ) in enumerate(results):
        ax = axes[0, col]
        lo = min(v.min() for v in dist.values() if len(v))
        hi = max(np.percentile(v, 99.5) for v in dist.values() if len(v))
        grid = np.linspace(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo), 400)
        for g in GROUPS:
            if not len(dist[g]):
                continue
            y = kde(dist[g], grid)
            ax.plot(grid, y, STYLES[g], color=COLORS[g], lw=2, label=g, solid_capstyle="round")
            xm = grid[int(np.argmax(y))]
            ax.annotate(g, (xm, y.max()), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8, color="#52514e")
        ax.set_title(name, fontsize=10, loc="left", pad=8)
        ax.set_xlabel("cosine distance to robot anchor")
        ax.set_ylabel("density" if col == 0 else "")
        ax.grid(True, color="#e8e7e1", lw=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if col == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")

        ax = axes[1, col]
        delta = summ["delta"]
        ax.hist(delta, bins=28, color="#8f8e86", edgecolor="#fcfcfb", linewidth=0.6)
        ax.axvline(0, color="#0b0b0b", lw=1.2)
        ax.set_xlabel(r"per-anchor  d(H-same) $-$ d(H-diff)")
        ax.set_ylabel("anchors" if col == 0 else "")
        ax.grid(True, axis="y", color="#e8e7e1", lw=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.annotate(f"same-action closer: {summ['win']*100:.0f}%\n"
                    f"top-1 retrieval: {summ['top1']*100:.0f}% "
                    f"(chance {summ['chance']*100:.0f}%)",
                    xy=(0.02, 0.95), xycoords="axes fraction", va="top", fontsize=8,
                    color="#52514e")
    fig.suptitle("HRDexDB paired human/robot: latent distance to a robot anchor "
                 "(same-action vs different-action)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = f"{OUT}/pair_latent_distance.png"
    fig.savefig(p, dpi=130)
    print(f"\nsaved {p}")

    distance_tables(d, f"{OUT}/distance_tables.md")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    main()

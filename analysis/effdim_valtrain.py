"""Effective dim: VAL vs TRAIN, single (m5-matching) + chunk granularity.

Loads both caches (cache.npz = val, cache_train.npz = train), computes the same
intrinsic-dim metrics as m5 (covariance eigenvalues -> participation ratio + #PCs
for 90/95/99% var, on per-dim z-scored features) for raw action / v3 / v4 at both
granularities, and reports val vs train side by side. Does NOT overwrite the val
effdim_compare.* outputs — writes effdim_valtrain.*.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from effdim_compare import analyze  # noqa: E402  (reuse identical math)

OUTDIR = Path(__file__).resolve().parent / "output" / "visual_sep_gr1"
CACHES = {"val": OUTDIR / "cache.npz", "train": OUTDIR / "cache_train.npz"}


def metrics_for(cache):
    d = np.load(cache, allow_pickle=True)
    A, Z3, Z4 = d["A"], d["Z3"], d["Z4"]
    N = A.shape[0]
    return {
        "N": int(N),
        "single": {"raw_action": analyze(A.reshape(-1, A.shape[-1])),
                   "v3": analyze(Z3.reshape(-1, Z3.shape[-1])),
                   "v4": analyze(Z4.reshape(-1, Z4.shape[-1]))},
        "chunk": {"raw_action": analyze(A.reshape(N, -1)),
                  "v3": analyze(Z3.reshape(N, -1)),
                  "v4": analyze(Z4.reshape(N, -1))}}


def main():
    res = {split: metrics_for(c) for split, c in CACHES.items()}
    json.dump(res, open(OUTDIR / "effdim_valtrain.json", "w"), indent=2)

    name = {"raw_action": "raw action", "v3": "v3 latent", "v4": "v4 latent"}
    units = {"single": {"raw_action": "timestep", "v3": "token", "v4": "token"},
             "chunk": {"raw_action": "16-step chunk", "v3": "chunk", "v4": "chunk"}}
    lines = ["=" * 104,
             f"Effective dim — VAL vs TRAIN   (gr1 1000demo, N_val={res['val']['N']}, N_train={res['train']['N']})",
             "=" * 104]
    for gran in ("single", "chunk"):
        lines.append(f"[{gran}]  " + ("sample=timestep/token, feature=per-step DoF (m5-matching)"
                                      if gran == "single" else "sample=chunk, feature=16 tokens x K"))
        lines.append(f"{'space':<12}{'unit':<15}{'nominal':>8}   "
                     f"{'PR val':>8}{'PR train':>9}   {'pc95 val':>9}{'pc95 tr':>8}   {'red val':>8}{'red tr':>8}")
        lines.append("-" * 104)
        for sp in ("raw_action", "v3", "v4"):
            v, t = res["val"][gran][sp], res["train"][gran][sp]
            lines.append(
                f"{name[sp]:<12}{units[gran][sp]:<15}{v['nominal_dim']:>8}   "
                f"{v['PR_corr']:>8.2f}{t['PR_corr']:>9.2f}   "
                f"{v['n_pc95_corr']:>9}{t['n_pc95_corr']:>8}   "
                f"{v['redundancy_corr']:>8.1f}{t['redundancy_corr']:>8.1f}")
        lines.append("-" * 104)
    lines += ["PR = participation ratio (effective dim);  red = nominal / PR;  pc95 = #PCs for 95% var (z-scored).",
              "single = m5's granularity.  Same N for val & train (balanced 167/task) so differences reflect data dist."]
    txt = "\n".join(lines)
    (OUTDIR / "effdim_valtrain.txt").write_text(txt + "\n")
    print(txt)

    # grouped bars: single granularity, PR val vs train for the 3 spaces
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, gran, ttl in ((axes[0], "single", "single action (m5-matching)"),
                          (axes[1], "chunk", "chunk-flattened latent")):
        spaces = [("raw_action", "raw\naction"), ("v3", "v3\nlatent"), ("v4", "v4\nlatent")]
        labels = [l for _k, l in spaces]
        pv = [res["val"][gran][k]["PR_corr"] for k, _ in spaces]
        pt = [res["train"][gran][k]["PR_corr"] for k, _ in spaces]
        x = np.arange(len(labels)); w = 0.38
        b1 = ax.bar(x - w / 2, pv, w, label="val", color="#7fa8d0", edgecolor="#5b6672")
        b2 = ax.bar(x + w / 2, pt, w, label="train", color="#2f6db5")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.1f}",
                        ha="center", va="bottom", fontsize=8.5, color="#333")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylabel("Effective dim (PR)", fontsize=10)
        ax.set_ylim(0, max(pv + pt) * 1.2)
        ax.set_title(ttl, fontsize=12, fontweight="bold")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("gr1 1000demo — tokenizer latent effective dim: val vs train",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUTDIR / "effdim_valtrain_pr.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")
    print("#### EFFDIM VALTRAIN DONE ####")


if __name__ == "__main__":
    main()

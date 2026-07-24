"""robocasa (OOD) — nominal vs EFFECTIVE dim (pc95) per method, m5-style bars.

For each method (raw action / v3 latent / v4 latent) compare the nominal dimension
against the effective dim measured as #PCs for 95% variance (pc95, z-scored).
Two panels: single (m5-matching, sample=timestep/token) and chunk (flattened).
Reads cache_robocasa.npz.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from effdim_compare import analyze  # noqa: E402

OUTDIR = Path(__file__).resolve().parent / "output" / "visual_sep_gr1"
ROBO = OUTDIR / "cache_robocasa.npz"
METRIC = "n_pc95_corr"
EFF_LABEL = "Effective dim (95% var)"


def main():
    d = np.load(ROBO, allow_pickle=True)
    A, Z3, Z4 = d["A"], d["Z3"], d["Z4"]
    N = A.shape[0]
    res = {
        "single": {"raw_action": analyze(A.reshape(-1, A.shape[-1])),
                   "v3": analyze(Z3.reshape(-1, Z3.shape[-1])),
                   "v4": analyze(Z4.reshape(-1, Z4.shape[-1]))},
        "chunk": {"raw_action": analyze(A.reshape(N, -1)),
                  "v3": analyze(Z3.reshape(N, -1)),
                  "v4": analyze(Z4.reshape(N, -1))}}

    spaces = [("raw_action", "raw\naction"), ("v3", "v3\nlatent"), ("v4", "v4\nlatent")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, gran, ttl in ((axes[0], "single", "single action (m5-matching)\nsample = timestep / token"),
                          (axes[1], "chunk", "chunk-flattened\nsample = 16 tokens × K")):
        labels = [l for _k, l in spaces]
        nominal = [res[gran][k]["nominal_dim"] for k, _ in spaces]
        eff = [res[gran][k][METRIC] for k, _ in spaces]
        redund = [n / e if e > 0 else float("inf") for n, e in zip(nominal, eff)]
        x = np.arange(len(labels)); w = 0.38
        ymax = max(nominal) * 1.30
        b1 = ax.bar(x - w / 2, nominal, w, label="Nominal dim", color="#b9c6d6", edgecolor="#7f8c9b")
        b2 = ax.bar(x + w / 2, eff, w, label=EFF_LABEL, color="#2f6db5")
        for b in b1:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax * 0.012, f"{b.get_height():.0f}",
                    ha="center", va="bottom", fontsize=9, color="#5b6672")
        for b in b2:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax * 0.012, f"{b.get_height():.0f}",
                    ha="center", va="bottom", fontsize=9, color="#2f6db5")
        for i, r in enumerate(redund):
            top = max(nominal[i], eff[i])
            ax.annotate(f"{r:.1f}×", xy=(i, top), xytext=(0, 20), textcoords="offset points",
                        ha="center", va="bottom", fontsize=12, fontweight="bold", color="#c0392b")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Dimensions", fontsize=10); ax.set_ylim(0, ymax)
        ax.set_title(ttl, fontsize=11.5, fontweight="bold")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("robocasa_gr1_tabletop (gr1_unified tokenizer, OOD) — nominal vs effective dim (pc95)",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005, "Effective dim = # PCs for 95% of variance (z-scored);   red × = Nominal / Effective",
             ha="center", va="bottom", fontsize=8, style="italic", color="#7f8c9b")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out = OUTDIR / "effdim_robocasa_pc95.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")
    for gran in ("single", "chunk"):
        for k, _ in spaces:
            r = res[gran][k]
            print(f"  [{gran}] {k:11} nominal={r['nominal_dim']:>5}  pc95={r['n_pc95_corr']:>4}  "
                  f"redund={r['nominal_dim']/r['n_pc95_corr']:.1f}x")
    print("#### DONE ####")


if __name__ == "__main__":
    main()

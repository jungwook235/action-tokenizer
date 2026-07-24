"""Effective dim of the gr1_unified tokenizer latents applied (OOD) to
robocasa_gr1_tabletop/sim_100demos, single (m5-matching) + chunk granularity.

Reads cache_robocasa.npz (A, Z3, Z4). Also prints the gr1_unified VAL numbers
(from cache.npz) side by side for context. Same math as m5 (participation ratio +
#PCs for 90/95/99% var, z-scored). CPU only.
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
GR1VAL = OUTDIR / "cache.npz"


def metrics_for(cache):
    d = np.load(cache, allow_pickle=True)
    A, Z3, Z4 = d["A"], d["Z3"], d["Z4"]
    N = A.shape[0]
    return {"N": int(N),
            "single": {"raw_action": analyze(A.reshape(-1, A.shape[-1])),
                       "v3": analyze(Z3.reshape(-1, Z3.shape[-1])),
                       "v4": analyze(Z4.reshape(-1, Z4.shape[-1]))},
            "chunk": {"raw_action": analyze(A.reshape(N, -1)),
                      "v3": analyze(Z3.reshape(N, -1)),
                      "v4": analyze(Z4.reshape(N, -1))}}


def main():
    robo = metrics_for(ROBO)
    gr1 = metrics_for(GR1VAL)
    json.dump({"robocasa": robo, "gr1_val": gr1}, open(OUTDIR / "effdim_robocasa.json", "w"), indent=2)

    name = {"raw_action": "raw action", "v3": "v3 latent", "v4": "v4 latent"}
    units = {"single": {"raw_action": "timestep", "v3": "token", "v4": "token"},
             "chunk": {"raw_action": "16-step chunk", "v3": "chunk", "v4": "chunk"}}
    lines = ["=" * 104,
             "Effective dim — gr1_unified tokenizer  |  robocasa_gr1_tabletop (OOD)  vs  gr1_unified VAL (in-dist)",
             f"  N_robocasa={robo['N']} (split=all)   N_gr1val={gr1['N']}",
             "  tokenizer: gr1_1000demos v3_recon_ln_bn16 / v4_recon_dino_bn64_l1_mse_naiveln_vae",
             "=" * 104]
    for gran in ("single", "chunk"):
        lines.append(f"[{gran}]  " + ("sample=timestep/token (m5-matching)" if gran == "single"
                                      else "sample=chunk (16 tokens x K flattened)"))
        lines.append(f"{'space':<12}{'unit':<15}{'nominal':>8}   "
                     f"{'PR robo':>9}{'PR gr1val':>10}   {'pc95 robo':>10}{'pc95 g1v':>9}   "
                     f"{'red robo':>9}{'red g1v':>8}")
        lines.append("-" * 104)
        for sp in ("raw_action", "v3", "v4"):
            r, g = robo[gran][sp], gr1[gran][sp]
            lines.append(
                f"{name[sp]:<12}{units[gran][sp]:<15}{r['nominal_dim']:>8}   "
                f"{r['PR_corr']:>9.2f}{g['PR_corr']:>10.2f}   "
                f"{r['n_pc95_corr']:>10}{g['n_pc95_corr']:>9}   "
                f"{r['redundancy_corr']:>9.1f}{g['redundancy_corr']:>8.1f}")
        lines.append("-" * 104)
    lines += ["PR = participation ratio (effective dim);  red = nominal/PR;  pc95 = #PCs for 95% var (z-scored).",
              "robocasa = gr1_unified tokenizer applied OOD; gr1val = the tokenizer's own (in-distribution) val."]
    txt = "\n".join(lines)
    (OUTDIR / "effdim_robocasa.txt").write_text(txt + "\n")
    print(txt)

    # bar chart: single granularity, PR robocasa vs gr1-val for the 3 spaces
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, gran, ttl in ((axes[0], "single", "single action (m5-matching)"),
                          (axes[1], "chunk", "chunk-flattened latent")):
        spaces = [("raw_action", "raw\naction"), ("v3", "v3\nlatent"), ("v4", "v4\nlatent")]
        labels = [l for _k, l in spaces]
        pr = [robo[gran][k]["PR_corr"] for k, _ in spaces]
        pg = [gr1[gran][k]["PR_corr"] for k, _ in spaces]
        x = np.arange(len(labels)); w = 0.38
        b1 = ax.bar(x - w / 2, pr, w, label="robocasa (OOD)", color="#c0714b", edgecolor="#7a4429")
        b2 = ax.bar(x + w / 2, pg, w, label="gr1_unified val", color="#2f6db5")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.1f}",
                        ha="center", va="bottom", fontsize=8.5, color="#333")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylabel("Effective dim (PR)", fontsize=10)
        ax.set_ylim(0, max(pr + pg) * 1.2)
        ax.set_title(ttl, fontsize=12, fontweight="bold")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("gr1_unified tokenizer latent effective dim: robocasa (OOD) vs gr1_unified val",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUTDIR / "effdim_robocasa_pr.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")
    print("#### EFFDIM ROBOCASA DONE ####")


if __name__ == "__main__":
    main()

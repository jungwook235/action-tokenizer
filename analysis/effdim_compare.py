"""Effective dim of gr1 latents at m5-matching granularity vs chunk-flattened.

m5 (m5_intrinsic_dim_*.py) computes intrinsic dim on SINGLE per-timestep actions:
sample = one timestep, feature = joint DoF (nominal = 29 for gr1 / 44 dual / 16 hand).
It does NOT flatten the 16-step chunk.

The tokenizer takes a 16-step chunk and emits 16 latent tokens of width K
(K=16 v3, K=64 v4). So there are two natural analogs, reported side by side:

  granularity   sample unit          feature space (nominal)
  --------------------------------------------------------------
  single        one timestep/token   raw 29  |  v3 16  |  v4 64      <- m5-matching
  chunk         one 16-step chunk    raw 464 |  v3 256 |  v4 1024    <- whole-chunk latent

Same math as m5: covariance eigenvalues -> participation ratio (effective dim) and
# PCs for 90/95/99 % variance, on per-dim z-scored (corr) features.

Reads the shared vsep cache (A, Z3, Z4). CPU only, no model load.
Outputs under analysis/output/visual_sep_gr1/:
    effdim_compare.json / .txt / effdim_compare_pr.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent


def eig_descending(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    w = np.linalg.eigvalsh(cov)
    return np.clip(w[::-1], 0.0, None)


def participation_ratio(eigs):
    s1, s2 = float(eigs.sum()), float((eigs ** 2).sum())
    return (s1 * s1) / s2 if s2 > 0 else 0.0


def n_pc_for(eigs, frac):
    total = eigs.sum()
    if total <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(eigs) / total, frac) + 1)


def zscore(X):
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - X.mean(axis=0, keepdims=True)) / sd


def analyze(X):
    """X: [n_samples, D] already in the chosen granularity. corr (z-scored) metrics."""
    X = X.astype(np.float64)
    D = X.shape[1]
    eigs = eig_descending(zscore(X))
    pr = participation_ratio(eigs)
    return {"nominal_dim": D, "n_samples": int(X.shape[0]),
            "PR_corr": pr, "redundancy_corr": D / pr if pr > 0 else float("inf"),
            "n_pc90_corr": n_pc_for(eigs, 0.90),
            "n_pc95_corr": n_pc_for(eigs, 0.95),
            "n_pc99_corr": n_pc_for(eigs, 0.99),
            "var_explained_corr": (np.cumsum(eigs) / eigs.sum()).tolist() if eigs.sum() > 0 else []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(_REPO_ROOT / "analysis" / "output" / "visual_sep_gr1" / "cache.npz"))
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    A, Z3, Z4 = d["A"], d["Z3"], d["Z4"]     # [N,16,29] [N,16,16] [N,16,64]
    N = A.shape[0]
    outdir = Path(args.cache).parent

    # single  = each timestep(raw) / token(latent) is a sample  -> m5-matching
    # chunk   = each 16-step chunk flattened is a sample
    res = {"N": int(N), "v3_ckpt": meta.get("v3_ckpt"), "v4_ckpt": meta.get("v4_ckpt"),
           "single": {
               "raw_action": analyze(A.reshape(-1, A.shape[-1])),   # [N*16, 29]
               "v3": analyze(Z3.reshape(-1, Z3.shape[-1])),         # [N*16, 16]
               "v4": analyze(Z4.reshape(-1, Z4.shape[-1]))},        # [N*16, 64]
           "chunk": {
               "raw_action": analyze(A.reshape(N, -1)),             # [N, 464]
               "v3": analyze(Z3.reshape(N, -1)),                    # [N, 256]
               "v4": analyze(Z4.reshape(N, -1))}}                   # [N, 1024]
    json.dump(res, open(outdir / "effdim_compare.json", "w"), indent=2)

    lines = ["=" * 100,
             f"Effective dim — m5-matching (single) vs chunk-flattened   (gr1 1000demo val, N={N})",
             f"  v3 ckpt: {res['v3_ckpt']}", f"  v4 ckpt: {res['v4_ckpt']}", "=" * 100,
             f"{'granularity':<12}{'space':<12}{'sample unit':<16}{'nominal':>8}{'#samples':>10}"
             f"{'PR':>8}{'red':>7}{'pc90':>6}{'pc95':>6}{'pc99':>6}", "-" * 100]
    units = {"single": {"raw_action": "timestep", "v3": "token", "v4": "token"},
             "chunk": {"raw_action": "16-step chunk", "v3": "chunk", "v4": "chunk"}}
    name = {"raw_action": "raw action", "v3": "v3 latent", "v4": "v4 latent"}
    for gran in ("single", "chunk"):
        for sp in ("raw_action", "v3", "v4"):
            r = res[gran][sp]
            lines.append(
                f"{gran:<12}{name[sp]:<12}{units[gran][sp]:<16}{r['nominal_dim']:>8}{r['n_samples']:>10}"
                f"{r['PR_corr']:>8.2f}{r['redundancy_corr']:>7.1f}"
                f"{r['n_pc90_corr']:>6}{r['n_pc95_corr']:>6}{r['n_pc99_corr']:>6}")
        lines.append("-" * 100)
    lines += [
        "single = m5's granularity (sample = one timestep/token; feature = per-step DoF).",
        "chunk  = whole 16-step chunk flattened (feature = 16 tokens x K).",
        "PR = participation ratio (effective dim);  red = nominal / PR;  pcNN = #PCs for NN% var (z-scored).",
        "NOTE: latent 'token' pooling treats the 16 chunk tokens as samples of one K-dim head,",
        "      the direct analog of m5 pooling the 16 timesteps as samples of one 29-dim action."]
    txt = "\n".join(lines)
    (outdir / "effdim_compare.txt").write_text(txt + "\n")
    print(txt)

    # bar chart: single (m5-matching) granularity, nominal vs effective for the 3 spaces
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, gran, ttl in ((axes[0], "single", "single action (m5-matching)"),
                          (axes[1], "chunk", "chunk-flattened latent")):
        spaces = [("raw_action", "raw\naction"), ("v3", "v3\nlatent"), ("v4", "v4\nlatent")]
        nominal = [res[gran][k]["nominal_dim"] for k, _ in spaces]
        eff = [res[gran][k]["PR_corr"] for k, _ in spaces]
        labels = [l for _k, l in spaces]
        x = np.arange(len(labels)); w = 0.38
        ymax = max(nominal) * 1.32
        b1 = ax.bar(x - w / 2, nominal, w, label="Nominal dim", color="#b9c6d6", edgecolor="#7f8c9b")
        b2 = ax.bar(x + w / 2, eff, w, label="Effective dim (PR)", color="#2f6db5")
        for b in b1:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax * 0.012, f"{b.get_height():.0f}",
                    ha="center", va="bottom", fontsize=8.5, color="#5b6672")
        for b in b2:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax * 0.012, f"{b.get_height():.1f}",
                    ha="center", va="bottom", fontsize=8.5, color="#2f6db5")
        for i, (n, e) in enumerate(zip(nominal, eff)):
            ax.annotate(f"{n/e:.1f}×", xy=(i, max(n, e)), xytext=(0, 18), textcoords="offset points",
                        ha="center", va="bottom", fontsize=11, fontweight="bold", color="#c0392b")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylabel("Dimensions", fontsize=10); ax.set_ylim(0, ymax)
        ax.set_title(ttl, fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("gr1 1000demo val — action vs tokenizer-latent effective dim", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = outdir / "effdim_compare_pr.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")
    print("#### EFFDIM COMPARE DONE ####")


if __name__ == "__main__":
    main()

"""Effective dimensionality of the v3 / v4 tokenizer LATENTS (gr1 1000demo val).

Same intrinsic-dim math as
  Isaac-GR00T/experiments/analysis/latent_vs_raw_dexterous/m5_intrinsic_dim_dexjoco_dual.py
but the "action group" is replaced by the tokenized latent itself:

    v3 latent  Z3  [N, 16 tokens, 16]  -> flatten [N, 256]   nominal dim = 256
    v4 latent  Z4  [N, 16 tokens, 64]  -> flatten [N, 1024]  nominal dim = 1024

For each latent we report the effective dimensionality of the point cloud:
    PR   = participation ratio  (Σλ)² / Σλ²   of the covariance eigenvalues
    pc90/95/99 = # principal components to reach 90/95/99 % of the variance

Preprocessing variants (like m5):
    corr   = per-dim z-score  (each latent coordinate standardized)  [PRIMARY]
    minmax = per-dim scale to [-1, 1]
    raw    = no scaling (covariance of the latent as-is; VAE μ already scaled)

Reads the shared vsep cache (Z3, Z4) — no model load, CPU only.
Outputs under analysis/output/visual_sep_gr1/:
    effdim_latent.json     full per-space metrics
    effdim_latent.txt      stdout summary
    effdim_latent_pr.png   nominal vs effective (PR) bars, v3 vs v4
    effdim_latent_pc95.png nominal vs effective (95% var) bars, v3 vs v4
    effdim_latent_spectrum.png  cumulative-variance curves
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


def minmax(X):
    lo, hi = X.min(0, keepdims=True), X.max(0, keepdims=True)
    rng = np.where((hi - lo) < 1e-8, 1.0, hi - lo)
    return (X - lo) / rng * 2.0 - 1.0


def analyze(Z):
    """Z: [N, ...] latent; flattened to [N, D]. Returns metrics for corr/minmax/raw."""
    X = Z.reshape(Z.shape[0], -1).astype(np.float64)
    D = X.shape[1]
    out = {"nominal_dim": D, "n_samples": int(X.shape[0])}
    variants = (("corr", zscore(X)), ("minmax", minmax(X)), ("raw", X))
    for tag, Xt in variants:
        eigs = eig_descending(Xt)
        pr = participation_ratio(eigs)
        out[f"PR_{tag}"] = pr
        out[f"redundancy_{tag}"] = D / pr if pr > 0 else float("inf")
        out[f"n_pc90_{tag}"] = n_pc_for(eigs, 0.90)
        out[f"n_pc95_{tag}"] = n_pc_for(eigs, 0.95)
        out[f"n_pc99_{tag}"] = n_pc_for(eigs, 0.99)
        out[f"var_explained_{tag}"] = (np.cumsum(eigs) / eigs.sum()).tolist() if eigs.sum() > 0 else []
    # per-dim raw std: reveals collapsed/dead latent coords (VAE prior collapse)
    sd = X.std(axis=0)
    out["per_dim_std_raw"] = sd.tolist()
    out["n_active_dims_1pct"] = int((sd > 0.01 * sd.max()).sum())  # dims with >1% of max std
    return out


def _bars(res, metric, out_png):
    spec = {"pr": ("PR_corr", "Effective dim (PR)", "{:.1f}"),
            "pc95": ("n_pc95_corr", "Effective dim (95% var)", "{:.0f}")}[metric]
    key, eff_label, fmt = spec
    groups = [("v3", "v3 (action-only)\n16×16=256"),
              ("v4", "v4 (DINO-fused)\n16×64=1024")]
    nominal = [res[k]["nominal_dim"] for k, _ in groups]
    eff = [res[k][key] for k, _ in groups]
    redund = [(n / e if e > 0 else float("inf")) for n, e in zip(nominal, eff)]
    labels = [l for _k, l in groups]

    x = np.arange(len(labels)); w = 0.38
    ymax = max(nominal) * 1.30
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    b1 = ax.bar(x - w / 2, nominal, w, label="Nominal dim", color="#b9c6d6", edgecolor="#7f8c9b")
    b2 = ax.bar(x + w / 2, eff, w, label=eff_label, color="#2f6db5")
    for b in b1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax * 0.012, f"{b.get_height():.0f}",
                ha="center", va="bottom", fontsize=9, color="#5b6672")
    for b in b2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax * 0.012, fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=9, color="#2f6db5")
    for i, r in enumerate(redund):
        top = max(nominal[i], eff[i])
        ax.annotate(f"{r:.1f}×", xy=(i, top), xytext=(0, 20), textcoords="offset points",
                    ha="center", va="bottom", fontsize=13, fontweight="bold", color="#c0392b")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Dimensions", fontsize=11); ax.set_ylim(0, ymax)
    ax.set_title("gr1 1000demo val — tokenizer latent effective dim", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    note = {"pr": "Effective dim = participation ratio of z-scored latent",
            "pc95": "Effective dim = # PCs for 95% of variance (z-scored latent)"}[metric]
    fig.text(0.5, 0.015, f"{note};   red × = Nominal / Effective (higher = more compressible)",
             ha="center", va="bottom", fontsize=7.5, style="italic", color="#7f8c9b")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"wrote {out_png}")


def _spectrum(res, out_png):
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for k, tag, col in (("v3", "v3 (action-only, 256)", "#1f77b4"),
                        ("v4", "v4 (DINO-fused, 1024)", "#d62728")):
        ve = np.asarray(res[k]["var_explained_corr"])
        ax.plot(np.arange(1, len(ve) + 1), ve, color=col, lw=1.8, label=tag)
        for frac, ls in ((0.90, ":"), (0.95, "--")):
            npc = res[k][f"n_pc{int(frac*100)}_corr"]
            ax.plot([npc], [ve[npc - 1]], "o", color=col, ms=5)
    for frac in (0.90, 0.95):
        ax.axhline(frac, color="0.7", lw=0.8, ls="--")
        ax.text(ax.get_xlim()[1], frac, f" {int(frac*100)}%", va="center", fontsize=8, color="0.5")
    ax.set_xlabel("# principal components", fontsize=11)
    ax.set_ylabel("cumulative variance explained", fontsize=11)
    ax.set_title("Latent variance spectrum (z-scored)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.grid(alpha=0.3); ax.set_ylim(0, 1.02)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(_REPO_ROOT / "analysis" / "output" / "visual_sep_gr1" / "cache.npz"))
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    Z3, Z4 = d["Z3"], d["Z4"]
    outdir = Path(args.cache).parent

    res = {"cache": str(args.cache),
           "v3_ckpt": meta.get("v3_ckpt"), "v4_ckpt": meta.get("v4_ckpt"),
           "N": int(Z3.shape[0]),
           "v3": analyze(Z3), "v4": analyze(Z4)}
    json.dump(res, open(outdir / "effdim_latent.json", "w"), indent=2)

    lines = []
    lines.append("=" * 96)
    lines.append(f"Effective dimensionality of tokenizer latents  (gr1 1000demo val, N={res['N']})")
    lines.append(f"  v3 ckpt: {res['v3_ckpt']}")
    lines.append(f"  v4 ckpt: {res['v4_ckpt']}")
    lines.append("=" * 96)
    hdr = (f"{'space':<6}{'nominal':>8}{'active':>8}"
           f"{'PR_corr':>9}{'red':>6}{'pc90':>6}{'pc95':>6}{'pc99':>6}"
           f"{'PR_raw':>9}{'pc95_raw':>9}")
    lines.append(hdr)
    lines.append("-" * 96)
    for k in ("v3", "v4"):
        r = res[k]
        lines.append(
            f"{k:<6}{r['nominal_dim']:>8}{r['n_active_dims_1pct']:>8}"
            f"{r['PR_corr']:>9.2f}{r['redundancy_corr']:>6.1f}"
            f"{r['n_pc90_corr']:>6}{r['n_pc95_corr']:>6}{r['n_pc99_corr']:>6}"
            f"{r['PR_raw']:>9.2f}{r['n_pc95_raw']:>9}")
    lines.append("-" * 96)
    lines.append("PR   = participation ratio (Σλ)²/Σλ² of covariance eigenvalues  (effective dim)")
    lines.append("red  = nominal / PR_corr  (redundancy; higher = more compressible)")
    lines.append("pcNN = # PCs for NN% variance (z-scored).  active = latent coords with std > 1% of max std")
    lines.append("corr = per-dim z-scored latent (primary);  raw = latent as-is (VAE μ / bottleneck).")
    txt = "\n".join(lines)
    (outdir / "effdim_latent.txt").write_text(txt + "\n")
    print(txt)

    _bars(res, "pr", outdir / "effdim_latent_pr.png")
    _bars(res, "pc95", outdir / "effdim_latent_pc95.png")
    _spectrum(res, outdir / "effdim_latent_spectrum.png")
    print(f"\nwrote JSON + charts under {outdir}")
    print("#### EFFDIM DONE ####")


if __name__ == "__main__":
    main()

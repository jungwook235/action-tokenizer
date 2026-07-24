"""③′  PARTIAL CORRELATION — does the latent track the visual AFTER controlling
for the action?  (real chunks only, no frame-swap, no OOD.)

The near-dup / spread figures are confounded: chunks that are "near-identical" in
action still differ a bit in action, and that residual action correlates with the
visual — so v3's latent also appears to move with the visual. The clean question:

    For random chunk PAIRS, is  Δlatent  correlated with  Δvisual  *once we condition
    on Δaction*?

  • v3 latent is a function of the action only ⇒ conditioned on Δaction, Δlatent
    carries no extra visual dependence ⇒ partial corr ≈ 0.
  • v4, if it encodes visual context ⇒ partial corr > 0.

We condition on Δaction NON-parametrically: bin pairs into Δaction quantiles and
z-score Δlatent / Δvisual WITHIN each bin, then correlate the pooled residuals.
We also report the correlation restricted to the near-duplicate-action pairs only
(Δaction in the smallest 1%), where the action is already ~constant.

Reads the shared cache (A, Z3, Z4, Vcontext). CPU only, no model load.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _std(X):
    return StandardScaler().fit_transform(X.reshape(X.shape[0], -1))


def partial_on_action(da, dv, dz, n_bins=30):
    """corr(dz, dv | da), conditioning on da by within-quantile-bin z-scoring."""
    qs = np.quantile(da, np.linspace(0, 1, n_bins + 1)); qs[-1] += 1e-9
    b = np.clip(np.digitize(da, qs) - 1, 0, n_bins - 1)
    zz = np.zeros_like(dz); vv = np.zeros_like(dv)
    for k in range(n_bins):
        m = b == k
        if m.sum() < 5:
            continue
        zz[m] = (dz[m] - dz[m].mean()) / (dz[m].std() + 1e-9)
        vv[m] = (dv[m] - dv[m].mean()) / (dv[m].std() + 1e-9)
    return float(np.corrcoef(zz, vv)[0, 1]), zz, vv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(_REPO_ROOT / "analysis" / "output" / "visual_sep_gr1" / "cache.npz"))
    ap.add_argument("--n-pairs", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    outdir = Path(args.cache).parent

    d = np.load(args.cache, allow_pickle=True)
    A, Z3, Z4, Vc = d["A"], d["Z3"], d["Z4"], d["Vcontext"]
    Xa, X3, X4, Xv = _std(A), _std(Z3), _std(Z4), _std(Vc)
    N = Xa.shape[0]

    rng = np.random.default_rng(args.seed)
    i = rng.integers(0, N, args.n_pairs); j = rng.integers(0, N, args.n_pairs); ok = i != j
    i, j = i[ok], j[ok]
    da = np.linalg.norm(Xa[i] - Xa[j], axis=1)
    dv = np.linalg.norm(Xv[i] - Xv[j], axis=1)
    d3 = np.linalg.norm(X3[i] - X3[j], axis=1)
    d4 = np.linalg.norm(X4[i] - X4[j], axis=1)

    # raw (uncontrolled) correlations
    r3_raw = float(np.corrcoef(d3, dv)[0, 1]); r4_raw = float(np.corrcoef(d4, dv)[0, 1])
    ra_v = float(np.corrcoef(da, dv)[0, 1])
    r3_a = float(np.corrcoef(d3, da)[0, 1]); r4_a = float(np.corrcoef(d4, da)[0, 1])
    # partial: control for action (also returns action-controlled residuals for plotting)
    p3, z3r, vvr = partial_on_action(da, dv, d3)
    p4, z4r, _ = partial_on_action(da, dv, d4)
    # near-dup-restricted: pairs with smallest 1% action distance
    thr = np.percentile(da, 1.0); nd = da <= thr
    c3_nd = float(np.corrcoef(d3[nd], dv[nd])[0, 1]); c4_nd = float(np.corrcoef(d4[nd], dv[nd])[0, 1])

    # figure: ACTION-CONTROLLED residuals — Δvisual(resid) vs Δlatent(resid); slope = partial corr
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    sub = rng.choice(len(da), min(8000, len(da)), replace=False)
    for a, zr, tag, col, pc in ((ax[0], z3r, "v3 (action-only)", "#1f77b4", p3),
                                (ax[1], z4r, "v4 (DINO-fused)", "#d62728", p4)):
        a.scatter(vvr[sub], zr[sub], s=5, alpha=0.18, color=col, linewidths=0)
        # trend line
        b1 = np.polyfit(vvr, zr, 1)
        xs = np.linspace(vvr.min(), vvr.max(), 50)
        a.plot(xs, np.polyval(b1, xs), color="k", lw=1.6)
        a.set_title(f"{tag}   partial r = {pc:.3f}", fontsize=12)
        a.set_xlabel("Δ visual  (action-controlled residual, z)")
        a.grid(alpha=0.2)
    ax[0].set_ylabel("Δ latent  (action-controlled residual, z)")
    fig.suptitle("③′  After removing Δaction: does Δlatent still track Δvisual?  "
                 "(real chunk pairs, no swap)\n"
                 f"v3 ≈ flat (latent = action code) · v4 tilts up (latent tracks visual beyond action)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    png = outdir / "partial_corr.png"
    fig.savefig(png, dpi=130); plt.close(fig)

    L = ["=" * 80, "③′  PARTIAL CORRELATION — latent vs visual, controlling for action (real pairs)",
         "=" * 80, f"N={N}  pairs={len(da)}  near-dup Δaction threshold (p1) = {thr:.2f}", "",
         "Reference correlations (uncontrolled):",
         f"  corr(Δaction, Δvisual)        = {ra_v:.3f}   (how coupled action & visual are)",
         f"  corr(Δlatent_v3, Δaction)     = {r3_a:.3f}",
         f"  corr(Δlatent_v4, Δaction)     = {r4_a:.3f}",
         f"  corr(Δlatent_v3, Δvisual) raw = {r3_raw:.3f}",
         f"  corr(Δlatent_v4, Δvisual) raw = {r4_raw:.3f}", "",
         "THE NUMBERS (visual dependence AFTER removing action):",
         f"  partial corr(Δlatent, Δvisual | Δaction):   v3 = {p3:.3f}    v4 = {p4:.3f}",
         f"  corr(Δlatent, Δvisual) on near-dup pairs :   v3 = {c3_nd:.3f}    v4 = {c4_nd:.3f}", "",
         "Read: v3 should be ~0 (latent is an action code; no visual dependence beyond action).",
         "      v4 > v3 (and > 0) ⇒ v4's latent tracks the visual EVEN after the action is",
         "      controlled — the clean, in-distribution, no-swap evidence.",
         f"figure -> {png}", "=" * 80]
    (outdir / "partial_corr.txt").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

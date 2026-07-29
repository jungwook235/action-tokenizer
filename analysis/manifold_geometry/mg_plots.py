"""Charts + markdown table for the manifold-geometry results.

Reads results/<embodiment>.json written by mg_run.py and produces, per
granularity:

  figs/<gran>_dimratio_vs_dof.png    Measurement 1 -- d/D vs nominal DoF, one
                                     series per estimator (PCA-95, PR, TwoNN)
  figs/<gran>_codimension.png        Measurement 1' -- D - d vs nominal DoF
  figs/<gran>_occupancy_vs_dof.png   Measurement 2 -- log eps-occupancy vs DoF
  figs/<gran>_occupancy_curves.png   Measurement 2 -- CDF(eps) log-log, all groups
  figs/<gran>_estimator_agreement.png  PCA/PR vs TwoNN scatter (robustness)
  tables/<gran>_summary.md / .csv

Colours encode embodiment, marker shape encodes dexterous vs arm vs other.
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from mg_embodiments import EMBODIMENTS, EMBODIMENT_ORDER  # noqa: E402

RESULT_DIR = os.path.join(_HERE, "results")
FIG_DIR = os.path.join(_HERE, "figs")
TAB_DIR = os.path.join(_HERE, "tables")

COLORS = {"bridge": "#8c8c8c", "robocasa_mg": "#5b8ff9", "gr1_tabletop": "#2f6db5",
          "gr1_unified_1000": "#7a3fa0", "dexjoco_single": "#e07b39",
          "dexjoco_dual": "#c0392b"}
MARKERS = {"dexterous": "o", "arm": "s", "other": "^"}

# null-calibration level used for the headline volume-deficit number
NULL_P = 0.01


def _role(emb_key, gname):
    spec = EMBODIMENTS[emb_key]
    if gname == spec.get("dexterous_group"):
        return "dexterous"
    if gname == spec.get("arm_group"):
        return "arm"
    return "other"


def load_rows(gran, ambient="gauss"):
    rows = []
    for key in EMBODIMENT_ORDER:
        fp = os.path.join(RESULT_DIR, f"{key}.json")
        if not os.path.exists(fp):
            continue
        blob = json.load(open(fp))
        groups = blob.get("granularity", {}).get(gran, {})
        for gname, r in groups.items():
            nn = r.get("nn", {}) or {}
            cd = r.get("corrdim", {}) or {}
            occ = (r.get("occupancy", {}) or {}).get(ambient, {}) or {}
            D = r["nominal_dim"]
            rows.append({
                "embodiment": key, "label": blob["label"], "group": gname,
                "role": _role(key, gname), "D": D,
                "D_eff_ambient": r.get("D_effective_ambient", D),
                "n_degenerate": r.get("n_degenerate_dims", 0),
                "n_samples": r.get("n_samples", 0),
                "PR": r.get("PR_corr", float("nan")),
                "pc90": r.get("n_pc90_corr", np.nan),
                "pc95": r.get("n_pc95_corr", np.nan),
                "pc99": r.get("n_pc99_corr", np.nan),
                "PR_minmax": r.get("PR_minmax", float("nan")),
                "twonn": nn.get("twonn_mean", float("nan")),
                "twonn_std": nn.get("twonn_std", float("nan")),
                "mle": nn.get("mle_mean", float("nan")),
                # mid-scale (cross-episode) window is the primary corr-dim; the
                # short-range one measures the trajectory curve, kept as diagnostic
                "corrdim": cd.get("id_mid", float("nan")),
                "corrdim_r2": cd.get("r2_mid", float("nan")),
                "corrdim_short": cd.get("id", float("nan")),
                "corrdim_short_r2": cd.get("r2", float("nan")),
                "corrdim_dupfrac": cd.get("dup_pair_frac", float("nan")),
                "local_slope_r": cd.get("local_slope_r", []),
                "local_slope": cd.get("local_slope", []),
                "tube_slope": occ.get("tube_slope", float("nan")),
                "tube_r2": occ.get("tube_slope_r2", float("nan")),
                "tail_slope": occ.get("tail_slope", float("nan")),
                "d_occ": occ.get("d_hat_from_tube", float("nan")),
                "occ_1x": occ.get("occ_at_1x_rmed", float("nan")),
                "occ_2x": occ.get("occ_at_2x_rmed", float("nan")),
                "occ_4x": occ.get("occ_at_4x_rmed", float("nan")),
                "occ_res": occ.get("occ_resolution", float("nan")),
                "r_med": occ.get("r_med_data_nn", float("nan")),
                "curve_eps": occ.get("curve_eps", []),
                "curve_eps_null": occ.get("curve_eps_null", []),
                "curve_cdf": occ.get("curve_cdf", []),
            })
            nc = (occ.get("null_calibrated") or {}).get(f"p{NULL_P:g}", {})
            rows[-1].update({
                "null_p": NULL_P,
                "occ_real_at_null": nc.get("occ_real", float("nan")),
                "null_censored": bool(nc.get("censored", False)),
                "deficit_log10": nc.get("volume_deficit_log10", float("nan")),
                "deficit_log10_bound": nc.get("volume_deficit_log10_bound",
                                              float("nan")),
            })
    return rows


def _style_ax(ax):
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _scatter_by_row(ax, rows, xkey, ykey, ykey_err=None):
    seen = set()
    for r in rows:
        c = COLORS[r["embodiment"]]
        m = MARKERS[r["role"]]
        lab = r["label"] if r["embodiment"] not in seen else None
        seen.add(r["embodiment"])
        y = r[ykey]
        if not np.isfinite(y):
            continue
        if ykey_err and np.isfinite(r.get(ykey_err, np.nan)):
            ax.errorbar(r[xkey], y, yerr=r[ykey_err], fmt=m, color=c, ms=7,
                        capsize=2, lw=1, label=lab, alpha=0.9)
        else:
            ax.plot(r[xkey], y, m, color=c, ms=7, label=lab, alpha=0.9)


def fig_dimratio(rows, gran, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    series = (("PR", "#2f6db5", "o", "PR / D"),
              ("pc95", "#7f8c9b", "x", "PCA-95% / D"),
              ("twonn", "#c0392b", "D", "TwoNN / D"),
              ("corrdim", "#2e8b57", "v", "corr-dim / D"))
    ax = axes[0]
    for r in rows:
        for key, col, mk, _ in series:
            v = r[key]
            if np.isfinite(v) and r["D"] > 0:
                ax.plot(r["D"], v / r["D"], mk, color=col, ms=6, alpha=0.75)
    for _key, col, mk, lab in series:
        ax.plot([], [], mk, color=col, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("nominal ambient dim  D")
    ax.set_ylabel("d / D   (dimension ratio)")
    ax.set_title(f"Measurement 1 — dimension ratio collapses with DoF ({gran})",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="0.7", lw=0.8, ls="--")
    ax.legend(frameon=False, fontsize=9)
    _style_ax(ax)

    ax = axes[1]
    _scatter_by_row(ax, rows, "D", "PR")
    lim = max(r["D"] for r in rows) * 1.1
    ax.plot([0, lim], [0, lim], "--", color="0.7", lw=0.9, label="d = D (full rank)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nominal ambient dim  D")
    ax.set_ylabel("effective dim  (PR)")
    ax.set_title("PR vs nominal — distance below the diagonal = redundancy",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style_ax(ax)

    fig.suptitle("marker: ● dexterous hand   ■ arm/eef   ▲ other   |   colour = embodiment",
                 fontsize=8.5, y=0.005, color="#7f8c9b")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_codimension(rows, gran, out):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for r in rows:
        if not np.isfinite(r["PR"]):
            continue
        ax.plot(r["D"], r["D"] - r["PR"], MARKERS[r["role"]],
                color=COLORS[r["embodiment"]], ms=7, alpha=0.9)
    xs = np.array(sorted({r["D"] for r in rows}), dtype=float)
    ax.plot(xs, xs, "--", color="0.7", lw=0.9, label="D − d = D  (d = 0)")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("nominal ambient dim  D")
    ax.set_ylabel("codimension  D − d   (d = PR)")
    ax.set_title(f"Measurement 1′ — codimension grows with DoF ({gran})",
                 fontsize=11, fontweight="bold")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[k],
                          label=EMBODIMENTS[k]["label"]) for k in EMBODIMENT_ORDER
               if any(r["embodiment"] == k for r in rows)]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_occupancy_vs_dof(rows, gran, out, ambient="gauss"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for r in rows:
        v = r["occ_real_at_null"]
        if not np.isfinite(v):
            continue
        censored = r.get("null_censored", False) or v <= 0
        y = r["occ_res"] if censored else v
        ax.plot(r["D"], y, MARKERS[r["role"]], color=COLORS[r["embodiment"]],
                ms=8, alpha=0.9, mfc="none" if censored else None)
        if censored:
            ax.annotate("", xy=(r["D"], y * 0.25), xytext=(r["D"], y),
                        arrowprops=dict(arrowstyle="->", color=COLORS[r["embodiment"]],
                                        lw=1))
    ax.axhline(NULL_P, color="#c0392b", lw=1.1, ls="--")
    ax.text(0.98, NULL_P * 1.35, f"null cloud = {NULL_P:g} by construction",
            transform=ax.get_yaxis_transform(), ha="right", fontsize=8,
            color="#c0392b")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nominal ambient dim  D")
    ax.set_ylabel("occupancy of the REAL cloud at the null-calibrated ε")
    ax.set_title(f"Measurement 2 — volume deficit vs a marginal-matched null\n"
                 f"({gran}, {ambient} ambient)", fontsize=11, fontweight="bold")
    ax.text(0.02, 0.03, "distance below the dashed line = orders of magnitude\n"
                        "less volume than a no-manifold cloud with identical marginals\n"
                        "hollow + arrow = 0 hits (1/M resolution limit)",
            transform=ax.transAxes, fontsize=7.5, color="#7f8c9b", va="bottom")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[k],
                          label=EMBODIMENTS[k]["label"]) for k in EMBODIMENT_ORDER
               if any(r["embodiment"] == k for r in rows)]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    _style_ax(ax)

    ax = axes[1]
    for r in rows:
        if np.isfinite(r["PR"]):
            ax.plot(r["D"], r["D"] - r["PR"], "_", color="0.55", ms=11)
        if np.isfinite(r["corrdim"]):
            ax.plot(r["D"], r["D"] - r["corrdim"], "v", color="#2e8b57", ms=6,
                    alpha=0.85)
        if np.isfinite(r["tube_slope"]):
            ax.plot(r["D"], r["tube_slope"], MARKERS[r["role"]],
                    color=COLORS[r["embodiment"]], ms=8, alpha=0.95)
    ax.plot([], [], "_", color="0.55", label="D − PR  (spectral codim)")
    ax.plot([], [], "v", color="#2e8b57", label="D − corr-dim  (nonlinear codim)")
    ax.plot([], [], "o", color="0.2", label="ε-tube slope  (where measurable)")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("nominal ambient dim  D")
    ax.set_ylabel("codimension estimate")
    ax.set_title("Codimension: three independent routes",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style_ax(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_occupancy_curves(rows, gran, out, ambient="gauss"):
    """Real vs null occupancy CDFs. The horizontal gap between a solid curve and
    its dotted twin at fixed occupancy is the volume deficit."""
    keep = [r for r in rows if r["role"] in ("dexterous", "arm")
            and len(r["curve_eps"]) > 5]
    if not keep:
        keep = [r for r in rows if len(r["curve_eps"]) > 5]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for r in keep:
        e = np.asarray(r["curve_eps"], dtype=float)
        e0 = np.asarray(r["curve_eps_null"], dtype=float)
        c = np.asarray(r["curve_cdf"], dtype=float)
        sc = np.sqrt(max(r["D_eff_ambient"], 1))
        ax.plot(e / sc, c, lw=1.6, alpha=0.9, color=COLORS[r["embodiment"]],
                ls="-" if r["role"] == "dexterous" else "--")
        if e0.size == c.size:
            ax.plot(e0 / sc, c, lw=1.0, alpha=0.5, color=COLORS[r["embodiment"]],
                    ls=":")
    ax.axhline(NULL_P, color="0.6", lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon\,/\,\sqrt{D}$   (ambient-scale normalised)")
    ax.set_ylabel(r"occupancy  $P(\mathrm{dist} < \varepsilon)$")
    ax.set_title(f"ε-occupancy CDF: real manifold vs marginal-matched null\n"
                 f"({gran}, {ambient} ambient) — rightward shift = thinner manifold",
                 fontsize=11, fontweight="bold")
    handles = [plt.Line2D([], [], color=COLORS[k], label=EMBODIMENTS[k]["label"])
               for k in EMBODIMENT_ORDER if any(r["embodiment"] == k for r in keep)]
    handles += [plt.Line2D([], [], color="0.4", ls=ls, label=lab)
                for ls, lab in (("-", "real, dexterous hand"), ("--", "real, arm/eef"),
                                (":", "null (shuffled columns)"))]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left", ncol=2)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_corrdim_scale(rows, gran, out):
    """Local correlation-dimension slope vs radius: robot actions are not
    scale-free, so the honest object is the whole curve, not one number."""
    keep = [r for r in rows if r["role"] in ("dexterous", "arm")
            and len(r.get("local_slope", [])) > 5]
    if not keep:
        return
    fig, ax = plt.subplots(figsize=(8, 5.4))
    for r in keep:
        rr = np.asarray(r["local_slope_r"], dtype=float)
        ss = np.asarray(r["local_slope"], dtype=float)
        sc = np.sqrt(max(r["D_eff_ambient"], 1))
        ax.plot(rr / sc, ss, lw=1.5, alpha=0.9, color=COLORS[r["embodiment"]],
                ls="-" if r["role"] == "dexterous" else "--")
        ax.plot([], [])
    for r in keep:
        if np.isfinite(r["PR"]):
            ax.plot(1.0, r["PR"], "_", color=COLORS[r["embodiment"]], ms=10,
                    alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"radius $r\,/\,\sqrt{D}$")
    ax.set_ylabel(r"local slope  $d\log C / d\log r$   (local intrinsic dim)")
    ax.set_title(f"Intrinsic dim is scale-dependent ({gran})\n"
                 "small r = trajectory curve, larger r = cross-episode manifold",
                 fontsize=11, fontweight="bold")
    handles = [plt.Line2D([], [], color=COLORS[k], label=EMBODIMENTS[k]["label"])
               for k in EMBODIMENT_ORDER if any(r["embodiment"] == k for r in keep)]
    handles += [plt.Line2D([], [], color="0.4", ls="-", label="dexterous hand"),
                plt.Line2D([], [], color="0.4", ls="--", label="arm/eef"),
                plt.Line2D([], [], color="0.4", ls="", marker="_",
                           label="PR (linear estimate)")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left", ncol=2)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_agreement(rows, gran, out):
    fig, ax = plt.subplots(figsize=(6.4, 6))
    for r in rows:
        if not (np.isfinite(r["PR"]) and np.isfinite(r["twonn"])):
            continue
        ax.errorbar(r["PR"], r["twonn"], yerr=r["twonn_std"], fmt=MARKERS[r["role"]],
                    color=COLORS[r["embodiment"]], ms=7, capsize=2, lw=1, alpha=0.9)
    vals = [v for r in rows for v in (r["PR"], r["twonn"]) if np.isfinite(v)]
    if vals:
        lim = [0, max(vals) * 1.1]
        ax.plot(lim, lim, "--", color="0.7", lw=0.9, label="TwoNN = PR")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    ax.set_xlabel("PR  (linear, z-scored covariance)")
    ax.set_ylabel("TwoNN  (nonlinear, NN-ratio)")
    ax.set_title(f"Estimator agreement ({gran})\nlinear and nonlinear estimators agreeing = robust low-dim claim",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def write_table(rows, gran, ambient):
    os.makedirs(TAB_DIR, exist_ok=True)
    cols = ["embodiment", "group", "role", "D", "D_eff_ambient", "n_samples",
            "PR", "pc90", "pc95", "pc99", "twonn", "twonn_std", "mle",
            "corrdim", "corrdim_r2", "corrdim_short", "tube_slope", "tube_r2",
            "tail_slope", "d_occ", "occ_1x", "occ_2x", "occ_4x", "occ_res",
            "r_med", "null_p", "occ_real_at_null", "deficit_log10"]
    csv = os.path.join(TAB_DIR, f"{gran}_summary.csv")
    with open(csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(
                f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                for c in cols) + "\n")

    md = os.path.join(TAB_DIR, f"{gran}_summary.md")
    with open(md, "w") as f:
        f.write(f"# Manifold geometry — {gran} granularity (ambient = {ambient})\n\n")
        f.write(
            "`d/D` uses PR. Intrinsic dim by four estimators: PR and PCA-k% "
            "(linear, upper bounds), TwoNN and corr-dim (nonlinear, computed after "
            "temporal decimation so they do not just measure the trajectory curve; "
            "corr-dim is the mid-scale window).\n\n"
            f"**Volume deficit** is the headline Measurement-2 number. ε is chosen "
            f"so a marginal-matched null cloud (each column permuted independently "
            f"— same per-dim distribution, no cross-DoF structure) is hit with "
            f"probability {NULL_P:g}; `occ_real @ null-1%` is what fraction of the "
            f"same ambient points land within that ε of the REAL data, and the "
            f"deficit is log10 of the ratio. −1 dex = the real manifold occupies "
            f"10× less volume than a structureless cloud with identical marginals. "
            f"`<1e-0X` / `<−X dex` are censored at the 1/M resolution limit, i.e. "
            f"lower bounds, not measured values.\n\n"
            "`tube slope` is the ε-occupancy log-log slope in the merged-tube "
            "regime (≈ codimension, a lower bound), `censored` where too few "
            "ambient points reach that regime. `tail slope` is the disjoint-ball "
            "diagnostic and should sit near D_eff — it is a health check on the "
            "estimator, not a result.\n\n")
        f.write("| embodiment | group | D | PR | d/D (PR) | pc90 | pc95 | pc99 | "
                "TwoNN | MLE | corr-dim | occ_real @ null-1% | volume deficit | "
                "tube slope | tail slope |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            occn = ("—" if not np.isfinite(r["occ_real_at_null"])
                    else f"<{r['occ_res']:.0e}" if r.get("null_censored")
                    else f"{r['occ_real_at_null']:.1e}")
            if np.isfinite(r["deficit_log10"]):
                dfc = f"{r['deficit_log10']:+.2f} dex"
            elif np.isfinite(r.get("deficit_log10_bound", np.nan)):
                dfc = f"<{r['deficit_log10_bound']:+.2f} dex"
            else:
                dfc = "—"
            tube = (f"{r['tube_slope']:.1f} (R²{r['tube_r2']:.2f})"
                    if np.isfinite(r["tube_slope"]) else "censored")
            tail = f"{r['tail_slope']:.1f}" if np.isfinite(r["tail_slope"]) else "—"
            f.write(f"| {r['embodiment']} | {r['group']} | {r['D']} | "
                    f"{r['PR']:.2f} | {r['PR'] / r['D']:.3f} | {r['pc90']} | "
                    f"{r['pc95']} | {r['pc99']} | "
                    f"{r['twonn']:.2f}±{r['twonn_std']:.2f} | {r['mle']:.2f} | "
                    f"{r['corrdim']:.2f} | {occn} | {dfc} | {tube} | {tail} |\n")
    print(f"wrote {csv}\nwrote {md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--granularity", nargs="+", default=["single", "chunk"])
    ap.add_argument("--ambient", default="gauss", choices=["gauss", "uniform"])
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    for gran in args.granularity:
        rows = load_rows(gran, ambient=args.ambient)
        if not rows:
            print(f"(no results for granularity={gran}, skip)")
            continue
        fig_dimratio(rows, gran, os.path.join(FIG_DIR, f"{gran}_dimratio_vs_dof.png"))
        fig_codimension(rows, gran, os.path.join(FIG_DIR, f"{gran}_codimension.png"))
        fig_occupancy_vs_dof(rows, gran,
                             os.path.join(FIG_DIR, f"{gran}_occupancy_vs_dof.png"),
                             ambient=args.ambient)
        fig_occupancy_curves(rows, gran,
                             os.path.join(FIG_DIR, f"{gran}_occupancy_curves.png"),
                             ambient=args.ambient)
        fig_corrdim_scale(rows, gran,
                          os.path.join(FIG_DIR, f"{gran}_corrdim_scale.png"))
        fig_agreement(rows, gran, os.path.join(FIG_DIR, f"{gran}_estimator_agreement.png"))
        write_table(rows, gran, args.ambient)
    print("#### MG PLOTS DONE ####")


if __name__ == "__main__":
    main()

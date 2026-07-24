"""Success-rate vs training-step curves.

Everything you normally tweak lives in the CONFIG block below:
  - DATA: benchmark -> method -> list of (step_in_1k, success_rate) points
  - METHOD_STYLE: color / marker / linestyle per method (order = legend order)
  - Axis labels, y-range, output path, figure size

Run:  python analysis/plot_success_rate.py
"""

import os
import re

import matplotlib.pyplot as plt

# ======================= CONFIG =======================

# benchmark name -> {method name -> [(step_k, success_rate), ...]}
# step is in units of 1k (e.g. 20 means 20k steps); success_rate in [0, 1]
DATA = {
    "GR00T N1.5 From-Scratch GR-1 Tabletop": {
        "Ours(1000)":     [(10, 23.5), (20, 32.5), (30, 35.5), (40, 41.2), (50, 43.8), (60, 44.9)],
        "Baseline(1000)": [(10, 6.7), (20, 9.3), (30, 17.6), (40, 23.9), (50, 29.6), (60, 33.3)],
        "Ours(100)":     [(60, 32.2)],
        "Baseline(100)": [(60, 23.8)],
    },
    "GR00T N1.5 Fine-tuned GR-1 Tabletop": {
        "Ours+(1000)":     [(10, 35.8), (20, 41.7), (30, 40.5), (40, 42.7), (50, 42.7), (60, 45.6)],
        "Ours(1000)":     [(60, 45.1)],
        "Baseline(1000)": [(60, 40.3)],
        "Ours(100)":     [(60, 42.3)],
        "Baseline(100)": [(60, 36.8)],
    },
    "GR00T N1.5 Fine-tuned GR-1 Tabletop 1000 demos": {
        "Ours":     [(60, 45.1)],
        "Baseline": [(60, 40.3)],
    },
    "GR00T N1.5 From-Scratch GR-1 Tabletop 1000 demos": {
        "Ours":     [(10, 23.5), (20, 32.5), (30, 35.5), (40, 41.2), (50, 43.8), (60, 44.9)],
        "Baseline": [(10, 6.7), (20, 9.3), (30, 17.6), (40, 23.9), (50, 29.6), (60, 33.3)],
    },
    "GR00T N1.5 Fine-tuned GR-1 Tabletop 100 demos": {
        "Ours":     [(60, 42.3)],
        "Baseline": [(60, 36.8)],
    },
    "GR00T N1.5 From-Scratch GR-1 Tabletop 100 demos": {
        "Ours":     [(60, 32.2)],
        "Baseline": [(60, 23.8)],
    },
    "Dit4Dit Video-frozen From-Scratch GR-1 Tabletop 1000 demos": {
        "Ours":     [(50, 47.7), (100, 50.4)],
        "Baseline": [(50, 39.2), (100, 50.4)],
    },
}

# method name -> style. Dict order decides legend order.
# Colors are a colorblind-safe categorical sequence — keep the order when adding methods.
METHOD_STYLE = {
    # plain names (single-setting benchmarks)
    "Baseline":       dict(color="#2a78d6", marker="o", linestyle="-"),
    "Ours":           dict(color="#e34948", marker="s", linestyle="-"),
    # demo-count variants: same hue per family, 1000 = solid/dark, 100 = dashed/light
    "Baseline(1000)": dict(color="#2a78d6", marker="o", linestyle="-"),
    "Baseline(100)":  dict(color="#86b6ef", marker="o", linestyle="--"),
    "Ours(1000)":     dict(color="#e34948", marker="s", linestyle="-"),
    "Ours(100)":      dict(color="#f0918f", marker="s", linestyle="--"),
    "Ours+(1000)":    dict(color="#4a3aa7", marker="^", linestyle="-"),
    # more methods: aqua "#1baf7a" (marker "D"), yellow "#eda100" (marker "v"),
    #               magenta "#e87ba4" (marker "P")
}

X_LABEL = "Training steps (×1k)"
Y_LABEL = "Success rate (%)"
Y_LIM = None                # e.g. (0, 100); None for auto

FIG_SIZE = (4.5, 3.5)       # width, height of each figure
OUTPUT_DIR = "analysis/success_rate_plots"   # one png + pdf per benchmark
DPI = 200

# ======================================================


def slugify(name):
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def plot_benchmark(benchmark, methods):
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for method, style in METHOD_STYLE.items():
        if method not in methods:
            continue
        points = sorted(methods[method])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(
            xs, ys,
            label=method,
            linewidth=2,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1,
            **style,
        )

    ax.set_title(benchmark, fontsize=10)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    if Y_LIM is not None:
        ax.set_ylim(*Y_LIM)

    # recessive chrome: light grid, no top/right spines
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")

    ax.legend(frameon=False)
    fig.tight_layout()

    base = os.path.join(OUTPUT_DIR, slugify(benchmark))
    fig.savefig(base + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {base}.png (+ .pdf)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for benchmark, methods in DATA.items():
        plot_benchmark(benchmark, methods)


if __name__ == "__main__":
    main()

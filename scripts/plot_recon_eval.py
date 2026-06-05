"""Compare tokenizer recon eval results across multiple runs.

Usage:
    python scripts/plot_recon_eval.py \\
        experiments/runs/recon_eval/gr1_v2_recon \\
        experiments/runs/recon_eval/gr1_v2_mask_recon \\
        experiments/runs/recon_eval/gr1_v2_state_pred_full_time \\
        experiments/runs/recon_eval/gr1_v2_state_pred_full_time_mask_statemask \\
        --output experiments/runs/recon_eval/comparison_gr1_v2.png

Each input path must contain a ``recon_eval.json`` produced by
``scripts/eval_tokenizer_recon.py``.

Plots:
  Row 1 (4 subplots): overall {norm,unnorm} L1 {mean,max}, one bar per run.
  Row 2 (2 subplots): per-key norm L1 mean / max, grouped bars per key.
  Row 3 (2 subplots): per-key unnorm L1 mean / max, grouped bars per key.

Each bar is annotated with the numeric value.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_eval_results(paths: List[str]) -> List[Tuple[str, dict]]:
    """Load ``recon_eval.json`` from each path. Returns ``[(label, data), ...]``."""
    results = []
    for p in paths:
        json_path = Path(p) / "recon_eval.json"
        if not json_path.exists():
            print(f"[WARN] {json_path} not found — skipping")
            continue
        with open(json_path) as f:
            data = json.load(f)
        label = Path(p).name
        results.append((label, data))
    return results


def aggregate_per_key(data: dict, metric_key: str) -> dict:
    """Reduce per-key per-dim list → single value per key.

    For ``*_mean`` metrics, average over dims. For ``*_max``, take max over dims.
    Matches the printed table format of ``eval_tokenizer_recon.py``.
    """
    is_max = metric_key.endswith("_max")
    out = {}
    for k, info in data["per_key"].items():
        vals = info.get(metric_key, [])
        if not vals:
            out[k] = 0.0
            continue
        out[k] = max(vals) if is_max else sum(vals) / len(vals)
    return out


def _annotate(ax, bars, fmt: str, fontsize: int = 7, rotation: int = 0):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
        )


def plot_overall(ax, results: List[Tuple[str, dict]], metric_key: str, title: str):
    labels = [r[0] for r in results]
    values = [r[1][metric_key] for r in results]
    cmap = plt.colormaps["tab10"]
    colors = [cmap(i % 10) for i in range(len(labels))]

    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("L1")
    ax.grid(axis="y", alpha=0.3)
    ax.margins(y=0.18)
    _annotate(ax, bars, "{:.5f}", fontsize=8)


def plot_per_key(ax, results: List[Tuple[str, dict]], metric_key: str, title: str):
    labels = [r[0] for r in results]
    first_keys = list(results[0][1]["per_key"].keys())
    n_keys = len(first_keys)
    n_runs = len(results)

    if n_runs == 0 or n_keys == 0:
        ax.set_visible(False)
        return

    width = 0.8 / n_runs
    x_base = np.arange(n_keys)
    cmap = plt.colormaps["tab10"]

    max_h = 0.0
    for i, (label, data) in enumerate(results):
        values_dict = aggregate_per_key(data, metric_key)
        values = [values_dict.get(k, 0.0) for k in first_keys]
        offsets = (i - (n_runs - 1) / 2.0) * width
        bars = ax.bar(
            x_base + offsets, values, width, label=label, color=cmap(i % 10)
        )
        max_h = max(max_h, max(values) if values else 0.0)
        _annotate(ax, bars, "{:.4f}", fontsize=6, rotation=45)

    ax.set_xticks(x_base)
    ax.set_xticklabels(
        [k.replace("action.", "") for k in first_keys],
        rotation=15,
        ha="right",
        fontsize=9,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("L1")
    ax.legend(fontsize=7, loc="upper left", ncol=min(2, n_runs))
    ax.grid(axis="y", alpha=0.3)
    ax.margins(y=0.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="Recon eval result directories (each containing recon_eval.json).",
    )
    parser.add_argument(
        "--output",
        default="recon_eval_comparison.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="Figure DPI."
    )
    parser.add_argument(
        "--title",
        default="Tokenizer Recon Eval Comparison",
        help="Top-level figure title.",
    )
    args = parser.parse_args()

    results = load_eval_results(args.paths)
    if not results:
        print("[ERROR] No valid results loaded — aborting.")
        return

    print(f"Loaded {len(results)} runs:")
    for label, data in results:
        print(
            f"  - {label} "
            f"(N={data.get('n_samples', '?')}, "
            f"D_norm={data.get('norm_action_dim', '?')}, "
            f"D_unnorm={data.get('unnorm_action_dim', '?')})"
        )

    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.32)

    # Row 1: overall metrics
    overall_specs = [
        ("overall_norm_l1_mean", "Overall norm L1 mean"),
        ("overall_norm_l1_max", "Overall norm L1 max"),
        ("overall_unnorm_l1_mean", "Overall unnorm L1 mean"),
        ("overall_unnorm_l1_max", "Overall unnorm L1 max"),
    ]
    for i, (key, title) in enumerate(overall_specs):
        ax = fig.add_subplot(gs[0, i])
        plot_overall(ax, results, key, title)

    # Row 2: per-key norm
    ax = fig.add_subplot(gs[1, :2])
    plot_per_key(ax, results, "norm_l1_mean", "Per-key norm L1 mean (avg over dims)")
    ax = fig.add_subplot(gs[1, 2:])
    plot_per_key(ax, results, "norm_l1_max", "Per-key norm L1 max (max over dims)")

    # Row 3: per-key unnorm
    ax = fig.add_subplot(gs[2, :2])
    plot_per_key(ax, results, "unnorm_l1_mean", "Per-key unnorm L1 mean (avg over dims)")
    ax = fig.add_subplot(gs[2, 2:])
    plot_per_key(ax, results, "unnorm_l1_max", "Per-key unnorm L1 max (max over dims)")

    fig.suptitle(args.title, fontsize=15, fontweight="bold", y=0.995)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison plot to {out_path}")


if __name__ == "__main__":
    main()


"""
python scripts/plot_recon_eval.py \
      experiments/runs/recon_eval/gr1_v2_recon \
      experiments/runs/recon_eval/gr1_v2_mask_recon \
      experiments/runs/recon_eval/gr1_v2_state_pred_full_time \
      experiments/runs/recon_eval/gr1_v2_state_pred_full_time_mask_statemask \
      --output experiments/runs/recon_eval/comparison_gr1_v2.png

python scripts/plot_recon_eval.py \
      experiments/runs/recon_eval/swx_v2_recon \
      experiments/runs/recon_eval/swx_v2_mask_recon \
      experiments/runs/recon_eval/swx_v2_state_pred_full_time \
      experiments/runs/recon_eval/swx_v2_state_pred_full_time_mask_statemask \
      --output experiments/runs/recon_eval/comparison_swx_v2.png

python scripts/plot_recon_eval.py \
      experiments/runs/recon_eval/robocasa_v2_mask_recon \
      experiments/runs/recon_eval/robocasa_v2_state_pred_full_time \
      experiments/runs/recon_eval/robocasa_v2_state_pred_full_time_mask_statemask \
      --output experiments/runs/recon_eval/comparison_robocasa_v2.png
"""
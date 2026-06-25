"""Side-by-side latent comparison of multiple tokenizers under ONE shared action
clustering (same #groups), so different tokenizers' latents are directly comparable.

For the SAME balanced sample of validation action chunks we:
  1. cluster the INPUT actions ONCE with KMeans(k)  → shared class labels,
  2. encode the chunks with EACH tokenizer → its latent z,
  3. t-SNE (and UMAP) of the action and of each tokenizer's latent, all colored by
     the SAME k action-class labels.

Output: one figure (rows = reducer, cols = [INPUT action] + one per tokenizer).
Because the actions/sample/seed are identical, the action panel and the coloring
are shared across tokenizers — only the latent layout differs.

Run from the action_tokenizer repo root, gr00t-actlat env, on a GPU.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gr00t.experiment.data_config_v3  # noqa: F401
from gr00t.data.dataset_action_frames_v4 import (  # noqa: E402
    ActionFramesCollatorV4, ActionFramesDatasetV4,
)
from gr00t.data.merge_norm_stats import apply_merged_normalization_metadata  # noqa: E402
from gr00t.model.action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper  # noqa: E402
from cluster_viz import tsne2d, umap2d, scatter  # noqa: E402

from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402


@torch.no_grad()
def sample_batches(args):
    """Return (actions[N,T,D], frame_x0[N,3,H,W], frame_x1, task_labels) for a
    balanced val sample shared across all tokenizers."""
    datasets, task_names = [], []
    for p in args.dataset_path:
        datasets.append(ActionFramesDatasetV4(
            dataset_path=p, data_config_name=args.data_config, embodiment_tag=args.embodiment_tag,
            split="val", val_ratio=args.val_ratio, val_seed=args.val_seed,
            normalization_mode=args.normalization_mode, image_size=args.image_size,
            video_backend=args.video_backend, use_fixed_val=True, fixed_val_path=args.fixed_val_path))
        task_names.append(Path(p).name)
    apply_merged_normalization_metadata(datasets, datasets)

    n_ds = len(datasets)
    per_ds = max(args.min_per_dataset, -(-args.target_total // n_ds))
    rng = np.random.default_rng(args.sample_seed)
    subsets, task_labels = [], []
    for ti, d in enumerate(datasets):
        n = len(d)
        k = min(per_ds, n)
        idx = rng.choice(n, size=k, replace=False)
        subsets.append(torch.utils.data.Subset(d, idx.tolist()))
        task_labels += [ti] * k
    concat = torch.utils.data.ConcatDataset(subsets)
    loader = torch.utils.data.DataLoader(concat, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, collate_fn=ActionFramesCollatorV4())
    A, X0, X1 = [], [], []
    for b in loader:
        A.append(b["action"]); X0.append(b["frame_x0"]); X1.append(b["frame_x1"])
    return (torch.cat(A), torch.cat(X0), torch.cat(X1), np.array(task_labels), task_names)


@torch.no_grad()
def encode_latent(ckpt, actions, x0, x1, device, batch=64):
    """Load a tokenizer and return its latent z [N, n_main, K] for the given chunks."""
    wrap = ActionLatentTokenizerWrapper.from_checkpoint(ckpt, device=device)
    wrap.eval()
    is_v4 = hasattr(wrap.tokenizer, "_is_v4") or hasattr(wrap.tokenizer, "_is_v5")
    outs = []
    N = actions.shape[0]
    for i in range(0, N, batch):
        a = actions[i:i + batch].to(device)
        if is_v4:
            g, t, h = wrap.encode(a, x0=x0[i:i + batch], x1=x1[i:i + batch])
        else:
            g, t, h = wrap.encode(a)
        outs.append(t.float().cpu())
    z = torch.cat(outs).numpy()
    K = z.shape[-1]
    del wrap
    torch.cuda.empty_cache()
    return z, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="output figure name")
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--dataset-path", nargs="+", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True, help="one short label per checkpoint")
    ap.add_argument("--k", type=int, default=10, help="shared #action clusters for coloring")
    ap.add_argument("--reducers", nargs="+", default=["tsne", "umap"])
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--normalization-mode", default="min_max")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--video-backend", default="decord")
    ap.add_argument("--val-ratio", type=float, default=0.003)
    ap.add_argument("--val-seed", type=int, default=42)
    ap.add_argument("--fixed-val-path", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--target-total", type=int, default=3000)
    ap.add_argument("--min-per-dataset", type=int, default=60)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    assert len(args.checkpoints) == len(args.labels)
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.sample_seed)

    outdir = _REPO_ROOT / "analysis" / "output" / "cluster"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[cmp] {args.tag}: sampling val chunks ...")
    actions, x0, x1, task_labels, task_names = sample_batches(args)
    N = actions.shape[0]
    print(f"[cmp] N={N} across {len(task_names)} tasks")

    # shared action clustering (fixed k)
    Xa = StandardScaler().fit_transform(actions.reshape(N, -1).numpy())
    labels = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit_predict(Xa)

    # latent for each tokenizer
    latents = []
    for ckpt, lab in zip(args.checkpoints, args.labels):
        print(f"[cmp] encoding {lab} ...")
        z, K = encode_latent(ckpt, actions, x0, x1, device, batch=args.batch_size)
        Xz = StandardScaler().fit_transform(z.reshape(N, -1))
        latents.append((f"{lab}  latent (K={K})", Xz))

    # embeddings
    panels_def = [("INPUT action", Xa)] + latents
    fns = {"tsne": tsne2d, "umap": umap2d}
    nrows = len(args.reducers)
    ncols = len(panels_def)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 4.0), squeeze=False)
    for r, red in enumerate(args.reducers):
        for c, (name, X) in enumerate(panels_def):
            print(f"[cmp] {red}: {name}")
            emb = fns[red](X, args.sample_seed)
            scatter(axes[r][c], emb, labels, f"{name}\n{red.upper()}  (color = action-cluster k={args.k})")
    fig.suptitle(f"{args.tag}  —  shared action clustering (k={args.k}), N={N}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = outdir / f"{args.tag}_compare_k{args.k}.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"[cmp] wrote {png}")


if __name__ == "__main__":
    main()

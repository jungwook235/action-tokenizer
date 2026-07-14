"""Collect a SHARED balanced validation sample and encode it with BOTH the
action-only (v3) and the DINO-fused (v4) tokenizer, so every downstream analysis
compares the two on the *exact same* action chunks / frames.

For each sampled val action chunk we store:
    A    = normalized input action chunk            [N, T, D]
    Z3   = v3 latent (action-only, K3)              [N, T, K3]
    Z4   = v4 latent mu (DINO-fused, K4)            [N, T, K4]   (mu, no VAE noise)
    V    = pooled DINO visual descriptor            [N, 2*C]     (mean f0 ‖ mean f1)
    Vd   = pooled DINO *dynamics* (f1-f0)           [N, C]
    task = source-dataset index                     [N]
plus, so the frame-thumbnail script can re-fetch the *identical* frames:
    samp_task[i], samp_local_idx[i]  → datasets[samp_task[i]][samp_local_idx[i]]

The v4 latent is the deterministic posterior mean ``mu`` (via _encode_mu_and_sample),
not a VAE sample, so the analysis is reproducible and free of sampling noise.

Everything is written to  analysis/output/visual_sep/cache.npz .

Run from the action_tokenizer repo root, gr00t-actlat env, on a GPU.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gr00t.experiment.data_config_v3  # noqa: F401,E402
from gr00t.data.dataset_action_frames_v4 import (  # noqa: E402
    ActionFramesCollatorV4, ActionFramesDatasetV4,
)
from gr00t.data.merge_norm_stats import apply_merged_normalization_metadata  # noqa: E402
from gr00t.model.action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper  # noqa: E402
from analyze_latents import _encode_mu_and_sample  # noqa: E402


# ----------------------------------------------------------------- sampling
def build_datasets(args):
    """Build the per-task val datasets (deterministic fixed-val) + merged norm."""
    datasets, task_names = [], []
    for p in args.dataset_path:
        datasets.append(ActionFramesDatasetV4(
            dataset_path=p, data_config_name=args.data_config,
            embodiment_tag=args.embodiment_tag, split="val",
            val_ratio=args.val_ratio, val_seed=args.val_seed,
            normalization_mode=args.normalization_mode, image_size=args.image_size,
            video_backend=args.video_backend, use_fixed_val=True,
            fixed_val_path=args.fixed_val_path))
        task_names.append(Path(p).name)
    apply_merged_normalization_metadata(datasets, datasets)
    return datasets, task_names


def sample_indices(datasets, args):
    """Return (samp_task[N], samp_local_idx[N]) — the shared balanced sample.

    Uses the SAME rng scheme as cluster_viz so the split/sample is reproducible.
    """
    n_ds = len(datasets)
    per_ds = max(args.min_per_dataset, -(-args.target_total // n_ds))
    rng = np.random.default_rng(args.sample_seed)
    samp_task, samp_local = [], []
    for ti, d in enumerate(datasets):
        n = len(d)
        k = min(per_ds, n)
        idx = rng.choice(n, size=k, replace=False)
        samp_task += [ti] * k
        samp_local += idx.tolist()
    return np.asarray(samp_task, dtype=np.int64), np.asarray(samp_local, dtype=np.int64)


def make_loader(datasets, samp_task, samp_local, args):
    subsets = []
    for ti, d in enumerate(datasets):
        loc = samp_local[samp_task == ti]
        subsets.append(torch.utils.data.Subset(d, loc.tolist()))
    concat = torch.utils.data.ConcatDataset(subsets)  # order == sorted-by-task == samp_* order
    return torch.utils.data.DataLoader(
        concat, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=ActionFramesCollatorV4())


# ----------------------------------------------------------------- encoding
@torch.no_grad()
def encode_v3(ckpt, loader, device):
    wrap = ActionLatentTokenizerWrapper.from_checkpoint(ckpt, device=device)
    wrap.eval()
    assert not wrap._is_v4(), "expected an action-only (v3) checkpoint for --v3-ckpt"
    dtype = wrap.tokenizer.encoder.action_proj.weight.dtype
    Z = []
    for b in loader:
        g, t, h = wrap.encode(b["action"].to(device).to(dtype))
        Z.append(t.float().cpu())
    z = torch.cat(Z).numpy()
    del wrap
    torch.cuda.empty_cache()
    return z


@torch.no_grad()
def encode_v4(ckpt, loader, device):
    """Return (A, Z4_mu, V_context[2C], Vd_dynamics[C]). A is collected here so
    the actions are guaranteed byte-identical to what v4 encoded."""
    wrap = ActionLatentTokenizerWrapper.from_checkpoint(ckpt, device=device)
    wrap.eval()
    assert wrap._is_v4(), "expected a DINO-fused (v4) checkpoint for --v4-ckpt"
    enc = wrap.tokenizer.encoder
    A, Z, Vc, Vd = [], [], [], []
    for b in loader:
        a = b["action"].to(device)
        f0, f1 = wrap._resolve_dino_feats(b["frame_x0"], b["frame_x1"], None, None, device)
        mu, _sig, _lv, _z = _encode_mu_and_sample(enc, a, f0, f1)  # deterministic mu
        A.append(a.float().cpu())
        Z.append(mu.float().cpu())
        f0m = f0.float().mean(dim=1).cpu()   # [B, C]
        f1m = f1.float().mean(dim=1).cpu()
        Vc.append(torch.cat([f0m, f1m], dim=1))         # [B, 2C] context
        Vd.append((f1.float() - f0.float()).mean(dim=1).cpu())  # [B, C] dynamics
    A = torch.cat(A).numpy()
    Z = torch.cat(Z).numpy()
    Vc = torch.cat(Vc).numpy()
    Vd = torch.cat(Vd).numpy()
    del wrap
    torch.cuda.empty_cache()
    return A, Z, Vc, Vd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-ckpt", required=True)
    ap.add_argument("--v4-ckpt", required=True)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--dataset-path", nargs="+", required=True)
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
    ap.add_argument("--out", default=str(_REPO_ROOT / "analysis" / "output" / "visual_sep" / "cache.npz"))
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("[collect] building val datasets ...")
    datasets, task_names = build_datasets(args)
    samp_task, samp_local = sample_indices(datasets, args)
    N = len(samp_task)
    print(f"[collect] N={N} across {len(task_names)} tasks: "
          + ", ".join(f"{i}={n}({int((samp_task==i).sum())})" for i, n in enumerate(task_names)))

    loader = make_loader(datasets, samp_task, samp_local, args)

    print("[collect] encoding v4 (+ actions + DINO visual) ...")
    A, Z4, Vc, Vd = encode_v4(args.v4_ckpt, loader, device)
    print(f"[collect]   A{A.shape}  Z4{Z4.shape}  Vcontext{Vc.shape}  Vdyn{Vd.shape}")

    # v3 needs an identically-ordered loader (fresh iterator, same order).
    loader3 = make_loader(datasets, samp_task, samp_local, args)
    print("[collect] encoding v3 (action-only) ...")
    Z3 = encode_v3(args.v3_ckpt, loader3, device)
    print(f"[collect]   Z3{Z3.shape}")
    assert Z3.shape[0] == N and Z4.shape[0] == N

    meta = dict(
        task_names=task_names, N=int(N),
        T=int(A.shape[1]), D=int(A.shape[2]), K3=int(Z3.shape[-1]), K4=int(Z4.shape[-1]),
        v3_ckpt=args.v3_ckpt, v4_ckpt=args.v4_ckpt,
        data_config=args.data_config, dataset_path=list(args.dataset_path),
        embodiment_tag=args.embodiment_tag, normalization_mode=args.normalization_mode,
        image_size=args.image_size, video_backend=args.video_backend,
        val_ratio=args.val_ratio, val_seed=args.val_seed, fixed_val_path=args.fixed_val_path,
        sample_seed=args.sample_seed, target_total=args.target_total,
        min_per_dataset=args.min_per_dataset,
    )
    np.savez_compressed(
        out, A=A, Z3=Z3, Z4=Z4, Vcontext=Vc, Vdyn=Vd,
        task=samp_task, samp_task=samp_task, samp_local=samp_local,
        meta=json.dumps(meta),
    )
    print(f"[collect] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

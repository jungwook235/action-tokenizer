"""Strided (stride-S) val collection for latent-distance vs image-distance analysis.

Motivation: if we sampled every action chunk, the "closest latent" pairs would be
dominated by temporally-adjacent chunks (they share almost all frames and an almost
identical action) — trivially close in BOTH latent and image. To measure whether
*latent* proximity carries *visual* information we instead take, per episode, chunks
at base_index stride S (default 20) so any near-latent pair is a genuinely different
moment, not a temporal neighbor.

For each strided chunk ("point") we store:
    A    = normalized action chunk           [N, T, D]
    Z3   = v3 latent (action-only)           [N, T, K3]
    Z4   = v4 latent mu (DINO-fused)         [N, T, K4]
    Vf0  = patch-mean DINO feature of x0     [N, C]   (chunk-start frame)
    Vf1  = patch-mean DINO feature of x1     [N, C]   (chunk-end frame)
    task, traj_id, base_index                [N]      (to mask same-episode pairs)
    samp_task, samp_local                    [N]      (to re-fetch identical frames)

Each point corresponds to TWO frames (x0, x1); the analysis defines image distance
as the mean of cosine(x0,x0') and cosine(x1,x1') on these DINO features.

Config / checkpoints are read from the existing val cache meta so this is directly
comparable to the vsep work (same v3/v4 ckpts, same fixed-val split).

Run from repo root, gr00t-actlat env, on a GPU.
Writes analysis/output/visual_sep_gr1/cache_stride{S}.npz.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gr00t.experiment.data_config_v3  # noqa: F401,E402
from vsep_collect import build_datasets, make_loader, encode_v3, encode_v4  # noqa: E402

VAL_CACHE = _REPO_ROOT / "analysis" / "output" / "visual_sep_gr1" / "cache.npz"
STRIDE = 20
OUT = _REPO_ROOT / "analysis" / "output" / "visual_sep_gr1" / f"cache_stride{STRIDE}.npz"


def strided_indices(datasets, stride):
    """Per (task, episode) take chunks at base_index stride S (0, S, 2S, ...)."""
    samp_task, samp_local, traj_ids, base_idx = [], [], [], []
    for ti, d in enumerate(datasets):
        assert hasattr(d, "all_steps"), "dataset missing all_steps (traj_id, base_index)"
        steps = np.asarray(d.all_steps)
        traj = steps[:, 0].astype(np.int64)
        base = steps[:, 1].astype(np.int64)
        for tj in np.unique(traj):
            g = np.where(traj == tj)[0]
            g = g[np.argsort(base[g], kind="stable")]  # order by step within episode
            sel = g[::stride]
            samp_task.extend([ti] * len(sel))
            samp_local.extend(sel.tolist())
            traj_ids.extend(traj[sel].tolist())
            base_idx.extend(base[sel].tolist())
    return (np.asarray(samp_task, np.int64), np.asarray(samp_local, np.int64),
            np.asarray(traj_ids, np.int64), np.asarray(base_idx, np.int64))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stride] device={device}")
    m = json.loads(str(np.load(VAL_CACHE, allow_pickle=True)["meta"]))
    args = SimpleNamespace(
        v3_ckpt=m["v3_ckpt"], v4_ckpt=m["v4_ckpt"], data_config=m["data_config"],
        dataset_path=m["dataset_path"], embodiment_tag=m["embodiment_tag"],
        normalization_mode=m["normalization_mode"], image_size=m["image_size"],
        video_backend=m["video_backend"], val_ratio=m["val_ratio"], val_seed=m["val_seed"],
        fixed_val_path=m["fixed_val_path"], batch_size=64, num_workers=8)

    print("[stride] building val datasets ...")
    datasets, task_names = build_datasets(args)
    samp_task, samp_local, traj_ids, base_idx = strided_indices(datasets, STRIDE)
    N = len(samp_task)
    n_ep = len(np.unique(np.stack([samp_task, traj_ids], 1), axis=0))
    print(f"[stride] stride={STRIDE}  N={N} chunks across {len(task_names)} tasks, "
          f"{n_ep} episodes  (mean {N/max(n_ep,1):.1f} chunks/episode)")

    loader = make_loader(datasets, samp_task, samp_local, args)
    print("[stride] encoding v4 (+actions +DINO) ...")
    A, Z4, Vc, Vd = encode_v4(args.v4_ckpt, loader, device)
    C = Vc.shape[1] // 2
    Vf0, Vf1 = Vc[:, :C].copy(), Vc[:, C:].copy()  # Vc = [meanf0 || meanf1]
    print(f"[stride]   A{A.shape} Z4{Z4.shape} Vf0{Vf0.shape} Vf1{Vf1.shape} (C={C})")

    loader3 = make_loader(datasets, samp_task, samp_local, args)
    print("[stride] encoding v3 ...")
    Z3 = encode_v3(args.v3_ckpt, loader3, device)
    print(f"[stride]   Z3{Z3.shape}")
    assert Z3.shape[0] == N == Z4.shape[0]

    meta = dict(
        stride=STRIDE, split="val", N=int(N), n_episodes=int(n_ep), task_names=task_names,
        T=int(A.shape[1]), D=int(A.shape[2]), K3=int(Z3.shape[-1]), K4=int(Z4.shape[-1]),
        C=int(C), v3_ckpt=args.v3_ckpt, v4_ckpt=args.v4_ckpt, data_config=args.data_config,
        dataset_path=list(args.dataset_path), embodiment_tag=args.embodiment_tag,
        normalization_mode=args.normalization_mode, image_size=args.image_size,
        video_backend=args.video_backend, val_ratio=args.val_ratio, val_seed=args.val_seed,
        fixed_val_path=args.fixed_val_path,
        note="strided(20) val; Vf0/Vf1 = patch-mean DINO of x0/x1; each point = 1 chunk = 2 frames")
    np.savez_compressed(
        OUT, A=A, Z3=Z3, Z4=Z4, Vf0=Vf0, Vf1=Vf1,
        task=samp_task, traj_id=traj_ids, base_index=base_idx,
        samp_task=samp_task, samp_local=samp_local, meta=json.dumps(meta))
    print(f"[stride] wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
    print("#### COLLECT STRIDE DONE ####")


if __name__ == "__main__":
    main()

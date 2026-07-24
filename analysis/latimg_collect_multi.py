"""Strided collection for latent-distance vs image-distance analysis (parametrized).

Same as latimg_collect.py but with configurable --stride and --outdir, so we can
build several strides (e.g. 5 and 20) into a dedicated output folder.

Per (task, episode) take chunks at base_index stride S. For each chunk store:
    A, Z3, Z4, Vf0, Vf1 (patch-mean DINO of x0/x1), task, traj_id, base_index.

Run from repo root, gr00t-actlat env, on a GPU.
Writes {outdir}/cache_stride{S}.npz.
"""

import argparse
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


def strided_indices(datasets, stride):
    samp_task, samp_local, traj_ids, base_idx = [], [], [], []
    for ti, d in enumerate(datasets):
        assert hasattr(d, "all_steps"), "dataset missing all_steps (traj_id, base_index)"
        steps = np.asarray(d.all_steps)
        traj = steps[:, 0].astype(np.int64)
        base = steps[:, 1].astype(np.int64)
        for tj in np.unique(traj):
            g = np.where(traj == tj)[0]
            g = g[np.argsort(base[g], kind="stable")]
            sel = g[::stride]
            samp_task.extend([ti] * len(sel))
            samp_local.extend(sel.tolist())
            traj_ids.extend(traj[sel].tolist())
            base_idx.extend(base[sel].tolist())
    return (np.asarray(samp_task, np.int64), np.asarray(samp_local, np.int64),
            np.asarray(traj_ids, np.int64), np.asarray(base_idx, np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    a = ap.parse_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"cache_stride{a.stride}.npz"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stride{a.stride}] device={device}")
    m = json.loads(str(np.load(VAL_CACHE, allow_pickle=True)["meta"]))
    args = SimpleNamespace(
        v3_ckpt=m["v3_ckpt"], v4_ckpt=m["v4_ckpt"], data_config=m["data_config"],
        dataset_path=m["dataset_path"], embodiment_tag=m["embodiment_tag"],
        normalization_mode=m["normalization_mode"], image_size=m["image_size"],
        video_backend=m["video_backend"], val_ratio=m["val_ratio"], val_seed=m["val_seed"],
        fixed_val_path=m["fixed_val_path"], batch_size=a.batch_size, num_workers=a.num_workers)

    print(f"[stride{a.stride}] building val datasets ...")
    datasets, task_names = build_datasets(args)
    samp_task, samp_local, traj_ids, base_idx = strided_indices(datasets, a.stride)
    N = len(samp_task)
    n_ep = len(np.unique(np.stack([samp_task, traj_ids], 1), axis=0))
    print(f"[stride{a.stride}] N={N} chunks, {n_ep} episodes (mean {N/max(n_ep,1):.1f}/ep)")

    loader = make_loader(datasets, samp_task, samp_local, args)
    print(f"[stride{a.stride}] encoding v4 ...")
    A, Z4, Vc, Vd = encode_v4(args.v4_ckpt, loader, device)
    C = Vc.shape[1] // 2
    Vf0, Vf1 = Vc[:, :C].copy(), Vc[:, C:].copy()
    print(f"[stride{a.stride}]   A{A.shape} Z4{Z4.shape} Vf0{Vf0.shape} Vf1{Vf1.shape} (C={C})")

    loader3 = make_loader(datasets, samp_task, samp_local, args)
    print(f"[stride{a.stride}] encoding v3 ...")
    Z3 = encode_v3(args.v3_ckpt, loader3, device)
    print(f"[stride{a.stride}]   Z3{Z3.shape}")
    assert Z3.shape[0] == N == Z4.shape[0]

    meta = dict(
        stride=a.stride, split="val", N=int(N), n_episodes=int(n_ep), task_names=task_names,
        T=int(A.shape[1]), D=int(A.shape[2]), K3=int(Z3.shape[-1]), K4=int(Z4.shape[-1]),
        C=int(C), v3_ckpt=args.v3_ckpt, v4_ckpt=args.v4_ckpt, data_config=args.data_config,
        dataset_path=list(args.dataset_path), note="Vf0/Vf1 = patch-mean DINO of x0/x1")
    np.savez_compressed(
        out, A=A, Z3=Z3, Z4=Z4, Vf0=Vf0, Vf1=Vf1,
        task=samp_task, traj_id=traj_ids, base_index=base_idx,
        samp_task=samp_task, samp_local=samp_local, meta=json.dumps(meta))
    print(f"[stride{a.stride}] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print(f"#### COLLECT STRIDE {a.stride} DONE ####")


if __name__ == "__main__":
    main()

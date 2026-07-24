"""Encode a balanced TRAIN-split sample with v3 + v4 (gr1), mirroring the val cache.

The vsep cache is the VAL split. To compute effective dim on TRAIN data we build the
TRAIN split (complement of the persistent fixed-val episodes) with the SAME config,
draw the SAME balanced per-task sample size, and encode with the SAME v3/v4 ckpts.

Reuses vsep_collect's encode/sample/loader helpers; only the dataset split differs
(split="train"). Config, dataset paths, and checkpoints are read from the existing
val cache meta so the two are directly comparable.

Run from repo root, gr00t-actlat env, on a GPU.
Writes analysis/output/visual_sep_gr1/cache_train.npz.
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
from gr00t.data.dataset_action_frames_v4 import ActionFramesDatasetV4  # noqa: E402
from gr00t.data.merge_norm_stats import apply_merged_normalization_metadata  # noqa: E402
from vsep_collect import sample_indices, make_loader, encode_v3, encode_v4  # noqa: E402

VAL_CACHE = _REPO_ROOT / "analysis" / "output" / "visual_sep_gr1" / "cache.npz"
OUT = _REPO_ROOT / "analysis" / "output" / "visual_sep_gr1" / "cache_train.npz"


def build_train_datasets(args):
    datasets, task_names = [], []
    for p in args.dataset_path:
        datasets.append(ActionFramesDatasetV4(
            dataset_path=p, data_config_name=args.data_config,
            embodiment_tag=args.embodiment_tag, split="train",
            val_ratio=args.val_ratio, val_seed=args.val_seed,
            normalization_mode=args.normalization_mode, image_size=args.image_size,
            video_backend=args.video_backend, use_fixed_val=True,
            fixed_val_path=args.fixed_val_path))
        task_names.append(Path(p).name)
    apply_merged_normalization_metadata(datasets, datasets)
    return datasets, task_names


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = json.loads(str(np.load(VAL_CACHE, allow_pickle=True)["meta"]))
    args = SimpleNamespace(
        v3_ckpt=m["v3_ckpt"], v4_ckpt=m["v4_ckpt"],
        data_config=m["data_config"], dataset_path=m["dataset_path"],
        embodiment_tag=m["embodiment_tag"], normalization_mode=m["normalization_mode"],
        image_size=m["image_size"], video_backend=m["video_backend"],
        val_ratio=m["val_ratio"], val_seed=m["val_seed"], fixed_val_path=m["fixed_val_path"],
        sample_seed=m["sample_seed"], target_total=m["target_total"],
        min_per_dataset=m["min_per_dataset"], batch_size=64, num_workers=8)

    print("[collect-train] building TRAIN datasets ...")
    datasets, task_names = build_train_datasets(args)
    samp_task, samp_local = sample_indices(datasets, args)
    N = len(samp_task)
    print(f"[collect-train] N={N} across {len(task_names)} tasks; "
          f"per-task={np.bincount(samp_task)[:5]}... (target/task="
          f"{max(args.min_per_dataset, -(-args.target_total//len(datasets)))})")

    loader = make_loader(datasets, samp_task, samp_local, args)
    print("[collect-train] encoding v4 (+actions +DINO) ...")
    A, Z4, Vc, Vd = encode_v4(args.v4_ckpt, loader, device)
    print(f"[collect-train]   A{A.shape} Z4{Z4.shape} Vc{Vc.shape} Vd{Vd.shape}")

    loader3 = make_loader(datasets, samp_task, samp_local, args)
    print("[collect-train] encoding v3 ...")
    Z3 = encode_v3(args.v3_ckpt, loader3, device)
    print(f"[collect-train]   Z3{Z3.shape}")
    assert Z3.shape[0] == N and Z4.shape[0] == N

    meta = dict(
        split="train", task_names=task_names, N=int(N),
        T=int(A.shape[1]), D=int(A.shape[2]), K3=int(Z3.shape[-1]), K4=int(Z4.shape[-1]),
        v3_ckpt=args.v3_ckpt, v4_ckpt=args.v4_ckpt,
        data_config=args.data_config, dataset_path=list(args.dataset_path),
        embodiment_tag=args.embodiment_tag, normalization_mode=args.normalization_mode,
        image_size=args.image_size, video_backend=args.video_backend,
        val_ratio=args.val_ratio, val_seed=args.val_seed, fixed_val_path=args.fixed_val_path,
        sample_seed=args.sample_seed, target_total=args.target_total,
        min_per_dataset=args.min_per_dataset)
    np.savez_compressed(OUT, A=A, Z3=Z3, Z4=Z4, Vcontext=Vc, Vdyn=Vd,
                        task=samp_task, samp_task=samp_task, samp_local=samp_local,
                        meta=json.dumps(meta))
    print(f"[collect-train] wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
    print("#### COLLECT TRAIN DONE ####")


if __name__ == "__main__":
    main()

"""Encode robocasa_gr1_tabletop/sim_100demos with the gr1_unified v3/v4 tokenizers.

Applies the SAME tokenizers used for the gr1 effective-dim work
(gr1_1000demos_v3_recon_ln_bn16 / v4_recon_dino_bn64_l1_mse_naiveln_vae, config
fourier_gr1_arms_waist) to a DIFFERENT dataset (robocasa_gr1_tabletop). This is an
OOD encoding — kept intentionally, per request, to reuse the existing tokenizer.

Uses split="all" (no train/val split, matching how the m5 GR1_TableTop chart was
built) and a bounded balanced sample. Saves latents + action only (no DINO visual)
to keep the cache small.

Run from repo root, gr00t-actlat env, on a GPU.
Writes analysis/output/visual_sep_gr1/cache_robocasa.npz.
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
OUT = _REPO_ROOT / "analysis" / "output" / "visual_sep_gr1" / "cache_robocasa.npz"
DATASET = "/storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos"
TARGET_TOTAL = 4000


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = json.loads(str(np.load(VAL_CACHE, allow_pickle=True)["meta"]))
    args = SimpleNamespace(
        v3_ckpt=m["v3_ckpt"], v4_ckpt=m["v4_ckpt"],
        data_config=m["data_config"], dataset_path=[DATASET],
        embodiment_tag=m["embodiment_tag"], normalization_mode=m["normalization_mode"],
        image_size=m["image_size"], video_backend=m["video_backend"],
        val_ratio=m["val_ratio"], val_seed=m["val_seed"], fixed_val_path=m["fixed_val_path"],
        sample_seed=0, target_total=TARGET_TOTAL, min_per_dataset=TARGET_TOTAL,
        batch_size=64, num_workers=8)

    print(f"[robocasa] building dataset (split=all) config={args.data_config} ...")
    d = ActionFramesDatasetV4(
        dataset_path=DATASET, data_config_name=args.data_config,
        embodiment_tag=args.embodiment_tag, split="all",
        val_ratio=args.val_ratio, val_seed=args.val_seed,
        normalization_mode=args.normalization_mode, image_size=args.image_size,
        video_backend=args.video_backend, use_fixed_val=False)
    datasets = [d]
    apply_merged_normalization_metadata(datasets, datasets)
    print(f"[robocasa] dataset len = {len(d)} chunks")

    # smoke: one sample -> action shape + frame keys, catch OOD/format mismatch early
    s = d[0]
    akey = "action" if "action" in s else [k for k in s if "action" in k][0]
    print(f"[robocasa] sample keys = {list(s.keys())}")
    a0 = np.asarray(s[akey])
    print(f"[robocasa] action shape = {a0.shape}  (tokenizer expects T={m['T']}, D={m['D']})")
    for fk in ("frame_x0", "frame_x1"):
        print(f"[robocasa]   {fk}: {'present' if fk in s else 'MISSING'}"
              + (f" shape={np.asarray(s[fk]).shape}" if fk in s else ""))

    samp_task, samp_local = sample_indices(datasets, args)
    N = len(samp_task)
    print(f"[robocasa] sampling N={N} (target={TARGET_TOTAL})")

    loader = make_loader(datasets, samp_task, samp_local, args)
    print("[robocasa] encoding v4 ...")
    A, Z4, Vc, Vd = encode_v4(args.v4_ckpt, loader, device)
    print(f"[robocasa]   A{A.shape} Z4{Z4.shape}")

    loader3 = make_loader(datasets, samp_task, samp_local, args)
    print("[robocasa] encoding v3 ...")
    Z3 = encode_v3(args.v3_ckpt, loader3, device)
    print(f"[robocasa]   Z3{Z3.shape}")
    assert Z3.shape[0] == N and Z4.shape[0] == N

    meta = dict(
        dataset=DATASET, split="all", ood=True, N=int(N),
        T=int(A.shape[1]), D=int(A.shape[2]), K3=int(Z3.shape[-1]), K4=int(Z4.shape[-1]),
        v3_ckpt=args.v3_ckpt, v4_ckpt=args.v4_ckpt, data_config=args.data_config,
        normalization_mode=args.normalization_mode, note="gr1_unified tokenizer applied OOD to robocasa_gr1_tabletop")
    # latents + action only (no DINO visual) to keep the cache small
    np.savez_compressed(OUT, A=A, Z3=Z3, Z4=Z4, meta=json.dumps(meta))
    print(f"[robocasa] wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
    print("#### COLLECT ROBOCASA DONE ####")


if __name__ == "__main__":
    main()

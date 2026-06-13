"""Diagnostic: how different are the two VGGT frame features (x0, x1)?

The V4 tokenizer feeds the encoder ``dino_diff = x1_feat - x0_feat`` and trains a
decoder to predict ``x1_feat`` from ``x0_feat`` + the action latent. If x0 and x1
features are nearly identical, the diff carries no signal and the prediction task is
trivially solved by the identity map ``x1 ≈ x0`` — i.e. learning is near-meaningless.

This script samples real (x0, x1) frame pairs from the dataset, extracts VGGT
features with the SAME settings as training (incl. --vggt-final-norm), and reports:

  1. Feature/diff magnitudes:   ||x0||, ||x1||, ||x1-x0||, relative change, cosine(x0,x1)
  2. The decisive baselines (the "is learning meaningful?" test):
       - identity MSE  = mean((x1-x0)^2)         → what the dino decoder must BEAT.
                          Compare to the logged ``loss_dino_mse``: if loss_dino_mse is
                          not clearly below this, the decoder basically learned identity.
       - random-pair   = same stats but x1 shuffled across the batch (unrelated frames)
                          → the scale of a "fully different" pair, for context.
  3. Degenerate samples: fraction where x0 and x1 are pixel-identical (short-episode
     clamping → exact-zero diff → dead training samples).

Per-sample scalars are aggregated so we report the DISTRIBUTION (p5/p50/p95), not just
the mean — temporal change is often bimodal (some chunks move a lot, some not at all).

Example:
  python scripts/diag_vggt_frame_diff.py \
      --dataset-path /NHNHOME/data/wook/dataset/robocasa_gr1_tabletop/sim_100demos \
      --data-config fourier_gr1_arms_waist \
      --embodiment-tag new_embodiment \
      --vggt-final-norm naive --num-samples 512 --batch-size 32
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gr00t.data.dataset_action_frames_v4 import (  # noqa: E402
    ActionFramesCollatorV4,
    ActionFramesDatasetV4,
)
from gr00t.utils.vggt_feature import VGGTFeatureExtractor  # noqa: E402


@dataclass
class Args:
    dataset_path: str
    """LeRobot dataset root (single path)."""
    data_config: str = "fourier_gr1_arms_waist"
    embodiment_tag: str = "new_embodiment"
    split: Literal["train", "val", "all"] = "train"

    # VGGT extractor — mirror the training script.
    vggt_model: str = "facebook/VGGT-1B"
    vggt_token_source: Literal["aggregator", "dpt_out2"] = "dpt_out2"
    vggt_image_size: int = 224
    vggt_final_norm: Literal["none", "naive"] = "naive"

    num_samples: int = 512
    batch_size: int = 32
    video_backend: str = "decord"
    num_workers: int = 8
    device: str = "cuda"
    seed: int = 0


def _pct(a: np.ndarray):
    return np.percentile(a, [5, 50, 95])


@torch.no_grad()
def main(args: Args):
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    dataset = ActionFramesDatasetV4(
        dataset_path=args.dataset_path,
        data_config_name=args.data_config,
        embodiment_tag=args.embodiment_tag,
        split=args.split,
        video_backend=args.video_backend,
        image_size=args.vggt_image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=ActionFramesCollatorV4(),
        drop_last=False,
    )

    extractor = VGGTFeatureExtractor(
        model_name=args.vggt_model,
        token_source=args.vggt_token_source,
        image_size=args.vggt_image_size,
        use_compile=False,
        final_norm=args.vggt_final_norm,
    ).to(device)
    extractor.eval()

    # Per-sample accumulators.
    cos_pair, rel_pair, mse_pair, l1_pair = [], [], [], []
    cos_rand, mse_rand = [], []
    rms0 = []
    n_identical = 0
    n_total = 0

    n_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        f0 = batch["frame_x0"].to(device).float() / 255.0  # [B,3,H,W]
        f1 = batch["frame_x1"].to(device).float() / 255.0
        B = f0.shape[0]
        n_total += B

        # Pixel-identical pairs (short-episode clamping → exact-zero diff).
        ident = (batch["frame_x0"] == batch["frame_x1"]).reshape(B, -1).all(dim=1)
        n_identical += int(ident.sum().item())

        x0, _ = extractor(f0)  # [B, Lp, C] fp32
        x1, _ = extractor(f1)
        x0 = x0.float()
        x1 = x1.float()

        diff = x1 - x0
        # Per-token norms, then average over tokens → per-sample scalar.
        n0 = x0.norm(dim=-1)  # [B, Lp]
        n1 = x1.norm(dim=-1)
        dn = diff.norm(dim=-1)
        denom = 0.5 * (n0 + n1) + 1e-8
        rel = (dn / denom).mean(dim=1)  # [B]
        cos = torch.cosine_similarity(x0, x1, dim=-1).mean(dim=1)  # [B]
        mse = (diff ** 2).mean(dim=(1, 2))  # [B]  == identity-baseline MSE per sample
        l1 = diff.abs().mean(dim=(1, 2))  # [B]

        cos_pair.append(cos.cpu().numpy())
        rel_pair.append(rel.cpu().numpy())
        mse_pair.append(mse.cpu().numpy())
        l1_pair.append(l1.cpu().numpy())
        rms0.append((x0 ** 2).mean(dim=(1, 2)).sqrt().cpu().numpy())

        # Random-pair baseline: x1 from a DIFFERENT sample (roll by 1).
        if B > 1:
            x1r = x1.roll(1, dims=0)
            diffr = x1r - x0
            cos_rand.append(
                torch.cosine_similarity(x0, x1r, dim=-1).mean(dim=1).cpu().numpy()
            )
            mse_rand.append((diffr ** 2).mean(dim=(1, 2)).cpu().numpy())

        print(f"  batch {bi + 1}/{n_batches} done (B={B})", flush=True)

    cos_pair = np.concatenate(cos_pair)
    rel_pair = np.concatenate(rel_pair)
    mse_pair = np.concatenate(mse_pair)
    l1_pair = np.concatenate(l1_pair)
    rms0 = np.concatenate(rms0)
    cos_rand = np.concatenate(cos_rand) if cos_rand else np.array([np.nan])
    mse_rand = np.concatenate(mse_rand) if mse_rand else np.array([np.nan])

    def line(name, a):
        p5, p50, p95 = _pct(a)
        print(f"  {name:<28} mean={a.mean():.4f}  std={a.std():.4f}  "
              f"p5={p5:.4f}  p50={p50:.4f}  p95={p95:.4f}")

    print("\n" + "=" * 78)
    print(f"VGGT frame-pair diagnostic  (source={args.vggt_token_source}, "
          f"final_norm={args.vggt_final_norm}, N={n_total} samples)")
    print("=" * 78)
    print(f"x0/x1 are obs index 0 and action_horizon-1 of each chunk (15 steps apart).")
    print(f"\n[degenerate] pixel-identical x0==x1 pairs: "
          f"{n_identical}/{n_total} ({100.0 * n_identical / max(1, n_total):.1f}%)")

    print("\n[1] magnitudes (per-sample, averaged over tokens)")
    line("feature RMS ||x0||/sqrt(C)", rms0)
    line("cosine(x0, x1)", cos_pair)
    line("relative change |x1-x0|/|x|", rel_pair)

    print("\n[2] identity-baseline reconstruction error  (decoder must beat this)")
    line("identity MSE = mean((x1-x0)^2)", mse_pair)
    line("identity L1  = mean|x1-x0|", l1_pair)
    print("    → compare 'identity MSE' to the logged loss_dino_mse. If loss_dino_mse")
    print("      is not clearly below it, the decoder mostly learned identity (trivial).")

    print("\n[3] random-pair baseline  (x1 from an unrelated sample — the 'fully different' scale)")
    line("cosine(x0, x1_random)", cos_rand)
    line("random-pair MSE", mse_rand)
    print("    → if temporal MSE (sec.2) << random MSE, frames share most structure and")
    print("      only a small temporal delta is learnable. If they're close, x0/x1 differ")
    print("      about as much as unrelated frames (strong signal).")
    print("=" * 78)


if __name__ == "__main__":
    main(tyro.cli(Args))



"""

PYTHONUNBUFFERED=1 python scripts/diag_vggt_frame_diff.py \
    --dataset-path /NHNHOME/data/wook/dataset/robocasa_gr1_tabletop/sim_100demos \
    --data-config fourier_gr1_arms_waist \
    --embodiment-tag new_embodiment \
    --vggt-final-norm naive \
    --num-samples 512 --batch-size 32
"""
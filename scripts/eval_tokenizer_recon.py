"""Tokenizer reconstruction evaluation on a fixed validation episode set.

Loads a trained ActionLatentTokenizer checkpoint (v1 / v2 / v3 / dimwise auto-detected)
and measures encode→decode L1 against the *normalized* and *unnormalized*
ground-truth actions on the validation split — using the same JSON-pinned
fixed val episodes as v3 training.

Outputs ``recon_eval.json`` with:
  - per_dim_norm_l1_mean / per_dim_norm_l1_max          (length = D_norm)
  - per_dim_unnorm_l1_mean / per_dim_unnorm_l1_max      (length = D_unnorm)
  - overall_*                                            (scalars)
  - per_key                                              (dict keyed by action.<sub>)
  - meta (checkpoint, dataset, data_config, ...)

Usage:
    python scripts/eval_tokenizer_recon.py \\
        --checkpoint-path checkpoints_action_tokenizer/.../checkpoint-100000 \\
        --dataset-path /path/to/dataset \\
        --data-config fourier_gr1_arms_waist \\
        --embodiment-tag new_embodiment \\
        --fixed-val-path /sjw_alinlab1/home/jungwook/Isaac-GR00T/experiments/fixed_val_splits/gr1_100demos.json \\
        --output-dir experiments/runs/recon_eval/<tag>
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import torch
import tyro
from torch.utils.data import DataLoader
from tqdm import tqdm

# Side-effect: register V3 q99 data configs (fourier_gr1_arms_waist_q99 등) into DATA_CONFIG_MAP
import gr00t.experiment.data_config_v3  # noqa: F401
from gr00t.data.dataset_action_only import ActionOnlyCollator
from gr00t.data.dataset_action_only_v3 import ActionOnlyDatasetV3
from gr00t.data.transform.concat import ConcatTransform
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.action_latent_tokenizer_wrapper import ActionLatentTokenizerWrapper


@dataclass
class ArgsConfig:
    # ── Required ──
    checkpoint_path: str
    """Tokenizer checkpoint directory (HF Trainer ``checkpoint-XXXX``) or single ``.pt`` file."""

    dataset_path: str
    """LeRobot-format dataset root."""

    data_config: str
    """Key in ``DATA_CONFIG_MAP`` (e.g. ``fourier_gr1_arms_waist``, ``single_panda_gripper``,
    ``bridge_flare_kty_actlat_fm``, ``fourier_gr1_arms_waist_q99``)."""

    output_dir: str
    """Where to write ``recon_eval.json``."""

    # ── Dataset / split ──
    embodiment_tag: str = "new_embodiment"
    normalization_mode: str = "min_max"
    """Fallback for keys NOT covered by ``data_config.action_normalization_modes``."""

    split: Literal["val", "train", "all"] = "val"
    val_ratio: float = 0.003
    val_seed: int = 42
    use_fixed_val: bool = True
    fixed_val_path: Optional[str] = None
    """Absolute path to the persisted JSON. None → ``<dataset>/meta/fixed_val_split.json``.
    Auto-created on first run if missing."""

    # ── Eval config ──
    target_tokens: Literal["all", "time", "global_time", "time_hand"] = "all"
    """Which token subset to round-trip through the tokenizer.
    ``all`` measures pure AE recon; ``time`` matches what the VLA actually sees."""

    batch_size: int = 256
    device: str = "cuda"
    num_workers: int = 8

    # ── Precision (match training-time eval) ──
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    """Forward/loss precision.
    Training uses ``bf16=True`` (HF Trainer mixed precision) + ``tf32=True``;
    pass ``--precision bf16`` to mirror that and reproduce wandb ``eval/recon_l1``.
    ``fp32`` (default) gives the most accurate L1 number."""

    tf32: bool = True
    """Allow TF32 matmul on Ampere+. Training had this on (training_args.tf32=True)."""

    # ── Output ──
    save_per_sample: bool = False
    """If True, also dump per-sample, per-step, per-dim abs errors as torch tensors
    (heavy — only enable for small val sets when debugging)."""


def _build_dataset(cfg: ArgsConfig) -> ActionOnlyDatasetV3:
    return ActionOnlyDatasetV3(
        dataset_path=cfg.dataset_path,
        data_config_name=cfg.data_config,
        embodiment_tag=cfg.embodiment_tag,
        split=cfg.split,
        val_ratio=cfg.val_ratio,
        val_seed=cfg.val_seed,
        normalization_mode=cfg.normalization_mode,
        video_backend="torchvision_av",  # unused by ActionOnly path
        use_fixed_val=cfg.use_fixed_val,
        fixed_val_path=cfg.fixed_val_path,
    )


def _get_concat_transform(ds: ActionOnlyDatasetV3) -> ConcatTransform:
    for t in ds.transforms.transforms:
        if isinstance(t, ConcatTransform):
            return t
    raise RuntimeError("ConcatTransform not found in dataset.transforms — pipeline mismatch.")


def _action_keys(ds: ActionOnlyDatasetV3) -> List[str]:
    """Public action_keys: prefer attr, fall back to data_config."""
    keys = getattr(ds, "_action_keys", None)
    if keys is not None:
        return list(keys)
    cfg = DATA_CONFIG_MAP[ds._data_config_name]  # type: ignore[attr-defined]
    return list(cfg.modality_config()["action"].modality_keys)


def _unnormalize_action(
    action_norm: torch.Tensor,
    transforms,
    action_keys: List[str],
) -> torch.Tensor:
    """Run ``transforms.unapply`` on a normalized action tensor.

    Args:
        action_norm: ``[B, T, D_norm]`` torch tensor on CPU.
        transforms: ``ComposedModalityTransform`` from the source dataset.
        action_keys: order of action.* keys — used to re-concat per-key outputs.

    Returns:
        ``[B, T, D_unnorm]`` torch tensor in original action units (rotation
        un-rotated, normalizer inverted). Note ``D_unnorm`` may differ from
        ``D_norm`` when rotation transforms are present (e.g. rotation_6d → axis_angle).
    """
    # transforms.unapply mutates the input dict (ConcatTransform.unapply pops "action").
    data = {"action": action_norm.detach().cpu()}
    out = transforms.unapply(data)
    parts = []
    for k in action_keys:
        v = out[k]
        # transforms may have cast to numpy if a StateActionToTensor unapply set numpy dtype.
        # Force back to torch for arithmetic.
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v)
        parts.append(v.float())
    return torch.cat(parts, dim=-1)


def _per_key_breakdown(
    per_dim: torch.Tensor,
    action_keys: List[str],
    dim_per_key: dict,
) -> dict:
    """Slice a per-dim metric (length D) into per-action-key chunks.

    Args:
        per_dim: ``[D]`` tensor.
        action_keys: ordered list of keys.
        dim_per_key: {key: int} dim count to slice.
    """
    out = {}
    cursor = 0
    for k in action_keys:
        d = dim_per_key.get(k, 0)
        if d == 0:
            out[k] = []
            continue
        out[k] = per_dim[cursor : cursor + d].tolist()
        cursor += d
    return out


def main(cfg: ArgsConfig) -> None:
    print("=" * 60)
    print("TOKENIZER RECON EVAL CONFIG:")
    print("=" * 60)
    for k, v in vars(cfg).items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TF32 toggle (training had it on).
    if cfg.tf32 and cfg.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    autocast_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[cfg.precision]
    print(f"[eval] precision={cfg.precision} (autocast dtype={autocast_dtype}), tf32={cfg.tf32}")

    # 1. Build dataset on the fixed val split.
    ds = _build_dataset(cfg)
    n = len(ds)
    assert n > 0, f"No samples in {cfg.split} split for {cfg.dataset_path} — check fixed-val path."
    print(f"[eval] {cfg.split} split sample count: {n:,}")

    # Action key ordering and per-key dims (post-rotation, post-normalize = what tokenizer sees).
    action_keys = _action_keys(ds)
    concat_tx = _get_concat_transform(ds)
    norm_dim_per_key = dict(concat_tx.action_dims)  # {"action.left_arm": 7, ...}

    # Original (pre-rotation, pre-normalize) per-key dims for unnorm slicing.
    # ConcatTransform.set_metadata stores post-rotation dims in self.action_dims; we need
    # the *original* shape for un-rotated keys.
    unnorm_dim_per_key = {}
    metadata = concat_tx.dataset_metadata
    for k in action_keys:
        modality, sub = k.split(".")
        modality_meta = getattr(metadata.modalities, modality)[sub]
        # shape is e.g. [3] or [4] (for quaternion). After unapply the rotation transform
        # restores this dim count.
        unnorm_dim_per_key[k] = int(modality_meta.shape[0])

    print(f"[eval] action_keys: {action_keys}")
    print(f"[eval] norm   dims per key: {norm_dim_per_key}")
    print(f"[eval] unnorm dims per key: {unnorm_dim_per_key}")

    # 2. Load tokenizer.
    wrapper = ActionLatentTokenizerWrapper.from_checkpoint(
        cfg.checkpoint_path, device=cfg.device
    )
    wrapper.eval()
    print(
        f"[eval] tokenizer loaded — type={wrapper.tokenizer_type}, "
        f"action_dim={wrapper.action_dim}, action_horizon={wrapper.action_horizon}"
    )

    # 3. DataLoader.
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=ActionOnlyCollator(),
        persistent_workers=cfg.num_workers > 0,
        pin_memory=cfg.device.startswith("cuda"),
    )

    # 4. Iterate, encode→decode, compute abs errors.
    norm_abs_chunks = []
    unnorm_abs_chunks = []
    n_seen = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[eval] recon"):
            gt_norm = batch["action"].to(cfg.device)  # [B, T, D_norm]

            # Mirror training's mixed-precision eval. Both pred and gt are cast to
            # the autocast dtype so |pred - gt| matches what F.l1_loss computed
            # inside the trainer.
            if autocast_dtype is not None and cfg.device.startswith("cuda"):
                gt_for_loss = gt_norm.to(dtype=autocast_dtype)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    if cfg.target_tokens == "all":
                        g, m, h = wrapper.encode(gt_for_loss)
                        pred_norm = wrapper.tokenizer.decode(g, m, h)
                    else:
                        lat = wrapper.get_latent_target(gt_for_loss, target_tokens=cfg.target_tokens)
                        pred_norm = wrapper.decode_latent(lat, target_tokens=cfg.target_tokens)
                pred_norm = pred_norm.to(dtype=autocast_dtype)
                norm_err = (pred_norm - gt_for_loss).abs().detach().cpu().float()
            else:
                if cfg.target_tokens == "all":
                    g, m, h = wrapper.encode(gt_norm)
                    pred_norm = wrapper.tokenizer.decode(g, m, h)
                else:
                    lat = wrapper.get_latent_target(gt_norm, target_tokens=cfg.target_tokens)
                    pred_norm = wrapper.decode_latent(lat, target_tokens=cfg.target_tokens)
                pred_norm = pred_norm.to(dtype=gt_norm.dtype)
                norm_err = (pred_norm - gt_norm).abs().detach().cpu().float()  # [B, T, D_norm]

            # Unnormalize ground-truth and prediction independently.
            # Always pass fp32 to the inverse transform (Normalizer.inverse uses
            # the original fp32 actions, so we keep the unnorm side honest).
            gt_un = _unnormalize_action(gt_norm.detach().cpu().float(), ds.transforms, action_keys)
            pred_un = _unnormalize_action(pred_norm.detach().cpu().float(), ds.transforms, action_keys)
            unnorm_err = (pred_un - gt_un).abs().float()  # [B, T, D_unnorm]

            norm_abs_chunks.append(norm_err)
            unnorm_abs_chunks.append(unnorm_err)
            n_seen += gt_norm.shape[0]

    norm_abs = torch.cat(norm_abs_chunks, dim=0)  # [N, T, D_norm]
    unnorm_abs = torch.cat(unnorm_abs_chunks, dim=0)  # [N, T, D_unnorm]

    # 5. Aggregate.
    per_dim_norm_mean = norm_abs.mean(dim=(0, 1))
    per_dim_norm_max = norm_abs.amax(dim=(0, 1))
    per_dim_unnorm_mean = unnorm_abs.mean(dim=(0, 1))
    per_dim_unnorm_max = unnorm_abs.amax(dim=(0, 1))

    metrics = {
        "n_samples": n_seen,
        "action_horizon": int(norm_abs.shape[1]),
        "norm_action_dim": int(norm_abs.shape[-1]),
        "unnorm_action_dim": int(unnorm_abs.shape[-1]),
        "per_dim_norm_l1_mean": per_dim_norm_mean.tolist(),
        "per_dim_norm_l1_max": per_dim_norm_max.tolist(),
        "per_dim_unnorm_l1_mean": per_dim_unnorm_mean.tolist(),
        "per_dim_unnorm_l1_max": per_dim_unnorm_max.tolist(),
        "overall_norm_l1_mean": float(per_dim_norm_mean.mean()),
        "overall_norm_l1_max": float(per_dim_norm_max.max()),
        "overall_unnorm_l1_mean": float(per_dim_unnorm_mean.mean()),
        "overall_unnorm_l1_max": float(per_dim_unnorm_max.max()),
    }

    # 6. Per-key breakdown (mean/max over each key's dims).
    # _per_key_breakdown maintains a cumulative cursor — call ONCE with the full
    # action_keys list per metric so each key gets its correct slice.
    norm_mean_chunks = _per_key_breakdown(per_dim_norm_mean, action_keys, norm_dim_per_key)
    norm_max_chunks = _per_key_breakdown(per_dim_norm_max, action_keys, norm_dim_per_key)
    unnorm_mean_chunks = _per_key_breakdown(per_dim_unnorm_mean, action_keys, unnorm_dim_per_key)
    unnorm_max_chunks = _per_key_breakdown(per_dim_unnorm_max, action_keys, unnorm_dim_per_key)

    per_key = {}
    for k in action_keys:
        per_key[k] = {
            "norm_dims": norm_dim_per_key.get(k, 0),
            "unnorm_dims": unnorm_dim_per_key.get(k, 0),
            "norm_l1_mean": norm_mean_chunks[k],
            "norm_l1_max": norm_max_chunks[k],
            "unnorm_l1_mean": unnorm_mean_chunks[k],
            "unnorm_l1_max": unnorm_max_chunks[k],
        }
    metrics["per_key"] = per_key

    meta = {
        "checkpoint_path": str(Path(cfg.checkpoint_path).resolve()),
        "dataset_path": str(Path(cfg.dataset_path).resolve()),
        "data_config": cfg.data_config,
        "embodiment_tag": cfg.embodiment_tag,
        "split": cfg.split,
        "use_fixed_val": cfg.use_fixed_val,
        "fixed_val_path": cfg.fixed_val_path,
        "val_ratio": cfg.val_ratio,
        "val_seed": cfg.val_seed,
        "target_tokens": cfg.target_tokens,
        "batch_size": cfg.batch_size,
        "precision": cfg.precision,
        "tf32": cfg.tf32,
        "tokenizer_type": wrapper.tokenizer_type,
        "tokenizer_action_dim": wrapper.action_dim,
        "tokenizer_action_horizon": wrapper.action_horizon,
        "tokenizer_emb_dim": wrapper.emb_dim,
        "tokenizer_num_global_tokens": wrapper.num_global_tokens,
        "tokenizer_num_main_tokens": wrapper.num_main_tokens,
        "tokenizer_num_hand_tokens": wrapper.num_hand_tokens,
        "action_keys": action_keys,
        "norm_dim_per_key": norm_dim_per_key,
        "unnorm_dim_per_key": unnorm_dim_per_key,
    }

    # 7. Pretty print.
    print("\n" + "=" * 60)
    print("RECON EVAL RESULTS")
    print("=" * 60)
    print(f"  N samples:         {metrics['n_samples']:,}")
    print(f"  action_horizon:    {metrics['action_horizon']}")
    print(f"  D (norm/unnorm):   {metrics['norm_action_dim']} / {metrics['unnorm_action_dim']}")
    print(
        f"  overall L1 norm:   mean={metrics['overall_norm_l1_mean']:.6f}  "
        f"max={metrics['overall_norm_l1_max']:.6f}"
    )
    print(
        f"  overall L1 unnorm: mean={metrics['overall_unnorm_l1_mean']:.6f}  "
        f"max={metrics['overall_unnorm_l1_max']:.6f}"
    )
    print("\n  Per-key (mean | max) — (norm | unnorm):")
    for k in action_keys:
        info = per_key[k]
        nm = sum(info["norm_l1_mean"]) / max(1, len(info["norm_l1_mean"]))
        nx = max(info["norm_l1_max"]) if info["norm_l1_max"] else 0.0
        um = sum(info["unnorm_l1_mean"]) / max(1, len(info["unnorm_l1_mean"]))
        ux = max(info["unnorm_l1_max"]) if info["unnorm_l1_max"] else 0.0
        print(f"    {k:30s}  norm={nm:.5f}/{nx:.5f}  unnorm={um:.5f}/{ux:.5f}")
    print("=" * 60 + "\n")

    # 8. Save JSON.
    out_path = out_dir / "recon_eval.json"
    payload = {**meta, **metrics}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[eval] Saved metrics to {out_path}")

    if cfg.save_per_sample:
        torch.save(
            {"norm_abs": norm_abs, "unnorm_abs": unnorm_abs},
            out_dir / "per_sample_abs.pt",
        )
        print(f"[eval] Saved per-sample abs tensors to {out_dir/'per_sample_abs.pt'}")


if __name__ == "__main__":
    cfg = tyro.cli(ArgsConfig)
    main(cfg)

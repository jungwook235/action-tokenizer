"""One-time script to generate proper metadata.json for robocasa actlat_fm VLA checkpoints.

The actlat_fm training script saves metadata as a flat config dict. The eval
policy (`gr00t.model.policy_actlat_fm.Gr00tPolicy._load_metadata`) expects
{"embodiment_tag": {statistics, modalities, ...}} format.

This reads stats from the robocasa dataset and generates the correct format in each
checkpoint's experiment_cfg/ directory, backing up the original flat config as
metadata_config.json (from which eval sbatch reads actlat_tokenizer_path +
actlat_target_tokens).

Usage:
    python scripts/generate_actlat_metadata_robocasa.py
"""

import json
import shutil
from pathlib import Path

import numpy as np

# ---------- Configuration ----------
DATASET_PATH = Path(
    "/storage1/sjw_dataset/dataset/huggingface/lerobot/shared/robocasa_preprocessed/robocasa_mg_gr00t_100"
)
CKPT_BASE = Path("/sjw_alinlab1/home/jungwook/Isaac-GR00T/checkpoints")
REPO_ROOT = Path("/sjw_alinlab1/home/jungwook/Isaac-GR00T")

CHECKPOINT_DIRS = [
    "vla_actlat_fm_robocasa_100demos/v2_hand_pred_norecon_mask_fullstate",
    "vla_actlat_fm_robocasa_100demos/v2_hand_pred_norecon_fullstate",
    "vla_actlat_fm_robocasa_100demos/v2_maskloss",
    "vla_actlat_fm_robocasa_100demos/v2_state_pred_full_time",
    "vla_actlat_fm_robocasa_100demos/v2_state_pred_full_time_mask",
    "vla_actlat_fm_robocasa_100demos/v2_state_pred_full_time_mask_statemask",
    "vla_nactlat_fm_robocasa_100demos/baseline",
]


def build_metadata(dataset_path: Path, embodiment_tag: str = "new_embodiment") -> dict:
    """Build DatasetMetadata-compatible dict from dataset meta files.

    Mirrors LeRobotSingleDataset._get_metadata() logic.
    """
    meta_dir = dataset_path / "meta"

    with open(meta_dir / "modality.json") as f:
        le_modality = json.load(f)

    with open(meta_dir / "info.json") as f:
        le_info = json.load(f)

    with open(meta_dir / "stats.json") as f:
        le_stats = json.load(f)

    modalities = {}
    for modality_type in ["state", "action"]:
        modalities[modality_type] = {}
        for subkey, meta in le_modality[modality_type].items():
            start, end = meta["start"], meta["end"]
            modalities[modality_type][subkey] = {
                "absolute": meta.get("absolute", True),
                "rotation_type": meta.get("rotation_type", None),
                "shape": [end - start],
                "continuous": True,
            }

    modalities["video"] = {}
    for new_key, vmeta in le_modality["video"].items():
        original_key = vmeta.get("original_key", new_key)
        le_video_info = le_info["features"][original_key]
        names = le_video_info["names"]
        shape = le_video_info["shape"]
        height = shape[names.index("height")]
        width = shape[names.index("width")]
        try:
            channels = shape[names.index("channel")]
            fps = le_video_info["video_info"]["video.fps"]
        except (ValueError, KeyError):
            channels = le_video_info["info"]["video.channels"]
            fps = le_video_info["info"]["video.fps"]
        modalities["video"][new_key] = {
            "resolution": [width, height],
            "channels": channels,
            "fps": fps,
        }

    statistics = {}
    for modality_type in ["state", "action"]:
        statistics[modality_type] = {}
        default_stats_key = (
            "observation.state" if modality_type == "state" else "action"
        )
        for subkey, meta in le_modality[modality_type].items():
            # modality.json may omit original_key; match gr00t/data/schema.py defaults
            original_key = meta.get("original_key") or default_stats_key
            start, end = meta["start"], meta["end"]
            indices = list(range(start, end))
            statistics[modality_type][subkey] = {}
            if original_key in le_stats:
                for stat_name, stat_values in le_stats[original_key].items():
                    arr = np.array(stat_values)
                    statistics[modality_type][subkey][stat_name] = arr[indices].tolist()

    dataset_metadata = {
        "statistics": statistics,
        "modalities": modalities,
        "embodiment_tag": embodiment_tag,
    }

    return {embodiment_tag: dataset_metadata}


def fix_tokenizer_path_in_config(config_path: Path, repo_root: Path = REPO_ROOT):
    """Ensure actlat_tokenizer_path in metadata_config.json is absolute."""
    if not config_path.exists():
        return
    with open(config_path) as f:
        config = json.load(f)
    tokenizer_path = config.get("actlat_tokenizer_path")
    if not tokenizer_path or Path(tokenizer_path).is_absolute():
        return
    absolute_path = str(repo_root / tokenizer_path)
    config["actlat_tokenizer_path"] = absolute_path
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  [fix-path] {config_path}: {tokenizer_path} -> {absolute_path}")


def process_checkpoint_dir(exp_cfg_dir: Path, metadata: dict):
    """Write proper metadata.json, backing up original as metadata_config.json."""
    metadata_path = exp_cfg_dir / "metadata.json"
    backup_path = exp_cfg_dir / "metadata_config.json"

    if metadata_path.exists():
        with open(metadata_path) as f:
            existing = json.load(f)
        if "new_embodiment" in existing and "statistics" in existing.get("new_embodiment", {}):
            print(f"  [skip] {metadata_path} already has correct format")
            fix_tokenizer_path_in_config(backup_path)
            return

        if not backup_path.exists():
            shutil.copy2(metadata_path, backup_path)
            print(f"  [backup] {metadata_path} -> {backup_path}")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"  [write] {metadata_path}")

    fix_tokenizer_path_in_config(backup_path)


def main():
    print(f"Building metadata from dataset: {DATASET_PATH}")
    metadata = build_metadata(DATASET_PATH)
    print(f"Metadata keys: {list(metadata.keys())}")

    for ckpt_rel in CHECKPOINT_DIRS:
        ckpt_dir = CKPT_BASE / ckpt_rel
        if not ckpt_dir.exists():
            print(f"\n[!] Checkpoint dir not found: {ckpt_dir}")
            continue

        print(f"\n=== {ckpt_rel} ===")

        top_cfg = ckpt_dir / "experiment_cfg"
        if top_cfg.exists():
            process_checkpoint_dir(top_cfg, metadata)

        for sub in sorted(ckpt_dir.iterdir()):
            if sub.is_dir() and sub.name.startswith("checkpoint-"):
                sub_cfg = sub / "experiment_cfg"
                if sub_cfg.exists():
                    process_checkpoint_dir(sub_cfg, metadata)

    print("\nDone!")


if __name__ == "__main__":
    main()

"""Persistent train/val episode split.

Writes / reads a JSON file that pins the train/val split for a dataset so the
exact same val set is reused across experiments. Two modes:

1. **Explicit path** (preferred): caller passes ``fixed_val_path``. The split
   is read from / written to that exact path. Multiple experiments pointing to
   the same path share the same split.
2. **Default path** (fallback): if ``fixed_val_path`` is ``None``, the split
   is stored at ``<dataset_path>/meta/fixed_val_split.json``.

The first run with a missing file generates the deterministic split (using
``val_seed`` and ``val_ratio``) and writes it. Subsequent runs use the file
verbatim — current ``val_seed`` / ``val_ratio`` CLI values are ignored once
the file exists, and a log line announces this.

A ``n_total`` consistency check guards against silent dataset growth.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np


FIXED_VAL_DEFAULT_FILENAME = "fixed_val_split.json"


def resolve_fixed_val_path(dataset_path: Path, override: Optional[str]) -> Path:
    """Return the absolute Path where the fixed-val JSON lives.

    If ``override`` is provided, use it as-is (interpreted as an absolute or
    cwd-relative path). Otherwise default to ``<dataset>/meta/fixed_val_split.json``.
    """
    if override:
        return Path(override)
    return Path(dataset_path) / "meta" / FIXED_VAL_DEFAULT_FILENAME


def load_or_create_fixed_split(
    dataset_path: Path,
    all_episode_ids: np.ndarray,
    val_seed: int,
    val_ratio: float,
    fixed_val_path: Optional[str] = None,
) -> dict:
    """Load existing fixed split, or compute & persist a new one.

    Args:
        dataset_path: dataset root (used for default file location and logging)
        all_episode_ids: full sorted list of episode IDs from episodes.jsonl
        val_seed: deterministic seed used IF the file does not yet exist
        val_ratio: validation ratio used IF the file does not yet exist
        fixed_val_path: optional explicit absolute path; ``None`` → default

    Returns:
        dict with keys: ``train_episode_ids`` (np.ndarray), ``val_episode_ids``
        (np.ndarray), ``source`` ('loaded' | 'created'), ``path`` (Path).
    """
    target = resolve_fixed_val_path(Path(dataset_path), fixed_val_path)
    n_total = int(len(all_episode_ids))

    if target.exists():
        with open(target, "r") as f:
            data = json.load(f)

        file_n_total = int(data.get("n_total", -1))
        assert file_n_total == n_total, (
            f"[fixed_val_split] n_total mismatch at {target}: "
            f"file={file_n_total} vs current dataset={n_total}. "
            f"Delete the file or use a different --fixed-val-path."
        )

        val_ids = np.asarray(data["val_episode_ids"], dtype=np.int64)
        train_ids = np.asarray(data["train_episode_ids"], dtype=np.int64)
        # Defensive: re-derive train from (all - val) if file was hand-edited
        # and ensure both lists are subsets of all_episode_ids.
        all_set = set(int(x) for x in all_episode_ids.tolist())
        for label, arr in [("val", val_ids), ("train", train_ids)]:
            missing = [int(x) for x in arr.tolist() if int(x) not in all_set]
            assert not missing, (
                f"[fixed_val_split] {label} contains episode_ids not in current dataset: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        print(
            f"[fixed_val_split] LOADED existing split from {target} "
            f"(val={len(val_ids)} / train={len(train_ids)} / total={n_total}, "
            f"file_seed={data.get('val_seed')}, file_ratio={data.get('val_ratio')})"
        )
        return {
            "train_episode_ids": train_ids,
            "val_episode_ids": val_ids,
            "source": "loaded",
            "path": target,
        }

    # File does not exist — compute and persist.
    n_val = max(1, int(n_total * val_ratio))
    rng = np.random.default_rng(val_seed)
    shuffled = rng.permutation(n_total)
    val_idx = np.sort(shuffled[:n_val])
    train_idx = np.sort(shuffled[n_val:])
    val_ids = np.asarray(all_episode_ids[val_idx], dtype=np.int64)
    train_ids = np.asarray(all_episode_ids[train_idx], dtype=np.int64)

    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "n_total": n_total,
        "val_seed": int(val_seed),
        "val_ratio": float(val_ratio),
        "val_episode_ids": [int(x) for x in val_ids.tolist()],
        "train_episode_ids": [int(x) for x in train_ids.tolist()],
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(target)

    print(
        f"[fixed_val_split] CREATED new split at {target} "
        f"(val={len(val_ids)} / train={len(train_ids)} / total={n_total}, "
        f"seed={val_seed}, ratio={val_ratio})"
    )
    return {
        "train_episode_ids": train_ids,
        "val_episode_ids": val_ids,
        "source": "created",
        "path": target,
    }


def get_fixed_split_for_split(
    dataset_path: Path,
    all_ids: np.ndarray,
    all_lengths: np.ndarray,
    split: str,
    val_seed: int,
    val_ratio: float,
    fixed_val_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ids, lengths)`` for the requested split, sorted ascending.

    ``split`` is "train" or "val". ``"all"`` should be handled by the caller
    (it does not need fixed-val resolution).
    """
    assert split in ("train", "val"), f"unsupported split: {split}"
    payload = load_or_create_fixed_split(
        dataset_path=dataset_path,
        all_episode_ids=all_ids,
        val_seed=val_seed,
        val_ratio=val_ratio,
        fixed_val_path=fixed_val_path,
    )
    selected_ids = (
        payload["val_episode_ids"] if split == "val" else payload["train_episode_ids"]
    )
    # Build index lookup for length alignment.
    id_to_idx = {int(eid): i for i, eid in enumerate(all_ids.tolist())}
    selected_idx = np.array([id_to_idx[int(eid)] for eid in selected_ids], dtype=np.int64)
    selected_idx.sort()  # match v2 deterministic ordering

    return all_ids[selected_idx], all_lengths[selected_idx]

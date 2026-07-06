"""Merge per-dataset normalization statistics (state and/or action) across
multiple single datasets and re-apply the merged statistics to each one.

The tokenizer training scripts (v3/v4/v5) combine several LeRobot datasets with
``torch.utils.data.ConcatDataset``, which does NOT merge normalization
statistics. Without this helper each dataset normalizes with its own
single-dataset min/max, so the same raw value maps to different normalized
values depending on which dataset it came from — and the distribution no longer
matches the Stage-2 VLA finetune (which uses ``LeRobotMixtureDataset``-merged
statistics) nor inference.

This computes the merged statistics (min-of-mins / max-of-maxs under the
``min_max`` mixing method, identical to ``LeRobotMixtureDataset``) for each
requested modality and writes them back into every dataset's transform, so a
ConcatDataset of them normalizes with the whole-mixture statistics.

Both ``state`` and ``action`` are merged by default. A dataset's transform only
reads the stats for the keys it actually normalizes, so merging a modality that
a given transform does not normalize (e.g. state in the action-only tokenizer)
is harmless — it simply future-proofs configs that DO normalize state.
"""

import copy
import json
from pathlib import Path

from gr00t.data.dataset import LeRobotMixtureDataset
from gr00t.data.schema import DatasetStatisticalValues


def apply_merged_normalization_metadata(
    datasets_for_stats,
    datasets_to_apply=None,
    percentile_mixing_method: str = "min_max",
    modalities=("state", "action"),
):
    """Merge stats from ``datasets_for_stats`` and apply to every dataset in
    ``datasets_to_apply`` (defaults to the same list).

    No-op (returns ``None``) when there is a single dataset: its own statistics
    already equal the whole-dataset statistics.

    Returns the merged ``DatasetMetadata`` (or ``None``).
    """
    if datasets_to_apply is None:
        datasets_to_apply = datasets_for_stats
    if len(datasets_for_stats) <= 1:
        return None

    # Under min_max mixing, min/max/q01/q99 are weight-independent (min-of-mins,
    # max-of-maxs); weights only affect mean/std. Use per-dataset sample counts.
    weights = [len(d) for d in datasets_for_stats]

    # First dataset's metadata is the template (modalities / embodiment tag);
    # only the requested modality statistics are swapped for the merged ones.
    merged_metadata = copy.deepcopy(datasets_for_stats[0].metadata)

    logged = []
    merged_stats_by_modality = {}
    for modality in modalities:
        per_ds_stats = [
            getattr(d.metadata.statistics, modality, None) for d in datasets_for_stats
        ]
        # Skip a modality that is missing/empty for any dataset (nothing to merge).
        if any(not s for s in per_ds_stats):
            continue

        per_task_stats = [
            {k: v.model_dump() for k, v in stats.items()} for stats in per_ds_stats
        ]
        merged = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=per_task_stats,
            dataset_sampling_weights=weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_stat = {k: DatasetStatisticalValues(**v) for k, v in merged.items()}
        merged_stats_by_modality[modality] = merged_stat
        setattr(merged_metadata.statistics, modality, merged_stat)
        first_key = next(iter(merged))
        logged.append(
            f"{modality}.{first_key} "
            f"min={merged[first_key]['min']} max={merged[first_key]['max']}"
        )

    # Apply the merged STATISTICS to each dataset while PRESERVING that dataset's
    # own modality metadata (video resolution, shapes, embodiment tag). Applying
    # dataset[0]'s whole metadata to everyone clobbers each dataset's video
    # resolution with dataset[0]'s — fatal when co-training datasets of different
    # native video resolutions (e.g. a 1280x800 dataset mixed with 256x256 ones),
    # because VideoToTensor.check_input then asserts the raw frames match
    # dataset[0]'s resolution. We swap only the merged state/action statistics.
    for d in datasets_to_apply:
        d_meta = copy.deepcopy(d.metadata)
        for modality, merged_stat in merged_stats_by_modality.items():
            setattr(d_meta.statistics, modality, merged_stat)
        d.set_transforms_metadata(d_meta)

    # Concise verification log — compare against the VLA's [norm] MERGED line;
    # they must match for Stage-1/Stage-2 consistency.
    print(
        f"[merge-stats] merged over {len(datasets_for_stats)} datasets: "
        + " | ".join(logged)
    )
    return merged_metadata


def save_normalization_stats(metadata, path):
    """Persist a dataset's normalization statistics to ``path`` as JSON.

    Writes the exact state/action ``min/max/mean/std/q01/q99`` that are actually
    applied at ``__getitem__`` time (i.e. the whole-mixture merged stats when
    training on multiple datasets) so inference / reproduction can reload them
    directly, instead of re-reading every source dataset's ``meta/stats.json``
    and re-running the mixture merge.

    Args:
        metadata: a ``DatasetMetadata`` (or any object exposing ``.statistics``
            and, optionally, ``.embodiment_tag``) — e.g. the value returned by
            :func:`apply_merged_normalization_metadata`, or a single dataset's
            ``.metadata`` when the merge was a no-op.
        path: destination file (parent dirs are created if missing).

    Returns:
        The ``Path`` written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    emb = getattr(metadata, "embodiment_tag", None)
    payload = {
        # str value ("new_embodiment") rather than the EmbodimentTag repr.
        "embodiment_tag": getattr(emb, "value", emb),
        # DatasetStatisticalValues has a json field_serializer that turns the
        # ndarrays into plain lists, so mode="json" yields a serializable dict.
        "statistics": metadata.statistics.model_dump(mode="json"),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path

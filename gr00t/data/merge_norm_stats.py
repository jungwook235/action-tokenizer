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
        setattr(
            merged_metadata.statistics,
            modality,
            {k: DatasetStatisticalValues(**v) for k, v in merged.items()},
        )
        first_key = next(iter(merged))
        logged.append(
            f"{modality}.{first_key} "
            f"min={merged[first_key]['min']} max={merged[first_key]['max']}"
        )

    for d in datasets_to_apply:
        d.set_transforms_metadata(merged_metadata)

    # Concise verification log — compare against the VLA's [norm] MERGED line;
    # they must match for Stage-1/Stage-2 consistency.
    print(
        f"[merge-stats] merged over {len(datasets_for_stats)} datasets: "
        + " | ".join(logged)
    )
    return merged_metadata

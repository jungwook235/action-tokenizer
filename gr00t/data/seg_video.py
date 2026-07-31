"""Segment (SAM3 cutout) video resolution shared by the V4 seg-DINO-stream datasets.

The cutout videos produced by ``analysis/sam3_masking/batch_sam3_robot_task.py`` mirror
the source LeRobot dataset layout under a separate root::

    <seg_root>/<dataset_dir_name>/cutout/chunk-000/<video_key>/episode_000000.mp4
    <source_root>/<dataset_dir_name>/videos/chunk-000/<video_key>/episode_000000.mp4

i.e. the ONLY differences are the root and the top-level ``videos`` → ``cutout``
component. Frame count / fps / resolution are identical to the source video (the
cutout is written frame-for-frame from it), so the SAME timestamp lookup used for the
RGB stream lands on the SAME step in the cutout stream.

Both the Stage-1 tokenizer dataset (``ActionFramesDatasetV4``) and the Stage-2 VLA
dataset (``LeRobotSingleDatasetActlatFMV4``) resolve seg paths through
:func:`seg_video_path_from_source`, so the two stages can never disagree.
"""

from pathlib import Path


def seg_dataset_dir(seg_root: str | Path, dataset_path: str | Path) -> Path:
    """``<seg_root>/<basename(dataset_path)>`` — the seg mirror of one dataset.

    Raises if the mirror directory is missing, so a typo'd root / a dataset with no
    cutouts fails at construction time rather than mid-training.
    """
    seg_dir = Path(seg_root) / Path(dataset_path).name
    if not seg_dir.is_dir():
        raise FileNotFoundError(
            f"segment (cutout) directory not found: {seg_dir}\n"
            f"  seg_root={seg_root}\n  dataset_path={dataset_path}\n"
            f"Expected <seg_root>/<dataset_dir_name>/<subdir>/chunk-XXX/<video_key>/"
            f"episode_XXXXXX.mp4"
        )
    return seg_dir


def seg_video_path_from_source(
    source_video_path: str | Path,
    dataset_path: str | Path,
    seg_dir: str | Path,
    seg_video_subdir: str = "cutout",
) -> Path:
    """Map a resolved source video path to its segment counterpart.

    Derived FROM the source path (rather than re-formatting the path pattern) so all
    of the base dataset's ``video_path``-pattern / ``original_key`` handling is reused
    verbatim and the two streams cannot drift apart.

    Args:
        source_video_path: absolute path returned by ``LeRobotSingleDataset.get_video_path``.
        dataset_path: the source dataset root (used to relativize).
        seg_dir: this dataset's seg mirror (see :func:`seg_dataset_dir`).
        seg_video_subdir: the seg mirror's top-level subdir ("cutout", "overlay", ...)
            replacing the source's "videos".
    """
    rel = Path(source_video_path).relative_to(Path(dataset_path))
    parts = rel.parts
    assert len(parts) >= 2, f"unexpected video relpath: {rel}"
    assert parts[0] == "videos", (
        f"expected the dataset's video_path pattern to start with 'videos/'; got "
        f"{rel!r}. The seg mirror layout replaces that component with "
        f"{seg_video_subdir!r}, so a different pattern needs explicit support."
    )
    return Path(seg_dir).joinpath(seg_video_subdir, *parts[1:])

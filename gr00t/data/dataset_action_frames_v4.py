"""ActionFramesDatasetV4: loads an action chunk + two aligned RGB frames.

Used by the V4 (RLA-DINO hybrid) tokenizer training. For each sample it returns:
  - ``action``  : [T, D] normalized action chunk (same pipeline as ActionOnlyDataset)
  - ``frame_x0``: [H, W, 3] uint8 RGB at chunk start (observation index 0)
  - ``frame_x1``: [H, W, 3] uint8 RGB at chunk end (observation index T-1)

With the optional segment (SAM3 cutout) stream enabled (``seg_dataset_root`` set) two
more frames are attached, read from the cutout mirror of the SAME dataset at the SAME
two steps and preprocessed through the SAME video transforms:
  - ``seg_x0``  : [H, W, 3] uint8 cutout at chunk start
  - ``seg_x1``  : [H, W, 3] uint8 cutout at chunk end
When ``seg_dataset_root`` is None (the default) nothing changes — the items, the
collator output and the trainer path are byte-identical to before.

DINO features are NOT computed here — the trainer runs the frozen extractor
on-the-fly. Frames are returned as resized uint8 (224x224 by default); the
trainer converts to float/255 before DINO.

The same class works for both ``gr1_unified`` and ``robocasa_gr1_tabletop``
(identical action keys + single ``video.ego_view`` camera); only the dataset path
and per-dataset normalization stats differ.

Reuses the persistent fixed-val split (``get_fixed_split_for_split``) and the
action transform pipeline (``ToTensor → StateActionTransform → Concat``) so the
action distribution matches V2/V3/VLA exactly.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from gr00t.data.dataset import (
    LE_ROBOT_EPISODE_FILENAME,
    LeRobotSingleDataset,
    ModalityConfig,
)
from gr00t.data.fixed_val_split import get_fixed_split_for_split
from gr00t.data.seg_video import seg_dataset_dir, seg_video_path_from_source
from gr00t.data.transform import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import VideoResize, VideoToNumpy, VideoToTensor
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.utils.video import get_frames_by_timestamps


class ActionFramesDatasetV4(LeRobotSingleDataset):
    """LeRobotSingleDataset variant loading action chunk + (x0, x1) frames.

    Args:
        dataset_path: dataset root.
        data_config_name: DATA_CONFIG_MAP key (e.g. "fourier_gr1_arms_waist").
        embodiment_tag: robot tag (e.g. "new_embodiment").
        split: "train" | "val" | "all".
        val_ratio / val_seed: split parameters.
        normalization_mode: fallback action normalization when the data_config
            does not define ``action_normalization_modes``.
        image_size: resize target (square) handed to the DINO extractor.
        use_fixed_val / fixed_val_path: persistent val split (same as V3).
        seg_dataset_root: root of the SAM3 cutout mirror (e.g.
            ``.../GR00T-X-Embodiment-Sim_sam3_robot_task``). None (default) disables
            the segment stream entirely — items keep their original keys.
        seg_video_subdir: subdir inside the mirror holding the videos ("cutout").
    """

    def __init__(
        self,
        dataset_path: str | Path,
        data_config_name: str,
        embodiment_tag: str,
        split: str = "all",
        val_ratio: float = 0.003,
        val_seed: int = 42,
        normalization_mode: str = "min_max",
        image_size: int = 224,
        video_backend: str = "decord",
        use_fixed_val: bool = True,
        fixed_val_path: Optional[str] = None,
        seg_dataset_root: Optional[str] = None,
        seg_video_subdir: str = "cutout",
    ):
        assert split in ("train", "val", "all"), f"split must be train/val/all: {split}"

        self._split = split
        self._val_ratio = val_ratio
        self._val_seed = val_seed
        self._use_fixed_val = use_fixed_val
        self._fixed_val_path = fixed_val_path

        data_config_cls = DATA_CONFIG_MAP[data_config_name]
        full_modality_configs = data_config_cls.modality_config()
        assert "action" in full_modality_configs, (
            f"data_config '{data_config_name}' has no action modality."
        )
        assert "video" in full_modality_configs, (
            f"data_config '{data_config_name}' has no video modality."
        )

        action_keys = list(full_modality_configs["action"].modality_keys)
        action_indices = list(full_modality_configs["action"].delta_indices)
        action_horizon = len(action_indices)

        video_keys = list(full_modality_configs["video"].modality_keys)
        assert len(video_keys) == 1, (
            f"V4 expects a single camera; got video_keys={video_keys}"
        )
        self._video_key = video_keys[0]

        # x0 = chunk start (index 0), x1 = chunk end (index action_horizon - 1).
        # Base dataset clamps to trajectory length, so short episodes give x0≈x1.
        self._video_indices = [0, action_horizon - 1]

        modality_configs = {
            "action": ModalityConfig(
                delta_indices=action_indices, modality_keys=action_keys
            ),
            "video": ModalityConfig(
                delta_indices=self._video_indices, modality_keys=video_keys
            ),
        }

        # Per-key normalization / rotation, respecting data_config (same as ActionOnly).
        action_normalization_modes = getattr(
            data_config_cls, "action_normalization_modes", None
        )
        if not action_normalization_modes:
            action_normalization_modes = {key: normalization_mode for key in action_keys}
        action_target_rotations = getattr(data_config_cls, "action_target_rotations", {}) or {}

        transforms = self._build_transforms(
            video_keys=video_keys,
            action_keys=action_keys,
            action_normalization_modes=action_normalization_modes,
            action_target_rotations=action_target_rotations,
            image_size=image_size,
        )

        super().__init__(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            transforms=transforms,
        )

        self._action_keys = action_keys
        self._action_horizon = action_horizon

        # ---- optional segment (SAM3 cutout) stream ----
        # Off by default (root=None): no extra attributes are used at __getitem__ time
        # and the returned item is byte-identical to before. When on, the cutout mirror
        # of this dataset is located eagerly (fails loud on a bad root) and a SEPARATE
        # instance of the same three video transforms is built for it, so the cutout
        # frames get byte-identical preprocessing to the RGB frames.
        self._seg_root = seg_dataset_root
        self._seg_video_subdir = seg_video_subdir
        self._seg_dir = None
        self._seg_transforms = None
        if seg_dataset_root is not None:
            self._seg_dir = seg_dataset_dir(seg_dataset_root, dataset_path)
            self._seg_transforms = ComposedModalityTransform(
                transforms=[
                    VideoToTensor(apply_to=video_keys),
                    VideoResize(
                        apply_to=video_keys,
                        height=image_size,
                        width=image_size,
                        interpolation="linear",
                    ),
                    VideoToNumpy(apply_to=video_keys),
                ]
            )
            self._seg_transforms.set_metadata(self.metadata)

    @staticmethod
    def _build_transforms(
        video_keys: list[str],
        action_keys: list[str],
        action_normalization_modes: dict[str, str],
        action_target_rotations: dict[str, str],
        image_size: int,
    ) -> ComposedModalityTransform:
        return ComposedModalityTransform(
            transforms=[
                # video: minimal (no crop/jitter) — DINO does its own normalization.
                VideoToTensor(apply_to=video_keys),
                VideoResize(
                    apply_to=video_keys,
                    height=image_size,
                    width=image_size,
                    interpolation="linear",
                ),
                VideoToNumpy(apply_to=video_keys),
                # action: numpy → tensor → (rotation +) normalize → concat
                StateActionToTensor(
                    apply_to=action_keys,
                    output_dtypes={key: torch.float32 for key in action_keys},
                ),
                StateActionTransform(
                    apply_to=action_keys,
                    normalization_modes=action_normalization_modes,
                    target_rotations=action_target_rotations,
                ),
                # Concat video into a single "video" tensor [T, V=1, H, W, C].
                # (Leaving video_concat_order=[] while a video key is present makes
                # ConcatTransform call np.concatenate([]) → "need at least one
                # array to concatenate". So we DO list the video key here.)
                ConcatTransform(
                    video_concat_order=video_keys,
                    state_concat_order=None,
                    action_concat_order=action_keys,
                ),
            ]
        )

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Episode-level split using the persistent fixed-val file when enabled."""
        episode_path = self._dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]

        all_ids = np.array([e["episode_index"] for e in episode_metadata])
        all_lengths = np.array([e["length"] for e in episode_metadata])

        if self._split == "all":
            return all_ids, all_lengths

        if self._use_fixed_val:
            ids, lengths = get_fixed_split_for_split(
                dataset_path=self._dataset_path,
                all_ids=all_ids,
                all_lengths=all_lengths,
                split=self._split,
                val_seed=self._val_seed,
                val_ratio=self._val_ratio,
                fixed_val_path=self._fixed_val_path,
            )
            print(
                f"[ActionFramesDatasetV4][{self._split}] {Path(self._dataset_path).name}: "
                f"{len(all_ids)} episodes → {len(ids)} used (fixed-val)"
            )
            return ids, lengths

        n_total = len(all_ids)
        n_val = max(1, int(n_total * self._val_ratio))
        rng = np.random.default_rng(self._val_seed)
        shuffled = rng.permutation(n_total)
        selected = np.sort(shuffled[:n_val]) if self._split == "val" else np.sort(shuffled[n_val:])
        print(
            f"[ActionFramesDatasetV4][{self._split}] {Path(self._dataset_path).name}: "
            f"{n_total} episodes → {len(selected)} used (val_ratio={self._val_ratio})"
        )
        return all_ids[selected], all_lengths[selected]

    def _load_seg_frames(self, trajectory_id: int, base_index: int) -> np.ndarray:
        """Cutout frames at the SAME two steps as (frame_x0, frame_x1): [2, S, S, C] uint8.

        Mirrors ``LeRobotSingleDataset.get_video`` exactly — same clamped step indices,
        same ``timestamp`` column, same decoder/backend — but reads the cutout mirror of
        the video ``get_video_path`` resolved. Requires ``self.curr_traj_data`` to be
        populated for this trajectory (``get_step_data`` in ``__getitem__`` does that).
        """
        assert self._seg_dir is not None, "segment stream is disabled"
        traj_index = self.get_trajectory_index(trajectory_id)
        step_indices = np.asarray(self._video_indices) + base_index
        step_indices = np.clip(step_indices, 0, self.trajectory_lengths[traj_index] - 1)

        sub_key = self._video_key.replace("video.", "")
        src_path = self.get_video_path(trajectory_id, sub_key)
        seg_path = seg_video_path_from_source(
            src_path, self._dataset_path, self._seg_dir, self._seg_video_subdir
        )

        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp in {trajectory_id=}"
        video_timestamp = self.curr_traj_data["timestamp"].to_numpy()[step_indices]

        frames = get_frames_by_timestamps(
            seg_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )  # [2, H, W, C] uint8, native resolution

        # Same VideoToTensor → VideoResize → VideoToNumpy chain the RGB stream goes
        # through, so the frozen extractor sees identically-preprocessed pixels.
        out = self._seg_transforms({self._video_key: np.asarray(frames)})
        return np.asarray(out[self._video_key])

    def __getitem__(self, index: int) -> dict:
        trajectory_id, base_index = self.all_steps[index]
        data = self.get_step_data(trajectory_id, base_index)
        data = self.transforms(data)

        action = data["action"]  # [T, D]
        # ConcatTransform merges the video key into "video" with a camera axis:
        # [T, V=1, H, W, C]. Fall back to the raw per-key name if not merged.
        frames = data["video"] if "video" in data else data[self._video_key]
        frames = np.asarray(frames)
        if frames.ndim == 5:  # [T, Cam, H, W, C] → drop single camera dim
            frames = frames[:, 0]
        frame_x0 = frames[0]  # [H, W, C]
        frame_x1 = frames[1] if frames.shape[0] > 1 else frames[0]

        item = {
            "action": action,
            "frame_x0": np.ascontiguousarray(frame_x0),
            "frame_x1": np.ascontiguousarray(frame_x1),
        }

        if self._seg_dir is not None:
            seg = self._load_seg_frames(trajectory_id, base_index)
            item["seg_x0"] = np.ascontiguousarray(seg[0])
            item["seg_x1"] = np.ascontiguousarray(seg[1] if seg.shape[0] > 1 else seg[0])

        return item


class ActionFramesCollatorV4:
    """Collate {action, frame_x0, frame_x1} into batched tensors.

    Frames are stacked as uint8 ``[B, 3, H, W]`` (channels-first); the trainer
    converts to float/255 before the DINO extractor.

    When the dataset also yields the segment (cutout) pair, ``seg_x0``/``seg_x1`` are
    stacked the same way. Their absence (the default) leaves the output dict exactly
    as before.
    """

    def __call__(self, features: list[dict]) -> dict:
        actions = torch.stack([f["action"] for f in features])  # [B, T, D]

        def stack_frames(key: str) -> torch.Tensor:
            arr = np.stack([np.asarray(f[key]) for f in features])  # [B, H, W, C] uint8
            t = torch.from_numpy(arr)
            return t.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        batch = {
            "action": actions,
            "frame_x0": stack_frames("frame_x0"),
            "frame_x1": stack_frames("frame_x1"),
        }
        if "seg_x0" in features[0]:
            batch["seg_x0"] = stack_frames("seg_x0")
            batch["seg_x1"] = stack_frames("seg_x1")
        return batch

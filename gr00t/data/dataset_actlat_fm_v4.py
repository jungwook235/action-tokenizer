"""V4 VLA dataset: LeRobotSingleDatasetActlatFM + (x0, x1) frame pair.

The V4 (RLA-DINO) tokenizer produces DINO-dependent latents, so at VLA training
time the latent target must be computed from the two frames (chunk start / end).
This subclass keeps the normal actlat_fm sample (state / eagle video / action)
exactly as-is and ADDITIONALLY attaches two raw RGB frames:

  - ``frame_x0`` : [3, H, W] uint8 at chunk start (observation index 0)
  - ``frame_x1`` : [3, H, W] uint8 at chunk end (observation index action_horizon-1)

These are loaded independently of the eagle ``video.ego_view`` modality (whose
``delta_indices`` stay ``[0]``), so the VLA backbone input is unchanged. The model
forward feeds ``frame_x0`` / ``frame_x1`` into ``get_latent_target`` only when the
tokenizer is V4; v2/v3 ignore them. Inference (decode-only) never needs frames.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from gr00t.data.dataset_actlat_fm import LeRobotSingleDatasetActlatFM
from gr00t.utils.video import get_frames_by_timestamps


class LeRobotSingleDatasetActlatFMV4(LeRobotSingleDatasetActlatFM):
    """actlat_fm dataset that also yields a (x0, x1) frame pair for V4.

    Extra args:
        frame_video_key: video modality key to read frames from (default
            "video.ego_view"). Must exist in the dataset's modality meta.
        frame_image_size: square resize applied to each frame (matches the V4
            tokenizer training; default 224).
        frame_action_horizon: chunk length; x1 is taken at base + (H - 1).
    """

    def __init__(
        self,
        *args,
        frame_video_key: str = "video.ego_view",
        frame_image_size: int = 224,
        frame_action_horizon: int = 16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._frame_video_key = frame_video_key
        self._frame_image_size = int(frame_image_size)
        self._frame_action_horizon = int(frame_action_horizon)

    def _resize_to_chw_uint8(self, frame_hwc: np.ndarray) -> np.ndarray:
        """[H, W, C] uint8 → [C, S, S] uint8 (bilinear resize to image_size).

        Byte-matches the Stage-1 V4 tokenizer's frame preprocessing
        (ActionFramesDatasetV4: VideoToTensor → VideoResize → VideoToNumpy), so
        the frozen DINO sees the SAME pixels at VLA-training latent-target time as
        it did during tokenizer training. The Stage-1 path resizes on a [0,1]
        float tensor with torchvision T.Resize(bilinear, antialias=True) and
        re-quantizes by truncation ((x*255).to(uint8), no clamp); we replicate
        that exactly: F.interpolate(bilinear, antialias=True) shares the same aten
        kernel and align_corners=False, and antialiased bilinear stays within
        [0,1] (convex weights) so no clamp is needed.
        """
        t = torch.from_numpy(np.ascontiguousarray(frame_hwc)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        s = self._frame_image_size
        t = F.interpolate(t, size=(s, s), mode="bilinear", align_corners=False, antialias=True)
        t = (t.squeeze(0) * 255.0).to(torch.uint8)  # truncate, matching VideoToNumpy
        return t.numpy()

    def _load_frame_pair(self, trajectory_id: int, base_index: int):
        """Load (x0, x1) frames at chunk start / end as [3, S, S] uint8."""
        H = self._frame_action_horizon
        step_indices = np.array([0, H - 1]) + base_index
        traj_idx = self.get_trajectory_index(trajectory_id)
        step_indices = np.clip(step_indices, 0, self.trajectory_lengths[traj_idx] - 1)

        key = self._frame_video_key
        assert key.startswith("video."), f"frame_video_key must start with 'video.': {key}"
        sub_key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, sub_key)

        assert self.curr_traj_data is not None and "timestamp" in self.curr_traj_data.columns
        timestamp = self.curr_traj_data["timestamp"].to_numpy()
        video_ts = timestamp[step_indices]

        frames = get_frames_by_timestamps(
            video_path.as_posix(),
            video_ts,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )  # [2, H, W, C] uint8
        frames = np.asarray(frames)
        f0 = self._resize_to_chw_uint8(frames[0])
        f1 = self._resize_to_chw_uint8(frames[1] if frames.shape[0] > 1 else frames[0])
        return f0, f1

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        # super().__getitem__ has populated self.curr_traj_data for this trajectory.
        trajectory_id, base_index = self.all_steps[index]
        f0, f1 = self._load_frame_pair(trajectory_id, base_index)
        item["frame_x0"] = f0  # [3, S, S] uint8
        item["frame_x1"] = f1
        return item

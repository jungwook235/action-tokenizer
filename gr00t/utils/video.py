# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch  # noqa: F401 # isort: skip
import torchvision  # noqa: F401 # isort: skip
import av
import cv2
import decord  # noqa: F401
import numpy as np
import os


def get_frames_by_indices(
    video_path: str,
    indices: list[int] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    if video_backend == "decord":
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "opencv":
        frames = []
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    else:
        raise NotImplementedError


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "torchvision_av",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    """Get frames from a video at specified timestamps.
    Args:
        video_path (str): Path to the video file.
        timestamps (list[int] | np.ndarray): Timestamps to retrieve frames for, in seconds.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
    Returns:
        np.ndarray: Frames at the specified timestamps.
    """
    if video_backend == "decord":
        #vr = decord.VideoReader(video_path, **video_backend_kwargs)
        try:
            vr = decord.VideoReader(video_path, **video_backend_kwargs)
            #print(f"Successfully loaded video: {video_path}")
            #print(f"File size: {os.path.getsize(video_path)} bytes")
            #print(f"Number of frames: {len(vr)}")
            #print(f"Frame shape: {vr.get_frame_shape()}")
            #print(f"Frame timestamps: {vr.get_frame_timestamp(range(len(vr)))}")
            #print(f"Frame indices: {vr.get_batch(range(len(vr)))}")
            #print(f"Frame data: {vr.get_batch(range(len(vr))).asnumpy()}")
            #print(f"Frame data shape: {vr.get_batch(range(len(vr))).asnumpy().shape}")
        except Exception as e:
            #print(f"\n[!!! ERROR !!!] Failed to load video: {video_path}")
            #print(f"File size: {os.path.getsize(video_path)} bytes")
            #print(f"Error: {e}")
            raise e  # 에러를 다시 발생시켜 학습을 멈춤
        num_frames = len(vr)
        # Retrieve the timestamps for each frame in the video
        frame_ts: np.ndarray = vr.get_frame_timestamp(range(num_frames))
        # Map each requested timestamp to the closest frame index
        # Only take the first element of the frame_ts array which corresponds to start_seconds
        indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "opencv":
        # Open the video file
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        # Retrieve the total number of frames
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate timestamps for each frame
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ts = np.arange(num_frames) / fps
        frame_ts = frame_ts[:, np.newaxis]  # Reshape to (num_frames, 1) for broadcasting
        # Map each requested timestamp to the closest frame index
        indices = np.abs(frame_ts - timestamps).argmin(axis=0)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    elif video_backend == "torchvision_av":
        # set backend
        torchvision.set_video_backend("pyav")
        # set a video stream reader
        reader = torchvision.io.VideoReader(video_path, "video")
        # set the first and last requested timestamps
        # Note: previous timestamps are usually loaded, since we need to access the previous key frame
        first_ts = timestamps[0]
        last_ts = timestamps[-1]
        # access closest key frame of the first requested frame
        # Note: closest key frame timestamp is usally smaller than `first_ts` (e.g. key frame can be the first frame of the video)
        # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
        reader.seek(first_ts, keyframes_only=True)
        # load all frames until last requested frame
        loaded_frames = []
        loaded_ts = []

        # ===== PATCH (PyAV codec leak fix): wrap reader use in try/finally =====
        # Why: PyAV's libavcodec contexts hold mmap pages / decoder threads that are
        # not always reclaimed by Python GC alone. After many thousands of opens
        # (worker x step), `codec.open()` fails with ENOMEM (errno 12). Explicitly
        # closing every stream's codec_context before closing the container, and
        # guaranteeing close happens even on exceptions, slows the leak drastically.
        try:
            for frame in reader:
                current_ts = frame["pts"]
                if current_ts in timestamps:
                    loaded_frames.append(frame["data"].numpy())
                    loaded_ts.append(current_ts)

                if current_ts >= last_ts:
                    break
                if len(loaded_frames) >= len(timestamps):
                    break
        finally:
            # ----- PATCH start: explicit codec_context + container teardown -----
            try:
                container = getattr(reader, "container", None)
                if container is not None:
                    for stream in container.streams:
                        cc = getattr(stream, "codec_context", None)
                        if cc is not None:
                            try:
                                cc.close()
                            except Exception:
                                pass
                    try:
                        container.close()
                    except Exception:
                        pass
            finally:
                del reader
            # ----- PATCH end -----

        if len(timestamps) != len(loaded_ts) and timestamps[0] == timestamps[1]:
            loaded_frames = loaded_frames * 2
            loaded_ts = loaded_ts * 2

        frames = np.array(loaded_frames)

        if len(loaded_ts) != len(timestamps):
            print("loaded_ts", loaded_ts)
            print("timestamps", timestamps)
            raise ValueError(f"len(loaded_ts) != len(timestamps): {len(loaded_ts)} != {len(timestamps)}")

        return frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError


def get_all_frames(
    video_path: str,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
    resize_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Get all frames from a video.
    Args:
        video_path (str): Path to the video file.
        video_backend (str, optional): Video backend to use. Defaults to "decord".
        video_backend_kwargs (dict, optional): Keyword arguments for the video backend.
        resize_size (tuple[int, int], optional): Resize size for the frames. Defaults to None.
    """
    if video_backend == "decord":
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(range(len(vr))).asnumpy()
    elif video_backend == "pyav":
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frame = frame.to_ndarray(format="rgb24")
            frames.append(frame)
        frames = np.array(frames)
    elif video_backend == "torchvision_av":
        # set backend and reader
        torchvision.set_video_backend("pyav")
        reader = torchvision.io.VideoReader(video_path, "video")
        frames = []
        for frame in reader:
            frames.append(frame["data"].numpy())
        frames = np.array(frames)
        frames = frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError(f"Video backend {video_backend} not implemented")
    # resize frames if specified
    if resize_size is not None:
        frames = [cv2.resize(frame, resize_size) for frame in frames]
        frames = np.array(frames)
    return frames

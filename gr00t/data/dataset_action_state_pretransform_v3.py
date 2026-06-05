"""V3 variant of PreTransformedActionStateDataset with persistent fixed-val split.

Same caching behavior as :class:`PreTransformedActionStateDataset` (action +
state + future + optional FAST tokens) but builds an :class:`ActionStateDatasetV3`
underneath so train/val split is read from / persisted to a JSON file.

The collator class is re-exported from the v2 file (no behavioral change).
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from gr00t.data.dataset import LE_ROBOT_EPISODE_FILENAME
from gr00t.data.dataset_action_state_pretransform import (
    ActionStateCollator,  # re-export
    ActionStateDataset,
)
from gr00t.data.fixed_val_split import get_fixed_split_for_split


class ActionStateDatasetV3(ActionStateDataset):
    """ActionStateDataset with optional persistent fixed-val split."""

    def __init__(
        self,
        dataset_path: str | Path,
        data_config_name: str,
        embodiment_tag: str,
        split: str = "all",
        val_ratio: float = 0.003,
        val_seed: int = 42,
        normalization_mode: str = "min_max",
        video_backend: str = "torchvision_av",
        use_fixed_val: bool = True,
        fixed_val_path: Optional[str] = None,
    ):
        self._use_fixed_val = use_fixed_val
        self._fixed_val_path = fixed_val_path
        super().__init__(
            dataset_path=dataset_path,
            data_config_name=data_config_name,
            embodiment_tag=embodiment_tag,
            split=split,
            val_ratio=val_ratio,
            val_seed=val_seed,
            normalization_mode=normalization_mode,
            video_backend=video_backend,
        )

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        episode_path = self._dataset_path / LE_ROBOT_EPISODE_FILENAME
        with open(episode_path, "r") as f:
            episode_metadata = [json.loads(line) for line in f]

        all_ids = np.array([e["episode_index"] for e in episode_metadata])
        all_lengths = np.array([e["length"] for e in episode_metadata])

        if self._split == "all":
            return all_ids, all_lengths

        if not self._use_fixed_val:
            return super()._get_trajectories()

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
            f"[ActionStateDatasetV3][{self._split}] {Path(self._dataset_path).name}: "
            f"전체 {len(all_ids)}개 에피소드 중 {len(ids)}개 사용 (fixed-val)"
        )
        return ids, lengths


class PreTransformedActionStateDatasetV3(Dataset):
    """V3 pre-transformed action+state dataset with fixed-val support.

    Mirrors :class:`PreTransformedActionStateDataset` 1-for-1; only the source
    dataset class differs.
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
        hand_state_dims: Optional[list[int]] = None,
        hand_pred_future_steps: Optional[list[int]] = None,
        cache_fast_tokens: bool = False,
        fast_tokenizer_path: str = "physical-intelligence/fast",
        fast_vocab_size: int = 2048,
        max_trajectories: Optional[int] = None,
        use_fixed_val: bool = True,
        fixed_val_path: Optional[str] = None,
    ):
        need_state = hand_state_dims is not None and len(hand_state_dims) > 0
        need_future = need_state and hand_pred_future_steps is not None and len(hand_pred_future_steps) > 0

        # Robocasa GR1 state-dim auto-expansion (mirrors v2 behavior).
        _ROBOCASA_DATA_CONFIGS = {"single_panda_gripper", "single_panda_gripper_actlat_fm"}
        if need_state and data_config_name in _ROBOCASA_DATA_CONFIGS:
            _robocasa_state_dim = 20
            if len(hand_state_dims) < _robocasa_state_dim:
                old_len = len(hand_state_dims)
                hand_state_dims = list(range(_robocasa_state_dim))
                print(
                    f"[PreTransformedActionStateDatasetV3] Robocasa data_config "
                    f"'{data_config_name}' detected: hand_state_dims auto-expanded "
                    f"{old_len} → {_robocasa_state_dim}"
                )

        source = ActionStateDatasetV3(
            dataset_path=dataset_path,
            data_config_name=data_config_name,
            embodiment_tag=embodiment_tag,
            split=split,
            val_ratio=val_ratio,
            val_seed=val_seed,
            normalization_mode=normalization_mode,
            use_fixed_val=use_fixed_val,
            fixed_val_path=fixed_val_path,
        )

        if max_trajectories is not None and max_trajectories > 0 and len(source.trajectory_ids) > max_trajectories:
            old_n_traj = len(source.trajectory_ids)
            source._trajectory_ids = source.trajectory_ids[:max_trajectories]
            source._trajectory_lengths = source.trajectory_lengths[:max_trajectories]
            source._all_steps = source._get_all_steps()
            print(
                f"[PreTransform v3][{split}] max_trajectories={max_trajectories} "
                f"-> trajectories {old_n_traj} -> {len(source.trajectory_ids)}"
            )

        n = len(source)
        assert n > 0, f"Dataset is empty: {dataset_path}"

        # --- Action cache ---
        first = source[0]
        first_action = first["action"]
        T, D = first_action.shape

        action_cache = torch.empty(n, T, D, dtype=torch.float32)
        action_cache[0] = first_action

        # --- State cache 준비 ---
        has_state = "state" in first and need_state
        state_cache = None
        if has_state:
            first_state = first["state"]
            hand_dim = len(hand_state_dims)
            state_cache = torch.empty(n, hand_dim, dtype=torch.float32)

        # --- Future state cache 준비 ---
        future_cache = None
        num_future = 0
        if need_future and has_state:
            num_future = len(hand_pred_future_steps)
            hand_dim = len(hand_state_dims)
            future_cache = torch.empty(n, num_future, hand_dim, dtype=torch.float32)

        # --- Main caching loop (current + future 함께) ---
        if future_cache is not None:
            traj_len_map = dict(
                zip(source.trajectory_ids.tolist(), source.trajectory_lengths.tolist())
            )
            loop_desc = f"[PreTransform v3][{split}] Caching action+state+future"
        else:
            traj_len_map = None
            loop_desc = f"[PreTransform v3][{split}] Caching action+state"

        for i in tqdm(range(n), desc=loop_desc):
            if i == 0:
                sample = first
            else:
                sample = source[i]

            action_cache[i] = sample["action"]

            if state_cache is not None:
                state_tensor = sample["state"]
                if state_tensor.ndim == 2:
                    state_tensor = state_tensor[0]
                state_cache[i] = state_tensor[hand_state_dims]

            if future_cache is not None:
                traj_id, base_idx = source.all_steps[i]
                traj_len = traj_len_map[traj_id]
                for fi, step_offset in enumerate(hand_pred_future_steps):
                    future_idx = min(base_idx + step_offset, traj_len - 1)
                    future_data = source.get_step_data(traj_id, future_idx)
                    future_data = source.transforms(future_data)

                    future_state = future_data["state"]
                    if future_state.ndim == 2:
                        future_state = future_state[0]
                    future_cache[i, fi] = future_state[hand_state_dims]

        # --- FAST token caching ---
        fast_cache = None
        if cache_fast_tokens:
            print(f"[PreTransform v3][{split}] Tokenizing actions with FAST tokenizer...")
            from transformers import AutoProcessor
            fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)

            actions_np = action_cache.numpy()
            batch_size = 1024
            all_tokens = []

            for start in tqdm(range(0, n, batch_size), desc=f"[FAST tokenize v3][{split}]"):
                end = min(start + batch_size, n)
                batch_actions = actions_np[start:end]
                encoded = fast_tokenizer(batch_actions)
                all_tokens.extend(encoded)

            max_len = max(len(t) for t in all_tokens)
            pad_id = fast_vocab_size
            fast_cache = torch.full((n, max_len), fill_value=pad_id, dtype=torch.long)
            for i, tokens in enumerate(all_tokens):
                if len(tokens) > 0:
                    fast_cache[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

            print(f"[PreTransform v3][{split}] FAST tokens: max_len={max_len}, vocab_size={fast_vocab_size}")

        self._action_cache = action_cache
        self._state_cache = state_cache
        self._future_cache = future_cache
        self._fast_cache = fast_cache

        mem_mb = action_cache.nbytes / 1024**2
        if state_cache is not None:
            mem_mb += state_cache.nbytes / 1024**2
        if future_cache is not None:
            mem_mb += future_cache.nbytes / 1024**2
        if fast_cache is not None:
            mem_mb += fast_cache.nbytes / 1024**2

        print(
            f"[PreTransformedActionStateDatasetV3][{split}] "
            f"Cached {n:,} samples, action=({T}, {D}), "
            f"state={'yes' if state_cache is not None else 'no'}, "
            f"future={'yes' if future_cache is not None else 'no'}, "
            f"fast_tokens={'yes' if fast_cache is not None else 'no'}, "
            f"memory={mem_mb:.1f} MB"
        )

    def __len__(self) -> int:
        return self._action_cache.shape[0]

    def __getitem__(self, index: int) -> dict:
        result = {"action": self._action_cache[index]}
        if self._state_cache is not None:
            result["hand_state"] = self._state_cache[index]
        if self._future_cache is not None:
            result["future_hand_states"] = self._future_cache[index]
        if self._fast_cache is not None:
            result["fast_tokens"] = self._fast_cache[index]
        return result


__all__ = [
    "ActionStateDatasetV3",
    "PreTransformedActionStateDatasetV3",
    "ActionStateCollator",
]

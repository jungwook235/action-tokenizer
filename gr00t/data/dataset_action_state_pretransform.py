"""
PreTransformedActionStateDataset: action + state 데이터를 미리 transform하여 메모리에 캐싱.

PreTransformedActionOnlyDataset과 동일한 패턴이지만 추가로:
- hand_state: 현재 hand state (hand_state_dims로 지정된 dim만)
- future_hand_states: 미래 시점의 hand state (hand_pred_future_steps로 설정)
- fast_tokens: FAST tokenizer로 action을 discrete token화한 결과

모든 캐시는 조건부 생성 (필요할 때만).
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from gr00t.data.dataset_action_only import ActionOnlyDataset
from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform import ComposedModalityTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.experiment.data_config import DATA_CONFIG_MAP


class ActionStateDataset(ActionOnlyDataset):
    """Action + State 데이터를 로드하는 Dataset.

    ActionOnlyDataset을 확장하여 state modality도 함께 로드.
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
        video_backend: str = "torchvision_av",
    ):
        self._split = split
        self._val_ratio = val_ratio
        self._val_seed = val_seed

        data_config_cls = DATA_CONFIG_MAP[data_config_name]
        full_modality_configs = data_config_cls.modality_config()
        assert "action" in full_modality_configs, f"data_config '{data_config_name}'에 action modality가 없습니다."

        # action + state modality config
        modality_config = {"action": full_modality_configs["action"]}
        action_keys = full_modality_configs["action"].modality_keys

        state_keys = []
        if "state" in full_modality_configs:
            modality_config["state"] = full_modality_configs["state"]
            state_keys = full_modality_configs["state"].modality_keys

        # Per-key normalization modes / target rotations from data_config.
        # data_config이 해당 속성을 정의했다면 respect — binary (robocasa gripper_close/control_mode),
        # rotation_6d (quat→6d) 등 semantic normalization 이 살아남.
        # 정의 안 되어 있으면 uniform fallback (기존 동작).
        action_normalization_modes = getattr(data_config_cls, "action_normalization_modes", None)
        if not action_normalization_modes:
            action_normalization_modes = {key: normalization_mode for key in action_keys}
        state_normalization_modes = getattr(data_config_cls, "state_normalization_modes", None)
        if not state_normalization_modes:
            state_normalization_modes = {key: normalization_mode for key in state_keys}
        state_target_rotations = getattr(data_config_cls, "state_target_rotations", {}) or {}
        action_target_rotations = getattr(data_config_cls, "action_target_rotations", {}) or {}

        transforms = self._build_action_state_transforms(
            action_keys=action_keys,
            state_keys=state_keys,
            action_normalization_modes=action_normalization_modes,
            state_normalization_modes=state_normalization_modes,
            state_target_rotations=state_target_rotations,
            action_target_rotations=action_target_rotations,
        )

        # ActionOnlyDataset.__init__이 아닌 LeRobotSingleDataset.__init__을 호출
        # (ActionOnlyDataset의 __init__이 action만 설정하므로 직접 부모 호출)
        from gr00t.data.dataset import LeRobotSingleDataset
        LeRobotSingleDataset.__init__(
            self,
            dataset_path=dataset_path,
            modality_configs=modality_config,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            transforms=transforms,
        )

        self._action_keys = action_keys
        self._state_keys = state_keys

    @staticmethod
    def _build_action_state_transforms(
        action_keys: list[str],
        state_keys: list[str],
        action_normalization_modes: dict[str, str],
        state_normalization_modes: dict[str, str],
        state_target_rotations: dict[str, str],
        action_target_rotations: dict[str, str] | None = None,
    ) -> ComposedModalityTransform:
        """Action + State 전용 transform 파이프라인 생성.

        data_config 에 정의된 per-key normalization_modes 와 state_target_rotations
        / action_target_rotations 를 그대로 respect 하므로 tokenizer 학습 시 VLA 와
        동일한 action/state 표현을 학습 타겟으로 삼을 수 있다.

        action_target_rotations 미지정(None/{}) 시 기존 관례(action 에 rotation 변환 미적용)
        그대로 유지 — backwards compat.
        """
        if action_target_rotations is None:
            action_target_rotations = {}

        transform_list = []

        # Action transforms — data_config 가 action_target_rotations 를 정의하면 respect
        # (e.g. bridge_flare_kty_actlat_fm: action.eef_rotation_delta → axis_angle).
        transform_list.extend([
            StateActionToTensor(
                apply_to=action_keys,
                output_dtypes={key: torch.float32 for key in action_keys},
            ),
            StateActionTransform(
                apply_to=action_keys,
                normalization_modes=action_normalization_modes,
                target_rotations=action_target_rotations,
            ),
        ])

        # State transforms (if any)
        if state_keys:
            transform_list.extend([
                StateActionToTensor(
                    apply_to=state_keys,
                    output_dtypes={key: torch.float32 for key in state_keys},
                ),
                StateActionTransform(
                    apply_to=state_keys,
                    normalization_modes=state_normalization_modes,
                    target_rotations=state_target_rotations,
                ),
            ])

        # Concat
        transform_list.append(
            ConcatTransform(
                video_concat_order=[],
                state_concat_order=state_keys if state_keys else None,
                action_concat_order=action_keys,
            ),
        )

        return ComposedModalityTransform(transforms=transform_list)

    def _get_needed_parquet_columns(self) -> list[str]:
        """Parquet 에서 로드할 최소 컬럼 집합.

        `self.modality_keys[modality]` 는 이미 `"action.xxx"` / `"state.xxx"` 형태 full key
        (ModalityConfig.modality_keys 원본 — 접두사 포함) 이므로 그대로 `get_key_meta` 에 넘긴다.
        robocasa 의 경우 {"action", "observation.state"} 2개로 귀결 → image dict 컬럼은 로드 안 함.
        """
        needed: set[str] = set()
        for modality in ("action", "state"):
            if modality not in self.modality_keys:
                continue
            for full_key in self.modality_keys[modality]:
                meta = self._lerobot_modality_meta.get_key_meta(full_key)
                if getattr(meta, "original_key", None):
                    needed.add(meta.original_key)
        return sorted(needed)

    def get_trajectory_data(self, trajectory_id: int):
        """LeRobotSingleDataset.get_trajectory_data override.

        두 가지 문제를 동시에 해결한다:
          1) column projection — 원본은 `pd.read_parquet(path)` 를 불러 image dict 컬럼
             (robocasa 의 observation.images.*) 까지 매번 로드/파싱. 필요한 컬럼만 projection.
          2) cache key 갱신 — 원본은 `self.curr_traj_id` 를 절대 재할당하지 않아
             같은 trajectory 연속 접근에도 cache miss 가 반복된다. 여기서 제대로 세팅.

        정규화 동치성: `meta/stats.json` 기반 통계로 StateActionTransform 이 적용되는 부분은
        전혀 건드리지 않는다. 읽어들이는 action/state ndarray 자체는 byte-by-byte 동일.
        """
        import pandas as pd

        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data

        chunk_index = self.get_episode_chunk(trajectory_id)
        parquet_path = self.dataset_path / self.data_path_pattern.format(
            episode_chunk=chunk_index, episode_index=trajectory_id
        )
        assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"

        needed_cols = self._get_needed_parquet_columns()
        df = pd.read_parquet(parquet_path, columns=needed_cols) if needed_cols else pd.read_parquet(parquet_path)

        self.curr_traj_id = trajectory_id
        self.curr_traj_data = df
        return df

    def __getitem__(self, index: int) -> dict:
        """action + state tensor를 반환."""
        trajectory_id, base_index = self.all_steps[index]
        data = self.get_step_data(trajectory_id, base_index)
        data = self.transforms(data)
        result = {"action": data["action"]}
        if "state" in data:
            result["state"] = data["state"]
        return result


class PreTransformedActionStateDataset(Dataset):
    """초기화 시 action + state 데이터를 미리 transform하여 메모리에 캐싱하는 Dataset.

    Args:
        dataset_path: 데이터셋 경로
        data_config_name: DATA_CONFIG_MAP의 키
        embodiment_tag: 로봇 태그
        split: "train", "val", 또는 "all"
        val_ratio: validation 에피소드 비율
        val_seed: split seed
        normalization_mode: 정규화 방식
        hand_state_dims: state에서 hand에 해당하는 dim indices (None이면 hand_state 캐싱 안함)
        hand_pred_future_steps: 미래 예측할 step 간격 리스트 (e.g., [8, 16])
        cache_fast_tokens: FAST tokenizer로 action을 tokenize하여 캐싱할지
        fast_tokenizer_path: FAST tokenizer 경로
        fast_vocab_size: FAST tokenizer vocab size (pad token으로 사용)
        max_trajectories: train/val split 이후 사용할 에피소드 상한 (None이면 전부)
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
    ):
        need_state = hand_state_dims is not None and len(hand_state_dims) > 0
        need_future = need_state and hand_pred_future_steps is not None and len(hand_pred_future_steps) > 0

        # Robocasa 데이터셋 보정: 기존 sbatch 는 --hand-state-dims 0..15 (16개) 로 들어오지만,
        # data_config 가 state_target   _rotations (quat→rotation_6d) 를 respect 하도록 바뀌면
        # 실제 state_dim 이 20 이 된다 (3 + 6 + 2 + 3 + 6). 토크나이저가 VLA 와 동일한 state 표현을
        # aux loss 로 학습하도록 hand_state_dims 를 0..19 로 확장.
        _ROBOCASA_DATA_CONFIGS = {"single_panda_gripper", "single_panda_gripper_actlat_fm"}
        if need_state and data_config_name in _ROBOCASA_DATA_CONFIGS:
            _robocasa_state_dim = 20
            if len(hand_state_dims) < _robocasa_state_dim:
                old_len = len(hand_state_dims)
                hand_state_dims = list(range(_robocasa_state_dim))
                print(
                    f"[PreTransformedActionStateDataset] Robocasa data_config "
                    f"'{data_config_name}' detected: hand_state_dims auto-expanded "
                    f"{old_len} → {_robocasa_state_dim} "
                    f"(state_target_rotations 로 quat→rotation_6d 변환 후 state_dim=20)"
                )

        # Source dataset 생성
        source = ActionStateDataset(
            dataset_path=dataset_path,
            data_config_name=data_config_name,
            embodiment_tag=embodiment_tag,
            split=split,
            val_ratio=val_ratio,
            val_seed=val_seed,
            normalization_mode=normalization_mode,
        )

        if max_trajectories is not None and max_trajectories > 0 and len(source.trajectory_ids) > max_trajectories:
            old_n_traj = len(source.trajectory_ids)
            source._trajectory_ids = source.trajectory_ids[:max_trajectories]
            source._trajectory_lengths = source.trajectory_lengths[:max_trajectories]
            source._all_steps = source._get_all_steps()
            print(
                f"[PreTransform][{split}] max_trajectories={max_trajectories} "
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
            first_state = first["state"]  # [obs_count, state_dim] or [state_dim]
            if first_state.ndim == 2:
                # observation_indices=[0] → 첫 번째 timestep만
                full_state_dim = first_state.shape[-1]
            else:
                full_state_dim = first_state.shape[0]
            hand_dim = len(hand_state_dims)
            state_cache = torch.empty(n, hand_dim, dtype=torch.float32)

        # --- Future state cache 준비 ---
        future_cache = None
        num_future = 0
        if need_future and has_state:
            num_future = len(hand_pred_future_steps)
            hand_dim = len(hand_state_dims)
            future_cache = torch.empty(n, num_future, hand_dim, dtype=torch.float32)

        # --- Main caching loop (current + future 를 하나의 pass 로 처리) ---
        # future 가 활성화된 경우 traj_id→traj_len 을 dict 로 precompute:
        #   원본 future 루프는 매 i 마다 `source.get_trajectory_index(traj_id)` 를 불러
        #   np.where 로 O(N_traj) 선형 탐색을 수행했는데, 그걸 O(1) lookup 으로 대체.
        # 또한 원본은 future 를 별도 루프로 돌려 parquet cache (curr_traj_data) 가
        # main 루프 종료 시점의 마지막 traj 로 고정돼 있어 future 루프 시작 시 전 traj 를
        # 전부 다시 로드했지만, 합치면 traj 당 parquet 1회로 충분.
        # 동치성: 각 i 에 대해 원본과 동일한 (traj_id, base_idx, step_offset) 조합으로
        #   `source.get_step_data` + `source.transforms` 를 호출하므로 캐시되는 값은 bit-identical.
        if future_cache is not None:
            traj_len_map = dict(
                zip(source.trajectory_ids.tolist(), source.trajectory_lengths.tolist())
            )
            loop_desc = f"[PreTransform][{split}] Caching action+state+future"
        else:
            traj_len_map = None
            loop_desc = f"[PreTransform][{split}] Caching action+state"

        for i in tqdm(range(n), desc=loop_desc):
            if i == 0:
                sample = first
            else:
                sample = source[i]

            action_cache[i] = sample["action"]

            if state_cache is not None:
                state_tensor = sample["state"]
                if state_tensor.ndim == 2:
                    state_tensor = state_tensor[0]  # 첫 번째 observation timestep
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
            print(f"[PreTransform][{split}] Tokenizing actions with FAST tokenizer...")
            from transformers import AutoProcessor
            fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)

            # Batch process
            actions_np = action_cache.numpy()  # [N, T, D]
            batch_size = 1024
            all_tokens = []

            for start in tqdm(range(0, n, batch_size), desc=f"[FAST tokenize][{split}]"):
                end = min(start + batch_size, n)
                batch_actions = actions_np[start:end]
                encoded = fast_tokenizer(batch_actions)
                all_tokens.extend(encoded)

            # Pad to max length
            max_len = max(len(t) for t in all_tokens)
            pad_id = fast_vocab_size  # pad token = vocab_size
            fast_cache = torch.full((n, max_len), fill_value=pad_id, dtype=torch.long)
            for i, tokens in enumerate(all_tokens):
                if len(tokens) > 0:
                    fast_cache[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)

            print(f"[PreTransform][{split}] FAST tokens: max_len={max_len}, vocab_size={fast_vocab_size}")

        # --- Store caches ---
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
            f"[PreTransformedActionStateDataset][{split}] "
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


class ActionStateCollator:
    """Action + state 관련 텐서를 batch로 묶는 collator."""

    def __call__(self, features: list[dict]) -> dict:
        batch = {}
        keys = features[0].keys()
        for key in keys:
            batch[key] = torch.stack([f[key] for f in features])
        return batch

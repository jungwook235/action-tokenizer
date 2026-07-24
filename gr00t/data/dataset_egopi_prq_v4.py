"""EgoPi {p,r,q} 15D action variant of the cached V4 tokenizer dataset.

Replicates the EgoPi cotrain action preprocessing (RLDX-1-egopi, egopi_cotrain.sh)
inside the Stage-1 V4 tokenizer pipeline, so BOTH the openarm robot data and the
rlwrld human pnp data land in ONE robot-anchored 15D {p,r,q} action space and can
share a SINGLE action encoder/decoder (one embodiment group):

  * robot (openarm_teleop_v3, 28D joints): {p,r} read from the precomputed openarm
    FK cache (per-object h5 built by RLDX-1-egopi/scripts/build_egopi_cache.py —
    ACTION columns, i.e. FK of the commanded joints), q = raw action[22:28] inspire
    joints. Episodes filtered by the per-episode left-arm gate (egopi_filter.json,
    ~208 eps kept), applied AFTER the fixed-val split so splits stay stable.
  * human (pnp_clean_260506, 30D eef_inspire): {p,r,q} via
    ``egopi_prq_mapping.human_to_prq`` over the right-hand slots
    (p - P_OFFSET, R_CONV @ R, q identity). All episodes kept.

Layout: [ p(3) | rot6d(6) | q(6) ] — identical to EgoPi (PRQ_DIM = 15).

Normalization is EgoPi's, NOT the LeRobot pipeline's: per-sub-key min-max to
[-1, 1] (``normalize_values_minmax`` formula) with the robot+human MERGED stats
(elementwise min-of-mins / max-of-maxs across the two source buckets of
egopi_prq_stats.json — exactly what StandardMixtureDataset.merge_statistics
produces for min/max under the shared EmbodimentTag.EGOPI), then clipped to
[-1, 1] (their processor default clip_outliers=True).

Frames/DINO features are untouched: this class subclasses
``CachedActionFramesDatasetV4`` and reuses its DINO-cache reader; the LeRobot
action transform output is simply ignored and replaced with the prq chunk.
Raw actions are read straight from the episode parquet (like EgoPi's
lerobot_episode_loader), converted per episode, and chunked with the same
first/last padding semantics as ``retrieve_data_and_pad`` (end rows repeat the
last row — base_index >= 0 so front padding never occurs).
"""

import json
from pathlib import Path
from typing import Literal, Optional

import h5py
import numpy as np
import pandas as pd
import torch

from gr00t.data import egopi_prq_mapping as _prq
from gr00t.data.dataset_dino_cache_v4 import CachedActionFramesDatasetV4

# robot openarm right hand joints in action(28) — same slice EgoPi uses on state.
_HAND_R = slice(22, 28)
_PRQ_KEYS = ("wrist_pos", "wrist_rot6d", "hand_q")  # concat order == 15D layout


def load_merged_prq_action_minmax(prq_stats_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Merged (robot ∪ human) per-dim action min/max, concatenated to (15,).

    Mirrors merge_statistics: global min/max across the per-source buckets.
    """
    with open(prq_stats_path, "r") as f:
        stats = json.load(f)
    mins, maxs = [], []
    for key in _PRQ_KEYS:
        per_src_min = [np.asarray(stats[src]["action"][key]["min"], np.float64) for src in ("robot", "human")]
        per_src_max = [np.asarray(stats[src]["action"][key]["max"], np.float64) for src in ("robot", "human")]
        mins.append(np.minimum.reduce(per_src_min))
        maxs.append(np.maximum.reduce(per_src_max))
    return np.concatenate(mins), np.concatenate(maxs)


def normalize_minmax(values: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    """EgoPi's normalize_values_minmax: linear map [min, max] → [-1, 1]; degenerate dims → 0."""
    normalized = np.zeros_like(values)
    mask = ~np.isclose(max_vals, min_vals)
    normalized[..., mask] = (values[..., mask] - min_vals[..., mask]) / (
        max_vals[..., mask] - min_vals[..., mask]
    )
    normalized[..., mask] = 2 * normalized[..., mask] - 1
    return normalized


class EgoPiPrqCachedDatasetV4(CachedActionFramesDatasetV4):
    """Cached V4 dataset whose action is the EgoPi 15D {p,r,q} chunk.

    Args (beyond CachedActionFramesDatasetV4):
        prq_mode: "robot" (openarm joints + FK cache) or "human" (eef_inspire 30D).
        prq_stats_path: egopi_prq_stats.json (robot/human buckets; merged here).
        fk_cache_h5: robot only — per-object FK cache h5 (ep_%06d groups with
            action_wrist_pos_R / action_wrist_rot_R).
        filter_json / filter_tag: robot only — egopi_filter.json + object tag
            (e.g. "bottle"); episodes with keep=false are dropped.
    """

    def __init__(
        self,
        *,
        prq_mode: Literal["robot", "human"],
        prq_stats_path: str | Path,
        fk_cache_h5: Optional[str | Path] = None,
        filter_json: Optional[str | Path] = None,
        filter_tag: Optional[str] = None,
        **kwargs,
    ):
        assert prq_mode in ("robot", "human"), f"prq_mode must be robot/human: {prq_mode}"
        if prq_mode == "robot":
            assert fk_cache_h5 is not None, "robot prq_mode requires fk_cache_h5"
            assert filter_json is not None and filter_tag is not None, (
                "robot prq_mode requires filter_json + filter_tag (left-arm gate)"
            )
        self._prq_mode = prq_mode
        self._fk_path = str(fk_cache_h5) if fk_cache_h5 is not None else None
        self._fk = None  # lazy per worker (h5py handles are not fork-safe)

        self._keep: Optional[set[int]] = None
        if prq_mode == "robot":
            with open(filter_json, "r") as f:
                filt = json.load(f)
            assert filter_tag in filt, f"filter_tag {filter_tag!r} not in {filter_json}"
            self._keep = {int(ep) for ep, v in filt[filter_tag].items() if v.get("keep")}

        self._prq_min, self._prq_max = load_merged_prq_action_minmax(prq_stats_path)
        self._prq_cache: dict[int, np.ndarray] = {}  # per-episode converted [L, 15]

        super().__init__(**kwargs)

    # ── episode filter (applied AFTER the fixed-val split → splits stay stable) ──
    def _get_trajectories(self):
        ids, lengths = super()._get_trajectories()
        if self._keep is None:
            return ids, lengths
        mask = np.array([int(i) in self._keep for i in ids])
        kept, kept_len = ids[mask], lengths[mask]
        print(
            f"[EgoPiPrq][{self._prq_mode}] {Path(self._dataset_path).name}: "
            f"left-arm gate {len(ids)} → {len(kept)} episodes"
        )
        if len(kept) == 0:
            print(
                f"[EgoPiPrq][WARN] {Path(self._dataset_path).name} split={self._split}: "
                f"0 episodes after filter (val episode may be gated out)"
            )
        return kept, kept_len

    # ── raw episode action → prq [L, 15] ──
    def _raw_episode_action(self, episode_id: int) -> np.ndarray:
        parquet = (
            self._dataset_path
            / f"data/chunk-{episode_id // 1000:03d}/episode_{episode_id:06d}.parquet"
        )
        df = pd.read_parquet(parquet, columns=["action"])
        return np.stack(df["action"].to_numpy())  # [L, 28|30]

    def _fk_group(self, episode_id: int):
        if self._fk is None:
            self._fk = h5py.File(self._fk_path, "r")
        return self._fk[f"ep_{int(episode_id):06d}"]

    def _episode_prq(self, episode_id: int) -> np.ndarray:
        cached = self._prq_cache.get(episode_id)
        if cached is not None:
            return cached
        action = self._raw_episode_action(episode_id)
        if self._prq_mode == "robot":
            g = self._fk_group(episode_id)
            awp, awr = g["action_wrist_pos_R"][...], g["action_wrist_rot_R"][...]
            T = min(len(action), len(awp))
            prq = np.stack(
                [_prq.robot_cache_to_prq(awp[t], awr[t], action[t, _HAND_R]) for t in range(T)]
            )
        else:
            prq = np.stack([_prq.human_to_prq(action[t]) for t in range(len(action))])
        self._prq_cache[episode_id] = prq
        return prq

    # ── sample: prq action chunk (normalized) + cached DINO feats ──
    def __getitem__(self, index: int) -> dict:
        trajectory_id, base_index = self.all_steps[index]
        trajectory_id, base_index = int(trajectory_id), int(base_index)

        prq = self._episode_prq(trajectory_id)  # [L, 15]
        L = prq.shape[0]
        idx = np.clip(np.arange(base_index, base_index + self._action_horizon), 0, L - 1)
        chunk = prq[idx].astype(np.float64)  # [T, 15]; end rows repeat last (first_last pad)

        chunk = normalize_minmax(chunk, self._prq_min, self._prq_max)
        chunk = np.clip(chunk, -1.0, 1.0)  # EgoPi processor clip_outliers=True

        x0_feat, x1_feat = self._reader.get_pair(trajectory_id, base_index)
        return {
            "action": torch.from_numpy(chunk.astype(np.float32)),  # [T, 15]
            "x0_feat": x0_feat,
            "x1_feat": x1_feat,
        }

"""Stage-2 VLA V4 cached dataset whose action is the EgoPi 15D {p,r,q} chunk.

Pairs with the ``openarm_prq`` embodiment of the finetuned soupv1 V4 tokenizer
(``recon_dino_bn64_l1_mse_naiveln_vae_embtok_finetune_openarm_prq_400k.sh``).
The tokenizer was trained (Stage-1, ``EgoPiPrqCachedDatasetV4``) on FK-converted
15D [p(3)|rot6d(6)|q(6)] actions normalized with the MERGED robot∪human EgoPi
min-max stats — NOT on the LeRobot pipeline's 28D joint actions. Stage-2 must
feed the tokenizer the exact same action representation, so this dataset:

  * produces the ordinary actlat_fm V4 cached sample (state / eagle video /
    cached x0_feat, x1_feat) UNCHANGED via the parent class, then
  * REPLACES ``item["action"]`` with the per-episode FK-cache-converted prq
    chunk [T, 15], normalized/clipped exactly like Stage-1
    (``normalize_minmax`` on the merged stats, clip to [-1, 1], end rows repeat
    the last row), and
  * applies the same per-episode left-arm gate (egopi_filter.json) AFTER the
    fixed-val split, so train/val episodes match Stage-1 exactly.

Robot-only (openarm_teleop_v3): the fk cache h5 and filter tag are derived from
the dataset directory name (bottle/cup/doll/snack), matching the per-object
layout of the Stage-1 embodiments config. The human pnp source is not used for
VLA training.

NOTE: the replaced action is consumed ONLY by the frozen tokenizer
(``get_latent_target``) — the VLA action head operates on the latent tokens, so
the [T, 15] shape never reaches the flow-matching head. The transform
pipeline's own (28D joint) action output is discarded.
"""

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from gr00t.data import egopi_prq_mapping as _prq
from gr00t.data.dataset_actlat_fm_v4_cached import LeRobotSingleDatasetActlatFMV4Cached
from gr00t.data.dataset_egopi_prq_v4 import (
    load_merged_prq_action_minmax,
    normalize_minmax,
)

# robot openarm right hand joints in action(28) — same slice Stage-1 uses.
_HAND_R = slice(22, 28)


class LeRobotSingleDatasetActlatFMV4CachedPrq(LeRobotSingleDatasetActlatFMV4Cached):
    """actlat_fm V4 cached dataset with the action replaced by the EgoPi prq chunk.

    Extra args (on top of ``LeRobotSingleDatasetActlatFMV4Cached``):
        prq_stats_path: egopi_prq_stats.json (robot/human buckets; merged here —
            identical to Stage-1).
        fk_cache_dir: directory of per-object FK cache h5 files; the file used is
            ``<fk_cache_dir>/<dataset_dir_name>.h5`` (ep_%06d groups with
            action_wrist_pos_R / action_wrist_rot_R).
        filter_json: egopi_filter.json; the tag used is the dataset dir name.
            Episodes with keep=false are dropped AFTER the fixed-val split.
    """

    def __init__(
        self,
        *args,
        prq_stats_path: str | Path,
        fk_cache_dir: str | Path,
        filter_json: str | Path,
        **kwargs,
    ):
        dataset_path = kwargs.get("dataset_path") or args[0]
        tag = Path(dataset_path).name
        fk_path = Path(fk_cache_dir) / f"{tag}.h5"
        assert fk_path.exists(), f"FK cache not found: {fk_path}"

        with open(filter_json, "r") as f:
            filt = json.load(f)
        assert tag in filt, f"filter tag {tag!r} not in {filter_json}"
        # set BEFORE super().__init__ — _get_trajectories runs inside it.
        self._keep: set[int] = {int(ep) for ep, v in filt[tag].items() if v.get("keep")}

        self._fk_path = str(fk_path)
        self._fk = None  # lazy per worker (h5py handles are not fork-safe)
        self._prq_min, self._prq_max = load_merged_prq_action_minmax(prq_stats_path)
        self._prq_cache: dict[int, np.ndarray] = {}

        super().__init__(*args, **kwargs)

    # ── left-arm gate (AFTER the fixed-val split → same episodes as Stage-1) ──
    def _get_trajectories(self):
        ids, lengths = super()._get_trajectories()
        mask = np.array([int(i) in self._keep for i in ids])
        kept, kept_len = ids[mask], lengths[mask]
        print(
            f"[ActlatFMPrq][{self.split}] {Path(self._dataset_path).name}: "
            f"left-arm gate {len(ids)} → {len(kept)} episodes"
        )
        # cup/doll: the single fixed-val episode is gated out → 0 val episodes.
        # Only the train split must be non-empty; the Stage-2 launcher skips
        # empty val datasets when building the eval mixture.
        assert self.split == "val" or len(kept) > 0, (
            f"{self._dataset_path} split={self.split}: 0 episodes after left-arm gate"
        )
        return kept, kept_len

    # ── raw episode action → prq [L, 15] (identical to Stage-1) ──
    def _raw_episode_action(self, episode_id: int) -> np.ndarray:
        parquet = (
            self._dataset_path
            / f"data/chunk-{episode_id // 1000:03d}/episode_{episode_id:06d}.parquet"
        )
        df = pd.read_parquet(parquet, columns=["action"])
        return np.stack(df["action"].to_numpy())  # [L, 28]

    def _fk_group(self, episode_id: int):
        if self._fk is None:
            self._fk = h5py.File(self._fk_path, "r")
        return self._fk[f"ep_{int(episode_id):06d}"]

    def _episode_prq(self, episode_id: int) -> np.ndarray:
        cached = self._prq_cache.get(episode_id)
        if cached is not None:
            return cached
        action = self._raw_episode_action(episode_id)
        g = self._fk_group(episode_id)
        awp, awr = g["action_wrist_pos_R"][...], g["action_wrist_rot_R"][...]
        T = min(len(action), len(awp))
        prq = np.stack(
            [_prq.robot_cache_to_prq(awp[t], awr[t], action[t, _HAND_R]) for t in range(T)]
        )
        self._prq_cache[episode_id] = prq
        return prq

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)  # normal cached V4 sample (x0_feat/x1_feat attached)

        trajectory_id, base_index = self.all_steps[index]
        trajectory_id, base_index = int(trajectory_id), int(base_index)

        prq = self._episode_prq(trajectory_id)  # [L, 15]
        L = prq.shape[0]
        idx = np.clip(np.arange(base_index, base_index + self._frame_action_horizon), 0, L - 1)
        chunk = prq[idx].astype(np.float64)  # [T, 15]; end rows repeat last

        chunk = normalize_minmax(chunk, self._prq_min, self._prq_max)
        chunk = np.clip(chunk, -1.0, 1.0)

        # Replace the LeRobot-pipeline (28D joint) action with the tokenizer's
        # prq action. Keep the container type the collator expects (np.stack).
        item["action"] = chunk.astype(np.float32)
        return item

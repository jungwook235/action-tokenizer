"""Cached-DINO variant of the Stage-2 VLA V4 dataset.

Drop-in replacement for ``LeRobotSingleDatasetActlatFMV4`` when a precomputed
DINO feature cache exists. The VLA backbone sample (state / eagle video / action)
is produced UNCHANGED by the actlat_fm base dataset; the only difference is that
instead of decoding the (frame_x0, frame_x1) pair and letting the model run DINO
on it, we attach the precomputed ``x0_feat`` / ``x1_feat`` straight from the cache.

The cache is the SAME one built for Stage-1 (the tokenizer was trained on those
exact features and the Stage-2 frame preprocessing byte-matches Stage-1), keyed by
``(model, final_norm, image_size, camera)`` — so no rebuild is needed between
stages.

Wiring: the actlat_fm collator (``_collate_actlat_fm``) stacks any unknown key
generically, so ``x0_feat`` / ``x1_feat`` reach the model inputs with no collator
change. The model forward then passes them to ``get_latent_target(..., x0_feat=,
x1_feat=)``, whose wrapper already prefers precomputed feats over raw frames
(``_resolve_dino_feats``). When the cache is not used these keys are simply absent
and the existing raw-frame path is untouched.
"""

from pathlib import Path
from typing import Optional

from gr00t.data.dataset_actlat_fm import LeRobotSingleDatasetActlatFM
from gr00t.data.dataset_actlat_fm_v4 import LeRobotSingleDatasetActlatFMV4
from gr00t.data.dino_feature_cache import DinoFeatureCacheReader, make_cache_key


class LeRobotSingleDatasetActlatFMV4Cached(LeRobotSingleDatasetActlatFMV4):
    """actlat_fm V4 dataset that yields cached (x0_feat, x1_feat) instead of frames.

    Extra args (on top of ``LeRobotSingleDatasetActlatFMV4``):
        feature_source / dino_model / dino_final_norm: must match the values used
            by ``scripts/precompute_dino_features.py`` (and the Stage-1 tokenizer),
            so the cache key resolves to the same directory. Validated against the
            cache's ``meta.json`` at construction.

    The cache key's ``image_size`` and ``camera`` come from the parent's
    ``frame_image_size`` / ``frame_video_key``.
    """

    def __init__(
        self,
        *args,
        feature_source: str = "dino",
        dino_model: str = "facebook/dinov2-large",
        dino_final_norm: str = "naive",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        key = make_cache_key(
            feature_source=feature_source,
            model_name=dino_model,
            final_norm=dino_final_norm,
            image_size=self._frame_image_size,
            video_key=self._frame_video_key,
        )
        self._reader = DinoFeatureCacheReader(
            self.dataset_path,
            key,
            action_horizon=self._frame_action_horizon,
            expect={
                "feature_source": feature_source,
                "model_name": dino_model,
                "final_norm": dino_final_norm,
                "image_size": self._frame_image_size,
                "video_key": self._frame_video_key,
            },
        )

    def __getitem__(self, index: int) -> dict:
        # Grandparent __getitem__ = the normal actlat_fm sample WITHOUT the V4
        # frame-pair decode (skipped — we use the cache instead).
        item = LeRobotSingleDatasetActlatFM.__getitem__(self, index)
        trajectory_id, base_index = self.all_steps[index]
        x0_feat, x1_feat = self._reader.get_pair(int(trajectory_id), int(base_index))
        item["x0_feat"] = x0_feat  # [Lp, C] (cache dtype); collator stacks generically
        item["x1_feat"] = x1_feat
        return item

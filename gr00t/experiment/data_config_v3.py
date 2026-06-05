"""V3 data configs that switch action normalization from min_max to q01/q99.

Adds two registry entries (no modification to ``data_config.py``):

* ``fourier_gr1_arms_waist_q99`` — Stage 1 tokenizer training (action-only).
* ``fourier_gr1_arms_waist_actlat_fm_q99`` — Stage 2 VLA training.

Both classes inherit their parent's full transform pipeline and override only
the action ``normalization_modes`` to ``q99``. State transforms (sin-cos /
rotation_6d) are inherited unchanged — q99 only applies to actions.

Importing this module triggers the registration into ``DATA_CONFIG_MAP``.
"""

from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.experiment.data_config import (
    DATA_CONFIG_MAP,
    FourierGr1ArmsWaistActlatFMDataConfig,
    FourierGr1ArmsWaistDataConfig,
)
from gr00t.model.transforms import GR00TInferTransform


_GR1_ARMS_WAIST_ACTION_KEYS = (
    "action.left_arm",
    "action.right_arm",
    "action.left_hand",
    "action.right_hand",
    "action.waist",
)


class FourierGr1ArmsWaistQ99DataConfig(FourierGr1ArmsWaistDataConfig):
    """``FourierGr1ArmsWaistDataConfig`` with q99 action normalization.

    The parent class does not declare ``action_normalization_modes`` at class
    level, so we add it here so :class:`ActionOnlyDataset` /
    :class:`ActionStateDataset` (which respect the attribute) pick up q99.
    """

    action_normalization_modes = {key: "q99" for key in _GR1_ARMS_WAIST_ACTION_KEYS}

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(
                apply_to=self.video_keys, height=224, width=224, interpolation="linear"
            ),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms (sin-cos, unchanged from parent)
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms with q99 instead of min_max
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys},
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TInferTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class FourierGr1ArmsWaistActlatFMQ99DataConfig(FourierGr1ArmsWaistActlatFMDataConfig):
    """``FourierGr1ArmsWaistActlatFMDataConfig`` with q99 action normalization.

    Mirrors the parent's transform pipeline except for action normalization.
    Used by stage 2 VLA training so the policy sees the same q99-normalized
    actions the v3 tokenizer was trained on.
    """

    action_normalization_modes = {key: "q99" for key in _GR1_ARMS_WAIST_ACTION_KEYS}

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(
                apply_to=self.video_keys, height=224, width=224, interpolation="linear"
            ),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys},
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TInferTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=29,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


# Registry side-effect — importing this module makes the keys visible to
# anything that consults ``DATA_CONFIG_MAP`` afterwards (training scripts,
# datasets, tyro Literal autocompletion if Literal is built lazily).
DATA_CONFIG_MAP["fourier_gr1_arms_waist_q99"] = FourierGr1ArmsWaistQ99DataConfig()
DATA_CONFIG_MAP["fourier_gr1_arms_waist_actlat_fm_q99"] = (
    FourierGr1ArmsWaistActlatFMQ99DataConfig()
)

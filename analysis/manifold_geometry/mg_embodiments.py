"""Embodiment registry for the manifold-geometry analysis.

Self-contained restatement of the action layouts used by the M5 scripts in
``Isaac-GR00T/experiments/analysis/latent_vs_raw_dexterous/`` (those files are
NOT imported or modified). Each entry gives, for one embodiment:

    paths   : list of lerobot dataset roots to pool (one entry per task)
    raw_dim : width of the raw parquet ``action`` column
    groups  : {group_name: [column indices into the raw action]}

Group slices come from each data-config's ``action_keys`` crossed with the
dataset's ``meta/modality.json``, exactly as in the M5 scripts:

  gr1_tabletop    fourier_gr1_arms_waist            raw 44, trained 29
  dexjoco_single  DexJoCoSingleArmMultiHorizon      raw 22
  dexjoco_dual    dexjoco_dual_arm                  raw 44
  robocasa_mg     SinglePandaGripper                raw 12
  bridge          BridgeFlareKTY                    raw  7
"""

import glob
import os


def _rng(a, b):
    return list(range(a, b))


# --------------------------------------------------------------------- gr1
_GR1_LEFT_ARM = _rng(0, 7)
_GR1_LEFT_HAND = _rng(7, 13)
_GR1_RIGHT_ARM = _rng(22, 29)
_GR1_RIGHT_HAND = _rng(29, 35)
_GR1_WAIST = _rng(41, 44)

_GR1_GROUPS = {
    "arm": _GR1_LEFT_ARM + _GR1_RIGHT_ARM,            # 14
    "hand": _GR1_LEFT_HAND + _GR1_RIGHT_HAND,         # 12 dexterous
    "waist": _GR1_WAIST,                              # 3
    "all_trained": (_GR1_LEFT_ARM + _GR1_RIGHT_ARM
                    + _GR1_LEFT_HAND + _GR1_RIGHT_HAND
                    + _GR1_WAIST),                    # 29
}

GR1 = {
    "label": "GR1 TableTop (100demo)",
    "paths": ["/storage1/sjw_dataset/dataset/robocasa_gr1_tabletop/sim_100demos"],
    "raw_dim": 44,
    "groups": dict(_GR1_GROUPS),
    "dexterous_group": "hand",
    "arm_group": "arm",
}

# The 24-task gr1_unified mixture (1000 demos/task) that the action-tokenizer
# effective-dim work uses. Same embodiment and same fourier_gr1_arms_waist
# layout as GR1 above, but a different, much larger and more diverse dataset --
# kept separate so the two are never silently pooled or compared as one.
_GR1U_ROOT = ("/storage1/sjw_dataset/dataset/"
              "PhysicalAI-Robotics-GR00T-X-Embodiment-Sim")

GR1_UNIFIED_1000 = {
    "label": "GR1 unified (24 tasks x 1000demo)",
    "paths": sorted(glob.glob(os.path.join(_GR1U_ROOT, "gr1_unified.*"))),
    "raw_dim": 44,
    "groups": dict(_GR1_GROUPS),
    "dexterous_group": "hand",
    "arm_group": "arm",
}

# ---------------------------------------------------------- dexjoco single
# NOTE: the original M5 script pointed at /sjw_alinlab2/home/hojin2/.cache/...,
# which no longer exists. Same v20 tasks live under the jungwook dataset root.
_DJ_BASE = "/sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20"
_DJ_TASKS = ["click_mouse", "fold_glasses", "hammer_nail",
             "pick_bucket", "water_plant", "pinch_tongs"]

DEXJOCO_SINGLE = {
    "label": "DexJoCo single-arm",
    "paths": [os.path.join(_DJ_BASE, t) for t in _DJ_TASKS],
    "raw_dim": 22,
    "groups": {
        "arm_pos": _rng(0, 3),        # 3
        "arm_rot": _rng(3, 6),        # 3 rotvec
        "arm_posrot": _rng(0, 6),     # 6
        "hand": _rng(6, 22),          # 16 dexterous
        "all_trained": _rng(0, 22),   # 22
    },
    "dexterous_group": "hand",
    "arm_group": "arm_posrot",
}

# ------------------------------------------------------------ dexjoco dual
_DJD_BASE = "/sjw_alinlab1/home/jungwook/dataset/dexjoco_lerobot/v20/bimanual"
_DJD_TASKS = ["bimanual_assembly", "bimanual_hanoi", "bimanual_microwave_cook",
              "bimanual_photograph", "bimanual_unlock_ipad"]
_DJD_R_ARM, _DJD_R_HAND = _rng(0, 6), _rng(6, 22)
_DJD_L_ARM, _DJD_L_HAND = _rng(22, 28), _rng(28, 44)

DEXJOCO_DUAL = {
    "label": "DexJoCo dual-arm",
    "paths": [os.path.join(_DJD_BASE, t) for t in _DJD_TASKS],
    "raw_dim": 44,
    "groups": {
        "right_arm": _DJD_R_ARM,                      # 6
        "left_arm": _DJD_L_ARM,                       # 6
        "arm": _DJD_R_ARM + _DJD_L_ARM,               # 12
        "right_hand": _DJD_R_HAND,                    # 16
        "left_hand": _DJD_L_HAND,                     # 16
        "hand": _DJD_R_HAND + _DJD_L_HAND,            # 32 dexterous
        "all_trained": _rng(0, 44),                   # 44
    },
    "dexterous_group": "hand",
    "arm_group": "arm",
}

# ------------------------------------------------------------- robocasa mg
ROBOCASA_MG = {
    "label": "Robocasa kitchen (Panda)",
    "paths": ["/storage1/sjw_dataset/dataset/huggingface/lerobot/shared/"
              "robocasa_preprocessed/robocasa_mg_gr00t_100"],
    "raw_dim": 12,
    "groups": {
        "base_motion": _rng(0, 4),                    # 4 (constant in this subset)
        "control_mode": _rng(4, 5),                   # 1 binary
        "eef_position": _rng(5, 8),                   # 3
        "eef_rotation": _rng(8, 11),                  # 3 axis-angle
        "gripper_close": _rng(11, 12),                # 1 binary
        "eef_posrot": _rng(5, 11),                    # 6
        "all_trained": _rng(0, 12),                   # 12
    },
    "dexterous_group": None,
    "arm_group": "eef_posrot",
}

# ----------------------------------------------------------------- bridge
BRIDGE = {
    "label": "Bridge (WidowX)",
    "paths": ["/storage1/sjw_dataset/dataset/huggingface/lerobot/shared/bridge_orig_lerobot"],
    "raw_dim": 7,
    "groups": {
        "eef_position": _rng(0, 3),                   # 3
        "eef_rotation": _rng(3, 6),                   # 3 euler rpy
        "gripper": _rng(6, 7),                        # 1
        "eef_posrot": _rng(0, 6),                     # 6
        "all_trained": _rng(0, 7),                    # 7
    },
    "dexterous_group": None,
    "arm_group": "eef_posrot",
}

EMBODIMENTS = {
    "gr1_tabletop": GR1,
    "gr1_unified_1000": GR1_UNIFIED_1000,
    "dexjoco_single": DEXJOCO_SINGLE,
    "dexjoco_dual": DEXJOCO_DUAL,
    "robocasa_mg": ROBOCASA_MG,
    "bridge": BRIDGE,
}

# Order used by the cross-embodiment charts (ascending dexterity).
EMBODIMENT_ORDER = ["bridge", "robocasa_mg", "gr1_tabletop", "gr1_unified_1000",
                    "dexjoco_single", "dexjoco_dual"]


def episode_files(paths):
    """All episode parquet paths under the given dataset roots, sorted per root."""
    out = []
    for p in paths:
        files = sorted(glob.glob(os.path.join(p, "data", "**", "*.parquet"),
                                 recursive=True))
        out.append((p, files))
    return out

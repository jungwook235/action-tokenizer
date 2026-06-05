"""
configs.py

Defines per-dataset configuration (kwargs) for each dataset in Open-X Embodiment.

Configuration adopts the following structure:
    image_obs_keys:
        primary: primary external RGB
        secondary: secondary external RGB
        wrist: wrist RGB

    depth_obs_keys:
        primary: primary external depth
        secondary: secondary external depth
        wrist: wrist depth

    # Always 8-dim =>> changes based on `StateEncoding`
    state_obs_keys:
        StateEncoding.POS_EULER:    EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) + Gripper Open/Close (1)
        StateEncoding.POS_QUAT:     EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
        StateEncoding.JOINT:        Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)

    state_encoding: Type of `StateEncoding`
    action_encoding: Type of action encoding (e.g., EEF Position vs. Joint Position)
"""

from enum import IntEnum

from gr00t.data.rlds.oxe.utils.droid_utils import zero_action_filter
from typing import Tuple
from gr00t.data.schema import StateActionMetadata, RotationType

# Defines Proprioceptive State Encoding Schemes
class StateEncoding(IntEnum):
    # fmt: off
    NONE = -1               # No Proprioceptive State
    POS_EULER = 1           # EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) + Gripper Open/Close (1)
    POS_QUAT = 2            # EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
    JOINT = 3               # Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)
    JOINT_BIMANUAL = 4      # Joint Angles (2 x [ Joint Angles (6) + Gripper Open/Close (1) ])
    AGIBOT_DEXHAND = 5      # AGIBOT DexHand State 
    AGIBOT_GRIPPER = 6      # AGIBOT Gripper State
    GALAXEA = 7             # GALAXEA State
    HUMANOID_EVERYDAY_G1 = 8   # HUMANOID Everyday State
    HUMANOID_EVERYDAY_H1 = 9   # HUMANOID Everyday State
    ACTION_NET = 10          # ACTION NET State
    NEURAL_GR1 = 11         # NEURAL GR1 State
    # fmt: on


# Defines Action Encoding Schemes
class ActionEncoding(IntEnum):
    # fmt: off
    EEF_POS = 1             # EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)
    JOINT_POS = 2           # Joint Delta Position (7) + Gripper Open/Close (1)
    JOINT_POS_BIMANUAL = 3  # Joint Delta Position (2 x [ Joint Delta Position (6) + Gripper Open/Close (1) ])
    EEF_R6 = 4              # EEF Delta XYZ (3) + R6 (6) + Gripper Open/Close (1)
    AGIBOT_DEXHAND = 5      # AGIBOT DexHand State
    AGIBOT_GRIPPER = 6      # AGIBOT Gripper State
    GALAXEA = 7             # GALAXEA State
    HUMANOID_EVERYDAY_G1 = 8   # HUMANOID Everyday State
    HUMANOID_EVERYDAY_H1 = 9   # HUMANOID Everyday State
    ACTION_NET = 10          # ACTION NET State
    NEURAL_GR1 = 11         # NEURAL GR1 State
    # fmt: on


def oxe_state_scheme_to_keys(state_encoding: StateEncoding) -> list[Tuple[str, int, int, StateActionMetadata]]:
    """Maps StateEncoding to corresponding state observation keys."""
    if state_encoding == StateEncoding.POS_EULER:
        metadata_eef_position = StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(3,),
            continuous=True,
        )
        metadata_eef_rotation = StateActionMetadata(
            absolute=True,
            rotation_type=RotationType.EULER_ANGLES_RPY,
            shape=(3,),
            continuous=True,
        )
        metadata_gripper_close = StateActionMetadata(
            absolute=False,
            rotation_type=None,
            shape=(1,),
            continuous=False,
        )
        return [
            ("eef_position", 0, 3, metadata_eef_position),
            ("eef_rotation", 3, 6, metadata_eef_rotation),
            ("gripper_close", 7, 8, metadata_gripper_close)
        ]
    elif state_encoding == StateEncoding.POS_QUAT:
        metadata_eef_position = StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(3,),
            continuous=True,
        )
        metadata_eef_rotation = StateActionMetadata(
            absolute=True,
            rotation_type=RotationType.QUATERNION,
            shape=(4,),
            continuous=True,
        )
        metadata_gripper_close = StateActionMetadata(
            absolute=False,
            rotation_type=None,
            shape=(1,),
            continuous=False,
        )
        return [
            ("eef_position", 0, 3, metadata_eef_position),
            ("eef_rotation", 3, 7, metadata_eef_rotation),
            ("gripper_close", 7, 8, metadata_gripper_close)
        ]
    elif state_encoding == StateEncoding.JOINT:
        metadata_joint_positions = StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(7,),
            continuous=True,
        )
        metadata_gripper_close = StateActionMetadata(
            absolute=False,
            rotation_type=None,
            shape=(1,),
            continuous=False,
        )
        return [
            ("joint_positions", 0, 7, metadata_joint_positions),
            ("gripper_close", 7, 8, metadata_gripper_close)
        ]
    elif state_encoding == StateEncoding.JOINT_BIMANUAL:
        metadata_left_joint_positions = StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(6,),
            continuous=True,
        )
        metadata_left_gripper_close = StateActionMetadata(
            absolute=False,
            rotation_type=None,
            shape=(1,),
            continuous=False,
        )
        metadata_right_joint_positions = StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(6,),
            continuous=True,
        )
        metadata_right_gripper_close = StateActionMetadata(
            absolute=False,
            rotation_type=None,
            shape=(1,),
            continuous=False,
        )
        return [
            ("left_joint_positions", 0, 6, metadata_left_joint_positions),
            ("left_gripper_close", 6, 7, metadata_left_gripper_close),
            ("right_joint_positions", 7, 13, metadata_right_joint_positions),
            ("right_gripper_close", 13, 14, metadata_right_gripper_close)
        ]
    elif state_encoding == StateEncoding.AGIBOT_DEXHAND:
        metadata_agibot_dexhand = StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(44,),
            continuous=True,
        )
        return [("agibot_dexhand", 0, 44, metadata_agibot_dexhand)]
    elif state_encoding == StateEncoding.AGIBOT_GRIPPER:
        metadata_agibot_gripper = StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(34,),
            continuous=True,
        )
        return [("agibot_gripper", 0, 34, metadata_agibot_gripper)]
    elif state_encoding == StateEncoding.GALAXEA:
        metadata_galaxea = StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(18,),
            continuous=True,
        )
        return [("galaxea", 0, 18, metadata_galaxea)]
    elif state_encoding == StateEncoding.NONE:
        return [(None, 0, 0, None)]
    elif state_encoding == StateEncoding.HUMANOID_EVERYDAY_G1:
        metadata_humanoid_everyday_g1 = StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(28,),
            continuous=True,
        )
        return [("humanoid_everyday_g1", 0, 28, metadata_humanoid_everyday_g1)]
    elif state_encoding == StateEncoding.HUMANOID_EVERYDAY_H1:
        metadata_humanoid_everyday_h1 = StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(26,),
            continuous=True,
        )
        return [("humanoid_everyday_h1", 0, 26, metadata_humanoid_everyday_h1)]
    elif state_encoding == StateEncoding.ACTION_NET:
        return [("action_net", 0, 44, StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(44,),
            continuous=True,
        ))]
    elif state_encoding == StateEncoding.NEURAL_GR1:
        return [("neural_gr1", 0, 44, StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(44,),
            continuous=True,
        ))]
    else:
        raise ValueError(f"Unsupported StateEncoding: {state_encoding}")


def oxe_action_scheme_to_keys(action_encoding: ActionEncoding) -> list[Tuple[str, int, int, StateActionMetadata]]:
    """Maps ActionEncoding to corresponding action keys."""
    if action_encoding == ActionEncoding.EEF_POS:
        metadata_eef_delta_position = StateActionMetadata(
            absolute=False,
            rotation_type=None,
            shape=(3,),
            continuous=True,
        )
        metadata_eef_delta_rotation = StateActionMetadata(
            absolute=False,
            rotation_type=RotationType.EULER_ANGLES_RPY,
            shape=(3,),
            continuous=True,
        )
        metadata_gripper_close = StateActionMetadata(
            absolute=False,
            rotation_type=None,
            shape=(1,),
            continuous=False,
        )
        return [
            ("eef_delta_position", 0, 3, metadata_eef_delta_position),
            ("eef_delta_rotation", 3, 6, metadata_eef_delta_rotation),
            ("gripper_close", 6, 7, metadata_gripper_close)
        ]
    elif action_encoding == ActionEncoding.JOINT_POS:
        metadata_joint_delta_positions = StateActionMetadata(
            absolute=False,
            rotation_type=None,
            shape=(7,),
            continuous=True,
        )
        metadata_gripper_close = StateActionMetadata(
            absolute=False,
            rotation_type=None,
            shape=(1,),
            continuous=False,
        )
        return [
            ("joint_delta_positions", 0, 7, metadata_joint_delta_positions),
            ("gripper_close", 7, 8, metadata_gripper_close)
        ]
    elif action_encoding == ActionEncoding.JOINT_POS_BIMANUAL:
        metadata_left_joint_delta_positions = StateActionMetadata(
            absolute=False,
            rotation_type=None,
            shape=(6,),
            continuous=True,
        )
        metadata_left_gripper_close = StateActionMetadata(
            absolute=False,
            rotation_type=None,
            shape=(1,),
            continuous=False,
        )
        metadata_right_joint_delta_positions = StateActionMetadata(
            absolute=False,
            rotation_type=None,
            shape=(6,),
            continuous=True,
        )
        metadata_right_gripper_close = StateActionMetadata(
            absolute=False,
            rotation_type=None,
            shape=(1,),
            continuous=False,
        )
        return [
            ("left_joint_delta_positions", 0, 6, metadata_left_joint_delta_positions),
            ("left_gripper_close", 6, 7, metadata_left_gripper_close),
            ("right_joint_delta_positions", 7, 13, metadata_right_joint_delta_positions),
            ("right_gripper_close", 13, 14, metadata_right_gripper_close)
        ]
    elif action_encoding == ActionEncoding.EEF_R6:
        metadata_eef_delta_position = StateActionMetadata(
            absolute=False,
            rotation_type=None,
            shape=(3,),
            continuous=True,
        )
        metadata_eef_rotation_6d = StateActionMetadata(
            absolute=False,
            rotation_type=RotationType.ROTATION_6D,
            shape=(6,),
            continuous=True,
        )
        metadata_gripper_close = StateActionMetadata(
            absolute=False,
            rotation_type=None,
            shape=(1,),
            continuous=False,
        )
        return [
            ("eef_delta_position", 0, 3, metadata_eef_delta_position),
            ("eef_rotation_6d", 3, 9, metadata_eef_rotation_6d),
            ("gripper_close", 9, 10, metadata_gripper_close)
        ]
    elif action_encoding == ActionEncoding.AGIBOT_DEXHAND:
        metadata_agibot_dexhand = StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(44,),
            continuous=True,
        )
        return [("agibot_dexhand", 0, 44, metadata_agibot_dexhand)]
    elif action_encoding == ActionEncoding.AGIBOT_GRIPPER:
        metadata_agibot_gripper = StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(34,),
            continuous=True,
        )
        return [("agibot_gripper", 0, 34, metadata_agibot_gripper)]
    elif action_encoding == ActionEncoding.GALAXEA:
        metadata_galaxea = StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(26,),
            continuous=True,
        )
        return [("galaxea", 0, 26, metadata_galaxea)]
    elif action_encoding == ActionEncoding.HUMANOID_EVERYDAY_G1:
        metadata_humanoid_everyday_g1 = StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(26,),
            continuous=True,
        )
        return [("humanoid_everyday_g1", 0, 26, metadata_humanoid_everyday_g1)]
    elif action_encoding == ActionEncoding.HUMANOID_EVERYDAY_H1:
        metadata_humanoid_everyday_h1 = StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(26,),
            continuous=True,
        )
        return [("humanoid_everyday_h1", 0, 26, metadata_humanoid_everyday_h1)]
    elif action_encoding == ActionEncoding.ACTION_NET:
        metadata_action_net = StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(44,),
            continuous=True,
        )
        return [("action_net", 0, 44, metadata_action_net)]
    elif action_encoding == ActionEncoding.NEURAL_GR1:
        metadata_neural_gr1 = StateActionMetadata(
            absolute=True,
            rotation_type=None,
            shape=(44,),
            continuous=True,
        )
        return [("neural_gr1", 0, 44, metadata_neural_gr1)]
    else:
        raise ValueError(f"Unsupported ActionEncoding: {action_encoding}")


# === Individual Dataset Configs ===
OXE_DATASET_CONFIGS = {
    "fractal20220817_data": { # good
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["base_pose_tool_reached", "gripper_closed"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 3.0,
    },
    "kuka": { # good
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [
            "clip_function_input/base_pose_tool_reached",
            "gripper_closed",
        ],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10.0,
    },
    "bridge_oxe": {  # Version of Bridge V2 in Open X-Embodiment mixture
        "image_obs_keys": {"primary": "image", "secondary": "image_1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],  #NOTE(TY): Fixed
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 5.0,
    },
    "bridge_orig": {  # Original version of Bridge V2 from project website
        "image_obs_keys": {"primary": "image_0", "secondary": "image_1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],  #NOTE(TY): Fixed
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 5.0,
    },
    "bridge_dataset": {  # Original version of Bridge V2 from project website
        "image_obs_keys": {"primary": "image_0", "secondary": "image_1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],  #NOTE(TY): Fixed
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 5.0,
    },
    "taco_play": {
        "image_obs_keys": {"primary": "rgb_static", "secondary": None, "wrist": "rgb_gripper"},
        ##NOTE(TY): erased depth information from tfrecord due to parse error #######################################
        ## 2025-11-05 11:06:50.829431: W tensorflow/core/framework/op_kernel.cc:1839]
        ## OP_REQUIRES failed at example_parsing_ops.cc:98 : INVALID_ARGUMENT: Key: steps/observation/depth_gripper.
        ## Can't parse serialized Example.
        # "depth_obs_keys": {"primary": "depth_static", "secondary": None, "wrist": "depth_gripper"},
        #############################################################################################################
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state_eef", None, "state_gripper"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 15.0,
    },
    "jaco_play": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "image_wrist"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state_eef", None, "state_gripper"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10.0,
    },
    "berkeley_cable_routing": {
        "image_obs_keys": {"primary": "image", "secondary": "top_image", "wrist": "wrist45_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["robot_state", None],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10.0,
    },
    "roboturk": {
        "image_obs_keys": {"primary": "front_rgb", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, None],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10.0,
    },
    "nyu_door_opening_surprising_effectiveness": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, None],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 3.0,
    },
    "viola": {
        "image_obs_keys": {"primary": "agentview_rgb", "secondary": None, "wrist": "eye_in_hand_rgb"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["joint_states", "gripper_states"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 20.0,
    },
    "berkeley_autolab_ur5": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "hand_image"},
        "depth_obs_keys": {"primary": "depth", "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 5.0,
    },
    "toto": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 30.0,
    },
    "language_table": {
        "image_obs_keys": {"primary": "rgb", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["effector_translation", None, None, None, None, None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10.0,
    },
    "columbia_cairlab_pusht_real": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["robot_state", None, None, None, None, None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["ee_position", "ee_orientation", None],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 20,
    },
    "nyu_rot_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 3,
    },
    "stanford_hydra_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],  #NOTE(TY): Fixed
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "austin_buds_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 20,
    },
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": "image_additional_view", "wrist": None},
        "depth_obs_keys": {"primary": "depth", "secondary": "depth_additional_view", "wrist": None},
        "state_obs_keys": ["eef_state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 3,
    },
    "maniskill_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": "depth", "secondary": None, "wrist": "wrist_depth"},
        "state_obs_keys": ["tcp_pose", "gripper_state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 20,
    },
    "furniture_bench_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "highres_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, None],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["joint_state", None],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 2,
    },
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],  #NOTE(TY): Fixed
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 3,
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 20,
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT, #NOTE(TY): Fixed (QUAT -> JOINT)
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 20,
    },
    "bc_z": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["present/xyz", "present/axis_angle", None, "present/sensed_close"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],  #NOTE(TY): Fixed
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"], #NOTE(TY): Fixed
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": "image2", "wrist": "hand_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["end_effector_pose", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "utokyo_xarm_bimanual_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["pose_r", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "robo_net": {
        "image_obs_keys": {"primary": "image", "secondary": "image1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],  #NOTE(TY): Fixed
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 1,
    },
    "berkeley_mvp_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "hand_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["pose", "gripper"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.JOINT_POS,
        "control_frequency": 5,
    },
    "berkeley_rpt_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "hand_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["joint_pos", "gripper"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
        "control_frequency": 30,
    },
    "kaist_nonprehensile_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "stanford_mask_vit_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],  #NOTE(TY): Fixed
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": None, # N/A in dataset list (quasistatic assumption)
    },
    "tokyo_u_lsmo_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],  #NOTE(TY): Fixed
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "dlr_sara_pour_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "dlr_sara_grid_clamp_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 5,
    },
    "asu_table_top_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],  #NOTE(TY): Fixed
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 12.5,
    },
    "stanford_robocook_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image_1", "secondary": "image_2", "wrist": None},
        "depth_obs_keys": {"primary": "depth_1", "secondary": "depth_2", "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],  #NOTE(TY): Fixed
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 5,
    },
    "imperialcollege_sawyer_wrist_cam": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, "state"],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["joint_state", "gripper_state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 20,
    },
    "uiuc_d3field": {
        "image_obs_keys": {"primary": "image_1", "secondary": "image_2", "wrist": None},
        "depth_obs_keys": {"primary": "depth_1", "secondary": "depth_2", "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, None],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 1,
    },
    "utaustin_mutex": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 20,
    },
    "berkeley_fanuc_manipulation": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["joint_state", None, "gripper_state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "cmu_playing_with_food": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "finger_vision_1"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "cmu_play_fusion": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 5,
    },
    "cmu_stretch": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],  #NOTE(TY): Fixed
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "berkeley_gnm_recon": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 3,
    },
    "berkeley_gnm_cory_hall": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 5,
    },
    "berkeley_gnm_sac_son": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "droid": {
        "image_obs_keys": {"primary": "exterior_image_1_left", "secondary": "exterior_image_2_left", "wrist": "wrist_image_left"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,  #NOTE(TY): Fixed (POS_QUAT -> POS_EULER)
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 15,
        "aux_kwargs": {
            "dataset_frame_transform_kwargs": {
                "chunk_filter_fn": zero_action_filter,
            },
        },
    },
    "fmb_dataset": {
        "image_obs_keys": {"primary": "image_side_1", "secondary": "image_side_2", "wrist": "image_wrist_1"},
        "depth_obs_keys": {"primary": "image_side_1_depth", "secondary": "image_side_2_depth", "wrist": "image_wrist_1_depth"},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_QUAT,  #NOTE(TY): Fixed (POS_EULER -> POS_QUAT)
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10,
    },
    "dobbe": {
        "image_obs_keys": {"primary": "wrist_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 3.75,
    },
    "roboset": {
        "image_obs_keys": {"primary": "image_left", "secondary": "image_right", "wrist": "image_wrist"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
        "control_frequency": 5,
    },
    "rh20t": {
        "image_obs_keys": {"primary": "image_front", "secondary": "image_side_right", "wrist": "image_wrist"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 10, # Standard RH20T frequency
    },
    ### T-DROID datasets
    "tdroid_carrot_in_bowl": {  # "put carrot in bowl" task, 50 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 5,
    },
    "tdroid_pour_corn_in_pot": {  # "pour corn from red bowl into steel pot" task, 50 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 5,
    },
    "tdroid_flip_pot_upright": {  # "flip pot upright" task, 10 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 5,
    },
    "tdroid_move_object_onto_plate": {  # "move <object> onto plate" task, 150 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 5,
    },
    "tdroid_knock_object_over": {  # "knock <object> over" task, 70 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 5,
    },
    "tdroid_cover_object_with_towel": {  # "cover <object> with towel" task, 45 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 5,
    },
    ### DROID Finetuning datasets
    "droid_wipe": {
        "image_obs_keys": {"primary": "exterior_image_2_left", "secondary": None, "wrist": "wrist_image_left"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 15, # Based on 'droid' (15)
    },
    ### LIBERO datasets (modified versions)
    "libero_spatial_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 20, # Standard Libero frequency
    },
    "libero_object_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 20,
    },
    "libero_goal_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 20,
    },
    "libero_10_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 20,
    },
    "libero_4_task_suites_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 20,
    },
    ### ALOHA fine-tuning datasets
    "aloha1_fold_shorts_20_demos": {
        "image_obs_keys": {"primary": "image", "secondary": None, "left_wrist": "left_wrist_image", "right_wrist": "right_wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
        "control_frequency": 50, # Based on 'aloha_mobile' (50)
    },
    "aloha1_fold_shirt_30_demos": {
        "image_obs_keys": {"primary": "image", "secondary": None, "left_wrist": "left_wrist_image", "right_wrist": "right_wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
        "control_frequency": 50,
    },
    "aloha1_scoop_X_into_bowl_45_demos": {
        "image_obs_keys": {"primary": "image", "secondary": None, "left_wrist": "left_wrist_image", "right_wrist": "right_wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
        "control_frequency": 50,
    },
    "aloha1_put_X_into_pot_300_demos": {
        "image_obs_keys": {"primary": "image", "secondary": None, "left_wrist": "left_wrist_image", "right_wrist": "right_wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
        "control_frequency": 50,
    },
    ### NOTE(TY): Added
    "agibot_dexhand": {
        "image_obs_keys":{"primary": "image_head", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.AGIBOT_DEXHAND,
        "action_encoding": ActionEncoding.AGIBOT_DEXHAND,
        "control_frequency": 30,
    },
    "agibot_gripper": {
        "image_obs_keys":{"primary": "image_head", "secondary": "image_left", "wrist": "image_right"}, # secondary: left wrist, wrist: right wrist
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.AGIBOT_GRIPPER,
        "action_encoding": ActionEncoding.AGIBOT_GRIPPER,
        "control_frequency": 30,
    },
    "galaxea": {
        "image_obs_keys": {"primary": "image_camera_head", "secondary": "image_camera_wrist_left", "wrist": "image_camera_wrist_right"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.GALAXEA,
        "action_encoding": ActionEncoding.GALAXEA,
        "control_frequency": 15,
    },
    "humanoid_everyday_g1": {
        "image_obs_keys": {"primary": "egocentric", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.HUMANOID_EVERYDAY_G1,
        "action_encoding": ActionEncoding.HUMANOID_EVERYDAY_G1,
    },
    "humanoid_everyday_h1": {
        "image_obs_keys": {"primary": "egocentric", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.HUMANOID_EVERYDAY_H1,
        "action_encoding": ActionEncoding.HUMANOID_EVERYDAY_H1,
    },
    "action_net": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.ACTION_NET,
        "action_encoding": ActionEncoding.ACTION_NET,
    },
    "neural_gr1": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.NEURAL_GR1,
        "action_encoding": ActionEncoding.NEURAL_GR1,
    },
    "neural_gr1_v2": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.NEURAL_GR1,
        "action_encoding": ActionEncoding.NEURAL_GR1,
    }
}

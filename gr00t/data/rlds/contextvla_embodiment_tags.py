"""
Mapping from RLDS dataset names to ContextVLA embodiment tag strings.

These strings are used in the ContextVLA chat template for the <embodiment_tag> placeholder.
The format matches the reference implementation in tmp_AlinVLA-VLM/qwenvl/data/__init__.py.
"""

# Mapping from RLDS dataset names to ContextVLA embodiment tag strings
EMBODIMENT_TAG_MAPPING = {
    "fractal20220817_data": "Embodiment Tag: RT-1 Robot Action Dataset, Robot: Google Robot, Morphology: Mobile Manipulator, Gripper: Default",
    "kuka": "Embodiment Tag: QT-Opt Dataset, Robot: Kuka iiwa, Morphology: Single Arm, Gripper: Default",
    "bridge_orig": "Embodiment Tag: Bridge Dataset, Robot: WidowX, Morphology: Single Arm, Gripper: Default",
    "taco_play": "Embodiment Tag: Freiburg Franka Play Dataset, Robot: Franka, Morphology: Single Arm, Gripper: Custom",
    "jaco_play": "Embodiment Tag: CLVR Jaco Play Dataset, Robot: Kinova Jaco 2, Morphology: Single Arm, Gripper: Default",
    "berkeley_cable_routing": "Embodiment Tag: Berkeley Cable Routing Dataset, Robot: Franka, Morphology: Single Arm, Gripper: Default",
    "roboturk": "Embodiment Tag: RoboTurk Dataset, Robot: Sawyer, Morphology: Single Arm, Gripper: Default",
    "viola": "Embodiment Tag: VIOLA Dataset, Robot: Franka, Morphology: Single Arm, Gripper: Default",
    "berkeley_autolab_ur5": "Embodiment Tag: Berkeley Autolab UR5 Dataset, Robot: UR5, Morphology: Single Arm, Gripper: Default",
    "toto": "Embodiment Tag: TOTO Dataset, Robot: Franka, Morphology: Single Arm, Gripper: Default",
    "language_table": "Embodiment Tag: Language Table Dataset, Robot: xArm6, Morphology: Single Arm, Gripper: Default",
    "stanford_hydra_dataset_converted_externally_to_rlds": "Embodiment Tag: Stanford HYDRA Dataset, Robot: Franka, Morphology: Single Arm, Gripper: Default",
    "austin_buds_dataset_converted_externally_to_rlds": "Embodiment Tag: Austin BUDS Dataset, Robot: Franka, Morphology: Single Arm, Gripper: Default",
    "nyu_franka_play_dataset_converted_externally_to_rlds": "Embodiment Tag: NYU Franka Play Dataset, Robot: Franka, Morphology: Single Arm, Gripper: Default",
    "furniture_bench_dataset_converted_externally_to_rlds": "Embodiment Tag: FurnitureBench Dataset, Robot: Franka, Morphology: Single Arm, Gripper: Default",
    "ucsd_kitchen_dataset_converted_externally_to_rlds": "Embodiment Tag: UCSD Kitchen Dataset, Robot: xArm7, Morphology: Single Arm, Gripper: Default",
    "austin_sailor_dataset_converted_externally_to_rlds": "Embodiment Tag: Austin Sailor Dataset, Robot: Franka, Morphology: Single Arm, Gripper: Default",
    "austin_sirius_dataset_converted_externally_to_rlds": "Embodiment Tag: Austin Sirius Dataset, Robot: Franka, Morphology: Single Arm, Gripper: Default",
    "dlr_edan_shared_control_converted_externally_to_rlds": "Embodiment Tag: DLR EDAN Dataset, Robot: DLR EDAN, Morphology: Mobile Manipulator, Gripper: Default",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": "Embodiment Tag: IAMLab CMU Dataset, Robot: Franka, Morphology: Single Arm, Gripper: Default",
    "utaustin_mutex": "Embodiment Tag: UT Austin Mutex Dataset, Robot: Franka, Morphology: Single Arm, Gripper: Default",
    "berkeley_fanuc_manipulation": "Embodiment Tag: Berkeley Fanuc Dataset, Robot: Fanuc Mate, Morphology: Single Arm, Gripper: Default",
    "cmu_stretch": "Embodiment Tag: CMU Stretch Dataset, Robot: Hello Stretch, Morphology: Mobile Manipulator, Gripper: Default",
    "bc_z": "Embodiment Tag: BC-Z Dataset, Robot: Google Robot, Morphology: Mobile Manipulator, Gripper: Default",
    "fmb_dataset": "Embodiment Tag: FMB Dataset, Robot: Franka, Morphology: Single Arm, Gripper: Default",
    "dobbe": "Embodiment Tag: DOBBE Dataset, Robot: Hello Stretch, Morphology: Mobile Manipulator, Gripper: Default",
    "droid": "Embodiment Tag: DROID Dataset, Robot: Franka, Morphology: Single Arm, Gripper: Default",

    # AGIBOT datasets
    "agibot_gripper": "Embodiment Tag: AGIBOT Dataset, Robot: Genie-1, Morphology: Bi-Manual",
    "agibot_dexhand": "Embodiment Tag: AGIBOT Dataset, Robot: Genie-1, Morphology: Bi-Manual",

    # GALAXEA datasets
    "galaxea": "Embodiment Tag: Galaxea Dataset, Robot: Galaxea R1, Morphology: Bi-Manual Mobile Manipulator, Gripper: Default",

    # EGODEX datasets
    "egodex_gr1": "Embodiment Tag: EGODEX-GR-1 Dataset, Robot: Human hand + GR-1, Morphology: Bi-Manual",

    # Default fallback
    "default": "Embodiment Tag: Unknown Dataset, Robot: Unknown, Morphology: Unknown, Gripper: Unknown",

    # HUMANOID GR1
    "action_net": "Embodiment Tag: ActionNet Dataset, Robot: GR1, Morphology: Humanoid, Hand: Default",
    "neural_gr1": "Embodiment Tag: Neural Trajectory GR1 Dataset, Robot: GR1, Morphology: Humanoid, Hand: Default",
    "humanoid_everyday_g1": "Embodiment Tag: Humanoid Everyday G1 Dataset, Robot: G1, Morphology: Humanoid, Hand: Dex3-1",
    "humanoid_everyday_h1": "Embodiment Tag: Humanoid Everyday H1 Dataset, Robot: H1, Morphology: Humanoid, Hand: Inspire",
}


#NOTE(MK): Embodiment tag mapping for LeRobot datasets uses `data_config` instead of `embodiment_tag`!
LEROBOT_EMBODIMENT_TAG_MAPPING = {
    "libero": "Embodiment Tag: LIBERO Dataset, Robot: Franka, Morphology: Single Arm, Gripper: Default",
    "single_panda_gripper": "Embodiment Tag: RoboCasa Kitchen Dataset, Robot: Franka, Morphology: Single Arm, Gripper: Default",
    "fourier_gr1_arms_waist": "Embodiment Tag: RoboCasa GR1 Dataset, Robot: GR1, Morphology: Humanoid, Hand: Default",
}


def get_contextvla_embodiment_tag_string(dataset_name: str) -> str:
    """
    Get ContextVLA embodiment tag string for a given dataset name.

    Args:
        dataset_name: RLDS dataset name (e.g., "bridgev2", "fmb", "nyu_franka_play_dataset_converted_externally_to_rlds", etc.)

    Returns:
        ContextVLA embodiment tag string for use in chat template (e.g., 
        "Embodiment Tag: Bridge Dataset, Robot: WidowX, Morphology: Single Arm, Gripper: Default")

    Note:
        This function handles special cases like "agibot_gripper" and "galaxea" that may appear
        as substrings in dataset names. It also provides a default fallback for unknown datasets.
    """
    # Handle special cases (matching logic from gr00t/data/rlds/dataset.py)
    normalized_name = dataset_name.lower()

    if "agibot_gripper" in normalized_name:
        dataset_name = "agibot_gripper"
    elif "agibot_dexhand" in normalized_name or ("agibot" in normalized_name and "gripper" not in normalized_name):
        dataset_name = "agibot_dexhand"
    elif "galaxea" in normalized_name:
        # Note: galaxea not in reference code, but keeping for consistency
        dataset_name = "galaxea"
    elif "egodex" in normalized_name or ("gr1" in normalized_name and "egodex" in normalized_name):
        dataset_name = "egodex_gr1"

    # Return mapped string or default
    return EMBODIMENT_TAG_MAPPING.get(
        dataset_name, 
        EMBODIMENT_TAG_MAPPING["default"]
    )


def get_contextvla_embodiment_tag_string_from_enum(embodiment_tag_value: str) -> str:
    """
    Get ContextVLA embodiment tag string for a given EmbodimentTag enum value (for LeRobot datasets).

    Args:
        embodiment_tag_value: EmbodimentTag enum value (e.g., "gr1", "new_embodiment", "droid", etc.)

    Returns:
        ContextVLA embodiment tag string for use in chat template (e.g., 
        "Embodiment Tag: GR1 Dataset, Robot: GR1, Morphology: Humanoid, Hand: Default")

    Note:
        This function is used for LeRobot datasets where we have an EmbodimentTag enum value
        instead of a dataset name. Falls back to default if the value is not found in the mapping.
    """
    return LEROBOT_EMBODIMENT_TAG_MAPPING.get(
        embodiment_tag_value,
        LEROBOT_EMBODIMENT_TAG_MAPPING.get("new_embodiment", EMBODIMENT_TAG_MAPPING["default"])
    )

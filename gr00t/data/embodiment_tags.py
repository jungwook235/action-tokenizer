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

from enum import Enum


class EmbodimentTag(Enum):
    GR1 = "gr1"
    """
    The GR1 dataset.
    """

    OXE_DROID = "droid"
    """
    The OxE Droid dataset.
    """

    AGIBOT_GENIE1 = "agibot_genie1"
    """
    The AgiBot Genie-1 with gripper dataset.
    """

    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment for finetuning.
    """

    ###OXE embodiment tags###
    OXE_FRACTAL = "fractal20220817_data"
    OXE_KUKA = "kuka"
    OXE_BRIDGE_ORIG = "bridge_orig"
    OXE_TACO = "taco_play"
    OXE_JACO = "jaco_play"
    OXE_BERKELEY_CABLE_ROUTING = "berkeley_cable_routing"
    OXE_ROBOTURK = "roboturk"
    OXE_VIOLA = "viola"
    OXE_BERKELEY_AUTOLAB_UR5 = "berkeley_autolab_ur5"
    OXE_TOTO = "toto"
    OXE_LANGUAGE_TABLE = "language_table"
    OXE_STANFORD_HYDRA = "stanford_hydra_dataset_converted_externally_to_rlds"
    OXE_AUSTIN_BUDS = "austin_buds_dataset_converted_externally_to_rlds"
    OXE_NYU_FRANKA_PLAY = "nyu_franka_play_dataset_converted_externally_to_rlds"
    OXE_FURNITURE_BENCH = "furniture_bench_dataset_converted_externally_to_rlds"
    OXE_UCSD_KITCHEN = "ucsd_kitchen_dataset_converted_externally_to_rlds"
    OXE_AUSTIN_SAILOR = "austin_sailor_dataset_converted_externally_to_rlds"
    OXE_AUSTIN_SIRIUS = "austin_sirius_dataset_converted_externally_to_rlds"
    OXE_DLR_EDAN_SHARED_CONTROL = "dlr_edan_shared_control_converted_externally_to_rlds"
    OXE_IAMLAB_CMU_PICKUP_INSERT = "iamlab_cmu_pickup_insert_converted_externally_to_rlds"
    OXE_UTAUSTIN_MUTEX = "utaustin_mutex"
    OXE_BERKELEY_FANUC_MANIPULATION = "berkeley_fanuc_manipulation"
    OXE_CMU_STRETCH = "cmu_stretch"
    OXE_BC_Z = "bc_z"
    OXE_FMB_DATASET = "fmb_dataset"
    OXE_DOBBE = "dobbe"
    OXE_AGIBOT_DEXHAND = "agibot_dexhand"
    OXE_AGIBOT_GRIPPER = "agibot_gripper"
    OXE_GALAXEA = "galaxea"

    OXE_HUMANOID_EVERYDAY_G1 = "humanoid_everyday_g1"
    OXE_HUMANOID_EVERYDAY_H1 = "humanoid_everyday_h1"
    OXE_ACTION_NET = "action_net"
    OXE_NEURAL_GR1 = "neural_gr1"


# Embodiment tag string: to projector index in the Action Expert Module
EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.NEW_EMBODIMENT.value: 31,
    EmbodimentTag.GR1.value: 24,
    EmbodimentTag.OXE_FRACTAL.value: 0,
    EmbodimentTag.OXE_KUKA.value: 1,
    EmbodimentTag.OXE_BRIDGE_ORIG.value: 2,
    EmbodimentTag.OXE_TACO.value: 3,
    EmbodimentTag.OXE_JACO.value: 4,
    EmbodimentTag.OXE_BERKELEY_CABLE_ROUTING.value: 5,
    EmbodimentTag.OXE_ROBOTURK.value: 6,
    EmbodimentTag.OXE_VIOLA.value: 7,
    EmbodimentTag.OXE_BERKELEY_AUTOLAB_UR5.value: 8,
    EmbodimentTag.OXE_TOTO.value: 9,
    EmbodimentTag.OXE_LANGUAGE_TABLE.value: 10,
    EmbodimentTag.OXE_STANFORD_HYDRA.value: 11,
    EmbodimentTag.OXE_AUSTIN_BUDS.value: 12,
    EmbodimentTag.OXE_NYU_FRANKA_PLAY.value: 13,
    EmbodimentTag.OXE_FURNITURE_BENCH.value: 14,
    EmbodimentTag.OXE_UCSD_KITCHEN.value: 15,
    EmbodimentTag.OXE_AUSTIN_SAILOR.value: 16,
    EmbodimentTag.OXE_DROID.value: 17,
    EmbodimentTag.OXE_AUSTIN_SIRIUS.value: 18,
    EmbodimentTag.OXE_DLR_EDAN_SHARED_CONTROL.value: 19,
    EmbodimentTag.OXE_IAMLAB_CMU_PICKUP_INSERT.value: 20,
    EmbodimentTag.OXE_UTAUSTIN_MUTEX.value: 21,
    EmbodimentTag.OXE_BERKELEY_FANUC_MANIPULATION.value: 22,
    EmbodimentTag.OXE_CMU_STRETCH.value: 23,
    EmbodimentTag.OXE_BC_Z.value: 25,
    EmbodimentTag.OXE_FMB_DATASET.value: 26,
    EmbodimentTag.OXE_DOBBE.value: 27,
    EmbodimentTag.OXE_AGIBOT_DEXHAND.value: 28,
    EmbodimentTag.OXE_AGIBOT_GRIPPER.value: 29,
    EmbodimentTag.OXE_GALAXEA.value: 30,
    EmbodimentTag.OXE_HUMANOID_EVERYDAY_G1.value: 32,
    EmbodimentTag.OXE_HUMANOID_EVERYDAY_H1.value: 33,
    EmbodimentTag.OXE_ACTION_NET.value: 34,
    EmbodimentTag.OXE_NEURAL_GR1.value: 35,
}

def get_name_from_value(value: int) -> str:
    for name, idx in EMBODIMENT_TAG_MAPPING.items():
        if idx == value:
            return name
    raise ValueError(f"EmbodimentTag with value {value} not found.")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from robolab.constants import BACKGROUND_ASSET_DIR
from robolab.core.events.randomize_background import randomize_dome_light_texture
from robolab.core.events.reset_pose import reset_pose_to_presets
from robolab.core.scenes.utils import import_scene
from robolab.variations.backgrounds import find_background_files

from .starting_states import load_preset_poses

PRESET_POSES = load_preset_poses()
BACKGROUND_FILE = find_background_files(BACKGROUND_ASSET_DIR, "home_office.exr")
CONTACT_OBJECT_LIST = ["cutting_board_a", "ceramic_mug", "coke", "table"]
SCENE = import_scene("mug_and_coke_on_cutting_board.usda", CONTACT_OBJECT_LIST)


@configclass
class SharedRandomization:
    """Single background, uniform sampling over the whole 100-row pool -- no split."""

    randomize_background = EventTerm(
        func=randomize_dome_light_texture,
        mode="reset",
        params={"background_files": [BACKGROUND_FILE]},
    )
    randomize_init_pose = EventTerm(
        func=reset_pose_to_presets,
        mode="reset",
        params={
            "presets": PRESET_POSES,
            "asset_cfg": ["cutting_board_a", "ceramic_mug", "coke"],
            "reset_to_default_otherwise": True,
        },
    )


@configclass
class TimeoutOnlyTerminations:
    """Success is decided entirely by the RLinf-side mixin in
    rlinf/envs/isaaclab/tasks/robolab_task.py, driven by `mode`
    (subtask_1 / subtask_2 / full). Both task files below use this unchanged.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

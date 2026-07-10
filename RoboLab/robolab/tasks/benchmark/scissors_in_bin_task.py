# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.events.reset_pose import reset_pose_uniform
from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.task.task import Task


@configclass
class ScissorsInitPoseRandomization:
    """Uniform ±10cm position randomization on the scissors."""
    randomize_init_pose = EventTerm(
        func=reset_pose_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": ["scissors"],
            "reset_to_default_otherwise": True,
            "use_collision_check": True,
        },
    )


@configclass
class ScissorsInBinTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": "scissors",
            "container": "bin_b04",
            "gripper_name": "gripper",
            "tolerance": 0.0,
            "require_contact_with": False,
            "require_gripper_detached": True,
        },
    )


@dataclass
class ScissorsInBinTask(Task):
    contact_object_list = ["scissors", "bin_b04", "table"]
    scene = import_scene("scissors_in_container.usda", contact_object_list)
    terminations = ScissorsInBinTerminations
    events = ScissorsInitPoseRandomization
    instruction = {
        "default": "Pick up the scissors and place them in the bin",
        "vague": "Put the tool in the container",
        "specific": "Grasp the scissors lying on the table and place them inside the large rectangular bin",
    }
    episode_length_s: int = 60
    attributes = ["semantics", "affordance"]
    subtasks = [
        pick_and_place(object=["scissors"], container="bin_b04", logical="all", score=1.0)
    ]

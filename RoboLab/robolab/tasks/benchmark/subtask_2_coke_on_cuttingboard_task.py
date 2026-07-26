# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from dataclasses import dataclass

from robolab.core.events.reset_pose import reset_to_captured_state
from robolab.core.task.conditionals import pick_and_place_on_surface
from robolab.core.task.task import Task

from robolab.tasks.benchmark._mug_coke_common import (
    CONTACT_OBJECT_LIST,
    SCENE,
    SharedRandomization,
    TimeoutOnlyTerminations,
)

# Set by RLinf (robolab_task.py) via the load_task_from_file module cache before events() runs.
# None by default, so this file falls back to SharedRandomization when run standalone.
RESET_STATES_PATH = None


def _events():
    """Reads RESET_STATES_PATH at call time (not import time), so injection from robolab_task.py
    takes effect before the randomization config is built.
    """
    path = sys.modules[__name__].RESET_STATES_PATH
    if not path:
        return SharedRandomization
    with open(path) as f:
        captured_states = [json.loads(line) for line in f if line.strip()]

    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.utils import configclass

    @configclass
    class CapturedStateRandomization:
        randomize_init_pose = EventTerm(
            func=reset_to_captured_state,
            mode="reset",
            params={
                "captured_states": captured_states,
                "object_names": ["cutting_board_a", "ceramic_mug", "coke"],
                "robot_name": "robot",
            },
        )

    return CapturedStateRandomization


@dataclass
class Subtask2CokeOnCuttingboardTask(Task):
    contact_object_list = CONTACT_OBJECT_LIST
    scene = SCENE
    terminations = TimeoutOnlyTerminations
    events = _events  # function reference; called lazily by the framework, not here
    instruction = {
        "default": "Pick up the coke can and place it on the cutting board"
    }  # placeholder
    episode_length_s: int = 50
    attributes = ["semantics"]
    subtasks = [
        pick_and_place_on_surface(
            object=["coke"], surface="cutting_board_a", logical="all", score=1.0
        )
    ]

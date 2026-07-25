# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from robolab.core.task.conditionals import pick_and_place_on_surface
from robolab.core.task.task import Task

from ._mug_coke_common import (
    CONTACT_OBJECT_LIST,
    SCENE,
    SharedRandomization,
    TimeoutOnlyTerminations,
)


@dataclass
class LongHorizonLatteartcupCokeTask(Task):
    contact_object_list = CONTACT_OBJECT_LIST
    scene = SCENE
    terminations = TimeoutOnlyTerminations
    events = SharedRandomization
    # Placeholder -- the RLinf-side mixin drives the real per-env instruction from mode/subtasks config.
    instruction = {
        "default": "Pick up the latte art cup and place it on the cutting board"
    }
    episode_length_s: int = (
        90  # long enough for the full stage; subtask_1 runs finish well before this
    )
    attributes = ["semantics", "conjunction"]
    subtasks = [
        pick_and_place_on_surface(
            object=["ceramic_mug"], surface="cutting_board_a", logical="all", score=0.5
        ),
        pick_and_place_on_surface(
            object=["coke"], surface="cutting_board_a", logical="all", score=0.5
        ),
    ]

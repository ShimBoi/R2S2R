# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from dataclasses import dataclass

from robolab.core.events.reset_pose import reset_to_captured_state
from robolab.core.task.task import Task

from robolab.tasks.benchmark._mug_coke_common import (
    CONTACT_OBJECT_LIST,
    SCENE,
    SharedRandomization,
    TimeoutOnlyTerminations,
)

# Set by RLinf (rlinf/envs/isaaclab/tasks/robolab_task.py) via the load_task_from_file module
# cache before events() runs, same mechanism as subtask_2_coke_on_cuttingboard_task.py. None
# (root pool) by default. This is the only thing injected into this module's namespace --
# predicate/predicate_args/instruction go straight to the RLinf-side mixin instead, since
# events() is evaluated by RoboLab's own task-registration machinery, which has no access to
# RLinf's Hydra config.
RESET_STATES_PATH = None


def _events():
    """Reads RESET_STATES_PATH at call time, so injection from robolab_task.py takes effect
    before the randomization config is built. The null-check is the root-vs-non-root
    distinction: every edge (root or not) uses this same generic task file with a different
    injected reset_states_path.
    """
    path = sys.modules[__name__].RESET_STATES_PATH
    if not path:
        return SharedRandomization()
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

    return CapturedStateRandomization()


@dataclass
class GenericSingleEdgeTask(Task):
    """Training task for exactly one edge of the discovered subtask tree, replacing what
    would otherwise be a hand-written, per-edge task file.

    Not hardcoded to any particular subtask: predicate, predicate_args, and instruction are
    injected by RLinf (robolab_task.py's mixin, mode="single_edge") from
    init_params.edge_spec.

    terminations is time_out only -- success/failure is decided by the mixin dispatching
    edge_spec["predicate"] by name via robolab.core.task.conditionals.
    """

    contact_object_list = CONTACT_OBJECT_LIST
    scene = SCENE
    terminations = TimeoutOnlyTerminations
    events = _events  # function reference; called lazily by the framework, not here
    # Placeholder -- the RLinf-side mixin (mode="single_edge") drives the real
    # instruction from init_params.edge_spec["instruction"].
    instruction = {"default": "Perform the assigned subtask"}
    episode_length_s: int = 50
    attributes = ["semantics"]

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

# Set by RLinf (rlinf/envs/isaaclab/tasks/robolab_task.py) via the load_task_from_file
# module cache before events() runs -- the *exact same* mechanism already used by
# subtask_2_coke_on_cuttingboard_task.py, unchanged. None (root pool) by default, so
# this file falls back to SharedRandomization when run standalone or when the edge
# being trained has no reset_states_path (i.e. it's a root edge, per PLAN.md section 8.0).
#
# This is deliberately the ONLY thing injected into this module's namespace. The rest
# of an edge spec -- predicate, predicate_args, instruction -- never needs to reach this
# file at all: it's only consumed by the RLinf-side mixin (NoAutoResetManagerBasedRLEnv
# in robolab_task.py), which already has direct access to the Hydra config
# (init_params.edge_spec) and doesn't need the module-injection trick. That trick exists
# here only because events() is evaluated by RoboLab's own task-registration machinery
# (auto_register_droid_envs / generate_task_env_cfg), a separate code path with no access
# to RLinf's cfg object -- reset_states_path is the one piece of the edge spec that
# genuinely has to cross that boundary.
RESET_STATES_PATH = None


def _events():
    """Reads RESET_STATES_PATH at call time (not import time), so injection from
    robolab_task.py takes effect before the randomization config is built. Identical
    logic to subtask_2_coke_on_cuttingboard_task.py's _events() -- the null-check here
    *is* the root-vs-non-root distinction for any edge in the discovered tree (PLAN.md
    section 8.0): no separate "root" task file is needed, every edge (root or not) uses
    this same generic task file, just with a different injected reset_states_path.
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
    """Training task for exactly one edge of the discovered subtask tree (PLAN.md
    section 8.0), replacing what would otherwise be a hand-written, per-edge hardcoded
    task file (as mug_on_cutting_board_task.py / subtask_2_coke_on_cuttingboard_task.py
    were for the original two-subtask pipeline).

    Not hardcoded to any particular subtask: which predicate defines success, its
    args, and the instruction shown to the policy are all injected by RLinf
    (robolab_task.py's mixin, mode="single_edge") from init_params.edge_spec, read
    directly off the Hydra config on the RLinf side -- see the RESET_STATES_PATH
    docstring above for why that part doesn't need the same module-injection trick.

    terminations is time_out only: success/failure for training purposes is decided
    entirely by the mixin dispatching edge_spec["predicate"] by name via
    robolab.core.task.conditionals, not by a termination term defined here.
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

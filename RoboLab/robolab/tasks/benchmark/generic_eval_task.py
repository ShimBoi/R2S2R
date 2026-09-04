# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from robolab.core.task.task import Task

from robolab.tasks.benchmark._mug_coke_common import (
    CONTACT_OBJECT_LIST,
    SCENE,
    SharedRandomization,
    TimeoutOnlyTerminations,
)


@dataclass
class GenericEvalTask(Task):
    """Eval-time ratchet task for the discovered subtask tree, generalizing
    long_horizon_latteartcup_coke_task.py's fixed two-instruction ratchet to however many
    steps a resolved plan has.

    init_params.plan is a plain ordered list of {"instruction", "predicate",
    "predicate_args"} dicts, produced once before rollout by resolve_plan() (Phase B). The
    RLinf-side mixin (robolab_task.py, mode="plan") walks that list step by step,
    dispatching each step's predicate by name via robolab.core.task.conditionals.

    events is always SharedRandomization -- eval always starts from scratch, unlike
    training edges which may reset from a non-root node's captured end-states.

    terminations is time_out only -- the mixin decides step-by-step success and overall
    episode termination.
    """

    contact_object_list = CONTACT_OBJECT_LIST
    scene = SCENE
    terminations = TimeoutOnlyTerminations
    events = SharedRandomization
    # Placeholder -- the RLinf-side mixin drives the real per-env instruction from
    # init_params.plan[0]["instruction"] at reset, then advances it step by step.
    instruction = {"default": "Complete the assigned sequence of subtasks"}
    episode_length_s: int = (
        90  # long enough for a multi-step plan; matches the existing long-horizon task.
    )
    attributes = ["semantics", "conjunction"]

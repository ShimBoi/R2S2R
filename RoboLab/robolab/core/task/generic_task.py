"""The generic task: runs the task specs given as a JSON list in ROBOLAB_TASK_SPEC_JSON (see robolab.core.task.task_spec)."""
import json
import os
from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
import torch
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.task import curriculum_conditionals
from robolab.core.task.task import Task
from robolab.core.task.task_spec import base_task_path, spec_of_env
from robolab.core.task.task_utils import load_task_from_file

_SPECS = json.loads(os.environ["ROBOLAB_TASK_SPEC_JSON"])
_BASE = load_task_from_file(str(base_task_path(_SPECS[0]["base_task"])))


def specs_success(env, successes):
    owner = torch.tensor(spec_of_env(env.num_envs, len(successes)), device=env.device)
    out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for i, success in enumerate(successes):
        result = getattr(curriculum_conditionals, success["predicate"])(env, **success["args"])
        out = torch.where(owner == i, torch.as_tensor(result, dtype=torch.bool, device=env.device), out)
    return out


@configclass
class GenericTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=specs_success, params={"successes": [spec["success"] for spec in _SPECS]})


@dataclass
class GenericTask(Task):
    contact_object_list = _BASE.contact_object_list
    scene = _BASE.scene
    terminations = GenericTaskTerminations
    events = _BASE.events
    instruction = {"default": _SPECS[0]["instruction"]}
    episode_length_s: int = _BASE.episode_length_s
    attributes = _BASE.attributes
    subtasks = None


del _BASE

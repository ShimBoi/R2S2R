"""The generic task: runs the task spec named by ROBOLAB_TASK_SPEC (see robolab.core.task.task_spec)."""
import os
from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.task.task_spec import base_task_class, load_spec, success_term, validate_spec
from robolab.core.task.task import Task

_SPEC = load_spec(os.environ["ROBOLAB_TASK_SPEC"])
_BASE = base_task_class(_SPEC)
validate_spec(_SPEC, _BASE.contact_object_list)


@configclass
class GenericTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = success_term(_SPEC)


@dataclass
class GenericTask(Task):
    contact_object_list = _BASE.contact_object_list
    scene = _BASE.scene
    terminations = GenericTaskTerminations
    events = _BASE.events
    instruction = {"default": _SPEC["instruction"]}
    episode_length_s: int = _SPEC.get("episode_s", _BASE.episode_length_s)
    attributes = _BASE.attributes
    subtasks = None


del _BASE

"""The generic task: runs the task spec given as JSON in ROBOLAB_TASK_SPEC_JSON (see robolab.core.task.task_spec)."""
import json
import os
from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.constants import TASK_DIR
from robolab.core.task import curriculum_conditionals
from robolab.core.task.task import Task
from robolab.core.task.task_utils import load_task_from_file, resolve_task_path

_SPEC = json.loads(os.environ["ROBOLAB_TASK_SPEC_JSON"])
_BASE = load_task_from_file(resolve_task_path(_SPEC["base_task"], TASK_DIR)[0])


@configclass
class GenericTaskTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=getattr(curriculum_conditionals, _SPEC["success"]["predicate"]),
                       params=dict(_SPEC["success"]["args"]))


@dataclass
class GenericTask(Task):
    contact_object_list = _BASE.contact_object_list
    scene = _BASE.scene
    terminations = GenericTaskTerminations
    events = _BASE.events
    instruction = {"default": _SPEC["instruction"]}
    episode_length_s: int = _BASE.episode_length_s
    attributes = _BASE.attributes
    subtasks = None


del _BASE

"""Task specs: a task written as data on top of an existing task file.

A spec is a JSON file:
  id           short name, unique within a run
  base_task    task file under robolab/tasks whose scene, objects, start events and episode length are reused
  instruction  the language instruction given to the policy
  success      {"predicate": <function in curriculum_conditionals>, "args": {...}}
  episode_s    optional; episode length in seconds, default the base task's
  init_states  optional; end-state dirs to start episodes from, default the base task's own start events
"""
import inspect
import json
import os

from isaaclab.managers import TerminationTermCfg as DoneTerm

from robolab.core.task import curriculum_conditionals

REQUIRED = ("id", "base_task", "instruction", "success")
OPTIONAL = ("episode_s", "init_states")
OBJECT_ARGS = {"object", "object1", "object2", "objects", "container", "surface", "reference_object",
               "reference_obj1", "reference_obj2", "require_contact_with"}


def load_spec(path):
    with open(path) as f:
        return json.load(f)


def base_task_class(spec):
    from robolab.constants import TASK_DIR
    from robolab.core.task.task_utils import load_task_from_file, resolve_task_path

    path, _ = resolve_task_path(spec["base_task"], TASK_DIR)
    return load_task_from_file(path)


def predicates():
    out = {}
    for name, fn in vars(curriculum_conditionals).items():
        if name.startswith(("_", "get_")) or not inspect.isfunction(fn):
            continue
        if fn.__module__ != curriculum_conditionals.__name__:
            continue
        params = list(inspect.signature(fn).parameters)
        if params and params[0] == "env":
            out[name] = fn
    return out


def _names(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    return []


def validate_spec(spec, objects):
    errors = [f"missing field '{k}'" for k in REQUIRED if k not in spec]
    errors += [f"unknown field '{k}'" for k in spec if k not in REQUIRED + OPTIONAL]
    if any(k not in spec for k in REQUIRED):
        raise ValueError(f"spec {spec.get('id', '?')}: " + "; ".join(errors))
    success = spec["success"]
    fn = predicates().get(success.get("predicate"))
    if fn is None:
        errors.append(f"unknown predicate '{success.get('predicate')}'")
    else:
        args = success.get("args", {})
        try:
            inspect.signature(fn).bind(None, **args)
        except TypeError as e:
            errors.append(f"bad arguments for {success['predicate']}: {e}")
        for k, v in args.items():
            if k in OBJECT_ARGS:
                errors += [f"unknown object '{n}' in {k}" for n in _names(v) if n not in objects]
    init_states = spec.get("init_states")
    if init_states is not None:
        if not isinstance(init_states, list) or not init_states:
            errors.append("init_states must be a non-empty list of end-state dirs")
        else:
            errors += [f"init_states dir {d} does not exist" for d in init_states if not os.path.isdir(d)]
    if errors:
        raise ValueError(f"spec {spec['id']}: " + "; ".join(errors))


def success_term(spec):
    return DoneTerm(func=predicates()[spec["success"]["predicate"]], params=dict(spec["success"].get("args", {})))

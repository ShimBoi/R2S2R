"""Task specs: a task written as data on top of an existing task file.

A spec is a JSON file:
  base_task    file name of a task under robolab/tasks whose scene, objects and start events are reused
  instruction  the language instruction given to the policy
  success      {"predicate": <function in curriculum_conditionals>, "args": {...}}
  init_states  optional; end-state dirs to start episodes from, default the base task's own start events

validate_spec needs only the standard library, so a spec can be checked before any simulator starts.
"""
import ast
import glob
import json
import math
import os
from pathlib import Path

from robolab.constants import TASK_DIR

REQUIRED = ("base_task", "instruction", "success")
OPTIONAL = ("init_states",)
SUCCESS_KEYS = ("predicate", "args")
PREDICATES_FILE = Path(__file__).resolve().parent / "curriculum_conditionals.py"
RESERVED_ARGS = {"env", "env_id"}
SUBJECT_ARGS = {"object", "object1", "objects"}
REFERENCE_ARGS = {"object2", "container", "surface", "reference_object", "reference_obj1", "reference_obj2"}
OBJECT_ARGS = SUBJECT_ARGS | REFERENCE_ARGS | {"require_contact_with", "ignore_objects", "table_name"}
GRIPPER_ALLOWED = {"object1", "object2", "require_contact_with"}
CHOICES = {
    "logical": ("all", "any", "choose"),
    "mode": ("bbox", "centroid"),
    "up_axis": ("x", "y", "z"),
    "axis": ("x", "y", "z", None),
    "order": ("bottom_to_top", "top_to_bottom", None),
    "gripper_name": ("gripper",),
    "robot_name": ("robot",),
}
FRAMES = {"world", "robot"}
MIN_OBJECTS = {"stacked": 2, "objects_in_line": 2}
GROUP_KEYS = ("object", "container", "tolerance", "logical", "K", "require_contact_with", "require_gripper_detached")
GROUP_REQUIRED = ("object", "container")


def _no_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ValueError(f"duplicate keys {duplicates}")
    return dict(pairs)


def _no_constant(name):
    raise ValueError(f"{name} is not allowed in a spec")


def load_spec(path):
    with open(path) as f:
        return json.load(f, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant)


def base_task_path(base_task):
    if not isinstance(base_task, str) or not base_task.endswith(".py") or os.sep in base_task or any(
            c in base_task for c in "*?["):
        raise ValueError(f"base_task must be a task file name like name_task.py, got {base_task!r}")
    matches = sorted(p for p in Path(TASK_DIR).rglob(base_task) if p.is_file())
    if len(matches) != 1:
        raise ValueError(f"base_task {base_task}: {len(matches)} matching files under {TASK_DIR}")
    return matches[0]


def base_task_objects(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "contact_object_list" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"base_task {path.name}: no literal contact_object_list")


def predicate_signatures():
    out = {}
    for node in ast.parse(PREDICATES_FILE.read_text()).body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith(("_", "get_")):
            continue
        params = node.args.args
        if not params or params[0].arg != "env":
            continue
        defaults = [None] * (len(params) - len(node.args.defaults)) + list(node.args.defaults)
        out[node.name] = {p.arg: (p.annotation, d) for p, d in zip(params, defaults)}
    return out


SIMPLE_TYPES = {
    "str": lambda v: isinstance(v, str),
    "bool": lambda v: isinstance(v, bool),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
    "dict": lambda v: isinstance(v, dict),
}


def _matches(value, ann):
    if isinstance(ann, ast.Constant) and ann.value is None:
        return value is None
    if isinstance(ann, ast.Name) and ann.id in SIMPLE_TYPES:
        return SIMPLE_TYPES[ann.id](value)
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        return _matches(value, ann.left) or _matches(value, ann.right)
    if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name):
        args = ann.slice.elts if isinstance(ann.slice, ast.Tuple) else [ann.slice]
        if ann.value.id == "Union":
            return any(_matches(value, a) for a in args)
        if ann.value.id == "list" and len(args) == 1:
            return isinstance(value, list) and all(_matches(v, args[0]) for v in value)
        if ann.value.id == "tuple":
            return isinstance(value, list) and len(value) == len(args) and all(
                _matches(v, a) for v, a in zip(value, args))
    raise ValueError(f"unsupported annotation {ast.unparse(ann)}")


def _names(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def _check_arg(predicate, name, value, ann, default, objects):
    if value is None and isinstance(default, ast.Constant) and default.value is None:
        return []
    try:
        if not _matches(value, ann):
            return [f"{predicate}: argument '{name}' = {json.dumps(value)} does not match {ast.unparse(ann)}"]
    except ValueError as e:
        return [f"{predicate}: argument '{name}': {e}"]
    if name in CHOICES and value not in CHOICES[name]:
        return [f"{predicate}: argument '{name}' must be one of {list(CHOICES[name])}, got {json.dumps(value)}"]
    errors = []
    if name in OBJECT_ARGS and objects is not None:
        if isinstance(value, list) and not value:
            errors.append(f"{predicate}: argument '{name}' must not be empty")
        allowed = objects | {"gripper"} if name in GRIPPER_ALLOWED else objects
        errors += [f"{predicate}: unknown object '{n}' in {name}" for n in _names(value) if n not in allowed]
    if name == "frame_of_reference" and objects is not None and value not in objects | FRAMES:
        errors.append(f"{predicate}: frame_of_reference must be one of {sorted(FRAMES)} or an object, got {value!r}")
    return errors


def _check_groups(groups, params, objects):
    errors = []
    for i, group in enumerate(groups):
        where = f"object_groups_in_containers groups[{i}]"
        if not isinstance(group, dict):
            errors.append(f"{where} must be an object")
            continue
        errors += [f"{where}: unknown key '{k}'" for k in group if k not in GROUP_KEYS]
        errors += [f"{where}: missing key '{k}'" for k in GROUP_REQUIRED if k not in group]
        for k, v in group.items():
            if k in GROUP_KEYS:
                errors += _check_arg(where, k, v, params[k][0], params[k][1], objects)
    return errors


def _check_success(success, objects):
    if not isinstance(success, dict) or set(success) != set(SUCCESS_KEYS):
        return [f"success must have exactly the keys {list(SUCCESS_KEYS)}"]
    predicate, args = success["predicate"], success["args"]
    if not isinstance(predicate, str):
        return [f"success.predicate must be a predicate name, got {json.dumps(predicate)}"]
    if not isinstance(args, dict):
        return ["success.args must be an object"]
    signatures = predicate_signatures()
    params = signatures.get(predicate)
    if params is None:
        return [f"unknown predicate '{predicate}'"]
    errors = [f"{predicate}: {k} cannot be set in a spec" for k in args if k in RESERVED_ARGS]
    errors += [f"{predicate}: unknown argument '{k}'" for k in args if k not in params]
    errors += [f"{predicate}: missing argument '{k}'" for k, (_, default) in params.items()
               if k not in RESERVED_ARGS and default is None and k not in args]
    for k, v in args.items():
        if k in params and k not in RESERVED_ARGS:
            errors += _check_arg(predicate, k, v, params[k][0], params[k][1], objects)
    minimum = MIN_OBJECTS.get(predicate, 0)
    if isinstance(args.get("objects"), list) and len(args["objects"]) < minimum:
        errors.append(f"{predicate}: needs at least {minimum} objects")
    subjects = {n for k in SUBJECT_ARGS & set(args) for n in _names(args[k])}
    references = {n for k in REFERENCE_ARGS & set(args) for n in _names(args[k])}
    errors += [f"{predicate}: '{n}' is both the object and the reference" for n in sorted(subjects & references)]
    if predicate == "object_groups_in_containers" and isinstance(args.get("groups"), list):
        if not args["groups"]:
            errors.append(f"{predicate}: groups must not be empty")
        errors += _check_groups(args["groups"], signatures["object_in_container"], objects)
    return errors


def _check_init_states(init_states):
    if not isinstance(init_states, list) or not init_states or not all(isinstance(d, str) for d in init_states):
        return ["init_states must be a non-empty list of end-state dirs"]
    errors = []
    for d in init_states:
        if not os.path.isdir(d):
            errors.append(f"init_states dir {d} does not exist")
        elif not glob.glob(os.path.join(d, "*.npz")) or not os.path.isfile(os.path.join(d, "names.json")):
            errors.append(f"init_states dir {d} has no saved end states")
    return errors


def validate_spec(spec, source="spec"):
    if not isinstance(spec, dict):
        raise ValueError(f"{source}: a spec must be a JSON object")
    errors = [f"missing field '{k}'" for k in REQUIRED if k not in spec]
    errors += [f"unknown field '{k}'" for k in spec if k not in REQUIRED + OPTIONAL]
    if errors:
        raise ValueError(f"{source}: " + "; ".join(errors))
    if not isinstance(spec["instruction"], str) or not spec["instruction"].strip():
        errors.append("instruction must be a non-empty string")
    objects = None
    try:
        objects = set(base_task_objects(base_task_path(spec["base_task"])))
    except (ValueError, SyntaxError) as e:
        errors.append(str(e))
    errors += _check_success(spec["success"], objects)
    if "init_states" in spec:
        errors += _check_init_states(spec["init_states"])
    if errors:
        raise ValueError(f"{source}: " + "; ".join(errors))

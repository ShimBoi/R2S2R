"""Task specs: a task written as data on top of an existing task file.

A spec is a JSON file:
  base_task    file name of a task under robolab/tasks whose scene, objects and start events are reused
  instruction  the language instruction given to the policy
  success      {"predicate": <function in curriculum_conditionals>, "args": {...}}
  init_states  optional; end-state dirs to start episodes from, default the base task's own start events

A run takes one spec or a list of specs on the same base task. The envs of each env worker are split into
equal contiguous blocks, one per spec, in list order (spec_of_env).

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
GENERIC_TASK_FILE = Path(__file__).resolve().parent / "generic_task.py"
RESERVED_ARGS = {"env", "env_id"}
SUBJECT_ARGS = ("object", "object1", "objects")
REFERENCE_ARGS = ("object2", "container", "surface", "reference_object", "reference_obj1", "reference_obj2")
ROLE_ARGS = SUBJECT_ARGS + REFERENCE_ARGS
OBJECT_ARGS = set(ROLE_ARGS) | {"require_contact_with", "ignore_objects", "table_name"}
MAY_BE_EMPTY = {"ignore_objects"}
GRIPPER_ALLOWED = {"object1", "object2", "require_contact_with"}
CONTACT_TARGET_ARGS = {"container", "reference_object"}
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
INT_LIMIT = 2**63


def _no_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ValueError(f"duplicate keys {duplicates}")
    return dict(pairs)


def _no_constant(name):
    raise ValueError(f"{name} is not allowed in a spec")


def load_spec(path):
    try:
        with open(path) as f:
            return json.load(f, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e


def base_task_path(base_task):
    if not isinstance(base_task, str):
        raise ValueError(f"base_task must be a task file name, got {json.dumps(base_task)}")
    matches = sorted(p for p in Path(TASK_DIR).rglob("*.py") if p.name == base_task)
    if len(matches) != 1:
        raise ValueError(f"base_task {json.dumps(base_task)}: {len(matches)} task files with that name "
                         f"under {TASK_DIR}")
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
        a = node.args
        if not a.args or a.args[0].arg != "env":
            continue
        if a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg:
            out[node.name] = "uses positional-only, keyword-only or variadic parameters, which a spec cannot set"
            continue
        defaults = [None] * (len(a.args) - len(a.defaults)) + list(a.defaults)
        out[node.name] = {p.arg: (p.annotation, d) for p, d in zip(a.args, defaults)}
    return out


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and abs(v) < INT_LIMIT


def _is_float(v):
    return _is_int(v) or (isinstance(v, float) and math.isfinite(v))


SIMPLE_TYPES = {
    "str": lambda v: isinstance(v, str),
    "bool": lambda v: isinstance(v, bool),
    "int": _is_int,
    "float": _is_float,
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
        kind = ann.value.id
        if kind == "Union":
            return any(_matches(value, a) for a in args)
        if kind == "Optional" and len(args) == 1:
            return value is None or _matches(value, args[0])
        if kind == "Literal" and all(isinstance(a, ast.Constant) for a in args):
            return any(type(value) is type(a.value) and value == a.value for a in args)
        if kind == "list" and len(args) == 1:
            return isinstance(value, list) and all(_matches(v, args[0]) for v in value)
        if kind == "tuple":
            return isinstance(value, list) and len(value) == len(args) and all(
                _matches(v, a) for v, a in zip(value, args))
    raise ValueError(f"unsupported annotation {ast.unparse(ann)}")


def _names(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def _check_arg(where, name, value, ann, default, objects):
    if value is None and isinstance(default, ast.Constant) and default.value is None:
        return []
    if ann is None:
        return [f"{where}: argument '{name}' has no type annotation, so a spec cannot set it"]
    try:
        if not _matches(value, ann):
            return [f"{where}: argument '{name}' = {json.dumps(value)} does not match {ast.unparse(ann)}"]
    except ValueError as e:
        return [f"{where}: argument '{name}': {e}"]
    is_literal = isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name) and ann.value.id == "Literal"
    if name in CHOICES and not is_literal and value not in CHOICES[name]:
        return [f"{where}: argument '{name}' must be one of {json.dumps(list(CHOICES[name]))}, got {json.dumps(value)}"]
    errors = []
    if name in OBJECT_ARGS and objects is not None:
        names = _names(value)
        if isinstance(value, list) and not value and name not in MAY_BE_EMPTY:
            errors.append(f"{where}: argument '{name}' must not be empty")
        if len(set(names)) != len(names):
            errors.append(f"{where}: argument '{name}' lists an object more than once")
        allowed = objects | {"gripper"} if name in GRIPPER_ALLOWED else objects
        errors += [f"{where}: unknown object {json.dumps(n)} in {name}" for n in names if n not in allowed]
    if name == "frame_of_reference" and objects is not None and value not in objects | FRAMES:
        errors.append(f"{where}: argument 'frame_of_reference' must be one of {json.dumps(sorted(FRAMES))} "
                      f"or an object, got {json.dumps(value)}")
    return errors


def _check_roles(where, args, params):
    errors = []
    seen = {}
    for k in ROLE_ARGS:
        for n in set(_names(args.get(k))):
            if n in seen:
                errors.append(f"{where}: {json.dumps(n)} is given as both {seen[n]} and {k}")
            else:
                seen[n] = k
    contact = args.get("require_contact_with")
    subjects = {n for k in SUBJECT_ARGS for n in _names(args.get(k))}
    errors += [f"{where}: require_contact_with names {json.dumps(n)}, which is the object itself"
               for n in sorted(subjects & set(_names(contact)))]
    if contact is True and not CONTACT_TARGET_ARGS & set(params):
        errors.append(f"{where}: require_contact_with true needs a container or reference_object, which this "
                      f"predicate has none of")
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
        errors += _check_roles(where, group, params)
    return errors


def _check_success(success, objects):
    if not isinstance(success, dict) or set(success) != set(SUCCESS_KEYS):
        return [f"success must have exactly the keys {json.dumps(list(SUCCESS_KEYS))}"]
    predicate, args = success["predicate"], success["args"]
    if not isinstance(predicate, str):
        return [f"success.predicate must be a predicate name, got {json.dumps(predicate)}"]
    if not isinstance(args, dict):
        return ["success.args must be an object"]
    signatures = predicate_signatures()
    params = signatures.get(predicate)
    if params is None:
        return [f"unknown predicate {json.dumps(predicate)}"]
    if isinstance(params, str):
        return [f"{predicate}: {params}"]
    errors = [f"{predicate}: argument '{k}' cannot be set in a spec" for k in args if k in RESERVED_ARGS]
    errors += [f"{predicate}: unknown argument '{k}'" for k in args if k not in params]
    errors += [f"{predicate}: missing argument '{k}'" for k, (_, default) in params.items()
               if k not in RESERVED_ARGS and default is None and k not in args]
    for k, v in args.items():
        if k in params and k not in RESERVED_ARGS:
            errors += _check_arg(predicate, k, v, params[k][0], params[k][1], objects)
    minimum = MIN_OBJECTS.get(predicate, 0)
    if isinstance(args.get("objects"), list) and len(args["objects"]) < minimum:
        errors.append(f"{predicate}: needs at least {minimum} objects")
    errors += _check_roles(predicate, args, params)
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


def load_specs(paths):
    paths = [paths] if isinstance(paths, str) else list(paths)
    if not paths:
        raise ValueError("task_spec must name at least one spec")
    names = [Path(p).stem for p in paths]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"task_spec: more than one spec file named {duplicates}")
    return names, [load_spec(p) for p in paths]


def validate_specs(paths):
    paths = [paths] if isinstance(paths, str) else list(paths)
    names, specs = load_specs(paths)
    for path, spec in zip(paths, specs):
        validate_spec(spec, source=path)
    base_tasks = sorted({spec["base_task"] for spec in specs})
    if len(base_tasks) > 1:
        raise ValueError(f"task_spec: all specs in a run must share one base_task, got {base_tasks}")
    return names, specs


def spec_of_env(num_envs, num_specs):
    if num_envs % num_specs:
        raise ValueError(f"{num_envs} envs per env worker cannot be split evenly over {num_specs} task specs")
    return [i // (num_envs // num_specs) for i in range(num_envs)]

"""Task specs: a task written as data on top of an existing task file.

A spec is a JSON file:
  base_task    task file under robolab/tasks whose scene, objects and start events are reused
  instruction  the language instruction given to the policy
  success      {"predicate": <function in curriculum_conditionals>, "args": {...}}
  init_states  optional; end-state dirs to start episodes from, default the base task's own start events

validate_spec needs only the standard library, so a spec can be checked before any simulator starts.
"""
import ast
import glob
import json
import os
from pathlib import Path

REQUIRED = ("base_task", "instruction", "success")
OPTIONAL = ("init_states",)
SUCCESS_KEYS = ("predicate", "args")
TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks"
PREDICATES_FILE = Path(__file__).resolve().parent / "curriculum_conditionals.py"
OBJECT_ARGS = {"object", "object1", "object2", "objects", "container", "surface", "reference_object",
               "reference_obj1", "reference_obj2", "require_contact_with", "intended_objects", "ignore_objects",
               "table_name"}
GROUP_OBJECT_KEYS = ("object", "container", "require_contact_with")
NON_OBJECT_CONTACTS = {"gripper"}
RESERVED_ARGS = {"env", "env_id"}


def load_spec(path):
    with open(path) as f:
        return json.load(f)


def base_task_path(base_task):
    direct = TASKS_DIR / base_task
    if direct.is_file():
        return direct
    matches = sorted(TASKS_DIR.rglob(base_task))
    if len(matches) != 1:
        raise ValueError(f"base_task {base_task}: {len(matches)} matching files under {TASKS_DIR}")
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
        params = node.args.args + node.args.kwonlyargs
        if not params or params[0].arg != "env":
            continue
        defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + list(node.args.defaults)
        defaults += list(node.args.kw_defaults)
        out[node.name] = {p.arg: (p.annotation, d is not None) for p, d in zip(params, defaults)}
    return out


def _matches(value, ann):
    if ann is None:
        return True
    if isinstance(ann, ast.Constant):
        return value is None if ann.value is None else value == ann.value
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        return _matches(value, ann.left) or _matches(value, ann.right)
    if isinstance(ann, ast.Name):
        return {
            "str": lambda v: isinstance(v, str),
            "bool": lambda v: isinstance(v, bool),
            "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "dict": lambda v: isinstance(v, dict),
            "list": lambda v: isinstance(v, list),
            "tuple": lambda v: isinstance(v, list),
        }.get(ann.id, lambda v: True)(value)
    if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name):
        args = ann.slice.elts if isinstance(ann.slice, ast.Tuple) else [ann.slice]
        kind = ann.value.id
        if kind in ("Union", "Optional"):
            return (kind == "Optional" and value is None) or any(_matches(value, a) for a in args)
        if kind == "Literal":
            return any(isinstance(a, ast.Constant) and value == a.value for a in args)
        if kind == "list":
            return isinstance(value, list) and all(_matches(v, args[0]) for v in value)
        if kind == "tuple":
            if not isinstance(value, list):
                return False
            if len(args) == 2 and isinstance(args[1], ast.Constant) and args[1].value is Ellipsis:
                return all(_matches(v, args[0]) for v in value)
            return len(value) == len(args) and all(_matches(v, a) for v, a in zip(value, args))
        if kind == "dict":
            return isinstance(value, dict)
    return True


def _names(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def _check_names(arg, value, objects):
    if objects is None:
        return []
    allowed = objects | NON_OBJECT_CONTACTS if arg.endswith("require_contact_with") else objects
    return [f"unknown object '{n}' in {arg}" for n in _names(value) if n not in allowed]


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


def _check_success(success, objects):
    if not isinstance(success, dict) or set(success) != set(SUCCESS_KEYS):
        return [f"success must have exactly the keys {list(SUCCESS_KEYS)}"]
    if not isinstance(success["args"], dict):
        return ["success.args must be an object"]
    params = predicate_signatures().get(success["predicate"])
    if params is None:
        return [f"unknown predicate '{success['predicate']}'"]
    args = success["args"]
    errors = [f"{k} cannot be set in a spec" for k in args if k in RESERVED_ARGS]
    errors += [f"unknown argument '{k}' for {success['predicate']}" for k in args if k not in params]
    errors += [f"missing argument '{k}' for {success['predicate']}"
               for k, (_, has_default) in params.items()
               if k not in RESERVED_ARGS and not has_default and k not in args]
    for k, v in args.items():
        if k not in params or k in RESERVED_ARGS:
            continue
        if not _matches(v, params[k][0]):
            errors.append(f"argument '{k}' = {json.dumps(v)} does not match {ast.unparse(params[k][0])}")
        if k in OBJECT_ARGS:
            errors += _check_names(k, v, objects)
        if k == "groups" and isinstance(v, list):
            for group in v:
                for gk in GROUP_OBJECT_KEYS:
                    if isinstance(group, dict) and gk in group:
                        errors += _check_names(f"groups.{gk}", group[gk], objects)
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
    except (ValueError, SyntaxError, TypeError) as e:
        errors.append(str(e))
    errors += _check_success(spec["success"], objects)
    if "init_states" in spec:
        errors += _check_init_states(spec["init_states"])
    if errors:
        raise ValueError(f"{source}: " + "; ".join(errors))

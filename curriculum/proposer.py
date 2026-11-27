"""Runs the open-ended-task-proposal skill on one frontier node: writes the request file, calls the skill headless
with `claude -p`, and reads back the candidates that pass the spec checks.

Request file (written here, read by the skill):
  node            {"id", "level", "paths": [[instruction, ...] from s0, one list per path that reaches the node]}
  scene           {"base_task", "task_file", "objects"}: the base task every candidate runs on, and its objects
  predicates      the success predicates: name, arguments and a one-line meaning
  tree            every task proposed so far: instruction, start node, status, screened successes out of n with
                  the level and checkpoint they were measured at, and the held-out eval rate after training
  images          optional renders or photos of the node's start states
  goal            optional user goal
  min_candidates  propose at least this many
  response        where the skill writes its response

Response file (written by the skill):
  {"node": id, "candidates": [{"spec": {base_task, instruction, success}, "difficulty_guess": 1-5,
   "rationale": str}], "notes": str}
"""
import ast
import json
import subprocess
from pathlib import Path

from robolab.core.task.task_spec import (
    PREDICATES_FILE,
    RESERVED_ARGS,
    base_task_objects,
    base_task_path,
    validate_spec,
)

REPO = Path(__file__).resolve().parents[1]
SKILL = "open-ended-task-proposal"
DIFFICULTIES = (1, 2, 3, 4, 5)


def predicate_menu():
    """[{"name", "args", "meaning"}] for every predicate a spec can use."""
    menu = []
    for node in ast.parse(PREDICATES_FILE.read_text()).body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith(("_", "get_")):
            continue
        a = node.args
        if not a.args or a.args[0].arg != "env" or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg:
            continue
        defaults = [None] * (len(a.args) - len(a.defaults)) + list(a.defaults)
        args = []
        for p, d in zip(a.args, defaults):
            if p.arg in RESERVED_ARGS:
                continue
            s = p.arg + (f": {ast.unparse(p.annotation)}" if p.annotation else "")
            args.append(s + (f" = {ast.unparse(d)}" if d is not None else ""))
        doc = (ast.get_docstring(node) or "").strip().split("\n")[0]
        menu.append({"name": node.name, "args": ", ".join(args), "meaning": doc})
    return menu


def build_request(path, node, base_task, tree=(), images=(), goal=None, min_candidates=3, response=None):
    """Writes the request file for `node` ({"id", "level", "paths"}) and returns its dict."""
    task_file = base_task_path(base_task)
    path = Path(path)
    request = {
        "node": node,
        "scene": {"base_task": base_task, "task_file": str(task_file), "objects": base_task_objects(task_file)},
        "predicates": predicate_menu(),
        "tree": list(tree),
        "images": [str(i) for i in images],
        "goal": goal,
        "min_candidates": min_candidates,
        "response": str(response or path.with_name("response.json")),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, indent=1))
    return request


def parse(request, response_text):
    """(accepted, cut) from a response: accepted candidates have a spec that passes validate_spec on the
    request's base task; cut ones carry the reason. Raises ValueError if the response itself is malformed."""
    response = json.loads(response_text)
    if not isinstance(response, dict) or not isinstance(response.get("candidates"), list):
        raise ValueError('the response must be a JSON object with a "candidates" list')
    if response.get("node") != request["node"]["id"]:
        raise ValueError(f'response "node" is {response.get("node")!r}, expected {request["node"]["id"]!r}')
    accepted, cut = [], []
    for i, c in enumerate(response["candidates"]):
        try:
            if not isinstance(c, dict) or not isinstance(c.get("spec"), dict):
                raise ValueError('a candidate must be an object with a "spec"')
            spec = c["spec"]
            if spec.get("base_task") != request["scene"]["base_task"]:
                raise ValueError(f'base_task must be {request["scene"]["base_task"]!r}')
            if "init_states" in spec:
                raise ValueError("init_states is set by the orchestrator, not the proposer")
            if c.get("difficulty_guess") not in DIFFICULTIES:
                raise ValueError(f"difficulty_guess must be one of {DIFFICULTIES}")
            if not isinstance(c.get("rationale"), str) or not c["rationale"].strip():
                raise ValueError("rationale must be a non-empty string")
            validate_spec(spec, source=f"candidate {i}")
            accepted.append(c)
        except ValueError as e:
            cut.append({"candidate": c, "reason": str(e)})
    return accepted, cut


def _prompt(request_path, error=None):
    p = (f"Use the {SKILL} skill on the request file {request_path}. "
         "Write the response file it names and nothing else.")
    if error:
        p += f" Your previous response could not be used: {error}. Write the response file again."
    return p


def run(request_path, model=None, timeout=1800, repo=REPO):
    """Runs the skill headless on a request file; retries once with the parse error attached.
    Returns (accepted, cut, notes); an unusable second response counts as an empty proposal."""
    request = json.loads(Path(request_path).read_text())
    response_path = Path(request["response"])
    allowed = f"Read Glob Grep Edit(/{response_path.resolve()}) Edit({PREDICATES_FILE.relative_to(Path(repo).resolve())})"
    error = None
    for _ in range(2):
        response_path.unlink(missing_ok=True)
        cmd = ["claude", "-p", _prompt(request_path, error), "--allowedTools", allowed]
        if model:
            cmd += ["--model", model]
        subprocess.run(cmd, cwd=repo, timeout=timeout, check=False, capture_output=True, text=True)
        try:
            if not response_path.exists():
                raise ValueError(f"no response file at {response_path}")
            accepted, cut = parse(request, response_path.read_text())
            return accepted, cut, json.loads(response_path.read_text()).get("notes", "")
        except (ValueError, json.JSONDecodeError) as e:
            error = str(e)
    return [], [], f"no usable response: {error}"

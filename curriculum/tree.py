"""A curriculum run's event log and the tree rebuilt from it.

events.jsonl is append-only and the source of truth; tree.json is rebuilt from it after every event, so a
restarted orchestrator or a dashboard only ever reads these two files. The run directory sits on scratch;
run.json, events.jsonl, tree.json and whatever is passed to Run.keep (specs, proposals, node states, final
LoRAs) are also copied to a mirror directory in $HOME, which is backed up.

Events, one JSON line each: {"t", "event", "level", "node", "task", "data"}.
  run_started      data: run config (also written to run.json)
  level_started    level
  proposed         task, node (the start node), data: {spec (the spec itself), spec_path, instruction,
                   difficulty_guess, rationale}
  cut              task, data: {reason}
  screened         task, data: {successes, n, ckpt}
  blocked, mastered, picked, not_picked   task
  train_started    task, data: {ckpt_in}
  train_step       task, data: {step, success}
  train_done       task, data: {ckpt, end_states}
  evaluated        task, data: {rate, n}
  states_captured  task, data: {dir}: the task's end states; creates its end node, or merges into the node of
                   an earlier task with the same spec hash
  hold_submitted   data: {job}
  level_done       level
  error            task (optional), data: {message}
  real_result      task, data: free form

Nodes: s0 is the scene's own start layouts, with one empty path. Every other node is reached by tasks; its
paths are the instruction lists from s0, one per path, and its states are the dirs captured at its tasks' ends.
"""
import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path

EVENTS = {
    "run_started", "level_started", "proposed", "cut", "screened", "blocked", "mastered", "picked", "not_picked",
    "train_started", "train_step", "train_done", "evaluated", "states_captured", "hold_submitted", "level_done",
    "error", "real_result",
}
STATUS_EVENTS = {"cut": "cut", "screened": "screened", "blocked": "blocked", "mastered": "mastered",
                 "picked": "picked", "not_picked": "not_picked", "train_started": "training",
                 "train_done": "trained"}
SCRATCH_RUNS = Path(os.environ.get("SCRATCH", "/tmp")) / "r2s2r" / "curriculum_runs"
HOME_RUNS = Path.home() / "curriculum_runs"


def spec_hash(spec):
    """The task's identity: the spec without its start states, so the same task from two nodes hashes alike."""
    body = {k: v for k, v in spec.items() if k != "init_states"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rebuild(events):
    """The tree, {"run", "levels", "nodes", "tasks", "holds", "errors"}, from a list of events."""
    tree = {"run": {}, "levels": {}, "nodes": {"s0": {"id": "s0", "level": 0, "paths": [[]], "incoming": [],
                                                       "states": []}},
            "tasks": {}, "holds": [], "errors": []}
    nodes, tasks = tree["nodes"], tree["tasks"]
    end_node_of_hash = {}
    for e in events:
        kind, t, d = e["event"], e["t"], e.get("data") or {}
        task = tasks.get(e.get("task"))
        if kind == "run_started":
            tree["run"] = {"started": t, **d}
        elif kind == "level_started":
            tree["levels"][str(e["level"])] = {"started": t, "done": None}
        elif kind == "level_done":
            tree["levels"].setdefault(str(e["level"]), {"started": None})["done"] = t
        elif kind == "proposed":
            tasks[e["task"]] = {
                "id": e["task"], "level": e["level"], "start_node": e["node"], "end_node": None,
                "spec": d["spec"], "spec_path": d.get("spec_path"), "spec_hash": spec_hash(d["spec"]),
                "instruction": d["instruction"],
                "difficulty_guess": d.get("difficulty_guess"), "rationale": d.get("rationale"),
                "status": "proposed", "history": [{"status": "proposed", "t": t}],
                "cut_reason": None, "screen": None, "train": None, "eval": None, "real": [],
            }
        elif kind == "hold_submitted":
            tree["holds"].append({"t": t, **d})
        elif kind == "error":
            tree["errors"].append({"t": t, "task": e.get("task"), **d})
        elif task is None:
            raise ValueError(f"event {kind} names unknown task {e.get('task')!r}")
        if task is None:
            continue
        if kind in STATUS_EVENTS:
            task["status"] = STATUS_EVENTS[kind]
            task["history"].append({"status": task["status"], "t": t})
        if kind == "cut":
            task["cut_reason"] = d.get("reason")
        elif kind == "screened":
            task["screen"] = {"successes": d["successes"], "n": d["n"], "ckpt": d.get("ckpt")}
        elif kind == "train_started":
            task["train"] = {"ckpt_in": d.get("ckpt_in"), "steps": [], "ckpt": None, "end_states": None}
        elif kind == "train_step":
            task["train"]["steps"].append([d["step"], d["success"]])
        elif kind == "train_done":
            task["train"].update(ckpt=d.get("ckpt"), end_states=d.get("end_states"))
        elif kind == "evaluated":
            task["eval"] = {"rate": d["rate"], "n": d["n"]}
        elif kind == "real_result":
            task["real"].append({"t": t, **d})
        elif kind == "states_captured":
            start = nodes[task["start_node"]]
            node_id = end_node_of_hash.get(task["spec_hash"])
            if node_id is None:
                node_id = f"s{len(nodes)}"
                nodes[node_id] = {"id": node_id, "level": start["level"] + 1, "paths": [], "incoming": [],
                                  "states": []}
                end_node_of_hash[task["spec_hash"]] = node_id
            node = nodes[node_id]
            node["incoming"].append(task["id"])
            node["states"].append(d["dir"])
            for p in start["paths"]:
                path = p + [task["instruction"]]
                if path not in node["paths"]:
                    node["paths"].append(path)
            task["end_node"] = node_id
    for n in nodes.values():
        n["children"] = [t["id"] for t in tasks.values() if t["start_node"] == n["id"]]
    return tree


class Run:
    """One curriculum run: its directory on scratch and its mirror in $HOME."""

    def __init__(self, run_id, root=SCRATCH_RUNS, mirror=HOME_RUNS):
        self.dir = Path(root) / run_id
        self.mirror = Path(mirror) / run_id if mirror else None
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.dir / "events.jsonl"

    def events(self):
        if not self.events_file.exists():
            return []
        return [json.loads(line) for line in self.events_file.read_text().splitlines() if line.strip()]

    def tree(self):
        return rebuild(self.events())

    def log(self, event, level=None, node=None, task=None, **data):
        """Appends one event, rebuilds tree.json and refreshes the $HOME mirror. Returns the new tree."""
        if event not in EVENTS:
            raise ValueError(f"unknown event {event!r}")
        e = {"t": _now(), "event": event, "level": level, "node": node, "task": task, "data": data}
        tree = rebuild(self.events() + [e])
        with open(self.events_file, "a") as f:
            f.write(json.dumps(e) + "\n")
        if event == "run_started":
            (self.dir / "run.json").write_text(json.dumps(data, indent=1))
        tmp = self.dir / "tree.json.tmp"
        tmp.write_text(json.dumps(tree, indent=1))
        tmp.replace(self.dir / "tree.json")
        for name in ("run.json", "events.jsonl", "tree.json"):
            if (self.dir / name).exists():
                self.keep(self.dir / name)
        return tree

    def keep(self, path):
        """Copies a file or directory under the run directory to the same place in the $HOME mirror."""
        if self.mirror is None:
            return None
        path = Path(path)
        dest = self.mirror / path.resolve().relative_to(self.dir.resolve())
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(path, dest)
        return dest

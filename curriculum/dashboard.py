"""The dashboard's data: one JSON snapshot of a run's tree, and a wait for the run's next event.

  python -m curriculum.dashboard export <run_dir> <out.json>
  python -m curriculum.dashboard wait <run_dir> [quiet_seconds]

`wait` returns once events.jsonl has grown and then stayed unchanged for `quiet_seconds` (default 20), so a burst
of events (a level's proposals, a screening's results) gives one update.
"""
import json
import sys
import time
from pathlib import Path

from curriculum import runner, tree as tree_mod

RECENT_EVENTS = 40


def _summary(e):
    d = e.get("data") or {}
    kind = e["event"]
    if kind == "proposed":
        return d.get("instruction", "")
    if kind == "cut":
        return d.get("reason", "")
    if kind == "screened":
        return f"{d['successes']}/{d['n']}"
    if kind == "train_step":
        return f"step {d['step']}: {d['success']:.2f}"
    if kind == "evaluated":
        return f"{d['rate']:.2f} over {d['n']}"
    if kind == "error":
        return d.get("message", "")
    if kind == "hold_submitted":
        return f"job {d.get('job')}"
    return ""


def _stage(tree, events):
    if not events:
        return "not started"
    last = events[-1]["event"]
    if last == "error":
        return "stopped on error"
    tasks = list(tree["tasks"].values())
    if any(t["status"] == "training" for t in tasks):
        return "training"
    if last == "level_done":
        level = events[-1]["level"]
        ended = [t for t in tasks if t["level"] == level and t["end_node"]]
        return "finished" if level >= tree["run"].get("max_depth", level) or not ended else "proposing"
    return {"run_started": "proposing", "level_started": "proposing", "proposed": "screening", "cut": "screening",
            "screened": "picking", "blocked": "picking", "mastered": "picking", "picked": "training",
            "not_picked": "training", "train_done": "evaluating", "evaluated": "evaluating",
            "states_captured": "capturing states"}.get(last, last)


def snapshot(run_dir, hold_left=None):
    """{status, nodes, tasks, steps, events}: lists of flat rows for the dashboard."""
    run_dir = Path(run_dir)
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines() if l.strip()]
    tree = tree_mod.rebuild(events)
    levels = [int(k) for k in tree["levels"]]
    active = [t["id"] for t in tree["tasks"].values() if t["status"] == "training"]
    status = {
        "run_id": tree["run"].get("run_id"), "started": tree["run"].get("started"),
        "level": max(levels) if levels else None, "max_depth": tree["run"].get("max_depth"),
        "k": tree["run"].get("k"), "train_steps": tree["run"].get("train_steps"),
        "stage": _stage(tree, events), "active_task": active[0] if active else None,
        "last_event": events[-1]["event"] if events else None, "last_event_at": events[-1]["t"] if events else None,
        "hold_left_hours": None if hold_left is None else round(hold_left / 3600, 1),
        "errors": len(tree["errors"]), "tasks": len(tree["tasks"]), "nodes": len(tree["nodes"]),
        "exported_at": tree_mod._now(),
    }
    nodes = [{
        "id": n["id"], "level": n["level"], "path": " → ".join(n["paths"][0]) if n["paths"] and n["paths"][0] else "start",
        "paths": len(n["paths"]), "incoming": ", ".join(n["incoming"]), "children": len(n["children"]),
        "state_dirs": len(n["states"]),
    } for n in tree["nodes"].values()]
    tasks, steps = [], []
    for t in tree["tasks"].values():
        train = t["train"] or {}
        curve = train.get("steps") or []
        steps += [{"task": t["id"], "step": s, "success": v} for s, v in curve]
        tasks.append({
            "id": t["id"], "level": t["level"], "start_node": t["start_node"], "end_node": t["end_node"],
            "instruction": t["instruction"], "difficulty_guess": t["difficulty_guess"], "status": t["status"],
            "screen_successes": (t["screen"] or {}).get("successes"), "screen_n": (t["screen"] or {}).get("n"),
            "screen_rate": None if not t["screen"] else t["screen"]["successes"] / t["screen"]["n"],
            "train_steps_done": len(curve), "train_first": curve[0][1] if curve else None,
            "train_last": curve[-1][1] if curve else None,
            "eval_rate": (t["eval"] or {}).get("rate"), "eval_n": (t["eval"] or {}).get("n"),
            "cut_reason": t["cut_reason"], "rationale": t["rationale"],
        })
    recent = [{"t": e["t"], "event": e["event"], "level": e.get("level"), "task": e.get("task"),
               "node": e.get("node"), "detail": _summary(e)} for e in events[-RECENT_EVENTS:]][::-1]
    return {"status": [status], "nodes": nodes, "tasks": tasks, "steps": steps, "events": recent}


def _hold_left():
    try:
        running = [j for j in runner.hold_jobs() if j["state"] == "RUNNING" and j["left"] is not None]
    except Exception:
        return None
    return max((j["left"] for j in running), default=None)


def export(run_dir, out):
    Path(out).write_text(json.dumps(snapshot(run_dir, _hold_left())))
    return out


def wait(run_dir, quiet=20, poll=5):
    """Returns the number of new event lines once events.jsonl has grown and then been unchanged for `quiet` s."""
    f = Path(run_dir) / "events.jsonl"
    count = lambda: len(f.read_text().splitlines()) if f.exists() else 0
    start = last = count()
    changed = None
    while True:
        time.sleep(poll)
        now = count()
        if now != last:
            last, changed = now, time.time()
        elif changed and time.time() - changed >= quiet:
            return last - start


def main(argv):
    if len(argv) >= 3 and argv[0] == "export":
        print(export(argv[1], argv[2]))
        return 0
    if len(argv) >= 2 and argv[0] == "wait":
        print(f"{wait(argv[1], *(int(a) for a in argv[2:3]))} new events")
        return 0
    print(__doc__.strip())
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

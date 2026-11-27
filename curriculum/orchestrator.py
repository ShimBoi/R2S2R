"""The curriculum level loop: propose, screen, pick, train, evaluate, capture, one level at a time.

    python -m curriculum.orchestrator run.json

run.json holds the config (DEFAULTS below lists every key). The run lives in a curriculum.tree.Run: every step
is an event in events.jsonl, and tree.json is rebuilt from it. Starting the orchestrator again on the same run
id replays the log and continues where it stopped: finished stages are skipped, and a task whose training was
cut off resumes from its last checkpoint with the steps it still has.

Each level L (tasks at level L start from the nodes at level L - 1; s0 is level 0):
  1. Propose: one proposal-skill call per frontier node. Each accepted spec gets an id and a file
     tasks/<id>/<id>.json with the node's state dirs as init_states (s0 has none). A spec already proposed
     from the same node is cut as a duplicate.
  2. Screen: every proposed task in one eval run, `screen_episodes` episodes each, with the current LoRA.
     0 successes: blocked; all: mastered; otherwise trainable.
  3. Pick: the k trainable tasks whose screened rate is closest to 50% (ties: spec hash); the rest not_picked.
  4. Train the picked tasks one after another, each from the LoRA the previous one ended on. Training success
     is read from the run's tensorboard every `poll` seconds and logged as train_step events.
  5. Evaluate every trained and mastered task with the level's final LoRA; mastered tasks save end states.
  6. Capture: each trained or mastered task's end states make (or merge into) its end node, the next frontier.
The run stops at max_depth, or when a level trains and masters nothing. A GPU job that fails is retried once
(from its last checkpoint for training); a second failure logs an error and stops the orchestrator.
"""
import glob
import json
import os
import sys
import threading
import time
from pathlib import Path

from curriculum import jobs, proposer, runner
from curriculum.tree import Run, spec_hash

DEFAULTS = {
    "run_id": None,
    "base_task": "mug_on_cutting_board_task.py",
    "k": 3,
    "max_depth": 3,
    "train_steps": 25,
    "save_every": 5,
    "screen_episodes": 16,
    "eval_episodes": 80,
    "eval_epochs": 1,
    "min_candidates": 3,
    "goal": None,
    "init_ckpt": None,
    "hold_script": "/scratch/11369/jshim12/r2s2r/hold.sbatch",
    "renew_below": 3 * 3600,
    "repo": str(runner.REPO),
    "venv": None,
    "model": None,
    "poll": 300,
}


class Stop(Exception):
    pass


def load_config(path):
    cfg = dict(DEFAULTS)
    cfg.update(json.loads(Path(path).read_text()))
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}")
    if not cfg["run_id"]:
        raise ValueError("run_id is required")
    return cfg


class Orchestrator:
    def __init__(self, cfg, run=None, propose=proposer.run, run_job=runner.run, hold=runner.ensure_hold,
                 sleep=time.sleep):
        self.cfg = cfg
        self.run = run or Run(cfg["run_id"])
        self._propose, self._run_job, self._hold, self._sleep = propose, run_job, hold, sleep

    # ---- helpers

    def tree(self):
        return self.run.tree()

    def log(self, event, **kw):
        return self.run.log(event, **kw)

    def current_ckpt(self):
        ckpt = self.cfg["init_ckpt"]
        for e in self.run.events():
            if e["event"] == "train_done" and e["data"].get("ckpt"):
                ckpt = e["data"]["ckpt"]
        return ckpt

    def gpu(self, job, task=None, what="job"):
        """Runs a GPU job on the hold node; retries once; logs an error and stops on a second failure."""
        for attempt in (1, 2):
            node, submitted = self._hold(self.cfg["hold_script"], renew_below=self.cfg["renew_below"])
            if submitted:
                self.log("hold_submitted", job=submitted)
            rc = self._run_job(job, node, repo=self.cfg["repo"], venv=self.cfg["venv"])
            if rc == 0:
                return
            self.log("error", task=task, message=f"{what} failed with exit code {rc} on {node} "
                                                   f"(attempt {attempt}); log: {job['out']}/job.log")
        raise Stop(f"{what} failed twice")

    @staticmethod
    def level_tasks(tree, level):
        return sorted((t for t in tree["tasks"].values() if t["level"] == level), key=lambda t: t["id"])

    # ---- stages

    def propose(self, level, node):
        tree = self.tree()
        out = self.run.dir / "levels" / str(level) / "proposals" / node["id"]
        summary = [{"id": t["id"], "instruction": t["instruction"], "success": t["spec"].get("success"),
                    "start_node": t["start_node"], "status": t["status"],
                    "screened": dict(t["screen"], level=t["level"]) if t["screen"] else None,
                    "eval": t["eval"]["rate"] if t["eval"] else None}
                   for t in tree["tasks"].values()]
        proposer.build_request(out / "request.json", {"id": node["id"], "level": node["level"],
                                                       "paths": node["paths"]},
                               self.cfg["base_task"], tree=summary, goal=self.cfg["goal"],
                               min_candidates=self.cfg["min_candidates"])
        accepted, cut, notes = self._propose(out / "request.json", model=self.cfg["model"])
        for f in ("request.json", "response.json"):
            if (out / f).exists():
                self.run.keep(out / f)
        if notes.startswith("no usable response"):
            self.log("error", message=f"proposal for {node['id']} at level {level}: {notes}")
        seen = {t["spec_hash"] for t in tree["tasks"].values() if t["start_node"] == node["id"]}
        candidates = [(c, None) for c in accepted] + [(c["candidate"], c["reason"]) for c in cut]
        for cand, reason in candidates:
            spec = cand.get("spec") if isinstance(cand, dict) else None
            if not isinstance(spec, dict) or not isinstance(spec.get("instruction"), str):
                self.log("error", message=f"unusable candidate from {node['id']}: {reason}")
                continue
            if node["states"]:
                spec = dict(spec, init_states=list(node["states"]))
            tid = f"t{len(self.tree()['tasks']) + 1:03d}"
            path = self.run.dir / "tasks" / tid / f"{tid}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(spec, indent=1))
            self.run.keep(path)
            self.log("proposed", level=level, node=node["id"], task=tid, spec=spec, spec_path=str(path),
                     instruction=spec["instruction"], difficulty_guess=cand.get("difficulty_guess"),
                     rationale=cand.get("rationale"))
            h = spec_hash(spec)
            if reason:
                self.log("cut", level=level, task=tid, reason=reason)
            elif h in seen:
                self.log("cut", level=level, task=tid, reason="duplicate of a task already proposed from this node")
            seen.add(h)

    def screen(self, level):
        todo = [t for t in self.level_tasks(self.tree(), level) if t["status"] == "proposed"]
        if not todo:
            return
        n = self.cfg["screen_episodes"]
        out = self.run.dir / "levels" / str(level) / "screen"
        ckpt = self.current_ckpt()
        specs = [t["spec_path"] for t in todo]
        job = jobs.screen(specs, out, ckpt=ckpt, episodes=n)
        self.gpu(job, what=f"screening level {level}")
        results = jobs.screen_results(out, specs, episodes=n)
        for t in todo:
            k = results[t["id"]]
            self.log("screened", level=level, task=t["id"], successes=k, n=n, ckpt=ckpt)
            if k == 0:
                self.log("blocked", level=level, task=t["id"])
            elif k == n:
                self.log("mastered", level=level, task=t["id"])

    def pick(self, level):
        tasks = self.level_tasks(self.tree(), level)
        if any(t["status"] in ("picked", "not_picked", "training", "trained") for t in tasks):
            return
        trainable = [t for t in tasks if t["status"] == "screened"]
        trainable.sort(key=lambda t: (abs(t["screen"]["successes"] / t["screen"]["n"] - 0.5), t["spec_hash"]))
        for i, t in enumerate(trainable):
            self.log("picked" if i < self.cfg["k"] else "not_picked", level=level, task=t["id"])

    @staticmethod
    def _checkpoints(task_dir):
        found = []
        for p in glob.glob(os.path.join(task_dir, "train*", "*", "checkpoints", "global_step_*")):
            found.append((os.path.dirname(os.path.dirname(os.path.dirname(p))), int(p.rsplit("_", 1)[-1]), p))
        return found

    def train(self, level, task, attempt=1):
        tid, task_dir = task["id"], self.run.dir / "tasks" / task["id"]
        steps, segments = self.cfg["train_steps"], sorted(glob.glob(str(task_dir / "train*")))
        done, ckpt_in = 0, self.current_ckpt()
        if task["status"] == "picked":
            self.log("train_started", level=level, task=tid, ckpt_in=ckpt_in)
        else:
            for seg in segments:
                ck = [c for c in self._checkpoints(task_dir) if c[0] == seg]
                if ck:
                    last = max(ck, key=lambda c: c[1])
                    done, ckpt_in = done + last[1], os.path.join(last[2], "actor")
        logged = len(task["train"]["steps"]) if task["train"] else 0
        if done >= steps:
            last = max(self._checkpoints(task_dir), key=lambda c: (c[0], c[1]))
            res = jobs.train_results(last[0])
            self.run.keep(res["ckpt"])
            self.log("train_done", level=level, task=tid, ckpt=res["ckpt"], end_states=res["end_states"])
            return
        out = task_dir / ("train" if not segments else f"train_r{len(segments)}")
        job = jobs.train(task["spec_path"], out, steps=steps - done, ckpt=ckpt_in, save_every=self.cfg["save_every"])

        def follow():
            nonlocal logged
            try:
                curve = sorted(jobs.scalars(str(out)).get("env/success_once", []))
            except FileNotFoundError:
                return
            for step, value in curve:
                if done + step + 1 > logged:
                    self.log("train_step", level=level, task=tid, step=done + step + 1, success=value)
                    logged = done + step + 1

        node, submitted = self._hold(self.cfg["hold_script"], renew_below=self.cfg["renew_below"])
        if submitted:
            self.log("hold_submitted", job=submitted)
        result = {}
        worker = threading.Thread(target=lambda: result.update(
            rc=self._run_job(job, node, repo=self.cfg["repo"], venv=self.cfg["venv"])))
        worker.start()
        while worker.is_alive():
            worker.join(self.cfg["poll"])
            follow()
        follow()
        if result.get("rc") != 0:
            self.log("error", task=tid, message=f"training {tid} failed with exit code {result.get('rc')} on "
                                                f"{node} (attempt {attempt}); log: {out}/job.log")
            if attempt == 2:
                raise Stop(f"training {tid} failed twice")
            return self.train(level, self.tree()["tasks"][tid], attempt=2)
        res = jobs.train_results(str(out))
        self.run.keep(res["ckpt"])
        self.log("train_done", level=level, task=tid, ckpt=res["ckpt"], end_states=res["end_states"])

    def evaluate(self, level):
        ckpt = self.current_ckpt()
        for t in self.level_tasks(self.tree(), level):
            if t["status"] not in ("trained", "mastered") or t["eval"]:
                continue
            out = self.run.dir / "tasks" / t["id"] / "eval"
            states = out / "end_states" if t["status"] == "mastered" else None
            job = jobs.evaluate(t["spec_path"], out, ckpt=ckpt, episodes=self.cfg["eval_episodes"],
                                epochs=self.cfg["eval_epochs"], save_states=states)
            self.gpu(job, task=t["id"], what=f"eval of {t['id']}")
            rate, n = jobs.eval_results(str(out))
            self.log("evaluated", level=level, task=t["id"], rate=rate, n=n)

    def capture(self, level):
        for t in self.level_tasks(self.tree(), level):
            if t["status"] not in ("trained", "mastered") or t["end_node"]:
                continue
            if t["status"] == "trained":
                d = t["train"]["end_states"]
            else:
                d = str(self.run.dir / "tasks" / t["id"] / "eval" / "end_states")
            if not d or not glob.glob(os.path.join(d, "*.npz")):
                self.log("error", task=t["id"], message="no end states were saved, so the task has no end node")
                continue
            self.run.keep(d)
            self.log("states_captured", level=level, task=t["id"], dir=d)

    # ---- the loop

    def go(self):
        if not self.run.events():
            self.log("run_started", **self.cfg)
        for level in range(1, self.cfg["max_depth"] + 1):
            tree = self.tree()
            if tree["levels"].get(str(level), {}).get("done"):
                continue
            frontier = [n for n in tree["nodes"].values() if n["level"] == level - 1]
            if level > 1:
                frontier = [n for n in frontier if n["incoming"]]
            if not frontier:
                break
            if str(level) not in tree["levels"]:
                self.log("level_started", level=level)
            proposed_from = {t["start_node"] for t in self.level_tasks(tree, level)}
            for node in frontier:
                if node["id"] not in proposed_from:
                    self.propose(level, node)
            self.screen(level)
            self.pick(level)
            for t in self.level_tasks(self.tree(), level):
                if t["status"] in ("picked", "training"):
                    self.train(level, t)
            self.evaluate(level)
            self.capture(level)
            self.log("level_done", level=level)
            if not any(t["end_node"] for t in self.level_tasks(self.tree(), level)):
                break


def main(argv):
    if len(argv) != 1:
        print(__doc__.strip().split("\n\n")[1])
        return 2
    cfg = load_config(argv[0])
    try:
        Orchestrator(cfg).go()
    except Stop as e:
        print(f"stopped: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

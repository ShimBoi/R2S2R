#!/usr/bin/env python3
"""Extracts a snapshot of the VLM-discovered subtask tree's progress into a single JSON blob,
for the live-progress dashboard artifact. Pure read-only data extraction -- never launches or
modifies anything. Safe to re-run at any time, as often as wanted, including while the driver is
mid-run.

Sources combined:
  - manifest.json           -- trained edges (ground truth for "finetuned" nodes)
  - driver_status.json      -- current in-flight edge/phase, if the driver is running
  - edge1_children.json /
    <node>_children.json    -- discovered-but-not-yet-trained candidate edges, if present
  - each trained edge's own training + collection logs -- before/after success-rate numbers

"Before finetuning" = the first rollout epoch's subtask_1_success_once in that edge's own
training log. "After finetuning" = the collection-eval pass's eval/subtask_1_success_once if
that log exists, else the last training rollout epoch's success_once as a fallback.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MANIFEST_DIR = Path("/scratch/cluster/jshim12/RLinf/logs/tree_manifest")
LOGS_ROOT = Path("/scratch/cluster/jshim12/RLinf/logs")
MANIFEST_PATH = MANIFEST_DIR / "manifest.json"
STATUS_PATH = MANIFEST_DIR / "driver_status.json"
OUT_PATH = MANIFEST_DIR / "dashboard" / "dashboard_data.json"

BOX_SUCCESS_RE = re.compile(r"subtask_1_success_once=([0-9.]+)")
WANDB_EVAL_SUCCESS_RE = re.compile(r"eval/subtask_1_success_once[':\s]+([0-9.]+)")

# Phase B (resolve_plan) composed-instruction eval results. These runs disable wandb, so
# metrics are printed as a raw Python dict repr via logging instead of wandb's console
# formatter, e.g.:
#   [INFO 02:41:03 RLinf] {'eval/subtask_2_success_once': array(0.4675, dtype=float32), ...,
#                          'eval/num_trajectories': 800}
# Values are either `array(<float>, dtype=float32)` or a bare int -- this regex handles both.
COMPOSED_EVAL_LINE_RE = re.compile(r"\[INFO [\d:]+ RLinf\]\s*\{.*'eval/success_once'.*\}")
COMPOSED_EVAL_KV_RE = re.compile(r"'eval/(\w+)':\s*(?:array\(([\-0-9.eE]+)|([0-9]+))")

# (slug, instruction) for the two Phase B composed language instructions evaluated.
COMPOSED_EVAL_TASKS = [
    (
        "mug_then_coke",
        "Place the ceramic mug on the cutting board, then place the coke can on the cutting board.",
    ),
    (
        "coke_then_mug",
        "Place the coke can on the cutting board, then place the ceramic mug on the cutting board.",
    ),
]


def parse_composed_eval_metrics(log_path: Path) -> dict | None:
    if not log_path.is_file():
        return None
    text = log_path.read_text(errors="replace")
    line_match = None
    for m in COMPOSED_EVAL_LINE_RE.finditer(text):
        line_match = m  # keep the last one, in case of a retry-within-log
    if line_match is None:
        return None
    line = line_match.group(0)
    metrics = {}
    for key, arr_val, int_val in COMPOSED_EVAL_KV_RE.findall(line):
        metrics[key] = float(arr_val) if arr_val else float(int_val)
    return metrics or None


def find_composed_eval_log(slug: str, *, baseline: bool) -> Path | None:
    suffix = "-baseline" if baseline else ""
    pattern = f"*-generic_eval_grpo_openpi_pi05-{slug}{suffix}"
    candidates = sorted(LOGS_ROOT.glob(pattern), key=lambda p: p.name)
    # glob("*-slug") also matches "*-slug-baseline" -- filter explicitly.
    if not baseline:
        candidates = [c for c in candidates if not c.name.endswith("-baseline")]
    for log_dir in reversed(candidates):  # most recent first
        log_path = log_dir / "eval_embodiment.log"
        metrics = parse_composed_eval_metrics(log_path)
        if metrics is not None:
            return log_dir
    return None


def build_composed_evals() -> list[dict]:
    manifest = load_json(MANIFEST_PATH, {"edges": []})
    edges_by_id = {e["id"]: e for e in manifest.get("edges", [])}
    plan_dir = MANIFEST_DIR / "eval_plans"

    results = []
    for slug, instruction in COMPOSED_EVAL_TASKS:
        plan_path = plan_dir / f"{slug}.json"
        resolved_sequence = None
        if plan_path.is_file():
            try:
                plan = json.loads(plan_path.read_text())
                # Match each plan step back to the manifest edge that produced it by walking the
                # tree (precondition == completed-so-far, exact match), not a flat content
                # search -- a parent/child pair can share an identical (predicate,
                # predicate_args) signature, differing only in precondition.
                resolved_sequence = []
                completed = frozenset()
                for step in plan:
                    match = next(
                        (
                            eid
                            for eid, e in edges_by_id.items()
                            if frozenset(e.get("precondition", [])) == completed
                            and e.get("predicate") == step.get("predicate")
                            and e.get("predicate_args") == step.get("predicate_args")
                        ),
                        None,
                    )
                    resolved_sequence.append(match or step.get("instruction"))
                    if match:
                        completed = completed | {match}
            except (json.JSONDecodeError, OSError):
                resolved_sequence = None

        after_dir = find_composed_eval_log(slug, baseline=False)
        before_dir = find_composed_eval_log(slug, baseline=True)
        after = parse_composed_eval_metrics(after_dir / "eval_embodiment.log") if after_dir else None
        before = parse_composed_eval_metrics(before_dir / "eval_embodiment.log") if before_dir else None

        results.append({
            "slug": slug,
            "instruction": instruction,
            "resolved_sequence": resolved_sequence,
            "before": before,
            "after": after,
            "before_log_dir": str(before_dir) if before_dir else None,
            "after_log_dir": str(after_dir) if after_dir else None,
        })
    return results


def load_json(path: Path, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default


def find_log_dir_for_checkpoint(checkpoint: str) -> Path | None:
    # checkpoint = <log_dir>/<config_name>/checkpoints/global_step_N/actor
    p = Path(checkpoint)
    # walk up: actor -> global_step_N -> checkpoints -> config_name -> log_dir
    try:
        return p.parents[3]
    except IndexError:
        return None


def before_after_from_training_log(log_dir: Path) -> tuple[float | None, float | None]:
    log_path = log_dir / "run_embodiment.log"
    if not log_path.is_file():
        return None, None
    text = log_path.read_text(errors="replace")
    matches = BOX_SUCCESS_RE.findall(text)
    if not matches:
        return None, None
    before = float(matches[0])
    after = float(matches[-1])
    return before, after


def after_from_collection_eval_log(edge_id: str) -> float | None:
    # Collection eval log dirs are named "<timestamp>-<config>-<edge_id>-collect"
    candidates = sorted(LOGS_ROOT.glob(f"*-{edge_id}-collect"), key=lambda p: p.name)
    for log_dir in reversed(candidates):  # most recent first
        log_path = log_dir / "eval_embodiment.log"
        if not log_path.is_file():
            continue
        text = log_path.read_text(errors="replace")
        m = WANDB_EVAL_SUCCESS_RE.findall(text)
        if m:
            return float(m[-1])
    return None


def main() -> None:
    manifest = load_json(MANIFEST_PATH, {"edges": [], "reset_paths_for_children": {}, "terminal_nodes": []})
    status = load_json(STATUS_PATH, None)

    trained_edges = {e["id"]: e for e in manifest.get("edges", [])}

    def edge_signature(e: dict) -> str:
        return json.dumps(
            [sorted(e.get("precondition", [])), e.get("predicate"), e.get("predicate_args", {})],
            sort_keys=True,
        )

    # Content-based signature (precondition + predicate + predicate_args), not id: a level-queue
    # file can contain a renamed duplicate of an already-trained edge (re-discovery + _dedupe_id
    # renaming a raw id collision) -- same edge, different id, must not show twice.
    trained_signatures = {edge_signature(e) for e in trained_edges.values()}

    nodes = []
    for edge_id, edge in trained_edges.items():
        log_dir = find_log_dir_for_checkpoint(edge["checkpoint"]) if edge.get("checkpoint") else None
        before, after_train = (None, None)
        if log_dir is not None:
            before, after_train = before_after_from_training_log(log_dir)
        after_eval = after_from_collection_eval_log(edge_id)
        after = after_eval if after_eval is not None else after_train

        nodes.append({
            "id": edge_id,
            "instruction": edge.get("instruction"),
            "predicate": edge.get("predicate"),
            "predicate_args": edge.get("predicate_args"),
            "precondition": edge.get("precondition", []),
            "produces_node": edge.get("produces_node", []),
            "reset_states_path": edge.get("reset_states_path"),
            "checkpoint": edge.get("checkpoint"),
            "status": "finetuned",
            "success_before": before,
            "success_after_training": after_train,
            "success_after_eval": after_eval,
            "success_after": after,
        })

    # Discovered-but-not-yet-trained candidates. Priority order, highest first:
    #   1. level_<depth>_queue.json -- authoritative once it exists: written after a whole
    #      level's edges are discovered and deduped, before any start training. Ids here are
    #      already final.
    #   2. root_candidates.json / *_children.json -- older preview files, fallback only for
    #      whatever a level-queue file doesn't cover yet. Can carry stale, non-deduped ids.
    pending_ids_seen = set()
    candidate_files = sorted(MANIFEST_DIR.glob("level_*_queue.json"))
    root_candidates_path = MANIFEST_DIR / "root_candidates.json"
    if root_candidates_path.is_file():
        candidate_files.append(root_candidates_path)
    candidate_files += sorted(MANIFEST_DIR.glob("*_children.json"))
    for children_path in candidate_files:
        children = load_json(children_path, [])
        for cand in children:
            if cand["id"] in trained_edges or cand["id"] in pending_ids_seen:
                continue
            is_current = bool(status and status.get("current_edge") == cand["id"])
            if not is_current and edge_signature(cand) in trained_signatures:
                # Same real edge as one already trained, a renamed duplicate from re-discovery
                # -- skip silently. `is_current` guard: the active edge must still render even
                # if its signature happened to coincide with an unrelated trained edge.
                pending_ids_seen.add(cand["id"])
                continue
            pending_ids_seen.add(cand["id"])
            if is_current and status.get("phase") == "FAILED":
                cand_status = "failed"
            elif is_current:
                cand_status = "training_in_progress"
            else:
                cand_status = "queued"
            # Surface a checkpoint that training produced before a later stage (e.g. end-state
            # collection) failed, so a failed node doesn't look like nothing happened.
            checkpoint = status.get("checkpoint") if (is_current and status.get("phase") == "FAILED") else None
            nodes.append({
                "id": cand["id"],
                "instruction": cand.get("instruction"),
                "predicate": cand.get("predicate"),
                "predicate_args": cand.get("predicate_args"),
                "precondition": cand.get("precondition", []),
                "produces_node": cand.get("produces_node", []),
                "reset_states_path": cand.get("reset_states_path"),
                "checkpoint": checkpoint,
                "status": cand_status,
                "error": status.get("error") if cand_status == "failed" else None,
                "success_before": None,
                "success_after_training": None,
                "success_after_eval": None,
                "success_after": None,
            })

    # Fallback for the currently-active edge when no preview file ever mentioned it. Its own
    # edge spec under edge_specs/<id>.json (written before every real launch) is always
    # authoritative once a stage has started.
    current_edge_id = status.get("current_edge") if status else None
    if current_edge_id and current_edge_id not in trained_edges and current_edge_id not in pending_ids_seen:
        spec_path = MANIFEST_DIR / "edge_specs" / f"{current_edge_id}.json"
        spec = load_json(spec_path, None)
        if spec is not None:
            cand_status = "failed" if status.get("phase") == "FAILED" else "training_in_progress"
            checkpoint = status.get("checkpoint") if cand_status == "failed" else None
            nodes.append({
                "id": current_edge_id,
                "instruction": spec.get("instruction"),
                "predicate": spec.get("predicate"),
                "predicate_args": spec.get("predicate_args"),
                "precondition": spec.get("precondition", []),
                "produces_node": spec.get("produces_node", []),
                "reset_states_path": spec.get("reset_states_path"),
                "checkpoint": checkpoint,
                "status": cand_status,
                "error": status.get("error") if cand_status == "failed" else None,
                "success_before": None,
                "success_after_training": None,
                "success_after_eval": None,
                "success_after": None,
            })

    data = {
        "generated_at_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "driver_status": status,
        "terminal_nodes": manifest.get("terminal_nodes", []),
        "nodes": nodes,
        "composed_evals": build_composed_evals(),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(data, indent=2))
    print(f"wrote {OUT_PATH} ({len(nodes)} nodes)")


if __name__ == "__main__":
    sys.exit(main())

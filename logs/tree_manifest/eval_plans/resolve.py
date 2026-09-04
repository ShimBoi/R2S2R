#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One-off script: verify Phase B's resolve_plan() end-to-end (real VLM calls, no GPU spend)
against the real, final manifest.json for the mug/coke/cutting-board tree, before any GPU eval
is launched.

Makes real network calls to the VLM (OPENAI_API_KEY must already be exported in the shell env
-- `set -a; source /scratch/cluster/jshim12/.env; set +a` -- before running this).

Does NOT write plan files for a task whose resolved sequence doesn't exactly match the expected
one, and does not run the bonus phrasing-robustness check unless both main tasks passed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RLINF_ROOT = Path("/scratch/cluster/jshim12/RLinf")
sys.path.insert(0, str(RLINF_ROOT))

from rlinf.envs.isaaclab.tasks.discovery.manifest import Manifest  # noqa: E402
from rlinf.envs.isaaclab.tasks.discovery.phase_b import (  # noqa: E402
    call_vlm_phase_b,
    resolve_plan,
)
from rlinf.envs.isaaclab.tasks.discovery.vlm_client import VLMCallError  # noqa: E402

MANIFEST_PATH = RLINF_ROOT / "logs" / "tree_manifest" / "manifest.json"
OUT_DIR = RLINF_ROOT / "logs" / "tree_manifest" / "eval_plans"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_A = (
    "Place the ceramic mug on the cutting board, then place the coke can on the cutting board."
)
TASK_B = (
    "Place the coke can on the cutting board, then place the ceramic mug on the cutting board."
)

# One differently-phrased variant per task, for the optional bonus robustness check (Step 1.5).
VARIANT_A = "First put the mug on the cutting board, then the coke can."
VARIANT_B = "I want both items on the cutting board, coke first."

EXPECTED = {
    "A": [
        "place_mug_on_cutting_board",
        "place_coke_on_cutting_board__given_place_mug_on_cutting_board",
    ],
    "B": [
        "place_coke_on_cutting_board",
        "place_ceramic_mug_on_cutting_board",
    ],
}

OUT_FILES = {
    "A": OUT_DIR / "mug_then_coke.json",
    "B": OUT_DIR / "coke_then_mug.json",
}


def make_logging_caller(call_log: list) -> callable:
    """Wrap call_vlm_phase_b so we can report the VLM's reasoning back, without changing
    resolve_plan()'s own logic (it still only calls this at genuine branch points).
    """

    def caller(overarching_task, completed_so_far, available_edges, state_description="(not provided)"):
        response = call_vlm_phase_b(
            overarching_task,
            completed_so_far=completed_so_far,
            available_edges=available_edges,
            state_description=state_description,
        )
        call_log.append(
            {
                "overarching_task": overarching_task,
                "completed_so_far": list(completed_so_far),
                "available_edge_ids": [e["id"] for e in available_edges],
                "response": response,
            }
        )
        return response

    return caller


def run_task(label: str, overarching_task: str, manifest: Manifest):
    print(f"\n=== Task {label}: {overarching_task!r} ===")
    call_log: list = []
    try:
        plan = resolve_plan(
            overarching_task, manifest, phase_b_caller=make_logging_caller(call_log)
        )
    except VLMCallError as exc:
        print(f"Task {label}: resolve_plan() RAISED VLMCallError: {exc}")
        return None, [], call_log
    ids = [e["id"] for e in plan]
    print(f"resolved edge-id sequence: {ids}")
    for e in plan:
        print(f"  - id={e['id']!r}")
        print(f"    instruction={e['instruction']!r}")
        print(f"    predicate={e['predicate']!r} predicate_args={e['predicate_args']!r}")
    print(f"VLM calls made for this task: {len(call_log)}")
    for i, call in enumerate(call_log):
        resp = call["response"]
        print(
            f"  call[{i}] at node={call['completed_so_far']!r} "
            f"available={call['available_edge_ids']!r}"
        )
        print(f"    -> status={resp.get('status')!r} subtask_id={resp.get('subtask_id')!r}")
        print(f"    -> reasoning: {resp.get('reasoning')!r}")
    return plan, ids, call_log


def main() -> int:
    manifest = Manifest(MANIFEST_PATH)
    print(f"Loaded manifest: {len(manifest)} edge(s) from {MANIFEST_PATH}")
    for e in manifest.all_edges():
        print(f"  edge id={e['id']!r} precondition={e['precondition']!r} produces={e['produces_node']!r}")

    results = {}
    for label, task in (("A", TASK_A), ("B", TASK_B)):
        plan, ids, call_log = run_task(label, task, manifest)
        expected = EXPECTED[label]
        ok = plan is not None and ids == expected
        results[label] = {"plan": plan, "ids": ids, "ok": ok, "call_log": call_log}
        print(f"Task {label}: expected={expected!r}")
        print(f"Task {label}: MATCH={ok}")
        if not ok:
            print(
                f"STOP for Task {label}: resolved sequence does NOT match the expected one "
                f"(or resolve_plan() raised) -- NOT writing a plan file, NOT proceeding to any "
                f"GPU eval for this task."
            )
            continue
        filtered = [
            {
                "predicate": e["predicate"],
                "predicate_args": e["predicate_args"],
                "instruction": e["instruction"],
            }
            for e in plan
        ]
        OUT_FILES[label].write_text(json.dumps(filtered, indent=2))
        print(f"Wrote verified plan for Task {label} -> {OUT_FILES[label]}")

    print("\n=== STEP 1 SUMMARY ===")
    for label in ("A", "B"):
        r = results[label]
        print(f"Task {label}: ids={r['ids']!r} match_expected={r['ok']}")
    all_ok = all(results[l]["ok"] for l in ("A", "B"))
    print(f"\nBoth tasks verified correct: {all_ok}")

    if all_ok:
        print("\n=== OPTIONAL BONUS: phrasing-robustness spot-check (no GPU, informational only) ===")
        for label, variant in (("A", VARIANT_A), ("B", VARIANT_B)):
            _, ids, _ = run_task(f"{label}-variant", variant, manifest)
            match = ids == EXPECTED[label]
            print(f"Variant for Task {label} ({variant!r}): ids={ids!r} matches_original_plan={match}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

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

"""Unit tests for rlinf.envs.isaaclab.tasks.discovery.phase_b.

Centerpiece: `test_resolve_plan_worked_example_*` reproduces PLAN.md section 8's exact worked
example (the 4-edge mug/coke tree) with a mocked VLM, and confirms `resolve_plan` makes exactly
one VLM call (at the root branch point) and returns the correct 2-step plan, for both the
"cup first" and "can first" VLM responses. Also covers the "unreachable"/"done" early-exit
paths. No real network calls anywhere in this file.
"""

import json

import pytest

from rlinf.envs.isaaclab.tasks.discovery.manifest import Manifest
from rlinf.envs.isaaclab.tasks.discovery.phase_b import (
    build_phase_b_prompt,
    call_vlm_phase_b,
    resolve_plan,
)
from rlinf.envs.isaaclab.tasks.discovery.vlm_client import VLMCallError

# The exact 4-edge tree from PLAN.md section 8 / section 8.1:
#
#                     {}
#                  /      \
#    {cup}                    {can}
#      |                        |
#    {cup, can-given-cup}    {can, cup-given-can}

CUP_EDGE = {
    "id": "latteartcup_on_cuttingboard",
    "precondition": [],
    "predicate": "object_on_top",
    "predicate_args": {"object": "latteartcup", "reference_object": "cuttingboard"},
    "instruction": "Pick up the latte art cup and place it on the cutting board",
    "objects_involved": ["latteartcup", "cuttingboard"],
    "produces_node": ["latteartcup_on_cuttingboard"],
}
CAN_EDGE = {
    "id": "coke_on_cuttingboard",
    "precondition": [],
    "predicate": "object_on_top",
    "predicate_args": {"object": "coke", "reference_object": "cuttingboard"},
    "instruction": "Pick up the coke can and place it on the cutting board",
    "objects_involved": ["coke", "cuttingboard"],
    "produces_node": ["coke_on_cuttingboard"],
}
CAN_GIVEN_CUP_EDGE = {
    "id": "coke_on_cuttingboard__given_cup",
    "precondition": ["latteartcup_on_cuttingboard"],
    "predicate": "object_on_top",
    "predicate_args": {"object": "coke", "reference_object": "cuttingboard"},
    "instruction": "Pick up the coke can and place it on the cutting board",
    "objects_involved": ["coke", "cuttingboard"],
    "produces_node": ["coke_on_cuttingboard__given_cup", "latteartcup_on_cuttingboard"],
}
CUP_GIVEN_CAN_EDGE = {
    "id": "latteartcup_on_cuttingboard__given_can",
    "precondition": ["coke_on_cuttingboard"],
    "predicate": "object_on_top",
    "predicate_args": {"object": "latteartcup", "reference_object": "cuttingboard"},
    "instruction": "Pick up the latte art cup and place it on the cutting board",
    "objects_involved": ["latteartcup", "cuttingboard"],
    "produces_node": ["coke_on_cuttingboard", "latteartcup_on_cuttingboard__given_can"],
}


@pytest.fixture
def worked_example_manifest(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    m.add_edge(CUP_EDGE)
    m.add_edge(CAN_EDGE)
    m.add_edge(CAN_GIVEN_CUP_EDGE)
    m.add_edge(CUP_GIVEN_CAN_EDGE)
    return m


def _counting_phase_b_caller(response):
    calls = []

    def caller(overarching_task, completed_so_far, available_edges, state_description="(not provided)"):
        calls.append(
            {
                "overarching_task": overarching_task,
                "completed_so_far": completed_so_far,
                "available_edges": available_edges,
            }
        )
        return response

    return caller, calls


def test_resolve_plan_worked_example_cup_first(worked_example_manifest):
    caller, calls = _counting_phase_b_caller(
        {"status": "next", "subtask_id": "latteartcup_on_cuttingboard", "reasoning": "cup first"}
    )

    plan = resolve_plan(
        "put the latte cup and the coke can on the cutting board",
        worked_example_manifest,
        phase_b_caller=caller,
    )

    # Exactly one VLM call (the root branch point) resolves the entire plan.
    assert len(calls) == 1
    assert calls[0]["completed_so_far"] == []
    assert {e["id"] for e in calls[0]["available_edges"]} == {
        "latteartcup_on_cuttingboard",
        "coke_on_cuttingboard",
    }

    # Correct 2-step plan: cup, then (mechanically, no further VLM call) coke-given-cup.
    assert [e["id"] for e in plan] == [
        "latteartcup_on_cuttingboard",
        "coke_on_cuttingboard__given_cup",
    ]


def test_resolve_plan_worked_example_can_first(worked_example_manifest):
    caller, calls = _counting_phase_b_caller(
        {"status": "next", "subtask_id": "coke_on_cuttingboard", "reasoning": "can first"}
    )

    plan = resolve_plan(
        "put the coke can and the latte cup on the cutting board",
        worked_example_manifest,
        phase_b_caller=caller,
    )

    assert len(calls) == 1
    assert [e["id"] for e in plan] == [
        "coke_on_cuttingboard",
        "latteartcup_on_cuttingboard__given_can",
    ]


def test_resolve_plan_unreachable_status_returns_empty_plan_no_further_calls(
    worked_example_manifest,
):
    caller, calls = _counting_phase_b_caller(
        {"status": "unreachable", "reasoning": "needs a 4th object not in this scene"}
    )
    plan = resolve_plan("do something impossible", worked_example_manifest, phase_b_caller=caller)
    assert plan == []
    assert len(calls) == 1


def test_resolve_plan_done_status_at_root_returns_empty_plan(worked_example_manifest):
    caller, calls = _counting_phase_b_caller(
        {"status": "done", "reasoning": "nothing needs doing"}
    )
    plan = resolve_plan("do nothing", worked_example_manifest, phase_b_caller=caller)
    assert plan == []
    assert len(calls) == 1


def test_resolve_plan_done_status_partway_through_stops_early(tmp_path):
    """`done` returned partway through a walk (not just at the root) should stop the walk right
    there, without raising, and without ever reaching the second branch point. Needs a tree with
    a branch point below the root (the 4-edge worked-example tree only branches at the root),
    so this uses a small ad hoc manifest instead.
    """
    m = Manifest(tmp_path / "manifest.json")
    m.add_edge(CUP_EDGE)  # {} -> {cup}, the only edge at root: mechanical, no VLM call
    # A second branch point at {cup}: two options.
    branch_a = dict(CAN_GIVEN_CUP_EDGE)
    branch_b = {
        "id": "some_other_thing_given_cup",
        "precondition": ["latteartcup_on_cuttingboard"],
        "predicate": "object_on_top",
        "predicate_args": {"object": "coke", "reference_object": "cuttingboard"},
        "instruction": "irrelevant",
        "objects_involved": ["coke", "cuttingboard"],
        "produces_node": ["latteartcup_on_cuttingboard", "some_other_thing_given_cup"],
    }
    m.add_edge(branch_a)
    m.add_edge(branch_b)

    calls = []

    def caller(overarching_task, completed_so_far, available_edges, state_description="(not provided)"):
        calls.append(completed_so_far)
        return {"status": "done", "reasoning": "stop here"}

    plan = resolve_plan("only place the cup", m, phase_b_caller=caller)

    # Mechanical step to {cup} (no VLM call), then a genuine branch point at {cup} where the
    # VLM says "done" -- the walk must stop there, not follow either branch.
    assert [e["id"] for e in plan] == ["latteartcup_on_cuttingboard"]
    assert len(calls) == 1
    assert calls[0] == ["latteartcup_on_cuttingboard"]


def test_resolve_plan_leaf_node_no_edges_returns_empty_plan_no_vlm_call(
    worked_example_manifest,
):
    caller, calls = _counting_phase_b_caller({"status": "done", "reasoning": "n/a"})
    leaf = frozenset({"coke_on_cuttingboard__given_cup", "latteartcup_on_cuttingboard"})
    plan = resolve_plan("anything", worked_example_manifest, node=leaf, phase_b_caller=caller)
    assert plan == []
    assert len(calls) == 0  # no branch point reached -- leaf has zero valid edges


def test_resolve_plan_raises_if_vlm_picks_an_id_not_offered(worked_example_manifest):
    caller, _ = _counting_phase_b_caller(
        {"status": "next", "subtask_id": "not_actually_offered", "reasoning": "oops"}
    )
    with pytest.raises(VLMCallError):
        resolve_plan("anything", worked_example_manifest, phase_b_caller=caller)


# ---------------------------------------------------------------------------
# call_vlm_phase_b -- structural validation, no network
# ---------------------------------------------------------------------------


def test_call_vlm_phase_b_parses_next_response():
    response = call_vlm_phase_b(
        "task", [], [CUP_EDGE],
        client_fn=lambda prompt, image_b64=None, model=None: json.dumps(
            {"status": "next", "subtask_id": "latteartcup_on_cuttingboard", "reasoning": "r"}
        ),
    )
    assert response["status"] == "next"
    assert response["subtask_id"] == "latteartcup_on_cuttingboard"


def test_call_vlm_phase_b_rejects_next_without_subtask_id():
    with pytest.raises(VLMCallError):
        call_vlm_phase_b(
            "task", [], [CUP_EDGE],
            client_fn=lambda prompt, image_b64=None, model=None: json.dumps(
                {"status": "next", "reasoning": "missing subtask_id"}
            ),
        )


def test_call_vlm_phase_b_rejects_unrecognized_status():
    with pytest.raises(VLMCallError):
        call_vlm_phase_b(
            "task", [], [CUP_EDGE],
            client_fn=lambda prompt, image_b64=None, model=None: json.dumps(
                {"status": "maybe", "reasoning": "?"}
            ),
        )


def test_build_phase_b_prompt_lists_available_edges():
    prompt = build_phase_b_prompt("overarching task text", [], [CUP_EDGE, CAN_EDGE])
    assert "overarching task text" in prompt
    assert "latteartcup_on_cuttingboard" in prompt
    assert "coke_on_cuttingboard" in prompt

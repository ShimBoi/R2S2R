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

"""Unit tests for rlinf.envs.isaaclab.tasks.discovery.orchestrator.

`discover_tree` is exercised against a mocked `call_vlm_phase_a` reproducing PLAN.md section
8.1's exact worked example. `train_sequence` is exercised with faked `run_training`/
`collect_end_states` (no real subprocess/GPU job ever launches). `launch_and_wait` -- the one
piece of *real*, not-invoked-for-training, subprocess-launching code -- gets its own test
against a trivial, instantaneous shell command, to confirm the session-detachment/polling/
exit-code plumbing itself actually works, without touching training.
"""

import json
import time

import pytest

import rlinf.envs.isaaclab.tasks.discovery.orchestrator as orchestrator
from rlinf.envs.isaaclab.tasks.discovery.manifest import Manifest
from rlinf.envs.isaaclab.tasks.discovery.orchestrator import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_TOTAL_EDGES,
    LaunchResult,
    TrainingConfigSpec,
    _dedupe_id,
    _make_default_collect_end_states,
    _precondition_suffix,
    discover_and_train,
    discover_and_train_by_level,
    discover_tree,
    generate_training_config,
    launch_and_wait,
    render_current_node,
    strip_actor_lora_path_override,
    train_sequence,
)

SCENE_OBJECTS = ["latteartcup", "coke", "cuttingboard"]

# Every test below that isn't specifically about grounding passes this no-op state source, to
# keep those tests hermetic/focused (real-grounding threading has its own dedicated tests
# further down, plus test_discovery_starting_states.py covers the summarizers themselves).
_NO_GROUNDING = lambda node: None  # noqa: E731


# ---------------------------------------------------------------------------
# discover_tree -- PLAN.md section 8.1's worked example
# ---------------------------------------------------------------------------


def _worked_example_phase_a_caller():
    """Returns (caller, calls) reproducing PLAN.md section 8.1 exactly:

    - node {} (root): two candidates (cup, can) -- a genuine branch.
    - node {cup}: exactly one candidate (coke given cup).
    - node {can}: exactly one candidate (cup given can) -- "mirror of call #2".
    - both {both, ...} leaves: [] -- nothing left, tree stops growing.
    """
    calls = []

    def caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        calls.append(sorted(satisfied_so_far))
        node = frozenset(satisfied_so_far)
        if node == frozenset():
            return [
                {
                    "id": "latteartcup_on_cuttingboard",
                    "predicate": "object_on_top",
                    "predicate_args": {
                        "object": "latteartcup",
                        "reference_object": "cuttingboard",
                        "require_gripper_detached": True,
                    },
                    "instruction": "Pick up the latte art cup and place it on the cutting board",
                    "objects_involved": ["latteartcup", "cuttingboard"],
                },
                {
                    "id": "coke_on_cuttingboard",
                    "predicate": "object_on_top",
                    "predicate_args": {
                        "object": "coke",
                        "reference_object": "cuttingboard",
                        "require_gripper_detached": True,
                    },
                    "instruction": "Pick up the coke can and place it on the cutting board",
                    "objects_involved": ["coke", "cuttingboard"],
                },
            ]
        if node == frozenset({"latteartcup_on_cuttingboard"}):
            return [
                {
                    "id": "coke_on_cuttingboard__given_cup",
                    "predicate": "object_on_top",
                    "predicate_args": {
                        "object": "coke",
                        "reference_object": "cuttingboard",
                        "require_gripper_detached": True,
                    },
                    "instruction": "Pick up the coke can and place it on the cutting board",
                    "objects_involved": ["coke", "cuttingboard"],
                }
            ]
        if node == frozenset({"coke_on_cuttingboard"}):
            return [
                {
                    "id": "latteartcup_on_cuttingboard__given_can",
                    "predicate": "object_on_top",
                    "predicate_args": {
                        "object": "latteartcup",
                        "reference_object": "cuttingboard",
                        "require_gripper_detached": True,
                    },
                    "instruction": "Pick up the latte art cup and place it on the cutting board",
                    "objects_involved": ["latteartcup", "cuttingboard"],
                }
            ]
        # Both {both, ...} leaves: nothing further.
        return []

    return caller, calls


def test_discover_tree_worked_example_topology(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    caller, calls = _worked_example_phase_a_caller()

    ordered_edges = discover_tree(
        manifest,
        scene_objects=SCENE_OBJECTS,
        max_depth=4,
        phase_a_caller=caller,
        predicate_module={"object_on_top"},
        get_starting_state_summary=_NO_GROUNDING,
    )

    # 4 edges total, topologically ordered: both root candidates appear before their (only)
    # children -- DFS order out of the sketch: [cup, coke_given_cup, coke, cup_given_can].
    assert [e["id"] for e in ordered_edges] == [
        "latteartcup_on_cuttingboard",
        "coke_on_cuttingboard__given_cup",
        "coke_on_cuttingboard",
        "latteartcup_on_cuttingboard__given_can",
    ]

    # Parents-before-children (topological order), the one hard requirement PLAN.md places on
    # ordering regardless of DFS-vs-BFS specifics.
    ids_in_order = [e["id"] for e in ordered_edges]
    for edge in ordered_edges:
        for precondition_id in edge["precondition"]:
            assert ids_in_order.index(precondition_id) < ids_in_order.index(edge["id"])

    # precondition / produces_node were filled in by discover_tree from the walk, not by the
    # (mocked) VLM.
    cup_edge = next(e for e in ordered_edges if e["id"] == "latteartcup_on_cuttingboard")
    assert cup_edge["precondition"] == []
    assert cup_edge["produces_node"] == ["latteartcup_on_cuttingboard"]

    coke_given_cup = next(
        e for e in ordered_edges if e["id"] == "coke_on_cuttingboard__given_cup"
    )
    assert coke_given_cup["precondition"] == ["latteartcup_on_cuttingboard"]
    assert sorted(coke_given_cup["produces_node"]) == sorted(
        ["latteartcup_on_cuttingboard", "coke_on_cuttingboard__given_cup"]
    )

    # Phase A called once per node visited: root, {cup}, {cup,coke_given_cup} (terminal),
    # {can}, {can,cup_given_can} (terminal) = 5 nodes total -- 6 in the earlier draft's count
    # was wrong; DFS visits each node exactly once regardless of branch, so 5.
    assert len(calls) == 5

    # Both true leaves were marked terminal.
    assert manifest.is_terminal(
        frozenset({"latteartcup_on_cuttingboard", "coke_on_cuttingboard__given_cup"})
    )
    assert manifest.is_terminal(
        frozenset({"coke_on_cuttingboard", "latteartcup_on_cuttingboard__given_can"})
    )
    # Root and the two depth-1 nodes were NOT marked terminal (they had candidates).
    assert not manifest.is_terminal(frozenset())
    assert not manifest.is_terminal(frozenset({"latteartcup_on_cuttingboard"}))


def test_discover_tree_filters_invalid_candidates_before_recursing(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")

    def caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        if satisfied_so_far:
            return []
        return [
            {
                "id": "valid_edge",
                "predicate": "object_on_top",
                "predicate_args": {"object": "coke", "reference_object": "cuttingboard"},
                "instruction": "ok",
                "objects_involved": ["coke", "cuttingboard"],
            },
            {
                "id": "invalid_edge_bad_object",
                "predicate": "object_on_top",
                "predicate_args": {"object": "banana"},
                "instruction": "not ok",
                "objects_involved": ["banana"],
            },
        ]

    ordered_edges = discover_tree(
        manifest,
        scene_objects=SCENE_OBJECTS,
        phase_a_caller=caller,
        predicate_module={"object_on_top"},
        get_starting_state_summary=_NO_GROUNDING,
    )
    assert [e["id"] for e in ordered_edges] == ["valid_edge"]


def test_discover_tree_respects_max_depth(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    call_count = {"n": 0}

    def caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        call_count["n"] += 1
        node = frozenset(satisfied_so_far)
        new_id = f"step_{len(node)}"
        return [
            {
                "id": new_id,
                "predicate": "object_on_top",
                "predicate_args": {},
                "instruction": "keep going",
                "objects_involved": [],
            }
        ]

    ordered_edges = discover_tree(
        manifest,
        scene_objects=SCENE_OBJECTS,
        max_depth=2,
        phase_a_caller=caller,
        predicate_module=None,
        get_starting_state_summary=_NO_GROUNDING,
    )
    # max_depth=2 means nodes of size >= 2 are not expanded -- exactly 2 edges get discovered
    # (root -> {step_0}, {step_0} -> {step_0, step_1}); the walk stops before ever calling
    # Phase A at a size-2 node.
    assert len(ordered_edges) == 2
    assert call_count["n"] == 2


def test_discover_tree_default_max_depth_and_max_total_edges_are_reasonable():
    # Regression guard for the coordinator-requested defaults: lower than the old max_depth=4
    # that let the real run explode to 52/23, and a real, finite max_total_edges safety valve
    # exists by default (was None/unbounded before).
    assert DEFAULT_MAX_DEPTH < 4
    assert DEFAULT_MAX_TOTAL_EDGES is not None
    assert 0 < DEFAULT_MAX_TOTAL_EDGES < 52


def test_discover_tree_max_total_edges_hard_cap(tmp_path):
    """A verbose model proposing many candidates per node must still be bounded by
    max_total_edges, independent of max_depth -- the structural safety-valve the coordinator
    asked for alongside the improved prompt.
    """
    manifest = Manifest(tmp_path / "manifest.json")

    def verbose_caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        # Always proposes 3 fresh candidates, however deep -- would run away indefinitely
        # (bounded only by max_depth) without the max_total_edges cap.
        node = frozenset(satisfied_so_far)
        prefix = "_".join(sorted(node)) or "root"
        return [
            {
                "id": f"{prefix}__opt{i}",
                "predicate": "object_on_top",
                "predicate_args": {},
                "instruction": "keep going",
                "objects_involved": [],
            }
            for i in range(3)
        ]

    ordered_edges = discover_tree(
        manifest,
        scene_objects=SCENE_OBJECTS,
        max_depth=10,  # deep enough that the depth cap would NOT be what stops this
        max_total_edges=5,
        phase_a_caller=verbose_caller,
        predicate_module=None,
        get_starting_state_summary=_NO_GROUNDING,
    )
    assert len(ordered_edges) == 5


def test_discover_tree_max_total_edges_none_disables_the_cap(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    caller, _ = _worked_example_phase_a_caller()
    ordered_edges = discover_tree(
        manifest,
        scene_objects=SCENE_OBJECTS,
        max_depth=4,
        max_total_edges=None,
        phase_a_caller=caller,
        predicate_module={"object_on_top"},
        get_starting_state_summary=_NO_GROUNDING,
    )
    assert len(ordered_edges) == 4  # unbounded cap, the real tree just naturally terminates


# ---------------------------------------------------------------------------
# discover_tree -- real starting-state grounding threading
# ---------------------------------------------------------------------------


def test_discover_tree_grounds_root_in_real_data_by_default(tmp_path):
    """With no injected state source, discover_tree must fall back to the REAL
    starting_states.summarize_starting_state (root -> real initial_conditions.json), not `None`
    -- this is the primary fix the coordinator asked for (real data replacing the screenshot).
    """
    manifest = Manifest(tmp_path / "manifest.json")
    seen_summaries = []

    def caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        seen_summaries.append(starting_state_summary)
        return []  # terminal immediately -- only care about what the root call received

    discover_tree(
        manifest,
        scene_objects=["ceramic_mug", "coke", "cutting_board_a"],
        phase_a_caller=caller,
        predicate_module=None,
    )
    assert len(seen_summaries) == 1
    assert seen_summaries[0] is not None
    assert "ceramic_mug: pos=(" in seen_summaries[0]


def test_discover_tree_injected_state_source_overrides_default(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    seen_summaries = []

    def caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        seen_summaries.append(starting_state_summary)
        return []

    discover_tree(
        manifest,
        scene_objects=SCENE_OBJECTS,
        phase_a_caller=caller,
        predicate_module=None,
        get_starting_state_summary=lambda node: "INJECTED_MARKER",
    )
    assert seen_summaries == ["INJECTED_MARKER"]


# ---------------------------------------------------------------------------
# discover_tree -- stable_base_objects threading (round 2: physical plausibility)
# ---------------------------------------------------------------------------


def test_discover_tree_stable_base_objects_disabled_by_default_end_to_end(tmp_path):
    """Same physically-dubious candidate as the enabled test below, but with no
    stable_base_objects passed -- must NOT be filtered (opt-in, not a silent default)."""
    manifest = Manifest(tmp_path / "manifest.json")

    def caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        if satisfied_so_far:
            return []
        return [
            {
                "id": "place_cutting_board_on_ceramic_mug",
                "predicate": "object_on_top",
                "predicate_args": {"object": "cutting_board_a", "reference_object": "ceramic_mug"},
                "instruction": "place the board on the mug",
                "objects_involved": ["cutting_board_a", "ceramic_mug"],
            }
        ]

    ordered_edges = discover_tree(
        manifest,
        scene_objects=["ceramic_mug", "cutting_board_a"],
        phase_a_caller=caller,
        predicate_module={"object_on_top"},
        get_starting_state_summary=lambda node: None,
    )
    assert [e["id"] for e in ordered_edges] == ["place_cutting_board_on_ceramic_mug"]


def test_discover_tree_stable_base_objects_filters_physically_implausible_bases_end_to_end(
    tmp_path,
):
    """End-to-end version of the coordinator's two round-2 bad examples: with
    stable_base_objects={"cutting_board_a"} opted in, discover_tree's real edge list must not
    contain either physically-implausible-base proposal, while a legitimate board-based one
    still comes through.
    """
    manifest = Manifest(tmp_path / "manifest.json")

    def caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        if satisfied_so_far:
            return []
        return [
            {
                "id": "coke_on_cutting_board",
                "predicate": "object_on_top",
                "predicate_args": {"object": "coke", "reference_object": "cutting_board_a"},
                "instruction": "place the coke on the board",
                "objects_involved": ["coke", "cutting_board_a"],
            },
            {
                "id": "place_cutting_board_on_ceramic_mug",
                "predicate": "object_on_top",
                "predicate_args": {"object": "cutting_board_a", "reference_object": "ceramic_mug"},
                "instruction": "place the board on the mug",
                "objects_involved": ["cutting_board_a", "ceramic_mug"],
            },
            {
                "id": "stack_coke_on_ceramic_mug",
                "predicate": "stacked",
                "predicate_args": {"objects": ["ceramic_mug", "coke"], "order": "bottom_to_top"},
                "instruction": "stack the coke on the mug",
                "objects_involved": ["ceramic_mug", "coke"],
            },
        ]

    ordered_edges = discover_tree(
        manifest,
        scene_objects=["ceramic_mug", "coke", "cutting_board_a"],
        phase_a_caller=caller,
        predicate_module={"object_on_top", "stacked"},
        get_starting_state_summary=lambda node: None,
        stable_base_objects={"cutting_board_a"},
    )
    assert [e["id"] for e in ordered_edges] == ["coke_on_cutting_board"]


# ---------------------------------------------------------------------------
# deterministic id deduplication (across the whole run, not per-node)
# ---------------------------------------------------------------------------


def test_precondition_suffix_root_and_short_precondition():
    assert _precondition_suffix([]) == "given_root"
    assert _precondition_suffix(["b", "a"]) == "given_a_b"  # sorted, joined


def test_precondition_suffix_falls_back_to_hash_when_unwieldy():
    long_precondition = [f"some_fairly_long_subtask_id_number_{i}" for i in range(5)]
    suffix = _precondition_suffix(long_precondition)
    assert suffix.startswith("given_")
    assert len(suffix) < len("given_" + "_".join(sorted(long_precondition)))
    # deterministic -- same input, same output, every time (not e.g. seeded from time/random).
    assert suffix == _precondition_suffix(long_precondition)


def test_dedupe_id_passthrough_when_no_collision():
    assert _dedupe_id("fresh_id", seen_ids=set(), precondition=[]) == "fresh_id"


def test_dedupe_id_qualifies_with_precondition_on_collision():
    seen = {"place_coke_on_cutting_board"}
    result = _dedupe_id(
        "place_coke_on_cutting_board", seen, precondition=["place_ceramic_mug_on_cutting_board"]
    )
    assert result == "place_coke_on_cutting_board__given_place_ceramic_mug_on_cutting_board"


def test_dedupe_id_numeric_fallback_when_even_qualified_id_collides():
    original = "place_coke_on_cutting_board"
    qualified = f"{original}__given_root"
    seen = {original, qualified}  # both the plain id AND its precondition-qualified form taken
    result = _dedupe_id(original, seen, precondition=[])
    assert result == f"{qualified}_2"
    assert result not in seen


def test_discover_tree_dedupes_colliding_ids_from_different_nodes(tmp_path):
    """The exact real failure mode observed across every real discover_tree() run so far: the
    VLM independently proposes the SAME id from two different nodes (different preconditions).
    Must be renamed deterministically, and the full resulting edge list must be feedable through
    Manifest.add_edge for every edge without a ManifestError.
    """
    manifest = Manifest(tmp_path / "manifest.json")

    def caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        node = frozenset(satisfied_so_far)
        if node == frozenset():
            return [
                {
                    "id": "place_ceramic_mug_on_cutting_board",
                    "predicate": "object_on_top",
                    "predicate_args": {"object": "ceramic_mug", "reference_object": "cutting_board_a"},
                    "instruction": "place the mug",
                    "objects_involved": ["ceramic_mug", "cutting_board_a"],
                },
                {
                    "id": "place_coke_on_cutting_board",
                    "predicate": "object_on_top",
                    "predicate_args": {"object": "coke", "reference_object": "cutting_board_a"},
                    "instruction": "place the coke",
                    "objects_involved": ["coke", "cutting_board_a"],
                },
            ]
        if node == frozenset({"place_ceramic_mug_on_cutting_board"}):
            # Same id as the root's coke candidate above -- a real collision, different node.
            return [
                {
                    "id": "place_coke_on_cutting_board",
                    "predicate": "object_on_top",
                    "predicate_args": {"object": "coke", "reference_object": "cutting_board_a"},
                    "instruction": "place the coke, mug already placed",
                    "objects_involved": ["coke", "cutting_board_a"],
                }
            ]
        return []  # every other node: terminal

    ordered_edges = discover_tree(
        manifest,
        scene_objects=["ceramic_mug", "coke", "cutting_board_a"],
        phase_a_caller=caller,
        predicate_module={"object_on_top"},
        get_starting_state_summary=lambda node: None,
    )

    ids = [e["id"] for e in ordered_edges]
    assert len(ids) == len(set(ids)), f"duplicate ids leaked through: {ids}"
    assert "place_coke_on_cutting_board" in ids  # the root's original, unqualified id survives
    # The second (colliding) one got renamed, qualified by its own precondition.
    renamed = [i for i in ids if i.startswith("place_coke_on_cutting_board__given_")]
    assert renamed == ["place_coke_on_cutting_board__given_place_ceramic_mug_on_cutting_board"]

    # The whole point: this list must be fully add_edge-able with no ManifestError.
    fresh_manifest = Manifest(tmp_path / "manifest_replay.json")
    for edge in ordered_edges:
        fresh_manifest.add_edge(edge)  # raises ManifestError on any remaining duplicate
    assert len(fresh_manifest) == len(ordered_edges)


def test_discover_and_train_dedupes_ids_across_the_walk(tmp_path):
    """Same collision scenario as above, but through the real (faked-training) interleaved
    discover_and_train() path -- confirms manifest.add_edge (called for real inside
    discover_and_train, not just simulated afterward) never raises ManifestError.
    """
    manifest = Manifest(tmp_path / "manifest.json")

    def caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        node = frozenset(satisfied_so_far)
        if node == frozenset():
            return [
                {
                    "id": "place_mug",
                    "predicate": "object_on_top",
                    "predicate_args": {},
                    "instruction": "place the mug",
                    "objects_involved": [],
                },
                {
                    "id": "place_coke",
                    "predicate": "object_on_top",
                    "predicate_args": {},
                    "instruction": "place the coke",
                    "objects_involved": [],
                },
            ]
        if node == frozenset({"place_mug"}):
            return [
                {
                    "id": "place_coke",  # collides with the root's second candidate above
                    "predicate": "object_on_top",
                    "predicate_args": {},
                    "instruction": "place the coke, mug already placed",
                    "objects_involved": [],
                }
            ]
        return []

    def fake_run_training(edge, current_checkpoint, config_spec):
        return f"/fake/checkpoints/{edge['id']}/actor"

    def fake_collect_end_states(edge, checkpoint, config_spec):
        return f"/fake/end_states/{edge['id']}.jsonl"

    discover_and_train(
        manifest,
        base_checkpoint=None,
        scene_objects=["ceramic_mug", "coke", "cutting_board_a"],
        logs_root=tmp_path / "logs",
        spec_dir=tmp_path / "specs",
        phase_a_caller=caller,
        predicate_module=None,
        run_training=fake_run_training,
        collect_end_states=fake_collect_end_states,
    )

    ids = [e["id"] for e in manifest.all_edges()]
    assert len(ids) == len(set(ids)), f"duplicate ids leaked through: {ids}"
    assert "place_coke" in ids
    assert "place_coke__given_place_mug" in ids


def test_discover_and_train_initial_edges_skips_phase_a_for_start_node(tmp_path):
    """Continuing a walk from a node already discovered interactively (e.g. by a coordinator
    before handing off to an unattended driver) must not pay for a second, redundant Phase A
    call for that same node -- it should train the given initial_edges directly and only call
    Phase A for nodes BELOW the start node.
    """
    manifest = Manifest(tmp_path / "manifest.json")
    calls = []

    def caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        calls.append(sorted(satisfied_so_far))
        return []  # terminal everywhere Phase A actually gets called

    initial_edges = [
        {
            "id": "already_discovered_edge",
            "predicate": "object_on_top",
            "predicate_args": {},
            "instruction": "already discovered elsewhere",
            "objects_involved": [],
            "precondition": ["place_mug_on_cutting_board"],
            "produces_node": ["already_discovered_edge", "place_mug_on_cutting_board"],
            "reset_states_path": "/real/end_states.jsonl",
        }
    ]

    final_checkpoint = discover_and_train(
        manifest,
        base_checkpoint="/fake/edge1_checkpoint/actor",
        scene_objects=["ceramic_mug", "coke", "cutting_board_a"],
        logs_root=tmp_path / "logs",
        spec_dir=tmp_path / "specs",
        node=frozenset({"place_mug_on_cutting_board"}),
        phase_a_caller=caller,
        predicate_module=None,
        initial_edges=initial_edges,
        run_training=lambda edge, ckpt, spec: f"/fake/checkpoints/{edge['id']}/actor",
        collect_end_states=lambda edge, ckpt, spec: f"/fake/end_states/{edge['id']}.jsonl",
    )

    # Zero Phase A calls for the start node -- only (if any) for nodes strictly below it. Here
    # the only node below it (the produced child) IS visited and DOES call Phase A once,
    # returning [] (terminal) -- confirming discovery below the seeded node still works normally.
    assert calls == [sorted(["already_discovered_edge", "place_mug_on_cutting_board"])]

    assert final_checkpoint == "/fake/checkpoints/already_discovered_edge/actor"
    assert [e["id"] for e in manifest.all_edges()] == ["already_discovered_edge"]
    assert manifest.get_edge("already_discovered_edge")["checkpoint"] == final_checkpoint


def test_render_current_node_stub_returns_none():
    assert render_current_node(frozenset()) is None
    assert render_current_node(frozenset({"anything"})) is None


# ---------------------------------------------------------------------------
# generate_training_config
# ---------------------------------------------------------------------------


def test_generate_training_config_writes_edge_spec_and_builds_overrides(tmp_path):
    edge = {
        "id": "coke_on_cuttingboard__given_cup",
        "precondition": ["latteartcup_on_cuttingboard"],
        "predicate": "object_on_top",
        "predicate_args": {"object": "coke", "reference_object": "cuttingboard"},
        "instruction": "Pick up the coke can and place it on the cutting board",
        "objects_involved": ["coke", "cuttingboard"],
        "produces_node": ["coke_on_cuttingboard__given_cup", "latteartcup_on_cuttingboard"],
    }
    spec_dir = tmp_path / "edge_specs"

    config = generate_training_config(
        edge,
        reset_states_path="/logs/cup_done_states.jsonl",
        spec_dir=spec_dir,
        lora_path="/logs/checkpoint_v1/actor",
    )

    assert isinstance(config, TrainingConfigSpec)
    assert config.config_name == "generic_single_edge_grpo_openpi_pi05"
    overrides_str = " ".join(config.overrides)
    assert "env.train.init_params.task_file=generic_single_edge_task.py" in overrides_str
    assert "reset_states_path=/logs/cup_done_states.jsonl" in overrides_str
    assert "+actor.model.lora_path=/logs/checkpoint_v1/actor" in overrides_str

    written_path = spec_dir / "coke_on_cuttingboard__given_cup.json"
    assert written_path.is_file()
    written = json.loads(written_path.read_text())
    assert written["id"] == edge["id"]
    assert written["predicate_args"] == edge["predicate_args"]

    # str(config) is what PLAN.md's sketch interpolates directly into the run_embodiment.sh
    # command line -- confirm that composition actually works.
    assert str(config).startswith("generic_single_edge_grpo_openpi_pi05 ")


def test_generate_training_config_root_edge_reset_path_null(tmp_path):
    edge = {
        "id": "latteartcup_on_cuttingboard",
        "precondition": [],
        "predicate": "object_on_top",
        "predicate_args": {"object": "latteartcup", "reference_object": "cuttingboard"},
        "instruction": "ok",
        "objects_involved": ["latteartcup", "cuttingboard"],
        "produces_node": ["latteartcup_on_cuttingboard"],
    }
    config = generate_training_config(
        edge, reset_states_path=None, spec_dir=tmp_path / "specs"
    )
    assert "reset_states_path=null" in " ".join(config.overrides)


def test_generate_training_config_mirrors_overrides_onto_env_eval(tmp_path):
    """Regression test for a real bug found while wiring up real training: eval_embodiment.sh
    (used both for a standalone eval and for this project's end-state-collection pass) runs
    against env.eval, not env.train (eval_embodied_agent.py forces runner.only_eval=True). Without
    also overriding env.eval.init_params.*, the collection pass would silently check the wrong
    predicate (whatever placeholder is baked into the env yaml) instead of the edge just trained.
    """
    edge = {
        "id": "some_edge",
        "precondition": [],
        "predicate": "object_on_top",
        "predicate_args": {"object": "coke", "reference_object": "cutting_board_a"},
        "instruction": "place the coke",
        "objects_involved": ["coke", "cutting_board_a"],
        "produces_node": ["some_edge"],
    }
    config = generate_training_config(
        edge, reset_states_path="/logs/states.jsonl", spec_dir=tmp_path / "specs"
    )
    overrides_str = " ".join(config.overrides)

    for env_key in ("train", "eval"):
        assert f"env.{env_key}.init_params.task_file=generic_single_edge_task.py" in overrides_str
        assert f"+env.{env_key}.init_params.edge_spec_path=" in overrides_str
        assert f"env.{env_key}.init_params.reset_states_path=/logs/states.jsonl" in overrides_str

    # Exactly one edge_spec_path per env key (train, eval) -- both point at the SAME file (one
    # edge, one spec file), not two different ones.
    edge_spec_path = tmp_path / "specs" / "some_edge.json"
    assert overrides_str.count(f"+env.train.init_params.edge_spec_path={edge_spec_path}") == 1
    assert overrides_str.count(f"+env.eval.init_params.edge_spec_path={edge_spec_path}") == 1


# ---------------------------------------------------------------------------
# strip_actor_lora_path_override / _make_default_collect_end_states -- regression tests for a
# real bug: reusing a training TrainingConfigSpec's overrides verbatim for the collection/eval
# stage duplicated `+actor.model.lora_path=`, which Hydra's add-only `+` prefix correctly
# refused with ConfigCompositionException on a real run (place_coke_on_cutting_board).
# ---------------------------------------------------------------------------


def test_strip_actor_lora_path_override_removes_single_and_double_plus():
    overrides = [
        "env.train.init_params.task_file=generic_single_edge_task.py",
        "+actor.model.lora_path=/logs/parent_checkpoint/actor",
        "env.train.init_params.reset_states_path=null",
    ]
    result = strip_actor_lora_path_override(overrides)
    assert result == [
        "env.train.init_params.task_file=generic_single_edge_task.py",
        "env.train.init_params.reset_states_path=null",
    ]

    # Hydra also allows `++` (force-override) for the same key -- must be stripped too.
    overrides_double_plus = [
        "env.train.init_params.task_file=generic_single_edge_task.py",
        "++actor.model.lora_path=/logs/parent_checkpoint/actor",
    ]
    assert strip_actor_lora_path_override(overrides_double_plus) == [
        "env.train.init_params.task_file=generic_single_edge_task.py",
    ]


def test_strip_actor_lora_path_override_noop_when_absent():
    overrides = [
        "env.train.init_params.task_file=generic_single_edge_task.py",
        "env.train.init_params.reset_states_path=null",
    ]
    assert strip_actor_lora_path_override(overrides) == overrides


def test_strip_actor_lora_path_override_does_not_touch_unrelated_lora_like_keys():
    # A key that merely CONTAINS "actor.model.lora_path" as a substring elsewhere, or a
    # differently-named key, must survive -- only an exact `+`/`++actor.model.lora_path=` match
    # is stripped.
    overrides = [
        "+actor.model.lora_rank=8",
        "+some.other.actor.model.lora_path=should_not_match",
    ]
    assert strip_actor_lora_path_override(overrides) == overrides


def test_make_default_collect_end_states_builds_cmd_with_exactly_one_lora_path(
    tmp_path, monkeypatch
):
    """The regression test for the real bug: build a training TrainingConfigSpec (which bakes
    in a warm-start +actor.model.lora_path), then confirm the DEFAULT collect_end_states
    (obtained via generate_training_config + _make_default_collect_end_states, the exact
    combination train_sequence/discover_and_train use when no custom collect_end_states is
    injected) constructs a command with the collection stage's OWN checkpoint as
    +actor.model.lora_path exactly once -- not the training-time parent checkpoint, and not
    duplicated -- and never uses the broken runner.ckpt_path=<LoRA dir> mechanism.
    """
    edge = {
        "id": "place_coke_on_cutting_board",
        "precondition": [],
        "predicate": "object_on_top",
        "predicate_args": {"object": "coke", "reference_object": "cutting_board_a"},
        "instruction": "place the coke",
        "objects_involved": ["coke", "cutting_board_a"],
        "produces_node": ["place_coke_on_cutting_board"],
    }
    config_spec = generate_training_config(
        edge,
        reset_states_path=None,
        spec_dir=tmp_path / "specs",
        lora_path="/logs/edge1_parent_checkpoint/actor",  # training-time warm-start
    )
    assert "+actor.model.lora_path=/logs/edge1_parent_checkpoint/actor" in config_spec.overrides

    captured_cmd = {}

    def fake_launch_and_wait(cmd, *, log_path, **kwargs):
        captured_cmd["cmd"] = cmd
        return LaunchResult(returncode=0, log_path=str(log_path))

    monkeypatch.setattr(orchestrator, "launch_and_wait", fake_launch_and_wait)

    collect_end_states = _make_default_collect_end_states(tmp_path / "logs")
    this_edges_own_checkpoint = "/logs/place_coke_on_cutting_board_run/actor"
    result_path = collect_end_states(edge, this_edges_own_checkpoint, config_spec)

    cmd = captured_cmd["cmd"]
    # Exactly one +actor.model.lora_path, and it's the COLLECTION stage's own checkpoint.
    assert cmd.count("actor.model.lora_path=") == 1
    assert f"+actor.model.lora_path={this_edges_own_checkpoint}" in cmd
    assert "edge1_parent_checkpoint" not in cmd  # the training-time one was stripped
    # The broken mechanism (crashes with IsADirectoryError on a LoRA adapter dir) must be gone.
    assert "runner.ckpt_path=" not in cmd
    assert result_path.endswith("place_coke_on_cutting_board_end_states.jsonl")


# ---------------------------------------------------------------------------
# train_sequence -- faked launch, no real subprocess/GPU job
# ---------------------------------------------------------------------------


def test_train_sequence_threads_checkpoint_and_updates_manifest_in_order(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    ordered_edges = [
        {
            "id": "latteartcup_on_cuttingboard",
            "precondition": [],
            "predicate": "object_on_top",
            "predicate_args": {"object": "latteartcup", "reference_object": "cuttingboard"},
            "instruction": "cup",
            "objects_involved": ["latteartcup", "cuttingboard"],
            "produces_node": ["latteartcup_on_cuttingboard"],
            "reset_states_path": None,
        },
        {
            "id": "coke_on_cuttingboard__given_cup",
            "precondition": ["latteartcup_on_cuttingboard"],
            "predicate": "object_on_top",
            "predicate_args": {"object": "coke", "reference_object": "cuttingboard"},
            "instruction": "coke given cup",
            "objects_involved": ["coke", "cuttingboard"],
            "produces_node": [
                "coke_on_cuttingboard__given_cup",
                "latteartcup_on_cuttingboard",
            ],
            "reset_states_path": "/logs/cup_done_states.jsonl",
        },
    ]

    training_calls = []
    collect_calls = []

    def fake_run_training(edge, current_checkpoint, config_spec):
        training_calls.append((edge["id"], current_checkpoint))
        return f"/fake/checkpoints/{edge['id']}/actor"

    def fake_collect_end_states(edge, checkpoint, config_spec):
        collect_calls.append((edge["id"], checkpoint))
        return f"/fake/end_states/{edge['id']}.jsonl"

    final_checkpoint = train_sequence(
        manifest,
        ordered_edges,
        base_checkpoint="/fake/base_checkpoint",
        logs_root=tmp_path / "logs",
        spec_dir=tmp_path / "specs",
        run_training=fake_run_training,
        collect_end_states=fake_collect_end_states,
    )

    # Checkpoint threading: edge 2's warm-start is edge 1's OUTPUT checkpoint, not the base.
    assert training_calls == [
        ("latteartcup_on_cuttingboard", "/fake/base_checkpoint"),
        ("coke_on_cuttingboard__given_cup", "/fake/checkpoints/latteartcup_on_cuttingboard/actor"),
    ]
    assert final_checkpoint == "/fake/checkpoints/coke_on_cuttingboard__given_cup/actor"

    # End-state collection happened against each edge's OWN freshly-trained checkpoint.
    assert collect_calls == [
        ("latteartcup_on_cuttingboard", "/fake/checkpoints/latteartcup_on_cuttingboard/actor"),
        (
            "coke_on_cuttingboard__given_cup",
            "/fake/checkpoints/coke_on_cuttingboard__given_cup/actor",
        ),
    ]

    # Manifest updated in order, with checkpoints recorded.
    assert [e["id"] for e in manifest.all_edges()] == [
        "latteartcup_on_cuttingboard",
        "coke_on_cuttingboard__given_cup",
    ]
    assert (
        manifest.get_edge("latteartcup_on_cuttingboard")["checkpoint"]
        == "/fake/checkpoints/latteartcup_on_cuttingboard/actor"
    )

    # Child reset-states path was recorded against the produced node, for the NEXT discovery
    # round's children to pick up.
    assert (
        manifest.get_reset_states_path_for_children(frozenset({"latteartcup_on_cuttingboard"}))
        == "/fake/end_states/latteartcup_on_cuttingboard.jsonl"
    )


def test_train_sequence_raises_on_training_failure_and_does_not_update_manifest(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    edge = {
        "id": "latteartcup_on_cuttingboard",
        "precondition": [],
        "predicate": "object_on_top",
        "predicate_args": {},
        "instruction": "cup",
        "objects_involved": ["latteartcup", "cuttingboard"],
        "produces_node": ["latteartcup_on_cuttingboard"],
        "reset_states_path": None,
    }

    def failing_run_training(edge, current_checkpoint, config_spec):
        raise RuntimeError("simulated training failure")

    with pytest.raises(RuntimeError):
        train_sequence(
            manifest,
            [edge],
            base_checkpoint="/fake/base",
            logs_root=tmp_path / "logs",
            spec_dir=tmp_path / "specs",
            run_training=failing_run_training,
            collect_end_states=lambda *a, **k: pytest.fail("should never be called"),
        )
    assert len(manifest) == 0


# ---------------------------------------------------------------------------
# discover_and_train -- the interleaved discover-then-train loop (architecture option (a))
# ---------------------------------------------------------------------------


def _write_fake_end_states_jsonl(path, scene_objects, marker_pos):
    with open(path, "w") as f:
        for _ in range(5):
            row = {
                "objects": {
                    obj: [marker_pos, marker_pos, marker_pos, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    for obj in scene_objects
                },
                "robot_joint_pos": [],
                "robot_joint_vel": [],
            }
            f.write(json.dumps(row) + "\n")


def test_discover_and_train_child_node_is_grounded_in_real_post_training_end_states(tmp_path):
    """The centerpiece test for architecture decision (a): a node below the root must be
    proposed using REAL data collected from actually training its parent edge -- not text-only
    reasoning, and not fabricated. Fakes training/collection (no GPU spend) but writes a real
    JSONL end-states file to disk in the real schema, exactly as a real collection run would --
    proving the plumbing that connects "training just happened" to "the next Phase A call sees
    it" actually works end to end.
    """
    manifest = Manifest(tmp_path / "manifest.json")
    scene_objects = ["ceramic_mug", "coke", "cutting_board_a"]
    calls = []  # (satisfied_so_far, starting_state_summary)

    def caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        calls.append((sorted(satisfied_so_far), starting_state_summary))
        if satisfied_so_far:
            return []  # terminal after depth 1 -- keep this test small
        return [
            {
                "id": "cup_edge",
                "predicate": "object_on_top",
                "predicate_args": {"object": "ceramic_mug", "reference_object": "cutting_board_a"},
                "instruction": "place the cup",
                "objects_involved": ["ceramic_mug", "cutting_board_a"],
            }
        ]

    def fake_run_training(edge, current_checkpoint, config_spec):
        return f"/fake/checkpoints/{edge['id']}/actor"

    def fake_collect_end_states(edge, checkpoint, config_spec):
        end_states_path = tmp_path / f"{edge['id']}_end_states.jsonl"
        # Distinctive, unrealistic marker position -- proves the CHILD node's grounding text
        # really came from this file, not coincidentally from the root's real preset data.
        _write_fake_end_states_jsonl(end_states_path, scene_objects, marker_pos=9.999)
        return str(end_states_path)

    final_checkpoint = discover_and_train(
        manifest,
        base_checkpoint=None,
        scene_objects=scene_objects,
        logs_root=tmp_path / "logs",
        spec_dir=tmp_path / "specs",
        phase_a_caller=caller,
        predicate_module=None,
        run_training=fake_run_training,
        collect_end_states=fake_collect_end_states,
    )

    assert final_checkpoint == "/fake/checkpoints/cup_edge/actor"
    assert len(calls) == 2

    root_satisfied, root_summary = calls[0]
    assert root_satisfied == []
    # Root grounding is real too (from the real initial_conditions.json), just not the focus
    # of this test -- confirm it's at least present and not the child's fake data.
    assert root_summary is not None
    assert "9.999" not in root_summary

    child_satisfied, child_summary = calls[1]
    assert child_satisfied == ["cup_edge"]
    # The key assertion: the child call's grounding contains the EXACT marker value written by
    # fake_collect_end_states above -- real data flowing from "training just happened" into the
    # very next discovery call, not a screenshot, not None, not root data reused.
    assert child_summary is not None
    assert "9.999" in child_summary
    assert "ACTUAL data from a trained policy" in child_summary

    # And the manifest was actually updated for real along the way (not just previewed).
    assert len(manifest) == 1
    assert manifest.get_edge("cup_edge")["checkpoint"] == "/fake/checkpoints/cup_edge/actor"
    assert manifest.is_terminal(frozenset({"cup_edge"}))


def test_discover_and_train_respects_max_total_edges(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")

    def verbose_caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        node = frozenset(satisfied_so_far)
        prefix = "_".join(sorted(node)) or "root"
        return [
            {
                "id": f"{prefix}__opt{i}",
                "predicate": "object_on_top",
                "predicate_args": {},
                "instruction": "keep going",
                "objects_involved": [],
            }
            for i in range(3)
        ]

    call_count = {"n": 0}

    def fake_run_training(edge, current_checkpoint, config_spec):
        call_count["n"] += 1
        return f"/fake/checkpoints/{call_count['n']}/actor"

    def fake_collect_end_states(edge, checkpoint, config_spec):
        return f"/fake/end_states/{edge['id']}.jsonl"

    discover_and_train(
        manifest,
        base_checkpoint=None,
        scene_objects=SCENE_OBJECTS,
        logs_root=tmp_path / "logs",
        spec_dir=tmp_path / "specs",
        max_depth=10,
        max_total_edges=4,
        phase_a_caller=verbose_caller,
        predicate_module=None,
        run_training=fake_run_training,
        collect_end_states=fake_collect_end_states,
    )
    # The manifest (real edges actually trained) must never exceed the cap, same guarantee as
    # discover_tree's preview cap, but now enforced against real (faked) training calls too.
    assert len(manifest) <= 4


def test_discover_and_train_raises_on_training_failure(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")

    def caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        return [
            {
                "id": "cup_edge",
                "predicate": "object_on_top",
                "predicate_args": {},
                "instruction": "place the cup",
                "objects_involved": [],
            }
        ]

    def failing_run_training(edge, current_checkpoint, config_spec):
        raise RuntimeError("simulated training failure")

    with pytest.raises(RuntimeError):
        discover_and_train(
            manifest,
            base_checkpoint=None,
            scene_objects=SCENE_OBJECTS,
            logs_root=tmp_path / "logs",
            spec_dir=tmp_path / "specs",
            phase_a_caller=caller,
            predicate_module=None,
            run_training=failing_run_training,
            collect_end_states=lambda *a, **k: pytest.fail("should never be called"),
        )
    assert len(manifest) == 0


# ---------------------------------------------------------------------------
# discover_and_train_by_level -- breadth-first walk (architecture correction)
# ---------------------------------------------------------------------------


def _synthetic_two_branch_phase_a_caller():
    """Root branches into two ({A}, {B}); each has exactly one child ({A,A2}, {B,B2}); both
    leaves terminal. The exact shape a DFS-vs-BFS ordering bug is observable on: DFS would
    train A, then A2 (A's child), THEN B -- BFS must train A and B (both depth 0) before
    EITHER A2 or B2 (depth 1).
    """

    def caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        # predicate_args is deliberately distinct per candidate (real object_on_top edges
        # always differ by which object they place -- an id collision aside, two genuinely
        # different subtasks should never have identical predicate_args) so
        # _find_matching_existing_edge's content-based matching can't spuriously conflate them.
        node = frozenset(satisfied_so_far)
        if node == frozenset():
            return [
                {
                    "id": "edge_A",
                    "predicate": "object_on_top",
                    "predicate_args": {"object": "A"},
                    "instruction": "do A",
                    "objects_involved": [],
                },
                {
                    "id": "edge_B",
                    "predicate": "object_on_top",
                    "predicate_args": {"object": "B"},
                    "instruction": "do B",
                    "objects_involved": [],
                },
            ]
        if node == frozenset({"edge_A"}):
            return [
                {
                    "id": "edge_A2",
                    "predicate": "object_on_top",
                    "predicate_args": {"object": "A2"},
                    "instruction": "do A2",
                    "objects_involved": [],
                }
            ]
        if node == frozenset({"edge_B"}):
            return [
                {
                    "id": "edge_B2",
                    "predicate": "object_on_top",
                    "predicate_args": {"object": "B2"},
                    "instruction": "do B2",
                    "objects_involved": [],
                }
            ]
        return []  # both {A,A2} and {B,B2} leaves: terminal

    return caller


def test_discover_and_train_by_level_visits_depth_0_fully_before_depth_1(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    caller = _synthetic_two_branch_phase_a_caller()

    train_order = []

    def fake_run_training(edge, current_checkpoint, config_spec):
        train_order.append(edge["id"])
        return f"/fake/checkpoints/{edge['id']}/actor"

    def fake_collect_end_states(edge, checkpoint, config_spec):
        return f"/fake/end_states/{edge['id']}.jsonl"

    final_checkpoint = discover_and_train_by_level(
        manifest,
        base_checkpoint=None,
        scene_objects=SCENE_OBJECTS,
        logs_root=tmp_path / "logs",
        spec_dir=tmp_path / "specs",
        phase_a_caller=caller,
        predicate_module=None,
        run_training=fake_run_training,
        collect_end_states=fake_collect_end_states,
    )

    # The crux of the fix: both depth-0 edges (A, B) trained before EITHER depth-1 edge (A2, B2)
    # -- a DFS walk would have produced ["edge_A", "edge_A2", "edge_B", "edge_B2"] instead.
    assert train_order[:2] == ["edge_A", "edge_B"] or train_order[:2] == ["edge_B", "edge_A"]
    assert set(train_order[:2]) == {"edge_A", "edge_B"}
    assert set(train_order[2:]) == {"edge_A2", "edge_B2"}
    assert len(train_order) == 4

    # Single continually-evolving checkpoint -- final result is whatever the LAST edge trained
    # produced, and every edge in the manifest has SOME checkpoint recorded (not asserting a
    # specific lineage shape beyond "one checkpoint per edge, threaded through in train order").
    assert final_checkpoint == f"/fake/checkpoints/{train_order[-1]}/actor"
    assert len(manifest) == 4
    assert {e["id"] for e in manifest.all_edges()} == {"edge_A", "edge_B", "edge_A2", "edge_B2"}


def test_discover_and_train_by_level_writes_level_queue_files_with_deduped_ids(tmp_path):
    """level_queue_dir (opt-in, for external progress dashboards): as soon as one depth's
    level_edges is fully computed, write it verbatim to level_<depth>_queue.json -- BEFORE any
    of that level's edges are trained, and with ids already deduped (not the raw,
    possibly-colliding ids a seeded/raw candidate file might have). Reproduces the real scenario
    that motivated this: a root candidate and a later depth-1 seed sharing a raw id
    ("shared_id") -- the queue file must show the depth-1 one already renamed.
    """
    manifest = Manifest(tmp_path / "manifest.json")
    queue_dir = tmp_path / "queues"

    root_candidates = [
        {
            "id": "shared_id",
            "predicate": "object_on_top",
            "predicate_args": {"object": "root_variant"},
            "instruction": "root variant",
            "objects_involved": [],
            "precondition": [],
            "produces_node": ["shared_id"],
            "reset_states_path": None,
        }
    ]
    depth1_candidates = [
        {
            "id": "shared_id",  # raw id collides with the root's -- must appear DEDUPED in the queue file
            "predicate": "object_on_top",
            "predicate_args": {"object": "given_shared_id_variant"},
            "instruction": "given shared_id is already done",
            "objects_involved": [],
            "precondition": ["shared_id"],
            "produces_node": ["shared_id", "shared_id"],
            "reset_states_path": "/real/some_end_states.jsonl",
        }
    ]

    def terminal_caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        return []

    discover_and_train_by_level(
        manifest,
        base_checkpoint=None,
        scene_objects=SCENE_OBJECTS,
        logs_root=tmp_path / "logs",
        spec_dir=tmp_path / "specs",
        phase_a_caller=terminal_caller,
        predicate_module=None,
        initial_edges_by_node={
            frozenset(): root_candidates,
            frozenset({"shared_id"}): depth1_candidates,
        },
        run_training=lambda edge, ckpt, spec: f"/fake/checkpoints/{edge['id']}/actor",
        collect_end_states=lambda edge, ckpt, spec: f"/fake/end_states/{edge['id']}.jsonl",
        level_queue_dir=queue_dir,
    )

    level0_path = queue_dir / "level_0_queue.json"
    level1_path = queue_dir / "level_1_queue.json"
    assert level0_path.is_file()
    assert level1_path.is_file()

    level0 = json.loads(level0_path.read_text())
    level1 = json.loads(level1_path.read_text())

    assert [e["id"] for e in level0] == ["shared_id"]
    # The crux: the depth-1 queue file shows the DEDUPED id, not the raw "shared_id" the seed
    # file itself had -- a dashboard reading this file needs no dedup logic of its own.
    assert len(level1) == 1
    assert level1[0]["id"] != "shared_id"
    assert level1[0]["id"].startswith("shared_id__given_")
    assert level1[0]["precondition"] == ["shared_id"]


def test_discover_and_train_by_level_no_queue_files_when_level_queue_dir_omitted(tmp_path):
    manifest = Manifest(tmp_path / "manifest.json")
    caller = _synthetic_two_branch_phase_a_caller()

    discover_and_train_by_level(
        manifest,
        base_checkpoint=None,
        scene_objects=SCENE_OBJECTS,
        logs_root=tmp_path / "logs",
        spec_dir=tmp_path / "specs",
        phase_a_caller=caller,
        predicate_module=None,
        run_training=lambda edge, ckpt, spec: f"/fake/checkpoints/{edge['id']}/actor",
        collect_end_states=lambda edge, ckpt, spec: f"/fake/end_states/{edge['id']}.jsonl",
    )
    # No level_queue_dir passed -- must be fully opt-in, no files written anywhere by default.
    assert not any(tmp_path.rglob("level_*_queue.json"))


def test_discover_and_train_by_level_threads_single_checkpoint_across_branches(tmp_path):
    """Confirms the "one checkpoint, not one per branch" invariant survives BFS re-ordering:
    every edge, across the ENTIRE run (not per-branch), warm-starts from whatever the single
    lineage's checkpoint happened to be immediately before it trained -- e.g. edge_B2 (a
    depth-1 edge under the edge_B branch) warm-starts from edge_A2's output (the edge trained
    immediately before it, from a DIFFERENT branch), not from edge_B's own output specifically.
    That's the concrete difference between "one checkpoint" and "one checkpoint per branch".
    """
    manifest = Manifest(tmp_path / "manifest.json")
    caller = _synthetic_two_branch_phase_a_caller()
    warm_starts = {}

    def fake_run_training(edge, current_checkpoint, config_spec):
        warm_starts[edge["id"]] = current_checkpoint
        return f"/fake/checkpoints/{edge['id']}/actor"

    def fake_collect_end_states(edge, checkpoint, config_spec):
        return f"/fake/end_states/{edge['id']}.jsonl"

    discover_and_train_by_level(
        manifest,
        base_checkpoint="/fake/base/actor",
        scene_objects=SCENE_OBJECTS,
        logs_root=tmp_path / "logs",
        spec_dir=tmp_path / "specs",
        phase_a_caller=caller,
        predicate_module=None,
        run_training=fake_run_training,
        collect_end_states=fake_collect_end_states,
    )

    depth0_order = [e for e in ["edge_A", "edge_B"] if warm_starts[e] == "/fake/base/actor"]
    assert len(depth0_order) == 1  # exactly one depth-0 edge warm-starts from the base
    first, second = (
        ("edge_A", "edge_B") if depth0_order == ["edge_A"] else ("edge_B", "edge_A")
    )
    assert warm_starts[first] == "/fake/base/actor"
    assert warm_starts[second] == f"/fake/checkpoints/{first}/actor"

    # Depth 1's frontier order is deterministic regardless of depth-0 training order (sorted by
    # node content: {edge_A} always precedes {edge_B} lexicographically) -- so edge_A2 always
    # trains right after depth 0 finishes, and edge_B2 always trains right after edge_A2.
    depth0_final_checkpoint = f"/fake/checkpoints/{second}/actor"
    assert warm_starts["edge_A2"] == depth0_final_checkpoint
    # The crux of this test: edge_B2 (edge_B's own branch) warm-starts from edge_A2's output
    # (a DIFFERENT branch's edge, trained immediately before it in the single lineage) -- NOT
    # from edge_B's own output. A "one checkpoint per branch" bug would produce the latter.
    assert warm_starts["edge_B2"] == "/fake/checkpoints/edge_A2/actor"


def test_discover_and_train_by_level_resumed_run_threads_checkpoint_from_manifest(tmp_path):
    """Regression test for a real bug caught on a live run: when RESUMING (both root edges
    already trained and persisted in the manifest from a PRIOR call -- exactly what happens on
    every driver relaunch, since root_candidates.json gets re-seeded every time), the
    "already trained, don't retrain" skip path did not update the running `checkpoint` variable
    at all -- so a depth-1 edge discovered/trained in THIS call warm-started from the stale
    `base_checkpoint` argument (root1's checkpoint) instead of root2's checkpoint (the real,
    single-lineage checkpoint AFTER both root edges were actually trained, in order, e.g. in an
    earlier call). Confirmed on a live run for place_mug_on_cutting_board__given_place_coke_on_
    cutting_board, which was launched with edge 1's original checkpoint instead of edge 2's.

    This test mirrors that exact shape: 2 root edges already in the manifest (root1's checkpoint
    added first, root2's second -- root2 is the REAL "current" single-lineage checkpoint), fed
    back in via initial_edges_by_node (as the driver always does), plus a depth-1 candidate
    triggered once both roots are recognized as already-done. The depth-1 edge's run_training
    call must receive root2's checkpoint as its warm-start argument -- NOT base_checkpoint
    (deliberately a third, distinct fake value here so the test can't pass by coincidental
    string equality with either real checkpoint).
    """
    manifest = Manifest(tmp_path / "manifest.json")

    # Both root edges already trained and persisted, in real chronological order (root1 then
    # root2) -- exactly what a real prior successful call leaves behind.
    manifest.add_edge(
        {
            "id": "root1",
            "predicate": "object_on_top",
            "predicate_args": {"object": "root1"},
            "instruction": "root1",
            "objects_involved": [],
            "precondition": [],
            "produces_node": ["root1"],
            "reset_states_path": None,
            "checkpoint": "/real/root1_checkpoint/actor",
        }
    )
    manifest.add_edge(
        {
            "id": "root2",
            "predicate": "object_on_top",
            "predicate_args": {"object": "root2"},
            "instruction": "root2",
            "objects_involved": [],
            "precondition": [],
            "produces_node": ["root2"],
            "reset_states_path": None,
            "checkpoint": "/real/root2_checkpoint/actor",  # the TRUE current single-lineage state
        }
    )
    manifest.set_reset_states_path_for_children(frozenset({"root2"}), "/real/root2_end_states.jsonl")

    root_candidates = [
        {
            "id": "root1",
            "predicate": "object_on_top",
            "predicate_args": {"object": "root1"},
            "instruction": "root1",
            "objects_involved": [],
        },
        {
            "id": "root2",
            "predicate": "object_on_top",
            "predicate_args": {"object": "root2"},
            "instruction": "root2",
            "objects_involved": [],
        },
    ]

    def caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        node = frozenset(satisfied_so_far)
        if node == frozenset({"root2"}):
            return [
                {
                    "id": "depth1_edge",
                    "predicate": "object_on_top",
                    "predicate_args": {"object": "depth1"},
                    "instruction": "depth1",
                    "objects_involved": [],
                }
            ]
        return []  # {root1} node: terminal (no children found from that branch in this test)

    warm_starts = {}

    def fake_run_training(edge, current_checkpoint, config_spec):
        warm_starts[edge["id"]] = current_checkpoint
        return f"/fake/checkpoints/{edge['id']}/actor"

    discover_and_train_by_level(
        manifest,
        base_checkpoint="/fake/DISTINCT_base_checkpoint/actor",  # must NOT be used for depth1
        scene_objects=SCENE_OBJECTS,
        logs_root=tmp_path / "logs",
        spec_dir=tmp_path / "specs",
        phase_a_caller=caller,
        predicate_module=None,
        initial_edges_by_node={frozenset(): root_candidates},
        run_training=fake_run_training,
        collect_end_states=lambda edge, ckpt, spec: f"/fake/end_states/{edge['id']}.jsonl",
    )

    # Neither root edge was retrained (both already existed).
    assert "root1" not in warm_starts
    assert "root2" not in warm_starts
    # The crux: depth1_edge must warm-start from root2's REAL checkpoint (the true current
    # single-lineage state after both roots), not base_checkpoint and not root1's checkpoint.
    assert "depth1_edge" in warm_starts
    assert warm_starts["depth1_edge"] == "/real/root2_checkpoint/actor"


def test_discover_and_train_by_level_initial_edges_by_node_skips_phase_a(tmp_path):
    """Seeding multiple already-discovered nodes at once (e.g. root candidates already known
    from an earlier call, AND a depth-1 node's children already discovered too) must skip Phase
    A for exactly those nodes -- the real scenario this was built for: edge 1 already trained,
    its children already discovered, continuing the walk from the true root without redoing
    either.
    """
    manifest = Manifest(tmp_path / "manifest.json")
    calls = []

    def caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        calls.append(sorted(satisfied_so_far))
        return []  # anything actually calling Phase A here is terminal -- we only care whether
        # it gets called at all for the seeded nodes

    root_candidates = [
        {
            "id": "edge_A",
            "predicate": "object_on_top",
            "predicate_args": {"object": "A"},
            "instruction": "do A",
            "objects_involved": [],
            "precondition": [],
            "produces_node": ["edge_A"],
            "reset_states_path": None,
        },
        {
            "id": "edge_B",
            "predicate": "object_on_top",
            "predicate_args": {"object": "B"},
            "instruction": "do B",
            "objects_involved": [],
            "precondition": [],
            "produces_node": ["edge_B"],
            "reset_states_path": None,
        },
    ]
    edge_a_children = [
        {
            "id": "edge_A2",
            "predicate": "object_on_top",
            "predicate_args": {"object": "A2"},
            "instruction": "do A2",
            "objects_involved": [],
            "precondition": ["edge_A"],
            "produces_node": ["edge_A", "edge_A2"],
            "reset_states_path": "/real/edge_A_end_states.jsonl",
        }
    ]

    discover_and_train_by_level(
        manifest,
        base_checkpoint=None,
        scene_objects=SCENE_OBJECTS,
        logs_root=tmp_path / "logs",
        spec_dir=tmp_path / "specs",
        phase_a_caller=caller,
        predicate_module=None,
        initial_edges_by_node={
            frozenset(): root_candidates,
            frozenset({"edge_A"}): edge_a_children,
        },
        run_training=lambda edge, ckpt, spec: f"/fake/checkpoints/{edge['id']}/actor",
        collect_end_states=lambda edge, ckpt, spec: f"/fake/end_states/{edge['id']}.jsonl",
    )

    # Phase A only ever called for nodes NOT seeded: {edge_B} (depth 1, discovered normally,
    # terminal) and {edge_A, edge_A2} (depth 2, below the seeded edge_A2 child, terminal).
    # Never called for the root ([]) or {edge_A} -- both were seeded via initial_edges_by_node.
    assert sorted(calls) == sorted([["edge_B"], ["edge_A", "edge_A2"]])
    assert [] not in calls
    assert ["edge_A"] not in calls


def test_discover_and_train_by_level_dedupes_colliding_ids_across_seeded_nodes(tmp_path):
    """Regression test for a real bug caught in a dry run before spending real GPU time:
    initial_edges_by_node's entries typically come from SEPARATE real Phase A calls (e.g. the
    root's candidates from one call, hours later a depth-1 node's children from another) that
    have no knowledge of each other, so they can collide on id even though they're semantically
    different edges -- exactly what happened with two independently-discovered
    "place_coke_on_cutting_board" candidates (one a root/from-scratch edge, one a "given mug is
    already placed" variant). Seeded edges bypass _discover_node_edges entirely (that's the
    whole point -- no redundant Phase A call), so they need their OWN dedup pass; without it,
    the second one hit Manifest.add_edge's duplicate-id ManifestError.
    """
    manifest = Manifest(tmp_path / "manifest.json")

    root_candidates = [
        {
            "id": "shared_id",  # collides with the depth-1 seed below, different precondition
            "predicate": "object_on_top",
            "predicate_args": {"object": "root_variant"},
            "instruction": "root variant",
            "objects_involved": [],
            "precondition": [],
            "produces_node": ["shared_id"],
            "reset_states_path": None,
        }
    ]
    depth1_candidates = [
        {
            "id": "shared_id",  # same literal id, genuinely different edge (different precondition)
            "predicate": "object_on_top",
            "predicate_args": {"object": "given_shared_id_variant"},
            "instruction": "given shared_id is already done",
            "objects_involved": [],
            "precondition": ["shared_id"],
            "produces_node": ["shared_id", "shared_id"],  # will be corrected by the dedup fixup
            "reset_states_path": "/real/some_end_states.jsonl",
        }
    ]

    def terminal_caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        return []

    trained_ids = []

    def fake_run_training(edge, checkpoint, config_spec):
        trained_ids.append(edge["id"])
        return f"/fake/checkpoints/{edge['id']}/actor"

    discover_and_train_by_level(
        manifest,
        base_checkpoint=None,
        scene_objects=SCENE_OBJECTS,
        logs_root=tmp_path / "logs",
        spec_dir=tmp_path / "specs",
        phase_a_caller=terminal_caller,
        predicate_module=None,
        initial_edges_by_node={
            frozenset(): root_candidates,
            frozenset({"shared_id"}): depth1_candidates,
        },
        run_training=fake_run_training,
        collect_end_states=lambda edge, ckpt, spec: f"/fake/end_states/{edge['id']}.jsonl",
    )

    # Both edges trained (no ManifestError), with distinct ids -- the second was renamed.
    assert len(trained_ids) == 2
    assert len(set(trained_ids)) == 2
    assert "shared_id" in trained_ids
    renamed = [i for i in trained_ids if i != "shared_id"]
    assert len(renamed) == 1
    assert renamed[0].startswith("shared_id__given_")

    # The renamed edge's produces_node was fixed up to match its new id, not left pointing at a
    # phantom node using the old (pre-rename) id.
    renamed_edge = manifest.get_edge(renamed[0])
    assert renamed_edge["produces_node"] == sorted(["shared_id", renamed[0]])


def test_discover_and_train_by_level_skips_retraining_already_trained_edge(tmp_path):
    """A candidate discovered/seeded whose id already exists in manifest (e.g. edge 1, trained
    in a prior run before this call) must not be retrained -- but its child node must still
    advance into the next level's frontier, using the manifest's EXISTING
    reset_states_path_for_children rather than re-collecting.
    """
    manifest = Manifest(tmp_path / "manifest.json")
    manifest.add_edge(
        {
            "id": "edge_A",
            "predicate": "object_on_top",
            "predicate_args": {},
            "instruction": "do A",
            "objects_involved": [],
            "precondition": [],
            "produces_node": ["edge_A"],
            "reset_states_path": None,
            "checkpoint": "/real/edge_A/actor",
        }
    )
    manifest.set_reset_states_path_for_children(
        frozenset({"edge_A"}), "/real/edge_A_end_states.jsonl"
    )

    def caller(
        object_list,
        satisfied_so_far,
        starting_state_summary=None,
        predicate_menu=None,
        *,
        scene_screenshot=None,
    ):
        node = frozenset(satisfied_so_far)
        if node == frozenset():
            # Re-discovering the root re-proposes edge_A (already trained) -- realistic: Phase A
            # has no memory of what's already trained, only sees satisfied_so_far=[].
            return [
                {
                    "id": "edge_A",
                    "predicate": "object_on_top",
                    "predicate_args": {},
                    "instruction": "do A",
                    "objects_involved": [],
                }
            ]
        if node == frozenset({"edge_A"}):
            return [
                {
                    "id": "edge_A2",
                    "predicate": "object_on_top",
                    "predicate_args": {},
                    "instruction": "do A2",
                    "objects_involved": [],
                }
            ]
        return []

    train_calls = []

    def fake_run_training(edge, checkpoint, config_spec):
        train_calls.append(edge["id"])
        return f"/fake/checkpoints/{edge['id']}/actor"

    discover_and_train_by_level(
        manifest,
        base_checkpoint="/real/edge_A/actor",
        scene_objects=SCENE_OBJECTS,
        logs_root=tmp_path / "logs",
        spec_dir=tmp_path / "specs",
        phase_a_caller=caller,
        predicate_module=None,
        run_training=fake_run_training,
        collect_end_states=lambda edge, ckpt, spec: (
            pytest.fail("should never collect end-states for the already-trained edge_A")
            if edge["id"] == "edge_A"
            else f"/fake/end_states/{edge['id']}.jsonl"
        ),
    )

    assert "edge_A" not in train_calls  # not retrained
    assert "edge_A2" in train_calls  # but its child still got discovered and trained
    assert len(manifest) == 2  # edge_A (pre-existing) + edge_A2 (newly trained)
    # The child node advanced using edge_A's EXISTING produces_node/reset path, not a
    # phantom re-derivation -- confirmed indirectly by edge_A2 actually having been
    # discovered from node {edge_A} (its precondition) and trained successfully above.
    assert manifest.get_edge("edge_A2")["precondition"] == ["edge_A"]


# ---------------------------------------------------------------------------
# launch_and_wait -- real plumbing, exercised against a trivial/instantaneous command only
# ---------------------------------------------------------------------------


def test_launch_and_wait_real_detached_launch_captures_exit_code_and_log(tmp_path):
    log_path = tmp_path / "job.log"
    result = launch_and_wait(
        "echo hello-from-detached-job; exit 3", log_path=log_path, poll_interval_s=0.05
    )
    assert isinstance(result, LaunchResult)
    assert result.returncode == 3
    assert not result.ok
    assert "hello-from-detached-job" in log_path.read_text()


def test_launch_and_wait_real_detached_launch_success(tmp_path):
    log_path = tmp_path / "job_ok.log"
    result = launch_and_wait("true", log_path=log_path, poll_interval_s=0.05)
    assert result.ok
    assert result.returncode == 0


def test_launch_and_wait_times_out_on_a_hanging_command(tmp_path):
    log_path = tmp_path / "job_hangs.log"
    with pytest.raises(TimeoutError):
        launch_and_wait(
            "sleep 30", log_path=log_path, poll_interval_s=0.05, timeout_s=0.3
        )

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

"""Unit tests for rlinf.envs.isaaclab.tasks.discovery.manifest.Manifest.

Pure-Python, no torch/isaaclab/GPU required -- these should pass under plain pytest.
"""

import pytest

from rlinf.envs.isaaclab.tasks.discovery.manifest import Manifest, ManifestError

CUP_EDGE = {
    "id": "latteartcup_on_cuttingboard",
    "precondition": [],
    "predicate": "object_on_top",
    "predicate_args": {"object": "latteartcup", "reference_object": "cuttingboard"},
    "instruction": "Pick up the latte art cup and place it on the cutting board",
    "objects_involved": ["latteartcup", "cuttingboard"],
    "produces_node": ["latteartcup_on_cuttingboard"],
}

COKE_GIVEN_CUP_EDGE = {
    "id": "coke_on_cuttingboard__given_cup",
    "precondition": ["latteartcup_on_cuttingboard"],
    "predicate": "object_on_top",
    "predicate_args": {"object": "coke", "reference_object": "cuttingboard"},
    "instruction": "Pick up the coke can and place it on the cutting board",
    "objects_involved": ["coke", "cuttingboard"],
    "produces_node": ["coke_on_cuttingboard__given_cup", "latteartcup_on_cuttingboard"],
}


def test_add_edge_and_all_edges_roundtrip(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    stored = m.add_edge(CUP_EDGE)
    assert stored["id"] == "latteartcup_on_cuttingboard"
    assert len(m) == 1
    assert m.all_edges()[0]["id"] == "latteartcup_on_cuttingboard"


def test_add_edge_persists_to_disk_and_reloads(tmp_path):
    path = tmp_path / "manifest.json"
    m1 = Manifest(path)
    m1.add_edge(CUP_EDGE)
    m1.add_edge(COKE_GIVEN_CUP_EDGE)

    m2 = Manifest(path)  # fresh instance, reads what m1 wrote
    assert len(m2) == 2
    assert {e["id"] for e in m2.all_edges()} == {
        "latteartcup_on_cuttingboard",
        "coke_on_cuttingboard__given_cup",
    }


def test_add_edge_rejects_missing_required_field(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    bad = dict(CUP_EDGE)
    del bad["predicate"]
    with pytest.raises(ManifestError):
        m.add_edge(bad)
    assert len(m) == 0


def test_add_edge_rejects_duplicate_id(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    m.add_edge(CUP_EDGE)
    with pytest.raises(ManifestError):
        m.add_edge(CUP_EDGE)
    assert len(m) == 1


def test_edges_from_exact_match(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    m.add_edge(CUP_EDGE)
    m.add_edge(COKE_GIVEN_CUP_EDGE)

    root_edges = m.edges_from(frozenset())
    assert [e["id"] for e in root_edges] == ["latteartcup_on_cuttingboard"]

    cup_edges = m.edges_from(frozenset({"latteartcup_on_cuttingboard"}))
    assert [e["id"] for e in cup_edges] == ["coke_on_cuttingboard__given_cup"]


def test_edges_from_does_not_subset_match(tmp_path):
    """The one negative case the exact-match rule exists for: a superset node must NOT match
    an edge whose precondition is a strict subset of it, even though every individual
    requirement in that subset is technically satisfied.
    """
    m = Manifest(tmp_path / "manifest.json")
    m.add_edge(CUP_EDGE)  # precondition == [] (root only)

    # A node that is a strict superset of CUP_EDGE's precondition ([]) -- e.g. some other
    # subtask already happened to be satisfied too. CUP_EDGE must NOT be offered here: its
    # reset-states distribution was collected under precondition==[] specifically, not under
    # "[] plus something else".
    superset_node = frozenset({"some_other_subtask_already_done"})
    assert m.edges_from(superset_node) == []

    # Sanity: it DOES match its own exact precondition.
    assert [e["id"] for e in m.edges_from(frozenset())] == ["latteartcup_on_cuttingboard"]


def test_edges_from_order_independent_and_dedupes(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    edge = dict(COKE_GIVEN_CUP_EDGE)
    edge["precondition"] = ["latteartcup_on_cuttingboard", "latteartcup_on_cuttingboard"]  # dup
    m.add_edge(edge)
    # Query with a differently-ordered/constructed node -- must still match.
    matches = m.edges_from({"latteartcup_on_cuttingboard"})
    assert len(matches) == 1


def test_get_producing_edge(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    m.add_edge(CUP_EDGE)
    m.add_edge(COKE_GIVEN_CUP_EDGE)

    assert m.get_producing_edge(frozenset()) is None  # nothing produces the root
    assert m.get_producing_edge(frozenset({"not_a_real_node"})) is None

    produced = m.get_producing_edge(frozenset({"latteartcup_on_cuttingboard"}))
    assert produced["id"] == "latteartcup_on_cuttingboard"


def test_mark_terminal_and_is_terminal(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    node = frozenset({"latteartcup_on_cuttingboard", "coke_on_cuttingboard__given_cup"})
    assert not m.is_terminal(node)
    m.mark_terminal(node)
    assert m.is_terminal(node)
    assert not m.is_terminal(frozenset())


def test_reset_states_path_for_children_set_and_get(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    assert m.get_reset_states_path_for_children(frozenset()) is None

    m.set_reset_states_path_for_children(frozenset(), "/logs/cup_done_states.jsonl")
    assert (
        m.get_reset_states_path_for_children(frozenset()) == "/logs/cup_done_states.jsonl"
    )

    node = frozenset({"latteartcup_on_cuttingboard"})
    m.set_reset_states_path_for_children(node, "/logs/both_states.jsonl")
    assert m.get_reset_states_path_for_children(node) == "/logs/both_states.jsonl"
    # unrelated node unaffected
    assert m.get_reset_states_path_for_children(frozenset({"coke_on_cuttingboard"})) is None


def test_reset_states_path_persists_across_reload(tmp_path):
    path = tmp_path / "manifest.json"
    m1 = Manifest(path)
    m1.set_reset_states_path_for_children(frozenset(), "/logs/root_children.jsonl")
    m1.mark_terminal(frozenset({"a", "b"}))

    m2 = Manifest(path)
    assert m2.get_reset_states_path_for_children(frozenset()) == "/logs/root_children.jsonl"
    assert m2.is_terminal(frozenset({"b", "a"}))


def test_get_edge(tmp_path):
    m = Manifest(tmp_path / "manifest.json")
    m.add_edge(CUP_EDGE)
    assert m.get_edge("latteartcup_on_cuttingboard")["instruction"].startswith("Pick up")
    assert m.get_edge("does_not_exist") is None

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

"""Unit tests for rlinf.envs.isaaclab.tasks.discovery.predicates.

Deliberately does NOT import RoboLab's real `conditionals` module (that requires isaaclab / a
GPU container) -- `list_predicate_names`/`build_predicate_menu` parse the source file with
`ast`, and `validate_phase_a` is exercised against a small fake "predicate module" (a plain
object with the right attributes) as well as a plain name set, to mirror both ways production
code might call it (a real imported module inside the container, or a precomputed name set).
"""

from types import SimpleNamespace

from rlinf.envs.isaaclab.tasks.discovery.predicates import (
    AMBIENT_STATE_PREDICATES,
    MANIPULATION_GOAL_PREDICATES,
    build_predicate_menu,
    default_conditionals_py_path,
    list_predicate_names,
    validate_phase_a,
)

KNOWN_OBJECTS = ["latteartcup", "coke", "cuttingboard"]


def _valid_candidate(**overrides):
    candidate = {
        "id": "coke_on_cuttingboard",
        "predicate": "object_on_top",
        "predicate_args": {"object": "coke", "reference_object": "cuttingboard"},
        "instruction": "Pick up the coke can and place it on the cutting board",
        "objects_involved": ["coke", "cuttingboard"],
    }
    candidate.update(overrides)
    return candidate


def test_default_conditionals_py_path_resolves_in_this_checkout():
    # This checkout does have RoboLab alongside RLinf -- confirm the path-discovery logic
    # actually finds the real file (a real regression check, not just "doesn't crash").
    path = default_conditionals_py_path()
    assert path is not None
    assert path.name == "conditionals.py"
    assert path.is_file()


def test_list_predicate_names_extracted_from_real_source_via_ast():
    names = list_predicate_names()
    # A handful of predicates confirmed present by grepping conditionals.py directly.
    for expected in ("object_on_top", "object_grabbed", "object_in_container", "stacked"):
        assert expected in names
    # PLAN.md says "~35 atomic predicates" -- confirm we're in that ballpark, not e.g. 3 or 300
    # (would indicate the AST walk is badly broken).
    assert 25 <= len(names) <= 45


def test_build_predicate_menu_uses_curated_markdown_by_default():
    menu = build_predicate_menu()
    assert "object_on_top" in menu
    # The curated conditionals.md has "use when" framing; a bare AST fallback wouldn't.
    assert "Use when" in menu or "use when" in menu.lower()


def test_build_predicate_menu_falls_back_to_ast_when_md_missing(tmp_path):
    missing_md = tmp_path / "does_not_exist.md"
    menu = build_predicate_menu(conditionals_md_path=missing_md)
    assert "object_on_top" in menu


# ---------------------------------------------------------------------------
# validate_phase_a
# ---------------------------------------------------------------------------


def test_validate_phase_a_accepts_well_formed_candidate_against_name_set():
    known_predicates = {"object_on_top", "object_grabbed"}
    accepted = validate_phase_a(
        [_valid_candidate()], known_objects=KNOWN_OBJECTS, predicate_module=known_predicates
    )
    assert len(accepted) == 1
    assert accepted[0]["id"] == "coke_on_cuttingboard"


def test_validate_phase_a_accepts_well_formed_candidate_against_fake_module():
    fake_module = SimpleNamespace(object_on_top=lambda *a, **k: None)
    accepted = validate_phase_a(
        [_valid_candidate()], known_objects=KNOWN_OBJECTS, predicate_module=fake_module
    )
    assert len(accepted) == 1


def test_validate_phase_a_accepts_against_real_conditionals_names():
    # Uses the AST-extracted name set as a stand-in for "the real conditionals module" --
    # exercises the actual predicate names this scene uses.
    real_names = list_predicate_names()
    accepted = validate_phase_a(
        [_valid_candidate()], known_objects=KNOWN_OBJECTS, predicate_module=real_names
    )
    assert len(accepted) == 1


def test_validate_phase_a_rejects_unknown_object_name():
    bad = _valid_candidate(objects_involved=["banana", "cuttingboard"])
    accepted = validate_phase_a(
        [bad], known_objects=KNOWN_OBJECTS, predicate_module={"object_on_top"}
    )
    assert accepted == []


def test_validate_phase_a_rejects_unknown_predicate_name():
    bad = _valid_candidate(predicate="teleport_object_by_magic")
    accepted = validate_phase_a(
        [bad], known_objects=KNOWN_OBJECTS, predicate_module={"object_on_top"}
    )
    assert accepted == []


def test_validate_phase_a_rejects_missing_required_field():
    bad = _valid_candidate()
    del bad["instruction"]
    accepted = validate_phase_a(
        [bad], known_objects=KNOWN_OBJECTS, predicate_module={"object_on_top"}
    )
    assert accepted == []


def test_validate_phase_a_mixed_batch_keeps_only_valid_ones():
    good = _valid_candidate(id="good_one")
    bad_object = _valid_candidate(id="bad_object_one", objects_involved=["nonexistent"])
    bad_predicate = _valid_candidate(id="bad_predicate_one", predicate="not_a_real_predicate")
    accepted = validate_phase_a(
        [good, bad_object, bad_predicate],
        known_objects=KNOWN_OBJECTS,
        predicate_module={"object_on_top"},
    )
    assert [c["id"] for c in accepted] == ["good_one"]


def test_validate_phase_a_on_reject_callback_receives_reason():
    bad = _valid_candidate(objects_involved=["banana"])
    rejections = []
    validate_phase_a(
        [bad],
        known_objects=KNOWN_OBJECTS,
        predicate_module={"object_on_top"},
        on_reject=lambda candidate, reason: rejections.append((candidate["id"], reason)),
    )
    assert len(rejections) == 1
    assert "banana" in rejections[0][1]


def test_validate_phase_a_permissive_when_predicate_module_none():
    # predicate_module=None means "no predicate validation possible" -- object-name checks
    # still apply.
    accepted = validate_phase_a(
        [_valid_candidate(predicate="anything_goes")],
        known_objects=KNOWN_OBJECTS,
        predicate_module=None,
    )
    assert len(accepted) == 1


# ---------------------------------------------------------------------------
# validate_phase_a -- hard ambient-predicate rejection (structural fix, round 2)
# ---------------------------------------------------------------------------


def test_validate_phase_a_hard_rejects_the_real_leaked_ambient_candidate():
    """Regression test for the exact candidate that leaked through prompt-only guidance on a
    real discover_tree() run: coke_next_to_ceramic_mug (object_next_to). Must now be rejected
    unconditionally, not just discouraged by wording.
    """
    leaked_candidate = {
        "id": "coke_next_to_ceramic_mug",
        "predicate": "object_next_to",
        "predicate_args": {"object": "coke", "reference_object": "ceramic_mug", "dist": 0.1},
        "instruction": "Move the coke can to be next to the ceramic mug.",
        "objects_involved": ["coke", "ceramic_mug"],
    }
    accepted = validate_phase_a(
        [leaked_candidate],
        known_objects=["ceramic_mug", "coke", "cutting_board_a"],
        predicate_module={"object_next_to"},  # even though the predicate name IS known/real
    )
    assert accepted == []


def test_validate_phase_a_ambient_rejection_gives_a_clear_reason():
    bad = _valid_candidate(predicate="object_upright", objects_involved=["cuttingboard"])
    rejections = []
    validate_phase_a(
        [bad],
        known_objects=KNOWN_OBJECTS,
        predicate_module={"object_upright"},
        on_reject=lambda candidate, reason: rejections.append(reason),
    )
    assert len(rejections) == 1
    assert "ambient" in rejections[0].lower()
    assert "object_upright" in rejections[0]


def test_validate_phase_a_accepts_manipulation_goal_predicates_by_default():
    for predicate in MANIPULATION_GOAL_PREDICATES:
        candidate = _valid_candidate(id=f"cand_{predicate}", predicate=predicate)
        accepted = validate_phase_a(
            [candidate], known_objects=KNOWN_OBJECTS, predicate_module={predicate}
        )
        assert accepted == [candidate], f"{predicate} should not be rejected"


def test_validate_phase_a_rejects_every_ambient_predicate_by_default():
    for predicate in AMBIENT_STATE_PREDICATES:
        candidate = _valid_candidate(id=f"cand_{predicate}", predicate=predicate)
        accepted = validate_phase_a(
            [candidate], known_objects=KNOWN_OBJECTS, predicate_module={predicate}
        )
        assert accepted == [], f"{predicate} should be hard-rejected by default"


def test_validate_phase_a_reject_ambient_predicates_flag_can_be_disabled():
    # Escape hatch exists (e.g. for a caller with a deliberately different policy) but defaults
    # to the strict/safe behavior.
    bad = _valid_candidate(predicate="object_next_to")
    accepted = validate_phase_a(
        [bad],
        known_objects=KNOWN_OBJECTS,
        predicate_module={"object_next_to"},
        reject_ambient_predicates=False,
    )
    assert accepted == [bad]


# ---------------------------------------------------------------------------
# validate_phase_a -- stable_base_objects (opt-in physical-plausibility filter, round 2)
# ---------------------------------------------------------------------------

SCENE_STABLE_BASES = {"cutting_board_a"}


def _on_top_candidate(object_, reference_object, id="cand"):
    return {
        "id": id,
        "predicate": "object_on_top",
        "predicate_args": {"object": object_, "reference_object": reference_object},
        "instruction": f"place {object_} on {reference_object}",
        "objects_involved": [object_, reference_object],
    }


def _stacked_candidate(objects, order="bottom_to_top", id="cand"):
    return {
        "id": id,
        "predicate": "stacked",
        "predicate_args": {"objects": objects, "order": order},
        "instruction": f"stack {objects}",
        "objects_involved": objects,
    }


def test_validate_phase_a_stable_base_objects_disabled_by_default():
    # The two real bad examples from the coordinator's round-2 report must NOT be rejected
    # unless a caller explicitly opts in via stable_base_objects -- disabled is the default.
    board_on_mug = _on_top_candidate("cutting_board_a", "ceramic_mug")
    accepted = validate_phase_a(
        [board_on_mug], known_objects=["ceramic_mug", "cutting_board_a"],
        predicate_module={"object_on_top"},
    )
    assert accepted == [board_on_mug]


def test_validate_phase_a_stable_base_objects_rejects_mug_as_base():
    # place_cutting_board_on_ceramic_mug: reference_object=ceramic_mug (the base) -- not stable.
    board_on_mug = _on_top_candidate("cutting_board_a", "ceramic_mug")
    accepted = validate_phase_a(
        [board_on_mug],
        known_objects=["ceramic_mug", "cutting_board_a"],
        predicate_module={"object_on_top"},
        stable_base_objects=SCENE_STABLE_BASES,
    )
    assert accepted == []


def test_validate_phase_a_stable_base_objects_rejects_coke_as_stacked_base():
    # stack_coke_ceramic_mug_cutting_board: objects=[coke, ceramic_mug, cutting_board_a],
    # bottom_to_top -- coke and ceramic_mug are both non-topmost (acting as bases).
    candidate = _stacked_candidate(["coke", "ceramic_mug", "cutting_board_a"])
    accepted = validate_phase_a(
        [candidate],
        known_objects=["coke", "ceramic_mug", "cutting_board_a"],
        predicate_module={"stacked"},
        stable_base_objects=SCENE_STABLE_BASES,
    )
    assert accepted == []


def test_validate_phase_a_stable_base_objects_rejects_mug_as_stacked_base():
    # stack_coke_on_ceramic_mug: objects=[ceramic_mug, coke], bottom_to_top -- mug is the base.
    candidate = _stacked_candidate(["ceramic_mug", "coke"])
    accepted = validate_phase_a(
        [candidate],
        known_objects=["ceramic_mug", "coke"],
        predicate_module={"stacked"},
        stable_base_objects=SCENE_STABLE_BASES,
    )
    assert accepted == []


def test_validate_phase_a_stable_base_objects_accepts_board_as_base():
    mug_on_board = _on_top_candidate("ceramic_mug", "cutting_board_a")
    coke_on_board = _on_top_candidate("coke", "cutting_board_a", id="coke_cand")
    accepted = validate_phase_a(
        [mug_on_board, coke_on_board],
        known_objects=["ceramic_mug", "coke", "cutting_board_a"],
        predicate_module={"object_on_top"},
        stable_base_objects=SCENE_STABLE_BASES,
    )
    assert accepted == [mug_on_board, coke_on_board]


def test_validate_phase_a_stable_base_objects_accepts_stack_with_board_as_only_base():
    # Stacking coke directly on the board (2-element stack, board as sole base) is fine.
    candidate = _stacked_candidate(["cutting_board_a", "coke"])
    accepted = validate_phase_a(
        [candidate],
        known_objects=["coke", "cutting_board_a"],
        predicate_module={"stacked"},
        stable_base_objects=SCENE_STABLE_BASES,
    )
    assert accepted == [candidate]


def test_validate_phase_a_stable_base_objects_permissive_when_stack_order_unspecified():
    # order != "bottom_to_top" -- can't confidently identify which element is the base, so this
    # deliberately does NOT enforce the filter rather than risk wrongly rejecting a valid
    # unordered stack.
    candidate = _stacked_candidate(["ceramic_mug", "coke"], order=None)
    accepted = validate_phase_a(
        [candidate],
        known_objects=["ceramic_mug", "coke"],
        predicate_module={"stacked"},
        stable_base_objects=SCENE_STABLE_BASES,
    )
    assert accepted == [candidate]


def test_validate_phase_a_stable_base_objects_does_not_affect_other_predicates():
    # A non-object_on_top/stacked predicate is untouched by this filter even if its objects
    # aren't in stable_base_objects.
    candidate = _valid_candidate(predicate="object_in_container", objects_involved=["coke", "cuttingboard"])
    accepted = validate_phase_a(
        [candidate],
        known_objects=KNOWN_OBJECTS,
        predicate_module={"object_in_container"},
        stable_base_objects=SCENE_STABLE_BASES,
    )
    assert accepted == [candidate]

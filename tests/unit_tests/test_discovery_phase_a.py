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

"""Unit tests for rlinf.envs.isaaclab.tasks.discovery.phase_a.

No real network calls: `call_vlm_phase_a` is exercised with an injected `client_fn`, never the
default `_raw_call`.
"""

import json

import pytest

from rlinf.envs.isaaclab.tasks.discovery.phase_a import (
    AMBIENT_STATE_PREDICATES,
    MANIPULATION_GOAL_PREDICATES,
    build_phase_a_prompt,
    call_vlm_phase_a,
)
from rlinf.envs.isaaclab.tasks.discovery.vlm_client import VLMCallError


def test_build_phase_a_prompt_includes_objects_and_satisfied_set():
    prompt = build_phase_a_prompt(
        object_list=["latteartcup", "coke", "cuttingboard"],
        satisfied_so_far=["latteartcup_on_cuttingboard"],
        predicate_menu="- object_on_top(...)",
    )
    assert "latteartcup" in prompt
    assert "['latteartcup_on_cuttingboard']" in prompt
    assert "object_on_top" in prompt
    # No starting_state_summary and no scene_screenshot given -- must fall back to the
    # explicit "no real data yet" note, not silently omit the section.
    assert "no real starting-state data available yet" in prompt


def test_build_phase_a_prompt_uses_real_starting_state_summary_when_given():
    real_summary = (
        "5 preset starting layouts exist in total; showing 5 representative samples:\n"
        "  sample #0:\n    coke: pos=(0.343, -0.391, 0.126) quat_wxyz=(0.230, 0.000, 0.000, 0.973)"
    )
    prompt = build_phase_a_prompt(
        object_list=["coke", "cuttingboard"],
        satisfied_so_far=[],
        starting_state_summary=real_summary,
        predicate_menu="menu",
    )
    # The real numeric data must appear verbatim -- not paraphrased, not dropped in favor of
    # the "no data available" fallback text.
    assert "pos=(0.343, -0.391, 0.126)" in prompt
    assert "no real starting-state data available yet" not in prompt


def test_build_phase_a_prompt_screenshot_is_only_a_fallback_when_no_real_data():
    prompt = build_phase_a_prompt(
        object_list=["coke"],
        satisfied_so_far=[],
        scene_screenshot="ZmFrZWJhc2U2NA==",
        predicate_menu="menu",
    )
    assert "an image is attached instead" in prompt

    # But real numeric data, when present, wins over the screenshot fallback text.
    prompt_with_both = build_phase_a_prompt(
        object_list=["coke"],
        satisfied_so_far=[],
        starting_state_summary="real data here",
        scene_screenshot="ZmFrZWJhc2U2NA==",
        predicate_menu="menu",
    )
    assert "real data here" in prompt_with_both
    assert "an image is attached instead" not in prompt_with_both


def test_build_phase_a_prompt_includes_anti_pattern_criteria_generically():
    """Round 3 fix: criteria (a)/(c)/(d) must describe their failure class abstractly (a "BAD
    PATTERN"), not name this scene's specific historical bad-example ids/objects -- see the
    dedicated scene-agnostic test below for the stronger, cross-scene version of this check.
    """
    prompt = build_phase_a_prompt(
        object_list=["ceramic_mug", "coke", "cutting_board_a"],
        satisfied_so_far=[],
        predicate_menu="menu",
    )
    # (a) not trivially satisfiable
    assert "TRIVIALLY" in prompt or "trivially" in prompt.lower()
    assert "BAD PATTERN" in prompt
    # (c) concrete/unambiguous
    assert "UNAMBIGUOUS" in prompt or "unambiguous" in prompt.lower()
    # (d) goal reached by manipulation, not a passive check
    assert "passive" in prompt.lower()
    # explicit creative/diverse instruction, so the fix doesn't collapse back to one template
    assert "CREATIVE" in prompt or "creative" in prompt.lower()
    assert "diverse" in prompt.lower() or "DIVERSE" in prompt


def test_build_phase_a_prompt_is_scene_agnostic_no_leaked_mug_coke_specifics():
    """Round 3 fix, the core regression test: build the prompt for a COMPLETELY DIFFERENT,
    made-up scene and confirm none of the mug/coke scene's object names or historical
    bad-example task ids leak into the criteria text. Per PLAN.md section 0.1, this discovery
    mechanism must be reusable, unmodified, for any future scene.
    """
    prompt = build_phase_a_prompt(
        object_list=["widget_a", "tray_b", "gizmo_c"],
        satisfied_so_far=[],
        predicate_menu="menu",
    )
    leaked_terms = [
        "ceramic_mug",
        "cutting_board_a",
        "cuttingboard",
        "latteartcup",
        # coke deliberately excluded -- "coke" is also an English word fragment risk-free to
        # check, but "coke_" (the id-prefix form) is the meaningful leak signal.
        "coke_",
        "ceramic_mug_right_of_coke",
        "check_ceramic_mug_upright",
        "align_objects_on_cuttingboard",
        "place_cutting_board_on_ceramic_mug",
        "stack_coke_ceramic_mug_cutting_board",
        "stack_coke_on_ceramic_mug",
    ]
    for term in leaked_terms:
        assert term not in prompt, f"scene-specific term {term!r} leaked into a generic prompt"

    # The made-up scene's own objects appear (proves object substitution actually works, this
    # isn't just an empty/broken template).
    assert "widget_a" in prompt
    assert "tray_b" in prompt
    assert "gizmo_c" in prompt

    # The criteria/guidance content itself is still present -- genericized, not deleted.
    assert "BAD PATTERN" in prompt
    assert "PHYSICALLY STABLE" in prompt
    assert "ALL FIVE" in prompt


def test_build_phase_a_prompt_includes_predicate_classification_guidance():
    prompt = build_phase_a_prompt(
        object_list=["coke"], satisfied_so_far=[], predicate_menu="menu"
    )
    assert "GOAL-REACHED-BY-MANIPULATION" in prompt
    assert "AMBIENT" in prompt
    for name in MANIPULATION_GOAL_PREDICATES:
        assert name in prompt
    for name in AMBIENT_STATE_PREDICATES:
        assert name in prompt
    # A couple of specific, load-bearing classification calls the coordinator asked for
    # explicitly: object_on_top is a real manipulation goal; object_upright/object_right_of
    # are ambient.
    assert "object_on_top" in MANIPULATION_GOAL_PREDICATES
    assert "object_upright" in AMBIENT_STATE_PREDICATES
    assert "object_right_of" in AMBIENT_STATE_PREDICATES


def test_build_phase_a_prompt_ambient_guidance_says_hard_reject_not_soft_discouragement():
    """Round 2 fix: the prompt must be honest that ambient predicates are now unconditionally
    and automatically rejected downstream (validate_phase_a), not just "discouraged unless
    justified" -- the old wording that didn't stop coke_next_to_ceramic_mug from leaking through.
    """
    prompt = build_phase_a_prompt(object_list=["coke"], satisfied_so_far=[], predicate_menu="menu")
    assert "automatically and unconditionally rejected" in prompt
    # The old soft-allow phrasing must be gone.
    assert "you must explicitly justify" not in prompt


def test_build_phase_a_prompt_includes_physical_stability_criterion_generically():
    """Round 3 fix: physical-stability guidance for object_on_top/stacked must reason about
    "wide/flat/low-profile" vs. "narrow/tall/top-heavy" shape properties in the abstract, deriving
    which named objects are which from SCENE OBJECTS at call time -- not naming this scene's
    specific objects as hardcoded good/bad bases inside the guidance/criteria text itself.
    """
    prompt = build_phase_a_prompt(
        object_list=["ceramic_mug", "coke", "cutting_board_a"],
        satisfied_so_far=[],
        predicate_menu="menu",
    )
    # Criterion (e) exists and the task now requires all five criteria.
    assert "ALL FIVE" in prompt
    assert "PHYSICALLY STABLE" in prompt

    # Generic shape-based reasoning language is present...
    assert "wide" in prompt.lower() and "flat" in prompt.lower()
    assert "narrow" in prompt.lower()
    assert "top-heavy" in prompt.lower() or "tippy" in prompt.lower()

    # ...but the SPECIFIC named objects are not called out as inherently good/bad bases in the
    # guidance text -- the model is pointed at "SCENE OBJECTS above" instead of a memorized list.
    assert "not from a fixed list" in prompt or "not a fixed rule" in prompt.lower()

    # Doesn't collapse back to "never place anything on anything" -- placing on a genuinely
    # wide/flat/stable object is still explicitly called out as fine, in the abstract.
    assert "fine" in prompt.lower()


def test_call_vlm_phase_a_threads_real_starting_state_summary_into_the_prompt():
    seen_prompts = []

    def fake_client(prompt, image_b64=None, model=None):
        seen_prompts.append(prompt)
        return "[]"

    call_vlm_phase_a(
        object_list=["coke"],
        satisfied_so_far=[],
        starting_state_summary="REAL_STATE_MARKER pos=(1,2,3)",
        predicate_menu="menu",
        client_fn=fake_client,
    )
    assert len(seen_prompts) == 1
    assert "REAL_STATE_MARKER pos=(1,2,3)" in seen_prompts[0]


def test_call_vlm_phase_a_returns_parsed_json_list():
    candidates = [
        {
            "id": "coke_on_cuttingboard",
            "predicate": "object_on_top",
            "predicate_args": {"object": "coke", "reference_object": "cuttingboard"},
            "instruction": "Pick up the coke can and place it on the cutting board",
            "objects_involved": ["coke", "cuttingboard"],
        }
    ]

    def fake_client(prompt, image_b64=None, model=None):
        assert "coke" in prompt  # object list did make it into the prompt
        return json.dumps(candidates)

    result = call_vlm_phase_a(
        object_list=["coke", "cuttingboard"],
        satisfied_so_far=[],
        predicate_menu="menu",
        client_fn=fake_client,
    )
    assert result == candidates


def test_call_vlm_phase_a_wraps_bare_object_response_as_single_item_list():
    """Regression test for a real failure hit on a live gpt-4o call: when there's exactly one
    candidate, the model sometimes returns a bare JSON object instead of a one-element JSON
    list, despite the prompt saying "a JSON list". This must be normalized, not raise.
    """
    bare_candidate = {
        "id": "coke_next_to_ceramic_mug",
        "predicate": "object_next_to",
        "predicate_args": {"object": "coke", "reference_object": "ceramic_mug", "dist": 0.05},
        "instruction": "Move the coke to be next to the ceramic mug.",
        "objects_involved": ["coke", "ceramic_mug"],
    }
    result = call_vlm_phase_a(
        object_list=["coke", "ceramic_mug"],
        satisfied_so_far=["something_already_done"],
        predicate_menu="menu",
        client_fn=lambda prompt, image_b64=None, model=None: json.dumps(bare_candidate),
    )
    assert result == [bare_candidate]


def test_call_vlm_phase_a_empty_list_is_valid():
    result = call_vlm_phase_a(
        object_list=["coke"],
        satisfied_so_far=["coke_on_cuttingboard"],
        predicate_menu="menu",
        client_fn=lambda prompt, image_b64=None, model=None: "[]",
    )
    assert result == []


def test_call_vlm_phase_a_rejects_non_list_non_dict_response():
    # A bare dict is now normalized (see the wrap test above, a real gpt-4o behavior); anything
    # that's neither a list nor a dict (e.g. a bare string/number) is still a hard error --
    # there's no reasonable single-candidate interpretation for that shape.
    with pytest.raises(VLMCallError):
        call_vlm_phase_a(
            object_list=["coke"],
            satisfied_so_far=[],
            predicate_menu="menu",
            client_fn=lambda prompt, image_b64=None, model=None: json.dumps("not a list or dict"),
        )


def test_call_vlm_phase_a_rejects_non_json_response():
    with pytest.raises(VLMCallError):
        call_vlm_phase_a(
            object_list=["coke"],
            satisfied_so_far=[],
            predicate_menu="menu",
            client_fn=lambda prompt, image_b64=None, model=None: "not json at all",
        )


def test_call_vlm_phase_a_default_client_fn_is_the_real_raw_call_but_unused_here():
    """Confirms the wiring (default `client_fn` really is `vlm_client._raw_call`, i.e. production
    code gets a real network call unless it overrides it) without ever invoking that default --
    every other test in this file passes an explicit fake `client_fn`, so the real network path
    is never exercised.
    """
    import inspect

    from rlinf.envs.isaaclab.tasks.discovery.vlm_client import _raw_call

    sig = inspect.signature(call_vlm_phase_a)
    assert sig.parameters["client_fn"].default is _raw_call

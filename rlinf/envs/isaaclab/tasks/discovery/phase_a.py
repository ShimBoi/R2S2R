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

"""Phase A: offline discovery -- propose the next edge(s) out of a given node.

Prompt builder + VLM call wrapper, per PLAN.md section 3. Stateful: takes the current node
(which subtasks are already satisfied) and asks what's plausible *from there* -- zero, one, or
several candidates.

REVISED TWICE (see coordinator follow-ups):

Round 1 -- a real ``discover_tree()`` run against the mug/coke scene (``max_depth=4``, the
original "propose ALL the plausible next atomic subtasks" prompt) produced 52 edges across 23
branching nodes -- exploding until the depth cap stopped it, not because the model ran out of
genuine ideas. Concretely it proposed things like ``ceramic_mug_right_of_coke`` (a spatial
relation that can already hold from the random initial scatter -- no manipulation required),
``check_ceramic_mug_upright`` (a passive orientation check, not a goal reached by doing
something), and ``align_objects_on_cutting_board`` (no concrete predicate/args -- not actually
specified). Fixed with explicit criteria (a)-(d) below, real numeric starting-state grounding
(``starting_state_summary``, see ``starting_states.py``) replacing the screenshot mechanism, and
a ``MANIPULATION_GOAL_PREDICATES``/``AMBIENT_STATE_PREDICATES`` classification given to the model
as prompt guidance.

Round 2 -- a follow-up real run (9 edges, much better) still let through
``coke_next_to_ceramic_mug`` (``object_next_to``, an ambient predicate -- prompt guidance alone
wasn't enough) and several physically-dubious base/stacking proposals:
``place_cutting_board_on_ceramic_mug`` (a flat board balanced on a mug's narrow rim) and
``stack_coke_ceramic_mug_cutting_board`` / ``stack_coke_on_ceramic_mug`` (something balanced on
top of a standing coke can, or on top of a mug). Two fixes:
  1. The manipulation-goal/ambient classification is now ALSO a hard, deterministic filter in
     ``predicates.validate_phase_a`` (``reject_ambient_predicates=True``, the default) -- not
     just prompt wording the model can ignore. See that function's docstring.
  2. A new criterion (e), PHYSICAL STABILITY, reasoned explicitly about which named objects make
     realistic bases for ``object_on_top``/``stacked``, using the two bad examples above by name.
     A matching *structural* check also exists now (``validate_phase_a``'s
     ``stable_base_objects`` parameter), opt-in rather than a blanket default -- see
     ``orchestrator.py``'s module docstring for the judgment call on why. A follow-up real run
     with ``stable_base_objects`` deliberately left off confirmed prompt-only guidance was NOT
     reliable on its own -- 2 of 13 proposals still used a small/narrow object as a stacking
     base -- so ``stable_base_objects`` is meant to be turned on for real usage.

Round 3 -- the coordinator/user flagged that criteria (a)-(e) as written in round 2 baked this
specific scene's object names and literal bad-example task ids directly into
``PHASE_A_PROMPT_TEMPLATE`` (e.g. naming ``ceramic_mug``/``coke``/``cutting_board_a`` and ids
like ``place_cutting_board_on_ceramic_mug`` in the criteria text itself). That's a real problem:
per PLAN.md section 0.1, this discovery mechanism is meant to be reused, unmodified, for future
scenes with entirely different objects -- a template hardcoded to this scene would carry
nonsensical, irrelevant examples into a totally different object set. Rewritten below so every
criterion (including physical stability) describes the *failure class* abstractly and reasons
from ``{object_list}``/``{satisfied_so_far}`` at call time, with zero literal object names or
task ids anywhere in ``PHASE_A_PROMPT_TEMPLATE`` or the guidance-block builders. Re-verified
against the real mug/coke scene afterwards (the only real testbed available) to confirm the
generic rewrite performs at least as well as the scene-specific wording it replaced -- see the
coordinator report for that run's results.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from .predicates import AMBIENT_STATE_PREDICATES, MANIPULATION_GOAL_PREDICATES, build_predicate_menu
from .vlm_client import VLMCallError, _raw_call, call_vlm

# Re-exported from predicates.py (single source of truth -- see that module's comment) so
# existing imports of `from phase_a import MANIPULATION_GOAL_PREDICATES` keep working, and so
# the prompt guidance built here can never list a different set than what validate_phase_a
# actually enforces.
__all__ = [
    "MANIPULATION_GOAL_PREDICATES",
    "AMBIENT_STATE_PREDICATES",
    "PHASE_A_PROMPT_TEMPLATE",
    "build_phase_a_prompt",
    "call_vlm_phase_a",
]


def _predicate_guidance_block() -> str:
    manip = ", ".join(MANIPULATION_GOAL_PREDICATES)
    ambient = ", ".join(AMBIENT_STATE_PREDICATES)
    return (
        "GOAL-REACHED-BY-MANIPULATION predicates (use these as a subtask's sole success "
        "criterion -- each describes a state that normally requires actually picking up/moving/\n"
        f"placing something to reach):\n  {manip}\n\n"
        "AMBIENT / PASSIVE / TRANSIENT predicates (DO NOT propose any of these as a subtask's "
        "sole success criterion -- each describes a relative position, an orientation, or a "
        "momentary gripper state that can already hold from the random initial scatter, or "
        f"isn't a completed goal at all):\n  {ambient}\n"
        "Candidates using one of these ambient predicates are automatically and unconditionally "
        "rejected by the validator downstream, regardless of how you word the instruction or "
        "justify it -- don't spend a candidate slot on one, propose a manipulation-goal "
        "predicate instead."
    )


def _physical_stability_guidance_block() -> str:
    return (
        "For OBJECT_ON_TOP and STACKED proposals specifically, reason about whether the BASE "
        "object (object_on_top's reference_object; every object except the topmost one in "
        "stacked's bottom-to-top order) is a realistic, physically stable resting surface for "
        "what would be placed on it. Do this concretely, from the real-world shape of the "
        "SPECIFIC objects named in SCENE OBJECTS above -- not from a fixed list of good/bad "
        "bases, since the object set changes from scene to scene. As a general principle: a "
        "wide, flat, low-profile object usually makes a stable base for something resting on "
        "top of it; a narrow, tall, top-heavy, or curved/rounded-top object usually does NOT, "
        "regardless of whether it happens to have a flat footprint of its own or whether the "
        "success-check geometry would technically register as satisfied at some instant. Apply "
        "this reasoning per-object to whatever objects actually exist in this scene."
    )


PHASE_A_PROMPT_TEMPLATE = """You are proposing the next training subtask(s) for a robot manipulation curriculum, built
incrementally as a decision tree over "which subtasks are already satisfied."

SCENE OBJECTS (exact names, use verbatim):
{object_list}

ALREADY-SATISFIED SUBTASKS AT THIS POINT IN THE TREE:
{satisfied_so_far}
  # e.g. [] for the root, or a list of previously-discovered subtask ids once some are done

STARTING STATE AT THIS POINT IN THE TREE (real, numeric data -- ground your proposals in this,
not a guess about "typical" layouts):
{starting_state_summary}

AVAILABLE SUCCESS-CHECK FUNCTIONS (use only these, never invent new ones):
{predicate_menu}

{predicate_guidance}

{physical_stability_guidance}

TASK
Propose the next atomic subtask(s) from this point in the tree. There may be more than one
reasonable option (e.g. at the root, several different objects/goals could plausibly go first),
exactly one (e.g. once something is already satisfied, there's often only one sensible thing
left), or none (return an empty list if nothing further is useful given what's already
satisfied). Keep proposing CREATIVE and DIVERSE manipulation tasks across calls -- do not
collapse to repeatedly proposing the same single goal/target for every object every time; the
goal is genuine variety across different objects, relations, and predicates, filtered for
validity, not narrowness.

Every candidate you propose MUST satisfy ALL FIVE of these, in order:

(a) NOT TRIVIALLY/RANDOMLY SATISFIABLE. Look at the STARTING STATE data above. If the proposed
    predicate would already evaluate true for a meaningful fraction of those starting layouts
    without the policy doing anything, it's not a real task -- reject it.
    BAD PATTERN (illustrative, not a specific example from this scene): a pure relative-position
    check between two objects (e.g. one being to a side of, in front of/behind, or near another)
    that could easily already hold by chance from an unbiased random scatter. If achieving the
    proposed configuration requires no manipulation in a large fraction of the starting layouts
    shown above, it fails this bar outright.

(b) ACHIEVABLE given the current layout/state. The manipulation implied must be physically
    sensible from a real starting position in the data above (e.g. don't propose placing an
    object that's already been placed and locked in by an earlier subtask in this same node).

(c) CONCRETE AND UNAMBIGUOUS -- fully specified by predicate + predicate_args, with no vague
    relational language left for someone else to interpret later.
    BAD PATTERN: a goal described only in vague relational language (e.g. objects being
    "arranged," "organized," or "aligned" with no concrete reference point or tolerance) that
    isn't one of the predicates above and isn't reducible to concrete predicate_args -- there's
    no way to check success programmatically from a description like that alone. Every candidate
    must map cleanly onto exactly one function from AVAILABLE SUCCESS-CHECK FUNCTIONS with fully
    filled-in predicate_args, not a description a human would still need to operationalize.

(d) A GOAL REACHED BY DOING SOMETHING, not a passive state check. See the GOAL-REACHED-BY-
    MANIPULATION vs. AMBIENT/PASSIVE/TRANSIENT predicate guidance above -- ambient predicates are
    hard-rejected downstream regardless of wording, so don't propose them at all.
    BAD PATTERN: a property check on a single object's own current state (its orientation,
    whether it's currently held, whether it has stopped moving) proposed as if it were a
    manipulation goal by itself. Objects are typically already in whatever their "normal" spawn
    state is, so checking that state alone usually isn't a goal that required picking anything up
    or moving it anywhere -- it's a passive property check, not a subtask.

(e) PHYSICALLY STABLE, for object_on_top/stacked proposals -- see the physical-stability
    guidance above. "Difficult but still physically possible" is fine; "no realistic stable rest
    state given these objects' real shapes" is not.
    BAD PATTERN: proposing that something rest balanced on top of an object that is narrow,
    tall, top-heavy, or has a curved/rounded top -- these usually have no realistic stable rest
    state for anything placed on them, regardless of what the geometry check would technically
    register at a single instant. Placing something directly on an object that's genuinely wide,
    flat, and low-profile is fine -- it's specifically a small/narrow/tippy object acting as the
    BASE for something else that's usually implausible, not the general idea of sequencing
    placements or stacking.

Do not propose a subtask that's already satisfied. Do not propose anything requiring an object not
in the scene or a predicate not in the list above.

OUTPUT: a JSON list (possibly empty). Each entry:
{{
  "id": "<snake_case_identifier, unique>",
  "predicate": "<exact function name from the list above>",
  "predicate_args": {{"<param_name>": "<exact object name or value>", ...}},
  "instruction": "<natural-language instruction>",
  "objects_involved": ["<object names used>"]
}}
Do not include "precondition" or "produces_node" -- those are filled in by the calling script from
ALREADY-SATISFIED, not by you, since they need to be exact copies, not paraphrased.
"""


def build_phase_a_prompt(
    object_list: Iterable[str],
    satisfied_so_far: Iterable[str],
    starting_state_summary: Optional[str] = None,
    predicate_menu: Optional[str] = None,
    *,
    scene_screenshot: Optional[str] = None,
) -> str:
    """Render the Phase A prompt.

    ``starting_state_summary`` is real, numeric starting-state text (see ``starting_states.py``)
    -- the primary grounding mechanism now. ``scene_screenshot`` (a base64 PNG) is a secondary,
    legacy fallback: only used to fill the starting-state section when no real numeric summary
    is available at all (``render_current_node`` is still a stub returning ``None``, so in
    practice this path is currently inert; kept only so a future image-grounding source has
    somewhere to plug in without another prompt rewrite).
    """
    if predicate_menu is None:
        predicate_menu = build_predicate_menu()

    if starting_state_summary:
        state_text = starting_state_summary
    elif scene_screenshot:
        state_text = "(no real numeric state data -- an image is attached instead)"
    else:
        state_text = (
            "(no real starting-state data available yet for this node -- reason structurally "
            "from the scene objects and ALREADY-SATISFIED set only, and be extra conservative "
            "about criterion (a): without real data you cannot confirm a proposal isn't "
            "trivially satisfiable, so prefer predicates/objects where that's obviously true "
            "regardless of exact starting positions, e.g. an object needs to move from one "
            "named region/container to another entirely different one.)"
        )

    return PHASE_A_PROMPT_TEMPLATE.format(
        object_list=list(object_list),
        satisfied_so_far=sorted(satisfied_so_far),
        starting_state_summary=state_text,
        predicate_menu=predicate_menu,
        predicate_guidance=_predicate_guidance_block(),
        physical_stability_guidance=_physical_stability_guidance_block(),
    )


def call_vlm_phase_a(
    object_list: Iterable[str],
    satisfied_so_far: Iterable[str],
    starting_state_summary: Optional[str] = None,
    predicate_menu: Optional[str] = None,
    *,
    scene_screenshot: Optional[str] = None,
    model: Optional[str] = None,
    client_fn: Callable[..., str] = _raw_call,
) -> list[dict[str, Any]]:
    """Call the VLM for Phase A and return its proposed candidates, UNVALIDATED.

    Always run the result through ``predicates.validate_phase_a`` before trusting it (that's
    what ``orchestrator.discover_tree``/``discover_and_train`` do) -- this function only builds
    the prompt, makes the call, and checks the response is at least shaped like a JSON list.
    """
    prompt = build_phase_a_prompt(
        object_list,
        satisfied_so_far,
        starting_state_summary,
        predicate_menu,
        scene_screenshot=scene_screenshot,
    )
    response = call_vlm(prompt, image_b64=scene_screenshot, model=model, client_fn=client_fn)
    parsed = response.parsed
    # Real-world robustness fix (hit on a live gpt-4o call, not hypothetical): when there's
    # exactly one candidate, models sometimes collapse "a JSON list with one entry" down to a
    # bare JSON object instead -- despite the prompt explicitly saying "a JSON list". A single
    # candidate dict has exactly the shape a Phase A entry should (id/predicate/... keys), so
    # this is an unambiguous, safe normalization, not a guess at unrelated malformed output.
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise VLMCallError(
            f"Phase A response must be a JSON list of candidates, got: {response.parsed!r}"
        )
    return parsed

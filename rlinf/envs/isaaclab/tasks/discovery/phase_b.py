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

"""Phase B: eval-time decomposition, precondition-aware. PLAN.md sections 5 and 5.1.

``resolve_plan`` is the important simplification from PLAN.md 5.1: the whole decomposition is
resolved ONCE, before the rollout starts, by walking the manifest from the root. The VLM is
called only at genuine branch points (a node with more than one valid edge); every other node
has exactly one valid edge and is taken mechanically. For the mug/coke scene this means exactly
one VLM call resolves an entire episode's plan. No VLM calls happen inside ``step()`` -- the
returned plan is a plain list the eval-time mixin (Agent A's territory) walks with coded
predicate checks only.
"""

from __future__ import annotations

from typing import Any, Callable, FrozenSet, Iterable, Optional

from .manifest import Manifest
from .vlm_client import VLMCallError, _raw_call, call_vlm

PHASE_B_PROMPT_TEMPLATE = """You are the high-level controller for a robot arm. A low-level policy executes exactly one
instruction at a time; you decide which instruction to give it next.

OVERARCHING TASK (from the human operator):
{overarching_task}

SUBTASKS ALREADY COMPLETED THIS EPISODE, IN ORDER:
{completed_so_far}
  # this defines which node of the trained tree you're currently at

SUBTASKS AVAILABLE FROM THIS EXACT POINT (precondition == completed_so_far exactly -- these are
the ONLY valid next choices; nothing else was ever trained from this specific state):
{available_edges}
  # e.g.:
  # - id: <some subtask id valid from exactly this point>
  #   instruction: "<the natural-language instruction for that subtask>"

CURRENT SCENE STATE (ground truth, computed programmatically -- trust this over anything you might
infer visually):
{state_description}

TASK
Decide the single next subtask to execute, chosen from SUBTASKS AVAILABLE FROM THIS EXACT POINT
only, to make progress on the overarching task.

Return ONLY JSON, exactly one of:
{{"status": "next", "subtask_id": "<id from the available list above>", "reasoning": "<one sentence>"}}
{{"status": "done", "reasoning": "<why the overarching task is now satisfied>"}}
{{"status": "unreachable", "reasoning": "<what's missing from the trained tree>"}}

Never return a subtask_id that isn't in the AVAILABLE list above -- not the broader set of
everything ever trained, only what's valid from exactly this point.
"""


def _format_available_edges(available_edges: Iterable[dict[str, Any]]) -> str:
    lines = []
    for e in available_edges:
        lines.append(f'  - id: {e["id"]}')
        lines.append(f'    instruction: "{e["instruction"]}"')
    return "\n".join(lines) if lines else "  (none)"


def build_phase_b_prompt(
    overarching_task: str,
    completed_so_far: Iterable[str],
    available_edges: Iterable[dict[str, Any]],
    state_description: str = "(not provided)",
) -> str:
    return PHASE_B_PROMPT_TEMPLATE.format(
        overarching_task=overarching_task,
        completed_so_far=sorted(completed_so_far),
        available_edges=_format_available_edges(available_edges),
        state_description=state_description,
    )


def call_vlm_phase_b(
    overarching_task: str,
    completed_so_far: Iterable[str],
    available_edges: Iterable[dict[str, Any]],
    state_description: str = "(not provided)",
    *,
    model: Optional[str] = None,
    client_fn: Callable[..., str] = _raw_call,
) -> dict[str, Any]:
    """Call the VLM for Phase B and return its parsed decision dict, structurally validated
    (has a recognized ``status``, and a ``subtask_id`` when ``status == "next"``) but NOT
    checked against ``available_edges`` -- that check happens in ``resolve_plan``, which is
    also where "the VLM invented an id that wasn't offered" would surface.
    """
    prompt = build_phase_b_prompt(
        overarching_task, completed_so_far, available_edges, state_description
    )
    response = call_vlm(prompt, model=model, client_fn=client_fn)
    parsed = response.parsed
    if not isinstance(parsed, dict) or "status" not in parsed:
        raise VLMCallError(f"Phase B response must be a JSON object with a status: {parsed!r}")
    if parsed["status"] == "next" and "subtask_id" not in parsed:
        raise VLMCallError(f'Phase B status "next" without a subtask_id: {parsed!r}')
    if parsed["status"] not in ("next", "done", "unreachable"):
        raise VLMCallError(f"Phase B returned an unrecognized status: {parsed['status']!r}")
    return parsed


def resolve_plan(
    overarching_task: str,
    manifest: Manifest,
    node: FrozenSet[str] = frozenset(),
    *,
    phase_b_caller: Callable[..., dict[str, Any]] = call_vlm_phase_b,
    state_description_fn: Optional[Callable[[FrozenSet[str]], str]] = None,
) -> list[dict[str, Any]]:
    """Walk the manifest from ``node`` (default: root), calling the VLM only at genuine branch
    points (>1 valid edge). Returns a plain ordered list of edge specs. No VLM calls happen
    after this returns -- the caller assigns the result to the eval-time mixin once, at reset.

    Exactly mirrors PLAN.md section 5.1's sketch, parameterized (manifest and the VLM caller are
    explicit arguments, not module globals) so it's directly unit-testable.
    """
    node = frozenset(node)
    plan: list[dict[str, Any]] = []
    while True:
        edges = manifest.edges_from(node)  # exact-match precondition lookup
        if not edges:
            break  # leaf -- nothing more to do
        elif len(edges) == 1:
            chosen = edges[0]  # no ambiguity, no VLM call
        else:
            state_description = state_description_fn(node) if state_description_fn else "(not provided)"
            response = phase_b_caller(
                overarching_task,
                completed_so_far=sorted(node),
                available_edges=edges,
                state_description=state_description,
            )
            if response["status"] != "next":
                break  # "done" or "unreachable"
            matches = [e for e in edges if e["id"] == response["subtask_id"]]
            if not matches:
                raise VLMCallError(
                    f"Phase B chose subtask_id {response['subtask_id']!r}, which isn't among "
                    f"the {[e['id'] for e in edges]} actually offered at node {sorted(node)!r}"
                )
            chosen = matches[0]
        plan.append(chosen)
        node = node | {chosen["id"]}
    return plan

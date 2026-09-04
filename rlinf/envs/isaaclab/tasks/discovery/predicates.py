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

"""Predicate menu extraction and Phase A candidate validation.

``conditionals.py`` (``RoboLab/robolab/core/task/conditionals.py``) is the ground truth for
which predicate names exist (~35 ``@atomic``/``@composite`` functions per PLAN.md). Importing
it for real pulls in ``isaaclab`` transitively (via ``robolab.core.world.world_state``), which
needs a GPU/container -- not something a plain ``pytest`` run should require. So predicate names
and signatures are extracted here by parsing the *source* with ``ast``, never by importing the
module. ``validate_phase_a`` is written to accept either that AST-derived name set OR the real,
live-imported ``conditionals`` module (production code inside the container can pass the actual
module, exactly as PLAN.md section 4's sketch does: ``predicate_module=conditionals``) -- both
paths go through the same ``hasattr``-style check.

A human-facing predicate menu (for the Phase A prompt, which wants "use when" / "key params"
framing, not just names) already exists, checked once, at
``RoboLab/skills/robolab-taskgen/references/conditionals.md`` -- PLAN.md's "extracted once from
conditionals.md" instruction turns out to already be done; ``build_predicate_menu`` prefers it
and only falls back to an AST-derived (name + docstring first line) listing if that file is ever
missing.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

_ATOMIC_DECORATOR_NAMES = {"atomic", "composite"}


def find_repo_root(start: Path) -> Optional[Path]:
    """Walk up from ``start`` looking for the directory that contains ``RoboLab/`` as a sibling.

    Avoids hardcoding a fixed number of ``.parent`` hops, which would silently break if this
    package ever moves. Returns ``None`` (rather than raising) if not found, since this module
    must stay importable even in a checkout that doesn't have RoboLab at all -- callers decide
    whether that's fatal. Public (not `_`-prefixed) because ``starting_states.py`` reuses it too
    -- one repo-root-finding implementation, not two copies that could drift.
    """
    for candidate in [start] + list(start.parents):
        if (candidate / "RoboLab").is_dir():
            return candidate
    return None


def default_conditionals_py_path() -> Optional[Path]:
    root = find_repo_root(Path(__file__).resolve())
    if root is None:
        return None
    path = root / "RoboLab" / "robolab" / "core" / "task" / "conditionals.py"
    return path if path.is_file() else None


def default_conditionals_md_path() -> Optional[Path]:
    root = find_repo_root(Path(__file__).resolve())
    if root is None:
        return None
    path = (
        root
        / "RoboLab"
        / "skills"
        / "robolab-taskgen"
        / "references"
        / "conditionals.md"
    )
    return path if path.is_file() else None


def list_predicate_signatures(
    conditionals_py_path: Optional[Union[str, Path]] = None,
) -> dict[str, dict[str, Any]]:
    """Parse ``conditionals.py`` with ``ast`` and return ``{name: {"args": [...], "doc": str}}``.

    Only top-level ``def``/``async def`` decorated with ``@atomic`` or ``@composite`` are
    included -- matches exactly what ``conditionals.py``'s own docstring says the module
    contains, and what a caller like ``getattr(conditionals, spec["predicate"])`` would
    actually be able to dispatch to at runtime.
    """
    path = Path(conditionals_py_path) if conditionals_py_path else default_conditionals_py_path()
    if path is None or not path.is_file():
        raise FileNotFoundError(
            "Could not locate conditionals.py "
            f"(looked for RoboLab/robolab/core/task/conditionals.py; got {path})"
        )

    tree = ast.parse(path.read_text(), filename=str(path))
    signatures: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorator_names = {
            d.id for d in node.decorator_list if isinstance(d, ast.Name)
        }
        if not decorator_names & _ATOMIC_DECORATOR_NAMES:
            continue
        args = [a.arg for a in node.args.args if a.arg not in ("env",)]
        doc = ast.get_docstring(node) or ""
        doc_first_line = doc.strip().splitlines()[0] if doc.strip() else ""
        signatures[node.name] = {"args": args, "doc": doc_first_line}
    return signatures


def list_predicate_names(
    conditionals_py_path: Optional[Union[str, Path]] = None,
) -> frozenset[str]:
    """Just the names, as a frozenset -- the membership set ``validate_phase_a`` checks against."""
    return frozenset(list_predicate_signatures(conditionals_py_path).keys())


def build_predicate_menu(
    *,
    prefer_md: bool = True,
    conditionals_md_path: Optional[Union[str, Path]] = None,
    conditionals_py_path: Optional[Union[str, Path]] = None,
) -> str:
    """Build the ``{predicate_menu}`` text for the Phase A prompt.

    Prefers the curated markdown reference (nicer "use when" framing for a VLM); falls back to
    an AST-derived plain listing (name + first docstring line + arg names) if that file is
    missing, so this doesn't hard-fail in a checkout without RoboLab's docs.
    """
    if prefer_md:
        md_path = (
            Path(conditionals_md_path) if conditionals_md_path else default_conditionals_md_path()
        )
        if md_path is not None and md_path.is_file():
            return md_path.read_text()

    signatures = list_predicate_signatures(conditionals_py_path)
    lines = []
    for name in sorted(signatures):
        sig = signatures[name]
        args_str = ", ".join(sig["args"])
        doc = f" -- {sig['doc']}" if sig["doc"] else ""
        lines.append(f"- {name}({args_str}){doc}")
    return "\n".join(lines)


# Classification of conditionals.py's predicates into "represents a goal actually reached by
# manipulating something" vs. "an ambient/passive property that's weak (or wrong) as a
# standalone task's *sole* success criterion". Single source of truth for both the Phase A
# prompt's guidance text (phase_a.py imports these rather than defining its own copy) and the
# hard structural filter in validate_phase_a below -- moved here (from phase_a.py) specifically
# so the prompt's claims about what's allowed and what actually gets enforced can never drift
# apart. Sourced from conditionals.md's own section groupings: "Containment & Placement" +
# "Stacking" + "Multi-Group" + object_picked_up are manipulation goals; "Spatial Relations"
# (pure relative position, no manipulation implied on its own) + object_upright/object_grabbed/
# object_dropped/objects_stationary + wrong_object_grabbed are ambient/transient/negative.
#
# Originally (first Phase A prompt revision) this classification was ONLY prompt guidance --
# "if you propose one of these, justify it". A real discover_tree() run still produced
# `coke_next_to_ceramic_mug` (object_next_to) despite that wording. Per the coordinator's
# follow-up, AMBIENT_STATE_PREDICATES is now also a hard rejection filter in validate_phase_a
# (reject_ambient_predicates=True by default) -- guaranteed, not hoped-for.
MANIPULATION_GOAL_PREDICATES = (
    "object_in_container",
    "object_on_top",
    "object_enclosed",
    "object_inside",
    "object_outside_of",
    "object_outside_of_and_on_surface",
    "object_picked_up",
    "object_at",
    "objects_in_line",
    "stacked",
    "object_groups_in_containers",
)

AMBIENT_STATE_PREDICATES = (
    "object_left_of",
    "object_right_of",
    "object_in_front_of",
    "object_behind",
    "object_above",
    "object_below",
    "object_below_top",
    "object_on_bottom",
    "object_on_center",
    "object_next_to",
    "object_between",
    "object_upright",
    "object_grabbed",
    "object_dropped",
    "objects_stationary",
    "wrong_object_grabbed",
)

# Predicates for which "reference_object" (object_on_top) / non-topmost "objects" entries
# (stacked, when order == "bottom_to_top") name the thing something else rests ON -- i.e. where
# physical-stability-of-the-base reasoning applies. See validate_phase_a's `stable_base_objects`
# parameter.
_BASE_BEARING_PREDICATES = {"object_on_top", "stacked"}


def _predicate_known(name: str, predicate_module: Any) -> bool:
    """True if ``name`` names a real predicate, per whatever ``predicate_module`` is.

    Accepts, in order of what's most convenient for the caller:
      - ``None``: no validation possible -- permissive (accepts everything). Callers that want
        strict validation should always pass a real module or name set.
      - a ``set``/``frozenset``/``list``/``tuple`` of names (e.g. from ``list_predicate_names``).
      - anything else (a real imported module, or any object with attribute access): checked via
        ``hasattr``/``callable``, exactly mirroring how the runtime mixin dispatches predicates
        (``getattr(conditionals, spec["predicate"])``) -- if ``getattr`` would work there, it
        passes validation here.
    """
    if predicate_module is None:
        return True
    if isinstance(predicate_module, (set, frozenset, list, tuple)):
        return name in predicate_module
    return hasattr(predicate_module, name) and callable(getattr(predicate_module, name))


def _base_objects_for_candidate(candidate: dict[str, Any]) -> Optional[list[str]]:
    """Which object name(s) act as the BASE (the thing rested ON) for a
    ``stacked``/``object_on_top`` candidate, or ``None`` if that can't be determined from this
    candidate (e.g. ``stacked`` without ``order == "bottom_to_top"`` -- ambiguous which end of
    the list is the base, so ``stable_base_objects`` deliberately does not enforce anything in
    that case rather than risk wrongly rejecting a valid unordered stack).
    """
    predicate = candidate.get("predicate")
    args = candidate.get("predicate_args")
    if not isinstance(args, dict):
        return None
    if predicate == "object_on_top":
        ref = args.get("reference_object")
        return [ref] if isinstance(ref, str) else None
    if predicate == "stacked":
        if args.get("order") != "bottom_to_top":
            return None
        objects = args.get("objects")
        if not isinstance(objects, (list, tuple)) or len(objects) < 2:
            return None
        return list(objects[:-1])  # everything except the topmost is acting as a base
    return None


def validate_phase_a(
    candidates: Iterable[dict[str, Any]],
    known_objects: Iterable[str],
    predicate_module: Any = None,
    *,
    reject_ambient_predicates: bool = True,
    stable_base_objects: Optional[Iterable[str]] = None,
    on_reject: Optional[Callable[[dict[str, Any], str], None]] = None,
) -> list[dict[str, Any]]:
    """Filter Phase A candidates down to well-formed, valid ones, per PLAN.md sections 3/4.

    Rejects (silently drops, unless ``on_reject`` is given a callback to observe why) any
    candidate that:
      - is missing a required field (``id``, ``predicate``, ``predicate_args``, ``instruction``,
        ``objects_involved``);
      - references an object name not in ``known_objects`` (checked against
        ``objects_involved``, the field the spec (PLAN.md section 2) designates for this);
      - uses a predicate in ``AMBIENT_STATE_PREDICATES`` (a pure relative-position/orientation/
        transient-gripper check -- can already hold from the random initial scatter, or isn't a
        completed manipulation goal on its own) -- HARD rejection, not just discouraged by
        prompt wording, when ``reject_ambient_predicates=True`` (the default). This is the
        structural fix for a real leak: prompt guidance alone still let
        ``coke_next_to_ceramic_mug`` (``object_next_to``) through on a live run.
      - for ``object_on_top``/``stacked`` (``order="bottom_to_top"``) candidates, when
        ``stable_base_objects`` is given: uses a base/reference object that ISN'T in
        ``stable_base_objects``. Opt-in (``None`` by default -- disabled) because "which objects
        make a physically stable base" is scene-specific, hand-curated knowledge (no geometry
        data available in this pipeline to derive it automatically), not a general rule this
        function should silently impose on every caller/scene. See ``orchestrator.py``'s
        docstring for the judgment call on when to actually pass this.
      - names a predicate not known to ``predicate_module`` (see ``_predicate_known``).

    Never raises on a malformed *candidate* -- a VLM proposing something invalid is an expected,
    not exceptional, outcome; the caller (``discover_tree``) just gets fewer/zero valid
    candidates back. Malformed input to this function itself (e.g. ``candidates`` not being a
    list of dicts) is allowed to raise naturally.
    """
    known_objects = set(known_objects)
    stable_base_objects_set = set(stable_base_objects) if stable_base_objects is not None else None
    required = ("id", "predicate", "predicate_args", "instruction", "objects_involved")

    accepted: list[dict[str, Any]] = []
    for candidate in candidates:
        reason = None
        missing = [f for f in required if f not in candidate]
        if missing:
            reason = f"missing required field(s): {missing}"
        elif not isinstance(candidate["objects_involved"], (list, tuple)):
            reason = "objects_involved must be a list"
        elif not isinstance(candidate["predicate_args"], dict):
            reason = "predicate_args must be a dict"
        else:
            unknown_objects = [
                obj for obj in candidate["objects_involved"] if obj not in known_objects
            ]
            if unknown_objects:
                reason = f"unknown object name(s): {unknown_objects}"
            elif reject_ambient_predicates and candidate["predicate"] in AMBIENT_STATE_PREDICATES:
                reason = (
                    "ambient/state-only predicate not allowed as a subtask's sole success "
                    f"criterion: {candidate['predicate']!r} (see AMBIENT_STATE_PREDICATES -- a "
                    "pure relative-position/orientation/transient-gripper check can already "
                    "hold from the random initial scatter, or isn't a completed manipulation "
                    "goal on its own)"
                )
            elif not _predicate_known(candidate["predicate"], predicate_module):
                reason = f"unknown predicate: {candidate['predicate']!r}"
            elif stable_base_objects_set is not None:
                base_objects = _base_objects_for_candidate(candidate)
                if base_objects is not None:
                    unstable_bases = [b for b in base_objects if b not in stable_base_objects_set]
                    if unstable_bases:
                        reason = (
                            f"physically implausible base for {candidate['predicate']!r}: "
                            f"{unstable_bases} (not in stable_base_objects={sorted(stable_base_objects_set)})"
                        )

        if reason is not None:
            if on_reject is not None:
                on_reject(candidate, reason)
            continue
        accepted.append(candidate)

    return accepted

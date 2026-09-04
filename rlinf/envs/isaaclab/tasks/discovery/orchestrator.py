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

"""The orchestrator: ``discover_tree()`` / ``discover_and_train()`` + ``train_sequence()``.

Two things this module deliberately does NOT do here (both by task scope, not by omission):
  - Nothing in this file invokes real multi-hour training. The real ``launch_and_wait``
    subprocess-launching implementation is written for correctness (see its docstring for how
    it applies the Stage 2 session-detachment crash lesson from CLAUDE.md) but is only exercised
    in tests against trivial, instantaneous shell commands, never real GPU jobs.
  - ``generate_training_config``'s exact CLI-override shape was a best-guess against PLAN.md
    section 8.0's description when first written; since then, reading Agent A's landed
    ``robolab_task.py`` confirmed the guess was correct (``env.train.init_params.edge_spec_path``
    is exactly the key Agent A's code reads) -- still worth a final integration-pass check, but
    no longer purely speculative.

ARCHITECTURE DEVIATION FROM PLAN.md section 4 (read this before touching ``discover_tree``):
a real ``discover_tree()`` run against the mug/coke scene surfaced two problems, one prompt-side
(fixed in ``phase_a.py`` -- see its module docstring) and one structural: PLAN.md section 4's
sketch discovers the *entire* tree upfront, before any training happens, then trains everything
in a second pass (``train_sequence``). That means every node below the root gets proposed with
NO real information about what the scene actually looks like once its ancestors' subtasks are
actually satisfied -- there's no real data to ground it in, because nothing has run yet. That
directly undermines the user's bar of "achievable given the current layout/state" for anything
below the root.

Two ways to reconcile this, both implemented here (deliberately, not just one silently chosen):

  - ``discover_tree()`` -- kept, mostly unchanged in spirit: a pure, no-training preview/dry-run
    walk of the whole tree. Still useful (and is what this task's real, cost-conscious API-call
    verification below uses -- one Phase-A-quality check, no GPU spend) precisely because it
    doesn't train anything. Grounding: the root call gets real data (see ``starting_states.py``);
    every node below the root gets real data too IF a prior training pass already populated
    ``manifest.get_reset_states_path_for_children(node)`` for it (e.g. a resumed/partial run),
    and an explicit "no real data yet, reason conservatively" note otherwise -- it never
    fabricates state data for a node nothing has actually run for.
  - ``discover_and_train()`` -- NEW, the recommended production entrypoint for growing a tree
    from scratch. Implements the interleaved loop the coordinator's option (a) describes:
    discover one node's candidate edges -> actually train each one -> actually collect its
    end-states -> only THEN discover that child's candidates (now with real grounding, because
    the manifest genuinely has real data for it by that point) -> recurse. This is the real,
    structural fix -- not a prompt-only patch -- because it's the only way for a node below the
    root to ever get real "achievable given current state" grounding at proposal time, which the
    user's bar requires, not just prefers.

**Chosen for production: discover_and_train() (option (a)).** Reasoning: the user's bar
explicitly includes "achievable given the current layout/state" for every proposal, not just the
root's; ``discover_tree()``'s upfront-preview shape can only ever satisfy that at the root by
construction, permanently, regardless of any further prompt tuning -- grounding is a data
availability problem there, not a wording problem. ``discover_and_train()`` costs real GPU hours
between every discovery step instead of amortizing discovery into one cheap upfront pass, which
is a real, accepted tradeoff (also raised, and accepted, by the coordinator) in exchange for
every node actually being grounded in what the trained policy really produces, not a guess.
``discover_tree()`` is kept (not deleted) specifically because a training-free preview is still
useful on its own terms (this task's real-API verification run needs exactly that), and because
the two entrypoints share their per-node discovery logic (``_discover_node_edges``) and grounding
mechanism (``starting_states.summarize_starting_state``) rather than duplicating it -- so fixing
the prompt or the grounding logic fixes both call paths at once.

JUDGMENT CALL -- ``stable_base_objects`` (physical-plausibility structural filter, round 2 of the
coordinator's Phase A tuning): a real run still proposed physically-dubious bases for
``object_on_top``/``stacked`` (a board balanced on a mug's rim, something balanced on a standing
coke can). ``predicates.validate_phase_a`` grew a ``stable_base_objects`` parameter that hard-
rejects any such candidate whose base/reference object isn't in that set -- but it's **opt-in**
(``None``/disabled unless a caller passes it), NOT wired as an unconditional default in
``discover_tree``/``discover_and_train`` below. Reasoning:
  - There's no geometry/dimension data in this pipeline to derive "what's a stable base" from
    first principles -- any such set is hand-curated, scene-specific knowledge (```{"cutting_board_a"}```
    for this exact scene), the same kind of fact table as the root-pose alias map in
    ``starting_states.py``. Baking a hardcoded default into a general-purpose validator would
    silently mis-validate a *different* scene with different objects (e.g. a tray or a plate that
    legitimately IS a good secondary base) unless every future caller remembers to override it.
  - A tempting alternative -- "a base is valid if it's already resting on cutting_board_a" (a
    topology check derivable from the tree structure, not hand-curated) -- was considered and
    rejected: it's actively wrong here. It would still allow "mug on top of coke" once coke is on
    the board, which is exactly one of the two bad real examples. Physical stability is a
    property of the object itself (its shape), not of where it currently sits, so no purely
    structural/topological rule over the manifest can substitute for at least some hand-specified
    per-object knowledge.
  - Given that, the honest thing to do is expose the mechanism, make it easy to opt into for a
    specific scene, and default it OFF so ``validate_phase_a`` doesn't quietly assume every caller
    is running this exact 3-object scene. The real-API verification run below DOES opt in
    (``stable_base_objects={"cutting_board_a"}``) since that's the correct, known-good set for
    this scene today.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, FrozenSet, Iterable, Optional

from .manifest import Manifest
from .phase_a import call_vlm_phase_a
from .predicates import validate_phase_a
from .starting_states import summarize_starting_state

# ---------------------------------------------------------------------------
# deterministic id deduplication (across the WHOLE discovery run, not per-node)
# ---------------------------------------------------------------------------
#
# Across every real discover_tree() run so far, the VLM has independently generated the same
# id (e.g. "place_coke_on_cutting_board") at multiple different nodes (different preconditions)
# in the same tree -- harmless in preview mode, but Manifest.add_edge correctly raises
# ManifestError on a duplicate id, which would crash a real train_sequence/discover_and_train
# run partway through, potentially after real GPU hours already spent on an earlier edge in
# that same run. Fixed deterministically here -- not by hoping the VLM does better -- by
# tracking every id seen anywhere else in the run (not just locally within one node's candidate
# list) and renaming a colliding one before it's used anywhere downstream (manifest lookups,
# generate_training_config's edge_spec file, checkpoint/log directory names, ...).

_MAX_PRECONDITION_SUFFIX_LEN = 60


def _precondition_suffix(precondition: Iterable[str]) -> str:
    joined = "_".join(sorted(precondition))
    if not joined:
        return "given_root"
    if len(joined) <= _MAX_PRECONDITION_SUFFIX_LEN:
        return f"given_{joined}"
    # Precondition list too long to make a readable suffix -- a short deterministic hash keeps
    # ids from growing unboundedly deep in the tree while still being reproducible.
    digest = hashlib.sha1(joined.encode()).hexdigest()[:8]
    return f"given_{digest}"


def _dedupe_id(candidate_id: str, seen_ids: set[str], precondition: Iterable[str]) -> str:
    """Return a version of ``candidate_id`` guaranteed not to collide with anything in
    ``seen_ids``, renaming deterministically (not randomly) if it does.

    First choice: qualify with the node's precondition (readable, and the natural
    disambiguator -- the same id proposed from two different nodes almost always means "the
    same kind of subtask, from a different starting point", which the precondition-qualified id
    describes accurately). Falls back to a numeric counter in the vanishingly unlikely case that
    even the qualified id collides (e.g. two different preconditions whose sorted-join happens
    to produce the same suffix, or the same id proposed twice from the exact same node).
    """
    if candidate_id not in seen_ids:
        return candidate_id
    qualified = f"{candidate_id}__{_precondition_suffix(precondition)}"
    if qualified not in seen_ids:
        return qualified
    i = 2
    candidate = f"{qualified}_{i}"
    while candidate in seen_ids:
        i += 1
        candidate = f"{qualified}_{i}"
    return candidate


def _find_matching_existing_edge(
    manifest: Manifest, candidate: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """Is ``candidate`` (already precondition/produces_node-assigned, and possibly already
    ``_dedupe_id``-renamed) the SAME real-world edge as one already in ``manifest`` -- e.g. a
    memoryless Phase A call re-proposing "place the mug" at the root even though that exact edge
    was already trained in a prior run?

    Deliberately NOT an id match (a renamed candidate's id, by construction, no longer equals
    the existing edge's id -- that's the whole point of the rename). Matches on exact
    precondition + predicate + predicate_args instead: same starting node, same success
    condition, is as strong a "this is the same subtask" signal as this pipeline has without
    real semantic understanding of instruction text. Returns ``None`` if nothing matches (a
    genuinely new edge).
    """
    target_precondition = tuple(sorted(candidate.get("precondition", [])))
    for existing in manifest.edges_from(target_precondition):
        if existing["predicate"] == candidate.get(
            "predicate"
        ) and existing["predicate_args"] == candidate.get("predicate_args"):
            return existing
    return None


# ---------------------------------------------------------------------------
# shared per-node discovery step (used by both discover_tree and discover_and_train)
# ---------------------------------------------------------------------------


def _discover_node_edges(
    manifest: Manifest,
    node: FrozenSet[str],
    *,
    scene_objects: Iterable[str],
    phase_a_caller: Callable[..., list[dict[str, Any]]],
    predicate_module: Any,
    predicate_menu: Optional[str],
    get_starting_state_summary: Optional[Callable[[FrozenSet[str]], Optional[str]]] = None,
    render_node: Optional[Callable[[FrozenSet[str]], Optional[str]]] = None,
    stable_base_objects: Optional[Iterable[str]] = None,
    seen_ids: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """One Phase A call + validation for a single node. Marks ``node`` terminal in ``manifest``
    if nothing valid comes back; does NOT add edges to the manifest (that's the caller's job,
    once -- and only if -- it decides to actually keep/train them) and does NOT recurse.

    ``stable_base_objects``: opt-in physical-plausibility filter for object_on_top/stacked
    candidates -- see the module docstring's "JUDGMENT CALL" section for why this isn't a
    default.

    ``seen_ids``: the shared, whole-run id-collision tracker (see ``_dedupe_id`` above). Callers
    (``discover_tree``/``discover_and_train``) must thread the SAME set object through every
    node visited in one run -- passing ``None`` (the default) makes this call self-contained
    (dedupes only within this one node's own candidate batch), which is enough for a
    standalone/test call but NOT enough to prevent cross-node collisions in a real multi-node
    walk; the two public entrypoints below always pass a real shared set.
    """
    node = frozenset(node)
    seen_ids = seen_ids if seen_ids is not None else set()

    if get_starting_state_summary is not None:
        starting_state_summary = get_starting_state_summary(node)
    else:
        starting_state_summary = summarize_starting_state(node, manifest, scene_objects)
    screenshot = render_node(node) if render_node is not None else None

    candidates = phase_a_caller(
        object_list=list(scene_objects),
        satisfied_so_far=sorted(node),
        starting_state_summary=starting_state_summary,
        scene_screenshot=screenshot,
        predicate_menu=predicate_menu,
    )
    candidates = validate_phase_a(
        candidates,
        known_objects=scene_objects,
        predicate_module=predicate_module,
        stable_base_objects=stable_base_objects,
    )
    if not candidates:
        manifest.mark_terminal(node)
        return []

    # Reset pool for this node's CHILDREN: whatever this node's own producing edge recorded
    # after collecting end-states (None at the root, or for any node not yet trained -- falls
    # back to the scene's default starting-state pool, exactly as the existing two-subtask plan
    # already does for subtask_1).
    reset_states_path = manifest.get_reset_states_path_for_children(node)

    edges = []
    for candidate in candidates:
        candidate = dict(candidate)
        final_id = _dedupe_id(candidate["id"], seen_ids, node)
        candidate["id"] = final_id
        seen_ids.add(final_id)
        candidate["precondition"] = sorted(node)
        candidate["produces_node"] = sorted(node | {final_id})
        candidate["reset_states_path"] = reset_states_path
        edges.append(candidate)
    return edges


# ---------------------------------------------------------------------------
# discover_tree -- pure preview/dry-run, no training (see module docstring)
# ---------------------------------------------------------------------------

# Defaults chosen after the real 52-edge/23-node runaway (see phase_a.py's module docstring):
# this 2-object scene's true meaningful depth is 2 (one node per object actually placed);
# DEFAULT_MAX_DEPTH=3 gives one level of headroom for a slightly richer future scene without
# permitting the old default of 4 (which is what let the explosion run that far before the cap
# -- not the root cause, which was the prompt, but worth tightening as defense in depth
# alongside it). DEFAULT_MAX_TOTAL_EDGES=30 is a structural safety valve independent of depth --
# a verbose model proposing many candidates per node could still blow up edge count within a
# shallow depth cap; 30 comfortably covers this scene's real 4-edge tree (and quite a bit of
# future headroom) while still bounding a runaway to a small, cheap, reviewable number rather
# than the 52 actually observed.
DEFAULT_MAX_DEPTH = 3
DEFAULT_MAX_TOTAL_EDGES = 30


def discover_tree(
    manifest: Manifest,
    node: FrozenSet[str] = frozenset(),
    *,
    scene_objects: Iterable[str],
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_total_edges: Optional[int] = DEFAULT_MAX_TOTAL_EDGES,
    visited: Optional[set] = None,
    phase_a_caller: Callable[..., list[dict[str, Any]]] = call_vlm_phase_a,
    predicate_module: Any = None,
    predicate_menu: Optional[str] = None,
    render_node: Optional[Callable[[FrozenSet[str]], Optional[str]]] = None,
    get_starting_state_summary: Optional[Callable[[FrozenSet[str]], Optional[str]]] = None,
    stable_base_objects: Optional[Iterable[str]] = None,
    _edge_budget: Optional[list] = None,
    _seen_ids: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Recursively discover edges via Phase A, without training anything yet.

    Returns a topologically-ordered list of edge dicts (parents always before children). Marks
    each node Phase A returns nothing for as terminal in ``manifest``; does not otherwise write
    to the manifest (edges are only persisted via ``manifest.add_edge`` once actually trained --
    by ``train_sequence`` if you're executing a previously-discovered preview list, or by
    ``discover_and_train`` if you want real per-node grounding -- see module docstring).

    ``max_total_edges`` is a hard structural cap on top of ``max_depth``: recursion stops the
    instant the running edge count (shared across the whole recursive walk via ``_edge_budget``,
    an internal parameter -- do not pass it yourself) reaches the cap, regardless of depth. Set
    ``None`` to disable (not recommended for an unattended run against a real VLM).

    ``stable_base_objects``: opt-in physical-plausibility filter, disabled (``None``) by default
    -- see the module docstring's "JUDGMENT CALL" section. Pass e.g. ``{"cutting_board_a"}`` for
    the mug/coke scene to reject object_on_top/stacked candidates that use a small/tippy object
    as the base.

    Every id in the returned list is guaranteed unique across the WHOLE returned list (see
    ``_dedupe_id`` above) -- seeded from any ids already in ``manifest`` too (defensive, in case
    this is a resumed/partial walk over a manifest a previous ``discover_and_train`` run already
    populated some edges into), so the result can always be fed through ``Manifest.add_edge`` for
    every edge without a ``ManifestError``. ``_edge_budget``/``_seen_ids`` are internal recursion
    state -- do not pass them yourself.
    """
    visited = visited if visited is not None else set()
    _edge_budget = _edge_budget if _edge_budget is not None else [0]
    if _seen_ids is None:
        _seen_ids = {e["id"] for e in manifest.all_edges()}  # seed from prior real edges, if any
    ordered_edges: list[dict[str, Any]] = []
    node = frozenset(node)
    if node in visited or len(node) >= max_depth:
        return ordered_edges
    if max_total_edges is not None and _edge_budget[0] >= max_total_edges:
        return ordered_edges
    visited.add(node)

    edges = _discover_node_edges(
        manifest,
        node,
        scene_objects=scene_objects,
        phase_a_caller=phase_a_caller,
        predicate_module=predicate_module,
        predicate_menu=predicate_menu,
        get_starting_state_summary=get_starting_state_summary,
        render_node=render_node,
        stable_base_objects=stable_base_objects,
        seen_ids=_seen_ids,
    )

    for edge in edges:
        if max_total_edges is not None and _edge_budget[0] >= max_total_edges:
            break
        ordered_edges.append(edge)
        _edge_budget[0] += 1
        ordered_edges += discover_tree(
            manifest,
            node | {edge["id"]},
            scene_objects=scene_objects,
            max_depth=max_depth,
            max_total_edges=max_total_edges,
            visited=visited,
            phase_a_caller=phase_a_caller,
            predicate_module=predicate_module,
            predicate_menu=predicate_menu,
            render_node=render_node,
            get_starting_state_summary=get_starting_state_summary,
            stable_base_objects=stable_base_objects,
            _edge_budget=_edge_budget,
            _seen_ids=_seen_ids,
        )
    return ordered_edges


# ---------------------------------------------------------------------------
# render_current_node
# ---------------------------------------------------------------------------


def render_current_node(
    node: FrozenSet[str], *, env: Any = None
) -> Optional[str]:
    """Best-effort scene screenshot (base64 PNG) -- secondary/legacy fallback.

    STUB. Superseded as the primary grounding mechanism by ``starting_states.py``'s real numeric
    starting-state data (see that module and ``phase_a.py``'s module docstring for why: the user
    wants real object poses, not an image). Still stubbed at ``None`` -- genuinely wiring this up
    needs a *live* IsaacLab env reset to the captured end-state for ``node`` (Agent A's
    ``reset_to_captured_state``/``robolab_task.py`` territory) plus a camera-frame grab, not
    something this module can stand up on its own without an env instance handed to it.
    """
    return None


# ---------------------------------------------------------------------------
# generate_training_config
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfigSpec:
    """A Hydra config name plus CLI overrides, ready to interpolate into
    ``run_embodiment.sh``/``eval_embodiment.sh`` exactly the way PLAN.md section 4's sketch does:
    ``f"bash examples/embodiment/run_embodiment.sh {config_path}"``.
    """

    config_name: str
    overrides: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return " ".join([self.config_name, *self.overrides])


_LORA_PATH_OVERRIDE_RE = re.compile(r"^\+{1,2}actor\.model\.lora_path=")


def strip_actor_lora_path_override(overrides: Iterable[str]) -> list[str]:
    """Remove any existing ``+actor.model.lora_path=...`` / ``++actor.model.lora_path=...``
    entry from a list of Hydra CLI overrides.

    Real, observed bug (not hypothetical): ``generate_training_config`` bakes
    ``+actor.model.lora_path=<parent checkpoint>`` into ``TrainingConfigSpec.overrides`` for the
    TRAINING stage's warm-start. A collection/eval pass against that SAME edge needs to load a
    DIFFERENT checkpoint -- the edge's own just-trained output, not its parent's -- via its own
    lora_path override. Reusing ``config_spec.overrides`` verbatim for that eval pass (as both
    ``_make_default_collect_end_states`` below and ``driver.py``'s ``make_collect_end_states``
    originally did) means Hydra sees ``+actor.model.lora_path=`` twice with two different
    values; its ``+`` prefix (add-new-key-only) correctly refuses the second one with
    ``hydra.errors.ConfigCompositionException: Could not append to config. An item is already
    at 'actor.model.lora_path'.`` -- confirmed on a real collection run for
    ``place_coke_on_cutting_board``. Any caller building an eval/collection override list from a
    training ``TrainingConfigSpec`` must strip the training-time entry first via this function,
    then add its own eval-appropriate one.
    """
    return [o for o in overrides if not _LORA_PATH_OVERRIDE_RE.match(o)]


def generate_training_config(
    edge: dict[str, Any],
    reset_states_path: Optional[str],
    *,
    spec_dir: str | Path,
    config_name: str = "generic_single_edge_grpo_openpi_pi05",
    lora_path: Optional[str] = None,
) -> TrainingConfigSpec:
    """Produce the CLI overrides for training one edge.

    Modeled on ``subtask_2_coke_on_cuttingboard_task.py``'s module-level ``RESET_STATES_PATH``
    injection mechanism. Confirmed (not just guessed) by reading Agent A's landed
    ``rlinf/envs/isaaclab/tasks/robolab_task.py`` and ``RoboLab/robolab/tasks/benchmark/
    generic_single_edge_task.py`` (NOT modified by this task -- read-only): the mixin really does
    read ``init_params.edge_spec_path`` (a path to a JSON file, exactly this function's shape)
    and falls back to an inline ``init_params.edge_spec`` dict if no path is given.
    ``generic_single_edge_grpo_openpi_pi05.yaml``/its ``env/`` counterpart now exist for real
    (``examples/embodiment/config/``), modeled on the proven ``tree_ext_smoketest_train.yaml``
    smoke test plus real-scale settings restored from ``mug_coke_subtask_1_grpo_openpi_pi05.yaml``.

    ``env.eval.*`` mirrors ``env.train.*`` (a real bug found and fixed here, not present in the
    first draft of this function): ``_default_collect_end_states`` below reuses this SAME
    ``TrainingConfigSpec`` against ``eval_embodiment.sh``, which runs against ``env.eval``, not
    ``env.train`` (``eval_embodied_agent.py`` forces ``runner.only_eval=True``, and the eval
    runner is built from ``env.eval``'s config, confirmed by reading that script). Without the
    ``env.eval.*`` overrides too, the end-state-collection pass would have run against whatever
    placeholder ``edge_spec``/``reset_states_path`` happens to be baked into the env yaml file,
    not the actual edge just trained -- meaning end-state collection would silently never trigger
    (the mixin's success dispatch would be checking the wrong predicate, or none at all). No
    separate eval-side config file is needed for this -- the same config, with both env.train and
    env.eval init_params pointed at the real edge, works for both `run_embodiment.sh` and
    `eval_embodiment.sh` (exactly like ``mug_coke_subtask_1_grpo_openpi_pi05.yaml`` already does
    for the hardcoded two-subtask pipeline).
    """
    spec_dir = Path(spec_dir)
    spec_dir.mkdir(parents=True, exist_ok=True)
    edge_spec_path = spec_dir / f"{edge['id']}.json"
    edge_spec_path.write_text(json.dumps(edge, indent=2, sort_keys=True, default=str))

    reset_states_override = str(reset_states_path) if reset_states_path else "null"
    overrides = []
    for env_key in ("train", "eval"):
        overrides.append(f"env.{env_key}.init_params.task_file=generic_single_edge_task.py")
        overrides.append(f"+env.{env_key}.init_params.edge_spec_path={edge_spec_path}")
        overrides.append(f"env.{env_key}.init_params.reset_states_path={reset_states_override}")
    if lora_path:
        overrides.append(f"+actor.model.lora_path={lora_path}")
    return TrainingConfigSpec(config_name=config_name, overrides=overrides)


# ---------------------------------------------------------------------------
# launch_and_wait -- real implementation, written for correctness, not invoked for real here
# ---------------------------------------------------------------------------


@dataclass
class LaunchResult:
    returncode: int
    log_path: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def launch_and_wait(
    cmd: str,
    *,
    log_path: str | Path,
    poll_interval_s: float = 30.0,
    timeout_s: Optional[float] = None,
) -> LaunchResult:
    """Launch a long-running shell command fully session-detached, then poll (not block-wait
    inside a caller that might itself get torn down) until it exits.

    Applies the Stage 2 crash lesson from CLAUDE.md directly, structurally rather than
    procedurally: that failure was traced to a backgrounded-but-not-detached training process
    dying the instant its parent shell session got torn down (by an agent-resume
    ``SendMessage``). The fix here doesn't rely on the *caller* remembering to be careful --
    ``setsid`` gives the child its own session (nothing above it to lose), stdio is fully
    redirected to ``log_path`` (never inherited from whatever's launching this), and the
    launcher process detaches immediately after recording the child's PID rather than blocking
    on it -- so a caller (e.g. an agent session) that itself gets rebuilt mid-flight doesn't take
    the training job down with it. Progress/completion is discovered by *polling* an exit-code
    file, mirroring "the coordinator should avoid nudging a subagent with a real background job
    in flight -- let it poll autonomously instead" from the same lesson.

    Not invoked against real training by anything in this package's tests (multi-hour GPU jobs
    are explicitly out of scope for this task) -- but IS exercised in a unit test against a
    trivial, instantaneous shell command, to verify the detachment/polling/exit-code plumbing
    itself actually works.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path = log_path.with_suffix(log_path.suffix + ".pid")
    exitcode_path = log_path.with_suffix(log_path.suffix + ".exitcode")
    exitcode_path.unlink(missing_ok=True)

    wrapped = (
        f"setsid bash -c '({cmd}) </dev/null >{shlex.quote(str(log_path))} 2>&1; "
        f"echo $? > {shlex.quote(str(exitcode_path))}' "
        f"< /dev/null > /dev/null 2>&1 & echo $! > {shlex.quote(str(pid_path))}"
    )
    # This `subprocess.run` only launches-and-backgrounds (the trailing `&` inside `wrapped`
    # returns control to bash almost immediately); it does not itself block for the command's
    # full duration, and closing its own stdio pipes here doesn't propagate to the
    # already-`setsid`-detached grandchild.
    subprocess.run(wrapped, shell=True, check=True)

    start = time.monotonic()
    while not exitcode_path.exists():
        if timeout_s is not None and (time.monotonic() - start) > timeout_s:
            raise TimeoutError(f"launch_and_wait timed out after {timeout_s}s: {cmd}")
        time.sleep(poll_interval_s)

    returncode = int(exitcode_path.read_text().strip())
    return LaunchResult(returncode=returncode, log_path=str(log_path))


# ---------------------------------------------------------------------------
# shared per-edge training helpers (used by both train_sequence and discover_and_train)
# ---------------------------------------------------------------------------


def _checkpoint_path(log_dir: str | Path, experiment_name: str, global_step: int) -> str:
    """Mirrors the documented checkpoint save-path convention (CLAUDE.md, confirmed against
    ``rlinf/runners/embodied_runner.py``'s ``_save_checkpoint``):
    ``<log_path>/<experiment_name>/checkpoints/global_step_<N>/actor``.
    """
    return str(Path(log_dir) / experiment_name / "checkpoints" / f"global_step_{global_step}" / "actor")


def _make_default_run_training(
    logs_root: Path, max_epochs: int
) -> Callable[[dict[str, Any], Optional[str], TrainingConfigSpec], str]:
    def _default_run_training(edge, current_checkpoint, config_spec: TrainingConfigSpec) -> str:
        log_dir = logs_root / f"{config_spec.config_name}-{edge['id']}"
        cmd = f"bash examples/embodiment/run_embodiment.sh {config_spec}"
        result = launch_and_wait(cmd, log_path=log_dir / "run_embodiment.log")
        if not result.ok:
            raise RuntimeError(f"training stage for edge {edge['id']!r} failed: {result}")
        return _checkpoint_path(log_dir, config_spec.config_name, max_epochs)

    return _default_run_training


def _make_default_collect_end_states(
    logs_root: Path,
) -> Callable[[dict[str, Any], str, TrainingConfigSpec], str]:
    def _default_collect_end_states(edge, checkpoint, config_spec: TrainingConfigSpec) -> str:
        log_dir = logs_root / f"{config_spec.config_name}-{edge['id']}"
        end_states_path = log_dir / f"{edge['id']}_end_states.jsonl"
        # Two real bugs fixed here (both confirmed on live runs, not hypothetical):
        #   1. config_spec.overrides may already contain a training-time
        #      `+actor.model.lora_path=<parent checkpoint>` (from generate_training_config's
        #      warm-start) -- must be stripped before adding this stage's OWN lora_path
        #      (the edge's just-trained checkpoint), or Hydra's `+` (add-only) prefix raises
        #      ConfigCompositionException on the second, colliding `actor.model.lora_path` key.
        #   2. `runner.ckpt_path=<checkpoint>` crashes (IsADirectoryError) for a LoRA adapter
        #      directory -- `torch.load()` expects a single .pt file. The confirmed-correct
        #      mechanism for evaluating/collecting against a LoRA checkpoint is
        #      `+actor.model.lora_path=<checkpoint>` (see CLAUDE.md; proven by Stage 4's real
        #      eval run and by this project's own real collection-run crash/fix history).
        overrides = strip_actor_lora_path_override(config_spec.overrides)
        cmd = (
            f"bash examples/embodiment/eval_embodiment.sh {config_spec.config_name} "
            f"{' '.join(overrides)} "
            f"+actor.model.lora_path={checkpoint} "
            f"env.eval.init_params.save_end_state_path={end_states_path}"
        )
        result = launch_and_wait(cmd, log_path=log_dir / "eval_embodiment.log")
        if not result.ok:
            raise RuntimeError(f"end-state collection for edge {edge['id']!r} failed: {result}")
        return str(end_states_path)

    return _default_collect_end_states


# ---------------------------------------------------------------------------
# train_sequence -- execute an already-decided, fixed edge list (e.g. from discover_tree())
# ---------------------------------------------------------------------------


def train_sequence(
    manifest: Manifest,
    ordered_edges: list[dict[str, Any]],
    base_checkpoint: Optional[str],
    *,
    logs_root: str | Path,
    spec_dir: str | Path,
    config_name: str = "generic_single_edge_grpo_openpi_pi05",
    max_epochs: int = 60,
    run_training: Optional[Callable[[dict[str, Any], Optional[str], TrainingConfigSpec], str]] = None,
    collect_end_states: Optional[
        Callable[[dict[str, Any], str, TrainingConfigSpec], str]
    ] = None,
) -> str:
    """The actual CRL sequence: ONE checkpoint, carried through every edge in order.

    Use this when you already have a fixed, decided edge list -- e.g. a human reviewed a
    ``discover_tree()`` preview and approved it as-is. For growing a tree from scratch, prefer
    ``discover_and_train()`` (see module docstring): this function trains a list that was
    discovered entirely without real state grounding below the root, since ``discover_tree()``
    never trains anything as it walks.

    ``run_training``/``collect_end_states`` are the two injection points -- production defaults
    shell out via ``launch_and_wait`` to ``run_embodiment.sh``/``eval_embodiment.sh``; tests
    inject fakes so no real subprocess/GPU job ever runs. Returns the final checkpoint path (the
    single, continually-trained policy after every edge).
    """
    logs_root = Path(logs_root)
    run_training = run_training or _make_default_run_training(logs_root, max_epochs)
    collect_end_states = collect_end_states or _make_default_collect_end_states(logs_root)

    current_checkpoint = base_checkpoint
    for edge in ordered_edges:
        config_spec = generate_training_config(
            edge,
            reset_states_path=edge.get("reset_states_path"),
            spec_dir=spec_dir,
            config_name=config_name,
            lora_path=current_checkpoint,
        )

        current_checkpoint = run_training(edge, current_checkpoint, config_spec)

        child_states_path = collect_end_states(edge, current_checkpoint, config_spec)

        edge = dict(edge)
        edge["checkpoint"] = current_checkpoint
        manifest.add_edge(edge)
        manifest.set_reset_states_path_for_children(edge["produces_node"], child_states_path)

    return current_checkpoint


# ---------------------------------------------------------------------------
# discover_and_train -- NEW: the interleaved discover-then-train loop (architecture option (a))
# ---------------------------------------------------------------------------


def discover_and_train(
    manifest: Manifest,
    base_checkpoint: Optional[str],
    *,
    scene_objects: Iterable[str],
    logs_root: str | Path,
    spec_dir: str | Path,
    node: FrozenSet[str] = frozenset(),
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_total_edges: Optional[int] = DEFAULT_MAX_TOTAL_EDGES,
    config_name: str = "generic_single_edge_grpo_openpi_pi05",
    max_epochs: int = 60,
    phase_a_caller: Callable[..., list[dict[str, Any]]] = call_vlm_phase_a,
    predicate_module: Any = None,
    predicate_menu: Optional[str] = None,
    render_node: Optional[Callable[[FrozenSet[str]], Optional[str]]] = None,
    stable_base_objects: Optional[Iterable[str]] = None,
    run_training: Optional[Callable[[dict[str, Any], Optional[str], TrainingConfigSpec], str]] = None,
    collect_end_states: Optional[
        Callable[[dict[str, Any], str, TrainingConfigSpec], str]
    ] = None,
    visited: Optional[set] = None,
    initial_edges: Optional[list[dict[str, Any]]] = None,
) -> str:
    """The recommended production entrypoint for growing a tree from scratch (architecture
    decision (a) -- see module docstring for the full "why" versus ``discover_tree()``).

    At each node: one Phase A call (grounded in whatever real starting-state data the manifest
    actually has for that node -- see ``starting_states.py``) -> validate -> for each accepted
    candidate, actually train it and collect its end-states -> record both in ``manifest`` for
    real -> only then recurse into that child, which will now find real grounding via
    ``manifest.get_reset_states_path_for_children`` because the training that just happened
    populated it. Returns the final checkpoint (the single, continually-trained policy).

    Like ``train_sequence``, ``run_training``/``collect_end_states`` are injectable and default
    to real ``launch_and_wait``-based implementations; nothing in this package's tests invokes
    them against real training.

    ``stable_base_objects``: opt-in physical-plausibility filter, disabled (``None``) by default
    -- see the module docstring's "JUDGMENT CALL" section.

    Ids are deduped deterministically across the WHOLE walk (see ``_dedupe_id`` above), seeded
    from any ids already in ``manifest`` (and from ``initial_edges``, if given) -- so every
    ``manifest.add_edge`` call below is guaranteed not to hit a duplicate-id ``ManifestError``
    from a same-run collision (a real, observed failure mode: the VLM independently proposing
    the same id at two different nodes).

    ``initial_edges``: skip the Phase A call for ``node`` (the starting node) and use this
    already-discovered, already-validated candidate list instead -- for continuing a walk whose
    starting node was discovered in a separate call (e.g. interactively, before handing off to
    an unattended driver), without paying for a redundant real API call to re-discover the exact
    same node. Only applies to ``node`` itself; every node below it is discovered normally.
    Callers are responsible for having already run these candidates through
    ``predicates.validate_phase_a`` (and deduped them against the manifest) themselves -- this
    function does not re-validate or re-dedupe ``initial_edges``, only what it discovers itself.
    """
    visited = visited if visited is not None else set()
    logs_root = Path(logs_root)
    run_training = run_training or _make_default_run_training(logs_root, max_epochs)
    collect_end_states = collect_end_states or _make_default_collect_end_states(logs_root)

    state = {"checkpoint": base_checkpoint}
    seen_ids = {e["id"] for e in manifest.all_edges()}  # seed from prior real edges, if any
    if initial_edges:
        seen_ids.update(e["id"] for e in initial_edges)
    start_node = frozenset(node)

    def _walk(node: FrozenSet[str]) -> None:
        node = frozenset(node)
        if node in visited or len(node) >= max_depth:
            return
        if max_total_edges is not None and len(manifest) >= max_total_edges:
            return
        visited.add(node)

        if node == start_node and initial_edges is not None:
            edges = initial_edges
        else:
            edges = _discover_node_edges(
                manifest,
                node,
                scene_objects=scene_objects,
                phase_a_caller=phase_a_caller,
                predicate_module=predicate_module,
                predicate_menu=predicate_menu,
                get_starting_state_summary=None,  # always real: summarize_starting_state via manifest
                render_node=render_node,
                stable_base_objects=stable_base_objects,
                seen_ids=seen_ids,
            )

        for edge in edges:
            if max_total_edges is not None and len(manifest) >= max_total_edges:
                break

            config_spec = generate_training_config(
                edge,
                reset_states_path=edge.get("reset_states_path"),
                spec_dir=spec_dir,
                config_name=config_name,
                lora_path=state["checkpoint"],
            )
            state["checkpoint"] = run_training(edge, state["checkpoint"], config_spec)
            child_states_path = collect_end_states(edge, state["checkpoint"], config_spec)

            trained_edge = dict(edge)
            trained_edge["checkpoint"] = state["checkpoint"]
            manifest.add_edge(trained_edge)
            manifest.set_reset_states_path_for_children(
                trained_edge["produces_node"], child_states_path
            )

            _walk(node | {edge["id"]})

    _walk(node)
    return state["checkpoint"]


# ---------------------------------------------------------------------------
# discover_and_train_by_level -- breadth-first counterpart (architecture correction)
# ---------------------------------------------------------------------------
#
# CORRECTION (real run, real bug): discover_and_train() above is pre-order DFS -- for each
# candidate at a node, it trains it AND FULLY RECURSES INTO IT (training every descendant)
# BEFORE ever returning to train that node's next sibling. Launched against the real mug/coke
# tree starting from a non-root node, this meant it would only ever walk ONE lineage below that
# node, permanently -- never train a sibling branch at all, let alone the tree's OTHER root
# child. That directly contradicts PLAN.md section 0's design (root branches into multiple
# children, ALL of them trained into the single evolving checkpoint, kept as separate leaves,
# not one lineage chosen and the rest abandoned) and isn't even the "do all of one branch before
# the other" DFS ordering PLAN.md leaves open as a valid choice among orderings -- it's strictly
# narrower than that: only one branch, ever.
#
# discover_and_train_by_level() fixes this by walking BREADTH-FIRST / level-order instead:
# every edge at depth D (the whole frontier of nodes at that depth) is discovered, trained, and
# has its end-states collected before depth D+1 is even discovered, let alone trained. The
# single continually-trained checkpoint is still threaded through every edge in the order
# actually trained (still one lineage, per PLAN.md section 0 -- level-order changes ONLY the
# iteration order, not the "one checkpoint" invariant). This is the recommended entrypoint for
# growing a tree from scratch going forward; discover_and_train() (DFS) is kept for whichever
# callers genuinely want depth-first (or a single already-known lineage) rather than deleted,
# since PLAN.md itself treats sibling-ordering as an open, paper-dependent question -- but level
# order is the correct default for "walk this whole tree autonomously," not DFS.


def discover_and_train_by_level(
    manifest: Manifest,
    base_checkpoint: Optional[str],
    *,
    scene_objects: Iterable[str],
    logs_root: str | Path,
    spec_dir: str | Path,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_total_edges: Optional[int] = DEFAULT_MAX_TOTAL_EDGES,
    config_name: str = "generic_single_edge_grpo_openpi_pi05",
    max_epochs: int = 60,
    phase_a_caller: Callable[..., list[dict[str, Any]]] = call_vlm_phase_a,
    predicate_module: Any = None,
    predicate_menu: Optional[str] = None,
    render_node: Optional[Callable[[FrozenSet[str]], Optional[str]]] = None,
    stable_base_objects: Optional[Iterable[str]] = None,
    run_training: Optional[Callable[[dict[str, Any], Optional[str], TrainingConfigSpec], str]] = None,
    collect_end_states: Optional[
        Callable[[dict[str, Any], str, TrainingConfigSpec], str]
    ] = None,
    initial_edges_by_node: Optional[dict[FrozenSet[str], list[dict[str, Any]]]] = None,
    level_queue_dir: Optional[str | Path] = None,
) -> str:
    """Breadth-first / level-order counterpart to ``discover_and_train`` -- see the section
    comment above for why this, not DFS, is the correct default for walking a whole tree
    unattended. Fully discovers + trains + collects every edge across the CURRENT depth's whole
    frontier before advancing to the next depth.

    ``level_queue_dir``: opt-in, purely additive progress logging for external tooling (e.g. a
    dashboard) -- if given, as soon as one depth's ``level_edges`` is fully computed (discovery
    done for every node at that depth, already deduped -- before any of them are trained), it's
    written verbatim to ``<level_queue_dir>/level_<depth>_queue.json`` (overwritten, not
    appended, if this depth is ever revisited). This is the single source of truth for "what's
    queued at the current level, with correct ids" -- deliberately NOT re-derived from
    ``initial_edges_by_node``'s raw seed files on the reader's side, since those still have
    PRE-dedup ids (e.g. two independently-discovered edges can legitimately share a raw id
    before ``_dedupe_id`` disambiguates them here) and re-implementing that dedup logic a second
    time elsewhere would be a real way for the two copies to drift. ``None`` (default) disables
    this entirely -- no behavior change, no file written, for any caller that doesn't need it.

    ``initial_edges_by_node``: like ``discover_and_train``'s ``initial_edges``, but generalized
    to seed MULTIPLE already-discovered nodes at once (not just the single starting node) --
    e.g. resuming with the root's candidates already known from an earlier real Phase A call
    AND a depth-1 node's children already discovered too, without paying for either call again.
    Any node not present as a key is discovered normally via ``phase_a_caller``. Values are
    trusted as already-validated (same contract as ``initial_edges``) -- this function does not
    re-run them through ``predicates.validate_phase_a``.

    A candidate matching an edge already in ``manifest`` (same precondition + predicate +
    predicate_args -- see ``_find_matching_existing_edge``, NOT an id match: a real, memoryless
    Phase A call re-proposing an already-trained edge -- e.g. the root's "place the mug" -- gets
    a *different* id than the original once ``_dedupe_id`` renames it, so matching has to be by
    content, not id) is recognized and NOT retrained -- its produces_node still advances into
    the next level's frontier using the EXISTING edge's own ``produces_node`` (not the
    candidate's, which may name a phantom node nothing was actually captured against), so the
    next level's discovery/grounding lines up with whatever real end-states already exist.

    Returns the final checkpoint (the single, continually-trained policy after every edge
    actually trained this call, in the order trained).
    """
    logs_root = Path(logs_root)
    run_training = run_training or _make_default_run_training(logs_root, max_epochs)
    collect_end_states = collect_end_states or _make_default_collect_end_states(logs_root)
    initial_edges_by_node = {
        frozenset(k): v for k, v in (initial_edges_by_node or {}).items()
    }

    seen_ids = {e["id"] for e in manifest.all_edges()}

    # Dedupe each seeded node's own candidates against the running seen_ids BEFORE using them
    # (real bug, caught in a dry run before spending real GPU time): initial_edges_by_node's
    # entries typically come from SEPARATE real Phase A calls made without knowledge of each
    # other (e.g. the root's candidates from one call, a depth-1 node's children from another,
    # possibly hours apart) -- they can legitimately collide on id even though they're
    # semantically different edges, exactly what _dedupe_id exists to catch. Unlike freshly
    # discovered edges (which flow through _discover_node_edges and get deduped there
    # automatically), seeded edges bypass that path entirely, so this function has to do it
    # explicitly. Iterates nodes in a fixed (sorted) order for determinism.
    for seed_node in sorted(initial_edges_by_node, key=sorted):
        deduped = []
        for candidate in initial_edges_by_node[seed_node]:
            candidate = dict(candidate)
            final_id = _dedupe_id(candidate["id"], seen_ids, seed_node)
            if final_id != candidate["id"]:
                candidate["id"] = final_id
                # produces_node was computed by whoever discovered this candidate, using the
                # OLD id -- keep it consistent with the rename.
                candidate["produces_node"] = sorted(seed_node | {final_id})
            seen_ids.add(final_id)
            deduped.append(candidate)
        initial_edges_by_node[seed_node] = deduped

    # Real bug, fixed here (confirmed on a live run: a depth-1 edge warm-started from edge 1's
    # checkpoint instead of edge 2's, after both root edges were already trained and persisted
    # in a PRIOR call). `base_checkpoint` is only the right starting point for a truly FRESH
    # walk (empty manifest). When RESUMING -- which every real driver relaunch does, since
    # root_candidates.json gets re-seeded every time and the "already trained, don't retrain"
    # branch below (`existing is not None: ... continue`) deliberately does NOT touch
    # `checkpoint` per-edge (touching it there risks rolling the checkpoint BACKWARD if an old,
    # out-of-order duplicate is matched mid-walk -- see that branch's own comment) -- the only
    # correct source of "what's the single lineage's current real state" is the manifest itself:
    # every edge is added to it via `manifest.add_edge` immediately after it's actually trained,
    # in that exact order, so `manifest.all_edges()`'s last entry IS the true current checkpoint,
    # full stop, regardless of which edges belonged to which depth or branch.
    checkpoint = manifest.all_edges()[-1]["checkpoint"] if len(manifest) > 0 else base_checkpoint
    visited_nodes: set[FrozenSet[str]] = set()
    frontier: set[FrozenSet[str]] = {frozenset()}
    depth = 0

    while frontier and depth < max_depth:
        if max_total_edges is not None and len(manifest) >= max_total_edges:
            break

        # Discover EVERY node at this level first (before training any of them) -- a level can
        # have more than one node (e.g. both root children's own children, once both root edges
        # are trained), and none of that level's discovery should depend on training order
        # within the level.
        level_edges: list[dict[str, Any]] = []
        for node in sorted(frontier, key=sorted):
            if node in visited_nodes:
                continue
            visited_nodes.add(node)

            if node in initial_edges_by_node:
                edges = initial_edges_by_node[node]
            else:
                edges = _discover_node_edges(
                    manifest,
                    node,
                    scene_objects=scene_objects,
                    phase_a_caller=phase_a_caller,
                    predicate_module=predicate_module,
                    predicate_menu=predicate_menu,
                    get_starting_state_summary=None,  # always real via manifest, see starting_states.py
                    render_node=render_node,
                    stable_base_objects=stable_base_objects,
                    seen_ids=seen_ids,
                )
            level_edges.extend(edges)

        if level_queue_dir is not None:
            queue_path = Path(level_queue_dir) / f"level_{depth}_queue.json"
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            queue_path.write_text(json.dumps(level_edges, indent=2, sort_keys=True, default=str))

        next_frontier: set[FrozenSet[str]] = set()
        for edge in level_edges:
            if max_total_edges is not None and len(manifest) >= max_total_edges:
                break

            existing = _find_matching_existing_edge(manifest, edge)
            if existing is not None:
                # Already trained (e.g. in a prior run before this call) -- don't retrain; just
                # advance its child node into the next level's frontier. Uses the EXISTING
                # edge's own produces_node, not the candidate's -- if _dedupe_id renamed this
                # candidate (its id collided with the already-trained edge, seeded into
                # seen_ids above), its own produces_node would name a phantom node nothing was
                # actually captured against; the manifest's real record is the source of truth.
                #
                # Deliberately does NOT touch `checkpoint` here (e.g. `checkpoint =
                # existing["checkpoint"]`) even though that looks tempting: this branch can fire
                # for an edge encountered anywhere in the walk, in whatever order `level_edges`
                # happens to list candidates, which is NOT guaranteed to match real chronological
                # training order in general. Setting `checkpoint` from an out-of-order match
                # could roll it BACKWARD to a stale value, discarding real progress from edges
                # trained after it. The single correct source for "current checkpoint" is set
                # once, before this loop, from `manifest.all_edges()`'s real chronological order
                # (see the comment above the `checkpoint = ...` initialization) -- this branch
                # just needs to leave it alone.
                next_frontier.add(frozenset(existing["produces_node"]))
                continue

            config_spec = generate_training_config(
                edge,
                reset_states_path=edge.get("reset_states_path"),
                spec_dir=spec_dir,
                config_name=config_name,
                lora_path=checkpoint,
            )
            checkpoint = run_training(edge, checkpoint, config_spec)
            child_states_path = collect_end_states(edge, checkpoint, config_spec)

            trained_edge = dict(edge)
            trained_edge["checkpoint"] = checkpoint
            manifest.add_edge(trained_edge)
            manifest.set_reset_states_path_for_children(
                trained_edge["produces_node"], child_states_path
            )

            next_frontier.add(frozenset(trained_edge["produces_node"]))

        frontier = next_frontier
        depth += 1

    return checkpoint

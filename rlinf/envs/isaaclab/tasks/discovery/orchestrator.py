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

``launch_and_wait`` never runs real multi-hour training here; it's exercised in tests only
against trivial, instantaneous shell commands.

Two tree-walk strategies, both kept:
  - ``discover_tree()`` -- pure preview/dry-run, no training. Nodes below the root only get real
    starting-state grounding if a prior training pass already populated
    ``manifest.get_reset_states_path_for_children(node)`` for them; otherwise discovery proceeds
    without state data for that node.
  - ``discover_and_train()`` / ``discover_and_train_by_level()`` -- the production entrypoints:
    discover one node's candidates, train each, collect end-states, only then discover its
    children (now grounded in real data the training just produced). Costs GPU time between every
    discovery step instead of discovering the whole tree upfront, in exchange for every node
    being grounded in what the trained policy actually produces rather than a guess.
  - ``discover_and_train_by_level()`` is breadth-first and is the recommended entrypoint for
    walking a whole tree unattended: it trains every edge at the current depth before discovering
    the next depth, so a root with multiple children gets ALL of them trained into the one
    checkpoint, not just the first one recursed into. ``discover_and_train()`` (DFS) fully
    recurses into each candidate before trying its next sibling, so from a non-root starting node
    it only ever walks one lineage -- kept for callers who genuinely want depth-first.

``stable_base_objects`` (in ``discover_tree``/``discover_and_train*`` below): opt-in
physical-plausibility filter passed through to ``predicates.validate_phase_a``, disabled
(``None``) by default. It's hand-curated, scene-specific knowledge (e.g. ``{"cutting_board_a"}``
for the mug/coke scene) with no way to derive it from first principles in this pipeline, so it's
never defaulted on -- a hardcoded default would silently mis-validate a different scene's objects.
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
# The VLM can independently propose the same id (e.g. "place_coke_on_cutting_board") at
# different nodes in the same tree. Manifest.add_edge raises on a duplicate id, which would
# crash a run partway through after GPU hours were already spent. Renamed deterministically here
# by tracking every id seen anywhere in the run and renaming a collision before it's used
# downstream (manifest lookups, edge_spec files, checkpoint/log directory names).

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
    """Return a version of ``candidate_id`` guaranteed not to collide with ``seen_ids``.

    Qualifies with the node's precondition first (readable, and usually the correct
    disambiguator). Falls back to a numeric counter if even the qualified id collides.
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
    """Is ``candidate`` the same real-world edge as one already in ``manifest``?

    Matches on precondition + predicate + predicate_args, not id (a `_dedupe_id`-renamed
    candidate's id no longer equals the existing edge's by construction). Returns ``None`` if
    nothing matches.
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
    if nothing valid comes back. Does not add edges to the manifest or recurse.

    ``seen_ids``: the shared, whole-run id-collision tracker (see ``_dedupe_id``). Callers must
    thread the SAME set through every node visited in one run; ``None`` dedupes only within this
    node's own candidate batch.
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

    # Reset pool for this node's children. None at the root or any untrained node -- falls back
    # to the scene's default starting-state pool.
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

# DEFAULT_MAX_DEPTH: this 2-object scene's meaningful depth is 2 (one node per object placed);
# 3 gives one level of headroom. DEFAULT_MAX_TOTAL_EDGES: a depth cap alone doesn't bound edge
# count if a node proposes many candidates -- 30 is a structural ceiling independent of depth.
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

    Returns a topologically-ordered list of edge dicts (parents before children). Marks each
    node Phase A returns nothing for as terminal; edges are only persisted via
    ``manifest.add_edge`` once actually trained (by ``train_sequence`` or ``discover_and_train``).

    ``max_total_edges`` is a hard cap on top of ``max_depth``: recursion stops the instant the
    running edge count reaches it, regardless of depth. ``None`` disables it.

    ``stable_base_objects``: opt-in physical-plausibility filter, disabled by default. Pass e.g.
    ``{"cutting_board_a"}`` to reject object_on_top/stacked candidates using a tippy base.

    Every returned id is unique across the whole list (seeded from ids already in ``manifest``),
    so it can always be fed through ``Manifest.add_edge`` without a ``ManifestError``.
    ``_edge_budget``/``_seen_ids`` are internal recursion state -- do not pass them yourself.
    """
    visited = visited if visited is not None else set()
    _edge_budget = _edge_budget if _edge_budget is not None else [0]
    if _seen_ids is None:
        _seen_ids = {e["id"] for e in manifest.all_edges()}
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
    """Best-effort scene screenshot (base64 PNG). STUB, always returns ``None``.

    ``starting_states.py``'s numeric starting-state data is the real grounding mechanism instead
    of an image. Wiring this up for real needs a live IsaacLab env reset to ``node``'s captured
    end-state plus a camera-frame grab -- not something this module can do without an env
    instance.
    """
    return None


# ---------------------------------------------------------------------------
# generate_training_config
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfigSpec:
    """A Hydra config name plus CLI overrides, e.g.
    ``f"bash examples/embodiment/run_embodiment.sh {config_spec}"``.
    """

    config_name: str
    overrides: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return " ".join([self.config_name, *self.overrides])


_LORA_PATH_OVERRIDE_RE = re.compile(r"^\+{1,2}actor\.model\.lora_path=")


def strip_actor_lora_path_override(overrides: Iterable[str]) -> list[str]:
    """Remove any existing ``+actor.model.lora_path=...`` / ``++actor.model.lora_path=...``
    entry from a list of Hydra CLI overrides.

    ``generate_training_config`` bakes the parent checkpoint's lora_path into
    ``TrainingConfigSpec.overrides`` for training's warm-start. A collection/eval pass against
    that same edge needs a different lora_path (the edge's own just-trained checkpoint) --
    reusing the overrides verbatim would give Hydra two `+actor.model.lora_path=` entries, and
    its add-only `+` prefix raises `ConfigCompositionException` on the second one. Strip the
    training-time entry first, then add the eval-appropriate one.
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

    The env mixin reads ``init_params.edge_spec_path`` (a JSON file, this function's shape),
    falling back to an inline ``init_params.edge_spec`` dict if no path is given.

    Overrides both ``env.train.*`` and ``env.eval.*``: ``eval_embodied_agent.py`` builds its
    runner from ``env.eval``'s config, so ``_default_collect_end_states``'s reuse of this same
    ``TrainingConfigSpec`` against `eval_embodiment.sh` needs `env.eval.init_params` pointed at
    the real edge too, or end-state collection silently checks the wrong predicate.
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
    """Launch a shell command fully session-detached, then poll for its exit code.

    ``setsid`` gives the child its own session and stdio is redirected to ``log_path``, so a
    caller that itself gets torn down mid-flight doesn't kill the training job with it.
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
    # The trailing `&` returns control almost immediately; this doesn't block for the command's
    # full duration.
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
    """``<log_path>/<experiment_name>/checkpoints/global_step_<N>/actor`` -- the save-path
    convention from ``rlinf/runners/embodied_runner.py``'s ``_save_checkpoint``.
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
        # config_spec.overrides may already carry a training-time lora_path -- strip it (see
        # strip_actor_lora_path_override) before adding this stage's own. Use
        # +actor.model.lora_path, not runner.ckpt_path: torch.load() on a LoRA adapter
        # directory raises IsADirectoryError.
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
    """The CRL sequence: one checkpoint, carried through every edge in order.

    Use for a fixed, already-decided edge list (e.g. an approved ``discover_tree()`` preview).
    For growing a tree from scratch, prefer ``discover_and_train()`` -- this trains a list that
    was discovered without real state grounding below the root.

    ``run_training``/``collect_end_states`` default to real ``launch_and_wait``-based
    implementations; tests inject fakes. Returns the final checkpoint path.
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
# discover_and_train -- the interleaved discover-then-train loop, depth-first
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
    """Grows a tree from scratch depth-first: discover, validate, train, collect end-states,
    then recurse into the child (now grounded via ``manifest.get_reset_states_path_for_children``,
    populated by the training that just happened). Returns the final checkpoint.

    ``stable_base_objects``: opt-in physical-plausibility filter, disabled by default.

    Ids are deduped across the whole walk (see ``_dedupe_id``), seeded from ids already in
    ``manifest`` and ``initial_edges``.

    ``initial_edges``: skip the Phase A call for the starting node and use this
    already-discovered, already-validated candidate list instead. Only applies to ``node``
    itself; not re-validated or re-deduped by this function.
    """
    visited = visited if visited is not None else set()
    logs_root = Path(logs_root)
    run_training = run_training or _make_default_run_training(logs_root, max_epochs)
    collect_end_states = collect_end_states or _make_default_collect_end_states(logs_root)

    state = {"checkpoint": base_checkpoint}
    seen_ids = {e["id"] for e in manifest.all_edges()}
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
# discover_and_train_by_level -- breadth-first counterpart
# ---------------------------------------------------------------------------
#
# discover_and_train() is pre-order DFS: for each candidate at a node, it trains it and fully
# recurses into it before trying that node's next sibling. From a non-root starting node with
# multiple children, this only ever walks one lineage -- a root's other children never get
# trained at all.
#
# discover_and_train_by_level() walks breadth-first instead: every edge at depth D is
# discovered, trained, and has its end-states collected before depth D+1 is discovered. Still
# one checkpoint threaded through every edge in training order -- level order only changes
# iteration order, not the single-checkpoint invariant. This is the recommended entrypoint for
# growing a tree unattended; discover_and_train() (DFS) is kept for callers who want a single
# known lineage.


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
    """Breadth-first counterpart to ``discover_and_train``. Fully discovers, trains, and
    collects every edge across the current depth's whole frontier before advancing to the next.

    ``level_queue_dir``: if given, each depth's fully-discovered (deduped) ``level_edges`` is
    written to ``<level_queue_dir>/level_<depth>_queue.json`` right before training starts on
    that depth. This is the source of truth for "what's queued, with final ids" -- readers
    shouldn't re-derive it from ``initial_edges_by_node``'s raw (pre-dedup) seed files. ``None``
    disables it.

    ``initial_edges_by_node``: seeds multiple already-discovered nodes at once (root candidates
    from one earlier call, a depth-1 node's children from another), skipping a redundant Phase A
    call for each. Values are trusted as already-validated.

    A candidate matching an edge already in ``manifest`` (by content -- see
    ``_find_matching_existing_edge``, not id, since a memoryless Phase A call re-proposing an
    already-trained edge gets a different id once ``_dedupe_id`` renames it) is not retrained;
    its child node advances into the next level's frontier using the existing edge's own
    ``produces_node``.

    Returns the final checkpoint.
    """
    logs_root = Path(logs_root)
    run_training = run_training or _make_default_run_training(logs_root, max_epochs)
    collect_end_states = collect_end_states or _make_default_collect_end_states(logs_root)
    initial_edges_by_node = {
        frozenset(k): v for k, v in (initial_edges_by_node or {}).items()
    }

    seen_ids = {e["id"] for e in manifest.all_edges()}

    # Seeded candidates bypass _discover_node_edges' automatic dedup, so do it explicitly here:
    # initial_edges_by_node's entries typically come from separate Phase A calls (root, then a
    # depth-1 node, possibly hours apart) and can collide on id despite being different edges.
    # Sorted iteration order for determinism.
    for seed_node in sorted(initial_edges_by_node, key=sorted):
        deduped = []
        for candidate in initial_edges_by_node[seed_node]:
            candidate = dict(candidate)
            final_id = _dedupe_id(candidate["id"], seen_ids, seed_node)
            if final_id != candidate["id"]:
                candidate["id"] = final_id
                # produces_node used the old id -- keep it consistent with the rename.
                candidate["produces_node"] = sorted(seed_node | {final_id})
            seen_ids.add(final_id)
            deduped.append(candidate)
        initial_edges_by_node[seed_node] = deduped

    # `base_checkpoint` is only correct for a fresh walk (empty manifest). On resume, the true
    # current checkpoint is the manifest's last-added edge -- edges are added in real training
    # order regardless of depth/branch, so `manifest.all_edges()[-1]` is authoritative.
    checkpoint = manifest.all_edges()[-1]["checkpoint"] if len(manifest) > 0 else base_checkpoint
    visited_nodes: set[FrozenSet[str]] = set()
    frontier: set[FrozenSet[str]] = {frozenset()}
    depth = 0

    while frontier and depth < max_depth:
        if max_total_edges is not None and len(manifest) >= max_total_edges:
            break

        # Discover every node at this level before training any of them -- discovery within a
        # level must not depend on training order within it.
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
                # Already trained -- advance using the EXISTING edge's produces_node (the
                # candidate's may name a phantom node if _dedupe_id renamed it). Deliberately
                # does not touch `checkpoint`: level_edges order doesn't match real chronological
                # training order, so setting it from an out-of-order match could roll it backward.
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

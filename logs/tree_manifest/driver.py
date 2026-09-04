#!/usr/bin/env python3
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

"""Autonomous driver: walks the VLM-discovered subtask tree for the mug/coke/cutting-board
scene, unattended, for however many real edges remain.

Uses ``discover_and_train_by_level()`` (breadth-first), starting from the true root
(``frozenset()``), which fully trains every edge at each depth before advancing to the next.

Seeded via ``INITIAL_EDGES_BY_NODE`` with already-discovered candidates so this driver pays for
zero redundant Phase A calls at startup: the root's candidates (``ROOT_CANDIDATES_PATH``) and
edge 1's children (``EDGE1_CHILDREN_PATH``). A third real root candidate
(``line_up_objects``/``objects_in_line``) is kept out and preserved at
``root_candidates_excluded_pending_confirmation.json`` rather than trained automatically.
``place_mug_on_cutting_board`` is already trained and persisted in the manifest;
``discover_and_train_by_level``'s content-based match recognizes it on the root frontier and
skips retraining, advancing straight to its known children.

``run_training``/``collect_end_states`` shell out to the apptainer container (same
binds/env vars as ``enter_robolab.sh``), fully session-detached via
``orchestrator.launch_and_wait``.

Meant to be launched via ``launch_driver.sh`` (setsid + redirected stdio + disown) so it survives
independent of any calling session for its entire multi-day lifetime.

Progress is written to ``STATUS_PATH`` (JSON, updated at every stage transition) alongside
stdout/stderr (redirected by the launcher to ``driver.log``).

Stops on a training/collection subprocess failing -- checked by exit code AND a scan of its own
log for fatal signatures, since a piped ``eval_embodiment.sh`` can report exit 0 on a real crash
-- or on hitting ``max_depth``/``max_total_edges``. Never auto-retries. Does not implement
crash-resume for the driver process itself: a relaunch after a crash redoes whatever level was
in progress.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

RLINF_ROOT = Path("/scratch/cluster/jshim12/RLinf")
sys.path.insert(0, str(RLINF_ROOT))

from rlinf.envs.isaaclab.tasks.discovery.manifest import Manifest  # noqa: E402
from rlinf.envs.isaaclab.tasks.discovery.orchestrator import (  # noqa: E402
    LaunchResult,
    TrainingConfigSpec,
    _checkpoint_path,
    discover_and_train_by_level,
    generate_training_config,
    launch_and_wait,
    strip_actor_lora_path_override,
)
from rlinf.envs.isaaclab.tasks.discovery.predicates import (  # noqa: E402
    build_predicate_menu,
    list_predicate_names,
)

# ---------------------------------------------------------------------------
# Fixed, scene-specific configuration for this run (mug/coke/cutting-board scene).
# ---------------------------------------------------------------------------

SCENE_OBJECTS = ["ceramic_mug", "coke", "cutting_board_a"]
STABLE_BASE_OBJECTS = {"cutting_board_a"}
CONFIG_NAME = "generic_single_edge_grpo_openpi_pi05"
MAX_EPOCHS = 60  # matches runner.max_epochs in that config -- final checkpoint is global_step_60

MANIFEST_DIR = RLINF_ROOT / "logs" / "tree_manifest"
MANIFEST_PATH = MANIFEST_DIR / "manifest.json"
SPEC_DIR = MANIFEST_DIR / "edge_specs"
LOGS_ROOT = RLINF_ROOT / "logs"
STATUS_PATH = MANIFEST_DIR / "driver_status.json"
ROOT_CANDIDATES_PATH = MANIFEST_DIR / "root_candidates.json"
EDGE1_CHILDREN_PATH = MANIFEST_DIR / "edge1_children.json"
LAUNCH_SCRIPTS_DIR = MANIFEST_DIR / "launch_scripts"
STAGE_INNER_SCRIPT = LAUNCH_SCRIPTS_DIR / "run_stage_inner.sh"

# Edge 1's checkpoint -- the base this continuation warm-starts from. Written by the
# interactive step that trained edge 1.
BASE_CHECKPOINT = (MANIFEST_DIR / "edge1_checkpoint.txt").read_text().strip()

FATAL_LOG_PATTERNS = (
    "Traceback (most recent call last)",
    "Error:",  # broad on purpose: catches IsADirectoryError:, RuntimeError:, KeyError:, ...
    "Exiting main process due to a failure",
    "Segmentation fault",
    "No device could be created",
    "CUDA out of memory",
)

APPTAINER_BASE_ARGS = [
    "apptainer",
    "exec",
    "--nv",
    "--bind",
    "/scratch/cluster/jshim12:/scratch/cluster/jshim12",
    "--bind",
    "/scratch/cluster/jshim12/vulkan_icd/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json",
    "--env",
    "VK_LOADER_DEBUG=all",
    "--env",
    "CARB_LOG_LEVEL=error",
    "--env",
    "PYTHONWARNINGS=ignore",
    "--env",
    "CPATH=",
    "--env",
    "C_INCLUDE_PATH=",
    "--env",
    "CPLUS_INCLUDE_PATH=",
    "--env",
    "CUDA_HOME=/scratch/cluster/jshim12/cuda/cuda-12.4",
    "--env",
    "CUDA_PATH=/scratch/cluster/jshim12/cuda/cuda-12.4",
    "--env",
    "TORCH_EXTENSIONS_DIR=/scratch/cluster/jshim12/.cache/torch_extensions",
    "--env",
    "ACCEPT_EULA=Y",
]
SANDBOX_PATH = "/scratch/cluster/jshim12/rlinf_sandbox/"


# ---------------------------------------------------------------------------
# status.json -- structured progress log, updated at every stage transition
# ---------------------------------------------------------------------------


def write_status(**kwargs) -> None:
    status: dict = {}
    if STATUS_PATH.exists():
        try:
            status = json.loads(STATUS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            status = {}
    status.update(kwargs)
    status["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    STATUS_PATH.write_text(json.dumps(status, indent=2, default=str))


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# apptainer-wrapped launch, verified by exit code AND a fatal-signature log scan (never one alone)
# ---------------------------------------------------------------------------


def _apptainer_cmd(env_vars: dict[str, str]) -> str:
    args = list(APPTAINER_BASE_ARGS)
    for key, value in env_vars.items():
        args += ["--env", f"{key}={value}"]
    args += [SANDBOX_PATH, "bash", str(STAGE_INNER_SCRIPT)]
    return " ".join(shlex.quote(a) for a in args)


def _launch_stage_and_verify(
    *,
    run_mode: str,
    config_name: str,
    overrides: list[str],
    log_dir: Path,
    lora_path: str = "",
    poll_interval_s: float = 120.0,
) -> str:
    """Launch one real training/collection stage inside the container, wait for it, and verify
    it actually succeeded (exit code 0 AND no fatal signature in its own log). Returns the log
    dir on success; raises RuntimeError on failure.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    overrides_file = log_dir / "overrides.txt"
    overrides_file.write_text("\n".join(overrides) + "\n")
    log_path = log_dir / ("run_embodiment.log" if run_mode == "train" else "eval_embodiment.log")

    env_vars = {
        "RUN_MODE": run_mode,
        "CONFIG_NAME": config_name,
        "LOG_DIR": str(log_dir),
        "OVERRIDES_FILE": str(overrides_file),
    }
    if run_mode == "eval":
        env_vars["LORA_PATH"] = lora_path

    cmd = _apptainer_cmd(env_vars)
    log(f"launching {run_mode} stage: log_dir={log_dir}")
    result: LaunchResult = launch_and_wait(cmd, log_path=log_path, poll_interval_s=poll_interval_s)

    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    fatal_hit = next((p for p in FATAL_LOG_PATTERNS if p in log_text), None)
    if not result.ok or fatal_hit is not None:
        raise RuntimeError(
            f"{run_mode} stage failed (log_dir={log_dir}, returncode={result.returncode}, "
            f"fatal_signature={fatal_hit!r}) -- see {log_path} for the full log."
        )
    log(f"{run_mode} stage succeeded: log_dir={log_dir}")
    return str(log_dir)


# ---------------------------------------------------------------------------
# run_training / collect_end_states -- the real, apptainer-wrapped injection points
# ---------------------------------------------------------------------------


def make_run_training():
    def run_training(edge: dict, current_checkpoint, config_spec: TrainingConfigSpec) -> str:
        label = f"{config_spec.config_name}-{edge['id']}"
        log_dir = LOGS_ROOT / f"{time.strftime('%Y%m%d-%H:%M:%S')}-{label}"
        write_status(
            phase="training",
            current_edge=edge["id"],
            log_dir=str(log_dir),
            warm_start_from=current_checkpoint,
        )
        _launch_stage_and_verify(
            run_mode="train",
            config_name=config_spec.config_name,
            overrides=config_spec.overrides,
            log_dir=log_dir,
        )
        checkpoint = _checkpoint_path(log_dir, config_spec.config_name, MAX_EPOCHS)
        if not Path(checkpoint).is_dir():
            raise RuntimeError(
                f"training stage for edge {edge['id']!r} exited cleanly but the expected "
                f"checkpoint directory doesn't exist: {checkpoint}"
            )
        write_status(phase="training_done", current_edge=edge["id"], checkpoint=checkpoint)
        return checkpoint

    return run_training


def make_collect_end_states():
    def collect_end_states(edge: dict, checkpoint: str, config_spec: TrainingConfigSpec) -> str:
        label = f"{config_spec.config_name}-{edge['id']}-collect"
        log_dir = LOGS_ROOT / f"{time.strftime('%Y%m%d-%H:%M:%S')}-{label}"
        end_states_path = MANIFEST_DIR / "end_states" / f"{edge['id']}_end_states.jsonl"
        end_states_path.parent.mkdir(parents=True, exist_ok=True)
        end_states_path.unlink(missing_ok=True)

        write_status(
            phase="collecting_end_states",
            current_edge=edge["id"],
            log_dir=str(log_dir),
            checkpoint=checkpoint,
        )
        # config_spec.overrides may already carry a training-time lora_path (the warm-start
        # override). This collection pass needs its own lora_path (passed below via
        # `lora_path=checkpoint`), so the stale one must be stripped first or Hydra raises
        # ConfigCompositionException on the duplicate key.
        overrides = strip_actor_lora_path_override(config_spec.overrides) + [
            f"env.eval.init_params.save_end_state_path={end_states_path}"
        ]
        _launch_stage_and_verify(
            run_mode="eval",
            config_name=config_spec.config_name,
            overrides=overrides,
            log_dir=log_dir,
            lora_path=checkpoint,
        )
        if not end_states_path.is_file() or end_states_path.stat().st_size == 0:
            raise RuntimeError(
                f"end-state collection for edge {edge['id']!r} exited cleanly (per log scan) "
                f"but produced no end-states file: {end_states_path}"
            )
        num_lines = sum(1 for line in end_states_path.open() if line.strip())
        if num_lines == 0:
            raise RuntimeError(
                f"end-state collection for edge {edge['id']!r} produced an empty end-states "
                f"file (0 valid lines): {end_states_path}"
            )
        write_status(
            phase="collecting_end_states_done",
            current_edge=edge["id"],
            end_states_path=str(end_states_path),
            end_states_count=num_lines,
        )
        return str(end_states_path)

    return collect_end_states


# ---------------------------------------------------------------------------
# resume affordance: a prior run trained an edge for real, then failed before its end-states
# were collected and persisted into the manifest. discover_and_train_by_level's "already
# trained, don't retrain" recognition only looks at the manifest, so without this a restart
# would retrain the same edge from scratch, wasting GPU hours already spent.
# ---------------------------------------------------------------------------


def _find_candidate_by_id(candidate_lists: list[list[dict]], edge_id: str) -> Optional[dict]:
    for candidates in candidate_lists:
        for candidate in candidates:
            if candidate["id"] == edge_id:
                return candidate
    return None


def _resume_dangling_trained_edge(
    manifest: Manifest,
    initial_edges_by_node: dict,
    collect_end_states_fn,
) -> None:
    """If STATUS_PATH shows an edge that finished training (has a real ``checkpoint``) but was
    never persisted into ``manifest``, finish it now: collect its end-states and persist it,
    without retraining. A no-op if there's nothing dangling to resume.
    """
    if not STATUS_PATH.exists():
        return
    try:
        status = json.loads(STATUS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        log("resume: driver_status.json exists but couldn't be parsed -- skipping resume check")
        return

    edge_id = status.get("current_edge")
    checkpoint = status.get("checkpoint")
    if not edge_id or not checkpoint:
        return  # nothing was ever fully trained in the run status.json describes
    if manifest.get_edge(edge_id) is not None:
        return  # already fully persisted -- nothing dangling
    if not Path(checkpoint).is_dir():
        log(
            f"resume: status.json names checkpoint {checkpoint!r} for edge {edge_id!r}, but it "
            f"doesn't exist on disk -- NOT resuming (would silently proceed on a phantom "
            f"checkpoint); leaving this for manual inspection."
        )
        return

    edge = _find_candidate_by_id(list(initial_edges_by_node.values()), edge_id)
    if edge is None:
        log(
            f"resume: found a dangling real checkpoint for edge {edge_id!r} "
            f"(checkpoint={checkpoint}) but couldn't find its candidate dict in any seeded "
            f"node's list -- NOT resuming automatically; needs manual intervention (the edge's "
            f"predicate/predicate_args/precondition/produces_node can't be safely guessed)."
        )
        return

    log(
        f"RESUMING: edge {edge_id!r} already trained for real (checkpoint={checkpoint}) in a "
        f"prior run that failed before collection/persistence -- collecting its end-states now, "
        f"WITHOUT retraining."
    )
    write_status(phase="resuming_collection", current_edge=edge_id, checkpoint=checkpoint)

    config_spec = generate_training_config(
        edge,
        reset_states_path=edge.get("reset_states_path"),
        spec_dir=SPEC_DIR,
        config_name=CONFIG_NAME,
        # Inert here (make_collect_end_states strips and substitutes its own lora_path) --
        # passed only so config_spec matches the original training run for log consistency.
        lora_path=status.get("warm_start_from"),
    )
    child_states_path = collect_end_states_fn(edge, checkpoint, config_spec)

    trained_edge = dict(edge)
    trained_edge["checkpoint"] = checkpoint
    manifest.add_edge(trained_edge)
    manifest.set_reset_states_path_for_children(trained_edge["produces_node"], child_states_path)
    log(
        f"RESUME COMPLETE: edge {edge_id!r} persisted to manifest "
        f"(reset_states_path_for_children -> {child_states_path})."
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    write_status(phase="starting", pid=os.getpid(), base_checkpoint=BASE_CHECKPOINT)
    log(f"driver starting, pid={os.getpid()}, base_checkpoint={BASE_CHECKPOINT}")

    if not STAGE_INNER_SCRIPT.is_file():
        raise FileNotFoundError(f"missing shared inner launch script: {STAGE_INNER_SCRIPT}")

    manifest = Manifest(MANIFEST_PATH)
    log(f"manifest loaded: {len(manifest)} edge(s) already present")

    initial_edges_by_node: dict = {}
    if ROOT_CANDIDATES_PATH.exists():
        root_candidates = json.loads(ROOT_CANDIDATES_PATH.read_text())
        initial_edges_by_node[frozenset()] = root_candidates
        log(f"seeded {len(root_candidates)} root candidate(s) from {ROOT_CANDIDATES_PATH}")
    if EDGE1_CHILDREN_PATH.exists():
        edge1_children = json.loads(EDGE1_CHILDREN_PATH.read_text())
        initial_edges_by_node[frozenset({"place_mug_on_cutting_board"})] = edge1_children
        log(
            f"seeded {len(edge1_children)} child candidate(s) for "
            f"{{place_mug_on_cutting_board}} from {EDGE1_CHILDREN_PATH}"
        )

    predicate_module = list_predicate_names()
    predicate_menu = build_predicate_menu()
    log(f"{len(predicate_module)} known predicates loaded")

    try:
        # Must run before discover_and_train_by_level, so its "already trained, don't retrain"
        # recognition picks up any dangling edge during root-frontier processing.
        _resume_dangling_trained_edge(manifest, initial_edges_by_node, make_collect_end_states())

        final_checkpoint = discover_and_train_by_level(
            manifest,
            base_checkpoint=BASE_CHECKPOINT,
            scene_objects=SCENE_OBJECTS,
            logs_root=LOGS_ROOT,
            spec_dir=SPEC_DIR,
            config_name=CONFIG_NAME,
            max_epochs=MAX_EPOCHS,
            predicate_module=predicate_module,
            predicate_menu=predicate_menu,
            stable_base_objects=STABLE_BASE_OBJECTS,
            initial_edges_by_node=initial_edges_by_node,
            run_training=make_run_training(),
            collect_end_states=make_collect_end_states(),
            # Progress logging for the dashboard: writes level_<depth>_queue.json to
            # MANIFEST_DIR as soon as each depth's discovery completes.
            level_queue_dir=MANIFEST_DIR,
        )
    except Exception as exc:
        write_status(phase="FAILED", error=str(exc), traceback=traceback.format_exc())
        log(f"DRIVER FAILED: {exc}")
        log(traceback.format_exc())
        return 1

    write_status(phase="DONE", final_checkpoint=final_checkpoint, total_edges=len(manifest))
    log(f"DRIVER DONE. final_checkpoint={final_checkpoint} total_edges={len(manifest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

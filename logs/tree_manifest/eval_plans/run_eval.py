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

"""Step 2 of the Phase B GPU-eval payoff milestone: launch the two real, GPU evals (one per
Step-1-verified resolved plan) SEQUENTIALLY, each using all 8 GPUs, via orchestrator.py's proven
launch_and_wait() (session-detached, exit-code-file-polled) wrapped in the SAME apptainer +
run_stage_inner.sh container-launch mechanism driver.py already uses for every real
training/eval stage in this project (imported directly from driver.py rather than
re-implemented, so this reuses the exact same proven APPTAINER_BASE_ARGS / SANDBOX_PATH /
FATAL_LOG_PATTERNS).

Verifies each run via BOTH exit code AND a scan of its own log for driver.py's
FATAL_LOG_PATTERNS -- never exit code alone (eval_embodiment.sh's `${CMD} 2>&1 | tee ...`
does not propagate a piped Python process's real exit status; a real, previously-hit bug in
this project). Checks nvidia-smi is back to ~0 MiB on all 8 GPUs before launching the second
job, and again after it, aborting (not launching the second job) if the first one failed for a
real reason.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

RLINF_ROOT = Path("/scratch/cluster/jshim12/RLinf")
sys.path.insert(0, str(RLINF_ROOT))
sys.path.insert(0, str(RLINF_ROOT / "logs" / "tree_manifest"))

from rlinf.envs.isaaclab.tasks.discovery.orchestrator import LaunchResult, launch_and_wait  # noqa: E402

import driver as _driver  # noqa: E402  -- reuse APPTAINER_BASE_ARGS/SANDBOX_PATH/FATAL_LOG_PATTERNS/STAGE_INNER_SCRIPT

FINAL_CHECKPOINT = (
    "/scratch/cluster/jshim12/RLinf/logs/"
    "20260806-11:00:25-generic_single_edge_grpo_openpi_pi05-"
    "place_coke_on_cutting_board__given_place_mug_on_cutting_board/"
    "generic_single_edge_grpo_openpi_pi05/checkpoints/global_step_60/actor"
)
CONFIG_NAME = "generic_eval_grpo_openpi_pi05"
EVAL_PLANS_DIR = RLINF_ROOT / "logs" / "tree_manifest" / "eval_plans"

TASKS = [
    ("mug_then_coke", EVAL_PLANS_DIR / "mug_then_coke.json"),
    ("coke_then_mug", EVAL_PLANS_DIR / "coke_then_mug.json"),
]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    print(f"[{ts}] {msg}", flush=True)


def apptainer_cmd(env_vars: dict) -> str:
    args = list(_driver.APPTAINER_BASE_ARGS)
    for k, v in env_vars.items():
        args += ["--env", f"{k}={v}"]
    args += [_driver.SANDBOX_PATH, "bash", str(_driver.STAGE_INNER_SCRIPT)]
    return " ".join(shlex.quote(a) for a in args)


def gpu_status() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    return out


def gpus_clean(gpu_text: str) -> bool:
    for line in gpu_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        mem = int(parts[1].split()[0])
        if mem > 10:  # small slack for driver bookkeeping noise
            return False
    return True


def run_one(slug: str, plan_path: Path) -> dict:
    if not Path(FINAL_CHECKPOINT).is_dir():
        raise RuntimeError(f"final checkpoint missing: {FINAL_CHECKPOINT}")
    ts = time.strftime("%Y%m%d-%H:%M:%S")
    log_dir = RLINF_ROOT / "logs" / f"{ts}-generic_eval_grpo_openpi_pi05-{slug}"
    log_dir.mkdir(parents=True, exist_ok=True)
    overrides = [f"env.eval.init_params.plan_path={plan_path}"]
    overrides_file = log_dir / "overrides.txt"
    overrides_file.write_text("\n".join(overrides) + "\n")
    log_path = log_dir / "eval_embodiment.log"

    env_vars = {
        "RUN_MODE": "eval",
        "CONFIG_NAME": CONFIG_NAME,
        "LOG_DIR": str(log_dir),
        "OVERRIDES_FILE": str(overrides_file),
        "LORA_PATH": FINAL_CHECKPOINT,
    }
    cmd = apptainer_cmd(env_vars)
    log(f"[{slug}] pre-launch GPU status:\n{gpu_status()}")
    log(f"[{slug}] launching. log_dir={log_dir}")
    log(f"[{slug}] cmd={cmd}")

    result: LaunchResult = launch_and_wait(cmd, log_path=log_path, poll_interval_s=120.0)

    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    fatal_hit = next((p for p in _driver.FATAL_LOG_PATTERNS if p in log_text), None)
    ok = result.ok and fatal_hit is None

    gpu_after = gpu_status()
    log(f"[{slug}] post-run GPU status:\n{gpu_after}")

    metrics = {}
    for line in log_text.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith("eval/") or "eval/subtask_1_success_once" in line_stripped \
           or "eval/subtask_2_success_once" in line_stripped or "eval/success_once" in line_stripped \
           or "eval/final_subtask_idx" in line_stripped or "num_trajectories" in line_stripped:
            metrics.setdefault("raw_metric_lines", []).append(line_stripped)

    r = {
        "slug": slug,
        "log_dir": str(log_dir),
        "log_path": str(log_path),
        "returncode": result.returncode,
        "fatal_hit": fatal_hit,
        "ok": ok,
        "gpus_clean_after": gpus_clean(gpu_after),
        "gpu_status_after": gpu_after,
    }
    log(f"[{slug}] RESULT: {json.dumps(r, indent=2)}")
    return r


def main() -> int:
    results = []
    for i, (slug, plan_path) in enumerate(TASKS):
        if not plan_path.is_file():
            log(f"SKIP {slug}: plan file missing: {plan_path}")
            continue

        if i > 0:
            gpu_text = gpu_status()
            log(f"pre-second-job GPU check:\n{gpu_text}")
            if not gpus_clean(gpu_text):
                log("ABORT: GPUs not clean before launching second job -- NOT launching.")
                break

        r = run_one(slug, plan_path)
        results.append(r)
        if not r["ok"]:
            log(f"STOPPING: {slug} failed (returncode={r['returncode']}, fatal_hit={r['fatal_hit']!r}) "
                f"-- not launching subsequent tasks automatically.")
            break

    log("\n=== STEP 2 SUMMARY ===")
    for r in results:
        log(json.dumps(r, indent=2))

    return 0 if results and all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

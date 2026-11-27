"""Runs a job from curriculum.jobs on the held GPU node over ssh, inside the container. The job's output goes to
<out>/job.log."""
import getpass
import os
import shlex
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def hold_node(job_name="hold"):
    """The node of the user's running Slurm job named `job_name`, or None."""
    out = subprocess.run(
        ["squeue", "-u", getpass.getuser(), "-h", "-n", job_name, "-t", "R", "-o", "%N"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return out[0] if out else None


def _seconds(left):
    """Seconds in a squeue time-left field: [D-]HH:MM:SS, MM:SS or SS; None when unlimited or unknown."""
    if not left or left in ("UNLIMITED", "INVALID", "NOT_SET"):
        return None
    days, _, clock = left.rpartition("-")
    parts = [int(p) for p in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return (int(days) if days else 0) * 86400 + h * 3600 + m * 60 + s


def hold_jobs(job_name="hold"):
    """The user's Slurm jobs named `job_name`: [{"id", "state", "left" (seconds or None), "node"}]."""
    out = subprocess.run(
        ["squeue", "-u", getpass.getuser(), "-h", "-n", job_name, "-o", "%i|%T|%L|%N"],
        capture_output=True, text=True, check=True,
    ).stdout
    jobs = []
    for line in out.splitlines():
        job_id, state, left, node = (line.split("|") + ["", "", "", ""])[:4]
        jobs.append({"id": job_id.strip(), "state": state.strip(), "left": _seconds(left.strip()),
                     "node": node.strip() or None})
    return jobs


def ensure_hold(script, renew_below=3 * 3600, job_name="hold", poll=60, wait=True):
    """Keeps a hold job ahead of its expiry. Submits `script` when no hold is queued and no running hold has
    more than `renew_below` seconds left, then (if `wait`) waits until a hold is running. Returns
    (node of the running hold with the most time left or None, id of the job submitted now or None)."""
    jobs = hold_jobs(job_name)
    running = [j for j in jobs if j["state"] == "RUNNING"]
    queued = [j for j in jobs if j["state"] == "PENDING"]
    submitted = None
    if not queued and not any(j["left"] is None or j["left"] > renew_below for j in running):
        out = subprocess.run(["sbatch", "--parsable", str(script)], capture_output=True, text=True, check=True)
        submitted = out.stdout.strip().split(";")[0]
    while wait and not running:
        time.sleep(poll)
        running = [j for j in hold_jobs(job_name) if j["state"] == "RUNNING"]
    best = max(running, key=lambda j: float("inf") if j["left"] is None else j["left"], default=None)
    return (best["node"] if best else None), submitted


def command(job, repo=REPO, venv=None, gpus="0,1,2,3"):
    """The shell command that runs `job` on the node."""
    q = shlex.quote
    out = job["out"]
    run = ["bash", "setup/horizon/enter_container.sh", "bash", job["script"], job["config"], *job["overrides"]]
    env = f"CUDA_VISIBLE_DEVICES={gpus}" + (f" R2S2R_VENV={q(str(venv))}" if venv else "")
    return (f"cd {q(str(repo))} && mkdir -p {q(out)} && "
            f"env {env} {' '.join(q(a) for a in run)} > {q(os.path.join(out, 'job.log'))} 2>&1")


def run(job, node, repo=REPO, venv=None, gpus="0,1,2,3", timeout=None):
    """Runs `job` on `node` and returns its exit code."""
    return subprocess.run(["ssh", node, command(job, repo, venv, gpus)], timeout=timeout).returncode

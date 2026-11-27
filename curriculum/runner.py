"""Runs a job from curriculum.jobs on the held GPU node over ssh, inside the container. The job's output goes to
<out>/job.log."""
import getpass
import os
import shlex
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def hold_node(job_name="hold"):
    """The node of the user's running Slurm job named `job_name`, or None."""
    out = subprocess.run(
        ["squeue", "-u", getpass.getuser(), "-h", "-n", job_name, "-t", "R", "-o", "%N"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return out[0] if out else None


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

# Portable setup (apptainer-based, host-OS-agnostic)

This document reproduces the exact working environment this project has been developed in —
RLinf + RoboLab (mug/coke/cutting-board scene) + the VLM-discovered subtask tree extension — on
**any machine that has [Apptainer](https://apptainer.org/) installed**, regardless of the host
OS. The host being RedHat, Ubuntu, or anything else doesn't matter: everything that cares about
OS (RoboLab/Isaac Sim need Ubuntu 22.04) runs *inside* an Apptainer sandbox built from a plain
`ubuntu:22.04` Docker base, not on the host directly.

Two layers, kept deliberately separate:

- **OS layer** — a single Apptainer sandbox (`rlinf_sandbox/`), built once from
  [`setup/rlinf_u22.def`](setup/rlinf_u22.def). Ubuntu 22.04 + system packages (git, build tools,
  Vulkan/GL libs, ffmpeg) + `uv`. This is the only part that's actually host-OS-dependent, and
  Apptainer neutralizes that entirely.
- **Project layer** — this repo, RoboLab, Isaac Sim, model checkpoints, and the Python venv, all
  living on the **host** filesystem (bind-mounted into the sandbox at runtime, not baked into it).
  This is what `setup/bootstrap.sh` provisions.

## Prerequisites

- Apptainer/Singularity installed on the host, with `--nv` GPU passthrough working (NVIDIA driver
  + `nvidia-container-cli` or equivalent already set up by your sysadmin — this project doesn't
  install the host driver).
- `apptainer build` needs either `--fakeroot` privileges (ask your sysadmin to add you to
  `/etc/subuid` / `/etc/subgid` if `apptainer build --fakeroot ...` fails) or access to
  `apptainer build --remote` (a free [Sylabs Cloud](https://cloud.sylabs.io/) account). This is
  the one step that's genuinely host/cluster-policy-dependent and can't be scripted around.
- Outbound network access to: Docker Hub (ubuntu base image), GitHub, Hugging Face,
  `download.isaacsim.omniverse.nvidia.com`, `developer.download.nvidia.com`, and (if you use the
  gsutil checkpoint route) `storage.googleapis.com`.
- **~60GB free disk**: sandbox (~1.5GB) + CUDA toolkit (~7GB) + Isaac Sim (~15GB) + PolaRiS-Hub
  (<2GB) + model checkpoint (~7GB) + Python venv/build artifacts (~10-15GB).
- An OpenAI (or Anthropic) API key if you plan to run the VLM-discovery pipeline
  (`rlinf/envs/isaaclab/tasks/discovery/`) — see [OpenAI key](#openai-key) below.

## Quickstart

```bash
git clone https://github.com/ShimBoi/R2S2R.git   # or wherever this repo lives
cd R2S2R/RLinf
bash setup/bootstrap.sh all
```

`bootstrap.sh` is organized into independent, idempotent stages — safe to re-run, and safe to
resume if one stage fails partway (network hiccup, disk fills up, etc.). Run a single stage with
`bash setup/bootstrap.sh <stage>`, or see everything it can do with `bash setup/bootstrap.sh
--help`. Stages, in order:

| Stage | What it does | Roughly how long |
|---|---|---|
| `sandbox` | `apptainer build --sandbox` from `setup/rlinf_u22.def` | 5-10 min |
| `cuda` | Downloads + extracts the CUDA 12.4 toolkit (user-space, no root, no driver) | 5-10 min |
| `vulkan-icd` | Writes the tiny `vulkan_icd/nvidia_icd.json` ICD manifest | instant |
| `robolab` | `git clone`s RoboLab (NVlabs/RoboLab) as `RLinf/RoboLab/` | 2-5 min |
| `isaac-sim` | Downloads + unzips the Isaac Sim 5.1.0 standalone build into `RLinf/isaac_sim/` | 15-30 min |
| `polaris-hub` | `hf download`s the PolaRiS-Hub scene/asset dataset | 5-10 min |
| `install` | Runs `requirements/install.sh embodied --model openpi --env isaaclab` **inside** the sandbox | 20-40 min |
| `model` | Downloads + converts the `pi05_droid_jointpos` checkpoint (needs `install` first, for the venv) | 15-30 min |
| `enter-script` | Generates `enter_container.sh` (this machine's equivalent of `enter_robolab.sh`) | instant |

Not automated — genuinely needs a human and a credential, see [step 9](#openai-key):

- The `.env` file holding your `OPENAI_API_KEY`.
- Whatever the *actual* GRPO/training checkpoints from your prior runs are (those live under
  `RLinf/logs/**/checkpoints/`, are not part of this repo, and are yours to `rsync`/`scp` over
  from the old machine if you want to resume from them rather than retrain from scratch).

## Step-by-step (what `bootstrap.sh` actually runs)

If a stage fails, or you want to understand/adapt what it's doing, here's the same sequence by
hand.

### 1. Build the OS sandbox

```bash
cd RLinf/setup
apptainer build --fakeroot --sandbox ../../rlinf_sandbox rlinf_u22.def
# or, if your cluster doesn't grant --fakeroot:
#   apptainer build --remote --sandbox ../../rlinf_sandbox rlinf_u22.def
```

This pulls `ubuntu:22.04` from Docker Hub and installs system packages + `uv` — see
[`setup/rlinf_u22.def`](setup/rlinf_u22.def) for the exact list. Nothing project-specific lives in
here; it's the same sandbox regardless of which machine or which branch of this repo you're on.

### 2. CUDA 12.4 toolkit (user-space, no root)

```bash
mkdir -p cuda && cd cuda
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sh cuda_12.4.0_550.54.14_linux.run --silent --toolkit --installpath="$(pwd)/cuda-12.4" --override --no-opengl-libs
rm cuda_12.4.0_550.54.14_linux.run
```

That exact runfile URL can rot — if it 404s, get the current one from NVIDIA's
[CUDA 12.4.0 archive page](https://developer.nvidia.com/cuda-12-4-0-download-archive) (choose
Linux → x86_64 → your distro → runfile (local)) and substitute it. `--toolkit` installs only the
toolkit (headers, libs, nvcc), not the driver — you're not touching host root either way.

### 3. Vulkan ICD manifest

```bash
mkdir -p vulkan_icd
cat > vulkan_icd/nvidia_icd.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.242"
    }
}
EOF
```

Bind-mounted into the sandbox at `/etc/vulkan/icd.d/nvidia_icd.json` by the entry script. Belt-and-suspenders from an earlier debugging pass — see [Troubleshooting](#troubleshooting) if you ever hit "No device could be created" / Vulkan `vk_icdGetInstanceProcAddr` failures again; last time this turned out to be a bad cluster node, not a real config problem, so don't assume it's this file first.

### 4. RoboLab

```bash
git clone https://github.com/NVlabs/RoboLab.git RLinf/RoboLab
```

Don't run RoboLab's own `uv sync` — this project drives RoboLab's task code through **RLinf's own
IsaacLab fork** (`github.com/RLinf/IsaacLab`, installed by step 7 below via `--env isaaclab`), not
RoboLab's native Isaac Sim 5.0 / IsaacLab 2.2.0 pin. RoboLab is used here purely as a sibling
directory of task/asset code (`RoboLab/robolab/tasks/benchmark/`,
`RoboLab/robolab/core/task/conditionals.py`, `RoboLab/assets/`) that `rlinf/envs/isaaclab/`
imports against at runtime.

### 5. Isaac Sim

```bash
mkdir -p RLinf/isaac_sim && cd RLinf/isaac_sim
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
rm isaac-sim-standalone-5.1.0-linux-x86_64.zip
```

Public download, no NGC login needed. `enter_robolab.sh`'s successor (`enter_container.sh`,
generated in step 8) sources `isaac_sim/setup_conda_env.sh` on every shell entry — that's not
optional, do it every time, not just once.

`bash requirements/install.sh embodied --env isaaclab` (step 7) *also* clones and installs
`github.com/RLinf/IsaacLab` via its own `isaaclab.sh --install`, which may provision its own copy
of Isaac Sim into the venv. Run step 7 first; only do this manual zip download if
`RLinf/isaac_sim/setup_conda_env.sh` doesn't already exist afterward.

### 6. PolaRiS-Hub (scene/asset dataset)

```bash
pip install --user huggingface-hub  # if `hf` isn't already on PATH
hf download owhan/PolaRiS-Hub --repo-type=dataset --local-dir RLinf/PolaRiS-Hub
```

### 7. Install RLinf + model deps, inside the sandbox

```bash
apptainer exec --nv --bind "$(pwd)":"$(pwd)" ../rlinf_sandbox/ bash -c '
    cd RLinf
    bash requirements/install.sh embodied --model openpi --env isaaclab
'
```

### 8. Model checkpoint

This project's configs reference `model.model_path: "./model/pi05_droid_jointpos"` — the openpi
π0.5 DROID-joint-position checkpoint, **not** RoboLab's or PolaRiS's own pretrained checkpoints.
Confirmed source: `gs://openpi-assets/checkpoints/pi05_droid_jointpos` (a public bucket — no GCP
account needed for read access). `bootstrap.sh`'s `model` stage automates this end to end
(download raw JAX checkpoint → convert to PyTorch via `openpi`'s own conversion script → place at
the exact path configs expect); by hand it's:

```bash
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_droid_jointpos RLinf/model/pi05_droid_jointpos.raw
git clone --depth 1 https://github.com/RLinf/openpi /tmp/openpi-src   # convert script isn't in the installed wheel
RLinf/.venv/bin/python /tmp/openpi-src/examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir RLinf/model/pi05_droid_jointpos.raw \
    --config_name pi05_droid_jointpos \
    --output_path RLinf/model/pi05_droid_jointpos
cp -r RLinf/model/pi05_droid_jointpos.raw/assets RLinf/model/pi05_droid_jointpos/
rm -rf RLinf/model/pi05_droid_jointpos.raw
```

`gsutil` needs the Google Cloud SDK (`pip install gsutil` is enough for this). The
`RLinf-Pi05-Polaris-droid_jointpos` checkpoint (`hf download RLinf/RLinf-Pi05-Polaris-droid_jointpos`)
is a *different*, PolaRiS-tuned checkpoint present in the old machine's `model/` directory but not
referenced by any config this project's tree-extension work actually uses — skip it unless you
specifically need it.

### 9. `.env` (OpenAI key) {#openai-key}

```bash
cat > .env <<'EOF'
OPENAI_API_KEY=sk-...
EOF
```

Never commit this file (it isn't — the outer workspace here isn't a git repo at all, and even if
it were, it'd need a `.gitignore` entry). `discovery/vlm_client.py` reads it via
`set -a; source .env; set +a` before any real VLM call — see that module's docstring. Cost-
sensitive: don't source it into a shell that's going to call `discover_tree()` / `resolve_plan()`
speculatively.

### 10. Entry script

`bootstrap.sh`'s `enter-script` stage writes `enter_container.sh` at the workspace root
(one level above `RLinf/`), parameterized to the actual paths it just set up — the same shape as
this project's own `enter_robolab.sh`, minus the hardcoded `/scratch/cluster/jshim12` paths.
Always enter the project through it, not `apptainer exec` by hand:

```bash
bash enter_container.sh
```

## Verification

Inside the container (`bash enter_container.sh` drops you into a shell already `cd`'d into
`RLinf/` with the venv active):

```bash
# Fast, no-GPU sanity check of the discovery package (import-light by design):
uv run pytest tests/unit_tests/test_discovery_*.py -q

# Cheapest real env-construction smoke test (skips Ray/the full trainer stack):
python -m rlinf.envs.isaaclab.tasks.robolab_task
```

The second one is the fastest way to find out whether Isaac Sim / Vulkan / the RoboLab scene are
actually wired up correctly before attempting a real multi-hour training run.

## Troubleshooting

- **Vulkan "No device could be created" / `vk_icdGetInstanceProcAddr` failures**: the last time
  this happened on the original machine, it was eventually root-caused to a bad cluster node, not
  a real driver/env misconfiguration — confirmed by the fact that even a minimal raw ctypes Vulkan
  probe hit the identical failure. Try a different node/allocation before re-debugging the ICD
  binding or `glibc_malloc_hook_shim`.
- **glibc malloc-hook symbol warnings** (`__malloc_hook` etc. undefined, or an Xorg `ErrorF`
  symbol): harmless on modern glibc (>=2.34 removed these hooks) but can produce noisy warnings
  when the host NVIDIA driver's `libnvidia-glcore.so` expects them. Optional fix, not required for
  correctness: `gcc -shared -fPIC -o setup/libmalloc_hook_shim.so
  <see setup/glibc_malloc_hook_shim.c>`, then `export
  LD_PRELOAD=$(pwd)/setup/libmalloc_hook_shim.so` before running IsaacLab/Isaac Sim commands.
- **`apptainer build` permission errors**: see the `--fakeroot` / `--remote` note under
  [Prerequisites](#prerequisites) — this is the one step that depends on how your specific
  cluster is configured, not something this repo can paper over.
- **Silent "success" that isn't**: `run_embodiment.sh`/`eval_embodiment.sh` piping through `tee`
  means a real Python traceback can still exit 0. Always grep the actual log for `Traceback` in
  addition to checking the exit code — several real bugs in this project's history were exactly
  this pattern.

#!/bin/bash
# Builds the venv (Isaac Sim 6.2 + Isaac Lab 3.0 + RLinf + openpi) for this branch.
# Run INSIDE the container: bash setup/horizon/enter_container.sh bash setup/horizon/install.sh
# Idempotent-ish: safe to re-run.
set -euo pipefail

R=${R2S2R_REPO:?run this through setup/horizon/enter_container.sh}
V=${R2S2R_VENV:?run this through setup/horizon/enter_container.sh}
W=${R2S2R_WORKSPACE:-$SCRATCH/r2s2r}
IL=$W/IsaacLab
export UV_CACHE_DIR=$W/.cache/uv
export UV_PYTHON_INSTALL_DIR=$W/.cache/uv-python
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=600
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y

[ -d "$IL/.git" ] || git clone --depth 1 -b v3.0.0 https://github.com/isaac-sim/IsaacLab "$IL"

# 1. Isaac Lab 3.0 + Isaac Sim 6.2 from Isaac Lab's own lockfile (torch 2.11+cu130, numpy 2).
cd "$IL"
UV_PROJECT_ENVIRONMENT="$V" uv sync --frozen --python 3.12 --extra isaacsim --extra rlinf

# Everything below runs from / so this repo's pyproject (torch==2.6.0 override) never applies.
cd /
PIP="uv pip install --python $V/bin/python"

# 2. RLinf itself (editable) + its embodied extras, without letting it move torch.
cat > $W/il3_constraints.txt <<EOF
torch==2.11.0
torchvision==0.26.0
numpy>=2
EOF
$PIP -c $W/il3_constraints.txt --no-deps -e "$R"
$PIP -c $W/il3_constraints.txt \
    "ray[default]>=2.47.0" "hydra-core==1.3.2" "datasets==3.6.0" torchdata scipy accelerate debugpy einops nvitop \
    pybind11 ninja pytest gsutil huggingface_hub icmplib "wandb>=0.19,<0.25.1" tensorboard \
    sortedcontainers peft timm "imageio[ffmpeg]" gymnasium gym omegaconf
$PIP -c $W/il3_constraints.txt -r "$R/requirements/embodied/envs/common.txt" || true

# 3. openpi (RLinf fork). Its torch 2.7.1 / numpy<2 / jax pins conflict with Isaac Lab 3, so install
#    it --no-deps and add only the runtime deps the PyTorch path needs.
$PIP --no-deps "git+https://github.com/RLinf/openpi"
$PIP --no-deps "openpi-client @ git+https://github.com/RLinf/openpi#subdirectory=packages/openpi-client"
$PIP -c $W/il3_constraints.txt \
    "transformers==4.53.2" sentencepiece "tyro>=0.9.5" "jaxtyping==0.2.36" "beartype==0.19.0" \
    ml_collections==1.0.0 "flax==0.10.2" "jax==0.5.3" "orbax-checkpoint==0.11.13" augmax dm-tree \
    equinox flatbuffers "fsspec[gcs]" imageio opencv-python-headless numpydantic tqdm-loggable \
    treescope rich polars "chex==0.1.90" safetensors msgpack websockets
# openpi.training.data_loader imports lerobot (openpi's pinned rev); no-deps so it can't move torch.
$PIP --no-deps "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
# datasets 3.6 (RLinf's pin; newer ones need huggingface_hub>=1, which transformers 4.53 forbids)
# caps fsspec at 2025.3, but torch 2.11/gcsfs need >=2026.6 -- override the cap. Plus lerobot's imports.
echo 'fsspec>=2026.6.0' > $W/fsspec_override.txt
$PIP --override $W/fsspec_override.txt "datasets==3.6.0" "fsspec>=2026.6.0" jsonlines deepdiff
SP=$($V/bin/python -c 'import site; print(site.getsitepackages()[0])')
cp -r "$SP/openpi/models_pytorch/transformers_replace/"* "$SP/transformers/"

# 4. Isaac Sim's aarch64 LD_PRELOAD check spawns a multiprocessing child, which crashes inside
#    RLinf's daemonic env worker (rlinf/envs/isaaclab/venv.py). Run it in-process there instead.
ISINIT="$SP/isaacsim/__init__.py"
grep -q "R2S2R daemon patch" "$ISINIT" || $V/bin/python - "$ISINIT" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "        queue = multiprocessing.Queue()\n        process = multiprocessing.Process("
new = ("        if multiprocessing.current_process().daemon:  # R2S2R daemon patch\n"
       "            import queue as _q\n"
       "            q = _q.SimpleQueue()\n"
       "            _aarch_preload_checking(q)\n"
       "            msg = q.get()\n"
       "            if msg:\n"
       "                sys.exit(msg)\n"
       "            return\n" + old)
assert old in s, "isaacsim __init__ layout changed"
open(p, "w").write(s.replace(old, new, 1))
PY

# 5. openpi's PaliGemma tokenizer; OPENPI_DATA_HOME points here (enter_container.sh).
[ -f $W/.cache/openpi/big_vision/paligemma_tokenizer.model ] || \
    $V/bin/hf download RLinf/openpi_tokenizer --local-dir $W/.cache/openpi

touch "$V/.install_done"
echo "[install] done: $V"

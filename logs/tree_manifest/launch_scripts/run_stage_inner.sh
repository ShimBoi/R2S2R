#!/bin/bash
# Shared inner script for every real training/collection stage the driver launches --
# written ONCE (not regenerated per edge) to avoid re-deriving the container setup boilerplate
# (and its nested-quoting risk, see launch_edge1_inner.sh's history) for every one of
# potentially many stages. All per-stage specifics come in via environment variables set by
# apptainer's --env (never string-interpolated into this file), and via OVERRIDES_FILE (one
# Hydra CLI override per line -- read with `mapfile` into an array so paths are never subject
# to word-splitting/quoting issues, however many there are or whatever characters they contain).
#
# Required env vars:
#   RUN_MODE        "train" or "eval"
#   CONFIG_NAME      Hydra config name (examples/embodiment/config/<CONFIG_NAME>.yaml)
#   LOG_DIR          passed straight through to run_embodiment.sh/eval_embodiment.sh
#   OVERRIDES_FILE   path to a file with one Hydra override per line
# Additional for RUN_MODE=eval:
#   LORA_PATH        checkpoint dir for +actor.model.lora_path (LoRA adapter, NOT runner.ckpt_path
#                     -- torch.load()-ing a LoRA adapter directory as runner.ckpt_path crashes
#                     with IsADirectoryError; confirmed by a real failed run, see CLAUDE.md /
#                     the coordinator report for this stage)
set -e

source /scratch/cluster/jshim12/RLinf/isaac_sim/setup_conda_env.sh
unset PYTHONHOME
unset VK_LOADER_DEBUG
export PATH="/scratch/cluster/jshim12/RLinf/.venv/bin:/scratch/cluster/jshim12/cuda/cuda-12.4/bin:$PATH"
export VIRTUAL_ENV=/scratch/cluster/jshim12/RLinf/.venv
export LD_LIBRARY_PATH=/.singularity.d/libs:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
cd /scratch/cluster/jshim12/RLinf

echo "=== run_stage_inner.sh starting: RUN_MODE=$RUN_MODE CONFIG_NAME=$CONFIG_NAME LOG_DIR=$LOG_DIR ==="

mapfile -t OVERRIDES < "$OVERRIDES_FILE"
echo "OVERRIDES (${#OVERRIDES[@]}):"
printf '  %s\n' "${OVERRIDES[@]}"

if [ "$RUN_MODE" = "train" ]; then
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    bash examples/embodiment/run_embodiment.sh "$CONFIG_NAME" "${OVERRIDES[@]}"
elif [ "$RUN_MODE" = "eval" ]; then
    export MUJOCO_GL=osmesa
    export PYOPENGL_PLATFORM=osmesa
    bash examples/embodiment/eval_embodiment.sh "$CONFIG_NAME" "+actor.model.lora_path=${LORA_PATH}" "${OVERRIDES[@]}"
else
    echo "unknown RUN_MODE: $RUN_MODE" >&2
    exit 1
fi

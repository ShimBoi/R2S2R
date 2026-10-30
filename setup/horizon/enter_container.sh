#!/bin/bash
# Enter the container on TACC Horizon (GB200 / aarch64) with the Isaac Sim 6.2 / Isaac Lab 3.0 venv.
# No args: interactive shell. With args: runs that command inside, e.g.
#   bash setup/horizon/enter_container.sh bash examples/embodiment/eval_embodiment.sh robolab_mug_grpo_openpi_pi05
#
# Paths (override via env):
#   R2S2R_WORKSPACE  holds the container sandbox and caches   (default: $SCRATCH/r2s2r)
#   R2S2R_VENV       the Python venv built by install.sh       (default: <repo>/.venv-il3)
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
W=${R2S2R_WORKSPACE:-$SCRATCH/r2s2r}
V=${R2S2R_VENV:-$R/.venv-il3}
module load tacc-apptainer >/dev/null 2>&1
# TACC's XALT tracker injects a bind path that doesn't exist in the container.
unset LD_PRELOAD SINGULARITY_BINDPATH APPTAINER_BINDPATH SINGULARITYENV_LD_PRELOAD APPTAINERENV_LD_PRELOAD OMP_NUM_THREADS

# Driver libs apptainer's --nv list doesn't know about (GB200 needs rmapi-tegra for Vulkan;
# NGX/OptiX for RTX denoising). Horizon nodes run different drivers, so bind this node's version.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
EXTRA_BINDS=()
for lib in libnvidia-rmapi-tegra libnvidia-tileiras libnvidia-present libnvidia-ngx libnvidia-api \
           libnvidia-allocator libnvidia-sandboxutils; do
    f=$(ls /usr/lib64/$lib.so.$DRV 2>/dev/null | head -1) || true
    [ -n "$f" ] && EXTRA_BINDS+=(--bind "$f:/usr/lib/aarch64-linux-gnu/$(basename "$f")" \
                                 --bind "$f:/usr/lib/aarch64-linux-gnu/$lib.so.1")
done

# Driver 610 can't create DLSS Ray Reconstruction, which Isaac Sim's default renderer needs
# (images come out as noise). Path tracing + the OptiX denoiser avoids NGX; it is ~2.3x slower
# to render. Don't compare evals across renderers. Set R2S2R_CARB_SETTINGS='{}' to opt out.
if [[ $DRV == 610.* && -z ${R2S2R_CARB_SETTINGS+x} ]]; then
    export R2S2R_CARB_SETTINGS='{"/rtx/rendermode":"PathTracing","/rtx/pathtracing/spp":8,"/rtx/pathtracing/totalSpp":8,"/rtx/pathtracing/optixDenoiser/enabled":true,"/rtx/post/aa/op":0}'
    echo "[enter_container] driver $DRV: using PathTracing + OptiX (R2S2R_CARB_SETTINGS)" >&2
fi

# srun/ssh shells may be pinned to one core; give the container the whole allocation.
exec taskset -c "$(cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/cpuset.cpus.effective)" \
apptainer exec --nv \
    "${EXTRA_BINDS[@]}" \
    --bind /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin \
    --bind "$W:$W" --bind "$R:$R" --bind "$V:$V" \
    --bind "$R/setup/horizon/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json" \
    --env CC=gcc --env CXX=g++ --env FC=gfortran --env F77=gfortran --env F90=gfortran \
    --env CARB_LOG_LEVEL=error \
    --env PYTHONWARNINGS=ignore \
    --env CPATH="" --env C_INCLUDE_PATH="" --env CPLUS_INCLUDE_PATH="" \
    --env TORCH_EXTENSIONS_DIR=$W/.cache/torch_extensions_il3 \
    --env HF_HOME=$W/.cache/huggingface \
    --env OPENPI_DATA_HOME=$W/.cache/openpi \
    --env XDG_CACHE_HOME=$W/.cache \
    --env UV_CACHE_DIR=$W/.cache/uv \
    --env ACCEPT_EULA=Y --env OMNI_KIT_ACCEPT_EULA=YES \
    --env R2S2R_REPO="$R" --env R2S2R_VENV="$V" \
    ${R2S2R_CARB_SETTINGS:+--env R2S2R_CARB_SETTINGS="$R2S2R_CARB_SETTINGS"} \
    "$W/rlinf_sandbox/" \
    bash -c '
        R=$R2S2R_REPO; V=$R2S2R_VENV
        unset PYTHONHOME
        # Drop host (TACC module) dirs from PATH; they hold Rocky-10/nvc binaries.
        export PATH="$V/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        unset LD_LIBRARY_PATH MPICC MPICXX CUDA_HOME CUDA_PATH TACC_CUDA_DIR
        export VIRTUAL_ENV=$V
        # Point Isaac Sim at the pip install; run_embodiment.sh otherwise exports placeholders.
        export ISAAC_PATH=$V/lib/python3.12/site-packages/isaacsim
        export EXP_PATH=$ISAAC_PATH/apps
        export CARB_APP_PATH=$ISAAC_PATH/kit
        # This checkout first, so it wins over whatever RLinf the venv was installed from.
        export PYTHONPATH="$R:$R/RoboLab${PYTHONPATH:+:$PYTHONPATH}"
        export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/.singularity.d/libs:$V/lib/python3.12/site-packages/nvidia/cu13/lib
        # Isaac Sim on aarch64 refuses to start unless libgomp is preloaded (system copy + the torch copy).
        export LD_PRELOAD="$(ls /lib/aarch64-linux-gnu/libgomp.so.1 $V/lib/python3.12/site-packages/torch.libs/libgomp* 2>/dev/null | paste -sd:)"
        cd $R
        if [ $# -gt 0 ]; then exec "$@"; else exec bash --norc --noprofile; fi
    ' enter_container "$@"

#!/bin/bash
# Isaac Lab 3.0 / Isaac Sim 6.2 counterpart of enter_container.sh (port/isaaclab3 branch).
# No args: interactive shell. With args: runs that command inside, e.g.
#   bash enter_container_il3.sh python -m rlinf.envs.isaaclab.tasks.robolab_task
# DLSS (NGX) and OptiX denoiser data aren't in apptainer's --nv library list; without them
# RTX renders come out unfiltered/speckled.
NGX_LIB=$(readlink -f /usr/lib64/libnvidia-ngx.so.1)
apptainer exec --nv \
    --bind "$NGX_LIB":/usr/lib/x86_64-linux-gnu/libnvidia-ngx.so.1 \
    --bind /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin \
    --bind /work2/11369/jshim12/stampede3:/work2/11369/jshim12/stampede3 \
    --bind /work2/11369/jshim12/stampede3/vulkan_icd/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json \
    --env CARB_LOG_LEVEL=error \
    --env PYTHONWARNINGS=ignore \
    --env CPATH="" \
    --env C_INCLUDE_PATH="" \
    --env CPLUS_INCLUDE_PATH="" \
    --env TORCH_EXTENSIONS_DIR=/work2/11369/jshim12/stampede3/.cache/torch_extensions_il3 \
    --env ACCEPT_EULA=Y \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    /work2/11369/jshim12/stampede3/rlinf_sandbox/ \
    bash -c '
        unset PYTHONHOME
        export PATH="/work2/11369/jshim12/stampede3/R2S2R/.venv-il3/bin:$PATH"
        export VIRTUAL_ENV=/work2/11369/jshim12/stampede3/R2S2R/.venv-il3
        # Point Isaac Sim at the pip install; run_embodiment.sh otherwise exports /path/to/isaac-sim placeholders.
        export ISAAC_PATH=/work2/11369/jshim12/stampede3/R2S2R/.venv-il3/lib/python3.12/site-packages/isaacsim
        export EXP_PATH=$ISAAC_PATH/apps
        export CARB_APP_PATH=$ISAAC_PATH/kit
        export PYTHONPATH="/work2/11369/jshim12/stampede3/R2S2R:/work2/11369/jshim12/stampede3/R2S2R/RoboLab${PYTHONPATH:+:$PYTHONPATH}"
        export LD_LIBRARY_PATH=/.singularity.d/libs:/usr/lib/x86_64-linux-gnu:/work2/11369/jshim12/stampede3/R2S2R/.venv-il3/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
        cd /work2/11369/jshim12/stampede3/R2S2R
        if [ $# -gt 0 ]; then exec "$@"; else exec bash --norc --noprofile; fi
    ' enter_container "$@"

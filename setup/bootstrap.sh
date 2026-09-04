#!/bin/bash
# Portable, host-OS-agnostic environment bootstrap. See ../SETUP.md for the narrative version of
# every stage below -- this script automates everything that's mechanical/deterministic; a few
# steps (model checkpoint source, .env secrets) are deliberately left to SETUP.md's manual section
# since they need a human decision or a credential this script has no business holding.
#
# Run from inside this repo's checkout, e.g.:
#   cd RLinf && bash setup/bootstrap.sh all
#
# Every stage is idempotent (safe to re-run) and independently invocable:
#   bash setup/bootstrap.sh sandbox
#   bash setup/bootstrap.sh cuda
#   ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLINF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$RLINF_DIR/.." && pwd)"

# ---- Configurable knobs (override via env before invoking, e.g. `CUDA_VERSION=12.6.0 ... `) ----
CUDA_VERSION="${CUDA_VERSION:-12.4.0}"
CUDA_DRIVER_BUILD="${CUDA_DRIVER_BUILD:-550.54.14}"
ISAAC_SIM_VERSION="${ISAAC_SIM_VERSION:-5.1.0}"
ROBOLAB_GIT_URL="${ROBOLAB_GIT_URL:-https://github.com/NVlabs/RoboLab.git}"
POLARIS_HUB_HF_REPO="${POLARIS_HUB_HF_REPO:-owhan/PolaRiS-Hub}"
SANDBOX_DIR="${SANDBOX_DIR:-$WORKSPACE_ROOT/rlinf_sandbox}"
CUDA_DIR="${CUDA_DIR:-$WORKSPACE_ROOT/cuda}"

STAGES=(sandbox cuda vulkan-icd robolab isaac-sim polaris-hub install enter-script)

log() { echo "[bootstrap] $*"; }
banner() { echo; echo "=== $* ==="; }

print_help() {
    cat <<EOF
Usage: bash setup/bootstrap.sh <stage|all> [more stages...]

Stages (run in this order for a fresh machine): ${STAGES[*]}

Env overrides:
  CUDA_VERSION (default $CUDA_VERSION), CUDA_DRIVER_BUILD (default $CUDA_DRIVER_BUILD)
  ISAAC_SIM_VERSION (default $ISAAC_SIM_VERSION)
  ROBOLAB_GIT_URL (default $ROBOLAB_GIT_URL)
  POLARIS_HUB_HF_REPO (default $POLARIS_HUB_HF_REPO)
  SANDBOX_DIR (default $SANDBOX_DIR), CUDA_DIR (default $CUDA_DIR)

Not covered here -- see SETUP.md's manual section:
  model checkpoint download, .env (OPENAI_API_KEY).
EOF
}

# ---------------------------------------------------------------------------
stage_sandbox() {
    banner "Building Apptainer sandbox ($SANDBOX_DIR)"
    if [ -d "$SANDBOX_DIR" ] && [ -f "$SANDBOX_DIR/etc/os-release" ]; then
        log "sandbox already exists at $SANDBOX_DIR, skipping (delete it to force a rebuild)"
        return 0
    fi
    if ! command -v apptainer >/dev/null 2>&1; then
        echo "apptainer not found on PATH -- install it first (see SETUP.md prerequisites)." >&2
        exit 1
    fi
    local build_flag="--fakeroot"
    log "attempting: apptainer build $build_flag --sandbox $SANDBOX_DIR $SCRIPT_DIR/rlinf_u22.def"
    if ! apptainer build $build_flag --sandbox "$SANDBOX_DIR" "$SCRIPT_DIR/rlinf_u22.def"; then
        log "--fakeroot build failed -- if your cluster doesn't grant fakeroot privileges, try:"
        log "  apptainer build --remote --sandbox $SANDBOX_DIR $SCRIPT_DIR/rlinf_u22.def"
        exit 1
    fi
}

stage_cuda() {
    banner "CUDA $CUDA_VERSION toolkit (user-space, no root, no driver)"
    local install_path="$CUDA_DIR/cuda-${CUDA_VERSION%.*}"
    # e.g. cuda-12.4 for CUDA_VERSION=12.4.0 -- matches this project's existing CUDA_HOME convention.
    if [ -f "$install_path/version.json" ]; then
        log "already present at $install_path, skipping"
        return 0
    fi
    mkdir -p "$CUDA_DIR"
    local runfile="cuda_${CUDA_VERSION}_${CUDA_DRIVER_BUILD}_linux.run"
    local url="https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/${runfile}"
    log "downloading $url"
    if ! wget -q --show-progress -O "$CUDA_DIR/$runfile" "$url"; then
        echo "Download failed -- the exact runfile URL/version pairing can rot." >&2
        echo "Get the current one from https://developer.nvidia.com/cuda-downloads (archive:" >&2
        echo "https://developer.nvidia.com/cuda-${CUDA_VERSION//./-}-download-archive ), then set" >&2
        echo "CUDA_VERSION/CUDA_DRIVER_BUILD to match, or download it manually into $CUDA_DIR/." >&2
        exit 1
    fi
    log "installing toolkit only to $install_path"
    sh "$CUDA_DIR/$runfile" --silent --toolkit --installpath="$install_path" --override --no-opengl-libs
    rm -f "$CUDA_DIR/$runfile"
    log "done: $install_path"
}

stage_vulkan_icd() {
    banner "Vulkan ICD manifest"
    local dir="$WORKSPACE_ROOT/vulkan_icd"
    mkdir -p "$dir"
    cat > "$dir/nvidia_icd.json" <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.242"
    }
}
EOF
    log "wrote $dir/nvidia_icd.json"
}

stage_robolab() {
    banner "RoboLab ($ROBOLAB_GIT_URL)"
    local dest="$RLINF_DIR/RoboLab"
    if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
        log "$dest already populated, skipping clone"
        return 0
    fi
    git clone "$ROBOLAB_GIT_URL" "$dest"
    log "cloned into $dest -- do NOT run RoboLab's own 'uv sync'; this project drives it through"
    log "RLinf's own IsaacLab fork instead (see the 'install' stage)."
}

stage_isaac_sim() {
    banner "Isaac Sim $ISAAC_SIM_VERSION standalone build"
    local dest="$RLINF_DIR/isaac_sim"
    if [ -f "$dest/setup_conda_env.sh" ]; then
        log "already present at $dest, skipping"
        return 0
    fi
    mkdir -p "$dest"
    local zip="isaac-sim-standalone-${ISAAC_SIM_VERSION}-linux-x86_64.zip"
    local url="https://download.isaacsim.omniverse.nvidia.com/${zip}"
    log "downloading $url (this is large, ~15GB, expect it to take a while)"
    wget -q --show-progress -O "$dest/$zip" "$url"
    log "unzipping"
    unzip -q "$dest/$zip" -d "$dest"
    rm -f "$dest/$zip"
    log "done: $dest"
    log "note: the 'install' stage's isaaclab.sh --install may provision its own Isaac Sim copy"
    log "into the venv -- run that stage first and only rely on this manual copy if"
    log "$dest/setup_conda_env.sh doesn't exist afterward."
}

stage_polaris_hub() {
    banner "PolaRiS-Hub dataset ($POLARIS_HUB_HF_REPO)"
    local dest="$RLINF_DIR/PolaRiS-Hub"
    if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
        log "$dest already populated, skipping"
        return 0
    fi
    if ! command -v hf >/dev/null 2>&1; then
        log "'hf' CLI not found -- installing huggingface-hub via pip --user"
        pip install --user -q huggingface-hub
    fi
    hf download "$POLARIS_HUB_HF_REPO" --repo-type=dataset --local-dir "$dest"
}

stage_install() {
    banner "RLinf + model deps (inside the sandbox)"
    if [ ! -d "$SANDBOX_DIR" ]; then
        echo "Sandbox not found at $SANDBOX_DIR -- run the 'sandbox' stage first." >&2
        exit 1
    fi
    if [ -f "$RLINF_DIR/.venv/bin/python" ]; then
        log "$RLINF_DIR/.venv already exists, skipping (delete it to force a reinstall)"
        return 0
    fi
    apptainer exec --nv \
        --bind "$WORKSPACE_ROOT:$WORKSPACE_ROOT" \
        "$SANDBOX_DIR/" \
        bash -c "cd '$RLINF_DIR' && bash requirements/install.sh embodied --model openpi --env isaaclab"
}

stage_enter_script() {
    banner "Generating enter_container.sh"
    local out="$WORKSPACE_ROOT/enter_container.sh"
    cat > "$out" <<EOF
#!/bin/bash
# Generated by RLinf/setup/bootstrap.sh -- this machine's equivalent of enter_robolab.sh.
apptainer exec --nv \\
    --bind $WORKSPACE_ROOT:$WORKSPACE_ROOT \\
    --bind $WORKSPACE_ROOT/vulkan_icd/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json \\
    --env VK_LOADER_DEBUG=all \\
    --env CARB_LOG_LEVEL=error \\
    --env PYTHONWARNINGS=ignore \\
    --env CPATH="" \\
    --env C_INCLUDE_PATH="" \\
    --env CPLUS_INCLUDE_PATH="" \\
    --env CUDA_HOME=$CUDA_DIR/cuda-${CUDA_VERSION%.*} \\
    --env CUDA_PATH=$CUDA_DIR/cuda-${CUDA_VERSION%.*} \\
    --env TORCH_EXTENSIONS_DIR=$WORKSPACE_ROOT/.cache/torch_extensions \\
    --env ACCEPT_EULA=Y \\
    $SANDBOX_DIR/ \\
    bash -c '
        source $RLINF_DIR/isaac_sim/setup_conda_env.sh
        unset PYTHONHOME
        unset VK_LOADER_DEBUG
        export PATH="$RLINF_DIR/.venv/bin:$CUDA_DIR/cuda-${CUDA_VERSION%.*}/bin:\$PATH"
        export VIRTUAL_ENV=$RLINF_DIR/.venv
        export LD_LIBRARY_PATH=/.singularity.d/libs:/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH
        cd $RLINF_DIR
        exec bash --norc --noprofile
    '
EOF
    chmod +x "$out"
    log "wrote $out -- enter the project with: bash $out"
}

# ---------------------------------------------------------------------------
run_stage() {
    case "$1" in
        sandbox) stage_sandbox ;;
        cuda) stage_cuda ;;
        vulkan-icd) stage_vulkan_icd ;;
        robolab) stage_robolab ;;
        isaac-sim) stage_isaac_sim ;;
        polaris-hub) stage_polaris_hub ;;
        install) stage_install ;;
        enter-script) stage_enter_script ;;
        *) echo "Unknown stage: $1 (see --help)" >&2; exit 1 ;;
    esac
}

if [ "$#" -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    print_help
    exit 0
fi

if [ "$1" = "all" ]; then
    for s in "${STAGES[@]}"; do run_stage "$s"; done
else
    for s in "$@"; do run_stage "$s"; done
fi

banner "Done"
log "next: bash $WORKSPACE_ROOT/enter_container.sh"
log "then verify per SETUP.md's Verification section"

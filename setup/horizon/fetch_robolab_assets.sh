#!/bin/bash
# Copies the RoboLab assets this repo doesn't vendor (robots, backgrounds, materials, fixtures)
# from upstream NVlabs/RoboLab into RoboLab/assets/. Needs git-lfs, so run it in the container:
#   bash setup/horizon/enter_container.sh bash setup/horizon/fetch_robolab_assets.sh
set -euo pipefail

R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
W=${R2S2R_WORKSPACE:-$SCRATCH/r2s2r}
UP=$W/.robolab-upstream
REV=b6cfcf6  # NVlabs/RoboLab commit these assets were taken from

if [ ! -d "$UP/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/NVlabs/RoboLab "$UP"
fi
git -C "$UP" checkout -q "$REV"

for d in robots backgrounds materials fixtures; do
    if [ -d "$R/RoboLab/assets/$d" ]; then
        echo "keep  RoboLab/assets/$d (already present)"
        continue
    fi
    git -C "$UP" lfs pull --include="assets/$d/**"
    cp -r "$UP/assets/$d" "$R/RoboLab/assets/$d"
    echo "added RoboLab/assets/$d"
done

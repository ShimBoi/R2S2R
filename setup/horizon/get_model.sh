#!/bin/bash
# Downloads the pi0.5 DROID joint-position checkpoint (JAX) and converts it to PyTorch at
# model/pi05_droid_jointpos, the path the robolab configs use. Run in the container:
#   bash setup/horizon/enter_container.sh bash setup/horizon/get_model.sh
set -euo pipefail

R=${R2S2R_REPO:?run this through setup/horizon/enter_container.sh}
W=${R2S2R_WORKSPACE:-$SCRATCH/r2s2r}
RAW=$R/model/pi05_droid_jointpos.raw
OUT=$R/model/pi05_droid_jointpos
SRC=$W/.openpi-src  # openpi source checkout, for its conversion script

[ -d "$RAW" ] || gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_droid_jointpos "$RAW"
[ -d "$SRC" ] || git clone -q --depth 1 https://github.com/RLinf/openpi "$SRC"

# The conversion script looks the config up in openpi's registry; pi05_droid_jointpos lives in
# RLinf's, so patch the lookup before running it.
export JAX_PLATFORMS=cpu
python - "$SRC/examples/convert_jax_model_to_pytorch.py" \
    --checkpoint_dir "$RAW" --config_name pi05_droid_jointpos --output_path "$OUT" <<'PY'
import runpy, sys
import openpi.training.config as openpi_config
from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT
_orig_get_config = openpi_config.get_config
openpi_config.get_config = lambda name: _CONFIGS_DICT[name] if name in _CONFIGS_DICT else _orig_get_config(name)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
PY
cp -r "$RAW/assets" "$OUT/"
echo "[get_model] done: $OUT"

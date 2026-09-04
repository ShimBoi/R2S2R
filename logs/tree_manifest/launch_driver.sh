#!/bin/bash
# Outer host-side launcher for driver.py -- fully session-detached (setsid + redirected stdio +
# disown), so the driver process survives independent of any agent/coordinator session for its
# entire multi-day lifetime. Requires OPENAI_API_KEY to already be exported in this shell before
# running (e.g. `set -a; source /scratch/cluster/jshim12/.env; set +a`) -- the driver process
# inherits this script's environment, and real Phase A calls need it.
set -e

DRIVER_DIR="/scratch/cluster/jshim12/RLinf/logs/tree_manifest"
DRIVER_LOG="${DRIVER_DIR}/driver.log"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY is not set in this shell -- source the .env file first." >&2
    exit 1
fi

cd /scratch/cluster/jshim12/RLinf
setsid bash -c "(/scratch/cluster/jshim12/RLinf/.venv/bin/python3 ${DRIVER_DIR}/driver.py) </dev/null >'${DRIVER_LOG}' 2>&1; echo \$? > '${DRIVER_LOG}.exitcode'" </dev/null >/dev/null 2>&1 &
disown
echo "driver launched (detached). log: ${DRIVER_LOG}"

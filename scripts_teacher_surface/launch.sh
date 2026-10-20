#!/usr/bin/env bash
set -euo pipefail
: "${V4_VENDOR:?Set V4_VENDOR to your ProtoMotions checkout}"
: "${ISAACLAB_PYTHON:?Set ISAACLAB_PYTHON to the simulation environment Python}"
if [ "$#" -lt 4 ]; then
  echo "Usage: launch.sh CONFIG NUM_ENVS MOTION_BANK TRACKER_CHECKPOINT" >&2
  exit 2
fi
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export V4_CONFIG="$1"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPTS:$V4_VENDOR:${PYTHONPATH:-}"
cd "$V4_VENDOR"
"$ISAACLAB_PYTHON" -u "$SCRIPTS/entry.py" \
  --checkpoint "$4" --simulator isaaclab --num-envs "$2" --headless --motion-file "$3"

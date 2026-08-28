#!/usr/bin/env bash
# Locked 3-GPU integrated semantic/hallucination/temporal development protocol.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  for candidate in python /opt/ptest/bin/python /opt/conda/envs/ptest/bin/python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import torch' >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: no Python interpreter that can import torch; set PYTHON_BIN" >&2
  exit 1
fi
export PYTHON_BIN

# Locked options are appended so a conflicting operator override cannot alter
# the scientific condition.
exec "$PYTHON_BIN" scripts/run_integrated_semantic_validation.py \
  "$@" \
  --devices cuda:0,cuda:1,cuda:2 \
  --seed 2025

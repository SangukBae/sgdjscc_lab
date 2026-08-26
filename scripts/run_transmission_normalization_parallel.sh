#!/usr/bin/env bash
# One-command three-GPU entry point. The Python orchestrator gives every GPU
# an independent output directory and merges results only after workers stop.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if [ -x /opt/conda/envs/ptest/bin/python ]; then
    PYTHON_BIN=/opt/conda/envs/ptest/bin/python
  else
    echo "ERROR: python not found; activate the ptest environment" >&2
    exit 1
  fi
fi

exec "$PYTHON_BIN" scripts/run_transmission_normalization_parallel.py "$@"

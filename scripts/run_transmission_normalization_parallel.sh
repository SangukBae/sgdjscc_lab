#!/usr/bin/env bash
# One-command three-GPU entry point. The Python orchestrator gives every GPU
# an independent output directory and merges results only after workers stop.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

_try_python() { command -v "$1" >/dev/null 2>&1 && "$1" -c "import torch" >/dev/null 2>&1; }

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -n "$PYTHON_BIN" ] && ! _try_python "$PYTHON_BIN"; then
  echo "ERROR: PYTHON_BIN=$PYTHON_BIN cannot import torch" >&2
  exit 1
fi
if [ -z "$PYTHON_BIN" ]; then
  for candidate in \
    python /opt/ptest/bin/python /opt/conda/envs/ptest/bin/python \
    "$HOME/anaconda3/envs/ptest/bin/python" "$HOME/miniconda3/envs/ptest/bin/python" \
    "$HOME/miniforge3/envs/ptest/bin/python"; do
    if _try_python "$candidate"; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: no Python interpreter that can import torch; set PYTHON_BIN=/path/to/python" >&2
  exit 1
fi

export PYTHON_BIN

exec "$PYTHON_BIN" scripts/run_transmission_normalization_parallel.py "$@"

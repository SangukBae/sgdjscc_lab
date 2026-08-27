#!/usr/bin/env bash
# Fixed-selector quantization-only reevaluation at the validated 10 dB
# decoder-step contract.  The scientific scope is intentionally locked:
# AWGN reference + float32 baseline + int16/int8/int6/int4, no SKEM and no
# keyframe-rate matching.  Extra arguments may select output/resume/smoke
# scope, but the locked options below are appended last and therefore win.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

exec bash scripts/run_transmission_normalization_parallel.sh \
  "$@" \
  --devices cuda:0,cuda:1,cuda:2 \
  --configs fixed_awgn,fixed_float32,fixed_int16,fixed_int8,fixed_int6,fixed_int4 \
  --digital-step-policy fixed_reference \
  --fixed-reference-snr-db 10 \
  --no-match-fixed-keyframes

#!/usr/bin/env bash
# Locked 3-GPU fixed-vs-SKEM exact actual-transmission matched-rate validation.
#
# fixed keeps the documented max-GOP baseline. SKEM is calibrated per video
# so the actual visual-transmitting frame count matches fixed exactly. The
# merged validator additionally requires <=1% raw bundle-byte mismatch and
# accounts padding so effective compared bytes are exactly equal.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

exec bash scripts/run_transmission_normalization_parallel.sh \
  "$@" \
  --devices cuda:0,cuda:1,cuda:2 \
  --configs fixed_float32,fixed_int16,fixed_int8,fixed_int6,fixed_int4,skem_float32,skem_int16,skem_int8,skem_int6,skem_int4 \
  --digital-step-policy fixed_reference \
  --fixed-reference-snr-db 10 \
  --psss-backend proxy \
  --match-actual-transmissions \
  --skip-keyframe-sweep \
  --skip-source-size-report

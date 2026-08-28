#!/usr/bin/env bash
# Locked three-GPU fixed-int4 edge/uncertainty transport ablation.
#
# Scope: baseline + five isolated edge profiles + five isolated uncertainty
# profiles + five predeclared combined candidates.  All rows use the validated
# fixed selector, int4 visual latent, fixed-reference 10 dB decoder policy,
# seed 2025 (unless explicitly overridden), and exact serialized bundle bytes.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GUIDE_PROFILES="baseline,edge_q4,edge_ds2,edge_ds4,edge_reuse2,edge_omit,uncertainty_q4,uncertainty_ds2,uncertainty_ds4,uncertainty_reuse2,uncertainty_omit,combined_q4,combined_ds2,combined_ds4,combined_reuse2,combined_q4_ds2_reuse2"

exec bash scripts/run_transmission_normalization_parallel.sh \
  "$@" \
  --devices cuda:0,cuda:1,cuda:2 \
  --configs fixed_int4 \
  --guide-profiles "$GUIDE_PROFILES" \
  --digital-step-policy fixed_reference \
  --fixed-reference-snr-db 10 \
  --no-match-fixed-keyframes \
  --skip-keyframe-sweep \
  --skip-source-size-report

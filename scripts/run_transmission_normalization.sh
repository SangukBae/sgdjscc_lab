#!/usr/bin/env bash
# run_transmission_normalization.sh – digital transmission normalization sweep.
#
# One-command entry point for the full fixed/SKEM x {float32,int16,int8,int6,
# int4} + AWGN-reference grid via scripts/run_transmission_reduction_eval.py,
# followed by scripts/summarize_transmission_normalization.py's separate
# quantization-effect / selector-effect tables. See docs/protocols/
# transmission_normalization.md for the full design note.
#
# Usage:
#   bash scripts/run_transmission_normalization.sh
#   bash scripts/run_transmission_normalization.sh --preflight-only
#   bash scripts/run_transmission_normalization.sh --dry-run
#   bash scripts/run_transmission_normalization.sh --resume outputs/transmission_normalization_20260826_120000
#
# Any real error aborts immediately (set -euo pipefail + explicit exit-code
# checks below) — this script never prints a success banner after a failed
# step.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ── defaults (overridable via flags below) ──────────────────────────────────
DEVICE="${DEVICE:-cuda:0}"
CONFIGS="${CONFIGS:-fixed_awgn,fixed_float32,fixed_int16,fixed_int8,fixed_int6,fixed_int4,skem_float32,skem_int16,skem_int8,skem_int6,skem_int4}"
VIDEO_IDS="${VIDEO_IDS:-}"
MAX_FRAMES="${MAX_FRAMES:-}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/etri_video_eval}"
OUTPUT_ROOT=""
MATCH_FIXED_KEYFRAMES=1
PREFLIGHT_ONLY=0
DRY_RUN=0
MIN_FREE_DISK_GIB=20

usage() {
  cat <<'EOF'
Usage: run_transmission_normalization.sh [options]

  --preflight-only            Run data/checkpoint/disk/GPU/CUDA/NVML checks and exit.
  --dry-run                   Print the exact commands that would run, without executing them.
  --resume DIR                Reuse an existing (possibly interrupted) output directory
                               instead of creating a new timestamped one. The underlying
                               python driver skips (video, config) pairs already recorded
                               in DIR/per_video_metrics.csv.
  --device DEVICE             Default: cuda:0. Use "cpu" to skip GPU/CUDA/NVML checks
                               (not recommended for a real sweep).
  --configs CSV                Comma-separated config list (default: the full
                               fixed/skem x {float32,int16,int8,int6,int4} grid + fixed_awgn).
  --video-ids CSV               Comma-separated subset of ETRI video keys (default: all).
  --max-frames N               Cap frames per video (smoke-test knob; default: all frames).
  --dataset-root PATH          Default: data/etri_video_eval under this checkout.
  --output-root PATH           Default: outputs/transmission_normalization_<timestamp>.
  --no-match-fixed-keyframes   Disable rate/keyframe-count matching between fixed and SKEM
                               (matching is ON by default -- see run_transmission_reduction_eval.py's
                               --match-fixed-keyframes).
  -h, --help                   Show this message.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --resume) OUTPUT_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --configs) CONFIGS="$2"; shift 2 ;;
    --video-ids) VIDEO_IDS="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --no-match-fixed-keyframes) MATCH_FIXED_KEYFRAMES=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[%s] FAILED: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

# ── locate a usable python (conda env "ptest" if available and not already active) ──
PYTHON_BIN="python"
if command -v conda >/dev/null 2>&1; then
  if [ "${CONDA_DEFAULT_ENV:-}" != "ptest" ] && conda env list 2>/dev/null | grep -qE '(^|[[:space:]])ptest([[:space:]]|$)'; then
    log "activating conda env 'ptest'"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ptest
  fi
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "no python interpreter on PATH (expected conda env 'ptest' or an equivalent environment with torch installed)"

# ── preflight: data / checkpoint / disk / GPU / CUDA / NVML ────────────────
run_preflight() {
  log "preflight: dataset root -> $DATASET_ROOT"
  [ -f "$DATASET_ROOT/manifest.csv" ] || fail "dataset manifest not found: $DATASET_ROOT/manifest.csv"

  log "preflight: checkpoints"
  local model_root
  model_root="$("$PYTHON_BIN" - <<'PYEOF'
import sys
sys.path.insert(0, "src")
from sgdjscc_lab.paths import model_root
print(model_root())
PYEOF
)" || fail "could not resolve model_root (check SGDJSCC_MODEL_ROOT / src/sgdjscc_lab/paths.py)"
  log "preflight: checkpoint root -> $model_root"
  local ckpt
  for ckpt in JSCC_model.pth diffusion_backbone.pth diffusion_controlnet.pth muge-epoch-19-checkpoint.pth; do
    [ -f "$model_root/$ckpt" ] || fail "missing checkpoint: $model_root/$ckpt (see CLAUDE.md's HuggingFace murjun/SGDJSCC download instructions)"
  done

  log "preflight: disk space"
  local out_check_dir free_kib free_gib
  out_check_dir="${OUTPUT_ROOT:-outputs}"
  mkdir -p "$out_check_dir"
  free_kib="$(df -Pk "$out_check_dir" | awk 'NR==2 {print $4}')"
  free_gib="$((free_kib / 1024 / 1024))"
  log "preflight: ${free_gib} GiB free at $out_check_dir"
  if [ "$free_gib" -lt "$MIN_FREE_DISK_GIB" ]; then
    fail "only ${free_gib} GiB free at $out_check_dir (need >= ${MIN_FREE_DISK_GIB} GiB for a full 10-video x 11-config sweep's recon frames/packets)"
  fi

  if [ "$DEVICE" = "cpu" ]; then
    log "preflight: --device cpu -- skipping GPU/CUDA/NVML checks (not recommended for a real sweep)"
    return 0
  fi

  log "preflight: GPU / nvidia-smi"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found on PATH -- no NVIDIA driver visible in this environment"
  fi
  nvidia-smi >/tmp/txnorm_nvidia_smi.$$ 2>&1 || {
    cat /tmp/txnorm_nvidia_smi.$$ >&2
    rm -f /tmp/txnorm_nvidia_smi.$$
    fail "nvidia-smi failed -- driver/NVML is not healthy in this environment"
  }
  rm -f /tmp/txnorm_nvidia_smi.$$

  log "preflight: torch CUDA / NVML via python"
  local cuda_check
  if ! cuda_check="$("$PYTHON_BIN" - <<'PYEOF' 2>&1
import torch
available = torch.cuda.is_available()
print(f"cuda_available={available}")
if not available:
    raise SystemExit(1)
print(f"device_count={torch.cuda.device_count()}")
print(f"device_name_0={torch.cuda.get_device_name(0)}")
PYEOF
)"; then
    echo "$cuda_check" >&2
    if echo "$cuda_check" | grep -qi "nvml"; then
      fail "torch reported an NVML error (see above) -- driver/container GPU passthrough is broken, not a code bug. Do not delete/recreate the container; report this state and stop."
    fi
    fail "torch.cuda.is_available() == False -- GPU is not visible to torch in this environment"
  fi
  log "preflight: $cuda_check" | tr '\n' ' '
  echo
}

run_preflight

if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
  log "preflight-only: all checks passed"
  exit 0
fi

# ── resolve output root ──────────────────────────────────────────────────
if [ -z "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="outputs/transmission_normalization_$(date +%Y%m%d_%H%M%S)"
  log "no --resume/--output-root given -- starting a fresh run at $OUTPUT_ROOT"
elif [ -d "$OUTPUT_ROOT" ] && [ -f "$OUTPUT_ROOT/per_video_metrics.csv" ]; then
  log "resuming existing run at $OUTPUT_ROOT (will skip already-completed (video, config) pairs)"
else
  log "using output root: $OUTPUT_ROOT"
fi

SWEEP_CMD=("$PYTHON_BIN" scripts/run_transmission_reduction_eval.py
  --output-root "$OUTPUT_ROOT"
  --dataset-root "$DATASET_ROOT"
  --configs "$CONFIGS"
  --device "$DEVICE"
)
[ -n "$VIDEO_IDS" ] && SWEEP_CMD+=(--video-ids "$VIDEO_IDS")
[ -n "$MAX_FRAMES" ] && SWEEP_CMD+=(--max-frames "$MAX_FRAMES")
[ "$MATCH_FIXED_KEYFRAMES" -eq 1 ] && SWEEP_CMD+=(--match-fixed-keyframes)

SUMMARIZE_CMD=("$PYTHON_BIN" scripts/summarize_transmission_normalization.py --run-root "$OUTPUT_ROOT")

if [ "$DRY_RUN" -eq 1 ]; then
  log "dry-run: would create $OUTPUT_ROOT and run:"
  printf '  %q' "${SWEEP_CMD[@]}"; echo
  printf '  %q' "${SUMMARIZE_CMD[@]}"; echo
  exit 0
fi

mkdir -p "$OUTPUT_ROOT"
LOG_FILE="$OUTPUT_ROOT/normalization_run.log"
log "sweep -> $LOG_FILE"
log "command: ${SWEEP_CMD[*]}"

if ! "${SWEEP_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
  fail "run_transmission_reduction_eval.py exited non-zero -- see $LOG_FILE. Re-run this same command (or with --resume $OUTPUT_ROOT) to continue from the last completed (video, config) pair."
fi

log "summarize -> quantization_effect.csv / selector_effect.csv"
log "command: ${SUMMARIZE_CMD[*]}"
if ! "${SUMMARIZE_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
  fail "summarize_transmission_normalization.py exited non-zero -- see $LOG_FILE"
fi

log "done. Results in $OUTPUT_ROOT:"
log "  per_video_metrics.csv / aggregate.csv        -- full quality + byte accounting"
log "  quantization_effect.csv                      -- bit_depth effect, selector held constant"
log "  selector_effect.csv                          -- fixed-vs-SKEM effect, bit_depth held constant"
log "  pareto_frontier.csv / summary.json            -- Pareto candidate under the quality gate"
log "  run_manifest.json                            -- commit/dirty, argv, resolved config, hashes, env"
log "  README.md                                    -- this run's own description"

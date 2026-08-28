#!/usr/bin/env bash
# run_transmission_normalization.sh – digital transmission normalization sweep.
#
# One-command entry point for the full fixed/SKEM x {float32,int16,int8,int6,
# int4} + AWGN-reference grid via scripts/run_transmission_reduction_eval.py,
# The Python driver also runs summarize_transmission_normalization.py before
# finalizing its manifest, so effect-table hashes cover the complete run. See docs/protocols/
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
MATCH_ACTUAL_TRANSMISSIONS=0
MATCHED_RATE_THRESHOLDS="${MATCHED_RATE_THRESHOLDS:--0.95,-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.999999}"
MATCHED_RATE_MAX_SEGMENT_LENGTHS="${MATCHED_RATE_MAX_SEGMENT_LENGTHS:-8,10,12,14,16,20,24,32,48,64,100}"
SKIP_KEYFRAME_SWEEP=0
SKIP_SOURCE_SIZE_REPORT=0
PREFLIGHT_ONLY=0
DRY_RUN=0
MIN_FREE_DISK_GIB=20
SEED="${SEED:-2025}"
DIGITAL_STEP_POLICY="${DIGITAL_STEP_POLICY:-fixed_reference}"
FIXED_REFERENCE_SNR_DB="${FIXED_REFERENCE_SNR_DB:-10}"
ABLATION_LABEL="${ABLATION_LABEL:-}"
PSSS_BACKEND="${PSSS_BACKEND:-proxy}"
PSSS_MODEL_ID="${PSSS_MODEL_ID:-}"
PSSS_DEVICE="${PSSS_DEVICE:-cpu}"
PSSS_DTYPE="${PSSS_DTYPE:-fp32}"
PSSS_THRESHOLD="${PSSS_THRESHOLD:-}"
PSSS_MAX_SEGMENT_LENGTH="${PSSS_MAX_SEGMENT_LENGTH:-}"
USE_SCENE_DETECTOR=0
RETRY_FAILED=0

usage() {
  cat <<'EOF'
Usage: run_transmission_normalization.sh [options]

  --preflight-only            Run data/checkpoint/disk/GPU/CUDA/NVML checks and exit.
  --dry-run                   Print the exact commands that would run, without executing them.
  --resume DIR                Reuse an existing (possibly interrupted) output directory
                               instead of creating a new timestamped one. The underlying
                               python driver refuses to continue if the run's conditions
                               (commit/dataset/config/checkpoint hash/seed/video list/
                               granularity/PSSS settings) differ from run_signature.json
                               already recorded there -- see docs/protocols/
                               transmission_normalization.md.
  --device DEVICE             Default: cuda:0. Use "cpu" to skip GPU/CUDA/NVML checks
                               (not recommended for a real sweep).
  --configs CSV                Comma-separated config list (default: the full
                               fixed/skem x {float32,int16,int8,int6,int4} grid + fixed_awgn).
  --video-ids CSV               Comma-separated subset of ETRI video keys (default: all).
  --max-frames N               Cap frames per video (smoke-test knob; default: all frames).
  --dataset-root PATH          Default: data/etri_video_eval under this checkout.
  --output-root PATH           Default: outputs/transmission_normalization_<timestamp>.
  --no-match-fixed-keyframes   Disable exact keyframe-count matching between fixed and SKEM
                               (ON by default -- FixedCountKeyframeSelector, see
                               run_transmission_reduction_eval.py's --match-fixed-keyframes).
  --match-actual-transmissions Keep fixed max-GOP unchanged and calibrate SKEM per video so
                               actual visual-transmitting frame counts match exactly.
  --matched-rate-thresholds CSV
                               PSSS threshold calibration grid for the exact mode.
  --matched-rate-max-segment-lengths CSV
                               Max-segment calibration grid for the exact mode.
  --skip-keyframe-sweep        Skip the unrelated diagnostic PSSS sweep.
  --skip-source-size-report    Skip the unrelated source MP4 size table.
  --seed N                     Base seed for Python/NumPy/PyTorch/CUDA (default: 2025).
  --digital-step-policy NAME   fixed_reference (default) | bitdepth_proxy | quant_nmse.
                               Anything but fixed_reference REQUIRES --ablation-label.
  --fixed-reference-snr-db DB  Decoder reference SNR for fixed_reference (default: 10).
  --ablation-label LABEL       Required when --digital-step-policy != fixed_reference.
  --psss-backend NAME           mock | proxy (default) | real -- see --psss-model-id.
  --psss-model-id ID            HF causal-LM/VLM id, required for --psss-backend real.
  --psss-device DEVICE          Device for the PSSS backend (default: cpu).
  --psss-dtype DTYPE            Dtype for the PSSS backend (default: fp32).
  --psss-threshold FLOAT        PSSS/SKEM selection threshold (python default: 0.35).
  --psss-max-segment-length N   PSSS/SKEM max segment length (python default: 16).
  --use-scene-detector          Combine real scene-change detection with SKEM/PSSS.
  --retry-failed                On resume, retry pairs already in failed_pairs.csv.
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
    --match-actual-transmissions) MATCH_ACTUAL_TRANSMISSIONS=1; MATCH_FIXED_KEYFRAMES=0; shift ;;
    --matched-rate-thresholds) MATCHED_RATE_THRESHOLDS="$2"; shift 2 ;;
    --matched-rate-max-segment-lengths) MATCHED_RATE_MAX_SEGMENT_LENGTHS="$2"; shift 2 ;;
    --skip-keyframe-sweep) SKIP_KEYFRAME_SWEEP=1; shift ;;
    --skip-source-size-report) SKIP_SOURCE_SIZE_REPORT=1; shift ;;
    --seed) SEED="$2"; shift 2 ;;
    --digital-step-policy) DIGITAL_STEP_POLICY="$2"; shift 2 ;;
    --fixed-reference-snr-db) FIXED_REFERENCE_SNR_DB="$2"; shift 2 ;;
    --ablation-label) ABLATION_LABEL="$2"; shift 2 ;;
    --psss-backend) PSSS_BACKEND="$2"; shift 2 ;;
    --psss-model-id) PSSS_MODEL_ID="$2"; shift 2 ;;
    --psss-device) PSSS_DEVICE="$2"; shift 2 ;;
    --psss-dtype) PSSS_DTYPE="$2"; shift 2 ;;
    --psss-threshold) PSSS_THRESHOLD="$2"; shift 2 ;;
    --psss-max-segment-length) PSSS_MAX_SEGMENT_LENGTH="$2"; shift 2 ;;
    --use-scene-detector) USE_SCENE_DETECTOR=1; shift ;;
    --retry-failed) RETRY_FAILED=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[%s] FAILED: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

# ── locate a usable python (explicit override, ptest env, Docker fallback) ──
_try_python() { command -v "$1" >/dev/null 2>&1 && "$1" -c "import torch" >/dev/null 2>&1; }

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ] && command -v conda >/dev/null 2>&1; then
  if [ "${CONDA_DEFAULT_ENV:-}" != "ptest" ] && conda env list 2>/dev/null | grep -qE '(^|[[:space:]])ptest([[:space:]]|$)'; then
    log "activating conda env 'ptest'"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ptest
  fi
  PYTHON_BIN="python"
fi
if [ -z "$PYTHON_BIN" ] || ! _try_python "$PYTHON_BIN"; then
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
[ -n "$PYTHON_BIN" ] && _try_python "$PYTHON_BIN" || \
  fail "no Python interpreter that can import torch; set PYTHON_BIN=/path/to/python"
export PYTHON_BIN

# ── preflight: data / checkpoint / disk / GPU / CUDA / NVML ────────────────
run_preflight() {
  log "preflight: exact git commit provenance"
  local git_check
  git_check="$("$PYTHON_BIN" - <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, "src")
from sgdjscc_lab.utils.run_manifest import UNKNOWN, get_git_state
state = get_git_state(Path.cwd())
print(f"commit={state['commit']} dirty={state['dirty']} branch={state['branch']}")
if state["commit"] == UNKNOWN:
    raise SystemExit(1)
if state["dirty"] is True:
    raise SystemExit(2)
if state["dirty"] == UNKNOWN:
    print("warning=tracked dirty state unavailable; set SGDJSCC_GIT_DIRTY=false after host verification")
PYEOF
)" || fail "git provenance failed: commit is unknown or tracked checkout is dirty. Keep .git mounted/install git, or inject verified SGDJSCC_GIT_COMMIT and SGDJSCC_GIT_DIRTY=false after host verification."
  log "preflight: $git_check"

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
  if ! cuda_check="$(TXNORM_DEVICE="$DEVICE" "$PYTHON_BIN" - <<'PYEOF' 2>&1
import os
import torch
available = torch.cuda.is_available()
print(f"cuda_available={available}")
if not available:
    raise SystemExit(1)
print(f"device_count={torch.cuda.device_count()}")
device = os.environ["TXNORM_DEVICE"]
index = int(device.split(":", 1)[1]) if ":" in device else 0
if index < 0 or index >= torch.cuda.device_count():
    raise SystemExit(f"invalid CUDA device ordinal: {device}")
print(f"selected_device={device} device_name={torch.cuda.get_device_name(index)}")
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

if [ "$DIGITAL_STEP_POLICY" != "fixed_reference" ] && [ -z "$ABLATION_LABEL" ]; then
  fail "--digital-step-policy $DIGITAL_STEP_POLICY requires --ablation-label (it is a decoder-step ablation, not the quantization comparison)"
fi
if [ "$PSSS_BACKEND" = "real" ] && [ -z "$PSSS_MODEL_ID" ]; then
  fail "--psss-backend real requires --psss-model-id"
fi

SWEEP_CMD=("$PYTHON_BIN" scripts/run_transmission_reduction_eval.py
  --output-root "$OUTPUT_ROOT"
  --dataset-root "$DATASET_ROOT"
  --configs "$CONFIGS"
  --device "$DEVICE"
  --seed "$SEED"
  --digital-step-policy "$DIGITAL_STEP_POLICY"
  --fixed-reference-snr-db "$FIXED_REFERENCE_SNR_DB"
  --psss-backend "$PSSS_BACKEND"
  --psss-device "$PSSS_DEVICE"
  --psss-dtype "$PSSS_DTYPE"
)
[ -n "$VIDEO_IDS" ] && SWEEP_CMD+=(--video-ids "$VIDEO_IDS")
[ -n "$MAX_FRAMES" ] && SWEEP_CMD+=(--max-frames "$MAX_FRAMES")
[ "$MATCH_FIXED_KEYFRAMES" -eq 1 ] && SWEEP_CMD+=(--match-fixed-keyframes)
[ "$MATCH_ACTUAL_TRANSMISSIONS" -eq 1 ] && SWEEP_CMD+=(
  --match-actual-transmissions
  --matched-rate-thresholds "$MATCHED_RATE_THRESHOLDS"
  --matched-rate-max-segment-lengths "$MATCHED_RATE_MAX_SEGMENT_LENGTHS"
)
[ "$SKIP_KEYFRAME_SWEEP" -eq 1 ] && SWEEP_CMD+=(--skip-keyframe-sweep)
[ "$SKIP_SOURCE_SIZE_REPORT" -eq 1 ] && SWEEP_CMD+=(--skip-source-size-report)
[ -n "$ABLATION_LABEL" ] && SWEEP_CMD+=(--ablation-label "$ABLATION_LABEL")
[ -n "$PSSS_MODEL_ID" ] && SWEEP_CMD+=(--psss-model-id "$PSSS_MODEL_ID")
[ -n "$PSSS_THRESHOLD" ] && SWEEP_CMD+=(--psss-threshold "$PSSS_THRESHOLD")
[ -n "$PSSS_MAX_SEGMENT_LENGTH" ] && SWEEP_CMD+=(--psss-max-segment-length "$PSSS_MAX_SEGMENT_LENGTH")
[ "$USE_SCENE_DETECTOR" -eq 1 ] && SWEEP_CMD+=(--use-scene-detector)
[ "$RETRY_FAILED" -eq 1 ] && SWEEP_CMD+=(--retry-failed)

if [ "$DRY_RUN" -eq 1 ]; then
  log "dry-run: would create $OUTPUT_ROOT and run:"
  printf '  %q' "${SWEEP_CMD[@]}"; echo
  exit 0
fi

mkdir -p "$OUTPUT_ROOT"
LOG_FILE="$OUTPUT_ROOT/normalization_run.log"
log "sweep -> $LOG_FILE"
log "command: ${SWEEP_CMD[*]}"

set +e
"${SWEEP_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
sweep_status="${PIPESTATUS[0]}"
set -e
if [ "$sweep_status" -eq 3 ]; then
  printf '[%s] COMPLETED_WITH_FAILURES: final summaries/manifests were written; inspect failed_pairs.csv and resume with --retry-failed.\n' "$(date +%H:%M:%S)" >&2
  exit 3
elif [ "$sweep_status" -ne 0 ]; then
  fail "run_transmission_reduction_eval.py exited non-zero -- see $LOG_FILE. Re-run this same command (or with --resume $OUTPUT_ROOT) to continue from the last completed (video, config) pair."
fi

log "done. Results in $OUTPUT_ROOT:"
log "  per_video_metrics.csv / aggregate.csv        -- full quality + bytes/video + bytes/frame"
log "  failed_pairs.csv                             -- (video, config) aborted at first non-finite value"
if [ "$DIGITAL_STEP_POLICY" = "fixed_reference" ]; then
  log "  quantization_effect.csv                      -- bit_depth effect, selector held constant"
  log "  selector_effect.csv                          -- fixed-vs-SKEM effect, bit_depth held constant"
else
  log "  quantization_effect_ablation.csv             -- decoder-step ablation table"
  log "  selector_effect_ablation.csv                 -- decoder-step ablation selector table"
fi
log "  rate_matching.csv                            -- fixed vs SKEM actual transmission/byte matching"
log "  pareto_frontier.csv / summary.json            -- Pareto candidate under the quality gate"
log "  run_manifest_initial.json / run_manifest.json -- commit/dirty, argv, resolved config, hashes, env"
log "  run_signature.json                           -- resume safety signature (refuses a mismatched resume)"
log "  README.md                                    -- this run's own description (한국어)"

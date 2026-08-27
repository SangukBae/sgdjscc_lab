#!/usr/bin/env bash
# run_float32_digital_diagnostics.sh – float32 digital quality diagnostic, one command.
#
# Drives scripts/diagnose_float32_digital_quality.py through a fixed sequence
# of stages (env/CUDA/data/checkpoint preflight -> tests+dry-run -> 1x1 path
# comparison + tensor contract -> 1x1 full ablation set -> 1x20 paired
# diagnostic -> 3 core-condition videos x 100 frames -> report), so a single
# command reproduces the whole diagnostic protocol on a server. See
# docs/protocols/float32_digital_diagnostics.md for the full design note.
#
# Usage:
#   bash scripts/run_float32_digital_diagnostics.sh                 # profile=full
#   bash scripts/run_float32_digital_diagnostics.sh --dry-run
#   bash scripts/run_float32_digital_diagnostics.sh --profile smoke
#   bash scripts/run_float32_digital_diagnostics.sh --profile short
#   bash scripts/run_float32_digital_diagnostics.sh --profile full
#   bash scripts/run_float32_digital_diagnostics.sh --resume outputs/f32dig_20260827_120000
#   bash scripts/run_float32_digital_diagnostics.sh --cuda-visible-devices 0
#
# Exit codes: 0 = every stage completed clean. 130 = interrupted (SIGINT/
# SIGTERM) — safe to re-run with --resume against the SAME output dir it
# printed. Any other non-zero = preflight failed (nothing ran) OR one or more
# independently-runnable stages failed (recorded; the remaining stages that
# do not depend on the failed one still ran — see the per-stage log lines).
#
# No stage is retried automatically (including on CUDA OOM) — each stage runs
# exactly once per invocation; re-running (optionally with --resume) is an
# explicit, observed decision, never an implicit loop.

set -uo pipefail  # deliberately not -e: independent stage failures must be
                   # recorded and the run must continue to the next stage.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${DEVICE:-cuda:0}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/etri_video_eval}"
OUTPUT_ROOT=""
PROFILE="full"
DRY_RUN=0
CUDA_VISIBLE_DEVICES_ARG=""
MIN_FREE_DISK_GIB=10
SEED="${SEED:-2025}"

# Core-condition videos (task spec: normal motion / semantic change / scene cut).
VIDEO_NORMAL_MOTION="01_person_walk"
VIDEO_SEMANTIC_CHANGE="07_person_enter"
VIDEO_SCENE_CUT="09_scene_cut_chair_car"

usage() {
  cat <<'EOF'
Usage: run_float32_digital_diagnostics.sh [options]

  --dry-run                    Print every stage's resolved plan (via each
                                python invocation's own --dry-run) and exit.
  --profile NAME                smoke | short | full (default: full).
  --resume DIR                 Reuse an existing output directory (must have
                                been created by a previous run of this script);
                                each stage resumes via its own subdirectory's
                                run_signature.json.
  --device DEVICE               Default: cuda:0. "cpu" skips GPU/CUDA/NVML
                                preflight checks (not recommended for a real run).
  --dataset-root PATH           Default: data/etri_video_eval under this checkout.
  --output-root PATH            Default: outputs/f32dig_<timestamp>.
  --cuda-visible-devices LIST   Sets CUDA_VISIBLE_DEVICES before every stage;
                                --device is still cuda:0 internally (isolated).
  --seed N                      Base seed (default: 2025).
  -h, --help                    Show this message.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --resume) OUTPUT_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_ARG="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$PROFILE" in
  smoke|short|full) ;;
  *) echo "ERROR: --profile must be smoke|short|full, got: $PROFILE" >&2; exit 2 ;;
esac

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
fail() { printf '[%s] FAILED: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; exit 1; }

if [ -n "$CUDA_VISIBLE_DEVICES_ARG" ]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_ARG"
  log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (internal --device stays $DEVICE)"
fi

STOPPED_EARLY=0
trap 'STOPPED_EARLY=1; log "received interrupt signal; current stage handles SIGINT/SIGTERM gracefully and will exit; re-run with --resume \"$OUTPUT_ROOT\" to continue."' INT TERM

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
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "no python interpreter on PATH (expected conda env 'ptest' or equivalent)"

# ── preflight: git / dataset / checkpoints / disk / GPU / CUDA / NVML ──────
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
PYEOF
)" || fail "git provenance failed: commit unknown or tracked checkout dirty."
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
    [ -f "$model_root/$ckpt" ] || fail "missing checkpoint: $model_root/$ckpt"
  done

  log "preflight: disk space"
  local out_check_dir free_kib free_gib
  out_check_dir="${OUTPUT_ROOT:-outputs}"
  mkdir -p "$out_check_dir"
  free_kib="$(df -Pk "$out_check_dir" | awk 'NR==2 {print $4}')"
  free_gib="$((free_kib / 1024 / 1024))"
  log "preflight: ${free_gib} GiB free at $out_check_dir"
  if [ "$free_gib" -lt "$MIN_FREE_DISK_GIB" ]; then
    fail "only ${free_gib} GiB free at $out_check_dir (need >= ${MIN_FREE_DISK_GIB} GiB)"
  fi

  if [ "$DEVICE" = "cpu" ]; then
    log "preflight: --device cpu -- skipping GPU/CUDA/NVML checks (not recommended for a real run)"
    return 0
  fi

  log "preflight: GPU / nvidia-smi"
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found on PATH"
  nvidia-smi >/tmp/f32dig_nvidia_smi.$$ 2>&1 || { cat /tmp/f32dig_nvidia_smi.$$ >&2; rm -f /tmp/f32dig_nvidia_smi.$$; fail "nvidia-smi failed"; }
  rm -f /tmp/f32dig_nvidia_smi.$$

  log "preflight: torch CUDA / NVML via python"
  local cuda_check
  if ! cuda_check="$(F32DIG_DEVICE="$DEVICE" "$PYTHON_BIN" - <<'PYEOF' 2>&1
import os
import torch
available = torch.cuda.is_available()
print(f"cuda_available={available}")
if not available:
    raise SystemExit(1)
print(f"device_count={torch.cuda.device_count()}")
device = os.environ["F32DIG_DEVICE"]
index = int(device.split(":", 1)[1]) if ":" in device else 0
if index < 0 or index >= torch.cuda.device_count():
    raise SystemExit(f"invalid CUDA device ordinal: {device}")
print(f"selected_device={device} device_name={torch.cuda.get_device_name(index)}")
PYEOF
)"; then
    echo "$cuda_check" >&2
    if echo "$cuda_check" | grep -qi "nvml"; then
      fail "torch reported an NVML error -- driver/container GPU passthrough is broken, not a code bug."
    fi
    fail "torch.cuda.is_available() == False -- GPU not visible to torch"
  fi
  log "preflight: $(echo "$cuda_check" | tr '\n' ' ')"
}

run_preflight

# ── resolve output root ─────────────────────────────────────────────────
if [ -z "$OUTPUT_ROOT" ]; then
  OUTPUT_ROOT="outputs/f32dig_$(date +%Y%m%d_%H%M%S)"
  log "no --resume/--output-root given -- starting a fresh run at $OUTPUT_ROOT"
else
  log "using output root: $OUTPUT_ROOT (resume-safe per stage via run_signature.json)"
fi
mkdir -p "$OUTPUT_ROOT"
RESUME_FLAG=""
[ -d "$OUTPUT_ROOT" ] && RESUME_FLAG="--resume"

# ── profile-dependent scope ─────────────────────────────────────────────
case "$PROFILE" in
  smoke)
    FRAMES_PAIRED="0-1"        # stage 5: 1 video x 2 frames
    FRAMES_CORE="0-2"          # stage 6: 3 videos x 3 frames
    ;;
  short)
    FRAMES_PAIRED="0-4"        # stage 5: 1 video x 5 frames
    FRAMES_CORE="0-9"          # stage 6: 3 videos x 10 frames
    ;;
  full)
    FRAMES_PAIRED="0-19"       # stage 5: 1 video x 20 frames
    FRAMES_CORE="0-99"         # stage 6: 3 videos x 100 frames
    ;;
esac

DIAG_CMD=("$PYTHON_BIN" scripts/diagnose_float32_digital_quality.py --dataset-root "$DATASET_ROOT" --device "$DEVICE" --seed "$SEED")

STAGE_FAILURES=0
run_stage() {
  local name="$1"; shift
  log "==== stage: $name ===="
  if [ "$DRY_RUN" -eq 1 ]; then
    "$@" --dry-run
    return $?
  fi
  "$@"
  local rc=$?
  if [ "$rc" -eq 130 ]; then
    log "stage '$name' interrupted (exit 130); stopping remaining stages."
    STOPPED_EARLY=1
  elif [ "$rc" -ne 0 ]; then
    log "stage '$name' FAILED (exit $rc) -- recorded; continuing with remaining independent stages."
    STAGE_FAILURES=$((STAGE_FAILURES + 1))
  else
    log "stage '$name' OK"
  fi
  return "$rc"
}

# ── stage 2: related tests + dry-run ────────────────────────────────────
log "==== stage: 2_tests_and_dry_run ===="
TEST_FILES=(tests/test_float32_digital_diagnostics.py tests/test_receiver_runtime.py
            tests/test_packet_bundle.py tests/test_wire_packet.py tests/test_digital_step_matching.py)
if [ "$DRY_RUN" -eq 1 ]; then
  log "dry-run: would run: $PYTHON_BIN -m pytest ${TEST_FILES[*]} -q"
else
  "$PYTHON_BIN" -m pytest "${TEST_FILES[@]}" -q
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log "stage '2_tests_and_dry_run' FAILED (exit $rc) -- recorded; continuing."
    STAGE_FAILURES=$((STAGE_FAILURES + 1))
  else
    log "stage '2_tests_and_dry_run' OK"
  fi
fi
[ "$STOPPED_EARLY" -eq 1 ] && { log "stopped after stage 2"; exit 130; }

# ── stage 3: 1 video x 1 frame, all 3 paths + tensor contract check ────
run_stage "3_single_frame_path_contract" \
  "${DIAG_CMD[@]}" --output-root "$OUTPUT_ROOT/stage3_single_frame_paths" \
  --video-ids "$VIDEO_NORMAL_MOTION" --frames 0 --ablations baseline $RESUME_FLAG
[ "$STOPPED_EARLY" -eq 1 ] && { log "stopped after stage 3"; exit 130; }

# ── stage 4: 1 video x 1 frame, full ablation set ───────────────────────
run_stage "4_single_frame_full_ablation" \
  "${DIAG_CMD[@]}" --output-root "$OUTPUT_ROOT/stage4_single_frame_ablations" \
  --video-ids "$VIDEO_NORMAL_MOTION" --frames 0 --ablations all $RESUME_FLAG
[ "$STOPPED_EARLY" -eq 1 ] && { log "stopped after stage 4"; exit 130; }

# ── stage 5: 1 video x 20 frames, paired diagnostic ─────────────────────
run_stage "5_paired_multi_frame" \
  "${DIAG_CMD[@]}" --output-root "$OUTPUT_ROOT/stage5_paired_frames" \
  --video-ids "$VIDEO_NORMAL_MOTION" --frames "$FRAMES_PAIRED" \
  --ablations baseline,diffusion_bypass_vae_direct $RESUME_FLAG
[ "$STOPPED_EARLY" -eq 1 ] && { log "stopped after stage 5"; exit 130; }

# ── stage 6: 3 core-condition videos x 100 frames, baseline only ───────
run_stage "6_core_conditions" \
  "${DIAG_CMD[@]}" --output-root "$OUTPUT_ROOT/stage6_core_conditions" \
  --video-ids "$VIDEO_NORMAL_MOTION,$VIDEO_SEMANTIC_CHANGE,$VIDEO_SCENE_CUT" \
  --frames "$FRAMES_CORE" --ablations baseline --no-instrument-tensors $RESUME_FLAG
[ "$STOPPED_EARLY" -eq 1 ] && { log "stopped after stage 6"; exit 130; }

if [ "$DRY_RUN" -eq 1 ]; then
  log "dry-run complete for all stages."
  exit 0
fi

# ── stage 7: validate results, hash, integrated report ──────────────────
log "==== stage: 7_validate_and_report ===="
{
  echo "# float32 digital diagnostics — integrated run ($PROFILE profile)"
  echo
  echo "- generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- output_root: $OUTPUT_ROOT"
  echo "- stage_failures: $STAGE_FAILURES"
  echo
  echo "## per-stage artifact hashes (sha256)"
  for stage_dir in "$OUTPUT_ROOT"/stage3_single_frame_paths "$OUTPUT_ROOT"/stage4_single_frame_ablations \
                   "$OUTPUT_ROOT"/stage5_paired_frames "$OUTPUT_ROOT"/stage6_core_conditions; do
    [ -d "$stage_dir" ] || continue
    echo
    echo "### $(basename "$stage_dir")"
    for f in run_manifest.json summary.json REPORT.md path_comparison.csv failed_cases.csv; do
      if [ -f "$stage_dir/$f" ]; then
        printf -- '- `%s`: `%s`\n' "$f" "$(sha256sum "$stage_dir/$f" | awk '{print $1}')"
      fi
    done
  done
} > "$OUTPUT_ROOT/INTEGRATED_REPORT.md"
log "wrote $OUTPUT_ROOT/INTEGRATED_REPORT.md"

log "done. Results in $OUTPUT_ROOT:"
log "  stage3_single_frame_paths/     -- 1 video x 1 frame, 3 paths, full tensor contract"
log "  stage4_single_frame_ablations/ -- 1 video x 1 frame, full ablation set"
log "  stage5_paired_frames/          -- 1 video x N frames, paired path comparison"
log "  stage6_core_conditions/        -- 3 core-condition videos x N frames, metrics only"
log "  INTEGRATED_REPORT.md           -- per-stage artifact hashes"
log "  each stageN_*/REPORT.md        -- per-stage verdict + evidence (see docs/protocols/float32_digital_diagnostics.md)"

if [ "$STAGE_FAILURES" -gt 0 ]; then
  log "COMPLETED_WITH_FAILURES: $STAGE_FAILURES stage(s) failed -- see logs above."
  exit 3
fi
exit 0

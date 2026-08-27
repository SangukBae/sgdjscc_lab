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
#   bash scripts/run_float32_digital_diagnostics.sh --parallel-devices 0,1,2
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
PARALLEL_DEVICES_ARG=""
PARALLEL_MODE=0
MIN_FREE_DISK_GIB=10
SEED="${SEED:-2025}"
FIXED_REFERENCE_SNR_DB="${FIXED_REFERENCE_SNR_DB:-10.0}"

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
  --parallel-devices LIST       Run the diagnostic on exactly three physical
                                GPUs (for example 0,1,2). Stages 3/4/5 run in
                                parallel, then the three stage-6 videos run
                                one per GPU in independent worker directories.
  --seed N                      Base seed (default: 2025).
  --fixed-reference-snr-db DB  Digital fixed-reference decoder SNR
                                (default: 10.0 dB; valid range: -20..60).
  -h, --help                    Show this message.
EOF
}

RESUME_REQUESTED=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --resume) OUTPUT_ROOT="$2"; RESUME_REQUESTED=1; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE_DEVICES_ARG="$2"; shift 2 ;;
    --parallel-devices) PARALLEL_DEVICES_ARG="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --fixed-reference-snr-db) FIXED_REFERENCE_SNR_DB="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$PROFILE" in
  smoke|short|full) ;;
  *) echo "ERROR: --profile must be smoke|short|full, got: $PROFILE" >&2; exit 2 ;;
esac

PARALLEL_GPU_IDS=()
if [ -n "$PARALLEL_DEVICES_ARG" ]; then
  [ -z "$CUDA_VISIBLE_DEVICES_ARG" ] || {
    echo "ERROR: pass only one of --cuda-visible-devices / --parallel-devices" >&2
    exit 2
  }
  IFS=',' read -r -a PARALLEL_GPU_IDS <<< "$PARALLEL_DEVICES_ARG"
  if [ "${#PARALLEL_GPU_IDS[@]}" -ne 3 ]; then
    echo "ERROR: --parallel-devices requires exactly three physical GPU indices" >&2
    exit 2
  fi
  declare -A _SEEN_PARALLEL_GPUS=()
  for _gpu in "${PARALLEL_GPU_IDS[@]}"; do
    if ! [[ "$_gpu" =~ ^[0-9]+$ ]] || [ -n "${_SEEN_PARALLEL_GPUS[$_gpu]:-}" ]; then
      echo "ERROR: --parallel-devices must contain three unique integer indices, got: $PARALLEL_DEVICES_ARG" >&2
      exit 2
    fi
    _SEEN_PARALLEL_GPUS[$_gpu]=1
  done
  unset _SEEN_PARALLEL_GPUS _gpu
  if [ "$DEVICE" != "cuda:0" ] && ! { [ "$DRY_RUN" -eq 1 ] && [ "$DEVICE" = "cpu" ]; }; then
    echo "ERROR: parallel mode uses isolated logical --device cuda:0 (only --dry-run may use --device cpu)" >&2
    exit 2
  fi
  PARALLEL_MODE=1
fi

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
fail() { printf '[%s] FAILED: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; exit 1; }

if [ -n "$CUDA_VISIBLE_DEVICES_ARG" ]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_ARG"
  log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (internal --device stays $DEVICE)"
fi

STOPPED_EARLY=0
ACTIVE_PIDS=()
handle_interrupt() {
  STOPPED_EARLY=1
  log "received interrupt signal; forwarding TERM to active workers; re-run with --resume \"$OUTPUT_ROOT\" to continue."
  local pid child
  for pid in "${ACTIVE_PIDS[@]}"; do
    # A background run_stage shell may be waiting on python/tee children.
    # Signal the children first so the diagnostic CLI can flush its current
    # group and return 130, then signal the wrapper itself.
    while read -r child; do
      [ -n "$child" ] && kill -TERM "$child" 2>/dev/null || true
    done < <(pgrep -P "$pid" 2>/dev/null || true)
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap handle_interrupt INT TERM

# ── locate a usable python (torch import must succeed) ─────────────────────
# Non-interactive shells (cron, CI runners, some SSH invocations) often do
# NOT have conda on PATH at all even though it is installed and this exact
# host normally uses it interactively -- silently falling through to the
# system "python" (no torch) used to fail deep inside preflight with an
# unhelpful ModuleNotFoundError. Resolution order: explicit PYTHON_BIN env
# var override > conda "ptest" env activation (if conda IS on PATH) > a
# working `python` already on PATH (the production Docker image exposes
# /opt/ptest/bin/python this way) > common ptest/conda install locations >
# fail with an actionable message. Every candidate is verified to actually
# import torch before being accepted -- "on PATH" alone is not enough.
_try_python() { command -v "$1" >/dev/null 2>&1 && "$1" -c "import torch" >/dev/null 2>&1; }

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -n "$PYTHON_BIN" ]; then
  log "using PYTHON_BIN from environment: $PYTHON_BIN"
elif command -v conda >/dev/null 2>&1; then
  if [ "${CONDA_DEFAULT_ENV:-}" != "ptest" ] && conda env list 2>/dev/null | grep -qE '(^|[[:space:]])ptest([[:space:]]|$)'; then
    log "activating conda env 'ptest'"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ptest
  fi
  PYTHON_BIN="python"
fi

if [ -z "$PYTHON_BIN" ] || ! _try_python "$PYTHON_BIN"; then
  log "conda env unavailable or unusable -- searching PATH and common ptest/conda locations"
  for cand in \
    "python" \
    "/opt/ptest/bin/python" \
    "$HOME/anaconda3/envs/ptest/bin/python" \
    "$HOME/miniconda3/envs/ptest/bin/python" \
    "$HOME/miniforge3/envs/ptest/bin/python" \
    "/opt/conda/envs/ptest/bin/python" \
    "/usr/local/anaconda3/envs/ptest/bin/python"; do
    if _try_python "$cand"; then
      PYTHON_BIN="$cand"
      log "found working interpreter: $PYTHON_BIN"
      break
    fi
  done
fi

if [ -z "$PYTHON_BIN" ] || ! _try_python "$PYTHON_BIN"; then
  fail "no python interpreter that can 'import torch' was found (checked PYTHON_BIN, conda env 'ptest', PATH, /opt/ptest, and common conda install paths). Set PYTHON_BIN=/path/to/python explicitly, or activate the ptest environment before running this script."
fi
log "python interpreter: $PYTHON_BIN ($("$PYTHON_BIN" -c 'import torch,sys; print(f"torch={torch.__version__} py={sys.version.split()[0]}")'))"

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
  local preflight_visible_devices="${CUDA_VISIBLE_DEVICES:-}"
  local expected_device_count=""
  local cuda_visibility_env=()
  if [ "$PARALLEL_MODE" -eq 1 ]; then
    preflight_visible_devices="$PARALLEL_DEVICES_ARG"
    expected_device_count="${#PARALLEL_GPU_IDS[@]}"
  fi
  if [ -n "$preflight_visible_devices" ]; then
    cuda_visibility_env=(env "CUDA_VISIBLE_DEVICES=$preflight_visible_devices")
  fi
  if ! cuda_check="$("${cuda_visibility_env[@]}" env \
      F32DIG_DEVICE="$DEVICE" F32DIG_EXPECTED_DEVICE_COUNT="$expected_device_count" \
      "$PYTHON_BIN" - <<'PYEOF' 2>&1
import os
import torch
available = torch.cuda.is_available()
print(f"cuda_available={available}")
if not available:
    raise SystemExit(1)
count = torch.cuda.device_count()
print(f"device_count={count}")
expected = os.environ.get("F32DIG_EXPECTED_DEVICE_COUNT")
if expected and count != int(expected):
    raise SystemExit(f"expected {expected} visible GPUs, got {count}")
device = os.environ["F32DIG_DEVICE"]
index = int(device.split(":", 1)[1]) if ":" in device else 0
if index < 0 or index >= count:
    raise SystemExit(f"invalid CUDA device ordinal: {device}")
print(f"selected_device={device} device_name={torch.cuda.get_device_name(index)}")
print("visible_device_names=" + ",".join(torch.cuda.get_device_name(i) for i in range(count)))
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
elif [ "$RESUME_REQUESTED" -eq 1 ]; then
  log "resuming existing output root: $OUTPUT_ROOT"
else
  log "using output root: $OUTPUT_ROOT (fresh run -- each stage's diagnose_float32_digital_quality.py "
  log "  invocation will itself refuse if this directory already has completed results; pass --resume "
  log "  explicitly to continue an interrupted run instead)"
fi
mkdir -p "$OUTPUT_ROOT"
# --resume is passed to every stage ONLY when the user explicitly asked for
# it (RESUME_REQUESTED=1, set only by --resume, never by --output-root or by
# the directory already existing) -- a fresh/new run must let each stage's
# python CLI refuse a non-empty output-root on its own rather than the shell
# silently making every invocation "a resume".
RESUME_FLAG=""
[ "$RESUME_REQUESTED" -eq 1 ] && RESUME_FLAG="--resume"

if [ "$PARALLEL_MODE" -eq 1 ]; then
  log "3-GPU parallel mode: physical GPUs $PARALLEL_DEVICES_ARG (each worker sees logical cuda:0)"
  log "parallel phase A: stage3->GPU${PARALLEL_GPU_IDS[0]}, stage4->GPU${PARALLEL_GPU_IDS[1]}, stage5->GPU${PARALLEL_GPU_IDS[2]}"
  log "parallel phase B: three core-condition videos run one per GPU"
fi
if [ "$DRY_RUN" -eq 0 ]; then
  F32DIG_OUTPUT_ROOT="$OUTPUT_ROOT" F32DIG_PROFILE="$PROFILE" \
  F32DIG_PARALLEL_DEVICES="$PARALLEL_DEVICES_ARG" F32DIG_SEQUENTIAL_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_ARG" \
  F32DIG_DEVICE="$DEVICE" F32DIG_SEED="$SEED" F32DIG_DATASET_ROOT="$DATASET_ROOT" \
  F32DIG_FIXED_REFERENCE_SNR_DB="$FIXED_REFERENCE_SNR_DB" \
  F32DIG_RESUME="$RESUME_REQUESTED" \
  "$PYTHON_BIN" - <<'PYEOF' || \
    fail "execution_plan.json is missing or differs from this invocation; keep the original mode/profile/devices/seed/dataset or use a fresh output root"
import json
import os
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, "src")
from sgdjscc_lab.utils.run_manifest import get_git_state

root = Path(os.environ["F32DIG_OUTPUT_ROOT"])
parallel_text = os.environ.get("F32DIG_PARALLEL_DEVICES", "")
devices = parallel_text.split(",") if parallel_text else []
parallel = bool(devices)
plan = {
    "schema_version": 2,
    "mode": "three_gpu_stage_and_video_parallel" if parallel else "sequential",
    "git": get_git_state(Path.cwd()),
    "profile": os.environ["F32DIG_PROFILE"],
    "physical_devices": devices if parallel else os.environ.get("F32DIG_SEQUENTIAL_VISIBLE_DEVICES") or "inherited",
    "logical_worker_device": "cuda:0" if parallel else os.environ["F32DIG_DEVICE"],
    "seed": int(os.environ["F32DIG_SEED"]),
    "fixed_reference_snr_db": float(os.environ["F32DIG_FIXED_REFERENCE_SNR_DB"]),
    "dataset_root": str(Path(os.environ["F32DIG_DATASET_ROOT"]).resolve()),
    "phase_a": ({
        "stage3_single_frame_paths": devices[0],
        "stage4_single_frame_ablations": devices[1],
        "stage5_paired_frames": devices[2],
    } if parallel else "sequential"),
    "phase_b": ({
        "01_person_walk": devices[0],
        "07_person_enter": devices[1],
        "09_scene_cut_chair_car": devices[2],
    } if parallel else "sequential"),
}
path = root / "execution_plan.json"
if path.exists():
    if json.loads(path.read_text(encoding="utf-8")) != plan:
        raise SystemExit(1)
else:
    if os.environ["F32DIG_RESUME"] == "1":
        raise SystemExit("resume requires the original execution_plan.json")
    fd, temporary = tempfile.mkstemp(dir=str(root), prefix=".execution_plan.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
PYEOF
fi

# ── profile-dependent scope ─────────────────────────────────────────────
case "$PROFILE" in
  smoke)
    FRAMES_PAIRED="0-1"        # stage 5: 1 video x 2 frames
    FRAMES_CORE="0-2"          # stage 6: 3 videos x 3 frames
    ;;
  short)
    FRAMES_PAIRED="0-19"       # stage 5: 1 video x 20 frames (same depth as full)
    FRAMES_CORE="0-9"          # stage 6: 3 videos x 10 frames
    ;;
  full)
    FRAMES_PAIRED="0-19"       # stage 5: 1 video x 20 frames
    FRAMES_CORE="0-99"         # stage 6: 3 videos x 100 frames
    ;;
esac

DIAG_CMD=(
  "$PYTHON_BIN" scripts/diagnose_float32_digital_quality.py
  --dataset-root "$DATASET_ROOT" --device "$DEVICE" --seed "$SEED"
  --fixed-reference-snr-db "$FIXED_REFERENCE_SNR_DB"
)

STAGE_FAILURES=0
STAGE_LOG_DIR="$OUTPUT_ROOT/stage_logs"
mkdir -p "$STAGE_LOG_DIR"
run_stage() {
  local name="$1"; shift
  local log_file="$STAGE_LOG_DIR/${name}.log"
  local start_ts end_ts duration_s
  start_ts=$(date +%s)
  log "==== stage: $name ===="
  if [ "$DRY_RUN" -eq 1 ]; then
    local dry_rc
    if "$@" --dry-run; then
      return 0
    else
      dry_rc=$?
      log "dry-run stage '$name' FAILED (exit $dry_rc)"
      STAGE_FAILURES=$((STAGE_FAILURES + 1))
      return "$dry_rc"
    fi
  fi
  {
    echo "===== attempt at $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    echo "command: $*"
    echo "---"
  } >> "$log_file"
  set +o pipefail
  "$@" 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  set -o pipefail
  end_ts=$(date +%s)
  duration_s=$((end_ts - start_ts))
  {
    echo "---"
    echo "end: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "duration_seconds: $duration_s"
    echo "exit_code: $rc"
  } >> "$log_file"
  if [ "$rc" -eq 130 ]; then
    log "stage '$name' interrupted (exit 130) after ${duration_s}s; stopping remaining stages. Log: $log_file"
    STOPPED_EARLY=1
  elif [ "$rc" -ne 0 ]; then
    log "stage '$name' FAILED (exit $rc) after ${duration_s}s -- recorded; continuing with remaining independent stages. Log: $log_file"
    STAGE_FAILURES=$((STAGE_FAILURES + 1))
  else
    log "stage '$name' OK (${duration_s}s). Log: $log_file"
  fi
  return "$rc"
}

PARALLEL_PIDS=()
PARALLEL_NAMES=()
launch_parallel_stage() {
  local physical_gpu="$1" name="$2"; shift 2
  log "launching parallel stage '$name' on physical GPU $physical_gpu (logical cuda:0)"
  (
    export CUDA_VISIBLE_DEVICES="$physical_gpu"
    run_stage "$name" "$@"
  ) &
  PARALLEL_PIDS+=("$!")
  PARALLEL_NAMES+=("$name")
  ACTIVE_PIDS+=("$!")
}

wait_parallel_stages() {
  local index pid name rc any_interrupted=0
  for index in "${!PARALLEL_PIDS[@]}"; do
    pid="${PARALLEL_PIDS[$index]}"
    name="${PARALLEL_NAMES[$index]}"
    if wait "$pid"; then
      rc=0
    else
      rc=$?
    fi
    if [ "$rc" -eq 130 ] || [ "$rc" -eq 143 ]; then
      log "parallel stage '$name' interrupted (exit $rc)"
      any_interrupted=1
    elif [ "$rc" -ne 0 ]; then
      log "parallel stage '$name' failed (exit $rc)"
      STAGE_FAILURES=$((STAGE_FAILURES + 1))
    else
      log "parallel stage '$name' completed"
    fi
  done
  PARALLEL_PIDS=()
  PARALLEL_NAMES=()
  ACTIVE_PIDS=()
  if [ "$any_interrupted" -eq 1 ] || [ "$STOPPED_EARLY" -eq 1 ]; then
    STOPPED_EARLY=1
    return 130
  fi
  return 0
}

# ── stage 2: related tests + dry-run ────────────────────────────────────
TEST_FILES=(tests/test_float32_digital_diagnostics.py tests/test_receiver_runtime.py
            tests/test_packet_bundle.py tests/test_wire_packet.py tests/test_digital_step_matching.py)
if [ "$DRY_RUN" -eq 1 ]; then
  log "==== stage: 2_tests_and_dry_run ===="
  log "dry-run: would run: $PYTHON_BIN -m pytest ${TEST_FILES[*]} -q"
else
  # pytest has no --dry-run flag, so this stage is logged inline rather than
  # via run_stage() (which always appends --dry-run in dry-run mode).
  name="2_tests_and_dry_run"
  log_file="$STAGE_LOG_DIR/${name}.log"
  start_ts=$(date +%s)
  log "==== stage: $name ===="
  {
    echo "===== attempt at $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    echo "command: $PYTHON_BIN -m pytest ${TEST_FILES[*]} -q"
    echo "---"
  } >> "$log_file"
  set +o pipefail
  "$PYTHON_BIN" -m pytest "${TEST_FILES[@]}" -q 2>&1 | tee -a "$log_file"
  rc=${PIPESTATUS[0]}
  set -o pipefail
  end_ts=$(date +%s); duration_s=$((end_ts - start_ts))
  {
    echo "---"; echo "end: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "duration_seconds: $duration_s"; echo "exit_code: $rc"
  } >> "$log_file"
  if [ "$rc" -ne 0 ]; then
    log "stage '$name' FAILED (exit $rc) after ${duration_s}s -- recorded; continuing. Log: $log_file"
    STAGE_FAILURES=$((STAGE_FAILURES + 1))
  else
    log "stage '$name' OK (${duration_s}s). Log: $log_file"
  fi
fi
[ "$STOPPED_EARLY" -eq 1 ] && { log "stopped after stage 2"; exit 130; }

# ── stages 3/4/5: independent diagnostic scopes ─────────────────────────
if [ "$PARALLEL_MODE" -eq 1 ]; then
  launch_parallel_stage "${PARALLEL_GPU_IDS[0]}" "3_single_frame_path_contract" \
    "${DIAG_CMD[@]}" --output-root "$OUTPUT_ROOT/stage3_single_frame_paths" \
    --video-ids "$VIDEO_NORMAL_MOTION" --frames 0 --ablations baseline $RESUME_FLAG
  launch_parallel_stage "${PARALLEL_GPU_IDS[1]}" "4_single_frame_full_ablation" \
    "${DIAG_CMD[@]}" --output-root "$OUTPUT_ROOT/stage4_single_frame_ablations" \
    --video-ids "$VIDEO_NORMAL_MOTION" --frames 0 --ablations all $RESUME_FLAG
  launch_parallel_stage "${PARALLEL_GPU_IDS[2]}" "5_paired_multi_frame" \
    "${DIAG_CMD[@]}" --output-root "$OUTPUT_ROOT/stage5_paired_frames" \
    --video-ids "$VIDEO_NORMAL_MOTION" --frames "$FRAMES_PAIRED" \
    --ablations baseline,diffusion_bypass_vae_direct $RESUME_FLAG
  wait_parallel_stages || true
  [ "$STOPPED_EARLY" -eq 1 ] && { log "stopped during parallel stages 3/4/5"; exit 130; }
else
  run_stage "3_single_frame_path_contract" \
    "${DIAG_CMD[@]}" --output-root "$OUTPUT_ROOT/stage3_single_frame_paths" \
    --video-ids "$VIDEO_NORMAL_MOTION" --frames 0 --ablations baseline $RESUME_FLAG
  [ "$STOPPED_EARLY" -eq 1 ] && { log "stopped after stage 3"; exit 130; }

  run_stage "4_single_frame_full_ablation" \
    "${DIAG_CMD[@]}" --output-root "$OUTPUT_ROOT/stage4_single_frame_ablations" \
    --video-ids "$VIDEO_NORMAL_MOTION" --frames 0 --ablations all $RESUME_FLAG
  [ "$STOPPED_EARLY" -eq 1 ] && { log "stopped after stage 4"; exit 130; }

  run_stage "5_paired_multi_frame" \
    "${DIAG_CMD[@]}" --output-root "$OUTPUT_ROOT/stage5_paired_frames" \
    --video-ids "$VIDEO_NORMAL_MOTION" --frames "$FRAMES_PAIRED" \
    --ablations baseline,diffusion_bypass_vae_direct $RESUME_FLAG
  [ "$STOPPED_EARLY" -eq 1 ] && { log "stopped after stage 5"; exit 130; }
fi

# ── stage 6: 3 independent core-condition videos, baseline only ────────
if [ "$PARALLEL_MODE" -eq 1 ]; then
  launch_parallel_stage "${PARALLEL_GPU_IDS[0]}" "6_core_normal_motion" \
    "${DIAG_CMD[@]}" --output-root "$OUTPUT_ROOT/stage6_core_conditions/worker_00_normal_motion" \
    --video-ids "$VIDEO_NORMAL_MOTION" --frames "$FRAMES_CORE" \
    --ablations baseline --no-instrument-tensors $RESUME_FLAG
  launch_parallel_stage "${PARALLEL_GPU_IDS[1]}" "6_core_semantic_change" \
    "${DIAG_CMD[@]}" --output-root "$OUTPUT_ROOT/stage6_core_conditions/worker_01_semantic_change" \
    --video-ids "$VIDEO_SEMANTIC_CHANGE" --frames "$FRAMES_CORE" \
    --ablations baseline --no-instrument-tensors $RESUME_FLAG
  launch_parallel_stage "${PARALLEL_GPU_IDS[2]}" "6_core_scene_cut" \
    "${DIAG_CMD[@]}" --output-root "$OUTPUT_ROOT/stage6_core_conditions/worker_02_scene_cut" \
    --video-ids "$VIDEO_SCENE_CUT" --frames "$FRAMES_CORE" \
    --ablations baseline --no-instrument-tensors $RESUME_FLAG
  wait_parallel_stages || true
  [ "$STOPPED_EARLY" -eq 1 ] && { log "stopped during parallel stage 6"; exit 130; }
else
  run_stage "6_core_conditions" \
    "${DIAG_CMD[@]}" --output-root "$OUTPUT_ROOT/stage6_core_conditions" \
    --video-ids "$VIDEO_NORMAL_MOTION,$VIDEO_SEMANTIC_CHANGE,$VIDEO_SCENE_CUT" \
    --frames "$FRAMES_CORE" --ablations baseline --no-instrument-tensors $RESUME_FLAG
  [ "$STOPPED_EARLY" -eq 1 ] && { log "stopped after stage 6"; exit 130; }
fi

if [ "$DRY_RUN" -eq 1 ]; then
  if [ "$STAGE_FAILURES" -gt 0 ]; then
    log "dry-run completed with $STAGE_FAILURES failed stage plan(s)."
    exit 3
  fi
  log "dry-run complete for all stages."
  exit 0
fi

# ── stage 7: validate results, hash, integrated report ──────────────────
log "==== stage: 7_validate_and_report ===="
if [ "$PARALLEL_MODE" -eq 1 ]; then
  STAGE7_DIRS="$OUTPUT_ROOT/stage3_single_frame_paths $OUTPUT_ROOT/stage4_single_frame_ablations $OUTPUT_ROOT/stage5_paired_frames $OUTPUT_ROOT/stage6_core_conditions/worker_00_normal_motion $OUTPUT_ROOT/stage6_core_conditions/worker_01_semantic_change $OUTPUT_ROOT/stage6_core_conditions/worker_02_scene_cut"
else
  STAGE7_DIRS="$OUTPUT_ROOT/stage3_single_frame_paths $OUTPUT_ROOT/stage4_single_frame_ablations $OUTPUT_ROOT/stage5_paired_frames $OUTPUT_ROOT/stage6_core_conditions"
fi
F32DIG_PROFILE="$PROFILE" F32DIG_OUTPUT_ROOT="$OUTPUT_ROOT" F32DIG_STAGE_FAILURES="$STAGE_FAILURES" \
F32DIG_STAGE_DIRS="$STAGE7_DIRS" F32DIG_PARALLEL_DEVICES="$PARALLEL_DEVICES_ARG" \
"$PYTHON_BIN" - <<'PYEOF'
import csv
import json
import math
import os
import sys
import hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

output_root = Path(os.environ["F32DIG_OUTPUT_ROOT"])
profile = os.environ["F32DIG_PROFILE"]
stage_failures = os.environ["F32DIG_STAGE_FAILURES"]
stage_dirs = [Path(p) for p in os.environ["F32DIG_STAGE_DIRS"].split() if p]
parallel_devices = os.environ.get("F32DIG_PARALLEL_DEVICES") or None

AUXILIARY_ABLATIONS = ("serialized_raw_edge", "awgn_edge_retransmit")
EVIDENCE_RANK = {
    "baseline_pending_vae_direct": 0,
    "baseline_only": 1,
    "baseline_with_vae_direct": 2,
    "auxiliary_edge_equalized": 1,
    "legacy_unspecified": 1,
}
QUALITY_GAP_PSNR_DB = 1.0
PATH_PARITY_PSNR_DB = 0.01


def logical_key(row):
    frame = row["frame"]
    try:
        frame = int(frame)
    except (TypeError, ValueError):
        pass
    return (row["video"], frame, row["ablation"])


def canonical_verdict(row):
    """Migrate the old overloaded label without changing raw evidence.

    Runs produced before ``no_issue_detected`` existed called both missing
    evidence and an observed sub-threshold quality gap ``inconclusive``.
    The reason string unambiguously distinguishes the latter.
    """
    verdict = row["verdict"]
    reason = str(row.get("reason", ""))
    if row.get("ablation") in AUXILIARY_ABLATIONS and verdict == "packet_tx_rx_issue":
        return "transport_delta_detected"
    if verdict == "inconclusive" and "no quality problem evidenced" in reason:
        return "no_issue_detected"
    return verdict


def metric_only_assessment(rows):
    """Assess a baseline group that intentionally has no tensor verdict."""
    by_path = {row.get("path"): row for row in rows if row.get("failed", "False").lower() != "true"}
    required = ("awgn", "digital_inprocess", "digital_wire")
    if any(path not in by_path for path in required):
        return {"status": "insufficient_evidence", "reason": "one or more required paths are missing"}
    try:
        awgn = float(by_path["awgn"]["psnr"])
        inprocess = float(by_path["digital_inprocess"]["psnr"])
        wire = float(by_path["digital_wire"]["psnr"])
    except (KeyError, TypeError, ValueError):
        return {"status": "insufficient_evidence", "reason": "PSNR is missing or non-numeric"}
    if not all(math.isfinite(value) for value in (awgn, inprocess, wire)):
        return {"status": "insufficient_evidence", "reason": "PSNR is non-finite"}
    gap = awgn - wire
    parity_delta = abs(inprocess - wire)
    bitexact = str(by_path["digital_wire"].get("roundtrip_bitexact", "")).lower() == "true"
    if not bitexact or parity_delta > PATH_PARITY_PSNR_DB:
        status = "path_metric_divergence_detected"
    elif gap >= QUALITY_GAP_PSNR_DB:
        status = "quality_regression_detected"
    else:
        status = "no_issue_detected"
    return {
        "status": status,
        "awgn_psnr": awgn,
        "digital_inprocess_psnr": inprocess,
        "digital_wire_psnr": wire,
        "wire_minus_awgn_psnr": wire - awgn,
        "inprocess_wire_abs_psnr": parity_delta,
        "roundtrip_bitexact": bitexact,
    }


def evidence_level(row):
    """Return an explicit or backward-compatible verdict evidence scope."""
    explicit = row.get("evidence_level")
    if explicit:
        return explicit
    if row.get("ablation") in AUXILIARY_ABLATIONS:
        return "auxiliary_edge_equalized"
    if row.get("status", "final") != "final":
        return "baseline_pending_vae_direct"
    if row.get("vae_direct_considered"):
        return "baseline_with_vae_direct"
    return "baseline_only"

lines = []
lines.append(f"# float32 digital diagnostics — integrated run ({profile} profile)")
lines.append("")
lines.append(f"- generated: {datetime.now(timezone.utc).isoformat()}")
lines.append(f"- output_root: {output_root}")
lines.append(f"- stage_failures: {stage_failures}")
lines.append(f"- execution_mode: {'three_gpu_parallel' if parallel_devices else 'sequential'}")
if parallel_devices:
    lines.append(f"- physical_devices: {parallel_devices}")
lines.append("")
lines.append("**The root-cause section consolidates each instrumented stage's own verdict (see docs/protocols/"
             "float32_digital_diagnostics.md for the classification criteria); it is NOT itself "
             "a new judgment, only a rollup of what each stage's own summary.json/verdicts.jsonl "
             "already recorded. Only `ablation == \"baseline\"` AND `status == \"final\"` rows feed "
             "any dominant-verdict tally below -- auxiliary edge-equalizing ablations and "
             "still-provisional baseline rows are listed separately, never summed into it. When stages "
             "overlap, the richest evidence wins (`baseline_with_vae_direct` > `baseline_only` > "
             "provisional); only disagreements at the same evidence level are conflicts. "
             "Stages run with `--no-instrument-tensors` have no root-cause verdict; their baseline "
             "CSV rows are instead included in a separate metric-only quality assessment using the "
             "documented 1 dB quality and 0.01 dB path-parity thresholds.**")
lines.append("")

# combined[key] = {"verdict", "status", "stage", "evidence_level"} -- ONE canonical entry per
# (video, frame, ablation) across ALL stages (stage3 and stage5, for
# example, both cover video 01/frame 0's baseline ablation, but stage5 also
# has VAE-direct evidence). Summing each stage's own verdict_summary counts
# directly would count that one frame twice. Precedence follows evidence
# richness: baseline_with_vae_direct > baseline_only > provisional. A verdict
# change while evidence gets richer is an expected scientific refinement, not
# a reproducibility conflict. Only DIFFERENT verdict labels at the SAME
# evidence level for the same key are flagged as conflicts.
combined = {}
combined_metric_only = {}
conflicts = []
per_stage_rows = []

for stage_dir in stage_dirs:
    if not stage_dir.is_dir():
        continue
    try:
        stage_label = str(stage_dir.relative_to(output_root))
    except ValueError:
        stage_label = stage_dir.name
    summary_path = stage_dir / "summary.json"
    n_frames = "?"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        n_frames = summary.get("counts", {}).get("n_frames_processed", "?")

    verdicts_path = stage_dir / "verdicts.jsonl"
    stage_baseline_final_counts = Counter()
    stage_n_provisional = 0
    stage_n_auxiliary = 0
    stage_evidence_counts = Counter()
    stage_latest = {}
    if verdicts_path.exists():
        # verdicts.jsonl is append-only: provisional -> final upgrades append a
        # new line for the same logical key. Reduce to the LAST line per key
        # before either per-stage counting or cross-stage merging; otherwise a
        # successfully upgraded verdict would still be reported as one stale
        # provisional row in the per-stage table.
        with verdicts_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = logical_key(row)
                stage_latest[key] = row

        for key, row in stage_latest.items():
            status = row.get("status", "final")
            level = evidence_level(row)
            entry = {
                "verdict": canonical_verdict(row), "status": status, "stage": stage_label,
                "evidence_level": level,
            }
            existing = combined.get(key)
            if existing is None:
                combined[key] = entry
            else:
                old_rank = EVIDENCE_RANK.get(existing["evidence_level"], 0)
                new_rank = EVIDENCE_RANK.get(level, 0)
                if new_rank > old_rank:
                    combined[key] = entry
                elif new_rank < old_rank:
                    pass
                elif existing["verdict"] != entry["verdict"]:
                    conflicts.append({
                        "key": key, "verdict_a": existing["verdict"], "stage_a": existing["stage"],
                        "verdict_b": entry["verdict"], "stage_b": entry["stage"],
                        "status": status, "evidence_level": level,
                    })
                    # Equal evidence level: keep the earlier stage
                    # deterministically, but surface and fail on the anomaly.

            if row["ablation"] == "baseline":
                stage_evidence_counts[level] += 1
                if status == "final":
                    stage_baseline_final_counts[canonical_verdict(row)] += 1
                else:
                    stage_n_provisional += 1
            elif row["ablation"] in AUXILIARY_ABLATIONS:
                stage_n_auxiliary += 1

    stage_metric_counts = Counter()
    path_csv = stage_dir / "path_comparison.csv"
    if path_csv.exists():
        metric_groups = {}
        with path_csv.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("ablation") != "baseline":
                    continue
                key = logical_key(row)
                metric_groups.setdefault(key, []).append(row)
        for key, rows in metric_groups.items():
            # Instrumented groups already have a richer tensor/root-cause
            # verdict. Metric-only assessment exists specifically to include
            # stage 6 and equivalent --no-instrument-tensors runs.
            if key in stage_latest:
                continue
            assessment = metric_only_assessment(rows)
            assessment["stage"] = stage_label
            existing = combined_metric_only.get(key)
            if existing is None:
                combined_metric_only[key] = assessment
            elif (
                {k: v for k, v in existing.items() if k != "stage"}
                != {k: v for k, v in assessment.items() if k != "stage"}
            ):
                conflicts.append({
                    "key": key,
                    "verdict_a": existing["status"], "stage_a": existing["stage"],
                    "verdict_b": assessment["status"], "stage_b": stage_label,
                    "status": "metric_only", "evidence_level": "path_metrics",
                })
            stage_metric_counts[assessment["status"]] += 1

    failed_csv = stage_dir / "failed_cases.csv"
    n_failed = max(0, sum(1 for _ in failed_csv.open(encoding="utf-8")) - 1) if failed_csv.exists() else 0
    stage_dominant = (
        max(stage_baseline_final_counts, key=stage_baseline_final_counts.get)
        if stage_baseline_final_counts else None
    )
    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(stage_baseline_final_counts.items())) or "(none)"
    evidence_str = ", ".join(f"{k}={v}" for k, v in sorted(stage_evidence_counts.items())) or "(none)"
    metric_str = ", ".join(f"{k}={v}" for k, v in sorted(stage_metric_counts.items())) or "(none)"
    per_stage_rows.append(
        f"| {stage_label} | {n_frames} | {stage_dominant or '(none)'} | {counts_str} "
        f"| {metric_str} | {evidence_str} | {stage_n_provisional} | {stage_n_auxiliary} | {n_failed} |"
    )

lines.append("## per-stage verdict summary (baseline, final only)")
lines.append("")
lines.append("| stage | n_frames_processed | dominant_verdict | baseline verdict counts | metric-only quality | evidence levels | provisional | auxiliary | failed_cases |")
lines.append("|---|---:|---|---|---|---|---:|---:|---:|")
lines.extend(per_stage_rows)

lines.append("")
lines.append("## overall (baseline, final only, deduplicated across stages by (video, frame))")
lines.append("")
overall_counts = Counter(
    entry["verdict"] for key, entry in combined.items()
    if key[2] == "baseline" and entry["status"] == "final"
)
n_provisional_overall = sum(
    1 for key, entry in combined.items() if key[2] == "baseline" and entry["status"] != "final"
)
if overall_counts:
    conclusive = {k: v for k, v in overall_counts.items() if k != "inconclusive"}
    overall_dominant = max(conclusive, key=conclusive.get) if conclusive else None
    lines.append(f"- dominant_verdict: `{overall_dominant or 'inconclusive'}`")
    for label, count in sorted(overall_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  - `{label}`: {count}")
else:
    lines.append("- no final baseline verdicts recorded across any stage (tensor instrumentation may "
                 "have been disabled, no stage completed, or all baseline verdicts are still provisional).")
if n_provisional_overall:
    lines.append(f"- provisional baseline verdicts (excluded above, waiting on VAE-direct evidence): {n_provisional_overall}")

lines.append("")
lines.append("## metric-only baseline quality assessment (no tensor/root-cause verdict)")
lines.append("")
lines.append(f"- thresholds: AWGN − digital wire PSNR < {QUALITY_GAP_PSNR_DB} dB; "
             f"|digital in-process − wire PSNR| <= {PATH_PARITY_PSNR_DB} dB; wire round-trip bit-exact")
metric_only_counts = Counter(item["status"] for item in combined_metric_only.values())
if metric_only_counts:
    for label, count in sorted(metric_only_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{label}`: {count}")
    numeric = [item for item in combined_metric_only.values() if "wire_minus_awgn_psnr" in item]
    if numeric:
        deltas = [item["wire_minus_awgn_psnr"] for item in numeric]
        parity = [item["inprocess_wire_abs_psnr"] for item in numeric]
        lines.append(f"- digital wire − AWGN PSNR: mean={sum(deltas)/len(deltas):.6f} dB, "
                     f"min={min(deltas):.6f} dB, max={max(deltas):.6f} dB")
        lines.append(f"- max |digital in-process − wire PSNR|: {max(parity):.9f} dB")
        lines.append(f"- bit-exact wire rows: {sum(item['roundtrip_bitexact'] for item in numeric)}/{len(numeric)}")
else:
    lines.append("- none recorded; all completed baseline groups had instrumented root-cause verdicts.")

lines.append("")
lines.append("## auxiliary evidence (serialized_raw_edge / awgn_edge_retransmit, never summed into the overall count)")
lines.append("")
auxiliary_counts = Counter(
    entry["verdict"] for key, entry in combined.items() if key[2] in AUXILIARY_ABLATIONS
)
if auxiliary_counts:
    for label, count in sorted(auxiliary_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{label}`: {count}")
else:
    lines.append("- none recorded (no stage ran serialized_raw_edge/awgn_edge_retransmit).")

lines.append("")
lines.append("## conflicts")
lines.append("")
if conflicts:
    lines.append(
        f"**{len(conflicts)} conflicting verdict(s) detected** -- the SAME (video, frame, ablation) "
        "got DIFFERENT verdict labels at the same evidence level from different stages. This is a "
        "reproducibility anomaly (non-determinism or a config mismatch between stages), not expected "
        "behavior -- treated as a stage-7 failure below."
    )
    for c in conflicts:
        lines.append(
            f"- `{c['key']}` ({c['status']}, `{c['evidence_level']}`): "
            f"`{c['verdict_a']}` (stage `{c['stage_a']}`) vs. "
            f"`{c['verdict_b']}` (stage `{c['stage_b']}`)"
        )
else:
    lines.append("- none detected.")

lines.append("")
lines.append("## run-level artifact hashes (sha256)")
lines.append("")
for name in ("execution_plan.json",):
    path = output_root / name
    if path.exists():
        lines.append(f"- `{name}`: `{hashlib.sha256(path.read_bytes()).hexdigest()}`")

lines.append("")
lines.append("## per-stage artifact hashes (sha256)")
for stage_dir in stage_dirs:
    if not stage_dir.is_dir():
        continue
    lines.append("")
    try:
        stage_label = str(stage_dir.relative_to(output_root))
    except ValueError:
        stage_label = stage_dir.name
    lines.append(f"### {stage_label}")
    for f in ("run_manifest.json", "summary.json", "REPORT.md", "path_comparison.csv",
              "failed_cases.csv", "verdicts.jsonl"):
        fp = stage_dir / f
        if fp.exists():
            h = hashlib.sha256(fp.read_bytes()).hexdigest()
            lines.append(f"- `{f}`: `{h}`")

(output_root / "INTEGRATED_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {output_root / 'INTEGRATED_REPORT.md'}")
if conflicts:
    print(f"CONFLICTS_DETECTED: {len(conflicts)}", file=sys.stderr)
    sys.exit(3)
PYEOF
python_rc=$?

if [ "$python_rc" -eq 0 ] && [ -s "$OUTPUT_ROOT/INTEGRATED_REPORT.md" ]; then
  log "wrote $OUTPUT_ROOT/INTEGRATED_REPORT.md"
elif [ "$python_rc" -eq 3 ] && [ -s "$OUTPUT_ROOT/INTEGRATED_REPORT.md" ]; then
  log "wrote $OUTPUT_ROOT/INTEGRATED_REPORT.md but detected verdict CONFLICTS across stages -- see its '## conflicts' section; treating as a failure."
  STAGE_FAILURES=$((STAGE_FAILURES + 1))
else
  log "stage '7_validate_and_report' FAILED (exit $python_rc) -- INTEGRATED_REPORT.md was not written (or is empty); per-stage REPORT.md/summary.json files under each stageN_*/ are still valid on their own."
  STAGE_FAILURES=$((STAGE_FAILURES + 1))
fi

log "done. Results in $OUTPUT_ROOT:"
log "  stage3_single_frame_paths/     -- 1 video x 1 frame, 3 paths, full tensor contract"
log "  stage4_single_frame_ablations/ -- 1 video x 1 frame, full ablation set"
log "  stage5_paired_frames/          -- 1 video x N frames, paired path comparison"
if [ "$PARALLEL_MODE" -eq 1 ]; then
  log "  stage6_core_conditions/worker_* -- 3 core-condition videos, one independent GPU worker each"
  log "  execution_plan.json            -- execution mode, physical GPU assignment, and resume contract"
else
  log "  stage6_core_conditions/        -- 3 core-condition videos x N frames, metrics only"
fi
log "  INTEGRATED_REPORT.md           -- per-stage artifact hashes"
log "  each stageN_*/REPORT.md        -- per-stage verdict + evidence (see docs/protocols/float32_digital_diagnostics.md)"

if [ "$STAGE_FAILURES" -gt 0 ]; then
  log "COMPLETED_WITH_FAILURES: $STAGE_FAILURES stage(s) failed -- see logs above."
  exit 3
fi
exit 0

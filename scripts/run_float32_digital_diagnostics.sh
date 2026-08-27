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

# ── locate a usable python (torch import must succeed) ─────────────────────
# Non-interactive shells (cron, CI runners, some SSH invocations) often do
# NOT have conda on PATH at all even though it is installed and this exact
# host normally uses it interactively -- silently falling through to the
# system "python" (no torch) used to fail deep inside preflight with an
# unhelpful ModuleNotFoundError. Resolution order: explicit PYTHON_BIN env
# var override > conda "ptest" env activation (if conda IS on PATH) > a
# fixed list of common conda install locations for envs/ptest/bin/python >
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
  log "conda not on PATH or 'python' cannot import torch -- searching common conda install locations"
  for cand in \
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
  fail "no python interpreter that can 'import torch' was found (checked PYTHON_BIN, conda env 'ptest', and common conda install paths). Set PYTHON_BIN=/path/to/envs/ptest/bin/python explicitly, or activate the ptest environment before running this script."
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
STAGE_LOG_DIR="$OUTPUT_ROOT/stage_logs"
mkdir -p "$STAGE_LOG_DIR"
run_stage() {
  local name="$1"; shift
  local log_file="$STAGE_LOG_DIR/${name}.log"
  local start_ts end_ts duration_s
  start_ts=$(date +%s)
  log "==== stage: $name ===="
  if [ "$DRY_RUN" -eq 1 ]; then
    "$@" --dry-run
    return $?
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
STAGE7_DIRS="$OUTPUT_ROOT/stage3_single_frame_paths $OUTPUT_ROOT/stage4_single_frame_ablations $OUTPUT_ROOT/stage5_paired_frames $OUTPUT_ROOT/stage6_core_conditions"
if F32DIG_PROFILE="$PROFILE" F32DIG_OUTPUT_ROOT="$OUTPUT_ROOT" F32DIG_STAGE_FAILURES="$STAGE_FAILURES" \
   F32DIG_STAGE_DIRS="$STAGE7_DIRS" "$PYTHON_BIN" - <<'PYEOF'
import json
import os
import hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

output_root = Path(os.environ["F32DIG_OUTPUT_ROOT"])
profile = os.environ["F32DIG_PROFILE"]
stage_failures = os.environ["F32DIG_STAGE_FAILURES"]
stage_dirs = [Path(p) for p in os.environ["F32DIG_STAGE_DIRS"].split() if p]

lines = []
lines.append(f"# float32 digital diagnostics — integrated run ({profile} profile)")
lines.append("")
lines.append(f"- generated: {datetime.now(timezone.utc).isoformat()}")
lines.append(f"- output_root: {output_root}")
lines.append(f"- stage_failures: {stage_failures}")
lines.append("")
lines.append("**This section consolidates each stage's own verdict (see docs/protocols/"
             "float32_digital_diagnostics.md for the classification criteria); it is NOT itself "
             "a new judgment, only a rollup of what each stage's own summary.json/verdicts.jsonl "
             "already recorded.**")
lines.append("")
lines.append("## per-stage verdict summary")
lines.append("")
lines.append("| stage | n_frames_processed | dominant_verdict | verdict counts | failed_cases |")
lines.append("|---|---:|---|---|---:|")

# (video, frame, ablation-kind) -> verdict label, deduplicated ACROSS stages
# -- stage3 and stage5 (for example) both cover video 01/frame 0's baseline
# ablation with the identical config, so summing each stage's own
# verdict_summary counts directly would count that one verdict twice. Reading
# each stage's verdicts.jsonl (not its pre-aggregated summary.json counts)
# and keying by the full (video, frame, ablation) triple across ALL stages
# combined lets the "overall" section count each real verdict exactly once.
combined_verdicts = {}
for stage_dir in stage_dirs:
    if not stage_dir.is_dir():
        continue
    summary_path = stage_dir / "summary.json"
    n_frames = "?"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        n_frames = summary.get("counts", {}).get("n_frames_processed", "?")

    verdicts_path = stage_dir / "verdicts.jsonl"
    stage_counts = Counter()
    if verdicts_path.exists():
        with verdicts_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                stage_counts[row["verdict"]] += 1
                combined_verdicts[(row["video"], row["frame"], row["ablation"])] = row["verdict"]

    failed_csv = stage_dir / "failed_cases.csv"
    n_failed = max(0, sum(1 for _ in failed_csv.open(encoding="utf-8")) - 1) if failed_csv.exists() else 0
    stage_conclusive = {k: v for k, v in stage_counts.items() if k != "inconclusive"}
    stage_dominant = max(stage_conclusive, key=stage_conclusive.get) if stage_conclusive else None
    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(stage_counts.items())) or "(none)"
    lines.append(f"| {stage_dir.name} | {n_frames} | {stage_dominant or 'inconclusive'} | {counts_str} | {n_failed} |")

lines.append("")
lines.append("## overall (all stages combined, deduplicated by (video, frame, ablation))")
lines.append("")
overall_counts = Counter(combined_verdicts.values())
if overall_counts:
    conclusive = {k: v for k, v in overall_counts.items() if k != "inconclusive"}
    overall_dominant = max(conclusive, key=conclusive.get) if conclusive else None
    lines.append(f"- dominant_verdict: `{overall_dominant or 'inconclusive'}`")
    for label, count in sorted(overall_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  - `{label}`: {count}")
else:
    lines.append("- no verdicts recorded across any stage (tensor instrumentation may have been "
                 "disabled, or no stage completed).")

lines.append("")
lines.append("## per-stage artifact hashes (sha256)")
for stage_dir in stage_dirs:
    if not stage_dir.is_dir():
        continue
    lines.append("")
    lines.append(f"### {stage_dir.name}")
    for f in ("run_manifest.json", "summary.json", "REPORT.md", "path_comparison.csv",
              "failed_cases.csv", "verdicts.jsonl"):
        fp = stage_dir / f
        if fp.exists():
            h = hashlib.sha256(fp.read_bytes()).hexdigest()
            lines.append(f"- `{f}`: `{h}`")

(output_root / "INTEGRATED_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {output_root / 'INTEGRATED_REPORT.md'}")
PYEOF
then
  integrated_report_ok=1
else
  integrated_report_ok=0
fi

if [ "$integrated_report_ok" -eq 1 ] && [ -s "$OUTPUT_ROOT/INTEGRATED_REPORT.md" ]; then
  log "wrote $OUTPUT_ROOT/INTEGRATED_REPORT.md"
else
  log "stage '7_validate_and_report' FAILED -- INTEGRATED_REPORT.md was not written (or is empty); per-stage REPORT.md/summary.json files under each stageN_*/ are still valid on their own."
  STAGE_FAILURES=$((STAGE_FAILURES + 1))
fi

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

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LIST_DIR="${LIST_DIR:-data/_lists/paper_like_multi}"
MODEL_ROOT="${MODEL_ROOT:-checkpoints}"
MUGE_REPR="${MUGE_REPR:-edge_uncertainty}"
GPUS_RAW="${GPUS:-0,1,2}"
IFS=', ' read -r -a GPUS <<< "$GPUS_RAW"

mkdir -p "$LIST_DIR"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/paper_like_stage23.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "No GPUs configured. Set GPUS, e.g. GPUS=0,1,2" >&2
  exit 1
fi

enqueue_work_units() {
  local out_file="$1"
  shift
  : > "$out_file"
  local root
  for root in "$@"; do
    [ -d "$root" ] || continue
    mapfile -t _subdirs < <(find "$root" -mindepth 1 -maxdepth 1 -type d | sort)
    if [ "${#_subdirs[@]}" -gt 0 ]; then
      printf '%s\n' "${_subdirs[@]}" >> "$out_file"
    else
      printf '%s\n' "$root" >> "$out_file"
    fi
  done
}

split_round_robin() {
  local src="$1"
  local prefix="$2"
  local idx=0
  local gpu_idx
  for gpu_idx in "${!GPUS[@]}"; do
    : > "${prefix}.${gpu_idx}"
  done
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    gpu_idx=$((idx % ${#GPUS[@]}))
    printf '%s\n' "$line" >> "${prefix}.${gpu_idx}"
    idx=$((idx + 1))
  done < "$src"
}

wait_for_pids() {
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

run_parallel_captions() {
  local worklist="$TMP_DIR/caption_units.txt"
  local prefix="$TMP_DIR/caption_units.part"
  local gpu_idx gpu units_file pid
  local -a pids=()

  enqueue_work_units "$worklist" \
    data/sa1b_images/train \
    data/sa1b_images/val
  split_round_robin "$worklist" "$prefix"

  echo "Caption workers: ${GPUS[*]}"
  for gpu_idx in "${!GPUS[@]}"; do
    units_file="${prefix}.${gpu_idx}"
    [ -s "$units_file" ] || continue
    gpu="${GPUS[$gpu_idx]}"
    (
      while IFS= read -r unit; do
        [ -n "$unit" ] || continue
        echo "[caption][gpu=$gpu] $unit"
        CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/generate_captions.py \
          --input "$unit" \
          --mode model \
          --device cuda:0
      done < "$units_file"
    ) &
    pids+=("$!")
  done

  wait_for_pids "${pids[@]}"
}

run_parallel_muge() {
  local worklist="$TMP_DIR/muge_units.txt"
  local prefix="$TMP_DIR/muge_units.part"
  local gpu_idx gpu units_file
  local -a pids=()

  enqueue_work_units "$worklist" \
    data/sa1b_images/train \
    data/sa1b_images/val \
    data/journey_pairs/train \
    data/journey_pairs/val \
    data/cc3m_pairs/train \
    data/cc3m_pairs/val \
    data/datacomp_pairs/train \
    data/datacomp_pairs/val \
    data/celeba_hq/train \
    data/celeba_hq/val
  split_round_robin "$worklist" "$prefix"

  echo "MuGE workers: ${GPUS[*]}"
  for gpu_idx in "${!GPUS[@]}"; do
    units_file="${prefix}.${gpu_idx}"
    [ -s "$units_file" ] || continue
    gpu="${GPUS[$gpu_idx]}"
    (
      while IFS= read -r unit; do
        [ -n "$unit" ] || continue
        echo "[muge][gpu=$gpu] $unit"
        CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/prepare_muge_edges.py \
          --input "$unit" \
          --model-root "$MODEL_ROOT" \
          --repr "$MUGE_REPR" \
          --device cuda:0
      done < "$units_file"
    ) &
    pids+=("$!")
  done

  wait_for_pids "${pids[@]}"
}

echo "[1/3] Generating BLIP-2 captions for SA-1B in parallel..."
run_parallel_captions

echo "[2/3] Building stage-2/3 file lists..."
find data/sa1b_images/train    -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | shuf -n 7000000 > "$LIST_DIR/sa1b_train.list"
find data/journey_pairs/train  -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | shuf -n 3000000 > "$LIST_DIR/journey_train.list"
find data/cc3m_pairs/train     -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | shuf -n 2000000 > "$LIST_DIR/cc3m_train.list"
find data/datacomp_pairs/train -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | shuf -n 2000000 > "$LIST_DIR/datacomp_train.list"
find data/celeba_hq/train      -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | shuf -n 30000   > "$LIST_DIR/celebahq_train.list"

cat \
  "$LIST_DIR/sa1b_train.list" \
  "$LIST_DIR/journey_train.list" \
  "$LIST_DIR/cc3m_train.list" \
  "$LIST_DIR/datacomp_train.list" \
  "$LIST_DIR/celebahq_train.list" \
  > "$LIST_DIR/stage23_train.list"

find data/sa1b_images/val    -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | shuf -n 5000 > "$LIST_DIR/sa1b_val.list"
find data/journey_pairs/val  -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | shuf -n 2000 > "$LIST_DIR/journey_val.list"
find data/cc3m_pairs/val     -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | shuf -n 2000 > "$LIST_DIR/cc3m_val.list"
find data/datacomp_pairs/val -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | shuf -n 2000 > "$LIST_DIR/datacomp_val.list"
find data/celeba_hq/val      -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | shuf -n 1000 > "$LIST_DIR/celebahq_val.list"

cat \
  "$LIST_DIR/sa1b_val.list" \
  "$LIST_DIR/journey_val.list" \
  "$LIST_DIR/cc3m_val.list" \
  "$LIST_DIR/datacomp_val.list" \
  "$LIST_DIR/celebahq_val.list" \
  > "$LIST_DIR/stage23_val.list"

echo "[3/3] Precomputing MuGE sidecars in parallel..."
run_parallel_muge

echo "Prepared paper-like stage-2/3 data under $LIST_DIR"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LIST_DIR="${LIST_DIR:-data/_lists/paper_like_multi}"
MODEL_ROOT="${MODEL_ROOT:-checkpoints}"
MUGE_REPR="${MUGE_REPR:-edge_uncertainty}"
CAPTION_DEVICE="${CAPTION_DEVICE:-cuda:0}"
MUGE_DEVICE="${MUGE_DEVICE:-cuda:0}"

mkdir -p "$LIST_DIR"

echo "[1/3] Generating BLIP-2 captions for SA-1B..."
python3 scripts/generate_captions.py \
  --input data/sa1b_images/train \
  --mode model \
  --device "$CAPTION_DEVICE"

python3 scripts/generate_captions.py \
  --input data/sa1b_images/val \
  --mode model \
  --device "$CAPTION_DEVICE"

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

echo "[3/3] Precomputing MuGE sidecars..."
for p in \
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
do
  python3 scripts/prepare_muge_edges.py \
    --input "$p" \
    --model-root "$MODEL_ROOT" \
    --repr "$MUGE_REPR" \
    --device "$MUGE_DEVICE"
done

echo "Prepared paper-like stage-2/3 data under $LIST_DIR"

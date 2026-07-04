#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NPROC="${NPROC:-3}"

echo "[1/5] Stage 1: JSCC"
torchrun --standalone --nproc_per_node="$NPROC" scripts/train.py \
  --config configs/paper_train_jscc_gan.yaml \
  --train-list data/imagenet/train \
  --val-list data/imagenet/val \
  --batch-size 21

echo "[2/5] Exporting JSCC checkpoint"
python3 scripts/export_checkpoint.py --stage jscc \
  --input outputs/checkpoints/paper_jscc_gan/best.pth \
  --output checkpoints/JSCC_model.pth --force

echo "[3/5] Stage 2: text-guided DM"
torchrun --standalone --nproc_per_node="$NPROC" scripts/train.py \
  --config configs/custom_paper_like/paper_train_text_dm_multi.yaml \
  --batch-size 21

echo "[4/5] Exporting diffusion backbone"
python3 scripts/export_checkpoint.py --stage text_dm \
  --input outputs/checkpoints/paper_text_dm_multi/best.pth \
  --output checkpoints/diffusion_backbone.pth --force

echo "[5/5] Edge codec + ControlNet"
torchrun --standalone --nproc_per_node="$NPROC" scripts/train.py \
  --config configs/custom_paper_like/paper_train_edge_codec_multi.yaml \
  --batch-size 21

torchrun --standalone --nproc_per_node="$NPROC" scripts/train.py \
  --config configs/custom_paper_like/paper_train_controlnet_multi.yaml \
  --batch-size 21

python3 scripts/export_checkpoint.py --stage controlnet \
  --input outputs/checkpoints/paper_controlnet_multi/best.pth \
  --output checkpoints/diffusion_controlnet.pth --force

echo "Paper-like multi-dataset training pipeline finished."

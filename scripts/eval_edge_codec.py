#!/usr/bin/env python
"""eval_edge_codec.py – Evaluate a trained edge JSCC codec (stage ``edge_codec``).

Loads a codec checkpoint produced by ``scripts/train.py --stage edge_codec`` and
reports edge-reconstruction quality on a validation set of edge maps: mean BCE,
soft-Dice, and (at a 0.5 threshold) IoU / F1.  This is the eval counterpart to
the edge-codec training stage; it does not touch the inference/eval pipelines.

Usage
-----
python scripts/eval_edge_codec.py \\
    --config configs/composed_train_edge_codec.yaml \\
    --checkpoint outputs/checkpoints/edge_codec/best.pth \\
    --val-list /data/edges/val/ --device cuda:0

The codec architecture (base_ch / norm / snr) is read from the SAME config used
for training, so the checkpoint loads cleanly.  ``--snr`` overrides the eval-time
channel SNR to sweep edge-link robustness.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch
from omegaconf import OmegaConf

from sgdjscc_lab.config import load_config, merge_cli_overrides

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("sgdjscc_lab.eval_edge_codec")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Edge-codec evaluation CLI")
    p.add_argument("--config", "-c", required=True)
    p.add_argument("--checkpoint", "-k", required=True,
                   help="Trained edge_codec checkpoint (e.g. .../edge_codec/best.pth)")
    p.add_argument("--val-list", default=None,
                   help="Folder of validation images/edge maps "
                        "(overrides val_input_path / train_input_path)")
    p.add_argument("--device", default=None)
    p.add_argument("--snr", type=float, default=None,
                   help="Override the edge-link SNR (dB) at eval time")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, device=args.device)
    if args.snr is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(
            {"train": {"edge_codec": {"snr_db": float(args.snr)}}}))

    from sgdjscc_lab.runtime import resolve_device
    device = resolve_device(str(cfg.get("device", "cpu")))

    from sgdjscc_lab.training.edge_transport import build_edge_codec
    codec = build_edge_codec(cfg, device)
    codec.load_codec_state(args.checkpoint, strict=False)
    codec.eval()

    val_path = (args.val_list
                or OmegaConf.select(cfg, "val_input_path", default=None)
                or OmegaConf.select(cfg, "train_input_path", default=None))
    if not val_path:
        sys.exit("Error: no validation data. Pass --val-list /path/to/edges/.")

    from sgdjscc_lab.data.datasets import build_dataloader_for_stage
    loader = build_dataloader_for_stage(
        val_path, cfg, shuffle=False, training=False, stage="edge_codec")

    n = 0
    sums = {"bce": 0.0, "dice": 0.0, "iou": 0.0, "f1": 0.0}
    for batch in loader:
        edge = batch["edge"].to(device)
        logits = codec.reconstruct(edge)
        prob = torch.sigmoid(logits)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, edge)
        # soft Dice SCORE (higher is better; 1.0 = perfect overlap)
        b = prob.shape[0]
        p = prob.reshape(b, -1); t = edge.reshape(b, -1)
        inter = (p * t).sum(1)
        dice_score = ((2 * inter + 1.0) / (p.sum(1) + t.sum(1) + 1.0)).mean()
        # hard IoU / F1 @0.5
        pb = (prob > 0.5).float().reshape(b, -1)
        tb = (edge > 0.5).float().reshape(b, -1)
        tp = (pb * tb).sum(1)
        fp = (pb * (1 - tb)).sum(1)
        fn = ((1 - pb) * tb).sum(1)
        iou = (tp / (tp + fp + fn + 1e-6)).mean()
        f1 = (2 * tp / (2 * tp + fp + fn + 1e-6)).mean()
        bs = edge.shape[0]
        sums["bce"] += float(bce) * bs
        sums["dice"] += float(dice_score) * bs   # Dice SCORE (higher = better)
        sums["iou"] += float(iou) * bs
        sums["f1"] += float(f1) * bs
        n += bs

    if n == 0:
        sys.exit("Error: validation loader yielded no samples.")
    logger.info("Edge-codec eval over %d samples (snr=%.1f dB):", n,
                float(OmegaConf.select(cfg, "train.edge_codec.snr_db", default=10.0)))
    logger.info("  BCE=%.4f  Dice=%.4f  IoU@0.5=%.4f  F1@0.5=%.4f",
                sums["bce"] / n, sums["dice"] / n, sums["iou"] / n, sums["f1"] / n)


if __name__ == "__main__":
    main()

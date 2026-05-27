#!/usr/bin/env python
"""infer_images.py – CLI entry point for sgdjscc_lab Phase-1/2 inference.

Usage examples
--------------
# Use defaults from configs/default.yaml
python scripts/infer_images.py --config configs/default.yaml

# Override input / output at runtime
python scripts/infer_images.py --config configs/default.yaml \\
    --input /path/to/images/ --output /path/to/out/ --snr 5 --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Make the src package importable when running without editable install ─────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sgdjscc_lab.config import load_config, merge_cli_overrides
from sgdjscc_lab.runtime import resolve_device, build_models
from sgdjscc_lab.pipelines.infer_pipeline import run_batch
from sgdjscc_lab.utils.seed import set_global_seed


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sgdjscc_lab")


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="sgdjscc_lab – SGDJSCC AWGN inference (Phase 1/2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config file (e.g. configs/default.yaml)",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Override config.input_path – single image file or folder.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Override config.output_dir – directory for reconstructed PNGs.",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=None,
        help="Override config.snr_db – AWGN channel SNR in dB.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override config.device – e.g. 'cuda:0' or 'cpu'.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── Load + merge config ───────────────────────────────────────────────────
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(
        cfg,
        input_path=args.input,
        output_dir=args.output,
        snr_db=args.snr,
        device=args.device,
    )

    logger.info("Config loaded from: %s", args.config)
    logger.info("  input_path  = %s", cfg.input_path)
    logger.info("  output_dir  = %s", cfg.output_dir)
    logger.info("  snr_db      = %.1f dB", cfg.snr_db)
    logger.info("  device      = %s", cfg.device)
    logger.info(
        "  use_semantic= %s  use_text=%s  use_controlnet=%s",
        cfg.use_semantic, cfg.use_text, cfg.use_controlnet,
    )

    set_global_seed(2025)

    # ── Resolve device ────────────────────────────────────────────────────────
    device = resolve_device(cfg.device)

    # ── Load models ───────────────────────────────────────────────────────────
    logger.info("Building models…")
    models = build_models(cfg, device)

    # ── Run inference ─────────────────────────────────────────────────────────
    run_batch(
        input_path=cfg.input_path,
        output_dir=cfg.output_dir,
        cfg=cfg,
        models=models,
    )

    logger.info("Done. Results saved to: %s", cfg.output_dir)


if __name__ == "__main__":
    main()

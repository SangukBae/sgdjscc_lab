#!/usr/bin/env python
"""evaluate.py – Full evaluation CLI for sgdjscc_lab Phase 3.

Usage examples
--------------
# Single SNR with default config
python scripts/evaluate.py --config configs/composed.yaml --snr 10

# SNR sweep writing to CSV
python scripts/evaluate.py --config configs/composed.yaml --snr-list -5,0,5,10,15,20,25

# Custom input / output CSV
python scripts/evaluate.py \\
    --config configs/composed.yaml \\
    --input /data/kodak/ \\
    --output-csv outputs/kodak_snr10.csv \\
    --snr 10 --device cuda:0

# Skip CLIP metrics (faster, quality metrics only)
python scripts/evaluate.py --config configs/composed.yaml --snr 10 --no-clip

Options
-------
--config        Path to YAML config file (required)
--input         Override config.input_path
--output-csv    Override eval_cfg.csv_path; default: outputs/results.csv
--snr           Single SNR value (dB). Cannot be used with --snr-list.
--snr-list      Comma-separated SNR values (dB). Cannot be used with --snr.
                If neither --snr nor --snr-list is given, the snr_list field
                from the loaded config (e.g. eval/default.yaml) is used.
--device        Override config.device
--no-clip       Disable CLIP / SRS metrics (quality metrics only)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Make the src package importable without editable install ──────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sgdjscc_lab.config import load_config, merge_cli_overrides
from sgdjscc_lab.utils.csv_logger import CSVLogger, RESULT_COLUMNS
from sgdjscc_lab.utils.metrics_io import summarize_metrics, format_summary_table
from sgdjscc_lab.pipelines.eval_pipeline import EvalContext, evaluate_single_snr, evaluate_snr_sweep


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sgdjscc_lab.evaluate")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="sgdjscc_lab Phase 3 – evaluation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config file (e.g. configs/composed.yaml)",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Override config.input_path – single image file or folder",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="CSV file path for results (overrides eval.csv_path in config)",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=None,
        help="Single AWGN channel SNR (dB)",
    )
    parser.add_argument(
        "--snr-list",
        default=None,
        help="Comma-separated list of SNR values in dB (e.g. '-5,0,5,10,15,20,25')",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override config.device (e.g. 'cuda:0' or 'cpu')",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable CLIP / SRS metrics (quality metrics only)",
    )
    return parser.parse_args()


def _parse_snr_list(snr_list_str: str) -> list:
    """Parse '−5,0,5,10' into [−5.0, 0.0, 5.0, 10.0]."""
    try:
        return [float(x.strip()) for x in snr_list_str.split(",")]
    except ValueError as exc:
        raise SystemExit(f"Invalid --snr-list: {snr_list_str!r}. Use comma-separated numbers.") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── Mutual exclusion check (before config load) ──────────────────────────
    if args.snr is not None and args.snr_list is not None:
        sys.exit("Error: --snr and --snr-list are mutually exclusive.")

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(
        cfg,
        input_path=args.input,
        device=args.device,
    )

    logger.info("Config: %s", args.config)
    logger.info("  input_path = %s", cfg.input_path)
    logger.info("  device     = %s", cfg.device)

    # ── Resolve CSV path ─────────────────────────────────────────────────────
    csv_path = args.output_csv
    if csv_path is None:
        csv_path = str(cfg.get("csv_path", "outputs/results.csv"))
    csv_path = Path(csv_path)

    # ── Build SNR list (CLI args take priority; fall back to cfg.snr_list) ───
    from omegaconf import OmegaConf as _OC
    if args.snr is not None:
        snr_list = [float(args.snr)]
    elif args.snr_list is not None:
        snr_list = _parse_snr_list(args.snr_list)
    else:
        cfg_snr = _OC.select(cfg, "snr_list", default=None)
        if cfg_snr is None:
            sys.exit(
                "Error: specify --snr or --snr-list, "
                "or set snr_list in the config (e.g. via eval/default.yaml)."
            )
        snr_list = [float(x) for x in _OC.to_container(cfg_snr, resolve=True)]
        logger.info("Using snr_list from config: %s", snr_list)

    logger.info("SNR list: %s dB", snr_list)
    logger.info("CSV output: %s", csv_path)

    # ── Read eval settings from config (populated via eval/default fragment) ─
    cfg_metrics = _OC.select(cfg, "metrics", default=None)
    clip_model_name = str(_OC.select(cfg, "clip_model_name", default="ViT-B/32"))
    srs_weights_raw = _OC.select(cfg, "semantic_reliability_weights", default=None)
    srs_weights = (
        _OC.to_container(srs_weights_raw, resolve=True)
        if srs_weights_raw is not None else None
    )

    # ── Build enabled_metrics (config base, then apply --no-clip) ────────────
    _clip_metrics = {
        "clip_image_image", "clip_text_image",
        "object_preservation_rate", "missing_object_rate",
        "additional_object_rate", "hallucination_score",
        "semantic_reliability_score",
    }
    if cfg_metrics is not None:
        enabled: set = set(_OC.to_container(cfg_metrics, resolve=True))
    else:
        enabled = {"psnr", "ssim", "lpips"} | _clip_metrics

    if args.no_clip:
        enabled -= _clip_metrics

    eval_ctx = EvalContext(
        enabled_metrics=enabled,
        clip_model_name=clip_model_name,
        srs_weights=srs_weights,
    )

    # ── Load models ──────────────────────────────────────────────────────────
    from sgdjscc_lab.runtime import resolve_device, build_models
    from sgdjscc_lab.utils.seed import set_global_seed

    set_global_seed(2025)
    device = resolve_device(cfg.device)

    logger.info("Building models…")
    models = build_models(cfg, device)

    # ── Run evaluation ───────────────────────────────────────────────────────
    with CSVLogger(csv_path, fieldnames=RESULT_COLUMNS) as csv_log:
        if len(snr_list) == 1:
            result = evaluate_single_snr(
                cfg, models, eval_ctx, snr_list[0], csv_logger=csv_log
            )
            all_rows = result["rows"]
        else:
            sweep = evaluate_snr_sweep(
                cfg, models, eval_ctx, snr_list, csv_logger=csv_log
            )
            all_rows = [r for v in sweep.values() for r in v.get("rows", [])]

    # ── Console summary ──────────────────────────────────────────────────────
    summary = summarize_metrics(all_rows)
    print("\n" + "=" * 66)
    print("  Evaluation complete")
    print("=" * 66)
    print(format_summary_table(summary))
    print(f"\n  CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()

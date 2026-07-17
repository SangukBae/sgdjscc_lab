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
from sgdjscc_lab.utils.csv_logger import (
    CSVLogger, RESULT_COLUMNS, PACKET_RESULT_COLUMNS_FULL,
)
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
    parser.add_argument(
        "--profile",
        default=None,
        choices=["paper", "extended", "full"],
        help="Metric profile (overrides config.metrics / metrics_profile): "
             "paper = PSNR/LPIPS/CLIP/FID (the paper's set); "
             "extended = + SSIM/object/hallucination/SRS; full = extended + FID.",
    )
    parser.add_argument(
        "--require-real-fid",
        action="store_true",
        help="Fail fast unless a real torchvision Inception-FID backend is "
             "available (rejects proxy/unavailable FID). Use for paper-comparable "
             "numbers; requires 'fid' to be an enabled metric.",
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

    # ── paper_mode (default off): enforce the paper-baseline EVAL guardrails ──
    # A config named "paper eval" must actually run the paper baseline: block
    # every extension feature (Phase 4/5, packet, regeneration, shared_vae).
    from sgdjscc_lab import paper_mode as _paper_mode
    try:
        _paper_mode.enforce_eval(cfg)
    except _paper_mode.PaperModeError as exc:
        sys.exit(f"Error: paper_mode eval guardrail violated.\n  {exc}")

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

    # ── Build enabled_metrics ────────────────────────────────────────────────
    # Precedence: --profile (CLI) > config.metrics_profile > config.metrics list
    # > built-in extended default. Profiles separate the paper's reported metric
    # set (PSNR/LPIPS/CLIP/FID) from the ETRI/extended metrics (SSIM/SRS/…).
    from sgdjscc_lab.utils.metric_profiles import (
        resolve_profile, columns_for_metrics, NON_PAPER_METRICS,
    )
    _clip_metrics = {
        "clip_image_image", "clip_text_image",
        "object_preservation_rate", "missing_object_rate",
        "additional_object_rate", "hallucination_score",
        "semantic_reliability_score",
    }
    cfg_profile = _OC.select(cfg, "metrics_profile", default=None)
    active_profile = args.profile or (str(cfg_profile) if cfg_profile else None)
    if active_profile is not None:
        enabled: set = resolve_profile(active_profile)
        logger.info("Metric profile: %s → %s", active_profile, sorted(enabled))
    elif cfg_metrics is not None:
        enabled = set(_OC.to_container(cfg_metrics, resolve=True))
    else:
        enabled = {"psnr", "ssim", "lpips"} | _clip_metrics

    if args.no_clip:
        enabled -= _clip_metrics

    # Flag non-paper metrics that are active (transparency for paper comparisons).
    _non_paper_active = sorted(enabled & NON_PAPER_METRICS)
    if _non_paper_active:
        logger.info("Active non-paper (ETRI/extended) metrics: %s", _non_paper_active)

    # paper_mode: now that --profile / --no-clip overrides are applied, verify the
    # FINAL metric set is the paper's (else the earlier "paper eval" intent is a lie).
    try:
        _paper_mode.enforce_eval_metrics(cfg, enabled, args.no_clip)
    except _paper_mode.PaperModeError as exc:
        sys.exit(f"Error: paper_mode eval guardrail violated.\n  {exc}")

    # ── Phase master switches ─────────────────────────────────────────────────
    from sgdjscc_lab.phase_gates import effective_flag as _eff, phase4_enabled, phase5_enabled
    _p4 = phase4_enabled(cfg)
    _p5 = phase5_enabled(cfg)
    if not _p4:
        logger.info("use_phase4=false: Phase 4 features disabled.")
    if not _p5:
        logger.info("use_phase5=false: Phase 5 features disabled.")

    # ── Phase 4-A packet-aware settings (gated by use_phase4) ────────────────
    use_packet = _eff(cfg, "use_packet_eval", phase=4)
    packet_weights_raw = _OC.select(cfg, "semantic_packet_weights", default=None)
    packet_weights = (
        _OC.to_container(packet_weights_raw, resolve=True)
        if packet_weights_raw is not None else None
    )
    packet_blend = float(_OC.select(cfg, "packet_blend", default=0.5))

    # ── Phase 5-C SRS-v2 + VQA + regeneration search (gated by use_phase5) ──
    use_srs_v2 = _eff(cfg, "use_srs_v2", phase=5)
    use_vqa = _eff(cfg, "use_vqa_hallucination", phase=5)
    use_regen_search = _eff(cfg, "use_regeneration_search", phase=5)
    srs_v2_weights_raw = _OC.select(cfg, "srs_v2_weights", default=None)
    srs_v2_weights = (
        _OC.to_container(srs_v2_weights_raw, resolve=True)
        if srs_v2_weights_raw is not None else None
    )
    vqa_backend_raw = _OC.select(cfg, "vqa_backend", default=None)
    vqa_backend_cfg = (
        _OC.to_container(vqa_backend_raw, resolve=True)
        if vqa_backend_raw is not None else None
    )

    # Resolve the compute device up front so the eval-side models (CLIP / packet
    # BLIP2 / VQA) run on the SAME device as the loaded models.  Otherwise they
    # default to CPU and fp16 (Half) ops crash ("slow_conv2d_cpu not implemented
    # for 'Half'"), silently emptying the semantic packets.
    from sgdjscc_lab.runtime import resolve_device, build_models
    from sgdjscc_lab.utils.seed import set_global_seed
    device = resolve_device(cfg.device)

    # Object presence judge settings (provisional CLIP probe; see
    # object_preservation.py). object_presence_threshold previously existed in
    # eval/default.yaml but was never wired to the evaluators — it now is.
    presence_threshold = float(_OC.select(cfg, "object_presence_threshold", default=0.25))
    presence_band = float(_OC.select(cfg, "object_presence_uncertain_band", default=0.0))

    eval_ctx = EvalContext(
        enabled_metrics=enabled,
        clip_model_name=clip_model_name,
        srs_weights=srs_weights,
        presence_threshold=presence_threshold,
        presence_uncertain_band=presence_band,
        packet_weights=packet_weights,
        packet_blend=packet_blend,
        use_srs_v2=use_srs_v2,
        srs_v2_weights=srs_v2_weights,
        use_vqa_hallucination=use_vqa,
        vqa_backend_cfg=vqa_backend_cfg,
        device=device,
    )
    if use_srs_v2:
        vqa_type = (vqa_backend_cfg or {}).get("type", "clip_fallback") if use_vqa else "off"
        logger.info("SRS-v2 verifier enabled (VQA=%s).", vqa_type)
    if use_regen_search:
        logger.info("Regeneration search enabled.")

    # CSV header: packet/SRS-v2/regen runs emit the extended packet columns;
    # otherwise a selected profile narrows the header to its metric set, derived
    # from the FINAL `enabled` set so --no-clip (and any other narrowing) keeps the
    # header and the computed metrics in sync (no orphan CLIP columns), else default.
    if use_packet or use_srs_v2 or use_regen_search:
        csv_columns = PACKET_RESULT_COLUMNS_FULL
    elif active_profile is not None:
        csv_columns = columns_for_metrics(enabled)
    else:
        csv_columns = RESULT_COLUMNS
    if use_packet:
        logger.info("Packet-aware evaluation enabled (srs_base / srs_packet).")

    # ── --require-real-fid: fail fast before the expensive eval ──────────────
    if args.require_real_fid:
        if "fid" not in enabled:
            sys.exit("Error: --require-real-fid but 'fid' is not an enabled metric. "
                     "Use --profile paper/full or add 'fid' to metrics_profile/metrics.")
        probe = eval_ctx._get_fid()
        if not probe.ensure_backend():
            sys.exit(
                "Error: --require-real-fid but no real Inception backend is "
                f"available (backend={probe.backend_name!r}). Install torchvision "
                "and ensure the Inception-v3 weights can be downloaded/cached "
                "(network on first use), then retry. To allow a proxy/None FID, "
                "drop --require-real-fid.")
        logger.info("Real Inception-FID backend verified (backend=inception).")

    # ── Load models ──────────────────────────────────────────────────────────
    set_global_seed(2025)

    logger.info("Building models…")
    models = build_models(cfg, device)

    # ── Run evaluation ───────────────────────────────────────────────────────
    with CSVLogger(csv_path, fieldnames=csv_columns) as csv_log:
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

    # FID is dataset/SNR-level (not in the per-metric mean table) — surface it with
    # its backend so a proxy FID is never mistaken for paper-comparable Inception-FID.
    fid_backends = {r.get("fid_backend") for r in all_rows if r.get("fid") is not None}
    if fid_backends:
        backend = next(iter(fid_backends)) if len(fid_backends) == 1 else "mixed"
        note = "" if backend == "inception" else "  (NOT paper-comparable)"
        print(f"\n  FID backend: {backend}{note}")

    print(f"\n  CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()

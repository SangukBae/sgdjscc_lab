#!/usr/bin/env python
"""report_transmission_accounting.py – ETRI 6차 (step 11-12) transmission
accounting + rate/reliability report CLI.

Two independent modes
----------------------
``--input`` (**from-run**): re-runs the frame sequence through the same
``TemporalPipeline`` construction ``scripts/evaluate_video.py`` uses (subject
to the same reconstruction-fidelity caveat as
``scripts/remeasure_video_metrics.py --from-run``: this recomputes
reuse/recompute/generate decisions from *cfg*, it does not replay a specific
prior run's exact decisions), then computes fresh transmission accounting +
a rate/reliability report from that result.

``--from-accounting-summary`` (**recombine existing outputs**): reads an
*already-produced* ``accounting_summary.json`` (written by
``evaluate_video.py``'s inline ``accounting.enabled: true`` path or by this
script's own ``--input`` mode) plus an existing ``temporal_metrics.csv`` row
and (optionally) an existing ``packet_match_report.json`` — and rebuilds just
the rate/reliability trade-off report from them, without re-running anything.
This is the "read temporal_frames/segments/metrics/packet reports from an
existing output folder" path.

**PoC accounting, not a real bitstream/CBR** — see
``accounting/bit_accounting.py``'s module docstring and
docs/etri_strategy.md 6차 구현 결과.

Usage
-----
# From-run: recompute accounting + rate/reliability from a frame folder
python scripts/report_transmission_accounting.py --config configs/composed_video.yaml \\
    --input /path/to/frames/ --no-models --accounting-output-dir ../outputs/accounting

# Recombine: rebuild just the rate/reliability report from an existing output folder
python scripts/report_transmission_accounting.py \\
    --from-accounting-summary ../outputs/accounting/accounting_summary.json \\
    --temporal-metrics-csv ../outputs/temporal_metrics.csv \\
    --packet-match-report-json ../outputs/packet_match_report.json \\
    --rate-reliability-json ../outputs/accounting/rate_reliability_summary.json \\
    --rate-reliability-curve-csv ../outputs/accounting/rate_reliability_curve.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sgdjscc_lab.report_transmission_accounting")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ETRI 6차 – transmission accounting + rate/reliability report CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", "-c", default=None,
                   help="Path to YAML config file (required for --input mode; optional for "
                        "--from-accounting-summary mode, only used for default output paths).")
    p.add_argument("--input", "-i", default=None,
                   help="Frame folder or video file — from-run mode: recompute accounting from scratch.")
    p.add_argument("--no-models", action="store_true",
                   help="--input mode only: skip SGD-JSCC model loading (identity reconstruction).")
    p.add_argument("--captions", default=None,
                   help="--input mode only: caption source (.txt file or per-frame directory).")
    p.add_argument("--device", default=None, help="Compute device override (--input mode only).")
    p.add_argument("--snr", type=float, default=None, help="AWGN SNR dB override (--input mode only).")

    p.add_argument("--from-accounting-summary", default=None,
                   help="Recombine mode: path to an existing accounting_summary.json.")
    p.add_argument("--temporal-metrics-csv", default=None,
                   help="Recombine mode: path to an existing temporal_metrics.csv (single data row read).")
    p.add_argument("--packet-match-report-json", default=None,
                   help="Optional: existing packet_match_report.json, used to compute mean_severity.")

    p.add_argument("--accounting-output-dir", default=None,
                   help="--input mode only: override accounting.output_dir for this run.")
    p.add_argument("--baseline", default=None,
                   help="--input mode only: override accounting.baseline for this run.")
    p.add_argument("--rate-reliability-json", default=None, help="Override rate_reliability.output_json path.")
    p.add_argument("--rate-reliability-curve-csv", default=None, help="Override rate_reliability.curve_csv path.")
    p.add_argument("--label", default=None, help="Row label for the rate/reliability curve CSV.")
    return p.parse_args()


def _load_temporal_metrics_row(path: str) -> dict:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"Error: {path} has no data rows.")
    row = rows[0]
    out = {}
    for k in ("ptc", "sfr", "sdi"):
        v = row.get(k)
        try:
            out[k] = float(v) if v not in (None, "", "None") else None
        except ValueError:
            out[k] = None
    return out


def _mean_severity_from_packet_report(path: str):
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    severities = [r.get("severity") for r in rows if isinstance(r, dict) and r.get("severity") is not None]
    return float(sum(severities) / len(severities)) if severities else None


def _run_from_run_mode(args: argparse.Namespace) -> None:
    from sgdjscc_lab.config import load_config, merge_cli_overrides

    if not args.config:
        sys.exit("Error: --config is required with --input.")
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, input_path=args.input, snr_db=args.snr, device=args.device)
    if args.accounting_output_dir:
        cfg = OmegaConf.merge(cfg, {"accounting": {"output_dir": args.accounting_output_dir}})
        from sgdjscc_lab.config import _resolve_paths  # re-resolve the interpolated sub-paths
        cfg = _resolve_paths(cfg, Path(args.config).resolve().parent)

    from sgdjscc_lab.phase_gates import phase4_enabled
    if not phase4_enabled(cfg):
        sys.exit("Error: use_phase4 is false — this CLI requires 'use_phase4: true' in your config.")

    from sgdjscc_lab.io import list_image_files, load_image_as_tensor
    from sgdjscc_lab.utils import video_io
    from sgdjscc_lab.video.scene_change_detector import SceneChangeDetector, SceneChangeConfig
    from sgdjscc_lab.video.keyframe_extractor import KeyframeExtractor
    from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline
    from sgdjscc_lab.evaluators.temporal_consistency import evaluate_sequence
    from sgdjscc_lab.guidance.semantic_packet_extractor import SemanticPacketExtractor

    if video_io.is_video_file(cfg.input_path):
        frames_root = Path(OmegaConf.select(cfg, "video_io.extracted_frames_dir", default="../outputs/video_frames"))
        extracted = video_io.extract_frames(cfg.input_path, frames_root / Path(cfg.input_path).stem)
        files = extracted["files"]
    else:
        files = list_image_files(cfg.input_path)
    frames = [load_image_as_tensor(f) for f in files]
    logger.info("Loaded %d ordered frames.", len(frames))

    captions = None
    if args.captions:
        cap_path = Path(args.captions)
        if cap_path.is_file():
            lines = cap_path.read_text(encoding="utf-8").splitlines()
            captions = [lines[i].strip() if i < len(lines) else "" for i in range(len(files))]
        elif cap_path.is_dir():
            captions = [
                (cap_path / f"{f.stem}.txt").read_text(encoding="utf-8").strip()
                if (cap_path / f"{f.stem}.txt").exists() else "" for f in files
            ]

    models = None
    clip_eval = None
    if not args.no_models:
        from sgdjscc_lab.runtime import resolve_device, build_models
        from sgdjscc_lab.utils.seed import set_global_seed
        from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator
        set_global_seed(2025)
        device = resolve_device(cfg.device)
        models = build_models(cfg, device)
        if hasattr(models, "jscc_model"):
            models.jscc_model.snr = float(cfg.snr_db)
        clip_eval = CLIPScoreEvaluator(model_name=str(cfg.get("clip_model_name", "ViT-B/32")), device=device)

    sc = OmegaConf.to_container(OmegaConf.select(cfg, "scene_change", default={}) or {}, resolve=True)
    scene_cfg = SceneChangeConfig(
        threshold=float(sc.get("threshold", 0.35)), hist_weight=float(sc.get("hist_weight", 1.0)),
        clip_weight=float(sc.get("clip_weight", 0.0)), lpips_weight=float(sc.get("lpips_weight", 0.0)),
        hist_bins=int(sc.get("hist_bins", 16)),
    )
    scene_detector = SceneChangeDetector(config=scene_cfg, clip_evaluator=clip_eval)
    max_gop = int(OmegaConf.select(cfg, "keyframe.max_gop", default=12))
    keyframe_extractor = KeyframeExtractor(scene_detector, max_gop=max_gop)

    packet_extractor = SemanticPacketExtractor(
        text_extractor=getattr(models, "text_extractor", None) if models else None,
        clip_evaluator=clip_eval,
    )

    def _caption_for(frame_id):
        if captions is None:
            return None
        fid = str(frame_id)
        if fid.startswith("recon_") and models is not None:
            return None
        try:
            idx = int(fid.split("_")[-1])
        except (ValueError, IndexError):
            return None
        return captions[idx] if 0 <= idx < len(captions) else None

    def packet_fn(frame, frame_id):
        return packet_extractor.extract(frame, frame_id=frame_id, caption=_caption_for(frame_id))

    if models is not None:
        from sgdjscc_lab.pipelines.eval_pipeline import _reconstruct_with_cfg

        def reconstruct_fn(frame, run_cfg):
            return _reconstruct_with_cfg(frame, models, run_cfg if run_cfg is not None else cfg)
    else:
        def reconstruct_fn(frame, run_cfg):
            return frame.clone()

    reuse_threshold = float(OmegaConf.select(cfg, "temporal.reuse_threshold", default=0.2))
    _sdt = OmegaConf.select(cfg, "temporal.semantic_delta_threshold", default=None)
    if _sdt is not None:
        reuse_threshold = float(_sdt)
    motion_threshold = OmegaConf.select(cfg, "temporal.motion_threshold", default=None)
    motion_weight = float(OmegaConf.select(cfg, "temporal.motion_weight", default=0.5))
    motion_grid = int(OmegaConf.select(cfg, "temporal.motion_grid", default=8))

    from sgdjscc_lab.phase_gates import effective_flag as _eff_video_gen
    enable_generate = _eff_video_gen(cfg, "use_video_gen", phase=4) and bool(
        OmegaConf.select(cfg, "video_generator.enabled", default=False)
    )
    video_generator = None
    conditioning_mode = str(OmegaConf.select(cfg, "video_generator.conditioning_mode", default="start_only"))
    generate_delta_min = generate_delta_max = generate_motion_max = None
    allow_ground_truth_reference = False
    if enable_generate:
        from sgdjscc_lab.video.video_generator import build_generator
        video_generator = build_generator(cfg)
        generate_delta_min = OmegaConf.select(cfg, "video_generator.generate_delta_min", default=None)
        generate_delta_max = OmegaConf.select(cfg, "video_generator.generate_delta_max", default=None)
        generate_motion_max = OmegaConf.select(cfg, "video_generator.generate_motion_max", default=None)
        allow_ground_truth_reference = bool(
            OmegaConf.select(cfg, "video_generator.allow_ground_truth_reference", default=False)
        )

    pipeline = TemporalPipeline(
        reconstruct_fn=reconstruct_fn, packet_fn=packet_fn,
        keyframe_extractor=keyframe_extractor, reuse_threshold=reuse_threshold,
        motion_threshold=(None if motion_threshold is None else float(motion_threshold)),
        motion_weight=motion_weight, motion_grid=motion_grid,
        diffusion_step=int(cfg.get("diffusion_step", 50)),
        enable_generate=enable_generate, video_generator=video_generator,
        generate_delta_min=generate_delta_min, generate_delta_max=generate_delta_max,
        generate_motion_max=generate_motion_max,
        allow_ground_truth_reference=allow_ground_truth_reference,
        conditioning_mode=conditioning_mode,
    )
    result = pipeline.run(frames)
    temporal_metrics = evaluate_sequence(result["records"])
    temporal_metrics.update(result["summary"])

    from sgdjscc_lab.pipelines.packet_verification import maybe_run as _maybe_run_verifier
    verifier_out = _maybe_run_verifier(result, cfg)
    mean_severity = None
    if verifier_out is not None:
        severities = [r.get("severity") for r in verifier_out["rows"] if r.get("severity") is not None]
        if severities:
            mean_severity = float(sum(severities) / len(severities))

    from sgdjscc_lab.pipelines.transmission_accounting import account_transmission, write_accounting

    accounting_result = account_transmission(
        result,
        baseline=str(args.baseline or OmegaConf.select(cfg, "accounting.baseline", default="naive_full_frame_packet")),
        latent_symbols_per_frame=OmegaConf.select(cfg, "accounting.latent_symbols_per_frame", default=None),
        edge_cr=float(OmegaConf.select(cfg, "accounting.edge_cr", default=0.078125)),
        symbols_per_bit_proxy=float(OmegaConf.select(cfg, "accounting.symbols_per_bit_proxy", default=1.0)),
    )
    write_accounting(
        accounting_result,
        frame_json=OmegaConf.select(cfg, "accounting.frame_json", default=None),
        frame_csv=OmegaConf.select(cfg, "accounting.frame_csv", default=None),
        segment_json=OmegaConf.select(cfg, "accounting.segment_json", default=None),
        segment_csv=OmegaConf.select(cfg, "accounting.segment_csv", default=None),
        summary_json=OmegaConf.select(cfg, "accounting.summary_json", default=None),
    )

    _write_rate_reliability(
        accounting_result["summary"], temporal_metrics, mean_severity,
        rate_reliability_json=args.rate_reliability_json or OmegaConf.select(cfg, "rate_reliability.output_json", default=None),
        curve_csv=args.rate_reliability_curve_csv or OmegaConf.select(cfg, "rate_reliability.curve_csv", default=None),
        label=args.label,
    )
    _print_summary(accounting_result["summary"])


def _write_rate_reliability(accounting_summary, temporal_metrics, mean_severity, rate_reliability_json, curve_csv, label):
    from sgdjscc_lab.pipelines.rate_reliability_report import (
        append_rate_reliability_row, build_rate_reliability_row, write_rate_reliability_summary,
    )

    row = build_rate_reliability_row(accounting_summary, temporal_metrics, mean_severity=mean_severity, label=label)
    if rate_reliability_json:
        write_rate_reliability_summary(row, rate_reliability_json)
    if curve_csv:
        append_rate_reliability_row(row, curve_csv)
    return row


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 66)
    print("  Transmission accounting complete (ETRI 6차, PoC — not a real bitstream/CBR)")
    print("=" * 66)
    for k in (
        "n_frames", "n_keyframes", "n_generate", "n_reused", "n_recompute",
        "total_bits", "total_channel_symbols", "total_semantic_units",
        "baseline", "baseline_bits", "baseline_channel_symbols",
        "bit_reduction", "symbol_reduction", "semantic_unit_reduction", "proxy_fraction",
    ):
        print(f"  {k:<26} {summary.get(k)}")


def _run_recombine_mode(args: argparse.Namespace) -> None:
    if not args.temporal_metrics_csv:
        sys.exit("Error: --temporal-metrics-csv is required with --from-accounting-summary.")
    with open(args.from_accounting_summary, encoding="utf-8") as fh:
        accounting_summary = json.load(fh)
    temporal_metrics = _load_temporal_metrics_row(args.temporal_metrics_csv)
    mean_severity = (
        _mean_severity_from_packet_report(args.packet_match_report_json)
        if args.packet_match_report_json else None
    )

    rate_reliability_json = args.rate_reliability_json
    curve_csv = args.rate_reliability_curve_csv
    if (not rate_reliability_json or not curve_csv) and args.config:
        from sgdjscc_lab.config import load_config
        cfg = load_config(args.config)
        rate_reliability_json = rate_reliability_json or OmegaConf.select(cfg, "rate_reliability.output_json", default=None)
        curve_csv = curve_csv or OmegaConf.select(cfg, "rate_reliability.curve_csv", default=None)
    if not rate_reliability_json and not curve_csv:
        sys.exit("Error: pass --rate-reliability-json and/or --rate-reliability-curve-csv "
                  "(or --config to read their defaults).")

    row = _write_rate_reliability(
        accounting_summary, temporal_metrics, mean_severity,
        rate_reliability_json=rate_reliability_json, curve_csv=curve_csv, label=args.label,
    )
    print("\n" + "=" * 66)
    print("  Rate/reliability report rebuilt from existing outputs (ETRI 6차)")
    print("=" * 66)
    for k, v in row.items():
        print(f"  {k:<26} {v}")


def main() -> None:
    args = _parse_args()
    if args.from_accounting_summary:
        _run_recombine_mode(args)
    elif args.input:
        _run_from_run_mode(args)
    else:
        sys.exit("Error: pass --input (from-run mode) or --from-accounting-summary (recombine mode).")


if __name__ == "__main__":
    main()

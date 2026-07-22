#!/usr/bin/env python
"""evaluate_video.py – Phase 4-B keyframe/temporal evaluation CLI.

Processes an ordered frame sequence — either a folder of frames or a video file
(mp4/avi/…, unpacked via utils/video_io) — through the keyframe-oriented
temporal pipeline:

  scene-change detection → keyframe / inter-frame split → per-frame
  reconstruction (full inference at keyframes, keyframe-reuse + semantic
  delta + optional motion gate at inter-frames) → temporal metrics.

Outputs
-------
- keyframe / GOP structure JSON       (cfg.keyframe_json)
- GOP/segment records JSON             (cfg.segment_json)
- per-sequence temporal metrics CSV    (cfg.temporal_csv) — includes the ETRI
  1차 provisional time-axis metrics PTC / SFR / SDI
- per-frame log CSV                    (cfg.frame_log_csv) — includes the
  reuse/recompute gate decision + motion score per frame
- reconstructed frame folder           (cfg.video_io.recon_frames_dir)
- optional reconstructed mp4           (--save-video / cfg.video_io.save_recon_video)

Usage
-----
python scripts/evaluate_video.py --config configs/composed_video.yaml \
    --input /path/to/ordered_frames/ --snr 5 --device cuda:0

# mp4 input + re-assembled mp4 output:
python scripts/evaluate_video.py --config configs/composed_video.yaml \
    --input /path/to/clip.mp4 --save-video --snr 5

# Dry run of the keyframe/delta logic without loading SGD-JSCC checkpoints:
python scripts/evaluate_video.py --config configs/composed_video.yaml \
    --input /path/to/frames/ --no-models
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

from sgdjscc_lab.config import load_config, merge_cli_overrides

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sgdjscc_lab.evaluate_video")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="sgdjscc_lab Phase 4-B – keyframe / temporal evaluation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    p.add_argument(
        "--input", "-i", default=None,
        help="Folder of ordered frames, or a video file (mp4/avi/mov/mkv/webm) "
        "that is unpacked into ordered frames first (needs cv2 or ffmpeg).",
    )
    p.add_argument("--snr", type=float, default=None, help="AWGN SNR (dB)")
    p.add_argument("--device", default=None, help="Compute device override")
    p.add_argument(
        "--save-video", action="store_true",
        help="Assemble the reconstructed frames into an mp4 "
        "(cfg.video_io.recon_video; also enabled via cfg.video_io.save_recon_video). "
        "Uses the source fps for video inputs, else cfg.video_io.fps (default 24).",
    )
    p.add_argument(
        "--no-models", action="store_true",
        help="Skip SGD-JSCC model loading; dry run with identity reconstruction. "
        "Validates keyframe/delta/temporal orchestration. Packets are empty unless "
        "--captions is supplied (then semantic delta/metrics are meaningful too).",
    )
    p.add_argument(
        "--captions", default=None,
        help="Optional captions source for building packets without BLIP2: either a "
        ".txt file (one caption per line, aligned to sorted frames) or a directory "
        "with a '<frame_stem>.txt' file per frame.",
    )
    return p.parse_args()


def _load_captions(captions_arg, files):
    """Return a list of captions aligned to *files*, or None if unavailable."""
    if captions_arg is None:
        return None
    path = Path(captions_arg)
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
        caps = [lines[i].strip() if i < len(lines) else "" for i in range(len(files))]
        return caps
    if path.is_dir():
        caps = []
        for f in files:
            side = path / f"{f.stem}.txt"
            caps.append(side.read_text(encoding="utf-8").strip() if side.exists() else "")
        return caps
    logger.warning("Captions path not found: %s", captions_arg)
    return None


def _load_frames(cfg, input_path: str):
    """Load ordered frames from a folder OR a video file.

    Video inputs are unpacked into cfg.video_io.extracted_frames_dir/<stem>/ as
    ordered PNGs first, then fed through the same loader as a frame folder.

    Returns ``(files, frames, source_fps)`` — source_fps is None for folders.
    """
    from sgdjscc_lab.io import list_image_files, load_image_as_tensor
    from sgdjscc_lab.utils import video_io

    source_fps = None
    if video_io.is_video_file(input_path):
        if video_io.get_backend() is None:
            sys.exit(
                "Error: video input given but no video backend is available "
                "(install opencv-python or make ffmpeg/ffprobe available on PATH)."
            )
        frames_root = Path(OmegaConf.select(
            cfg, "video_io.extracted_frames_dir", default="../outputs/video_frames"))
        out_dir = frames_root / Path(input_path).stem
        extracted = video_io.extract_frames(input_path, out_dir)
        files = extracted["files"]
        source_fps = extracted["fps"]
    else:
        files = list_image_files(input_path)
    frames = [load_image_as_tensor(f) for f in files]
    return files, frames, source_fps


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, input_path=args.input, snr_db=args.snr, device=args.device)

    # Phase 4-B requires the use_phase4 master switch.
    from sgdjscc_lab.phase_gates import phase4_enabled
    if not phase4_enabled(cfg):
        sys.exit(
            "Error: use_phase4 is false — Phase 4-B temporal/video evaluation "
            "requires 'use_phase4: true' in your config.\n"
            "Add it to configs/eval/default.yaml or your composed config, "
            "or use configs/composed_phase5_full.yaml for the full Phase 4+5 stack."
        )

    logger.info("Video eval config: %s", args.config)
    logger.info("  input_path = %s", cfg.input_path)

    files, frames, source_fps = _load_frames(cfg, cfg.input_path)
    logger.info("Loaded %d ordered frames.", len(frames))

    captions = _load_captions(args.captions, files)
    if captions is not None:
        logger.info("Loaded %d captions from %s", len(captions), args.captions)

    # ── Scene detector / keyframe extractor ──────────────────────────────────
    from sgdjscc_lab.video.scene_change_detector import SceneChangeDetector, SceneChangeConfig
    from sgdjscc_lab.video.keyframe_extractor import KeyframeExtractor

    sc = OmegaConf.to_container(OmegaConf.select(cfg, "scene_change", default={}) or {}, resolve=True)
    scene_cfg = SceneChangeConfig(
        threshold=float(sc.get("threshold", 0.35)),
        hist_weight=float(sc.get("hist_weight", 1.0)),
        clip_weight=float(sc.get("clip_weight", 0.0)),
        lpips_weight=float(sc.get("lpips_weight", 0.0)),
        hist_bins=int(sc.get("hist_bins", 16)),
    )
    max_gop = int(OmegaConf.select(cfg, "keyframe.max_gop", default=12))
    reuse_threshold = float(OmegaConf.select(cfg, "temporal.reuse_threshold", default=0.2))
    # ETRI-plan alias: temporal.semantic_delta_threshold overrides reuse_threshold.
    _sdt = OmegaConf.select(cfg, "temporal.semantic_delta_threshold", default=None)
    if _sdt is not None:
        reuse_threshold = float(_sdt)
    # Motion gate (dual gate, ETRI 1차 step 3). Default None = disabled →
    # identical to the legacy semantic-delta-only reuse decision.
    motion_threshold = OmegaConf.select(cfg, "temporal.motion_threshold", default=None)
    motion_weight = float(OmegaConf.select(cfg, "temporal.motion_weight", default=0.5))
    motion_grid = int(OmegaConf.select(cfg, "temporal.motion_grid", default=8))
    if motion_threshold is not None:
        logger.info("Motion gate enabled: threshold=%.4g weight=%.2f grid=%d",
                    float(motion_threshold), motion_weight, motion_grid)

    # ── Build reconstruct_fn / packet_fn ─────────────────────────────────────
    models = None
    clip_eval = None
    if not args.no_models:
        from sgdjscc_lab.runtime import resolve_device, build_models
        from sgdjscc_lab.utils.seed import set_global_seed
        from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator
        set_global_seed(2025)
        device = resolve_device(cfg.device)
        logger.info("Building SGD-JSCC models…")
        models = build_models(cfg, device)
        if hasattr(models, "jscc_model"):
            models.jscc_model.snr = float(cfg.snr_db)
        clip_eval = CLIPScoreEvaluator(model_name=str(cfg.get("clip_model_name", "ViT-B/32")), device=device)

    scene_detector = SceneChangeDetector(
        config=scene_cfg, clip_evaluator=clip_eval,
        use_lpips=bool(sc.get("use_lpips", False)),
    )
    keyframe_extractor = KeyframeExtractor(scene_detector, max_gop=max_gop)

    # Packet extractor
    from sgdjscc_lab.guidance.semantic_packet_extractor import SemanticPacketExtractor
    packet_extractor = SemanticPacketExtractor(
        text_extractor=getattr(models, "text_extractor", None) if models else None,
        clip_evaluator=clip_eval,
    )

    # Honest warning when packets cannot carry semantics.
    if captions is None and getattr(models, "text_extractor", None) is None and clip_eval is None:
        logger.warning(
            "No caption/CLIP source: packets will be EMPTY (no objects/scene). "
            "Keyframe/delta orchestration is still exercised, but semantic delta, "
            "transmitted_units and packet SRS are not meaningful. Pass --captions "
            "or run without --no-models for semantic evaluation."
        )

    def _caption_for(frame_id):
        """Look up the caption for a 'frame_NNNNN' / 'recon_NNNNN' id (or None).

        Provided captions describe the *original* frames.  For reconstructed
        frames we only reuse the caption in the identity dry-run (recon == orig);
        with real models we return None so the recon's *own* semantics are
        extracted (needed for hallucination / missing-object detection).
        """
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
        # Dry run: identity reconstruction (validates keyframe/delta orchestration).
        def reconstruct_fn(frame, run_cfg):
            return frame.clone()

    # SRS function (packet-aware). Presence settings wired per ETRI step 0
    # (provisional CLIP probe; see evaluators/object_preservation.py).
    from sgdjscc_lab.evaluators.semantic_reliability import SemanticReliabilityEvaluator
    srs_eval = SemanticReliabilityEvaluator(
        clip_evaluator=clip_eval,
        presence_threshold=float(OmegaConf.select(cfg, "object_presence_threshold", default=0.25)),
        presence_uncertain_band=float(OmegaConf.select(cfg, "object_presence_uncertain_band", default=0.0)),
    )

    def srs_fn(op, rp):
        return srs_eval.score_packet(op, rp, srs_base=None)["srs_packet"]

    # ── Run temporal pipeline ────────────────────────────────────────────────
    from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline
    from sgdjscc_lab.evaluators.temporal_consistency import evaluate_sequence

    pipeline = TemporalPipeline(
        reconstruct_fn=reconstruct_fn,
        packet_fn=packet_fn,
        keyframe_extractor=keyframe_extractor,
        delta=None,
        cfg=cfg,
        reuse_threshold=reuse_threshold,
        motion_threshold=(None if motion_threshold is None else float(motion_threshold)),
        motion_weight=motion_weight,
        motion_grid=motion_grid,
        diffusion_step=int(cfg.get("diffusion_step", 50)),
        srs_fn=srs_fn,
    )
    result = pipeline.run(frames)

    temporal_metrics = evaluate_sequence(result["records"])
    temporal_metrics.update(result["summary"])

    # ── Packet Verifier + controller (ETRI 2차 step 7; gated, default OFF) ────
    # Adds severity/controller_decision to frame_records and a verifier_summary
    # to segment_records IN PLACE when enabled; a no-op otherwise (see
    # pipelines/packet_verification.py::maybe_run docstring for the exact contract).
    from sgdjscc_lab.pipelines.packet_verification import maybe_run as _maybe_run_verifier
    verifier_out = _maybe_run_verifier(result, cfg)
    if verifier_out is not None:
        logger.info("Packet verifier: %d frame(s) verified.", len(verifier_out["rows"]))

    # ── Persist outputs ──────────────────────────────────────────────────────
    kf_json = Path(OmegaConf.select(cfg, "keyframe_json", default="../outputs/keyframes.json"))
    kf_json.parent.mkdir(parents=True, exist_ok=True)
    structure = dict(result["keyframe_structure"])
    structure["files"] = [f.name for f in files]
    with open(kf_json, "w", encoding="utf-8") as fh:
        json.dump(structure, fh, indent=2)
    logger.info("Keyframe structure → %s", kf_json)

    frame_csv = Path(OmegaConf.select(cfg, "frame_log_csv", default="../outputs/temporal_frames.csv"))
    frame_csv.parent.mkdir(parents=True, exist_ok=True)
    flogs = result["frame_records"]
    if flogs:
        with open(frame_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flogs[0].keys()))
            w.writeheader()
            w.writerows(flogs)
    logger.info("Per-frame log → %s", frame_csv)

    tcsv = Path(OmegaConf.select(cfg, "temporal_csv", default="../outputs/temporal_metrics.csv"))
    tcsv.parent.mkdir(parents=True, exist_ok=True)
    with open(tcsv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(temporal_metrics.keys()))
        w.writeheader()
        w.writerow(temporal_metrics)
    logger.info("Temporal metrics → %s", tcsv)

    # ── GOP/segment records (ETRI 1차 step 4) ────────────────────────────────
    seg_json = Path(OmegaConf.select(cfg, "segment_json", default="../outputs/segments.json"))
    seg_json.parent.mkdir(parents=True, exist_ok=True)
    with open(seg_json, "w", encoding="utf-8") as fh:
        json.dump(result["segment_records"], fh, indent=2)
    logger.info("Segment records → %s", seg_json)

    # ── Reconstructed frames / video (ETRI 1차 step 1) ───────────────────────
    saved_frames = []
    if bool(OmegaConf.select(cfg, "video_io.save_recon_frames", default=True)):
        from torchvision.utils import save_image as tv_save_image
        recon_dir = Path(OmegaConf.select(
            cfg, "video_io.recon_frames_dir", default="../outputs/recon_frames"))
        recon_dir.mkdir(parents=True, exist_ok=True)
        # Remove recon_*.png left by a previous run so the folder holds exactly
        # this run's frames (a shorter re-run must not keep older tail frames).
        # Only that pattern is deleted — other files in the folder are kept.
        stale = sorted(recon_dir.glob("recon_*.png"))
        for f in stale:
            f.unlink()
        if stale:
            logger.info("Removed %d stale recon_*.png from %s", len(stale), recon_dir)
        for rec in result["records"]:
            if rec.recon is None:
                continue
            fp = recon_dir / f"recon_{rec.index:05d}.png"
            tv_save_image(rec.recon.cpu().float().clamp(0, 1), str(fp))
            saved_frames.append(fp)
        logger.info("Reconstructed frames → %s (%d files)", recon_dir, len(saved_frames))

    want_video = args.save_video or bool(
        OmegaConf.select(cfg, "video_io.save_recon_video", default=False))
    if want_video:
        from sgdjscc_lab.utils import video_io
        if not saved_frames:
            logger.warning("--save-video: no reconstructed frames were saved "
                           "(video_io.save_recon_frames is false?) — skipping mp4.")
        elif video_io.get_backend() is None:
            logger.warning("--save-video: no video backend (cv2/ffmpeg) available — skipping mp4.")
        else:
            fps = OmegaConf.select(cfg, "video_io.fps", default=None) or source_fps or 24.0
            vpath = Path(OmegaConf.select(
                cfg, "video_io.recon_video", default="../outputs/recon.mp4"))
            video_io.write_video(saved_frames, vpath, fps=float(fps))
            logger.info("Reconstructed video → %s", vpath)

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  Temporal evaluation complete")
    print("=" * 66)
    for k, v in temporal_metrics.items():
        print(f"  {k:<32} {v}")
    print(f"\n  Keyframes: {result['keyframe_structure']['keyframes']}")
    print(f"  Segments:  {len(result['segment_records'])}")


if __name__ == "__main__":
    main()

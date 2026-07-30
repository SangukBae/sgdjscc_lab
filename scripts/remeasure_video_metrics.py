#!/usr/bin/env python
"""remeasure_video_metrics.py – ETRI 5차 (step 9) held-out re-measurement CLI.

Re-reads a frame sequence through the SAME keyframe-oriented temporal
pipeline ``evaluate_video.py`` uses (so it sees the same
``orig_packet``/``recon_packet``/reconstructed-frame records 1~4차 already
produce), then recomputes packet-verifier reports + PTC/SFR/SDI **twice**:
"clip_only" (the plain CLIP-derived packet comparison 1~4차 already reported)
and "calibrated" (routed through the ETRI 5차 presence-calibration structure —
see ``evaluators/presence_calibration.py``). With the default config
(``verifier.use_presence_calibration: false``), "calibrated" is identical to
"clip_only" — this script only starts producing a non-zero diff once a real
presence calibrator (a second backend, e.g. OWLv2/VQA) is actually configured
and available.

Scope note: this reproduces the ETRI 5차 scaffold's promise — a
verifier-agnostic held-out remeasurement structure — not a verified
generation-quality or hallucination-detection improvement. Every report this
script produces is tagged ``metric_role: "held_out"``.

Optional object vocabulary filter (``verifier.object_vocabulary_filter.enabled: true``
in *--config*): strips non-object caption-noun contamination (count words like
"one", action/event words like "walking", scene words like "sidewalk") out of
the object metric before it reaches either the clip_only or calibrated column
— see ``evaluators/object_vocabulary_filter.py`` and
``pipelines/heldout_remeasurement.py``'s "Object vocabulary filter" docstring
section. Disabled by default; enabled in ``configs/etri_video_eval_ensemble.yaml``.

Alternative input #1: previously-saved packet JSON pairs (``--from-packets``),
for when you already have ``<stem>.orig_packet.json`` / ``<stem>.packet.json``
files on disk (the still-image pipeline's ``packet_dir`` convention — see
``utils/packet_io.py``) and don't want to re-run reconstruction at all. This
mode has no reconstructed-frame tensor, so calibration is limited to
image-free backends (``mock``/``gt``).

Alternative input #2: a completed run's saved frame folders
(``--from-recon-frames <run_dir>``), for reusing an already-finished
real-model batch (e.g. ``outputs/etri_video_eval_real_full_step50/baseline/
<video>/``) WITHOUT re-running reconstruction. Reads ``<run_dir>/
extracted_frames/`` (original input) + ``<run_dir>/recon_frames/`` (the
actual reconstruction that run produced) directly, so the reconstructed
image tensor IS byte-identical to that run — unlike ``--from-run``, which
always reconstructs fresh (see ``_build_items_from_run``'s fidelity note).
The packets themselves are freshly re-extracted from those frames (no packet
JSON was saved by that run) — see
``pipelines.heldout_remeasurement.items_from_recon_frame_dirs``'s docstring
for exactly what that does and does not preserve. This is the mode to use
for a real OWLv2/VQA sanity check against already-produced real-model
frames without spending GPU time on reconstruction again.

Usage
-----
# Re-run reconstruction (mock reconstruct_fn without --no-models omitted):
python scripts/remeasure_video_metrics.py --config configs/composed_video.yaml \\
    --input /path/to/frames/ --no-models --captions captions.txt

# From previously-saved packet JSON pairs (no reconstruction re-run):
python scripts/remeasure_video_metrics.py --config configs/composed_video.yaml \\
    --from-packets /path/to/packet_dir/

# From an already-completed run's saved frames (no reconstruction re-run,
# real reconstructed image available for owlv2/vqa/clip backends):
python scripts/remeasure_video_metrics.py --config configs/etri_video_eval_owlv2.yaml \\
    --from-recon-frames outputs/etri_video_eval_real_full_step50/baseline/01_person_walk \\
    --captions data/etri_video_eval/captions/01_person_walk.txt \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from omegaconf import OmegaConf

from sgdjscc_lab.config import load_config, merge_cli_overrides

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sgdjscc_lab.remeasure_video_metrics")


def _load_captions(captions_arg: Optional[str], files) -> Optional[list]:
    """Load per-frame caption strings aligned to *files* (order/length match).

    *captions_arg* may be a ``.txt`` file (one caption per line, indexed by
    frame position) or a directory of ``<stem>.txt`` files (one per frame,
    matched by filename stem) — the same two conventions
    ``generate_captions.py`` and ``run_etri_video_eval.py`` already use.
    Returns ``None`` if *captions_arg* is falsy (caller then gets no captions
    at all, not a list of empty strings).
    """
    if not captions_arg:
        return None
    cap_path = Path(captions_arg)
    if cap_path.is_file():
        lines = cap_path.read_text(encoding="utf-8").splitlines()
        return [lines[i].strip() if i < len(lines) else "" for i in range(len(files))]
    if cap_path.is_dir():
        return [
            (cap_path / f"{f.stem}.txt").read_text(encoding="utf-8").strip()
            if (cap_path / f"{f.stem}.txt").exists() else "" for f in files
        ]
    raise FileNotFoundError(f"--captions path not found: {cap_path}")


def _load_gt_metadata(gt_metadata_arg: Optional[str]) -> Optional[dict]:
    """Load ``--gt-metadata`` for the ``gt`` presence backend, accepting
    EITHER an already-converted ``{item_id: {object_name: bool}}`` mapping OR
    the raw ``data/etri_video_eval/gt/<video>.json`` segment-level format
    (``{"n_frames": ..., "segments": [...]}``) — the latter is auto-converted
    via ``pipelines.heldout_remeasurement.convert_gt_to_presence()``.

    Passing the raw segment-level file straight through unconverted used to
    silently produce a presence map whose keys never match any item_id
    convention at all (not an error — just quietly useless). Returns ``None``
    if *gt_metadata_arg* is falsy.
    """
    if not gt_metadata_arg:
        return None
    from sgdjscc_lab.pipelines.heldout_remeasurement import (
        convert_gt_to_presence, looks_like_segment_level_gt,
    )
    raw = json.loads(Path(gt_metadata_arg).read_text(encoding="utf-8"))
    if looks_like_segment_level_gt(raw):
        logger.info("--gt-metadata %s looks like raw segment-level GT — "
                    "auto-converting via convert_gt_to_presence().", gt_metadata_arg)
        return convert_gt_to_presence(raw)
    return raw


def _resolve_recon_frame_device(device_arg: Optional[str]):
    """Resolve the device used by ``--from-recon-frames`` packet re-extraction.

    This mode does not run JSCC/diffusion reconstruction, but it still builds
    CLIP for packet re-extraction and image-based presence backends may also
    use CUDA. Prefer GPU automatically when CUDA is available; keep an
    explicit ``--device`` override for operators who need a specific GPU or
    CPU fallback.
    """
    from sgdjscc_lab.runtime import resolve_device

    if device_arg:
        return resolve_device(str(device_arg))

    try:
        import torch
        auto = "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception as exc:  # noqa: BLE001 - broken driver/import should not hide CLI fallback
        logger.warning("Could not probe CUDA availability (%s); using CPU for packet re-extraction.", exc)
        auto = "cpu"
    return resolve_device(auto)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ETRI 5차 – held-out clip_only vs calibrated packet-verifier remeasurement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    p.add_argument("--input", "-i", default=None, help="Frame folder or video file (--from-run mode)")
    p.add_argument("--from-packets", default=None,
                   help="Directory of saved <stem>.orig_packet.json/<stem>.packet.json pairs "
                        "(utils/packet_io.py convention) — skips reconstruction entirely.")
    p.add_argument("--from-recon-frames", default=None,
                   help="A completed run's directory containing extracted_frames/ + recon_frames/ "
                        "(e.g. outputs/etri_video_eval_real_full_step50/baseline/<video>) — reuses "
                        "that run's actual reconstructed images (no reconstruction re-run); packets "
                        "are freshly re-extracted from the frames. See "
                        "pipelines.heldout_remeasurement.items_from_recon_frame_dirs.")
    p.add_argument("--no-models", action="store_true",
                   help="--from-run mode only: skip SGD-JSCC model loading (identity reconstruction).")
    p.add_argument("--captions", default=None,
                   help="Caption source (.txt file or per-frame directory) — used by --from-run mode "
                        "and by --from-recon-frames mode (for the reference/original frame's packet).")
    p.add_argument("--gt-metadata", default=None,
                   help="Optional JSON file for the 'gt' presence backend: either an already-converted "
                        "{item_id: {object_name: bool}} mapping, or the raw data/etri_video_eval/gt/"
                        "<video>.json segment-level format (auto-converted via convert_gt_to_presence()).")
    p.add_argument("--device", default=None,
                   help="Compute device override. In --from-recon-frames mode, default is auto "
                        "(cuda:0 when CUDA is available, otherwise cpu).")
    p.add_argument("--snr", type=float, default=None, help="AWGN SNR dB override (--from-run mode)")
    return p.parse_args()


def _build_items_from_run(cfg, args):
    """Re-run the frame sequence through TemporalPipeline (same construction
    as scripts/evaluate_video.py, INCLUDING its motion gate and 3~4차
    generate/bidirectional settings when the config enables them) and convert
    the records into RemeasurementItems.

    Fidelity note: this reconstructs frames from scratch using *cfg* — it
    does not replay a specific prior run's actual decisions. If a diffusion
    reconstruction is stochastic (no fixed seed) or *cfg* has drifted from
    whatever produced an earlier ``packet_match_report.json``, the
    reuse/recompute/generate decisions recomputed here can differ from that
    original run. For a byte-for-byte remeasurement of one specific prior
    run's numbers, prefer ``--from-packets`` against packets saved by that
    run (no reconstruction is repeated there). A saved-FrameRecord loader
    (bypassing reconstruction entirely) is a natural follow-up if this
    distinction matters for your use case.
    """
    from sgdjscc_lab.pipelines.heldout_remeasurement import items_from_temporal_records
    from sgdjscc_lab.video.scene_change_detector import SceneChangeDetector, SceneChangeConfig
    from sgdjscc_lab.video.keyframe_extractor import KeyframeExtractor
    from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline
    from sgdjscc_lab.io import list_image_files, load_image_as_tensor
    from sgdjscc_lab.utils import video_io

    if video_io.is_video_file(cfg.input_path):
        frames_root = Path(OmegaConf.select(cfg, "video_io.extracted_frames_dir", default="../outputs/video_frames"))
        extracted = video_io.extract_frames(cfg.input_path, frames_root / Path(cfg.input_path).stem)
        files = extracted["files"]
    else:
        files = list_image_files(cfg.input_path)
    frames = [load_image_as_tensor(f) for f in files]
    logger.info("Loaded %d ordered frames.", len(frames))

    captions = _load_captions(args.captions, files)

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

    from sgdjscc_lab.guidance.semantic_packet_extractor import SemanticPacketExtractor
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

    # Mirror evaluate_video.py's generate-branch gate (ETRI 3차/4차) so a
    # remeasurement run reconstructs frames under the SAME reuse/recompute/
    # generate policy the original run used, not silently a reuse/recompute-only
    # pipeline. Still subject to this function's fidelity note above.
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
        logger.info("Remeasuring with generate branch enabled: backend=%s conditioning_mode=%s",
                    video_generator.backend_name, conditioning_mode)

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
    return items_from_temporal_records(result["records"])


def _build_items_from_recon_frames(run_dir: str, args, gt_metadata_by_id=None):
    """Build items from a completed run's saved ``extracted_frames/`` +
    ``recon_frames/`` (see
    ``pipelines.heldout_remeasurement.items_from_recon_frame_dirs`` for the
    fidelity note on what is/isn't byte-identical to the original run). No
    reconstruction and no diffusion/JSCC model load happens here — only a
    CLIP evaluator is loaded, for packet re-extraction.
    """
    from sgdjscc_lab.pipelines.heldout_remeasurement import items_from_recon_frame_dirs
    from sgdjscc_lab.guidance.semantic_packet_extractor import SemanticPacketExtractor
    from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator
    from sgdjscc_lab.io import list_image_files

    run_path = Path(run_dir)
    recon_dir = run_path / "recon_frames"
    if not recon_dir.is_dir():
        sys.exit(f"Error: {recon_dir} not found — expected a run directory produced with "
                  f"video_io.save_recon_frames: true (e.g. an evaluate_video.py / "
                  f"run_etri_video_eval.py baseline-stage output directory).")

    extracted_root = run_path / "extracted_frames"
    extracted_dir = extracted_root
    if extracted_root.is_dir():
        # utils/video_io.py's extract_frames() nests frames one level under the
        # video's filename stem (extracted_frames/<video_stem>/frame_*.png) for
        # mp4 inputs — auto-descend into that single subdirectory when the top
        # level itself has no frame images.
        try:
            list_image_files(extracted_root)
        except FileNotFoundError:
            subdirs = [p for p in extracted_root.iterdir() if p.is_dir()]
            if len(subdirs) == 1:
                extracted_dir = subdirs[0]
    if not extracted_dir.is_dir():
        sys.exit(f"Error: {extracted_root} not found — needed to build the reference/original "
                  f"packet (video_io.extracted_frames_dir was not saved for this run).")

    orig_files = list_image_files(extracted_dir)
    captions = _load_captions(args.captions, orig_files)

    roles = None
    temporal_frames_csv = run_path / "temporal_frames.csv"
    if temporal_frames_csv.exists():
        import csv as _csv
        with open(temporal_frames_csv, newline="", encoding="utf-8") as fh:
            rows = sorted(_csv.DictReader(fh), key=lambda r: int(r["index"]))
        if len(rows) == len(orig_files):
            roles = [r.get("role") for r in rows]
        else:
            logger.warning(
                "%s has %d row(s) but %d extracted frame(s) were found — skipping "
                "role lookup (SDI will be None instead of misaligned).",
                temporal_frames_csv, len(rows), len(orig_files))

    device = _resolve_recon_frame_device(args.device)
    logger.info("Loading CLIP for packet re-extraction (device=%s) — "
                "no diffusion/JSCC model is loaded in this mode.", device)
    clip_eval = CLIPScoreEvaluator(device=device)
    packet_extractor = SemanticPacketExtractor(clip_evaluator=clip_eval, device=device)

    items = items_from_recon_frame_dirs(
        extracted_dir, recon_dir, packet_extractor,
        captions=captions, roles=roles, gt_metadata_by_id=gt_metadata_by_id,
    )
    logger.info("Loaded %d item(s) from %s (extracted_frames=%s, recon_frames=%s)",
                len(items), run_path, extracted_dir, recon_dir)
    return items


def _build_items_from_packets(packet_dir: str, gt_metadata_by_id=None):
    from sgdjscc_lab.pipelines.heldout_remeasurement import items_from_saved_packets
    from sgdjscc_lab.utils.packet_io import ORIG_PACKET_SUFFIX, PACKET_SUFFIX

    pairs = []
    for orig_path in sorted(glob.glob(str(Path(packet_dir) / f"*{ORIG_PACKET_SUFFIX}"))):
        stem = Path(orig_path).name[: -len(ORIG_PACKET_SUFFIX)]
        recon_path = Path(packet_dir) / f"{stem}{PACKET_SUFFIX}"
        if recon_path.exists():
            pairs.append((stem, orig_path, str(recon_path)))
    if not pairs:
        sys.exit(f"Error: no <stem>{ORIG_PACKET_SUFFIX} / <stem>{PACKET_SUFFIX} pairs found under {packet_dir}")
    logger.info("Loaded %d saved packet pairs from %s", len(pairs), packet_dir)
    return items_from_saved_packets(pairs, gt_metadata_by_id=gt_metadata_by_id)


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, input_path=args.input, snr_db=args.snr, device=args.device)

    from sgdjscc_lab.phase_gates import phase4_enabled
    if not phase4_enabled(cfg):
        sys.exit("Error: use_phase4 is false — the 5차 remeasurement pipeline requires 'use_phase4: true'.")

    gt_metadata_by_id = _load_gt_metadata(args.gt_metadata)

    if args.from_packets:
        items = _build_items_from_packets(args.from_packets, gt_metadata_by_id=gt_metadata_by_id)
    elif args.from_recon_frames:
        items = _build_items_from_recon_frames(args.from_recon_frames, args, gt_metadata_by_id=gt_metadata_by_id)
    else:
        if not args.input:
            sys.exit("Error: pass --input (frame folder/video), --from-packets <dir>, "
                      "or --from-recon-frames <run_dir>.")
        items = _build_items_from_run(cfg, args)

    presence_calibrator = None
    if bool(OmegaConf.select(cfg, "verifier.use_presence_calibration", default=False)):
        from sgdjscc_lab.evaluators.presence_calibration import build_presence_calibrator
        presence_calibrator = build_presence_calibrator(cfg)
        if presence_calibrator is not None:
            logger.info("Presence calibration ENABLED: mode=%s backends=%s",
                        presence_calibrator.mode, sorted(presence_calibrator.backends))
    if presence_calibrator is None:
        logger.info("Presence calibration disabled/unavailable — 'calibrated' will equal 'clip_only'.")

    from sgdjscc_lab.evaluators.object_vocabulary_filter import build_object_vocabulary_filter
    object_vocabulary_filter = build_object_vocabulary_filter(cfg)
    if object_vocabulary_filter is not None:
        logger.info("Object vocabulary filter ENABLED (use_gt_vocabulary=%s, %d excluded term(s)) — "
                    "count/action/scene tokens (e.g. 'one'/'walking'/'sidewalk') are excluded from "
                    "object metrics for both clip_only and calibrated.",
                    object_vocabulary_filter.use_gt_vocabulary, len(object_vocabulary_filter.excluded_terms))
    else:
        logger.info("Object vocabulary filter disabled — object metrics unchanged from prior behaviour.")

    from sgdjscc_lab.pipelines.heldout_remeasurement import remeasure, write_remeasurement
    result = remeasure(items, presence_calibrator=presence_calibrator, object_vocabulary_filter=object_vocabulary_filter)

    write_remeasurement(
        result,
        clip_only_json=OmegaConf.select(cfg, "heldout.clip_only_json", default=None),
        clip_only_csv=OmegaConf.select(cfg, "heldout.clip_only_csv", default=None),
        calibrated_json=OmegaConf.select(cfg, "heldout.calibrated_json", default=None),
        calibrated_csv=OmegaConf.select(cfg, "heldout.calibrated_csv", default=None),
        metric_delta_json=OmegaConf.select(cfg, "heldout.output_json", default=None),
        metric_delta_csv=OmegaConf.select(cfg, "heldout.output_csv", default=None),
    )

    print("\n" + "=" * 66)
    print("  Held-out remeasurement complete (ETRI 5차)")
    print("=" * 66)
    print(f"  n_items: {result['clip_only']['metrics']['n_items']}")
    for k in ("mean_severity", "ptc", "sfr", "sdi"):
        a = result["clip_only"]["metrics"].get(k)
        b = result["calibrated"]["metrics"].get(k)
        print(f"  {k:<16} clip_only={a}  calibrated={b}")
    print("\n  NOTE: mock backends only unless a real OWLv2/VQA backend was")
    print("  explicitly configured — this is a structural remeasurement, not a")
    print("  verified accuracy/quality result. See docs/etri_strategy.md 5차 구현 결과.")


if __name__ == "__main__":
    main()

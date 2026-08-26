#!/usr/bin/env python
"""run_transmission_reduction_eval.py – Transmission-reduction sweep driver.

Compares, per ETRI video, the *full* transmission size (visual latent +
caption + edge/structural guide + keyframe/manifest info — see
``sgdjscc_lab.transmission.packet_bundle``) against reconstruction quality
measured over the **whole video** (via the real ``video.temporal_pipeline``
orchestration, not just the selected keyframes), across:

    keyframe selector : fixed (scene-change + max_gop)  vs
                         SKEM/PSSS (real, optionally ALSO scene-change-forced
                         via ``keyframe.psss.use_scene_detector``)
    channel transport  : analog AWGN (baseline, kept separate — never used as
                          the digital Pareto baseline) vs digital_packet at
                          32-bit (lossless), 16-bit (reliable high-precision),
                          8/6/4-bit (real quantization + bit-packed binary
                          packet bundle, see ``sgdjscc_lab.transmission``)

Architecture
------------
``video.temporal_pipeline.TemporalPipeline`` drives the temporal policy. For
each digitally transmitted frame the sender serializes all visual patches,
per-patch captions, edge + uncertainty, patch layout and keyframe manifest.
The receiver reconstructs from those bytes through
``transmission.receiver_runtime``; the exact same artifact is saved and
counted. AWGN remains an analog baseline: its visual samples use the original
production path, while a shadow pass records the exact digital side-info bytes
and analog visual-symbol count as separate domains.

Scope note (exact vs. estimate — see ``transmission.byte_accounting``):
    - ``latent_elements`` / ``source_packet_bits`` / packet-bundle component
      bytes: EXACT (real serialized bytes, never estimated).
    - ``analog_channel_symbols``: exact count of real-valued channel symbols
      for a frame whose visual latent went over AWGN — never expressed as
      bytes, and never left as a fabricated 0 for a digital frame (it is
      ``None``/blank instead).
    - ``estimated_digital_channel_symbols`` / ``estimated_wire_bytes``:
      labeled proxy estimates (modulation/FEC assumptions), literally
      ``"unavailable"`` when no ``--bits-per-symbol`` was given — never a
      silently-fabricated number.
    - ``source_size_report.csv``: exact **source** MP4 file sizes only. A
      real H.264/H.265/AV1 CRF-sweep + quality-matched crossing-point search
      is a separate, already-existing tool
      (``scripts/benchmark_etri_video_rate.py``) — this script does not
      duplicate it; see that script's own docstring for the full comparison.

Example
-------
Correctness smoke run with explicit baselines:
    python scripts/run_transmission_reduction_eval.py \\
        --video-ids 01_person_walk --configs fixed_awgn,fixed_int16,skem_int4 \\
        --max-frames 20 --device cuda:0 --output-root outputs/transmission_reduction_smoke

Full sweep (all 10 videos, full config grid + keyframe sweep):
    python scripts/run_transmission_reduction_eval.py \\
        --configs fixed_awgn,fixed_int16,fixed_int8,fixed_int6,fixed_int4,skem_int16,skem_int8,skem_int6,skem_int4 \\
        --device cuda:0 --output-root outputs/transmission_reduction_$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

CHANNEL_CONFIGS = {
    "awgn": {"channel": "awgn", "bit_depth": None},
    "int8": {"channel": "digital_packet", "bit_depth": 8},
    "int6": {"channel": "digital_packet", "bit_depth": 6},
    "int4": {"channel": "digital_packet", "bit_depth": 4},
    # "Reliable digital" baselines: never AWGN — real digital transport with
    # negligible (int16) or zero (float32) quantization error, the fair
    # comparison point for what int8/6/4 give up. See CHANNEL_CONFIGS docstring
    # note in transmission/quantization.py.
    "int16": {"channel": "digital_packet", "bit_depth": 16},
    "float32": {"channel": "digital_packet", "bit_depth": 32},
}
SELECTORS = ("fixed", "skem")
ALL_CONFIGS = [f"{sel}_{ch}" for sel in SELECTORS for ch in CHANNEL_CONFIGS]
# fixed baseline (analog) + fixed/SKEM reliable-digital baseline (int16) +
# fixed/SKEM lossy digital (int8/6/4). float32 is available (ALL_CONFIGS) but
# not in the default grid. int16 is kept as the provisional reliable-digital
# reference; configurations with invalid frames cannot become a Pareto baseline.
DEFAULT_CONFIGS = [
    "fixed_awgn",
    "fixed_int16", "fixed_int8", "fixed_int6", "fixed_int4",
    "skem_int16", "skem_int8", "skem_int6", "skem_int4",
]
# Frame decisions that actually transmit a new visual latent (see
# video/temporal_pipeline.py: "reuse" replays the keyframe reconstruction
# in-memory, "generate" synthesizes from a mock backend — neither transmits a
# new latent, so neither gets a packet bundle / byte accounting entry).
TRANSMITTING_DECISIONS = ("keyframe", "recompute_semantic", "recompute_motion")

DEFAULT_PSSS_THRESHOLDS = (0.25, 0.35, 0.45, 0.55)
DEFAULT_MAX_SEGMENT_LENGTHS = (12, 16, 24, 32)

# Quality-degradation gate (task spec): a candidate config is "in budget" when
# it does not lose more than this much quality vs the reliable-digital baseline.
QUALITY_GATE = {"psnr_drop_db": 0.5, "ssim_drop": 0.01, "lpips_rise": 0.02}
# Preference order for the Pareto-frontier quality baseline: a real DIGITAL
# reliable baseline first (never AWGN — see module docstring's "fair baseline"
# note); AWGN is only ever a last-resort fallback (flagged explicitly).
BASELINE_PREFERENCE = ["fixed_float32", "fixed_int16", "skem_float32", "skem_int16", "fixed_awgn"]


def _load_manifest_reader():
    spec = importlib.util.spec_from_file_location(
        "_txred_run_etri_video_eval", _REPO_ROOT / "scripts" / "run_etri_video_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.read_manifest


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real packet-bundle/quantization/SKEM transmission-reduction sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-root", default=str(_REPO_ROOT / "data/etri_video_eval"))
    p.add_argument(
        "--config", default=None,
        help="Optional composed video config. Use the Wan/SKEM recipe here to exercise "
             "the real generator; otherwise the benchmark-safe default video config is used.",
    )
    p.add_argument("--output-root", required=True)
    p.add_argument("--video-ids", default=None, help="Comma-separated subset, default = all in manifest.")
    p.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                    help="Comma-separated subset of " + ",".join(ALL_CONFIGS))
    p.add_argument("--snr", type=float, default=10.0, help="AWGN baseline SNR (dB); irrelevant to digital configs.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-frames", type=int, default=None,
                    help="Cap frames per video fed to TemporalPipeline (smoke-test knob); default = all.")
    p.add_argument("--fps", type=float, default=None, help="recon.mp4 output fps; default = source fps.")
    # keyframe selection
    p.add_argument("--psss-threshold", type=float, default=0.35)
    p.add_argument("--psss-max-segment-length", type=int, default=16)
    p.add_argument("--psss-backend", default="proxy", choices=["mock", "proxy", "real"],
                    help="mock/proxy are NOT real PSSS — see docs/lgvsc_psss_skem_readiness.md.")
    p.add_argument("--psss-model-id", default=None, help="HF causal-LM/VLM id, required for --psss-backend real.")
    p.add_argument("--psss-device", default="cpu")
    p.add_argument("--psss-dtype", default="fp32")
    p.add_argument("--use-scene-detector", action="store_true",
                    help="Combine real scene-change detection with the SKEM/PSSS selector "
                         "(forces a keyframe at every detected scene boundary, structurally — "
                         "not inferred from any reason string).")
    p.add_argument("--fixed-max-gop", type=int, default=16, help="max_gop for the fixed selector.")
    p.add_argument("--skip-keyframe-sweep", action="store_true",
                    help="Skip the threshold x max_segment_length PSSS sweep (keyframe_sweep.csv).")
    p.add_argument("--skip-source-size-report", action="store_true",
                    help="Skip source_size_report.csv (exact source MP4 sizes only).")
    p.add_argument("--granularity", default="per_tensor", choices=["per_tensor", "per_channel"])
    p.add_argument("--no-lpips", action="store_true")
    # accounting estimates (labeled proxy; omit for "unavailable")
    p.add_argument("--bits-per-symbol", type=float, default=None,
                    help="Modulation assumption for estimated_digital_channel_symbols; "
                         "omitted = reported as 'unavailable', never fabricated.")
    p.add_argument("--code-rate", type=float, default=1.0, help="FEC code rate for estimated_wire_bytes.")
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_frames(video_path: Path, work_dir: Path):
    from sgdjscc_lab.utils.video_io import extract_frames
    from sgdjscc_lab.io import load_image_as_tensor

    info = extract_frames(video_path, work_dir)
    tensors = [load_image_as_tensor(f) for f in info["files"]]
    return tensors, info


def _load_captions(captions_path: Optional[Path], n_frames: int) -> Optional[List[str]]:
    if captions_path is None:
        return None
    lines = [ln.strip() for ln in captions_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None
    if len(lines) == 1:
        lines = lines * n_frames
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Keyframe selection (real scene-change + real PSSS/SKEM combination)
# ─────────────────────────────────────────────────────────────────────────────

def _psss_backend_cfg(args) -> Dict[str, Any]:
    return {
        "proxy": {"model_name": "ViT-B/32"},
        "real": {"model_id": args.psss_model_id, "device": args.psss_device, "dtype": args.psss_dtype},
    }


def _build_selector(name: str, captions: Optional[List[str]], threshold: float, max_segment_length: int, args):
    from omegaconf import OmegaConf
    from sgdjscc_lab.video.keyframe_extractor import build_caption_fn, build_keyframe_extractor
    from sgdjscc_lab.video.scene_change_detector import SceneChangeDetector

    scene_detector = SceneChangeDetector()

    if name == "fixed":
        cfg = OmegaConf.create({"keyframe": {"selector": "fixed", "max_gop": args.fixed_max_gop}})
        return build_keyframe_extractor(cfg, scene_detector=scene_detector)

    caption_source = "captions_file" if captions else "mock"
    caption_fn = build_caption_fn(caption_source, captions=captions)
    cfg = OmegaConf.create({
        "keyframe": {
            "selector": "psss",
            "psss": {
                "backend": args.psss_backend,
                "threshold": threshold,
                "max_segment_length": max_segment_length,
                "min_segment_length": 1,
                "caption_source": caption_source,
                "use_scene_detector": bool(args.use_scene_detector),
                **_psss_backend_cfg(args),
            },
        },
    })
    return build_keyframe_extractor(
        cfg, scene_detector=scene_detector, caption_fn=caption_fn,
    )


@dataclass
class KeyframeSelection:
    video: str
    selector: str
    threshold: Optional[float]
    max_segment_length: int
    n_frames: int
    keyframe_indices: List[int]
    n_keyframes: int
    force_reason: Dict[int, str]     # structured — never inferred from reason prose
    reasons: Dict[int, str]           # human-readable prose, informational only
    psss_scores: List[Dict]
    psss_backend_kind: Optional[str]  # "mock" | "proxy" | "real" — never conflated in results


def _select_keyframes(video_key, frames, captions, selector_name, threshold, max_segment_length, args) -> KeyframeSelection:
    selector = _build_selector(selector_name, captions, threshold, max_segment_length, args)
    result = selector.extract(frames)
    return _selection_from_result(
        video_key, selector_name, threshold, max_segment_length, len(frames), result
    )


def _selection_from_result(
    video_key, selector_name, threshold, max_segment_length, n_frames, result,
) -> KeyframeSelection:
    keyframes = list(result["keyframes"])
    reasons = {int(k): v for k, v in dict(result.get("keyframe_reasons", {})).items()}
    # Structured field — video/skem_selector.py's real force_reason (never
    # inferred by pattern-matching the prose `reasons` string). "fixed"
    # selector has no force_reason concept (KeyframeExtractor doesn't produce
    # per-decision categories) — every keyframe is simply "selected".
    raw_force = dict(result.get("force_reason", {}))
    force_reason = {int(k): v for k, v in raw_force.items()} if raw_force else {
        k: ("first_frame" if k == 0 else "selected") for k in keyframes
    }
    return KeyframeSelection(
        video=video_key, selector=selector_name, threshold=threshold,
        max_segment_length=max_segment_length, n_frames=n_frames,
        keyframe_indices=keyframes, n_keyframes=len(keyframes),
        force_reason=force_reason, reasons=reasons,
        psss_scores=list(result.get("psss_scores", [])),
        psss_backend_kind=result.get("psss_backend_kind"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Config / model setup
# ─────────────────────────────────────────────────────────────────────────────

_CFG_FRAGMENTS = (
    "base/channel/awgn", "base/model/sgdjscc", "base/infer/awgn",
    "base/eval/default", "base/video/default",
)


def _make_cfg(output_root: Path, model_root: Path, snr_db: float, config_path=None):
    """Compose a real config via the project's own fragment set (config.py's
    _defaults_ mechanism) rather than hand-rolling a minimal dict — guarantees
    every key run_single_image()/_jscc_forward() expects is present with its
    real default, exactly as every other entry point gets it."""
    from omegaconf import OmegaConf
    from sgdjscc_lab.config import load_config

    if config_path:
        source_path = Path(config_path).resolve()
        cfg = load_config(source_path)
        composed_path = output_root / "configs" / "source_config.txt"
        composed_path.parent.mkdir(parents=True, exist_ok=True)
        composed_path.write_text(str(source_path) + "\n", encoding="utf-8")
    else:
        composed_path = output_root / "configs" / "composed.yaml"
        composed_path.parent.mkdir(parents=True, exist_ok=True)
        frag_paths = [str((_REPO_ROOT / "configs" / f)) for f in _CFG_FRAGMENTS]
        composed_path.write_text(
            "_defaults_: [" + ", ".join(f'"{p}"' for p in frag_paths) + "]\n",
            encoding="utf-8",
        )
        cfg = load_config(composed_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({
        "model_root": str(model_root),
        "snr_db": float(snr_db),
        "use_phase4": True,
    }))
    return cfg


def _build_models(cfg, device):
    from sgdjscc_lab.runtime import build_models
    return build_models(cfg, device)


# ─────────────────────────────────────────────────────────────────────────────
# Full-video reconstruction via the REAL TemporalPipeline path
# ─────────────────────────────────────────────────────────────────────────────

LATENT_ELEMENTS_PER_PATCH = 16 * 16 * 16  # fixed VAE architecture constant


def _set_channel(models, channel_kind: str, bit_depth: Optional[int], granularity: str):
    from sgdjscc_lab.channels import DigitalPacketChannel

    if channel_kind == "awgn":
        models.jscc_model.channel_model = None  # original AWGN path, untouched
        return None
    ch = DigitalPacketChannel(bit_depth=bit_depth, granularity=granularity, channel_dim=1)
    models.jscc_model.channel_model = ch
    return ch


def _run_temporal_pipeline(
    frames, models, cfg, keyframe_extractor, captions=None, *, channel_kind="awgn",
    bit_depth=None, granularity="per_tensor", video_key="", config_name="",
    selected_keyframes=None,
):
    from sgdjscc_lab.pipelines.eval_pipeline import _reconstruct_with_cfg
    from sgdjscc_lab.guidance.semantic_packet_extractor import SemanticPacketExtractor
    from sgdjscc_lab.transmission.receiver_runtime import (
        encode_frame_to_bundle_bytes, reconstruct_frame_from_bundle_bytes,
    )
    from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline

    frame_index_by_id = {id(frame): i for i, frame in enumerate(frames)}
    transmitted_bundles = {}
    transmitted_semantic_packets = {}

    def reconstruct_fn(frame, run_cfg):
        resolved_cfg = run_cfg if run_cfg is not None else cfg
        index = frame_index_by_id[id(frame)]
        if channel_kind == "awgn":
            return _reconstruct_with_cfg(frame, models, resolved_cfg)
        data, n_elements = encode_frame_to_bundle_bytes(
            frame, models, resolved_cfg, bit_depth=bit_depth,
            granularity=granularity, keyframe_index=index,
            manifest={"video": video_key, "config": config_name},
            selected_keyframes=selected_keyframes,
            caption_override=(captions[index] if captions and index < len(captions) else None),
            semantic_packet=transmitted_semantic_packets.get(index),
        )
        # This dictionary is sender output storage only.  The receiver call
        # below is deliberately passed bytes, models and config — no frame or
        # sender-side tensors can cross the boundary.
        transmitted_bundles[index] = (data, n_elements)
        return reconstruct_frame_from_bundle_bytes(data, models, resolved_cfg)

    packet_extractor = SemanticPacketExtractor(
        text_extractor=getattr(models, "text_extractor", None),
        clip_evaluator=None, device=models.device,
    )

    def _caption_for(frame_id):
        fid = str(frame_id)
        if fid.startswith("recon_") or captions is None:
            return None
        try:
            idx = int(fid.split("_")[-1])
        except (ValueError, IndexError):
            return None
        return captions[idx] if 0 <= idx < len(captions) else None

    def packet_fn(frame, frame_id):
        packet = packet_extractor.extract(
            frame, frame_id=str(frame_id), caption=_caption_for(frame_id)
        )
        fid = str(frame_id)
        if fid.startswith("frame_"):
            index = int(fid.split("_")[-1])
            # Model the actual sender/receiver boundary for temporal decisions:
            # deterministic JSON bytes cross the link; TemporalPipeline sees
            # only the freshly parsed receiver-side object.
            payload = json.dumps(
                packet, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            parsed = json.loads(payload.decode("utf-8"))
            transmitted_semantic_packets[index] = parsed
            return parsed
        return packet

    from omegaconf import OmegaConf
    reuse_threshold = float(OmegaConf.select(cfg, "temporal.reuse_threshold", default=0.2))
    semantic_threshold = OmegaConf.select(cfg, "temporal.semantic_delta_threshold", default=None)
    if semantic_threshold is not None:
        reuse_threshold = float(semantic_threshold)
    motion_threshold = OmegaConf.select(cfg, "temporal.motion_threshold", default=None)

    from sgdjscc_lab.phase_gates import effective_flag
    enable_generate = effective_flag(cfg, "use_video_gen", phase=4) and bool(
        OmegaConf.select(cfg, "video_generator.enabled", default=False)
    )
    video_generator = None
    if enable_generate:
        from sgdjscc_lab.video.video_generator import build_generator
        video_generator = build_generator(cfg)

    pipeline = TemporalPipeline(
        reconstruct_fn=reconstruct_fn, packet_fn=packet_fn,
        keyframe_extractor=keyframe_extractor, cfg=cfg,
        reuse_threshold=reuse_threshold,
        motion_threshold=(None if motion_threshold is None else float(motion_threshold)),
        motion_weight=float(OmegaConf.select(cfg, "temporal.motion_weight", default=0.5)),
        motion_grid=int(OmegaConf.select(cfg, "temporal.motion_grid", default=8)),
        diffusion_step=int(cfg.get("diffusion_step", 50)),
        enable_generate=enable_generate,
        video_generator=video_generator,
        generate_delta_min=OmegaConf.select(cfg, "video_generator.generate_delta_min", default=None),
        generate_delta_max=OmegaConf.select(cfg, "video_generator.generate_delta_max", default=None),
        generate_motion_max=OmegaConf.select(cfg, "video_generator.generate_motion_max", default=None),
        allow_ground_truth_reference=bool(OmegaConf.select(
            cfg, "video_generator.allow_ground_truth_reference", default=False
        )),
        conditioning_mode=str(OmegaConf.select(
            cfg, "video_generator.conditioning_mode", default="start_only"
        )),
    )
    return pipeline.run(frames), transmitted_bundles, transmitted_semantic_packets


def _shadow_measure_frame(
    frames, index, models, cfg, channel_kind, bit_depth, granularity,
    video_key, config_name, selected_keyframes=None, caption_override=None,
    semantic_packet=None,
):
    """Build the analog baseline's digital side-information envelope.

    Digital configurations do not use this shadow path: their exact bytes are
    captured from the very bundle consumed by the real receiver.
    """
    from sgdjscc_lab.transmission.packet_bundle import parse_bundle
    from sgdjscc_lab.transmission.receiver_runtime import encode_frame_to_bundle_bytes

    data, n_elements = encode_frame_to_bundle_bytes(
        frames[index], models, cfg, bit_depth=bit_depth, granularity=granularity,
        keyframe_index=index, manifest={"video": video_key, "config": config_name},
        selected_keyframes=selected_keyframes,
        visual_is_analog=(channel_kind == "awgn"),
        caption_override=caption_override,
        semantic_packet=semantic_packet,
    )
    return parse_bundle(data), data, n_elements


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

def run(argv=None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output_root)

    configs = [c for c in args.configs.split(",") if c]
    for c in configs:
        if c not in ALL_CONFIGS:
            raise ValueError(f"unknown config {c!r}; expected one of {ALL_CONFIGS}")
    if args.psss_backend == "real" and not args.psss_model_id:
        raise ValueError("--psss-backend real requires --psss-model-id.")

    for sub in ("packets", "recon_videos", "configs", "logs"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    read_manifest = _load_manifest_reader()
    entries = read_manifest(Path(args.dataset_root))
    if args.video_ids:
        wanted = set(args.video_ids.split(","))
        entries = [e for e in entries if e["key"] in wanted]

    import torch
    device = torch.device(args.device)

    from sgdjscc_lab.evaluators.quality import QualityEvaluator
    quality_evaluator = QualityEvaluator(use_lpips=not args.no_lpips, device=device)

    from sgdjscc_lab.paths import model_root as _model_root
    cfg = _make_cfg(output_root, _model_root(), args.snr, config_path=args.config)
    models = _build_models(cfg, device)

    keyframe_rows: List[Dict[str, Any]] = []
    packet_rows: List[Dict[str, Any]] = []
    per_video_rows: List[Dict[str, Any]] = []
    keyframe_sweep_rows: List[Dict[str, Any]] = []

    log_path = output_root / "logs" / "run.log"
    with open(log_path, "a", encoding="utf-8") as log_fh:
        def log(msg):
            line = f"[{time.strftime('%H:%M:%S')}] {msg}"
            print(line)
            log_fh.write(line + "\n")
            log_fh.flush()

        for entry in entries:
            video_key = entry["key"]
            work_dir = output_root / "logs" / f"{video_key}_frames"
            log(f"loading frames for {video_key} ...")
            frames, info = _load_frames(entry["processed"], work_dir)
            if args.max_frames is not None:
                frames = frames[: args.max_frames]
            captions = _load_captions(entry["captions"], len(frames))

            if not args.skip_keyframe_sweep:
                for th in DEFAULT_PSSS_THRESHOLDS:
                    for max_len in DEFAULT_MAX_SEGMENT_LENGTHS:
                        sel = _select_keyframes(video_key, frames, captions, "skem", th, max_len, args)
                        keyframe_sweep_rows.append({
                            "video": video_key, "threshold": th, "max_segment_length": max_len,
                            "psss_backend_kind": sel.psss_backend_kind,
                            "n_frames": sel.n_frames, "n_keyframes": sel.n_keyframes,
                            "keyframe_indices": json.dumps(sel.keyframe_indices),
                        })
                        log(f"  keyframe_sweep {video_key} th={th} max_len={max_len} -> "
                            f"{sel.n_keyframes} keyframes ({sel.psss_backend_kind})")

            for config_name in configs:
                sel_name, ch_name = config_name.split("_", 1)
                channel_kind = "awgn" if ch_name == "awgn" else "digital_packet"
                bit_depth = CHANNEL_CONFIGS[ch_name]["bit_depth"]

                keyframe_extractor = _build_selector(
                    sel_name, captions, args.psss_threshold, args.psss_max_segment_length, args
                )
                # PSSS/scene detection may be expensive.  Compute it exactly
                # once, then give TemporalPipeline a cache wrapper so the same
                # structure drives both reconstruction and accounting.
                selection_result = keyframe_extractor.extract(frames)
                sel = _selection_from_result(
                    video_key, sel_name, args.psss_threshold,
                    args.psss_max_segment_length, len(frames), selection_result,
                )

                class _CachedExtractor:
                    def extract(self, _frames):
                        return selection_result

                log(f"[{video_key}][{config_name}] running TemporalPipeline over {len(frames)} frames "
                    f"(selector={sel_name}, channel={ch_name}, psss_backend_kind={sel.psss_backend_kind})")

                _set_channel(models, channel_kind, bit_depth, args.granularity)
                start = time.time()
                result, transmitted_bundles, transmitted_semantic_packets = _run_temporal_pipeline(
                    frames, models, cfg, _CachedExtractor(), captions,
                    channel_kind=channel_kind, bit_depth=bit_depth,
                    granularity=args.granularity, video_key=video_key,
                    config_name=config_name, selected_keyframes=sel.keyframe_indices,
                )
                video_elapsed = time.time() - start
                records = sorted(result["records"], key=lambda r: r.index)

                # ── save every reconstructed frame + assemble recon.mp4 ──
                from sgdjscc_lab.io import save_tensor_as_image
                from sgdjscc_lab.utils.video_io import write_video

                recon_dir = output_root / "recon_videos" / video_key / config_name
                frame_files = []
                video_psnr, video_ssim, video_lpips = [], [], []
                n_nan_frames = 0
                for rec in records:
                    if rec.recon is None:
                        continue
                    if torch.isnan(rec.recon).any() or torch.isinf(rec.recon).any():
                        # Known, reproducible fragility (not a bug in this
                        # feature's own code — see README's "Known limitations"):
                        # jscc.snr_prediction_net (the blind SNR predictor used
                        # by _compute_step's step_style="continuous" branch) was
                        # only ever trained on AWGN-shaped degradation. Coarse
                        # digital quantization (observed with bit_depth=8 on
                        # some real frames) can push its predicted_signal_scale
                        # to >= 1, making cur_step <= 0 and
                        # 10*log10(1/cur_step - 1) evaluate log10 of a
                        # non-positive number -> NaN, which then propagates
                        # through the (otherwise correct, untouched per this
                        # repo's algorithm-preservation invariant) diffusion
                        # decode. Excluded from the quality average rather than
                        # silently poisoning mean_psnr/ssim/lpips into "nan",
                        # and always counted so it's visible, never hidden.
                        n_nan_frames += 1
                        log(f"  [{video_key}][{config_name}] frame {rec.index}: NaN/Inf reconstruction "
                            f"(quality gate fails; invalid frame is not written to recon.mp4)")
                        continue
                    fpath = recon_dir / f"frame_{rec.index:05d}.png"
                    save_tensor_as_image(rec.recon, fpath)
                    frame_files.append(fpath)
                    original = frames[rec.index]
                    h = min(rec.recon.shape[-2], original.shape[-2])
                    w = min(rec.recon.shape[-1], original.shape[-1])
                    m = quality_evaluator.evaluate(original[..., :h, :w], rec.recon[..., :h, :w])
                    video_psnr.append(m["psnr"])
                    video_ssim.append(m["ssim"])
                    if m["lpips"] is not None:
                        video_lpips.append(m["lpips"])
                if len(frame_files) == len(records):
                    fps = args.fps or info.get("fps") or 24.0
                    write_video(frame_files, recon_dir / "recon.mp4", fps=fps)
                elif n_nan_frames:
                    (recon_dir / "INVALID_RECONSTRUCTION.json").parent.mkdir(parents=True, exist_ok=True)
                    (recon_dir / "INVALID_RECONSTRUCTION.json").write_text(json.dumps({
                        "n_frames_total": len(records),
                        "n_nan_or_inf_frames": n_nan_frames,
                        "reason": "recon.mp4 omitted because at least one reconstructed frame was non-finite",
                    }, indent=2), encoding="utf-8")

                # ── shadow accounting pass: only frames that transmitted a new latent ──
                video_bytes = 0
                video_symbols_analog = 0
                video_symbols_latent = 0
                transmitting = [r for r in records if r.decision in TRANSMITTING_DECISIONS]
                for rec in records:
                    visual_transmitted = rec.decision in TRANSMITTING_DECISIONS
                    semantic_packet = transmitted_semantic_packets.get(rec.index)
                    if semantic_packet is None:
                        raise RuntimeError(f"missing serialized semantic packet for frame {rec.index}")

                    if not visual_transmitted:
                        from sgdjscc_lab.transmission.packet_bundle import (
                            build_side_info_bundle, serialize_bundle,
                        )
                        bundle = build_side_info_bundle(
                            keyframe_index=rec.index,
                            manifest={
                                "video": video_key, "config": config_name,
                                "decision": rec.decision,
                                "selected_keyframes": sel.keyframe_indices,
                            },
                            semantic_packet=semantic_packet,
                        )
                        serialized = serialize_bundle(bundle)
                        n_elements = 0
                    elif channel_kind == "awgn":
                        bundle, serialized, n_elements = _shadow_measure_frame(
                            frames, rec.index, models, cfg, channel_kind, bit_depth,
                            args.granularity, video_key, config_name,
                            selected_keyframes=sel.keyframe_indices,
                            caption_override=(
                                captions[rec.index] if captions and rec.index < len(captions) else None
                            ),
                            semantic_packet=semantic_packet,
                        )
                    else:
                        if rec.index not in transmitted_bundles:
                            raise RuntimeError(
                                f"missing receiver-consumed bundle for transmitting frame {rec.index}"
                            )
                        serialized, n_elements = transmitted_bundles[rec.index]
                        from sgdjscc_lab.transmission.packet_bundle import parse_bundle
                        bundle = parse_bundle(serialized)
                    from sgdjscc_lab.transmission.byte_accounting import measure_frame_transmission

                    measurement = measure_frame_transmission(
                        bundle, latent_elements=n_elements,
                        visual_is_analog=(channel_kind == "awgn" and visual_transmitted),
                        bits_per_symbol=args.bits_per_symbol, code_rate=args.code_rate,
                    )
                    video_symbols_latent += n_elements
                    video_bytes += bundle.total_exact_bytes()
                    if channel_kind == "awgn" and visual_transmitted:
                        video_symbols_analog += n_elements

                    bundle_dir = output_root / "packets" / video_key / config_name
                    bundle_dir.mkdir(parents=True, exist_ok=True)
                    (bundle_dir / f"frame_{rec.index:05d}.sgbundle").write_bytes(serialized)

                    force_reason = (
                        sel.force_reason.get(rec.index, "selected") if visual_transmitted else ""
                    )
                    m_dict = measurement.as_dict()
                    keyframe_rows.append({
                        "video": video_key, "config": config_name, "frame_index": rec.index,
                        "selector": sel_name, "decision": rec.decision, "force_reason": force_reason,
                        "visual_transmitted": visual_transmitted,
                        "reason": sel.reasons.get(rec.index, ""),
                        "psss_backend_kind": sel.psss_backend_kind,
                        **m_dict,
                    })

                    breakdown_rows = {
                        "caption_bytes": sum(
                            it.byte_len for it in bundle.items if it.name in ("caption", "captions")
                        ),
                        "edge_bytes": bundle.get("edge").byte_len if bundle.get("edge") else 0,
                        "edge_uncertainty_bytes": (
                            bundle.get("edge_uncertainty").byte_len
                            if bundle.get("edge_uncertainty") else 0
                        ),
                        "manifest_bytes": bundle.get("manifest").byte_len if bundle.get("manifest") else 0,
                        "semantic_packet_bytes": (
                            bundle.get("semantic_packet").byte_len
                            if bundle.get("semantic_packet") else 0
                        ),
                        "visual_bytes": sum(it.byte_len for it in bundle.items if it.name.startswith("visual")),
                        "bundle_overhead_bytes": bundle.overhead_exact_bytes(),
                        "total_bundle_bytes": bundle.total_exact_bytes(),
                        "video": video_key, "config": config_name, "frame_index": rec.index,
                        "bit_depth": bit_depth if bit_depth is not None else "",
                    }
                    packet_rows.append(breakdown_rows)

                n = max(len(video_psnr), 1)
                n_kf_in_gop = sel.n_keyframes
                per_video_rows.append({
                    "video": video_key, "config": config_name, "selector": sel_name,
                    "channel": ch_name, "bit_depth": bit_depth if bit_depth is not None else "",
                    "psss_backend_kind": sel.psss_backend_kind,
                    "n_frames_total": len(frames), "n_transmitting_frames": len(transmitting),
                    "n_keyframes_selected": n_kf_in_gop, "n_nan_or_inf_frames": n_nan_frames,
                    "n_quality_frames": len(video_psnr),
                    "valid_frame_ratio": len(video_psnr) / max(len(frames), 1),
                    "mean_psnr": sum(video_psnr) / n if video_psnr else float("nan"),
                    "mean_ssim": sum(video_ssim) / n if video_ssim else float("nan"),
                    "mean_lpips": (sum(video_lpips) / len(video_lpips)) if video_lpips else "",
                    "latent_elements_total": video_symbols_latent,
                    "analog_channel_symbols_total": video_symbols_analog if channel_kind == "awgn" else "",
                    "source_packet_bits_total": video_bytes * 8,
                    "digital_side_information_bytes_total": video_bytes if channel_kind == "awgn" else "",
                    "total_bundle_bytes": video_bytes,
                    "analog_no_wire_bytes": channel_kind == "awgn",
                    "visual_transport_complete": channel_kind != "awgn",
                    "total_elapsed_s": round(video_elapsed, 3),
                })
                nan_note = f" ({n_nan_frames} NaN/Inf frames excluded)" if n_nan_frames else ""
                log(f"[{video_key}][{config_name}] frames={len(frames)} transmitting={len(transmitting)} "
                    f"mean_psnr={per_video_rows[-1]['mean_psnr']:.4f}{nan_note} "
                    f"bytes={per_video_rows[-1]['total_bundle_bytes']}")

    _write_csv(output_root / "per_video_metrics.csv", per_video_rows)
    _write_csv(output_root / "keyframe_selection.csv", keyframe_rows)
    _write_csv(output_root / "packet_components.csv", packet_rows)
    _write_csv(output_root / "keyframe_sweep.csv", keyframe_sweep_rows)
    aggregate_rows = _aggregate(per_video_rows)
    _write_csv(output_root / "aggregate.csv", aggregate_rows)
    pareto_rows, baseline_info = _pareto_frontier(aggregate_rows)
    _write_csv(output_root / "pareto_frontier.csv", pareto_rows)

    source_size_rows: List[Dict[str, Any]] = []
    if not args.skip_source_size_report:
        source_size_rows = _source_size_report(entries)
    _write_csv(output_root / "source_size_report.csv", source_size_rows)

    _write_readme_and_summary(output_root, args, per_video_rows, pareto_rows, baseline_info)
    return 0


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(per_video_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_config: Dict[str, List[Dict[str, Any]]] = {}
    for r in per_video_rows:
        by_config.setdefault(r["config"], []).append(r)
    out = []
    for config, rows in by_config.items():
        n = len(rows)
        byte_rows = [r["total_bundle_bytes"] for r in rows if r["total_bundle_bytes"] != ""]
        lpips_rows = [r["mean_lpips"] for r in rows if r["mean_lpips"] != ""]
        out.append({
            "config": config,
            "selector": rows[0]["selector"],
            "channel": rows[0]["channel"],
            "bit_depth": rows[0]["bit_depth"],
            "psss_backend_kind": rows[0]["psss_backend_kind"],
            "n_videos": n,
            "total_frames": sum(r.get("n_frames_total", 1) for r in rows),
            "total_quality_frames": sum(
                r.get("n_quality_frames", r.get("n_frames_total", 1)) for r in rows
            ),
            "valid_frame_ratio": (
                sum(r.get("n_quality_frames", r.get("n_frames_total", 1)) for r in rows)
                / max(sum(r.get("n_frames_total", 1) for r in rows), 1)
            ),
            "mean_psnr": sum(r["mean_psnr"] for r in rows) / n,
            "mean_ssim": sum(r["mean_ssim"] for r in rows) / n,
            "mean_lpips": (sum(lpips_rows) / len(lpips_rows)) if lpips_rows else "",
            "mean_latent_elements": sum(r["latent_elements_total"] for r in rows) / n,
            "mean_total_bundle_bytes": (sum(byte_rows) / len(byte_rows)) if byte_rows else "",
            "analog_no_wire_bytes": rows[0]["analog_no_wire_bytes"],
            "total_nan_or_inf_frames": sum(r.get("n_nan_or_inf_frames", 0) for r in rows),
        })
    return out


def _pareto_frontier(aggregate_rows: List[Dict[str, Any]]):
    by_config = {r["config"]: r for r in aggregate_rows}
    baseline = None
    baseline_config = None
    for candidate in BASELINE_PREFERENCE:
        if candidate in by_config and by_config[candidate].get("total_nan_or_inf_frames", 0) == 0:
            baseline = by_config[candidate]
            baseline_config = candidate
            break
    baseline_is_analog = bool(baseline and baseline.get("analog_no_wire_bytes"))
    baseline_info = {
        "baseline_config": baseline_config,
        "baseline_is_analog": baseline_is_analog,
        "baseline_valid": baseline is not None,
    }
    if baseline is None:
        return [], baseline_info

    candidates = [r for r in aggregate_rows
                  if r["config"] != baseline_config and r["mean_total_bundle_bytes"] != ""]
    in_budget = []
    for r in candidates:
        psnr_drop = baseline["mean_psnr"] - r["mean_psnr"]
        ssim_drop = baseline["mean_ssim"] - r["mean_ssim"]
        lpips_rise = (
            (r["mean_lpips"] - baseline["mean_lpips"])
            if (r["mean_lpips"] != "" and baseline["mean_lpips"] != "") else None
        )
        invalid_frames = int(r.get("total_nan_or_inf_frames", 0))
        valid_frame_ratio = float(r.get("valid_frame_ratio", 1.0))
        ok = (
            invalid_frames == 0
            and valid_frame_ratio == 1.0
            and psnr_drop <= QUALITY_GATE["psnr_drop_db"]
            and ssim_drop <= QUALITY_GATE["ssim_drop"]
            and (lpips_rise is None or lpips_rise <= QUALITY_GATE["lpips_rise"])
        )
        row = dict(r)
        row["baseline_config"] = baseline_config
        row["psnr_drop_db"] = psnr_drop
        row["ssim_drop"] = ssim_drop
        row["lpips_rise"] = lpips_rise if lpips_rise is not None else ""
        row["within_quality_gate"] = ok
        row["quality_gate_failure_reason"] = (
            "non_finite_frames" if invalid_frames else
            "incomplete_quality_coverage" if valid_frame_ratio < 1.0 else
            "quality_threshold" if not ok else ""
        )
        in_budget.append(row)

    selected = [r for r in in_budget if r["within_quality_gate"]]
    pool = selected if selected else in_budget  # spec: if none qualify, report nearest, don't hide it
    pool_sorted = sorted(pool, key=lambda r: r["mean_total_bundle_bytes"])
    for i, r in enumerate(pool_sorted):
        r["rank"] = i
        r["selected_as_smallest_in_budget"] = bool(selected) and i == 0
    return pool_sorted, baseline_info


def _source_size_report(entries) -> List[Dict[str, Any]]:
    rows = []
    for entry in entries:
        try:
            size = entry["processed"].stat().st_size
        except OSError:
            continue
        rows.append({
            "video": entry["key"],
            "source_file_bytes": size,
            "note": "exact source MP4 size only — this is NOT a codec-vs-semantic "
                    "quality comparison. Run scripts/benchmark_etri_video_rate.py "
                    "separately (against this run's recon_videos/<video>/<config>/recon.mp4) "
                    "for a real H.264/H.265/AV1 CRF sweep + quality-matched crossing-point search.",
        })
    return rows


def _write_readme_and_summary(output_root, args, per_video_rows, pareto_rows, baseline_info):
    summary = {
        "output_root": str(output_root),
        "configs_run": args.configs.split(","),
        "n_videos": len({r["video"] for r in per_video_rows}),
        "quality_gate": QUALITY_GATE,
        "pareto_baseline": baseline_info,
        "pareto_selected": next((r for r in pareto_rows if r.get("selected_as_smallest_in_budget")), None),
        "psss_backend_requested": args.psss_backend,
        "psss_model_id": args.psss_model_id,
        "use_scene_detector": bool(args.use_scene_detector),
        "bits_per_symbol": args.bits_per_symbol,
        "code_rate": args.code_rate,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if not baseline_info["baseline_valid"]:
        baseline_note = (
            "Pareto baseline: unavailable. Include a valid reliable-digital config "
            "(preferably float32/int16) or fixed_awgn to build a comparison frontier."
        )
    else:
        baseline_note = (
            f"Pareto baseline: `{baseline_info['baseline_config']}` "
            + ("(**analog fallback** — no valid reliable digital baseline (int16/float32) was available; "
               "quality-gate comparisons against an analog baseline mix quantization loss with AWGN "
               "noise and should be treated cautiously)."
               if baseline_info["baseline_is_analog"] else "(a reliable digital baseline, never AWGN).")
        )

    readme = f"""# transmission_reduction run — {output_root.name}

Real packet-bundle (all visual patches + per-patch captions + edge/uncertainty + manifest) transmission-size
accounting, full-video quality via the real `TemporalPipeline` path, and
SKEM/PSSS keyframe selection (optionally scene-change-combined). See the
module docstring of `scripts/run_transmission_reduction_eval.py` for the
exact-vs-estimate accounting boundaries and the digital bundle-only receiver boundary.

{baseline_note}

- `per_video_metrics.csv` / `aggregate.csv` — full-video quality (PSNR/SSIM/
  LPIPS over every reconstructed frame, not just keyframes) + exact
  transmission-bundle bytes per (video, config).
- `keyframe_selection.csv` — per transmitting frame: decision, structured
  `force_reason` (`first_frame`|`scene_change`|`max_segment_length`|`psss`|
  `selected` — never inferred from prose), the 5-field measurement schema
  (`latent_elements`/`analog_channel_symbols`/`source_packet_bits`/
  `estimated_digital_channel_symbols`/`estimated_wire_bytes`), and
  `psss_backend_kind` (`mock`|`proxy`|`real` — never conflated).
- `packet_components.csv` — exact per-frame bundle byte breakdown (caption/
  edge/edge-uncertainty/visual/manifest payloads + container overhead), with
  `total_bundle_bytes` equal to the actual `.sgbundle` file size.
- `packets/<video>/<config>/frame_NNNNN.sgbundle` — the actual serialized
  transmission bundles (visual+caption+edge+manifest) a receiver would parse.
- `recon_videos/<video>/<config>/recon.mp4` + `frame_*.png` — the FULL
  reconstructed video (every frame, not just keyframes).
- `keyframe_sweep.csv` — PSSS threshold x max_segment_length grid (selection
  only, no reconstruction); reports `psss_backend_kind` per row.
- `pareto_frontier.csv` — smallest-bytes config meeting the quality gate
  (PSNR drop <= {QUALITY_GATE['psnr_drop_db']} dB, SSIM drop <= {QUALITY_GATE['ssim_drop']},
  LPIPS rise <= {QUALITY_GATE['lpips_rise']}) against the reliable-digital
  baseline above; if none qualify, the nearest candidates are still listed.
- `source_size_report.csv` — exact source MP4 sizes only (see note above).
- `summary.json` — run configuration + selected config + baseline used.

Known limitations:
- `--psss-backend {args.psss_backend}` was used for keyframe selection this
  run — only `real` (with `--psss-model-id`) is genuine PSSS (an actual
  causal-LM/VLM's yes/no token probability); `mock`/`proxy` are explicitly
  NOT real PSSS (see `video/psss.py`'s module docstring) and every CSV/JSON
  in this run tags rows with `psss_backend_kind` so this is never conflated.
- `estimated_digital_channel_symbols`/`estimated_wire_bytes` are labeled
  proxy estimates (`{'unavailable — no --bits-per-symbol given' if args.bits_per_symbol is None else f'bits_per_symbol={args.bits_per_symbol}'}`)
  — no real modulator/FEC coder exists in this codebase.
- Digital configs reconstruct from the exact `.sgbundle` bytes saved in this
  run. AWGN cannot be reconstructed from a byte bundle because its visual
  waveform is analog; for AWGN, `analog_channel_symbols_total` and the exact
  digital caption/edge/manifest bytes are reported as separate domains.
- **Known numerical fragility at coarse digital quantization** (found via GPU
  verification, not this feature's own bug): `jscc.snr_prediction_net` (the
  blind SNR predictor `_compute_step()` uses, `pipelines/infer_pipeline.py`,
  untouched by this feature) was only ever trained on AWGN-shaped
  degradation. On some real frames, bit_depth=8 quantization pushes its
  predicted signal scale to >= 1, making `10*log10(1/cur_step - 1)` evaluate
  `log10` of a non-positive number -> NaN, which then propagates through the
  (otherwise correct) diffusion decode. This driver detects it
  (`n_nan_or_inf_frames` in per_video_metrics.csv/aggregate.csv). Any such
  config unconditionally fails the quality gate, cannot be Pareto-selected,
  and does not get a misleading `recon.mp4`; finite-frame diagnostic means
  remain available together with `valid_frame_ratio`.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(run())

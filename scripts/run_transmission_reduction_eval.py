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
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# Hard dependency (not soft/optional): reproducibility manifests are a
# required part of this feature, not a best-effort extra. run_manifest.py
# is import-light (no torch) — see its own module docstring — so importing
# it at module level does not slow down a plain --help/--dry-run invocation,
# and a missing/broken module fails IMMEDIATELY (import error at script
# start) rather than degrading to a "status: unavailable" placeholder deep
# into a run.
from sgdjscc_lab.utils import run_manifest as rm  # noqa: E402

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
# Per-video calibration grid used only by --match-actual-transmissions.  PSSS
# S_rel lives in (-1, 1); the endpoints make "almost every semantic change"
# and "almost no semantic change" representable while keeping the search
# deterministic and small enough to run before the expensive reconstructions.
DEFAULT_MATCHED_RATE_THRESHOLDS = tuple(
    round(-0.95 + 0.05 * index, 10) for index in range(39)
) + (0.999999,)
DEFAULT_MATCHED_RATE_MAX_SEGMENT_LENGTHS = (8, 10, 12, 14, 16, 20, 24, 32, 48, 64, 100)

# Quality-degradation gate (task spec): a candidate config is "in budget" when
# it does not lose more than this much quality vs the reliable-digital baseline.
QUALITY_GATE = {"psnr_drop_db": 0.5, "ssim_drop": 0.01, "lpips_rise": 0.02}
# Preference order for the Pareto-frontier quality baseline: float32
# (lossless) or int16 (near-lossless, real affine quantization) ONLY — never
# AWGN (mixing analog noise with quantization loss is not a meaningful
# "reliable digital" reference, see module docstring's "fair baseline" note)
# and never a lossier bit_depth even as a fallback. A config only qualifies
# if it additionally has zero non-finite frames (checked in
# _pareto_frontier) — "정상" int16 in the task sense, not merely "int16".
BASELINE_PREFERENCE = ["fixed_float32", "fixed_int16", "skem_float32", "skem_int16"]


def _load_manifest_reader():
    spec = importlib.util.spec_from_file_location(
        "_txred_run_etri_video_eval", _REPO_ROOT / "scripts" / "run_etri_video_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.read_manifest


def _run_effect_summarizer(output_root: Path) -> None:
    """Write effect tables before the final manifest hashes its artifacts."""
    spec = importlib.util.spec_from_file_location(
        "_txred_summarize_transmission_normalization",
        _REPO_ROOT / "scripts" / "summarize_transmission_normalization.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    rc = mod.run(["--run-root", str(output_root)])
    if rc != 0:
        raise RuntimeError(f"effect summarizer exited with status {rc}")


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
    p.add_argument(
        "--match-fixed-keyframes", action="store_true",
        help="Per video, replace the fixed selector with FixedCountKeyframeSelector so its "
             "keyframe count exactly matches this run's SKEM selection for the same video. "
             "Rate matching additionally requires the measured bundle-byte tolerance, instead of comparing "
             "fixed's fixed --fixed-max-gop against whatever count SKEM happens to pick.",
    )
    p.add_argument(
        "--match-actual-transmissions", action="store_true",
        help="Keep the fixed max-GOP selector unchanged and calibrate SKEM per video so the "
             "actual number of visual-transmitting decisions (keyframe + recompute_semantic/"
             "recompute_motion) exactly matches fixed. This is the fair matched-rate mode; "
             "it is mutually exclusive with legacy --match-fixed-keyframes.",
    )
    p.add_argument(
        "--matched-rate-thresholds",
        default=",".join(str(value) for value in DEFAULT_MATCHED_RATE_THRESHOLDS),
        help="Comma-separated deterministic PSSS threshold search grid used by "
             "--match-actual-transmissions.",
    )
    p.add_argument(
        "--matched-rate-max-segment-lengths",
        default=",".join(str(value) for value in DEFAULT_MATCHED_RATE_MAX_SEGMENT_LENGTHS),
        help="Comma-separated positive max-segment search grid used by "
             "--match-actual-transmissions.",
    )
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
    # reproducibility
    p.add_argument("--seed", type=int, default=2025,
                    help="Base seed for Python/NumPy/PyTorch/CUDA. A deterministic per-(video, "
                         "frame) seed derived from this is set before each frame's reconstruction "
                         "so different configs reconstructing the SAME frame share the same RNG "
                         "state as much as possible (see utils/seed.py::derive_frame_seed).")
    p.add_argument(
        "--retry-failed", action="store_true",
        help="On resume, retry pairs already recorded in failed_pairs.csv. By default failed "
             "pairs are skipped, preventing duplicate failure rows.",
    )
    # digital step-matching policy (see pipelines/infer_pipeline.py::DIGITAL_STEP_POLICIES)
    p.add_argument(
        "--digital-step-policy", default="fixed_reference",
        choices=["fixed_reference", "bitdepth_proxy", "quant_nmse"],
        help="How the diffusion decoder step is derived for a digital_packet frame. "
             "'fixed_reference' (default): every bit_depth decoded as if it were the clean "
             "float32 reference -- the quantization-effect comparison isolates raw quantization "
             "distortion, uncontaminated by the decoder also adapting its diffusion strength per "
             "bit_depth (any run with a non-default value here is a decoder-step ablation, not "
             "the quantization comparison -- see --ablation-label). 'bitdepth_proxy': a "
             "deterministic bit_depth-only heuristic, explicitly labeled a proxy, never a real "
             "channel SNR. 'quant_nmse': the sender's OWN measured quantization SNR for this "
             "exact packet, read by the receiver from the packet's own metadata.",
    )
    p.add_argument(
        "--fixed-reference-snr-db", type=float, default=10.0,
        help="Decoder reference SNR used by digital-step-policy=fixed_reference. "
             "Recorded in resolved config, per-video/aggregate rows, summary, manifest, and "
             "resume signature so a 10 dB quantization run cannot be resumed at another step.",
    )
    p.add_argument(
        "--ablation-label", default=None,
        help="Free-form label recorded in per_video_metrics.csv/aggregate.csv/summary.json "
             "identifying this run as a decoder-step ablation (e.g. 'bitdepth_proxy_ablation') "
             "rather than the quantization comparison. Required when --digital-step-policy is "
             "not 'fixed_reference', so a bitdepth_proxy/quant_nmse run can never be silently "
             "mixed into a quantization_effect.csv table meant to isolate quantization alone.",
    )
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


def _build_selector(
    name: str, captions: Optional[List[str]], threshold: float, max_segment_length: int, args,
    fixed_count_override: Optional[int] = None,
):
    """*fixed_count_override*, when given, builds the "fixed" selector as a
    :class:`~sgdjscc_lab.video.keyframe_extractor.FixedCountKeyframeSelector`
    producing EXACTLY that many keyframes (not an approximation via
    ``max_gop`` — see ``run()``'s per-video block, which passes SKEM's own
    achieved keyframe count here for an exact fixed-vs-SKEM keyframe-count
    match)."""
    from omegaconf import OmegaConf
    from sgdjscc_lab.video.keyframe_extractor import build_caption_fn, build_keyframe_extractor
    from sgdjscc_lab.video.scene_change_detector import SceneChangeDetector

    scene_detector = SceneChangeDetector()

    if name == "fixed":
        if fixed_count_override is not None:
            cfg = OmegaConf.create({
                "keyframe": {"selector": "fixed_count", "fixed_count": {"count": int(fixed_count_override)}},
            })
            return build_keyframe_extractor(cfg, scene_detector=scene_detector)
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


def _select_keyframes(
    video_key, frames, captions, selector_name, threshold, max_segment_length, args,
    fixed_count_override: Optional[int] = None,
) -> KeyframeSelection:
    selector = _build_selector(
        selector_name, captions, threshold, max_segment_length, args,
        fixed_count_override=fixed_count_override,
    )
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


def _parse_float_grid(text: str, *, name: str) -> List[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain one or more finite comma-separated floats")
    return sorted(set(values))


def _parse_positive_int_grid(text: str, *, name: str) -> List[int]:
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values or any(value < 1 for value in values):
        raise ValueError(f"{name} must contain one or more positive comma-separated integers")
    return sorted(set(values))


class _CachingPsssBackend:
    """Memoise PSSS results across the matched-rate threshold/grid search.

    The selector is autoregressive, so different thresholds revisit many of
    the same caption pairs.  Real/proxy backends can be expensive; caching the
    immutable score preserves the exact decision while avoiding repeated
    model calls.  Provenance attributes are delegated to the wrapped backend.
    """

    def __init__(self, backend) -> None:
        self.backend = backend
        self.cache: Dict[tuple, Any] = {}

    def score(self, info_a, info_b, semantic_focus):
        key = (str(info_a or ""), str(info_b or ""), str(semantic_focus))
        if key not in self.cache:
            self.cache[key] = self.backend.score(info_a, info_b, semantic_focus)
        return self.cache[key]

    def __getattr__(self, name):
        return getattr(self.backend, name)


def _extract_rate_planning_packets(frames, captions, models) -> List[Dict[str, Any]]:
    """Extract the exact sender-side semantic packets once for rate planning.

    The canonical JSON round-trip mirrors ``_run_temporal_pipeline``'s actual
    sender/receiver boundary.  Temporal reuse/recompute decisions depend only
    on these original-frame packets and the selected keyframe anchors, not on
    the reconstructed pixels or digital bit depth.
    """
    from sgdjscc_lab.guidance.semantic_packet_extractor import SemanticPacketExtractor

    extractor = SemanticPacketExtractor(
        text_extractor=getattr(models, "text_extractor", None),
        clip_evaluator=None,
        device=models.device,
    )
    packets: List[Dict[str, Any]] = []
    for index, frame in enumerate(frames):
        caption = captions[index] if captions and index < len(captions) else None
        packet = extractor.extract(frame, frame_id=f"frame_{index:05d}", caption=caption)
        payload = json.dumps(
            packet, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        packets.append(json.loads(payload.decode("utf-8")))
    return packets


def _planned_transmitting_indices(
    selection_result: Dict[str, Any], semantic_packets: List[Dict[str, Any]],
    *, reuse_threshold: float,
) -> List[int]:
    """Mirror the default TemporalPipeline visual-transmission decisions.

    This planner deliberately supports the matched-rate wrapper's locked
    default contract only: semantic gate enabled, motion gate disabled and
    video-generation branch disabled.  Under that contract an inter frame
    transmits iff its semantic delta from the current keyframe anchor is not
    below ``reuse_threshold``.  The full run verifies the planned count
    against the actual ``FrameRecord.decision`` values and fails closed on a
    mismatch.
    """
    from sgdjscc_lab.video.semantic_delta import SemanticDelta

    keyframes = {int(index) for index in selection_result.get("keyframes", [])}
    if semantic_packets and 0 not in keyframes:
        raise ValueError("matched-rate selector must include frame 0 as a keyframe")
    delta = SemanticDelta()
    anchor = None
    transmitting: List[int] = []
    for index, packet in enumerate(semantic_packets):
        if index in keyframes:
            anchor = packet
            transmitting.append(index)
            continue
        if anchor is None:
            raise ValueError(f"no keyframe anchor available for frame {index}")
        if delta.compute(anchor, packet)["magnitude"] >= float(reuse_threshold):
            transmitting.append(index)
    return transmitting


def _build_actual_transmission_plan(
    video_key: str, frames, captions, models, cfg, args,
) -> tuple:
    """Choose a per-video SKEM operating point matching fixed transmissions.

    Returns ``(plan_row, fixed_result, skem_result)``.  Candidate selection is
    lexicographic: actual transmitting-frame count error first, then keyframe
    count distance (a useful byte-proximity proxy once visual transmission
    counts are equal), then distance from the documented default SKEM knobs.
    An exact transmission-count candidate is mandatory; the expensive full
    reconstructions never start for a video whose rate cannot be calibrated.
    """
    from omegaconf import OmegaConf

    motion_threshold = OmegaConf.select(cfg, "temporal.motion_threshold", default=None)
    generate_enabled = bool(OmegaConf.select(cfg, "video_generator.enabled", default=False))
    if motion_threshold is not None or generate_enabled:
        raise ValueError(
            "--match-actual-transmissions requires temporal.motion_threshold=null and "
            "video_generator.enabled=false so the precomputed semantic schedule exactly "
            "matches TemporalPipeline decisions"
        )
    reuse_threshold = float(OmegaConf.select(cfg, "temporal.reuse_threshold", default=0.2))
    semantic_threshold = OmegaConf.select(cfg, "temporal.semantic_delta_threshold", default=None)
    if semantic_threshold is not None:
        reuse_threshold = float(semantic_threshold)

    semantic_packets = _extract_rate_planning_packets(frames, captions, models)
    fixed_extractor = _build_selector(
        "fixed", captions, args.psss_threshold, args.psss_max_segment_length, args,
    )
    fixed_result = fixed_extractor.extract(frames)
    fixed_tx = _planned_transmitting_indices(
        fixed_result, semantic_packets, reuse_threshold=reuse_threshold,
    )
    fixed_keyframes = list(fixed_result.get("keyframes", []))

    thresholds = _parse_float_grid(
        args.matched_rate_thresholds, name="--matched-rate-thresholds",
    )
    max_lengths = _parse_positive_int_grid(
        args.matched_rate_max_segment_lengths,
        name="--matched-rate-max-segment-lengths",
    )
    candidates = []
    caching_backend = None
    for threshold in thresholds:
        for max_length in max_lengths:
            extractor = _build_selector(
                "skem", captions, threshold, max_length, args,
            )
            if caching_backend is None:
                caching_backend = _CachingPsssBackend(extractor.psss_backend)
            extractor.psss_backend = caching_backend
            result = extractor.extract(frames)
            transmitting = _planned_transmitting_indices(
                result, semantic_packets, reuse_threshold=reuse_threshold,
            )
            keyframes = list(result.get("keyframes", []))
            score = (
                abs(len(transmitting) - len(fixed_tx)),
                abs(len(keyframes) - len(fixed_keyframes)),
                abs(float(threshold) - float(args.psss_threshold)),
                abs(int(max_length) - int(args.psss_max_segment_length)),
                float(threshold), int(max_length),
            )
            candidates.append((score, threshold, max_length, result, transmitting))

    candidates.sort(key=lambda item: item[0])
    _, threshold, max_length, skem_result, skem_tx = candidates[0]
    exact_candidates = sum(
        len(item[4]) == len(fixed_tx) for item in candidates
    )
    exact = len(skem_tx) == len(fixed_tx)
    if not exact:
        raise RuntimeError(
            f"[{video_key}] no SKEM candidate matched fixed's actual visual-transmission "
            f"count={len(fixed_tx)} across {len(candidates)} candidates; closest selected "
            f"{len(skem_tx)}. Expand --matched-rate-thresholds/"
            "--matched-rate-max-segment-lengths before running full reconstruction."
        )

    plan_row = {
        "video": video_key,
        "mode": "actual_transmissions",
        "fixed_max_gop": int(args.fixed_max_gop),
        "reuse_threshold": reuse_threshold,
        "psss_backend_kind": skem_result.get("psss_backend_kind", ""),
        "selected_psss_threshold": float(threshold),
        "selected_psss_max_segment_length": int(max_length),
        "fixed_n_keyframes": len(fixed_keyframes),
        "skem_n_keyframes": len(skem_result.get("keyframes", [])),
        "target_n_transmitting_frames": len(fixed_tx),
        "skem_planned_n_transmitting_frames": len(skem_tx),
        "transmitting_frame_count_exact": exact,
        "fixed_keyframe_indices": json.dumps(fixed_keyframes),
        "skem_keyframe_indices": json.dumps(list(skem_result.get("keyframes", []))),
        "fixed_transmitting_indices": json.dumps(fixed_tx),
        "skem_transmitting_indices": json.dumps(skem_tx),
        "n_candidates_evaluated": len(candidates),
        "n_exact_transmission_count_candidates": exact_candidates,
    }
    return plan_row, fixed_result, skem_result


# ─────────────────────────────────────────────────────────────────────────────
# Config / model setup
# ─────────────────────────────────────────────────────────────────────────────

_CFG_FRAGMENTS = (
    "base/channel/awgn", "base/model/sgdjscc", "base/infer/awgn",
    "base/eval/default", "base/video/default",
)


def _make_cfg(
    output_root: Path,
    model_root: Path,
    snr_db: float,
    config_path=None,
    *,
    fixed_reference_snr_db: float = 10.0,
):
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
        "digital_fixed_reference_snr_db": float(fixed_reference_snr_db),
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
    selected_keyframes=None, log_fn=None, digital_step_policy="fixed_reference",
    base_seed=None,
):
    """Reconstruct *frames* via TemporalPipeline.

    Returns ``(pipeline_result, transmitted_bundles, transmitted_semantic_packets,
    quantization_diagnostics)``
    on success — success means every frame's reconstruction was finite by
    construction (enforced deep inside the forward pass by ``pipelines/
    infer_pipeline.py``'s ``assert_finite`` stage guards).

    If any frame hits a :class:`~sgdjscc_lab.utils.finite_checks.NonFiniteError`,
    this does NOT substitute a placeholder and keep going: the exception is
    enriched with ``{"video", "config", "frame_index"}`` (added to
    ``exc.context``, while this closure still has all three in scope) and
    RE-RAISED, aborting this ``(video, config)`` pair immediately — the
    caller (``run()``'s per-config loop) catches it, records the failure, and
    moves on to the next pair. Any OTHER (unexpected) exception is not caught
    here at all and stops the whole run — only ``NonFiniteError`` gets this
    special "abort just this pair" handling.
    """
    from sgdjscc_lab.pipelines.eval_pipeline import _reconstruct_with_cfg
    from sgdjscc_lab.guidance.semantic_packet_extractor import SemanticPacketExtractor
    from sgdjscc_lab.transmission.receiver_runtime import (
        encode_frame_to_bundle_bytes, reconstruct_frame_from_bundle_bytes,
    )
    from sgdjscc_lab.utils.finite_checks import NonFiniteError
    from sgdjscc_lab.utils.seed import derive_frame_seed, set_global_seed
    from sgdjscc_lab.video.temporal_pipeline import TemporalPipeline

    log_fn = log_fn or (lambda msg: None)
    frame_index_by_id = {id(frame): i for i, frame in enumerate(frames)}
    transmitted_bundles = {}
    transmitted_semantic_packets = {}
    quantization_diagnostics: List[Dict[str, Any]] = []

    def reconstruct_fn(frame, run_cfg):
        resolved_cfg = run_cfg if run_cfg is not None else cfg
        index = frame_index_by_id[id(frame)]
        if base_seed is not None:
            # Deterministic per-(video, frame) seed shared across every
            # config reconstructing this same frame, so a residual
            # RNG-dependent step (e.g. an unconditional diffusion draw) is
            # aligned across the configs being compared instead of adding
            # incidental RNG-draw noise on top of the real channel/selector
            # difference under test.
            set_global_seed(derive_frame_seed(base_seed, video_key, index))
        try:
            if channel_kind == "awgn":
                return _reconstruct_with_cfg(frame, models, resolved_cfg)
            frame_quantization_diagnostics: List[Dict[str, Any]] = []
            data, n_elements = encode_frame_to_bundle_bytes(
                frame, models, resolved_cfg, bit_depth=bit_depth,
                granularity=granularity, keyframe_index=index,
                manifest={"video": video_key, "config": config_name},
                selected_keyframes=selected_keyframes,
                caption_override=(captions[index] if captions and index < len(captions) else None),
                semantic_packet=transmitted_semantic_packets.get(index),
                include_quantization_error_metadata=(digital_step_policy == "quant_nmse"),
                quantization_diagnostics=frame_quantization_diagnostics,
            )
            for diagnostic in frame_quantization_diagnostics:
                quantization_diagnostics.append({
                    "video": video_key,
                    "config": config_name,
                    "frame_index": index,
                    "wire_metadata": digital_step_policy == "quant_nmse",
                    **diagnostic,
                })
            # This dictionary is sender output storage only.  The receiver call
            # below is deliberately passed bytes, models and config — no frame or
            # sender-side tensors can cross the boundary.
            transmitted_bundles[index] = (data, n_elements)
            return reconstruct_frame_from_bundle_bytes(
                data, models, resolved_cfg, digital_step_policy=digital_step_policy,
            )
        except NonFiniteError as exc:
            exc.context.update({"video": video_key, "config": config_name, "frame_index": index})
            log_fn(
                f"  [{video_key}][{config_name}] frame {index}: non-finite at stage={exc.stage!r} "
                f"({exc.n_nan} NaN, {exc.n_inf} Inf / {exc.numel} elements) — "
                "ABORTING this (video, config) pair, not processing further frames"
            )
            raise

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
    return (
        pipeline.run(frames), transmitted_bundles, transmitted_semantic_packets,
        quantization_diagnostics,
    )


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
    if args.digital_step_policy != "fixed_reference" and not args.ablation_label:
        raise ValueError(
            f"--digital-step-policy={args.digital_step_policy!r} is a decoder-step ablation, "
            "not the quantization comparison ('fixed_reference' is) -- pass --ablation-label "
            "so this run can never be silently mixed into quantization_effect.csv."
        )
    if args.match_fixed_keyframes and args.match_actual_transmissions:
        raise ValueError(
            "--match-fixed-keyframes and --match-actual-transmissions are mutually exclusive"
        )
    if args.match_actual_transmissions:
        fixed_channels = {
            name.split("_", 1)[1] for name in configs if name.startswith("fixed_")
        }
        skem_channels = {
            name.split("_", 1)[1] for name in configs if name.startswith("skem_")
        }
        if fixed_channels != skem_channels or not fixed_channels or "awgn" in fixed_channels:
            raise ValueError(
                "--match-actual-transmissions requires paired fixed/skem digital configs "
                "with identical channel sets and no AWGN row"
            )
        _parse_float_grid(
            args.matched_rate_thresholds, name="--matched-rate-thresholds",
        )
        _parse_positive_int_grid(
            args.matched_rate_max_segment_lengths,
            name="--matched-rate-max-segment-lengths",
        )

    for sub in ("packets", "recon_videos", "configs", "logs"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    read_manifest = _load_manifest_reader()
    entries = read_manifest(Path(args.dataset_root))
    if args.video_ids:
        wanted = set(args.video_ids.split(","))
        entries = [e for e in entries if e["key"] in wanted]
    expected_video_keys = {e["key"] for e in entries}

    from sgdjscc_lab.paths import model_root as _model_root
    cfg = _make_cfg(
        output_root,
        _model_root(),
        args.snr,
        config_path=args.config,
        fixed_reference_snr_db=args.fixed_reference_snr_db,
    )

    # Run signature + resume-safety check FIRST (before any heavy model load):
    # a resume targeting the same --output-root under DIFFERENT conditions
    # (different commit/dataset/config/checkpoint/seed/video list/frame cap/
    # granularity/PSSS settings/eval options) is refused immediately, and the
    # INITIAL manifest/run spec is recorded before any video is processed.
    signature = _build_run_signature(args, cfg, entries, _model_root())
    if signature.get("git_commit") == rm.UNKNOWN:
        raise SystemExit(
            "experiment provenance unavailable: git commit is unknown. Install git in the "
            "container, keep the checkout's .git metadata mounted, or inject the verified host "
            "commit with SGDJSCC_GIT_COMMIT before running the sweep."
        )
    if signature.get("git_dirty") is True:
        raise SystemExit(
            "experiment checkout has tracked/indexed changes relative to HEAD. Commit or restore "
            "them before running so the recorded commit fully identifies the executed code."
        )
    _check_resume_signature(output_root, signature)

    per_video_rows_initial: List[Dict[str, Any]] = [
        _coerce_per_video_row(r) for r in _read_csv_dicts(output_root / "per_video_metrics.csv")
    ]
    failed_pairs: List[Dict[str, Any]] = _read_csv_dicts(output_root / "failed_pairs.csv")
    if not (output_root / "run_manifest_initial.json").exists():
        _write_manifest(
            args, output_root, cfg, per_video_rows_initial, failed_pairs,
            signature, phase="initial",
        )

    from sgdjscc_lab.utils.finite_checks import NonFiniteError
    from sgdjscc_lab.utils.seed import set_global_seed
    set_global_seed(int(args.seed))

    import torch
    device = torch.device(args.device)

    from sgdjscc_lab.evaluators.quality import QualityEvaluator
    quality_evaluator = QualityEvaluator(use_lpips=not args.no_lpips, device=device)

    models = _build_models(cfg, device)

    # Resume: reload any prior (possibly interrupted) run's CSVs from this same
    # output_root so already-completed (video, config) pairs are skipped, not
    # redone. An empty/fresh output_root reloads nothing (normal full run).
    keyframe_rows: List[Dict[str, Any]] = _read_csv_dicts(output_root / "keyframe_selection.csv")
    packet_rows: List[Dict[str, Any]] = _read_csv_dicts(output_root / "packet_components.csv")
    keyframe_sweep_rows: List[Dict[str, Any]] = _read_csv_dicts(output_root / "keyframe_sweep.csv")
    quantization_diagnostic_rows: List[Dict[str, Any]] = _read_csv_dicts(
        output_root / "quantization_diagnostics.csv"
    )
    matched_rate_plan_rows: List[Dict[str, Any]] = _read_csv_dicts(
        output_root / "matched_rate_plan.csv"
    )
    matched_rate_plan_by_video = {
        row["video"]: row for row in matched_rate_plan_rows
    }
    per_video_rows: List[Dict[str, Any]] = per_video_rows_initial
    done_pairs = {(r["video"], r["config"]) for r in per_video_rows}
    failed_pair_keys = {(r["video"], r["config"]) for r in failed_pairs}
    done_sweep_videos = {r["video"] for r in keyframe_sweep_rows}

    log_path = output_root / "logs" / "run.log"
    with open(log_path, "a", encoding="utf-8") as log_fh:
        def log(msg):
            line = f"[{time.strftime('%H:%M:%S')}] {msg}"
            print(line)
            log_fh.write(line + "\n")
            log_fh.flush()

        if done_pairs:
            log(f"resume: {len(done_pairs)} (video,config) pair(s) already completed in {output_root} — will skip those")
        if failed_pair_keys:
            action = "will retry once" if args.retry_failed else "will skip (use --retry-failed to retry)"
            log(f"resume: {len(failed_pair_keys)} failed (video,config) pair(s) already recorded — {action}")

        for entry in entries:
            video_key = entry["key"]
            pending_configs = [
                name for name in configs
                if (video_key, name) not in done_pairs
                and (args.retry_failed or (video_key, name) not in failed_pair_keys)
            ]
            needs_keyframe_sweep = (
                not args.skip_keyframe_sweep and video_key not in done_sweep_videos
            )
            if not pending_configs and not needs_keyframe_sweep:
                log(f"resume: [{video_key}] no pending config or keyframe sweep, skipping video load")
                continue
            work_dir = output_root / "logs" / f"{video_key}_frames"
            log(f"loading frames for {video_key} ...")
            frames, info = _load_frames(entry["processed"], work_dir)
            if args.max_frames is not None:
                frames = frames[: args.max_frames]
            captions = _load_captions(entry["captions"], len(frames))

            fixed_count_target = None
            matched_plan_row = None
            selector_results: Dict[str, Any] = {}
            selector_summaries: Dict[str, KeyframeSelection] = {}
            # Derive this from the full run config, not only pending pairs: a
            # partial resume may have just fixed or just SKEM left, while its
            # selector plan must remain the same paired plan as the original.
            requested_selectors = {name.split("_", 1)[0] for name in configs}
            if args.match_actual_transmissions:
                matched_plan_row, fixed_result, skem_result = _build_actual_transmission_plan(
                    video_key, frames, captions, models, cfg, args,
                )
                existing_plan = matched_rate_plan_by_video.get(video_key)
                if existing_plan is not None:
                    stable_fields = (
                        "mode", "fixed_max_gop", "reuse_threshold", "psss_backend_kind",
                        "selected_psss_threshold", "selected_psss_max_segment_length",
                        "fixed_n_keyframes", "skem_n_keyframes",
                        "target_n_transmitting_frames", "skem_planned_n_transmitting_frames",
                        "transmitting_frame_count_exact", "fixed_keyframe_indices",
                        "skem_keyframe_indices", "fixed_transmitting_indices",
                        "skem_transmitting_indices", "n_candidates_evaluated",
                        "n_exact_transmission_count_candidates",
                    )
                    mismatches = [
                        field for field in stable_fields
                        if str(existing_plan.get(field, "")) != str(matched_plan_row.get(field, ""))
                    ]
                    if mismatches:
                        raise RuntimeError(
                            f"[{video_key}] matched-rate plan changed on resume for fields "
                            f"{mismatches}; refusing to mix selector schedules"
                        )
                else:
                    matched_rate_plan_rows.append(matched_plan_row)
                    matched_rate_plan_by_video[video_key] = matched_plan_row
                    _write_csv(output_root / "matched_rate_plan.csv", matched_rate_plan_rows)

                selector_results["fixed"] = fixed_result
                selector_results["skem"] = skem_result
                selector_summaries["fixed"] = _selection_from_result(
                    video_key, "fixed", args.psss_threshold,
                    args.psss_max_segment_length, len(frames), fixed_result,
                )
                selector_summaries["skem"] = _selection_from_result(
                    video_key, "skem", matched_plan_row["selected_psss_threshold"],
                    matched_plan_row["selected_psss_max_segment_length"], len(frames), skem_result,
                )
                log(
                    f"  match_actual_transmissions {video_key}: fixed max_gop={args.fixed_max_gop} "
                    f"and calibrated SKEM threshold={matched_plan_row['selected_psss_threshold']} "
                    f"max_segment={matched_plan_row['selected_psss_max_segment_length']} both plan "
                    f"{matched_plan_row['target_n_transmitting_frames']} visual transmissions "
                    f"({matched_plan_row['n_exact_transmission_count_candidates']} exact candidates)"
                )
            elif "skem" in requested_selectors or args.match_fixed_keyframes:
                skem_extractor = _build_selector(
                    "skem", captions, args.psss_threshold,
                    args.psss_max_segment_length, args,
                )
                selector_results["skem"] = skem_extractor.extract(frames)
                selector_summaries["skem"] = _selection_from_result(
                    video_key, "skem", args.psss_threshold,
                    args.psss_max_segment_length, len(frames), selector_results["skem"],
                )
            if args.match_fixed_keyframes and not args.match_actual_transmissions:
                ref_sel = selector_summaries["skem"]
                if 1 <= ref_sel.n_keyframes <= len(frames):
                    fixed_count_target = ref_sel.n_keyframes
                    log(f"  match_fixed_keyframes {video_key}: skem selected {ref_sel.n_keyframes} "
                        f"keyframes over {len(frames)} frames -> fixed selector forced to exactly "
                        f"{fixed_count_target} keyframes (FixedCountKeyframeSelector)")
                else:
                    log(f"  match_fixed_keyframes {video_key}: skem selected {ref_sel.n_keyframes} "
                        f"keyframes, which FixedCountKeyframeSelector cannot represent over "
                        f"{len(frames)} frames -- falling back to --fixed-max-gop "
                        f"({args.fixed_max_gop}); keyframe_count_matched=False for this video")

            if "fixed" in requested_selectors and not args.match_actual_transmissions:
                fixed_extractor = _build_selector(
                    "fixed", captions, args.psss_threshold,
                    args.psss_max_segment_length, args,
                    fixed_count_override=fixed_count_target,
                )
                selector_results["fixed"] = fixed_extractor.extract(frames)
                selector_summaries["fixed"] = _selection_from_result(
                    video_key, "fixed", args.psss_threshold,
                    args.psss_max_segment_length, len(frames), selector_results["fixed"],
                )

            run_keyframe_sweep = "skem" in requested_selectors and not args.skip_keyframe_sweep
            if run_keyframe_sweep and video_key not in done_sweep_videos:
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
                _write_csv(output_root / "keyframe_sweep.csv", keyframe_sweep_rows)
            elif run_keyframe_sweep:
                log(f"resume: keyframe_sweep already recorded for {video_key}, skipping recompute")

            for config_name in configs:
                if (video_key, config_name) in done_pairs:
                    log(f"resume: [{video_key}][{config_name}] already completed, skipping")
                    continue
                if (video_key, config_name) in failed_pair_keys and not args.retry_failed:
                    log(f"resume: [{video_key}][{config_name}] previously failed, skipping")
                    continue
                sel_name, ch_name = config_name.split("_", 1)
                channel_kind = "awgn" if ch_name == "awgn" else "digital_packet"
                bit_depth = CHANNEL_CONFIGS[ch_name]["bit_depth"]

                # PSSS/scene detection is computed once per video/selector and
                # reused across every bit depth/channel configuration.
                selection_result = selector_results[sel_name]
                sel = selector_summaries[sel_name]

                class _CachedExtractor:
                    def extract(self, _frames):
                        return selection_result

                log(f"[{video_key}][{config_name}] running TemporalPipeline over {len(frames)} frames "
                    f"(selector={sel_name}, channel={ch_name}, psss_backend_kind={sel.psss_backend_kind})")

                _set_channel(models, channel_kind, bit_depth, args.granularity)
                start = time.time()
                try:
                    (
                        result, transmitted_bundles, transmitted_semantic_packets,
                        pair_quantization_diagnostics,
                    ) = _run_temporal_pipeline(
                        frames, models, cfg, _CachedExtractor(), captions,
                        channel_kind=channel_kind, bit_depth=bit_depth,
                        granularity=args.granularity, video_key=video_key,
                        config_name=config_name, selected_keyframes=sel.keyframe_indices,
                        log_fn=log, digital_step_policy=args.digital_step_policy,
                        base_seed=int(args.seed),
                    )
                except NonFiniteError as exc:
                    failed_pairs = [
                        r for r in failed_pairs
                        if (r["video"], r["config"]) != (video_key, config_name)
                    ]
                    failed_pairs.append({
                        "video": video_key, "config": config_name,
                        "failure_stage": exc.stage, "frame_index": exc.context.get("frame_index", ""),
                        "n_nan": exc.n_nan, "n_inf": exc.n_inf, "numel": exc.numel,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    })
                    log(f"[{video_key}][{config_name}] ABORTED: non-finite at stage={exc.stage!r} "
                        f"frame={exc.context.get('frame_index', '?')} ({exc.n_nan} NaN, {exc.n_inf} Inf) "
                        "— this pair produces no per_video_metrics.csv row; moving to next pair")
                    _write_csv(output_root / "failed_pairs.csv", failed_pairs)
                    continue
                # A successful explicit retry replaces (rather than merely
                # coexisting with) the prior failure record. Keep that record
                # until success so a process killed mid-retry remains safely
                # resumable as a known failed pair.
                failed_pairs = [
                    r for r in failed_pairs
                    if (r["video"], r["config"]) != (video_key, config_name)
                ]
                _write_csv(output_root / "failed_pairs.csv", failed_pairs)
                quantization_diagnostic_rows.extend(pair_quantization_diagnostics)
                video_elapsed = time.time() - start
                records = sorted(result["records"], key=lambda r: r.index)

                # ── save every reconstructed frame + assemble recon.mp4 ──
                from sgdjscc_lab.io import save_tensor_as_image
                from sgdjscc_lab.utils.video_io import write_video

                recon_dir = output_root / "recon_videos" / video_key / config_name
                frame_files = []
                video_psnr, video_ssim, video_lpips = [], [], []
                for rec in records:
                    if rec.recon is None:
                        continue
                    if torch.isnan(rec.recon).any() or torch.isinf(rec.recon).any():
                        # Must be unreachable: every reconstruct_fn call is
                        # guarded by assert_finite at every internal stage
                        # (see pipelines/infer_pipeline.py), and any failure
                        # there raises NonFiniteError, caught above, which
                        # aborts this pair before reaching here. If this ever
                        # fires, it is a real gap in stage coverage — treat it
                        # as the unexpected bug it is, not a data point to
                        # quietly exclude.
                        raise RuntimeError(
                            f"[{video_key}][{config_name}] frame {rec.index}: non-finite reconstruction "
                            "was not caught by any assert_finite stage -- this is a coverage gap in "
                            "pipelines/infer_pipeline.py, not an expected data point; add the missing "
                            "assert_finite call rather than suppressing this"
                        )
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
                if frame_files:
                    fps = args.fps or info.get("fps") or 24.0
                    write_video(frame_files, recon_dir / "recon.mp4", fps=fps)

                # ── shadow accounting pass: only frames that transmitted a new latent ──
                video_bytes = 0
                video_symbols_analog = 0
                video_symbols_latent = 0
                transmitting = [r for r in records if r.decision in TRANSMITTING_DECISIONS]
                if args.match_actual_transmissions:
                    actual_indices = [record.index for record in transmitting]
                    expected_indices = json.loads(
                        matched_plan_row[f"{sel_name}_transmitting_indices"]
                    )
                    if actual_indices != expected_indices:
                        raise RuntimeError(
                            f"[{video_key}][{config_name}] planned vs actual visual-transmission "
                            f"schedule mismatch: planned={expected_indices}, actual={actual_indices}"
                        )
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
                if sel_name == "fixed" and args.match_fixed_keyframes:
                    if fixed_count_target is not None:
                        fixed_selector_kind = "fixed_count"
                        keyframe_count_matched = (n_kf_in_gop == fixed_count_target)
                    else:
                        fixed_selector_kind = "fixed_max_gop"
                        keyframe_count_matched = False  # invalid SKEM count forced fixed-max-GOP fallback
                elif sel_name == "fixed":
                    fixed_selector_kind = "fixed_max_gop"
                    keyframe_count_matched = ""  # matching not requested -- not applicable
                else:
                    fixed_selector_kind = ""
                    keyframe_count_matched = ""
                per_video_rows.append({
                    "video": video_key, "config": config_name, "selector": sel_name,
                    "channel": ch_name, "bit_depth": bit_depth if bit_depth is not None else "",
                    "psss_backend_kind": sel.psss_backend_kind,
                    "digital_step_policy": (args.digital_step_policy if channel_kind == "digital_packet" else ""),
                    "fixed_reference_snr_db": (
                        args.fixed_reference_snr_db
                        if channel_kind == "digital_packet" and args.digital_step_policy == "fixed_reference"
                        else ""
                    ),
                    "ablation_label": (
                        args.ablation_label if channel_kind == "digital_packet" else ""
                    ),
                    "n_frames_total": len(frames), "n_transmitting_frames": len(transmitting),
                    "n_keyframes_selected": n_kf_in_gop, "n_nan_or_inf_frames": 0,
                    "fixed_selector_kind": fixed_selector_kind,
                    "fixed_count_target": (fixed_count_target if fixed_count_target is not None else ""),
                    "fixed_max_gop_used": (args.fixed_max_gop if fixed_selector_kind == "fixed_max_gop" else ""),
                    "keyframe_count_matched": keyframe_count_matched,
                    "matched_rate_mode": (
                        "actual_transmissions" if args.match_actual_transmissions else ""
                    ),
                    "matched_rate_plan_exact": (
                        bool(matched_plan_row["transmitting_frame_count_exact"])
                        if args.match_actual_transmissions else ""
                    ),
                    "matched_rate_target_n_transmitting_frames": (
                        int(matched_plan_row["target_n_transmitting_frames"])
                        if args.match_actual_transmissions else ""
                    ),
                    "selected_psss_threshold": (
                        float(matched_plan_row["selected_psss_threshold"])
                        if args.match_actual_transmissions and sel_name == "skem" else ""
                    ),
                    "selected_psss_max_segment_length": (
                        int(matched_plan_row["selected_psss_max_segment_length"])
                        if args.match_actual_transmissions and sel_name == "skem" else ""
                    ),
                    "nonfinite_stages": "",
                    "n_quality_frames": len(video_psnr),
                    # Denominator is ALL frames, not just transmitting ones:
                    # TemporalPipeline sets rec.recon (and this loop scores it)
                    # for every decision (keyframe/reuse/recompute_*/generate)
                    # -- reused frames legitimately get PSNR/SSIM against the
                    # stale reconstruction they reused, which is itself a real
                    # quality signal (temporal-reuse cost), not something to
                    # exclude from the denominator.
                    "valid_frame_ratio": (len(video_psnr) / len(frames)) if frames else 1.0,
                    "mean_psnr": sum(video_psnr) / n if video_psnr else float("nan"),
                    "mean_ssim": sum(video_ssim) / n if video_ssim else float("nan"),
                    "mean_lpips": (sum(video_lpips) / len(video_lpips)) if video_lpips else "",
                    "latent_elements_total": video_symbols_latent,
                    "analog_channel_symbols_total": video_symbols_analog if channel_kind == "awgn" else "",
                    "source_packet_bits_total": video_bytes * 8,
                    "digital_side_information_bytes_total": video_bytes if channel_kind == "awgn" else "",
                    "total_bundle_bytes": video_bytes,
                    "total_bundle_bytes_per_frame": video_bytes / max(len(frames), 1),
                    "analog_no_wire_bytes": channel_kind == "awgn",
                    "visual_transport_complete": channel_kind != "awgn",
                    "total_elapsed_s": round(video_elapsed, 3),
                })
                log(f"[{video_key}][{config_name}] frames={len(frames)} transmitting={len(transmitting)} "
                    f"mean_psnr={per_video_rows[-1]['mean_psnr']:.4f} "
                    f"bytes/video={per_video_rows[-1]['total_bundle_bytes']} "
                    f"bytes/frame={per_video_rows[-1]['total_bundle_bytes_per_frame']:.1f}")

                # Persist after every (video, config) pair — not just at the end —
                # so a killed/crashed run resumes from the last completed pair
                # instead of losing all progress (see done_pairs / --resume above).
                _write_csv(output_root / "per_video_metrics.csv", per_video_rows)
                _write_csv(output_root / "keyframe_selection.csv", keyframe_rows)
                _write_csv(output_root / "packet_components.csv", packet_rows)
                _write_csv(
                    output_root / "quantization_diagnostics.csv",
                    quantization_diagnostic_rows,
                )

    _write_csv(output_root / "per_video_metrics.csv", per_video_rows)
    _write_csv(output_root / "keyframe_selection.csv", keyframe_rows)
    _write_csv(output_root / "packet_components.csv", packet_rows)
    _write_csv(output_root / "quantization_diagnostics.csv", quantization_diagnostic_rows)
    _write_csv(output_root / "keyframe_sweep.csv", keyframe_sweep_rows)
    _write_csv(output_root / "matched_rate_plan.csv", matched_rate_plan_rows)
    _write_csv(output_root / "failed_pairs.csv", failed_pairs)
    aggregate_rows = _aggregate(per_video_rows, expected_video_keys=expected_video_keys)
    _write_csv(output_root / "aggregate.csv", aggregate_rows)
    pareto_rows, baseline_info = _pareto_frontier(aggregate_rows)
    _write_csv(output_root / "pareto_frontier.csv", pareto_rows)

    if args.match_fixed_keyframes or args.match_actual_transmissions:
        rate_matching_rows = _compute_rate_matching(per_video_rows)
        _write_csv(output_root / "rate_matching.csv", rate_matching_rows)

    source_size_rows: List[Dict[str, Any]] = []
    if not args.skip_source_size_report:
        source_size_rows = _source_size_report(entries)
    _write_csv(output_root / "source_size_report.csv", source_size_rows)

    _write_readme_and_summary(output_root, args, per_video_rows, pareto_rows, baseline_info, failed_pairs)
    _run_effect_summarizer(output_root)
    _write_manifest(
        args, output_root, cfg, per_video_rows, failed_pairs,
        signature, phase="final",
    )
    if failed_pairs:
        print(
            f"completed_with_failures: {len(failed_pairs)} failed (video, config) pair(s) "
            "-- see failed_pairs.csv; use --retry-failed with the same output root to retry"
        )
        return 3
    return 0


_CHECKPOINT_NAMES = (
    "JSCC_model.pth", "diffusion_backbone.pth", "diffusion_controlnet.pth",
    "muge-epoch-19-checkpoint.pth",
)


def _checkpoint_hashes(model_root: Path) -> Dict[str, str]:
    """``{name: sha256}`` for every checkpoint file that exists under *model_root*."""
    return {
        name: rm.sha256_file(model_root / name)
        for name in _CHECKPOINT_NAMES if (model_root / name).exists()
    }


def _build_run_signature(args, cfg, entries, model_root: Path) -> Dict[str, Any]:
    """Everything that must be IDENTICAL for a ``--resume`` to safely continue
    a prior run in this same ``--output-root`` (task requirement): commit,
    dataset/config/checkpoint hashes, seed, video list + frame counts,
    granularity, PSSS settings, eval options. Written once at the start of a
    fresh run (``run_signature.json``) and re-derived + compared on every
    invocation targeting an existing output_root — see ``_check_resume_signature``.
    """
    from omegaconf import OmegaConf

    git_state = rm.get_git_state(_REPO_ROOT)
    dataset_manifest = Path(args.dataset_root) / "manifest.csv"
    dataset_hash = rm.sha256_file(dataset_manifest) if dataset_manifest.exists() else rm.UNKNOWN

    try:
        resolved_config = OmegaConf.to_container(cfg, resolve=True)
        config_hash = hashlib.sha256(
            json.dumps(resolved_config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
    except Exception:  # noqa: BLE001 — signature must still be buildable if config dump fails
        config_hash = rm.UNKNOWN

    video_frame_counts = {
        e["key"]: int(e["row"]["n_frames"]) if "n_frames" in e.get("row", {}) else None
        for e in entries
    }
    dataset_artifact_sha256: Dict[str, Dict[str, str]] = {}
    for entry in entries:
        item_hashes: Dict[str, str] = {}
        for field in ("processed", "captions", "gt"):
            value = entry.get(field)
            if value is not None and Path(value).is_file():
                item_hashes[field] = rm.sha256_file(value)
        dataset_artifact_sha256[entry["key"]] = item_hashes

    return {
        "git_commit": git_state["commit"],
        "git_dirty": git_state["dirty"],
        "git_branch": git_state["branch"],
        "dataset_root": str(args.dataset_root),
        "dataset_manifest_sha256": dataset_hash,
        "dataset_artifact_sha256": dataset_artifact_sha256,
        "resolved_config_sha256": config_hash,
        "checkpoint_sha256": _checkpoint_hashes(model_root),
        "seed": args.seed,
        "video_keys": sorted(e["key"] for e in entries),
        "video_frame_counts": video_frame_counts,
        "max_frames_cap": args.max_frames,
        "granularity": args.granularity,
        "psss": {
            "backend": args.psss_backend, "model_id": args.psss_model_id,
            "device": args.psss_device, "dtype": args.psss_dtype,
            "threshold": args.psss_threshold, "max_segment_length": args.psss_max_segment_length,
            "use_scene_detector": bool(args.use_scene_detector),
        },
        "eval_options": {
            "no_lpips": bool(args.no_lpips), "bits_per_symbol": args.bits_per_symbol,
            "code_rate": args.code_rate,
        },
        "configs": sorted(c for c in args.configs.split(",") if c),
        "device": args.device,
        "physical_cuda_device": os.environ.get("SGDJSCC_PHYSICAL_CUDA_DEVICE", args.device),
        "digital_step_policy": args.digital_step_policy,
        "fixed_reference_snr_db": args.fixed_reference_snr_db,
        "ablation_label": args.ablation_label,
        "match_fixed_keyframes": bool(args.match_fixed_keyframes),
        "match_actual_transmissions": bool(args.match_actual_transmissions),
        "matched_rate_byte_tolerance": (
            ACTUAL_TRANSMISSION_BYTE_TOLERANCE
            if args.match_actual_transmissions else RATE_MATCH_BYTE_TOLERANCE
        ),
        "match_actual_transmissions": bool(args.match_actual_transmissions),
        "matched_rate_thresholds": _parse_float_grid(
            args.matched_rate_thresholds, name="--matched-rate-thresholds",
        ),
        "matched_rate_max_segment_lengths": _parse_positive_int_grid(
            args.matched_rate_max_segment_lengths,
            name="--matched-rate-max-segment-lengths",
        ),
        "fixed_max_gop": args.fixed_max_gop,
    }


def _diff_signature(old: Dict[str, Any], new: Dict[str, Any]) -> str:
    """Human-readable per-key diff for a resume signature mismatch."""
    lines = []
    keys = sorted(set(old) | set(new))
    for key in keys:
        if old.get(key) != new.get(key):
            lines.append(f"  {key}:\n    was: {json.dumps(old.get(key), default=str)}\n    now: {json.dumps(new.get(key), default=str)}")
    return "\n".join(lines)


def _check_resume_signature(output_root: Path, signature: Dict[str, Any]) -> None:
    """Refuse to continue if *output_root* already has a run_signature.json
    that differs from *signature* — a resume must be the SAME run, not a
    different one silently writing into the same directory. Writes the
    signature (first run in this output_root) when none exists yet."""
    sig_path = output_root / "run_signature.json"
    if not sig_path.exists():
        sig_path.parent.mkdir(parents=True, exist_ok=True)
        sig_path.write_text(json.dumps(signature, indent=2, sort_keys=True), encoding="utf-8")
        return
    existing = json.loads(sig_path.read_text(encoding="utf-8"))
    if existing != signature:
        diff = _diff_signature(existing, signature)
        raise SystemExit(
            f"resume signature mismatch at {output_root} — refusing to continue with "
            f"different run conditions than the run already recorded there:\n{diff}\n"
            "Use a different --output-root for a genuinely different run, or match the "
            "original conditions exactly to resume."
        )


def _write_manifest(
    args, output_root: Path, cfg, per_video_rows: List[Dict[str, Any]],
    failed_pairs: List[Dict[str, Any]], signature: Dict[str, Any], phase: str,
) -> None:
    """Reproducibility manifest via utils/run_manifest.py (hard dependency —
    see the module-level ``from sgdjscc_lab.utils import run_manifest as rm``
    import; there is no soft-dependency fallback).

    *phase* is ``"initial"`` (written once at the very start of a run, before
    any video is processed — the intended run spec) or ``"final"`` (written
    once at the end, after every artifact CSV/JSON/README this run produces
    has been written — includes their sha256, see ``_hash_output_artifacts``).
    """
    from omegaconf import OmegaConf

    from sgdjscc_lab.paths import model_root as _model_root

    checkpoints = {
        name: (_model_root() / name) for name in _CHECKPOINT_NAMES if (_model_root() / name).exists()
    }

    try:
        resolved_config = OmegaConf.to_container(cfg, resolve=True)
    except Exception:  # noqa: BLE001 — manifest generation must never crash a completed sweep
        resolved_config = None

    failure_stages: Dict[str, int] = {}
    for row in failed_pairs:
        stage = str(row.get("failure_stage") or "unknown")
        failure_stages[stage] = failure_stages.get(stage, 0) + 1
    run_status = "completed_with_failures" if failed_pairs else (
        "completed" if phase == "final" else "running"
    )
    extra: Dict[str, Any] = {
        "configs_run": args.configs.split(","),
        "phase": phase,
        "run_status": run_status,
        "run_signature": signature,
        "logical_device": args.device,
        "physical_cuda_device": os.environ.get("SGDJSCC_PHYSICAL_CUDA_DEVICE", args.device),
    }
    if phase == "final":
        extra["output_artifact_sha256"] = _hash_output_artifacts(output_root)

    cuda_device_index = 0
    if str(args.device).startswith("cuda:"):
        cuda_device_index = int(str(args.device).split(":", 1)[1])

    manifest = rm.build_run_manifest(
        run_id=output_root.name,
        command_argv=sys.argv,
        command_source="captured",
        seed=int(args.seed),
        resolved_config=resolved_config,
        dataset_ref=str(args.dataset_root),
        dataset_hash=signature.get("dataset_manifest_sha256", rm.UNKNOWN),
        checkpoints=checkpoints,
        exact_fields=[
            "latent_elements", "source_packet_bits", "header_bytes", "shape_bytes",
            "scale_zp_bytes", "metadata_bytes", "payload_bytes", "checksum_bytes",
            "total_bundle_bytes", "quant_mse", "quant_signal_power", "quant_snr_db",
        ],
        proxy_fields=[
            "estimated_digital_channel_symbols", "estimated_wire_bytes",
            "bitdepth_proxy (pipelines/infer_pipeline.py::_digital_quant_snr_db)",
        ],
        nan_or_failure_counts={
            "total_nan_or_inf_frames": (
                sum(int(r.get("n_nan_or_inf_frames", 0)) for r in per_video_rows)
                + len(failed_pairs)
            ),
            "n_failed_pairs": len(failed_pairs),
            "failed_pair_nan_values": sum(int(float(r.get("n_nan") or 0)) for r in failed_pairs),
            "failed_pair_inf_values": sum(int(float(r.get("n_inf") or 0)) for r in failed_pairs),
            "failure_stages": failure_stages,
        },
        repo_root=_REPO_ROOT,
        cuda_device_index=cuda_device_index,
        extra=extra,
    )
    rm.write_run_manifest(output_root / f"run_manifest_{phase}.json", manifest)
    if phase == "final":
        # Keep the stable, documented name (results_registry.md, README links)
        # pointing at the final manifest.
        rm.write_run_manifest(output_root / "run_manifest.json", manifest)


_ARTIFACT_HASH_FILES = (
    "aggregate.csv", "per_video_metrics.csv", "pareto_frontier.csv",
    "keyframe_selection.csv", "packet_components.csv", "keyframe_sweep.csv",
    "rate_matching.csv", "matched_rate_plan.csv", "matched_rate_quality_effect.csv",
    "matched_rate_validation.json",
    "MATCHED_RATE_REPORT.md", "failed_pairs.csv", "source_size_report.csv",
    "quantization_diagnostics.csv", "quantization_effect.csv",
    "selector_effect.csv", "quantization_effect_ablation.csv",
    "selector_effect_ablation.csv", "normalization_effect_summary.json",
    "summary.json", "README.md",
)


def _hash_output_artifacts(output_root: Path) -> Dict[str, str]:
    """sha256 of every core output artifact that exists in *output_root*
    (task requirement: "최종 CSV·JSON·README 등 핵심 artifact의 SHA-256도 기록").
    A file this run did not produce (e.g. rate_matching.csv when
    --match-fixed-keyframes was not used) is simply absent from the dict —
    never a fabricated hash."""
    return {
        name: rm.sha256_file(output_root / name)
        for name in _ARTIFACT_HASH_FILES if (output_root / name).exists()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Resume support — reload a prior (possibly interrupted) run's CSVs so
# already-completed (video, config) pairs are skipped instead of redone.
# ─────────────────────────────────────────────────────────────────────────────

_PER_VIDEO_INT_FIELDS = (
    "n_frames_total", "n_transmitting_frames", "n_keyframes_selected",
    "n_nan_or_inf_frames", "n_quality_frames", "latent_elements_total",
    "source_packet_bits_total", "total_bundle_bytes",
    "matched_rate_target_n_transmitting_frames", "selected_psss_max_segment_length",
)
_PER_VIDEO_FLOAT_FIELDS = (
    "valid_frame_ratio", "mean_psnr", "mean_ssim", "total_elapsed_s",
    "total_bundle_bytes_per_frame",
)
_PER_VIDEO_OPTIONAL_FLOAT_FIELDS = (
    "mean_lpips", "analog_channel_symbols_total", "digital_side_information_bytes_total",
    "fixed_reference_snr_db", "selected_psss_threshold",
)
_PER_VIDEO_BOOL_FIELDS = ("analog_no_wire_bytes", "visual_transport_complete")
_PER_VIDEO_OPTIONAL_BOOL_FIELDS = (
    "keyframe_count_matched", "matched_rate_plan_exact",
)  # "" means "not applicable", never coerced to False


def _read_csv_dicts(path: Path) -> List[Dict[str, Any]]:
    """Load a CSV previously written by :func:`_write_csv`, or ``[]`` if absent/empty.

    Used only for resume: reloads a prior (possibly interrupted) run's output
    so the main sweep can skip (video, config) pairs already recorded there.
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _coerce_per_video_row(row: Dict[str, str]) -> Dict[str, Any]:
    """Cast a CSV-reloaded per_video_metrics.csv row back to the types the
    fresh in-memory rows use, so `_aggregate`/`_pareto_frontier` (which do
    arithmetic on these fields) treat resumed and freshly-computed rows
    identically."""
    out = dict(row)
    for key in _PER_VIDEO_INT_FIELDS:
        if key in out and out[key] != "":
            out[key] = int(float(out[key]))
    for key in _PER_VIDEO_FLOAT_FIELDS:
        if key in out and out[key] != "":
            out[key] = float(out[key])
    for key in _PER_VIDEO_OPTIONAL_FLOAT_FIELDS:
        if key in out and out[key] not in ("", None):
            out[key] = float(out[key])
    for key in _PER_VIDEO_BOOL_FIELDS:
        if key in out:
            out[key] = str(out[key]).strip().lower() == "true"
    for key in _PER_VIDEO_OPTIONAL_BOOL_FIELDS:
        if key in out and out[key] != "":
            out[key] = str(out[key]).strip().lower() == "true"
    if out.get("bit_depth", "") != "":
        out["bit_depth"] = int(out["bit_depth"])
    if out.get("psss_backend_kind", "") == "":
        out["psss_backend_kind"] = None
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Atomically (over)write *path* — write to a sibling temp file, then
    ``os.replace`` it into place. ``os.replace`` is atomic on POSIX/Windows,
    so a reader (or a resumed run's ``_read_csv_dicts``) never observes a
    half-written file, and a process killed mid-write leaves the PREVIOUS
    complete version at *path* rather than a truncated/corrupt one."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            if rows:
                fieldnames = list(rows[0].keys())
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _aggregate(
    per_video_rows: List[Dict[str, Any]],
    expected_video_keys: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """One row per config, averaged across the videos that completed it.

    *expected_video_keys* (when given) is the full set of video keys this run
    intended to cover (from the dataset manifest / ``--video-ids``) — used to
    set ``all_expected_videos_present`` per config (task requirement: a
    config missing some expected video's pair — still running, or a config
    every video hasn't reached yet — is not a valid baseline/Pareto/effect
    candidate; see ``_pareto_frontier``). A ``(video, config)`` pair that
    raised ``NonFiniteError`` never produced a ``per_video_rows`` entry (see
    ``run()``'s per-config loop) — every row here IS therefore already
    "successful" in the sense that zero non-finite frames occurred within it.
    """
    by_config: Dict[str, List[Dict[str, Any]]] = {}
    for r in per_video_rows:
        by_config.setdefault(r["config"], []).append(r)
    out = []
    for config, rows in by_config.items():
        n = len(rows)
        video_keys = {r["video"] for r in rows}
        byte_rows = [r["total_bundle_bytes"] for r in rows if r["total_bundle_bytes"] != ""]
        bytes_per_frame_rows = [
            r["total_bundle_bytes_per_frame"] for r in rows if r.get("total_bundle_bytes_per_frame", "") != ""
        ]
        lpips_rows = [r["mean_lpips"] for r in rows if r["mean_lpips"] != ""]
        # "" (not applicable -- matching wasn't requested/relevant for this
        # row, e.g. skem_* rows or a run without --match-fixed-keyframes)
        # must never silently count as "not matched" -- only rows with an
        # applicable True/False value are considered.
        kfcm_applicable = [r["keyframe_count_matched"] for r in rows if r.get("keyframe_count_matched", "") != ""]
        n_quality = sum(r.get("n_quality_frames", 0) for r in rows)
        n_transmitting = sum(r.get("n_transmitting_frames", 0) for r in rows)
        all_finite_metrics = all(
            math.isfinite(r["mean_psnr"]) and math.isfinite(r["mean_ssim"])
            and (r["mean_lpips"] == "" or math.isfinite(r["mean_lpips"]))
            for r in rows
        )
        out.append({
            "config": config,
            "selector": rows[0]["selector"],
            "channel": rows[0]["channel"],
            "bit_depth": rows[0]["bit_depth"],
            "psss_backend_kind": rows[0]["psss_backend_kind"],
            "digital_step_policy": rows[0].get("digital_step_policy", ""),
            "fixed_reference_snr_db": rows[0].get("fixed_reference_snr_db", ""),
            "ablation_label": rows[0].get("ablation_label", ""),
            "n_videos": n,
            "video_keys": ",".join(sorted(video_keys)),
            "n_videos_expected": (len(expected_video_keys) if expected_video_keys is not None else n),
            "all_expected_videos_present": (
                (expected_video_keys is None) or (video_keys == expected_video_keys)
            ),
            "total_frames": sum(r.get("n_frames_total", 1) for r in rows),
            "total_quality_frames": n_quality,
            "total_transmitting_frames": n_transmitting,
            "valid_frame_ratio": (n_quality / max(sum(r.get("n_frames_total", 1) for r in rows), 1)),
            "all_finite_metrics": all_finite_metrics,
            "mean_psnr": sum(r["mean_psnr"] for r in rows) / n,
            "mean_ssim": sum(r["mean_ssim"] for r in rows) / n,
            "mean_lpips": (sum(lpips_rows) / len(lpips_rows)) if lpips_rows else "",
            "mean_latent_elements": sum(r["latent_elements_total"] for r in rows) / n,
            "mean_total_bundle_bytes_per_video": (sum(byte_rows) / len(byte_rows)) if byte_rows else "",
            "mean_total_bundle_bytes_per_frame": (
                (sum(bytes_per_frame_rows) / len(bytes_per_frame_rows)) if bytes_per_frame_rows else ""
            ),
            "mean_n_keyframes_selected": sum(r.get("n_keyframes_selected", 0) for r in rows) / n,
            "keyframe_count_matched": (all(bool(v) for v in kfcm_applicable) if kfcm_applicable else ""),
            "analog_no_wire_bytes": rows[0]["analog_no_wire_bytes"],
            "visual_transport_complete": rows[0].get("visual_transport_complete", False),
            "total_nan_or_inf_frames": sum(r.get("n_nan_or_inf_frames", 0) for r in rows),
            "nonfinite_stages": ",".join(sorted({
                s for r in rows for s in str(r.get("nonfinite_stages", "")).split(",") if s
            })),
        })
    return out


def _pareto_frontier(aggregate_rows: List[Dict[str, Any]]):
    """Validity gate (task requirement, ALL must hold for baseline OR candidate):
    zero non-finite frames, PSNR/SSIM/LPIPS all finite, valid_frame_ratio == 1
    (all reconstructed video frames — see _aggregate), every expected video present.
    A candidate additionally needs its video_keys to equal the BASELINE's
    video_keys (never compared across mismatched video sets)."""
    by_config = {r["config"]: r for r in aggregate_rows}

    def _row_is_valid(r: Dict[str, Any]) -> bool:
        return (
            int(r.get("total_nan_or_inf_frames", 0)) == 0
            and bool(r.get("all_finite_metrics", False))
            and float(r.get("valid_frame_ratio", 0.0)) == 1.0
            and bool(r.get("all_expected_videos_present", False))
        )

    baseline = None
    baseline_config = None
    for candidate in BASELINE_PREFERENCE:
        if candidate in by_config and _row_is_valid(by_config[candidate]):
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

    baseline_video_keys = set(str(baseline.get("video_keys", "")).split(",")) if baseline.get("video_keys") else set()

    candidates = [
        r for r in aggregate_rows
        if r["config"] != baseline_config
        and r["mean_total_bundle_bytes_per_video"] != ""
        and r.get("channel") != "awgn"
        and not bool(r.get("analog_no_wire_bytes", False))
        and bool(r.get("visual_transport_complete", False))
    ]
    in_budget = []
    for r in candidates:
        psnr_drop = baseline["mean_psnr"] - r["mean_psnr"]
        ssim_drop = baseline["mean_ssim"] - r["mean_ssim"]
        lpips_rise = (
            (r["mean_lpips"] - baseline["mean_lpips"])
            if (r["mean_lpips"] != "" and baseline["mean_lpips"] != "") else None
        )
        r_video_keys = set(str(r.get("video_keys", "")).split(",")) if r.get("video_keys") else set()
        same_video_set = r_video_keys == baseline_video_keys
        row_valid = _row_is_valid(r)
        ok = (
            row_valid
            and same_video_set
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
            "non_finite_frames" if int(r.get("total_nan_or_inf_frames", 0)) else
            "non_finite_metrics" if not r.get("all_finite_metrics", False) else
            "incomplete_quality_coverage" if float(r.get("valid_frame_ratio", 0.0)) != 1.0 else
            "missing_expected_video" if not r.get("all_expected_videos_present", False) else
            "video_set_mismatch_vs_baseline" if not same_video_set else
            "quality_threshold" if not ok else ""
        )
        in_budget.append(row)

    selected = [r for r in in_budget if r["within_quality_gate"]]
    pool = selected if selected else in_budget  # spec: if none qualify, report nearest, don't hide it
    pool_sorted = sorted(pool, key=lambda r: r["mean_total_bundle_bytes_per_video"])
    for i, r in enumerate(pool_sorted):
        r["rank"] = i
        r["selected_as_smallest_in_budget"] = bool(selected) and i == 0
    return pool_sorted, baseline_info


# Relative byte difference within which a fixed/SKEM pair at the same
# keyframe count is actually called "rate-matched" (task requirement: never
# use that label just because keyframe counts matched -- bytes must also be
# close in practice).
RATE_MATCH_BYTE_TOLERANCE = 0.10
ACTUAL_TRANSMISSION_BYTE_TOLERANCE = 0.01


def _compute_rate_matching(per_video_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare fixed vs SKEM rates using the requested matching contract.

    Legacy ``--match-fixed-keyframes`` retains its 10% raw-byte rule.  The
    exact mode requires equal *actual visual-transmission* counts and at most
    1% raw bundle-byte mismatch.  It then explicitly accounts padding on the
    smaller side, so the effective bytes used for the paired quality
    comparison are exactly equal rather than merely described as close.
    """
    by_video_channel: Dict[tuple, Dict[str, Dict[str, Any]]] = {}
    for r in per_video_rows:
        if r["channel"] == "awgn":
            continue
        by_video_channel.setdefault((r["video"], r["channel"]), {})[r["selector"]] = r

    rows = []
    for (video, channel), by_selector in sorted(by_video_channel.items()):
        fixed_row = by_selector.get("fixed")
        skem_row = by_selector.get("skem")
        if fixed_row is None or skem_row is None:
            continue
        fixed_bytes = fixed_row["total_bundle_bytes"]
        skem_bytes = skem_row["total_bundle_bytes"]
        larger = max(fixed_bytes, skem_bytes)
        byte_diff_ratio = (abs(fixed_bytes - skem_bytes) / larger) if larger > 0 else 0.0
        keyframe_count_matched = (
            int(fixed_row["n_keyframes_selected"])
            == int(skem_row["n_keyframes_selected"])
        )
        transmission_count_matched = (
            int(fixed_row["n_transmitting_frames"])
            == int(skem_row["n_transmitting_frames"])
        )
        mode = str(
            fixed_row.get("matched_rate_mode")
            or skem_row.get("matched_rate_mode")
            or "legacy_keyframes"
        )
        if mode == "actual_transmissions":
            tolerance = ACTUAL_TRANSMISSION_BYTE_TOLERANCE
            raw_rate_matched = transmission_count_matched and byte_diff_ratio <= tolerance
        else:
            tolerance = RATE_MATCH_BYTE_TOLERANCE
            raw_rate_matched = keyframe_count_matched and byte_diff_ratio <= tolerance
        fixed_padding = (larger - fixed_bytes) if raw_rate_matched else 0
        skem_padding = (larger - skem_bytes) if raw_rate_matched else 0
        rows.append({
            "video": video, "channel": channel,
            "matched_rate_mode": mode,
            "fixed_n_keyframes": fixed_row["n_keyframes_selected"],
            "skem_n_keyframes": skem_row["n_keyframes_selected"],
            "keyframe_count_matched": keyframe_count_matched,
            "fixed_n_transmitting_frames": fixed_row["n_transmitting_frames"],
            "skem_n_transmitting_frames": skem_row["n_transmitting_frames"],
            "transmitting_frame_count_matched": transmission_count_matched,
            "fixed_total_bundle_bytes": fixed_bytes,
            "skem_total_bundle_bytes": skem_bytes,
            "fixed_total_bundle_bytes_per_frame": fixed_row["total_bundle_bytes_per_frame"],
            "skem_total_bundle_bytes_per_frame": skem_row["total_bundle_bytes_per_frame"],
            "byte_diff_ratio": byte_diff_ratio,
            "byte_diff_ratio_tolerance": tolerance,
            "raw_rate_matched": raw_rate_matched,
            "fixed_padding_bytes": fixed_padding,
            "skem_padding_bytes": skem_padding,
            "fixed_effective_total_bytes": fixed_bytes + fixed_padding,
            "skem_effective_total_bytes": skem_bytes + skem_padding,
            "effective_bytes_exact": (
                raw_rate_matched
                and fixed_bytes + fixed_padding == skem_bytes + skem_padding
            ),
            "rate_matched": raw_rate_matched,
        })
    return rows


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


def _write_readme_and_summary(output_root, args, per_video_rows, pareto_rows, baseline_info, failed_pairs):
    summary = {
        "output_root": str(output_root),
        "configs_run": args.configs.split(","),
        "n_videos": len({r["video"] for r in per_video_rows}),
        "n_failed_pairs": len(failed_pairs),
        "run_status": "completed_with_failures" if failed_pairs else "completed",
        "quality_gate": QUALITY_GATE,
        "pareto_baseline": baseline_info,
        "pareto_selected": next((r for r in pareto_rows if r.get("selected_as_smallest_in_budget")), None),
        "psss_backend_requested": args.psss_backend,
        "psss_model_id": args.psss_model_id,
        "use_scene_detector": bool(args.use_scene_detector),
        "bits_per_symbol": args.bits_per_symbol,
        "code_rate": args.code_rate,
        "digital_step_policy": args.digital_step_policy,
        "fixed_reference_snr_db": args.fixed_reference_snr_db,
        "ablation_label": args.ablation_label,
        "match_fixed_keyframes": bool(args.match_fixed_keyframes),
        "seed": args.seed,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    skem_backend_kinds = sorted({
        r["psss_backend_kind"] for r in per_video_rows
        if r["selector"] == "skem" and r.get("psss_backend_kind")
    })
    ablation_note = (
        f"- **ablation 실행**: `--digital-step-policy {args.digital_step_policy}` "
        f"(label: `{args.ablation_label}`) — `quantization_effect_ablation.csv`로 분리 기록"
        if args.digital_step_policy != "fixed_reference" else
        f"- 양자화 비교 policy: `fixed_reference` ({args.fixed_reference_snr_db:g}dB; "
        "모든 bit_depth를 동일 step으로 디코딩 — decoder step 변화가 섞이지 않은 순수 양자화 효과)"
    )
    baseline_note = (
        "- Pareto baseline: **unavailable** — `fixed_float32`/`fixed_int16`/`skem_float32`/"
        "`skem_int16` 중 non-finite 0건·기대 영상 전부 완료인 config가 없음 "
        "(AWGN은 baseline 후보에서 완전히 제외, fallback 없음)"
        if not baseline_info["baseline_valid"] else
        f"- Pareto baseline: `{baseline_info['baseline_config']}` "
        "(reliable digital — float32 무손실 또는 int16 near-lossless, non-finite 0건, "
        "기대 영상 전부 완료 — AWGN 아님)"
    )

    readme = f"""# transmission_reduction run — {output_root.name}

- 개요
  - 실제 packet bundle(visual latent + caption + edge/uncertainty + manifest)
    전송량 회계, `TemporalPipeline` 전체 영상 품질, SKEM/PSSS keyframe 선택 비교
  - exact-vs-estimate 회계 경계와 digital receiver bytes-only 경계는
    `scripts/run_transmission_reduction_eval.py` 모듈 docstring 참고
- run 상태
  - 완료 (video, config) 쌍: {len(per_video_rows)}개
  - 실패 (video, config) 쌍: {len(failed_pairs)}개 (`failed_pairs.csv`) — 첫 non-finite 발생 즉시
    해당 pair 중단, 다음 pair로 진행. 재개 시 기본 skip, `--retry-failed`일 때만 재시도
  - digital step 정책: `{args.digital_step_policy}`
  - fixed-reference SNR: `{args.fixed_reference_snr_db:g}dB`
{ablation_note}
  - rate 정합: `{"ON — fixed max-GOP은 유지하고 SKEM을 영상별 보정해 실제 visual 전송 프레임 수를 정확히 일치; raw bytes 1% gate + 작은 쪽 padding 계상" if args.match_actual_transmissions else ("ON — FixedCountKeyframeSelector로 fixed를 SKEM과 정확히 동일 개수로 강제 (legacy)" if args.match_fixed_keyframes else "OFF")}`
{baseline_note}
- 산출물
  - `per_video_metrics.csv` / `aggregate.csv` — 영상 전체 품질(PSNR/SSIM/LPIPS) +
    정확한 전송 bytes. **`total_bundle_bytes`는 bytes/video**, **`total_bundle_bytes_per_frame`은
    bytes/frame**(전체 프레임 기준) — 단위 혼동 금지
  - `failed_pairs.csv` — 중단된 (video, config): 실패 stage·frame·NaN/Inf 수
  - `matched_rate_plan.csv` (`--match-actual-transmissions`) — fixed max-GOP schedule,
    영상별 선택된 SKEM threshold/max-segment, 계획 visual 전송 index와 exact-count 검증
  - `rate_matching.csv` (rate matching 사용 시) — 영상×channel별 fixed vs SKEM의 실제
    전송 프레임 수·raw bytes·byte 차이. actual-transmission mode는 raw 차이
    **{int(ACTUAL_TRANSMISSION_BYTE_TOLERANCE * 100)}% 이내**를 요구하고 작은 쪽 padding을 계상한
    effective bytes가 정확히 같을 때만 검증 통과. legacy keyframe mode는 {int(RATE_MATCH_BYTE_TOLERANCE * 100)}% 기준
  - `keyframe_selection.csv` — 프레임별 decision, 구조화된 `force_reason`,
    5필드 회계 스키마, `psss_backend_kind`(`mock`|`proxy`|`real`)
  - `packet_components.csv` — 실제 `.sgbundle` byte의 정확한 구성 breakdown
  - `quantization_diagnostics.csv` — packet별 실측 NMSE/SNR 분석값. `quant_nmse`에서만
    receiver 입력 metadata로 전송되며, 다른 정책에서는 bundle bytes에 포함되지 않음
  - `packets/<video>/<config>/frame_NNNNN.sgbundle` — 실제 직렬화 전송 bundle
  - `recon_videos/<video>/<config>/recon.mp4` + `frame_*.png` — 전체 복원 영상
  - `keyframe_sweep.csv` — PSSS threshold x max_segment_length grid (선택만, 복원 없음)
  - `pareto_frontier.csv` — quality gate(PSNR 저하 <= {QUALITY_GATE['psnr_drop_db']}dB,
    SSIM 저하 <= {QUALITY_GATE['ssim_drop']}, LPIPS 상승 <= {QUALITY_GATE['lpips_rise']}) 통과 config 중
    bytes/video 최소. 유효 조건: non-finite 0·PSNR/SSIM/LPIPS 전부 finite·
    valid_frame_ratio==1·기대 영상 전부 완료·baseline과 영상 집합 동일. 미달이어도
    가장 가까운 후보를 숨기지 않고 나열. AWGN 행은 visual waveform bytes가 없으므로
    Pareto 후보에서 제외하고 참고 기준으로만 유지
  - `run_manifest_initial.json` / `run_manifest_final.json` (= `run_manifest.json`) /
    `run_signature.json` — 재현성(commit·dataset/config/checkpoint hash·seed·환경) 및
    resume 안전성 서명. `run_manifest.json`에 핵심 artifact SHA-256 포함
  - `summary.json` — run 설정 + 선택된 config + baseline

한계:
- `--psss-backend {args.psss_backend}` 사용 — `real`(+`--psss-model-id`)만 진짜 PSSS이고
  `mock`/`proxy`는 진짜 PSSS가 아님(`video/psss.py` 참고). 이번 run의 SKEM 행에 실제 관측된
  backend: `{', '.join(skem_backend_kinds) if skem_backend_kinds else '(skem 행 없음)'}`
  — `psss_backend_kind` 컬럼으로 항상 구분되며, `real`이 아니면 결과를 "real SKEM"으로
  표기하지 않음
- `estimated_digital_channel_symbols`/`estimated_wire_bytes`는 labeled proxy
  (`{'unavailable — --bits-per-symbol 미지정' if args.bits_per_symbol is None else f'bits_per_symbol={args.bits_per_symbol}'}`)
  — 실제 변조/FEC 코더 없음
- digital config는 이번 run이 저장한 실제 `.sgbundle` byte로부터 복원. AWGN은 아날로그
  파형이라 byte bundle로 복원 불가 — `analog_channel_symbols_total`과 digital
  caption/edge/manifest byte를 별도 도메인으로 기록
- `bitdepth_proxy` 정책은 bit_depth만으로 결정되는 휴리스틱이며 실측 채널 SNR이 아님
  (`pipelines/infer_pipeline.py::_digital_quant_snr_db`). `quant_nmse` 정책만 송신단이
  실측한 quantization SNR(패킷 metadata로 전송, receiver가 패킷 자체에서 읽음 — 전역
  channel 객체 아님)을 사용
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(run())

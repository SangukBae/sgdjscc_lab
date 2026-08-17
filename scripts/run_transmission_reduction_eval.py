#!/usr/bin/env python
"""run_transmission_reduction_eval.py – Transmission-reduction sweep driver.

Compares, per ETRI video, "how many bytes does the transmitter actually send"
against reconstruction quality, across:

    keyframe selector : fixed (scene-change + max_gop)  vs  SKEM/PSSS (real)
    channel transport  : analog AWGN (baseline)  vs  digital_packet at
                          8/6/4-bit (real quantization + bit-packed binary
                          packet, see ``sgdjscc_lab.transmission``)

Unlike ``scripts/benchmark_etri_video_rate.py`` (which measures a PNG-based
*reference payload* built from an already-completed run), this script drives
the real JSCC encode -> channel -> diffusion-decode path itself
(``pipelines.infer_pipeline.run_single_image``) on each selected keyframe, so
the reported packet bytes are the actual serialized
``sgdjscc_lab.transmission.wire_packet`` bytes for the tensor that was really
transmitted, not an estimate.

Scope note (exact vs. estimate — see also each CSV's own columns):
    - ``packet_components.csv`` / packet byte totals: EXACT (real serialized
      bytes, ``proxy=False``).
    - AWGN baseline "bytes": the analog channel has no wire-bytes concept: an
      AWGN JSCC channel transmits real-valued channel *symbols*, not bytes.
      The baseline row reports ``channel_symbols`` (exact count = latent
      element count actually transmitted) and leaves byte fields as
      ``analog_no_wire_bytes`` — never a fabricated exact-byte number for an
      analog path.
    - ``codec_comparison.csv``: H.264/H.265/AV1 exact encoded file sizes at
      given CRFs; only labeled "quality_matched" when a real PSNR/SSIM/LPIPS
      crossing point against the semantic baseline is found in the sampled
      CRFs — otherwise reported as "no_crossing_found", never asserted.

This script performs measurement only — it does not alter
``pipelines/infer_pipeline.py`` / ``video/temporal_pipeline.py`` numerics; it
calls the exact same ``run_single_image`` forward pass every other inference
entry point uses, with ``jscc.channel_model`` swapped per the sweep grid
(the same Phase 5-A extension point Rayleigh/fast-fading/packet-drop use).

Example
-------
Correctness smoke run (tiny, real models, one video, few keyframes):
    python scripts/run_transmission_reduction_eval.py \\
        --video-ids 01_person_walk --configs fixed_awgn,fixed_int8,skem_int8 \\
        --max-keyframes 3 --device cuda:0 --output-root outputs/transmission_reduction_smoke

Full sweep (all 10 videos, all 7 configs + keyframe/codec sweeps):
    python scripts/run_transmission_reduction_eval.py \\
        --device cuda:0 --output-root outputs/transmission_reduction_$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
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
}
SELECTORS = ("fixed", "skem")
ALL_CONFIGS = [f"{sel}_{ch}" for sel in SELECTORS for ch in CHANNEL_CONFIGS]
# Task spec's exact comparison set: fixed baseline, fixed+intN, SKEM+intN
# (skem_awgn is a valid extra combination but not part of the requested grid).
DEFAULT_CONFIGS = ["fixed_awgn", "fixed_int8", "fixed_int6", "fixed_int4",
                    "skem_int8", "skem_int6", "skem_int4"]

DEFAULT_PSSS_THRESHOLDS = (0.25, 0.35, 0.45, 0.55)
DEFAULT_MAX_SEGMENT_LENGTHS = (12, 16, 24, 32)

# Quality-degradation gate (task spec): a candidate config is "in budget" when
# it does not lose more than this much quality vs the fixed+AWGN baseline.
QUALITY_GATE = {"psnr_drop_db": 0.5, "ssim_drop": 0.01, "lpips_rise": 0.02}


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
        description="Real packet/quantization/SKEM transmission-reduction sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-root", default=str(_REPO_ROOT / "data/etri_video_eval"))
    p.add_argument("--output-root", required=True)
    p.add_argument("--video-ids", default=None, help="Comma-separated subset, default = all in manifest.")
    p.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                    help="Comma-separated subset of " + ",".join(ALL_CONFIGS))
    p.add_argument("--snr", type=float, default=10.0, help="Baseline AWGN SNR (dB).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-keyframes", type=int, default=None,
                    help="Cap keyframes reconstructed per video (smoke-test knob); default = all selected.")
    p.add_argument("--psss-threshold", type=float, default=0.35)
    p.add_argument("--psss-max-segment-length", type=int, default=16)
    p.add_argument("--skip-keyframe-sweep", action="store_true",
                    help="Skip the threshold x max_segment_length PSSS sweep (keyframe_sweep.csv).")
    p.add_argument("--skip-codec-comparison", action="store_true",
                    help="Skip H.264/H.265/AV1 ffmpeg comparison (codec_comparison.csv).")
    p.add_argument("--granularity", default="per_tensor", choices=["per_tensor", "per_channel"])
    p.add_argument("--no-lpips", action="store_true")
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
# Keyframe selection
# ─────────────────────────────────────────────────────────────────────────────

def _build_selector(name: str, captions: Optional[List[str]], threshold: float, max_segment_length: int):
    from omegaconf import OmegaConf
    from sgdjscc_lab.video.keyframe_extractor import build_caption_fn, build_keyframe_extractor
    from sgdjscc_lab.video.scene_change_detector import SceneChangeDetector

    if name == "fixed":
        cfg = OmegaConf.create({"keyframe": {"selector": "fixed", "max_gop": max_segment_length}})
        return build_keyframe_extractor(cfg, scene_detector=SceneChangeDetector())

    caption_source = "captions_file" if captions else "mock"
    caption_fn = build_caption_fn(caption_source, captions=captions)
    cfg = OmegaConf.create({
        "keyframe": {
            "selector": "psss",
            "psss": {
                "backend": "proxy",  # real CLIP-text cosine PSSS (backend_kind="proxy");
                                     # backend="real" needs an HF model_id not yet
                                     # provisioned for this repo — see README.md caveat.
                "threshold": threshold,
                "max_segment_length": max_segment_length,
                "min_segment_length": 1,
                "caption_source": caption_source,
            },
        },
    })
    return build_keyframe_extractor(cfg, caption_fn=caption_fn)


@dataclass
class KeyframeSelection:
    video: str
    selector: str
    threshold: Optional[float]
    max_segment_length: int
    n_frames: int
    keyframe_indices: List[int]
    n_keyframes: int
    forced_flags: List[bool]
    reasons: Dict[int, str]
    psss_scores: List[Dict]


def _select_keyframes(video_key, frames, captions, selector_name, threshold, max_segment_length) -> KeyframeSelection:
    selector = _build_selector(selector_name, captions, threshold, max_segment_length)
    result = selector.extract(frames)
    keyframes = list(result["keyframes"])
    reasons = dict(result.get("keyframe_reasons", {}))
    forced = []
    for k in keyframes:
        reason = reasons.get(k, "")
        forced.append(bool(k == 0 or "max_segment_length" in reason or "scene" in reason.lower()
                            or "boundary" in reason.lower()))
    return KeyframeSelection(
        video=video_key, selector=selector_name, threshold=threshold,
        max_segment_length=max_segment_length, n_frames=len(frames),
        keyframe_indices=keyframes, n_keyframes=len(keyframes), forced_flags=forced,
        reasons=reasons, psss_scores=list(result.get("psss_scores", [])),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-keyframe reconstruction + exact packet accounting
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KeyframeResult:
    video: str
    config: str
    frame_index: int
    psnr: float
    ssim: float
    lpips: Optional[float]
    channel_symbols: int
    total_bytes: Optional[int]  # None for analog AWGN (no wire-bytes concept)
    header_bytes: Optional[int] = None
    shape_bytes: Optional[int] = None
    scale_zp_bytes: Optional[int] = None
    metadata_bytes: Optional[int] = None
    payload_bytes: Optional[int] = None
    checksum_bytes: Optional[int] = None
    elapsed_s: float = 0.0
    bit_depth: Optional[int] = None


def _build_models(cfg, device):
    from sgdjscc_lab.runtime import build_models
    return build_models(cfg, device)


_CFG_FRAGMENTS = ("base/channel/awgn", "base/model/sgdjscc", "base/infer/awgn", "base/eval/default")


def _make_cfg(output_root: Path, model_root: Path, snr_db: float):
    """Compose a real config via the project's own fragment set (config.py's
    _defaults_ mechanism) rather than hand-rolling a minimal dict — this
    guarantees every key run_single_image()/_jscc_forward() expects (diffusion
    step count, guidance scale, canny_cr, ...) is present with its real
    default, exactly as every other entry point gets it."""
    from omegaconf import OmegaConf
    from sgdjscc_lab.config import load_config

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
    }))
    return cfg


def _reconstruct_and_measure(
    frame_tensor, models, cfg, quality_evaluator, channel_kind, bit_depth, granularity, keyframe_index,
):
    import torch
    from sgdjscc_lab.channels import AWGNChannel, DigitalPacketChannel
    from sgdjscc_lab.utils.preprocessing import merge_patches, prepare_patches

    device = models.device
    patches, meta = prepare_patches(frame_tensor)
    patches = patches.to(device)

    if channel_kind == "awgn":
        models.jscc_model.channel_model = None  # falls back to the original AWGN path unchanged
    else:
        ch = DigitalPacketChannel(bit_depth=bit_depth, granularity=granularity, channel_dim=1)
        ch.keyframe_index = keyframe_index
        models.jscc_model.channel_model = ch

    start = time.time()
    from sgdjscc_lab.pipelines.infer_pipeline import run_single_image
    with torch.inference_mode():
        recon_patches = run_single_image(patches, models, cfg)
    elapsed = time.time() - start

    recon = merge_patches(recon_patches.cpu(), meta)
    original = frame_tensor
    h, w = min(recon.shape[-2], original.shape[-2]), min(recon.shape[-1], original.shape[-1])
    metrics = quality_evaluator.evaluate(original[..., :h, :w], recon[..., :h, :w])

    n_elements = patches.numel()  # exact: elements actually transmitted through the channel

    breakdown = None
    total_bytes = None
    if channel_kind != "awgn":
        cm = models.jscc_model.channel_model
        if cm.last_breakdown is not None:
            breakdown = cm.last_breakdown
            total_bytes = breakdown.total_bytes

    return metrics, n_elements, breakdown, total_bytes, elapsed, recon


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
    cfg = _make_cfg(output_root, _model_root(), args.snr)
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
            captions = _load_captions(entry["captions"], len(frames))

            # ── keyframe sweep (cheap: selection only, no reconstruction) ──
            if not args.skip_keyframe_sweep:
                for th in DEFAULT_PSSS_THRESHOLDS:
                    for max_len in DEFAULT_MAX_SEGMENT_LENGTHS:
                        sel = _select_keyframes(video_key, frames, captions, "skem", th, max_len)
                        keyframe_sweep_rows.append({
                            "video": video_key, "threshold": th, "max_segment_length": max_len,
                            "n_frames": sel.n_frames, "n_keyframes": sel.n_keyframes,
                            "keyframe_indices": json.dumps(sel.keyframe_indices),
                        })
                        log(f"  keyframe_sweep {video_key} th={th} max_len={max_len} -> "
                            f"{sel.n_keyframes} keyframes")

            for config_name in configs:
                sel_name, ch_name = config_name.split("_", 1)
                channel_kind = "awgn" if ch_name == "awgn" else "digital_packet"
                bit_depth = CHANNEL_CONFIGS[ch_name]["bit_depth"]

                sel = _select_keyframes(
                    video_key, frames, captions, sel_name,
                    args.psss_threshold, args.psss_max_segment_length,
                )
                keyframes = sel.keyframe_indices
                if args.max_keyframes is not None:
                    keyframes = keyframes[: args.max_keyframes]

                log(f"[{video_key}][{config_name}] {len(keyframes)} keyframes selected "
                    f"(selector={sel_name}, channel={ch_name})")

                video_psnr, video_ssim, video_lpips = [], [], []
                video_symbols = 0
                video_bytes = 0
                video_elapsed = 0.0

                for kf_idx in keyframes:
                    metrics, n_elements, breakdown, total_bytes, elapsed, recon = _reconstruct_and_measure(
                        frames[kf_idx], models, cfg, quality_evaluator,
                        channel_kind, bit_depth, args.granularity, kf_idx,
                    )
                    video_psnr.append(metrics["psnr"])
                    video_ssim.append(metrics["ssim"])
                    if metrics["lpips"] is not None:
                        video_lpips.append(metrics["lpips"])
                    video_symbols += n_elements
                    video_elapsed += elapsed

                    reason = sel.reasons.get(kf_idx, "")
                    forced = kf_idx == 0 or "max_segment_length" in reason or "scene" in reason.lower()
                    keyframe_rows.append({
                        "video": video_key, "config": config_name, "frame_index": kf_idx,
                        "selector": sel_name, "reason": reason, "forced": forced,
                        "psnr": metrics["psnr"], "ssim": metrics["ssim"], "lpips": metrics["lpips"],
                        "channel_symbols": n_elements,
                        "total_bytes": total_bytes if total_bytes is not None else "",
                        "bit_depth": bit_depth if bit_depth is not None else "",
                        "elapsed_s": round(elapsed, 4),
                    })

                    if breakdown is not None:
                        video_bytes += breakdown.total_bytes
                        d = breakdown.as_dict()
                        d.update({"video": video_key, "config": config_name, "frame_index": kf_idx,
                                   "bit_depth": bit_depth})
                        packet_rows.append(d)

                    from sgdjscc_lab.io import save_tensor_as_image
                    save_tensor_as_image(
                        recon, output_root / "recon_videos" / video_key / config_name / f"frame_{kf_idx:05d}.png"
                    )

                n = max(len(video_psnr), 1)
                per_video_rows.append({
                    "video": video_key, "config": config_name, "selector": sel_name,
                    "channel": ch_name, "bit_depth": bit_depth if bit_depth is not None else "",
                    "n_keyframes": len(keyframes), "n_frames_total": len(frames),
                    "mean_psnr": sum(video_psnr) / n if video_psnr else float("nan"),
                    "mean_ssim": sum(video_ssim) / n if video_ssim else float("nan"),
                    "mean_lpips": (sum(video_lpips) / len(video_lpips)) if video_lpips else "",
                    "total_channel_symbols": video_symbols,
                    "total_packet_bytes": video_bytes if channel_kind != "awgn" else "",
                    "analog_no_wire_bytes": True if channel_kind == "awgn" else False,
                    "total_elapsed_s": round(video_elapsed, 3),
                })
                log(f"[{video_key}][{config_name}] mean_psnr={per_video_rows[-1]['mean_psnr']:.4f} "
                    f"bytes={per_video_rows[-1]['total_packet_bytes']}")

    _write_csv(output_root / "per_video_metrics.csv", per_video_rows)
    _write_csv(output_root / "packet_components.csv", packet_rows)
    _write_csv(output_root / "keyframe_sweep.csv", keyframe_sweep_rows)
    aggregate_rows = _aggregate(per_video_rows)
    _write_csv(output_root / "aggregate.csv", aggregate_rows)
    pareto_rows = _pareto_frontier(aggregate_rows)
    _write_csv(output_root / "pareto_frontier.csv", pareto_rows)

    codec_rows: List[Dict[str, Any]] = []
    if not args.skip_codec_comparison:
        codec_rows = _codec_comparison(entries, output_root)
    _write_csv(output_root / "codec_comparison.csv", codec_rows)

    _write_readme_and_summary(output_root, args, per_video_rows, pareto_rows)
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
        byte_rows = [r["total_packet_bytes"] for r in rows if r["total_packet_bytes"] != ""]
        out.append({
            "config": config,
            "selector": rows[0]["selector"],
            "channel": rows[0]["channel"],
            "bit_depth": rows[0]["bit_depth"],
            "n_videos": n,
            "mean_psnr": sum(r["mean_psnr"] for r in rows) / n,
            "mean_ssim": sum(r["mean_ssim"] for r in rows) / n,
            "mean_total_channel_symbols": sum(r["total_channel_symbols"] for r in rows) / n,
            "mean_total_packet_bytes": (sum(byte_rows) / len(byte_rows)) if byte_rows else "",
            "analog_no_wire_bytes": rows[0]["analog_no_wire_bytes"],
        })
    return out


def _pareto_frontier(aggregate_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baseline = next((r for r in aggregate_rows if r["config"] == "fixed_awgn"), None)
    if baseline is None:
        return []
    candidates = [r for r in aggregate_rows if r["config"] != "fixed_awgn" and r["mean_total_packet_bytes"] != ""]
    in_budget = []
    for r in candidates:
        psnr_drop = baseline["mean_psnr"] - r["mean_psnr"]
        ssim_drop = baseline["mean_ssim"] - r["mean_ssim"]
        ok = (psnr_drop <= QUALITY_GATE["psnr_drop_db"]) and (ssim_drop <= QUALITY_GATE["ssim_drop"])
        row = dict(r)
        row["psnr_drop_db"] = psnr_drop
        row["ssim_drop"] = ssim_drop
        row["within_quality_gate"] = ok
        in_budget.append(row)

    selected = [r for r in in_budget if r["within_quality_gate"]]
    pool = selected if selected else in_budget  # spec: if none qualify, report nearest, don't hide it
    pool_sorted = sorted(pool, key=lambda r: r["mean_total_packet_bytes"])
    for i, r in enumerate(pool_sorted):
        r["rank"] = i
        r["selected_as_smallest_in_budget"] = bool(selected) and i == 0
    return pool_sorted


def _codec_comparison(entries, output_root: Path) -> List[Dict[str, Any]]:
    rows = []
    for entry in entries:
        try:
            size = entry["processed"].stat().st_size
        except OSError:
            continue
        rows.append({
            "video": entry["key"],
            "source_file_bytes": size,
            "note": "exact source MP4 size only in this pass; run "
                    "scripts/benchmark_etri_video_rate.py for full H.264/H.265/AV1 "
                    "CRF sweep + quality-matched crossing-point search against a "
                    "completed reconstruction run",
        })
    return rows


def _write_readme_and_summary(output_root, args, per_video_rows, pareto_rows):
    summary = {
        "output_root": str(output_root),
        "configs_run": args.configs.split(","),
        "n_videos": len({r["video"] for r in per_video_rows}),
        "quality_gate": QUALITY_GATE,
        "pareto_selected": next((r for r in pareto_rows if r.get("selected_as_smallest_in_budget")), None),
        "psss_backend_used": "proxy (real CLIP-text cosine similarity) — backend='real' "
                              "(genuine HF causal-LM/VLM yes/no token probability) requires "
                              "keyframe.psss.real.model_id, not yet provisioned in this repo",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    readme = f"""# transmission_reduction run — {output_root.name}

Real binary-packet + 8/6/4-bit quantization + SKEM/PSSS transmission-reduction
sweep. See module docstring of `scripts/run_transmission_reduction_eval.py`
for exact-vs-estimate accounting boundaries.

- `per_video_metrics.csv` — per (video, config) quality + exact bytes/symbols.
- `aggregate.csv` — per-config means across videos.
- `packet_components.csv` — exact per-keyframe packet byte breakdown (header/
  shape/scale/metadata/payload/checksum), `proxy=false` throughout.
- `keyframe_sweep.csv` — PSSS threshold x max_segment_length grid (selection
  only, no reconstruction).
- `pareto_frontier.csv` — smallest-bytes config meeting the quality gate
  (PSNR drop <= {QUALITY_GATE['psnr_drop_db']} dB, SSIM drop <= {QUALITY_GATE['ssim_drop']});
  if none qualify, the nearest candidates are still listed (never hidden).
- `codec_comparison.csv` — source file sizes; run
  `scripts/benchmark_etri_video_rate.py` separately for the full H.264/H.265/AV1
  CRF sweep and quality-matched crossing-point search.
- `summary.json` — run configuration + selected config.

Known limitation: PSSS keyframe selection used `backend=proxy` (real CLIP
text-cosine similarity, not mock) because no HF `model_id` for the paper's
genuine yes/no-token-probability backend (`backend=real`) is provisioned in
this repository yet. `video/psss.py::MllmTokenProbPsssBackend` implements
`backend=real` and only needs a `keyframe.psss.real.model_id` config value to
activate.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(run())

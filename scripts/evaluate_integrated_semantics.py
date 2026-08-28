#!/usr/bin/env python
"""Evaluate saved integrated-run frames with the real CLIP/OWLv2/VQA ensemble.

One process owns one GPU and reuses all three presence models across every
assigned video/profile/policy.  Reconstruction is never repeated.  Reports
are written pair-by-pair atomically, so ``--resume`` safely skips completed
pairs after an interruption.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

POLICIES = ("full50", "few10", "vae_direct")
PROFILES = (
    "baseline", "combined_ds4",
    "candidate_edge_ds4_uncertainty_omit", "candidate_both_omit",
)
BACKENDS = ("clip", "owlv2", "vqa")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-root", default=str(ROOT / "data/etri_video_eval"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def read_csv(path: Path):
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _worker_plan(run_root: Path, worker_id: str):
    path = run_root / "reconstruction" / "full50" / "parallel_plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    worker = next((item for item in plan["assignments"] if item["worker_id"] == worker_id), None)
    if worker is None:
        raise ValueError(f"worker {worker_id!r} absent from {path}")
    return plan, worker


def preflight(run_root: Path, worker_id: str, dataset_root: Path, *, require_outputs: bool):
    config = ROOT / "configs/experiments/etri_video_eval/etri_video_eval_ensemble.yaml"
    if not config.is_file():
        raise FileNotFoundError(config)
    for sub in ("manifest.csv",):
        if not (dataset_root / sub).is_file():
            raise FileNotFoundError(dataset_root / sub)
    if require_outputs:
        plan, worker = _worker_plan(run_root, worker_id)
        for policy in POLICIES:
            policy_root = run_root / "reconstruction" / policy
            child = json.loads((policy_root / "parallel_plan.json").read_text(encoding="utf-8"))
            child_worker = next(item for item in child["assignments"] if item["worker_id"] == worker_id)
            if child_worker["videos"] != worker["videos"]:
                raise RuntimeError(f"video assignment changed for {policy}/{worker_id}")
            summary = json.loads((policy_root / "summary.json").read_text(encoding="utf-8"))
            if summary.get("run_status") != "completed":
                raise RuntimeError(f"reconstruction {policy} is not complete: {summary.get('run_status')}")
        return plan, worker
    return None, None


def _captions(path: Path, n: int):
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    return [(lines[i].strip() if i < len(lines) else "") for i in range(n)]


def _roles(path: Path, video: str, config: str, n: int):
    rows = [r for r in read_csv(path) if r.get("video") == video and r.get("config") == config]
    by_index = {int(r["frame_index"]): r.get("decision") for r in rows}
    if len(by_index) != n:
        raise RuntimeError(f"role provenance incomplete for {video}/{config}: {len(by_index)}/{n}")
    return ["keyframe" if by_index[i] == "keyframe" else "inter" for i in range(n)]


def _backend_counts(result):
    counts = {name: 0 for name in BACKENDS}
    n_objects = 0
    uncalibrated_mismatches = 0
    for row in result["calibrated"]["rows"]:
        raw = row.get("raw_clip_result") or {}
        mismatches = len(raw.get("missing_objects") or []) + len(raw.get("additional_objects") or [])
        calibrated = row.get("calibrated_presence_result") or []
        if mismatches and not calibrated:
            uncalibrated_mismatches += mismatches
        for item in calibrated:
            n_objects += 1
            contributing = set(item.get("contributing_backends") or [])
            missing = set(BACKENDS) - contributing
            if missing:
                raise RuntimeError(
                    f"presence ensemble silently degraded; missing {sorted(missing)} for "
                    f"object {item.get('object_name')!r}"
                )
            for name in BACKENDS:
                counts[name] += int(name in contributing)
    if uncalibrated_mismatches:
        raise RuntimeError(f"{uncalibrated_mismatches} object mismatches had no calibrated evidence")
    return n_objects, counts


def _metric_fields(prefix, metrics):
    names = (
        "n_items", "mean_severity", "total_missing_objects", "total_additional_objects",
        "temporal_srs", "srs_flicker", "object_identity_consistency",
        "temporal_hallucination_rate", "ptc", "sfr", "sdi",
    )
    return {f"{prefix}_{name}": metrics.get(name) for name in names}


def run(argv=None):
    args = parse_args(argv)
    run_root = Path(args.run_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    if args.preflight_only:
        preflight(run_root, args.worker_id, dataset_root, require_outputs=False)
        from sgdjscc_lab.config import load_config
        load_config(ROOT / "configs/experiments/etri_video_eval/etri_video_eval_ensemble.yaml")
        print(f"semantic preflight passed for {args.worker_id}")
        return 0

    _plan, worker = preflight(run_root, args.worker_id, dataset_root, require_outputs=True)
    out_root = run_root / "semantic" / "workers" / args.worker_id
    raw_root = out_root / "pair_reports"
    status_path = out_root / "integrated_semantic_rows.csv"
    completed = {
        (r["video"], r["decoder_policy"], r["guide_profile"]): r
        for r in read_csv(status_path)
    } if args.resume else {}

    import torch
    from omegaconf import OmegaConf
    from sgdjscc_lab.config import load_config
    from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator
    from sgdjscc_lab.evaluators.object_vocabulary_filter import build_object_vocabulary_filter
    from sgdjscc_lab.evaluators.presence_calibration import build_presence_calibrator
    from sgdjscc_lab.guidance.semantic_packet_extractor import SemanticPacketExtractor
    from sgdjscc_lab.pipelines.heldout_remeasurement import (
        convert_gt_to_presence, items_from_recon_frame_dirs, remeasure,
    )

    device = torch.device(args.device)
    cfg = load_config(ROOT / "configs/experiments/etri_video_eval/etri_video_eval_ensemble.yaml")
    cfg = OmegaConf.merge(cfg, {"verifier": {"presence_backend_cfg": {"vqa": {
        "vqa_backend": {"device": str(args.device)}
    }}}})
    calibrator = build_presence_calibrator(cfg)
    if calibrator is None or tuple(calibrator.backends) != BACKENDS:
        raise RuntimeError(f"expected real ensemble backends {BACKENDS}, got {tuple(calibrator.backends) if calibrator else None}")
    clip = CLIPScoreEvaluator(model_name=str(cfg.get("clip_model_name", "ViT-B/32")), device=device)
    packet_extractor = SemanticPacketExtractor(
        text_extractor=None, clip_evaluator=clip, device=device,
    )
    closed_cfg = OmegaConf.merge(cfg, {"verifier": {"object_vocabulary_filter": {
        "enabled": True, "use_gt_vocabulary": True,
    }}})
    open_cfg = OmegaConf.merge(cfg, {"verifier": {"object_vocabulary_filter": {
        "enabled": True, "use_gt_vocabulary": False,
    }}})
    closed_filter = build_object_vocabulary_filter(closed_cfg)
    open_filter = build_object_vocabulary_filter(open_cfg)

    rows_by_key = dict(completed)
    for policy in POLICIES:
        worker_root = run_root / "reconstruction" / policy / "workers" / args.worker_id
        quality = {(r["video"], r["config"]): r for r in read_csv(worker_root / "per_video_metrics.csv")}
        for video in worker["videos"]:
            original_dir = worker_root / "logs" / f"{video}_frames"
            gt_raw = json.loads((dataset_root / "gt" / f"{video}.json").read_text(encoding="utf-8"))
            gt = convert_gt_to_presence(gt_raw)
            for profile in PROFILES:
                key = (video, policy, profile)
                if key in completed:
                    continue
                config_name = f"fixed_int4__{profile}"
                recon_dir = worker_root / "recon_videos" / video / config_name
                frame_count = len(list(recon_dir.glob("frame_*.png")))
                if frame_count < 1:
                    raise RuntimeError(f"no reconstructed frames at {recon_dir}")
                captions = _captions(dataset_root / "captions" / f"{video}.txt", frame_count)
                roles = _roles(worker_root / "keyframe_selection.csv", video, config_name, frame_count)
                items = items_from_recon_frame_dirs(
                    original_dir, recon_dir, packet_extractor,
                    captions=captions, roles=roles, gt_metadata_by_id=gt,
                )
                closed = remeasure(items, presence_calibrator=calibrator, object_vocabulary_filter=closed_filter)
                opened = remeasure(items, presence_calibrator=calibrator, object_vocabulary_filter=open_filter)
                closed_n, closed_counts = _backend_counts(closed)
                open_n, open_counts = _backend_counts(opened)
                q = quality.get((video, config_name))
                if q is None:
                    raise RuntimeError(f"missing pixel/rate row for {video}/{policy}/{config_name}")
                row = {
                    "video": video, "decoder_policy": policy,
                    "decoder_mode": q.get("decoder_mode"),
                    "diffusion_step": q.get("diffusion_step"),
                    "effective_diffusion_step": q.get("effective_diffusion_step"),
                    "config": config_name, "guide_profile": profile,
                    "physical_cuda_device": os.environ.get("SGDJSCC_PHYSICAL_CUDA_DEVICE", args.device),
                    "n_frames": frame_count,
                    "mean_psnr": q["mean_psnr"], "mean_ssim": q["mean_ssim"],
                    "mean_lpips": q["mean_lpips"], "total_bundle_bytes": q["total_bundle_bytes"],
                    "total_elapsed_s": q["total_elapsed_s"],
                    "closed_calibrated_objects": closed_n,
                    "open_calibrated_objects": open_n,
                    **{f"closed_backend_{k}": v for k, v in closed_counts.items()},
                    **{f"open_backend_{k}": v for k, v in open_counts.items()},
                    **_metric_fields("closed", closed["calibrated"]["metrics"]),
                    **_metric_fields("open", opened["calibrated"]["metrics"]),
                }
                for name, value in row.items():
                    if isinstance(value, float) and not math.isfinite(value):
                        raise RuntimeError(f"non-finite integrated metric {name} for {key}")
                atomic_json(raw_root / policy / video / f"{profile}.json", {
                    "row": row, "closed_world": closed, "open_world": opened,
                    "configured_backends": list(BACKENDS),
                })
                rows_by_key[key] = row
                write_csv(status_path, [rows_by_key[k] for k in sorted(rows_by_key)])
                print(f"completed semantic pair {video}/{policy}/{profile}", flush=True)

    expected = len(worker["videos"]) * len(POLICIES) * len(PROFILES)
    if len(rows_by_key) != expected:
        raise RuntimeError(f"semantic worker incomplete: {len(rows_by_key)}/{expected}")
    atomic_json(out_root / "worker_summary.json", {
        "status": "completed", "worker_id": args.worker_id,
        "physical_cuda_device": os.environ.get("SGDJSCC_PHYSICAL_CUDA_DEVICE", args.device),
        "n_pairs": len(rows_by_key), "videos": worker["videos"],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

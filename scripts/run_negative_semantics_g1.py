#!/usr/bin/env python3
"""Run the frozen, resumable single-GPU negative-semantics G1 experiment.

G1 is inference and evaluation, not training.  The formal run uses only the
frozen OVIS Pilot split; the sealed held-out split is never read.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sgdjscc_lab.evaluators.negative_semantics import (  # noqa: E402
    select_global_threshold,
    summarize_g1,
)


PROTOCOL_PATH = ROOT / "configs/experiments/negative_semantics/g1_protocol.yaml"
G1_DATASET = ROOT / "data/negative_semantics/g1/pilot"
G1_V1_0_PROTOCOL_SHA256 = "b9cd11c0b132ec5e05a6a5f88a15989915b3c99811d8a542af4f968d3ab53e19"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _load_protocol() -> Dict[str, Any]:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(PROTOCOL_PATH), resolve=True)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_dataset() -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "_prepare_negative_semantics_g1", ROOT / "scripts/prepare_negative_semantics_g1.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.prepare(ROOT, G1_DATASET)


def _require_g0_pass() -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/audit_negative_semantics_g0.py"),
        "--repo-root", str(ROOT), "--require-pass",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True)
    if completed.returncode != 0:
        raise SystemExit("G0 audit did not pass; G1 is not permitted")


def _git_state() -> Dict[str, Any]:
    def output(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    dirty = (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0
        or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0
    )
    return {"commit": output("rev-parse", "HEAD"), "branch": output("rev-parse", "--abbrev-ref", "HEAD"), "dirty": dirty}


def _resolve_hf_snapshot(model_id: str) -> Dict[str, str]:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo_id=model_id, local_files_only=True)).resolve()
    return {"model_id": model_id, "snapshot_path": str(path), "revision": path.name}


def _preflight(protocol: Mapping[str, Any], formal: bool) -> Dict[str, Any]:
    import torch
    from sgdjscc_lab.paths import model_root

    git = _git_state()
    if formal and git["dirty"]:
        raise SystemExit(
            "formal G1 requires a tracked-clean checkout; commit the G1 implementation first"
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this environment")
    cuda_index = torch.cuda.current_device()
    checkpoints = [
        "JSCC_model.pth", "diffusion_backbone.pth", "diffusion_controlnet.pth",
        "muge-epoch-19-checkpoint.pth",
    ]
    checkpoint_root = Path(model_root())
    missing = [name for name in checkpoints if not (checkpoint_root / name).is_file()]
    if missing:
        raise SystemExit(f"missing SGD-JSCC checkpoints: {missing}")
    caption = _resolve_hf_snapshot(protocol["reconstruction"]["captions"]["model_id"])
    evaluator = _resolve_hf_snapshot(protocol["evaluator"]["model_id"])
    selector_ids = list(protocol["evaluator"]["selector_model_ids"])
    separation = evaluator["model_id"] not in selector_ids
    if not separation:
        raise SystemExit("evaluator and selector share weights, violating the frozen G1 protocol")
    return {
        "status": "PASSED",
        "git": git,
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(cuda_index),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(cuda_index).total_memory,
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_sha256": {name: _sha256(checkpoint_root / name) for name in checkpoints},
        "caption_model": caption,
        "evaluator_model": evaluator,
        "selector_model_ids": selector_ids,
        "selector_weight_separation": separation,
    }


def _resized_size(size: tuple[int, int], long_side: int) -> tuple[int, int]:
    width, height = size
    scale = min(1.0, float(long_side) / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _resize_pil(image, long_side: int):
    from PIL import Image

    target = _resized_size(image.size, long_side)
    if target != image.size:
        image = image.resize(target, Image.Resampling.BILINEAR)
    return image


def _pad_pil(image, pad_multiple: int):
    from PIL import Image

    width, height = image.size
    target_width = ((width + pad_multiple - 1) // pad_multiple) * pad_multiple
    target_height = ((height + pad_multiple - 1) // pad_multiple) * pad_multiple
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    canvas.paste(image, (left, top))
    return canvas


def _transform_pil(image, long_side: int, pad_multiple: int):
    return _pad_pil(_resize_pil(image, long_side), pad_multiple)


def _selected_videos(
    protocol: Mapping[str, Any], gt_index: Mapping[str, Any], smoke: bool
) -> List[str]:
    if not smoke:
        return sorted(gt_index["videos"])
    # Select a frozen-event video (rather than an arbitrary directory) for the
    # structural smoke. The smoke remains explicitly non-evidence.
    video_id = str(protocol["gate"]["smoke_video_id"])
    if video_id not in gt_index["videos"]:
        raise RuntimeError(f"frozen smoke video is not in Pilot: {video_id}")
    return [video_id]


def _caption_transform(snapshot: Mapping[str, str], long_side: int) -> Dict[str, Any]:
    return {
        "long_side": int(long_side),
        "rule": "aspect_resize_before_internal_model_padding",
        "model_revision": snapshot["revision"],
    }


def _adopt_v1_0_captions(
    prior_run: Path,
    gt_index: Mapping[str, Any],
    video_ids: Sequence[str],
    snapshot: Mapping[str, str],
    *,
    long_side: int,
) -> Dict[str, Any]:
    """Reuse v1.0 captions, which were generated before the failed model padding stage."""
    prior_run = Path(prior_run).resolve()
    spec = _load_json(prior_run / "run_spec.json")
    status = _load_json(prior_run / "caption_status.json")
    if spec.get("protocol_sha256") != G1_V1_0_PROTOCOL_SHA256 or spec.get("smoke") is not False:
        raise SystemExit("caption donor is not the invalidated formal G1 v1.0 run")
    donor_video_ids = list(spec.get("video_ids") or [])
    if not set(video_ids).issubset(donor_video_ids):
        raise SystemExit("caption donor does not cover the requested G1 v1.1 videos")
    if (
        status.get("status") != "GENERATED"
        or int(status.get("video_count", -1)) != len(donor_video_ids)
        or status.get("model_revision") != snapshot["revision"]
    ):
        raise SystemExit("caption donor did not finish the frozen caption stage")
    transform = _caption_transform(snapshot, long_side)
    caption_dir = G1_DATASET / "captions"
    hashes = {}
    for video_id in video_ids:
        path = caption_dir / f"{video_id}.txt"
        expected = len(gt_index["videos"][video_id]["frame_names"])
        lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        if len(lines) != expected or any(not line.strip() for line in lines):
            raise SystemExit(f"caption donor artifact is incomplete for {video_id}")
        hashes[video_id] = _sha256(path)
        _atomic_json(caption_dir / f"{video_id}.meta.json", transform)
    return {
        "status": "ADOPTED",
        "reason": "v1.0 captions precede and are independent of the failed internal padding stage",
        "source_run": str(prior_run),
        "source_protocol_sha256": spec["protocol_sha256"],
        "source_git_commit": spec["git_commit"],
        "model_revision": snapshot["revision"],
        "video_count": len(video_ids),
        "caption_sha256": hashes,
    }


def _generate_captions(
    gt_index: Mapping[str, Any], video_ids: Sequence[str], snapshot: Mapping[str, str],
    *, device: str, long_side: int,
) -> Dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    caption_dir = G1_DATASET / "captions"
    expected = {
        video_id: len(gt_index["videos"][video_id]["frame_names"])
        for video_id in video_ids
    }
    pending = []
    transform = _caption_transform(snapshot, long_side)
    for video_id in video_ids:
        path = caption_dir / f"{video_id}.txt"
        metadata_path = caption_dir / f"{video_id}.meta.json"
        lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        metadata = _load_json(metadata_path) if metadata_path.is_file() else None
        if (
            len(lines) != expected[video_id]
            or any(not line.strip() for line in lines)
            or metadata != transform
        ):
            pending.append(video_id)
    if not pending:
        return {"status": "REUSED", "video_count": len(video_ids), "model_revision": snapshot["revision"]}

    print(f"[G1] loading BLIP2 caption model; pending videos={len(pending)}", flush=True)
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(snapshot["snapshot_path"], local_files_only=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        snapshot["snapshot_path"], torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    for video_number, video_id in enumerate(pending, start=1):
        metadata = gt_index["videos"][video_id]
        source_dir = ROOT / metadata["source_path"]
        captions = []
        for frame_name in metadata["frame_names"]:
            with Image.open(source_dir / frame_name) as source:
                image = _resize_pil(source.convert("RGB"), long_side)
            inputs = processor(images=image, return_tensors="pt").to(device, dtype)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=30, num_beams=1)
            caption = processor.batch_decode(generated, skip_special_tokens=True)[0]
            captions.append(" ".join(caption.strip().split()) or "an image")
        target = caption_dir / f"{video_id}.txt"
        temporary = target.with_suffix(".txt.tmp")
        temporary.write_text("\n".join(captions) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        _atomic_json(caption_dir / f"{video_id}.meta.json", transform)
        print(f"[G1] captions {video_number}/{len(pending)}: {video_id}", flush=True)
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "GENERATED", "video_count": len(video_ids), "model_revision": snapshot["revision"]}


class _OwlV2Scorer:
    """Method-blind scorer: its public call receives only an image."""

    def __init__(self, snapshot: Mapping[str, str], queries: Mapping[str, str], device: str):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.device = device
        self.concepts = list(queries)
        self.query_text = [queries[concept] for concept in self.concepts]
        self.processor = Owlv2Processor.from_pretrained(snapshot["snapshot_path"], local_files_only=True)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            snapshot["snapshot_path"], local_files_only=True
        ).to(device).eval()

    def score_image(self, image) -> Dict[str, float]:
        inputs = self.processor(
            text=[self.query_text], images=image, return_tensors="pt"
        ).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = self.torch.tensor([image.size[::-1]])
        result = self.processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=target_sizes
        )[0]
        values = {concept: 0.0 for concept in self.concepts}
        for score, label in zip(result["scores"], result["labels"]):
            index = int(label.item())
            if 0 <= index < len(self.concepts):
                concept = self.concepts[index]
                values[concept] = max(values[concept], float(score.item()))
        return values

    def close(self) -> None:
        del self.model, self.processor
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _score_sequence(
    scorer: _OwlV2Scorer,
    frame_paths: Sequence[Path],
    cache_path: Path,
    *,
    long_side: int | None,
    pad_multiple: int | None,
    crop_size: tuple[int, int] | None,
    cache_metadata: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    from PIL import Image

    names = [path.name for path in frame_paths]
    if cache_path.is_file():
        cached = _load_json(cache_path)
        if (
            cached.get("frame_names") == names
            and cached.get("concepts") == scorer.concepts
            and cached.get("metadata") == dict(cache_metadata)
        ):
            return cached["scores"]
    rows = []
    for index, path in enumerate(frame_paths):
        with Image.open(path) as source:
            image = source.convert("RGB")
            if crop_size is not None:
                target_width, target_height = crop_size
                width, height = image.size
                if target_width > width or target_height > height:
                    raise RuntimeError(
                        f"evaluation crop {crop_size} exceeds image size {image.size}: {path}"
                    )
                left = (width - target_width) // 2
                top = (height - target_height) // 2
                image = image.crop((left, top, left + target_width, top + target_height))
            if long_side is not None:
                image = _resize_pil(image, long_side)
            if pad_multiple is not None:
                image = _pad_pil(image, pad_multiple)
            scores = scorer.score_image(image)
        rows.append({"frame_index": index, "frame_name": path.name, "scores": scores})
    _atomic_json(cache_path, {
        "frame_names": names,
        "concepts": scorer.concepts,
        "metadata": dict(cache_metadata),
        "scores": rows,
    })
    return rows


def _calibrate(
    protocol: Mapping[str, Any], gt_index: Mapping[str, Any], video_ids: Sequence[str],
    preflight: Mapping[str, Any], run_root: Path, *, device: str, max_frames: int | None,
    smoke: bool,
) -> Dict[str, Any]:
    evaluator = protocol["evaluator"]
    spatial = protocol["reconstruction"]["spatial_transform"]
    scorer = _OwlV2Scorer(preflight["evaluator_model"], evaluator["queries"], device)
    rows = []
    try:
        for number, video_id in enumerate(video_ids, start=1):
            metadata = gt_index["videos"][video_id]
            names = metadata["frame_names"][:max_frames] if max_frames else metadata["frame_names"]
            paths = [ROOT / metadata["source_path"] / name for name in names]
            scored = _score_sequence(
                scorer, paths, run_root / "evaluator/source" / f"{video_id}.json",
                long_side=int(spatial["image_long_side"]),
                pad_multiple=None,
                crop_size=None,
                cache_metadata={
                    "kind": "source", "source_sha256": metadata["source_sha256"],
                    "model_revision": preflight["evaluator_model"]["revision"],
                    "spatial_transform": spatial,
                },
            )
            for item in scored:
                present = set(metadata["present_concepts"][item["frame_index"]])
                for concept, score in item["scores"].items():
                    rows.append({
                        "video_id": video_id, "frame_index": item["frame_index"],
                        "concept_id": concept, "score": score,
                        "gt_present": concept in present,
                    })
            print(f"[G1] evaluator calibration {number}/{len(video_ids)}: {video_id}", flush=True)
    finally:
        scorer.close()
    calibration_cfg = evaluator["calibration"]
    try:
        report = select_global_threshold(
            rows, evaluator["threshold_grid"],
            min_recall=float(calibration_cfg["source_gt_recall_min"]),
            max_false_positive_rate=float(calibration_cfg["source_gt_false_positive_rate_max"]),
        )
    except ValueError as error:
        if not smoke:
            raise
        report = {
            "status": "NOT_EVIDENCE_SMOKE", "selected_threshold": 0.10,
            "reason": str(error), "candidates": [],
        }
    report.update({
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "evaluator_model": preflight["evaluator_model"],
        "selector_model_ids": preflight["selector_model_ids"],
        "selector_weight_separation": preflight["selector_weight_separation"],
        "calibration_uses_reconstruction_outputs": False,
        "smoke": smoke,
    })
    _atomic_json(run_root / "evaluator/evaluator_freeze.json", report)
    if not smoke and report["status"] != "PASSED":
        raise SystemExit(
            "frozen OWLv2 calibration did not satisfy recall/FPR constraints; formal reconstruction stopped"
        )
    return report


def _run_reconstructions(
    protocol: Mapping[str, Any], video_ids: Sequence[str], run_root: Path,
    *, device: str, policies: Mapping[str, int], seeds: Sequence[int], max_frames: int | None,
) -> None:
    recon = protocol["reconstruction"]
    spatial = recon["spatial_transform"]
    for policy, steps in policies.items():
        for seed in seeds:
            output = run_root / "reconstruction" / policy / f"seed_{seed}"
            command = [
                sys.executable, str(ROOT / "scripts/run_transmission_reduction_eval.py"),
                "--dataset-root", str(G1_DATASET),
                "--config", str(ROOT / recon["config"]),
                "--output-root", str(output),
                "--video-ids", ",".join(video_ids),
                "--configs", recon["base_config"],
                "--guide-profiles", recon["guide_profile"],
                "--decoder-mode", "diffusion",
                "--diffusion-step", str(steps),
                "--device", device,
                "--seed", str(seed),
                "--snr", str(recon["snr_db"]),
                "--digital-step-policy", recon["digital_step_policy"],
                "--fixed-reference-snr-db", str(recon["fixed_reference_snr_db"]),
                "--fixed-max-gop", str(recon["fixed_max_gop"]),
                "--image-long-side", str(spatial["image_long_side"]),
                "--image-pad-multiple", str(spatial["pad_to_multiple"]),
                "--fps", "1",
                "--skip-keyframe-sweep", "--skip-source-size-report", "--no-lpips",
            ]
            if max_frames is not None:
                command.extend(["--max-frames", str(max_frames)])
            print(f"[G1] reconstruction policy={policy} seed={seed} (resume-safe)", flush=True)
            completed = subprocess.run(command, cwd=ROOT)
            if completed.returncode != 0:
                raise SystemExit(
                    f"reconstruction failed: policy={policy}, seed={seed}, exit={completed.returncode}"
                )


def _score_reconstructions(
    protocol: Mapping[str, Any], gt_index: Mapping[str, Any], video_ids: Sequence[str],
    preflight: Mapping[str, Any], run_root: Path, *, device: str,
    policies: Mapping[str, int], seeds: Sequence[int], max_frames: int | None,
) -> List[Dict[str, Any]]:
    evaluator = protocol["evaluator"]
    recon = protocol["reconstruction"]
    config_name = f"{recon['base_config']}__{recon['guide_profile']}"
    scorer = _OwlV2Scorer(preflight["evaluator_model"], evaluator["queries"], device)
    rows: List[Dict[str, Any]] = []
    try:
        total = len(policies) * len(seeds) * len(video_ids)
        completed = 0
        for policy in policies:
            for seed in seeds:
                for video_id in video_ids:
                    metadata = gt_index["videos"][video_id]
                    expected = len(metadata["frame_names"])
                    if max_frames is not None:
                        expected = min(expected, max_frames)
                    recon_dir = (
                        run_root / "reconstruction" / policy / f"seed_{seed}" /
                        "recon_videos" / video_id / config_name
                    )
                    paths = [recon_dir / f"frame_{index:05d}.png" for index in range(expected)]
                    missing = [path for path in paths if not path.is_file()]
                    if missing:
                        raise SystemExit(
                            f"missing {len(missing)} reconstruction frames for {policy}/{seed}/{video_id}"
                        )
                    from PIL import Image
                    source_first = ROOT / metadata["source_path"] / metadata["frame_names"][0]
                    with Image.open(source_first) as source_image:
                        content_size = _resized_size(
                            source_image.size,
                            int(recon["spatial_transform"]["image_long_side"]),
                        )
                    scored = _score_sequence(
                        scorer, paths,
                        run_root / "evaluator/reconstruction" / policy / f"seed_{seed}" / f"{video_id}.json",
                        long_side=None,
                        pad_multiple=None,
                        crop_size=content_size,
                        cache_metadata={
                            "kind": "reconstruction", "model_revision": preflight["evaluator_model"]["revision"],
                            "frame_count": expected,
                            "evaluation_crop_size": list(content_size),
                        },
                    )
                    for item in scored:
                        present = set(metadata["present_concepts"][item["frame_index"]])
                        for concept, score in item["scores"].items():
                            rows.append({
                                "video_id": video_id, "seed": int(seed), "policy": policy,
                                "frame_index": item["frame_index"], "concept_id": concept,
                                "score": score, "gt_present": concept in present,
                            })
                    completed += 1
                    print(f"[G1] reconstruction evaluator {completed}/{total}: {policy}/{seed}/{video_id}", flush=True)
    finally:
        scorer.close()
    _atomic_jsonl(run_root / "evaluator/detection_rows.jsonl", rows)
    return rows


def _write_summary(
    protocol: Mapping[str, Any], gt_index: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any], run_root: Path, *, smoke: bool,
    policies: Mapping[str, int], seeds: Sequence[int], preflight: Mapping[str, Any],
) -> Dict[str, Any]:
    threshold = float(calibration["selected_threshold"])
    metric_cfg = protocol["metrics"]
    report = summarize_g1(
        rows, gt_index["events"], threshold=threshold,
        primary_policy=protocol["gate"]["primary_policy"],
        bootstrap_iterations=int(metric_cfg["statistics"]["bootstrap_iterations"]),
        ghost_horizon=int(metric_cfg["ghost_survival_auc"]["horizon_annotated_timesteps"]),
    )
    gate_cfg = protocol["gate"]
    primary = report["by_policy"].get(gate_cfg["primary_policy"], {})
    checks = {
        "formal_not_smoke": not smoke,
        "evaluator_calibration_passed": calibration["status"] == "PASSED",
        "selector_weight_separation": bool(preflight["selector_weight_separation"]),
        "all_policies_present": set(report["by_policy"]) == set(protocol["reconstruction"]["policies"]),
        "all_seeds_present": {int(row["seed"]) for row in rows} == set(protocol["reconstruction"]["seeds"]),
        "h_add_prevalence": (primary.get("h_add") or 0.0) >= float(gate_cfg["h_add_min"]),
        "additional_event_count": primary.get("additional_event_count", 0) >= int(gate_cfg["additional_events_min"]),
        "affected_video_count": primary.get("affected_video_count", 0) >= int(gate_cfg["affected_videos_min"]),
        "affected_seed_count": primary.get("affected_seed_count", 0) >= int(gate_cfg["affected_seeds_min"]),
        "not_concentrated_in_one_video": primary.get("max_single_video_event_share", 1.0) <= float(gate_cfg["max_single_video_event_share"]),
        "not_concentrated_in_one_seed": primary.get("max_single_seed_event_share", 1.0) <= float(gate_cfg["max_single_seed_event_share"]),
    }
    report.update({
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "evidence_scope": "NOT_EVIDENCE_SMOKE" if smoke else "OVIS_PILOT_G1",
        "policies": dict(policies), "seeds": list(seeds),
        "gate_checks": checks,
        "gate_status": "NOT_EVIDENCE" if smoke else ("PASSED" if all(checks.values()) else "NOT_PASSED"),
        "heldout_accessed": False,
        "git_commit": preflight["git"]["commit"],
    })
    _atomic_json(run_root / "g1_summary.json", report)
    csv_path = run_root / "g1_policy_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "policy", "h_add", "absent_opportunities", "additional_detections",
            "additional_event_count", "affected_video_count", "affected_seed_count",
            "max_single_video_event_share", "max_single_seed_event_share",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for policy, values in report["by_policy"].items():
            writer.writerow({"policy": policy, **{field: values.get(field) for field in fields[1:]}})
    return report


def run(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Two-frame structural GPU check; never paper evidence.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--reuse-captions-from-run", type=Path, default=None,
        help="Adopt fully completed source-only captions from the invalidated formal G1 v1.0 run.",
    )
    args = parser.parse_args(argv)

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    protocol = _load_protocol()
    if protocol["freeze_status"] != "frozen" or protocol["scope"]["heldout_access"] != "prohibited":
        raise SystemExit("invalid or unfrozen G1 protocol")
    _require_g0_pass()
    preparation = _prepare_dataset()
    gt_index = _load_json(G1_DATASET / "ground_truth_index.json")
    preflight = _preflight(protocol, formal=not args.smoke and not args.preflight_only)
    _atomic_json(run_root / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    video_ids = _selected_videos(protocol, gt_index, args.smoke)
    policies = (
        {"few10": int(protocol["reconstruction"]["policies"]["few10"])}
        if args.smoke else {key: int(value) for key, value in protocol["reconstruction"]["policies"].items()}
    )
    seeds = [int(protocol["reconstruction"]["seeds"][0])] if args.smoke else [int(value) for value in protocol["reconstruction"]["seeds"]]
    max_frames = 2 if args.smoke else None
    run_spec = {
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "preparation_manifest_sha256": _sha256(G1_DATASET / "preparation_manifest.json"),
        "git_commit": preflight["git"]["commit"],
        "git_dirty": preflight["git"]["dirty"],
        "smoke": args.smoke, "video_ids": video_ids, "policies": policies,
        "seeds": seeds, "max_frames": max_frames, "device": args.device,
        "heldout_accessed": False,
    }
    spec_path = run_root / "run_spec.json"
    if spec_path.is_file() and _load_json(spec_path) != run_spec:
        raise SystemExit("resume run_spec mismatch; choose another --run-root")
    _atomic_json(spec_path, run_spec)

    if args.reuse_captions_from_run is not None:
        adoption = _adopt_v1_0_captions(
            args.reuse_captions_from_run, gt_index, video_ids, preflight["caption_model"],
            long_side=int(protocol["reconstruction"]["spatial_transform"]["image_long_side"]),
        )
        _atomic_json(run_root / "caption_adoption.json", adoption)
    captions = _generate_captions(
        gt_index, video_ids, preflight["caption_model"], device=args.device,
        long_side=int(protocol["reconstruction"]["spatial_transform"]["image_long_side"]),
    )
    _atomic_json(run_root / "caption_status.json", captions)
    calibration = _calibrate(
        protocol, gt_index, video_ids, preflight, run_root,
        device=args.device, max_frames=max_frames, smoke=args.smoke,
    )
    _run_reconstructions(
        protocol, video_ids, run_root, device=args.device,
        policies=policies, seeds=seeds, max_frames=max_frames,
    )
    rows = _score_reconstructions(
        protocol, gt_index, video_ids, preflight, run_root, device=args.device,
        policies=policies, seeds=seeds, max_frames=max_frames,
    )
    report = _write_summary(
        protocol, gt_index, rows, calibration, run_root, smoke=args.smoke,
        policies=policies, seeds=seeds, preflight=preflight,
    )
    print(f"[G1] complete: {report['gate_status']} -> {run_root / 'g1_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

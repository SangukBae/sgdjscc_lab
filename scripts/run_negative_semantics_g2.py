#!/usr/bin/env python3
"""Run the frozen SV0/G2 Oracle ABSENT receiver-controllability experiment.

The four arms share fixed_int4 packets, captions, selector decisions, frame
seeds and diffusion-step budgets.  Only an out-of-band receiver negative-text
condition changes.  Oracle results are therefore mechanism-feasibility
evidence, never a rate-bearing communication result.
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

from sgdjscc_lab.evaluators.negative_semantics_g2 import summarize_g2  # noqa: E402
from sgdjscc_lab.guidance.negative_conditioning import (  # noqa: E402
    G2_ARMS,
    build_g2_condition_manifests,
    validate_g2_condition_manifest,
)


PROTOCOL_PATH = ROOT / "configs/experiments/negative_semantics/g2_oracle_absent_protocol.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _load_protocol() -> Dict[str, Any]:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(PROTOCOL_PATH), resolve=True)


def _load_g1_module():
    name = "_negative_semantics_g1_for_g2"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/run_negative_semantics_g1.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git_state() -> Dict[str, Any]:
    def output(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    dirty = (
        subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0
        or subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0
    )
    return {
        "commit": output("rev-parse", "HEAD"),
        "branch": output("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": dirty,
    }


def _require_file_hash(relative: str, expected: str) -> Path:
    path = ROOT / relative
    if not path.is_file():
        raise SystemExit(f"missing frozen G2 dependency: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise SystemExit(
            f"frozen G2 dependency hash mismatch: {relative}: expected={expected}, actual={actual}"
        )
    return path


def _dependency_state(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    dependency = protocol["dependency"]
    paths = {}
    for name in ("g1_detection_rows", "g1_evaluator_freeze", "g1_v1_2_summary"):
        paths[name] = _require_file_hash(
            str(dependency[name]), str(dependency[f"{name}_sha256"])
        )
    for name in ("ground_truth_index", "preparation_manifest"):
        paths[name] = _require_file_hash(
            str(protocol["data"][name]), str(protocol["data"][f"{name}_sha256"])
        )
    evaluator_freeze = _load_json(paths["g1_evaluator_freeze"])
    if evaluator_freeze.get("status") != "PASSED":
        raise SystemExit("G1 source-only evaluator calibration is not PASSED")
    if float(evaluator_freeze.get("selected_threshold")) != float(protocol["evaluator"]["threshold"]):
        raise SystemExit("G2 detector threshold differs from frozen G1 calibration")
    derived = _load_json(paths["g1_v1_2_summary"])
    scientific_status = (
        (derived.get("gate_reapplication") or {}).get("effective_seed_status")
        or "UNKNOWN"
    )
    if scientific_status != "PASSED" and not bool(
        dependency["allow_exploratory_g2_while_g1_scientific_gate_not_passed"]
    ):
        raise SystemExit("G1 scientific gate is not PASSED and exploratory G2 is disabled")
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "sha256": {key: _sha256(value) for key, value in paths.items()},
        "g1_scientific_gate_status": scientific_status,
        "evaluator_freeze": evaluator_freeze,
    }


def _preflight(protocol: Mapping[str, Any], dependency: Mapping[str, Any], *, formal: bool) -> Dict[str, Any]:
    import torch
    from sgdjscc_lab.paths import model_root

    git = _git_state()
    if formal and git["dirty"]:
        raise SystemExit("formal G2 requires a tracked-clean checkout; commit the implementation first")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this environment")
    checkpoint_root = Path(model_root())
    checkpoint_names = (
        "JSCC_model.pth", "diffusion_backbone.pth", "diffusion_controlnet.pth",
        "muge-epoch-19-checkpoint.pth",
    )
    missing = [name for name in checkpoint_names if not (checkpoint_root / name).is_file()]
    if missing:
        raise SystemExit(f"missing SGD-JSCC checkpoints: {missing}")
    evaluator_snapshot = _load_g1_module()._resolve_hf_snapshot(protocol["evaluator"]["model_id"])
    frozen_revision = dependency["evaluator_freeze"]["evaluator_model"]["revision"]
    if evaluator_snapshot["revision"] != frozen_revision:
        raise SystemExit("locally resolved OWLv2 revision differs from the frozen G1 evaluator")
    cuda_index = torch.cuda.current_device()
    return {
        "status": "PASSED",
        "git": git,
        "python": sys.executable,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(cuda_index),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(cuda_index).total_memory,
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_sha256": {
            name: _sha256(checkpoint_root / name) for name in checkpoint_names
        },
        "evaluator_model": evaluator_snapshot,
        "g1_scientific_gate_status": dependency["g1_scientific_gate_status"],
    }


def _selected_videos(protocol: Mapping[str, Any], gt_index: Mapping[str, Any], smoke: bool) -> List[str]:
    if smoke:
        video_id = str(protocol["gate"]["smoke_video_id"])
        if video_id not in gt_index["videos"]:
            raise SystemExit(f"smoke video is absent from G2 Pilot: {video_id}")
        return [video_id]
    values = sorted(gt_index["videos"])
    if len(values) != int(protocol["data"]["expected_videos"]):
        raise SystemExit("G2 Pilot video count differs from the frozen protocol")
    return values


def _verify_captions(protocol: Mapping[str, Any], gt_index: Mapping[str, Any], video_ids: Sequence[str]) -> Dict[str, str]:
    caption_dir = ROOT / protocol["data"]["dataset_root"] / "captions"
    hashes = {}
    for video_id in video_ids:
        path = caption_dir / f"{video_id}.txt"
        lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        expected = len(gt_index["videos"][video_id]["frame_names"])
        if len(lines) != expected or any(not line.strip() for line in lines):
            raise SystemExit(f"missing/incomplete frozen source caption file: {video_id}")
        hashes[video_id] = _sha256(path)
    return hashes


def _materialize_conditions(
    protocol: Mapping[str, Any], gt_index: Mapping[str, Any], video_ids: Sequence[str],
    run_root: Path,
) -> Dict[str, Path]:
    condition = protocol["negative_conditioning"]
    manifests = build_g2_condition_manifests(
        gt_index, video_ids,
        concepts=protocol["evaluator"]["concepts"],
        queries=protocol["evaluator"]["queries"],
        frequency_ranking=condition["frequency_ranking"],
        random_seed=int(condition["random_seed"]),
    )
    if list(condition["arms"]) != list(G2_ARMS):
        raise SystemExit("G2 protocol arm order differs from the implemented four-arm contract")
    expected = {
        video_id: len(gt_index["videos"][video_id]["frame_names"])
        for video_id in video_ids
    }
    paths = {}
    for arm in G2_ARMS:
        validate_g2_condition_manifest(manifests[arm], expected)
        path = run_root / "conditions" / f"{arm}.json"
        if path.is_file() and _load_json(path) != manifests[arm]:
            raise SystemExit(f"resume condition mismatch for {arm}; choose another --run-root")
        _atomic_json(path, manifests[arm])
        paths[arm] = path
    return paths


def _run_reconstructions(
    protocol: Mapping[str, Any], video_ids: Sequence[str], conditions: Mapping[str, Path],
    run_root: Path, *, device: str, policies: Mapping[str, int], seeds: Sequence[int],
    max_frames: int | None, smoke: bool,
) -> None:
    recon = protocol["reconstruction"]
    spatial = recon["spatial_transform"]
    for arm in G2_ARMS:
        for policy, steps in policies.items():
            for seed in seeds:
                output = run_root / "reconstruction" / arm / policy / f"seed_{seed}"
                command = [
                    sys.executable, str(ROOT / "scripts/run_transmission_reduction_eval.py"),
                    "--dataset-root", str(ROOT / protocol["data"]["dataset_root"]),
                    "--config", str(ROOT / recon["config"]),
                    "--output-root", str(output),
                    "--video-ids", ",".join(video_ids),
                    "--configs", recon["base_config"],
                    "--guide-profiles", recon["guide_profile"],
                    "--decoder-mode", recon["decoder_mode"],
                    "--diffusion-step", str(steps),
                    "--device", device,
                    "--seed", str(seed),
                    "--snr", str(recon["snr_db"]),
                    "--digital-step-policy", recon["digital_step_policy"],
                    "--fixed-reference-snr-db", str(recon["fixed_reference_snr_db"]),
                    "--fixed-max-gop", str(recon["fixed_max_gop"]),
                    "--image-long-side", str(spatial["image_long_side"]),
                    "--image-pad-multiple", str(spatial["pad_to_multiple"]),
                    "--negative-condition-manifest", str(conditions[arm]),
                    "--fps", "1", "--skip-keyframe-sweep", "--skip-source-size-report",
                ]
                if smoke:
                    command.append("--no-lpips")
                if max_frames is not None:
                    command.extend(["--max-frames", str(max_frames)])
                print(f"[G2] reconstruction arm={arm} policy={policy} seed={seed}", flush=True)
                completed = subprocess.run(command, cwd=ROOT)
                if completed.returncode != 0:
                    raise SystemExit(
                        f"G2 reconstruction failed: arm={arm}, policy={policy}, "
                        f"seed={seed}, exit={completed.returncode}"
                    )


def _source_score_lookup(
    protocol: Mapping[str, Any], dependency: Mapping[str, Any], video_ids: Sequence[str],
) -> Dict[tuple, float]:
    root = Path(protocol["dependency"]["g1_run_root"])
    if not root.is_absolute():
        root = ROOT / root
    revision = dependency["evaluator_freeze"]["evaluator_model"]["revision"]
    concepts = set(protocol["evaluator"]["concepts"])
    lookup = {}
    for video_id in video_ids:
        path = root / "evaluator/source" / f"{video_id}.json"
        if not path.is_file():
            raise SystemExit(f"missing frozen G1 source score cache: {path}")
        payload = _load_json(path)
        if set(payload.get("concepts") or []) != concepts:
            raise SystemExit(f"source score concept grid mismatch: {video_id}")
        metadata = payload.get("metadata") or {}
        if metadata.get("model_revision") != revision:
            raise SystemExit(f"source score model revision mismatch: {video_id}")
        for item in payload.get("scores") or []:
            for concept, score in item["scores"].items():
                lookup[(video_id, int(item["frame_index"]), concept)] = float(score)
    return lookup


def _score_reconstructions(
    protocol: Mapping[str, Any], dependency: Mapping[str, Any], gt_index: Mapping[str, Any],
    video_ids: Sequence[str], run_root: Path, *, device: str,
    policies: Mapping[str, int], seeds: Sequence[int], max_frames: int | None,
) -> List[Dict[str, Any]]:
    g1 = _load_g1_module()
    recon = protocol["reconstruction"]
    config_name = f"{recon['base_config']}__{recon['guide_profile']}"
    source = _source_score_lookup(protocol, dependency, video_ids)
    scorer = g1._OwlV2Scorer(
        dependency["evaluator_freeze"]["evaluator_model"],
        protocol["evaluator"]["queries"], device,
    )
    rows: List[Dict[str, Any]] = []
    try:
        total = len(G2_ARMS) * len(policies) * len(seeds) * len(video_ids)
        completed = 0
        for arm in G2_ARMS:
            for policy in policies:
                for seed in seeds:
                    for video_id in video_ids:
                        metadata = gt_index["videos"][video_id]
                        expected = len(metadata["frame_names"])
                        if max_frames is not None:
                            expected = min(expected, max_frames)
                        recon_dir = (
                            run_root / "reconstruction" / arm / policy / f"seed_{seed}" /
                            "recon_videos" / video_id / config_name
                        )
                        paths = [recon_dir / f"frame_{index:05d}.png" for index in range(expected)]
                        if any(not path.is_file() for path in paths):
                            raise SystemExit(f"missing G2 reconstruction frames: {arm}/{policy}/{seed}/{video_id}")
                        from PIL import Image
                        source_first = ROOT / metadata["source_path"] / metadata["frame_names"][0]
                        with Image.open(source_first) as source_image:
                            content_size = g1._resized_size(
                                source_image.size, int(recon["spatial_transform"]["image_long_side"])
                            )
                        scored = g1._score_sequence(
                            scorer, paths,
                            run_root / "evaluator/reconstruction" / arm / policy /
                            f"seed_{seed}" / f"{video_id}.json",
                            long_side=None, pad_multiple=None, crop_size=content_size,
                            cache_metadata={
                                "kind": "g2_reconstruction", "arm": arm, "policy": policy,
                                "seed": int(seed),
                                "model_revision": dependency["evaluator_freeze"]["evaluator_model"]["revision"],
                                "frame_count": expected, "evaluation_crop_size": list(content_size),
                            },
                        )
                        for item in scored:
                            frame_index = int(item["frame_index"])
                            present = set(metadata["present_concepts"][frame_index])
                            for concept, score in item["scores"].items():
                                key = (video_id, frame_index, concept)
                                if key not in source:
                                    raise SystemExit(f"missing source-paired detector score: {key}")
                                rows.append({
                                    "video_id": video_id, "arm": arm, "policy": policy,
                                    "seed": int(seed), "frame_index": frame_index,
                                    "concept_id": concept, "score": float(score),
                                    "source_score": source[key], "gt_present": concept in present,
                                })
                        completed += 1
                        print(f"[G2] evaluator {completed}/{total}: {arm}/{policy}/{seed}/{video_id}", flush=True)
    finally:
        scorer.close()
        gc.collect()
    _atomic_jsonl(run_root / "evaluator/detection_rows.jsonl", rows)
    return rows


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"missing G2 child artifact: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _collect_accounting_and_verify_matching(
    protocol: Mapping[str, Any], video_ids: Sequence[str], run_root: Path,
    policies: Mapping[str, int], seeds: Sequence[int], *, smoke: bool,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    accounting: List[Dict[str, Any]] = []
    schedules: Dict[tuple, List[tuple]] = {}
    row_lookup: Dict[tuple, Dict[str, str]] = {}
    fields = (
        "config", "channel", "bit_depth", "digital_step_policy", "fixed_reference_snr_db",
        "decoder_mode", "diffusion_step", "effective_diffusion_step", "n_frames_total",
        "n_transmitting_frames", "n_keyframes_selected", "latent_elements_total",
        "source_packet_bits_total", "total_bundle_bytes",
    )
    for arm in G2_ARMS:
        for policy in policies:
            for seed in seeds:
                child = run_root / "reconstruction" / arm / policy / f"seed_{seed}"
                rows = _read_csv(child / "per_video_metrics.csv")
                by_video = {row["video"]: row for row in rows}
                if set(by_video) != set(video_ids) or len(rows) != len(video_ids):
                    raise SystemExit(f"G2 child video grid mismatch: {arm}/{policy}/{seed}")
                selection_rows = _read_csv(child / "keyframe_selection.csv")
                for video_id in video_ids:
                    row = by_video[video_id]
                    if row.get("negative_condition_arm") != arm:
                        raise SystemExit(f"child did not record expected condition arm: {arm}/{video_id}")
                    key = (policy, int(seed), video_id)
                    row_lookup[(arm, *key)] = row
                    schedules[(arm, *key)] = [
                        (item["frame_index"], item["decision"], item["visual_transmitted"],
                         item["source_packet_bits"])
                        for item in selection_rows if item["video"] == video_id
                    ]
                    accounting.append({
                        **row, "video_id": video_id, "arm": arm,
                        "policy": policy, "seed": int(seed),
                    })
    mismatches = []
    reference = G2_ARMS[0]
    for policy in policies:
        for seed in seeds:
            for video_id in video_ids:
                key = (policy, int(seed), video_id)
                ref_row = row_lookup[(reference, *key)]
                ref_values = {field: ref_row.get(field, "") for field in fields}
                ref_schedule = schedules[(reference, *key)]
                for arm in G2_ARMS[1:]:
                    candidate = row_lookup[(arm, *key)]
                    different = {
                        field: {"reference": ref_values[field], "candidate": candidate.get(field, "")}
                        for field in fields if candidate.get(field, "") != ref_values[field]
                    }
                    if schedules[(arm, *key)] != ref_schedule:
                        different["frame_schedule"] = "different"
                    if different:
                        mismatches.append({
                            "arm": arm, "policy": policy, "seed": int(seed),
                            "video_id": video_id, "differences": different,
                        })
    if mismatches:
        raise SystemExit(f"G2 matched-compute/rate contract failed: {mismatches[:3]}")
    report = {
        "status": "NOT_EVIDENCE_SMOKE" if smoke else "PASSED",
        "reference_arm": reference,
        "matched_fields": list(fields) + ["frame_schedule"],
        "comparison_count": (len(G2_ARMS) - 1) * len(policies) * len(seeds) * len(video_ids),
        "mismatch_count": 0,
        "oracle_side_input_rate_accounted": False,
    }
    _atomic_json(run_root / "matched_compute_audit.json", report)
    _atomic_csv(run_root / "accounting_rows.csv", accounting)
    return accounting, report


def _artifact_hashes(run_root: Path) -> Dict[str, str]:
    names = (
        "run_spec.json", "preflight.json", "matched_compute_audit.json",
        "accounting_rows.csv", "evaluator/detection_rows.jsonl", "g2_summary.json",
    )
    values = {}
    for name in names:
        path = run_root / name
        if path.is_file():
            values[name] = _sha256(path)
    for arm in G2_ARMS:
        path = run_root / "conditions" / f"{arm}.json"
        if path.is_file():
            values[str(path.relative_to(run_root))] = _sha256(path)
    return values


def run(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Two-frame GPU wiring check; never evidence.")
    parser.add_argument("--prepare-only", action="store_true", help="Materialize/validate condition files without CUDA.")
    parser.add_argument("--preflight-only", action="store_true", help="Run dependency/GPU preflight and stop.")
    args = parser.parse_args(argv)

    protocol = _load_protocol()
    if protocol["freeze_status"] != "frozen" or protocol["scope"]["heldout_access"] != "prohibited":
        raise SystemExit("invalid or unfrozen G2 protocol")
    if protocol["reconstruction"]["base_config"] != "fixed_int4":
        raise SystemExit("G2 protocol must keep fixed_int4 in every arm")
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    # G0 audit and G1 dataset preparation touch no held-out media. Keep the
    # same frozen materialization used by G1.
    _load_g1_module()._require_g0_pass()
    _load_g1_module()._prepare_dataset()
    dependency = _dependency_state(protocol)
    gt_index = _load_json(ROOT / protocol["data"]["ground_truth_index"])
    video_ids = _selected_videos(protocol, gt_index, args.smoke)
    caption_hashes = _verify_captions(protocol, gt_index, video_ids)
    conditions = _materialize_conditions(protocol, gt_index, video_ids, run_root)

    if args.prepare_only:
        report = {
            "status": "PREPARED_NOT_VALIDATED", "protocol_sha256": _sha256(PROTOCOL_PATH),
            "video_ids": video_ids, "condition_sha256": {
                arm: _sha256(path) for arm, path in conditions.items()
            }, "heldout_accessed": False,
        }
        _atomic_json(run_root / "preparation.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    preflight = _preflight(
        protocol, dependency, formal=not args.smoke and not args.preflight_only
    )
    _atomic_json(run_root / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    policies = (
        {"few10": int(protocol["reconstruction"]["policies"]["few10"])}
        if args.smoke else {
            key: int(value) for key, value in protocol["reconstruction"]["policies"].items()
        }
    )
    seeds = [int(protocol["reconstruction"]["seeds"][0])] if args.smoke else [
        int(value) for value in protocol["reconstruction"]["seeds"]
    ]
    max_frames = 2 if args.smoke else None
    run_spec = {
        "protocol_id": protocol["protocol_id"], "protocol_sha256": _sha256(PROTOCOL_PATH),
        "git_commit": preflight["git"]["commit"], "git_dirty": preflight["git"]["dirty"],
        "smoke": bool(args.smoke), "video_ids": video_ids, "policies": policies,
        "seeds": seeds, "arms": list(G2_ARMS), "max_frames": max_frames,
        "device": args.device, "caption_sha256": caption_hashes,
        "condition_sha256": {arm: _sha256(path) for arm, path in conditions.items()},
        "g1_dependency_sha256": dependency["sha256"],
        "g1_scientific_gate_status": dependency["g1_scientific_gate_status"],
        "oracle_side_input": "ORACLE_EVAL_ONLY", "oracle_rate_accounted": False,
        "heldout_accessed": False,
    }
    spec_path = run_root / "run_spec.json"
    if spec_path.is_file() and _load_json(spec_path) != run_spec:
        raise SystemExit("resume run_spec mismatch; choose another --run-root")
    _atomic_json(spec_path, run_spec)

    _run_reconstructions(
        protocol, video_ids, conditions, run_root, device=args.device,
        policies=policies, seeds=seeds, max_frames=max_frames, smoke=args.smoke,
    )
    rows = _score_reconstructions(
        protocol, dependency, gt_index, video_ids, run_root, device=args.device,
        policies=policies, seeds=seeds, max_frames=max_frames,
    )
    accounting, matched = _collect_accounting_and_verify_matching(
        protocol, video_ids, run_root, policies, seeds, smoke=args.smoke,
    )
    if args.smoke:
        report = {
            "protocol_id": protocol["protocol_id"],
            "evidence_scope": "NOT_EVIDENCE_SMOKE", "gate_status": "NOT_EVIDENCE",
            "detection_row_count": len(rows), "accounting_row_count": len(accounting),
            "matched_compute": matched, "heldout_accessed": False,
        }
    else:
        report = summarize_g2(
            rows, accounting, threshold=float(protocol["evaluator"]["threshold"]),
            arms=list(G2_ARMS), policies=list(policies),
            primary_policy=str(protocol["gate"]["primary_policy"]),
            bootstrap_iterations=int(protocol["metrics"]["statistics"]["bootstrap_iterations"]),
            gate=protocol["gate"],
        )
        dependency_passed = dependency["g1_scientific_gate_status"] == "PASSED"
        report.update({
            "protocol_id": protocol["protocol_id"], "protocol_sha256": _sha256(PROTOCOL_PATH),
            "evidence_scope": (
                "OVIS_PILOT_G2" if dependency_passed
                else "EXPLORATORY_MECHANISM_FEASIBILITY_G1_NOT_PASSED"
            ),
            "gate_status": (
                report["provisional_gate_status"] if dependency_passed
                else "NOT_CONFIRMATORY_DEPENDENCY_G1_NOT_PASSED"
            ),
            "g1_scientific_gate_status": dependency["g1_scientific_gate_status"],
            "matched_compute": matched, "oracle_side_input": "ORACLE_EVAL_ONLY",
            "oracle_rate_accounted": False, "heldout_accessed": False,
            "git_commit": preflight["git"]["commit"],
        })
    _atomic_json(run_root / "g2_summary.json", report)
    _atomic_json(run_root / "artifact_checksums.json", _artifact_hashes(run_root))
    print(f"[G2] complete: {report['gate_status']} -> {run_root / 'g2_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

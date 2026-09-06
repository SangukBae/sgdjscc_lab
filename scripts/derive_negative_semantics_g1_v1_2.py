#!/usr/bin/env python3
"""Derive the G1 v1.2 amendment summary from an already-frozen v1.1 run.

Read-only with respect to the source run: this script never writes into
``--run-root`` (the frozen ``outputs/negative_semantics_g1_pilot_rtx4080_v1_1``
by default) and performs no GPU inference, reconstruction, or evaluator
scoring of its own. It re-derives metrics purely from artifacts already on
disk -- ``evaluator/detection_rows.jsonl``, the cached ``evaluator/source/
*.json`` OWLv2 scores, the frozen dataset's ``ground_truth_index.json`` event
list, and the reconstruction PNG frames themselves -- and writes a new,
separate output directory documented in
``docs/experiments/2026-09-06_negative_semantics_g1_v1_2_amendment.md``.

Before touching any of that, it verifies the source run's provenance chain
end to end: the *current* ``g1_protocol.yaml`` and dataset
``preparation_manifest.json``/``ground_truth_index.json`` must still hash to
what ``run_spec.json`` (and, transitively, ``evaluator_freeze.json``) recorded
at execution time. If the frozen protocol or dataset materialization has
since been mutated, this refuses to derive against a moving target rather
than silently applying today's config to yesterday's data.

See ``src/sgdjscc_lab/evaluators/negative_semantics.py::summarize_g1_v1_2``
for what changes vs. the frozen v1.1 ``summarize_g1``: effective-seed
deduplication (the decoder never consumes ``seed``, so 2025/2026/2027 are
bit-identical reconstructions) and a source-paired additional-object split
(reconstruction-introduced vs. carried over from an already-positive source
frame). Unlike the first version of this script, seed collapse is never
accepted on detector-score equality alone: by default this independently
hashes every reconstruction frame and cross-validates that hash-based
evidence against the score-based grouping before treating the decoder as
deterministic (see ``--declared-deterministic-decoder`` below).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sgdjscc_lab.evaluators.negative_semantics import (  # noqa: E402
    cross_validate_seed_evidence,
    effective_seed_groups,
    effective_seed_groups_from_hashes,
    summarize_g1_v1_2,
)
from sgdjscc_lab.utils.frame_hash import frame_tree_sha256  # noqa: E402


DEFAULT_RUN_ROOT = ROOT / "outputs/negative_semantics_g1_pilot_rtx4080_v1_1"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2"
G1_DATASET = ROOT / "data/negative_semantics/g1/pilot"
PROTOCOL_PATH = ROOT / "configs/experiments/negative_semantics/g1_protocol.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_protocol() -> Dict[str, Any]:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(PROTOCOL_PATH), resolve=True)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """Write CSV atomically (build in memory, then rename into place) with a
    forced LF line terminator -- csv.writer defaults to CRLF regardless of
    how the file is opened, which silently turns a git-tracked CSV into a
    CRLF file the moment any row is rewritten (see results/registry.csv).
    """
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())
    temporary.replace(path)


def _verify_source_integrity(run_root: Path) -> Dict[str, Any]:
    """Fail closed before reading any heavy artifact if the source run's
    provenance chain no longer matches what is currently on disk.
    """
    run_spec_path = run_root / "run_spec.json"
    if not run_spec_path.is_file():
        raise SystemExit(f"missing run_spec.json: {run_spec_path}")
    run_spec = _load_json(run_spec_path)
    if run_spec.get("smoke") is not False:
        raise SystemExit("refusing to derive v1.2 metrics from a smoke (non-evidence) run")

    if not PROTOCOL_PATH.is_file():
        raise SystemExit(f"missing frozen protocol file: {PROTOCOL_PATH}")
    current_protocol_sha256 = _sha256(PROTOCOL_PATH)
    recorded_protocol_sha256 = run_spec.get("protocol_sha256")
    if recorded_protocol_sha256 != current_protocol_sha256:
        raise SystemExit(
            "g1_protocol.yaml has changed since the v1.1 formal run: run_spec.json "
            f"recorded protocol_sha256={recorded_protocol_sha256!r}, the current file "
            f"hashes to {current_protocol_sha256!r}. Refusing to derive v1.2 metrics "
            "against a mutated frozen protocol -- re-freeze/version the protocol first."
        )

    dataset_manifest_path = G1_DATASET / "preparation_manifest.json"
    if not dataset_manifest_path.is_file():
        raise SystemExit(f"missing dataset preparation manifest: {dataset_manifest_path}")
    dataset_manifest = _load_json(dataset_manifest_path)
    current_manifest_sha256 = _sha256(dataset_manifest_path)
    recorded_manifest_sha256 = run_spec.get("preparation_manifest_sha256")
    if recorded_manifest_sha256 != current_manifest_sha256:
        raise SystemExit(
            "data/negative_semantics/g1/pilot/preparation_manifest.json has changed "
            f"since the v1.1 run (recorded {recorded_manifest_sha256!r}, current "
            f"{current_manifest_sha256!r}); refusing to derive against a changed "
            "dataset materialization."
        )

    gt_index_path = G1_DATASET / "ground_truth_index.json"
    if not gt_index_path.is_file():
        raise SystemExit(f"missing ground_truth_index.json: {gt_index_path}")
    current_gt_index_sha256 = _sha256(gt_index_path)
    recorded_gt_index_sha256 = dataset_manifest.get("g1_ground_truth_index_sha256")
    if recorded_gt_index_sha256 != current_gt_index_sha256:
        raise SystemExit(
            "ground_truth_index.json does not match the hash recorded in "
            f"preparation_manifest.json (recorded {recorded_gt_index_sha256!r}, "
            f"current {current_gt_index_sha256!r}); refusing to derive against "
            "changed ground truth."
        )

    for relative in ("evaluator/detection_rows.jsonl", "evaluator/evaluator_freeze.json"):
        path = run_root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"missing or empty required source artifact: {path}")

    evaluator_freeze = _load_json(run_root / "evaluator" / "evaluator_freeze.json")
    if evaluator_freeze.get("protocol_sha256") != current_protocol_sha256:
        raise SystemExit(
            "evaluator_freeze.json's own recorded protocol_sha256 does not match "
            "the current g1_protocol.yaml; evaluator calibration and the protocol "
            "file have drifted apart."
        )
    if evaluator_freeze.get("status") != "PASSED":
        raise SystemExit("source-only evaluator calibration did not pass in the source run; cannot derive")
    threshold = evaluator_freeze.get("selected_threshold")
    if threshold is None:
        raise SystemExit("evaluator_freeze.json has no selected_threshold")

    gt_index = _load_json(gt_index_path)
    missing_source_cache = [
        video_id for video_id in gt_index["videos"]
        if not (run_root / "evaluator" / "source" / f"{video_id}.json").is_file()
    ]
    if missing_source_cache:
        raise SystemExit(f"missing cached source-only OWLv2 scores for: {missing_source_cache}")

    return {
        "run_spec": run_spec,
        "evaluator_freeze": evaluator_freeze,
        "threshold": float(threshold),
        "gt_index": gt_index,
        "protocol_sha256": current_protocol_sha256,
        "dataset_manifest_sha256": current_manifest_sha256,
        "gt_index_sha256": current_gt_index_sha256,
    }


def _load_source_rows(run_root: Path, gt_index: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Reconstruct flat (video, frame, concept, score) rows from the cached
    per-video source-scoring files, the same cache `_calibrate` reads in
    `run_negative_semantics_g1.py`. gt_present is not needed downstream
    (only reconstruction rows carry gt_present in `raw_and_paired_h_add`),
    so this loader intentionally omits it.
    """
    rows: List[Dict[str, Any]] = []
    source_dir = run_root / "evaluator" / "source"
    for video_id in gt_index["videos"]:
        cached = _load_json(source_dir / f"{video_id}.json")
        for item in cached["scores"]:
            for concept, score in item["scores"].items():
                rows.append({
                    "video_id": video_id,
                    "frame_index": item["frame_index"],
                    "concept_id": concept,
                    "score": score,
                })
    return rows


def build_reconstruction_frame_hash_manifest(
    run_root: Path, gt_index: Mapping[str, Any], *,
    base_config: str, guide_profile: str, policies: Sequence[str], seeds: Sequence[int],
) -> Dict[str, Dict[int, str]]:
    """Independently hash every reconstruction frame per (policy, seed).

    This is the "separate evidence from score equality" item 5 requires: a
    detector-score match does not prove the underlying pixels are identical,
    so this reads and hashes the actual PNG files the score rows were
    computed from. Pure I/O + hashlib -- no GPU/torch needed.
    """
    config_name = f"{base_config}__{guide_profile}"
    manifest: Dict[str, Dict[int, str]] = {}
    for policy in policies:
        manifest[policy] = {}
        for seed in seeds:
            per_video_hashes = []
            for video_id in sorted(gt_index["videos"]):
                metadata = gt_index["videos"][video_id]
                frame_count = len(metadata["frame_names"])
                recon_dir = (
                    run_root / "reconstruction" / policy / f"seed_{seed}" /
                    "recon_videos" / video_id / config_name
                )
                paths = [recon_dir / f"frame_{index:05d}.png" for index in range(frame_count)]
                missing = [path for path in paths if not path.is_file()]
                if missing:
                    raise SystemExit(
                        f"missing {len(missing)} reconstruction frames while building the "
                        f"frame-hash manifest: {policy}/seed_{seed}/{video_id}"
                    )
                per_video_hashes.append(f"{video_id}:{frame_tree_sha256(paths)}")
            manifest[policy][seed] = hashlib.sha256(
                "\n".join(per_video_hashes).encode("utf-8")
            ).hexdigest()
    return manifest


def _resolve_seed_evidence(
    run_root: Path, gt_index: Mapping[str, Any], detection_rows: Sequence[Mapping[str, Any]],
    reconstruction_cfg: Mapping[str, Any], *, operator_declared: bool,
) -> Dict[str, Any]:
    """Decide whether seed collapse may be treated as deterministic, and on
    what grounds -- an explicit operator flag, or independent frame-hash
    verification. Raises (via cross_validate_seed_evidence or SystemExit)
    rather than defaulting to "trust it" when evidence disagrees or frames
    are missing.
    """
    score_groups = effective_seed_groups(detection_rows)
    if operator_declared:
        return {
            "declared_deterministic_decoder": True,
            "evidence_method": "operator_declared",
            "score_seed_groups": score_groups,
            "hash_seed_groups": None,
            "frame_hash_manifest": None,
        }
    seeds = [int(row["seed"]) for row in detection_rows]
    policies = sorted({str(row["policy"]) for row in detection_rows})
    frame_hash_manifest = build_reconstruction_frame_hash_manifest(
        run_root, gt_index,
        base_config=reconstruction_cfg["base_config"],
        guide_profile=reconstruction_cfg["guide_profile"],
        policies=policies, seeds=sorted(set(seeds)),
    )
    hash_groups = effective_seed_groups_from_hashes(frame_hash_manifest)
    cross_validate_seed_evidence(score_groups, hash_groups)
    return {
        "declared_deterministic_decoder": True,
        "evidence_method": "score_and_hash_dual_verified",
        "score_seed_groups": score_groups,
        "hash_seed_groups": hash_groups,
        "frame_hash_manifest": frame_hash_manifest,
    }


def _check_output_root_resume(output_root: Path, signature: str, *, allow_overwrite: bool) -> None:
    if not output_root.exists():
        return
    existing_entries = list(output_root.iterdir())
    if not existing_entries:
        return
    summary_path = output_root / "g1_summary_v1_2.json"
    if not summary_path.is_file():
        if allow_overwrite:
            return
        raise SystemExit(
            f"--output-root {output_root} is non-empty but has no g1_summary_v1_2.json "
            "to compare a resume signature against; pass --allow-overwrite to replace "
            "it deliberately."
        )
    existing = _load_json(summary_path)
    existing_signature = existing.get("derivation_signature")
    if existing_signature == signature:
        return  # Identical inputs/evidence/code path -- an idempotent resume.
    if allow_overwrite:
        return
    raise SystemExit(
        f"--output-root {output_root} already holds a v1.2 derivation with a "
        f"different derivation_signature ({existing_signature!r} vs new "
        f"{signature!r}). Refusing to silently overwrite a result that may be "
        "registered as frozen in results/registry.csv. If this is a deliberate "
        "re-derivation (e.g. after a methodology fix), first preserve the old "
        "result under a new versioned results/ path, then pass --allow-overwrite."
    )


def derive(
    run_root: Path, output_root: Path, *,
    declared_deterministic_decoder: bool, allow_overwrite: bool = False,
) -> Dict[str, Any]:
    run_root = run_root.resolve()
    output_root = output_root.resolve()
    if output_root == run_root or output_root.is_relative_to(run_root):
        raise SystemExit(
            "refusing to write the v1.2 derivation inside the frozen v1.1 run_root; "
            "pass a separate --output-root"
        )

    integrity = _verify_source_integrity(run_root)
    gt_index = integrity["gt_index"]
    threshold = integrity["threshold"]

    detection_rows_path = run_root / "evaluator" / "detection_rows.jsonl"
    detection_rows = _load_jsonl(detection_rows_path)
    source_rows = _load_source_rows(run_root, gt_index)

    protocol = _load_protocol()
    metric_cfg, gate_cfg, reconstruction_cfg = protocol["metrics"], protocol["gate"], protocol["reconstruction"]
    primary_policy = gate_cfg["primary_policy"]

    evidence = _resolve_seed_evidence(
        run_root, gt_index, detection_rows, reconstruction_cfg,
        operator_declared=declared_deterministic_decoder,
    )

    detection_rows_sha256 = _sha256(detection_rows_path)
    evaluator_freeze_sha256 = _sha256(run_root / "evaluator" / "evaluator_freeze.json")
    signature_payload = {
        "run_root": str(run_root),
        "source_protocol_sha256": integrity["protocol_sha256"],
        "source_dataset_manifest_sha256": integrity["dataset_manifest_sha256"],
        "source_gt_index_sha256": integrity["gt_index_sha256"],
        "source_detection_rows_sha256": detection_rows_sha256,
        "source_evaluator_freeze_sha256": evaluator_freeze_sha256,
        "declared_deterministic_decoder": evidence["declared_deterministic_decoder"],
        "evidence_method": evidence["evidence_method"],
        "frame_hash_manifest_sha256": (
            _sha256_json(evidence["frame_hash_manifest"]) if evidence["frame_hash_manifest"] else None
        ),
    }
    signature = _sha256_json(signature_payload)
    _check_output_root_resume(output_root, signature, allow_overwrite=allow_overwrite)

    report = summarize_g1_v1_2(
        detection_rows, gt_index["events"], source_rows,
        threshold=float(threshold), primary_policy=primary_policy,
        bootstrap_iterations=int(metric_cfg["statistics"]["bootstrap_iterations"]),
        ghost_horizon=int(metric_cfg["ghost_survival_auc"]["horizon_annotated_timesteps"]),
        declared_deterministic_decoder=evidence["declared_deterministic_decoder"],
        gate_cfg=gate_cfg,
    )
    report.update({
        "protocol_id": "negative_semantics_g1_v1_2",
        "amendment_record": "docs/experiments/2026-09-06_negative_semantics_g1_v1_2_amendment.md",
        "source_run_root": str(run_root),
        "source_run_git_commit": integrity["run_spec"].get("git_commit"),
        "source_protocol_sha256": integrity["protocol_sha256"],
        "source_dataset_manifest_sha256": integrity["dataset_manifest_sha256"],
        "source_gt_index_sha256": integrity["gt_index_sha256"],
        "source_detection_rows_sha256": detection_rows_sha256,
        "source_detection_rows_count": len(detection_rows),
        "source_evaluator_freeze_sha256": evaluator_freeze_sha256,
        "seed_evidence_method": evidence["evidence_method"],
        "seed_evidence_hash_groups": evidence["hash_seed_groups"],
        "reconstruction_frame_hash_manifest": evidence["frame_hash_manifest"],
        "derivation_signature": signature,
        "heldout_accessed": False,
        "read_only_derivation": True,
    })

    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / "g1_summary_v1_2.json", report)

    fields = [
        "policy", "declared_seed_count", "effective_seed_count",
        "raw_h_add", "raw_additional_detections", "raw_additional_event_count",
        "effective_h_add", "effective_additional_detections", "effective_additional_event_count",
        "source_paired_h_add_effective", "source_paired_additional_detections_effective",
        "source_paired_opportunities_effective",
    ]
    csv_rows = []
    for policy in sorted(report["seed_groups"]):
        seed_info = report["seed_groups"][policy]
        raw = report["raw_v1_1_view"]["by_policy"][policy]
        effective = report["effective_seed_view"]["by_policy"][policy]
        paired_effective = report["source_paired_additional_object"]["effective_seeds"][policy]
        csv_rows.append({
            "policy": policy,
            "declared_seed_count": seed_info["declared_seed_count"],
            "effective_seed_count": seed_info["effective_seed_count"],
            "raw_h_add": raw["h_add"],
            "raw_additional_detections": raw["additional_detections"],
            "raw_additional_event_count": raw["additional_event_count"],
            "effective_h_add": effective["h_add"],
            "effective_additional_detections": effective["additional_detections"],
            "effective_additional_event_count": effective["additional_event_count"],
            "source_paired_h_add_effective": paired_effective["source_paired_h_add"],
            "source_paired_additional_detections_effective": paired_effective["source_paired_additional_detections"],
            "source_paired_opportunities_effective": paired_effective["source_paired_opportunities"],
        })
    _atomic_csv(output_root / "g1_policy_summary_v1_2.csv", fields, csv_rows)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--declared-deterministic-decoder", action="store_true", default=False,
        help=(
            "Operator override: treat any seed collapse found in the detection "
            "rows as deterministic-decoder-caused WITHOUT independently hashing "
            "reconstruction frames to verify it. Default is False -- by default "
            "this script hashes every reconstruction frame and cross-validates "
            "that against the detector-score-based grouping "
            "(cross_validate_seed_evidence), refusing to proceed if they disagree "
            "or frames are missing. Pass this flag only when the frame trees are "
            "not available to hash (e.g. a stripped-down checkout) and you accept "
            "the score-only evidence on its own."
        ),
    )
    parser.add_argument(
        "--allow-overwrite", action="store_true", default=False,
        help=(
            "Allow overwriting a non-empty --output-root whose existing "
            "g1_summary_v1_2.json has a different derivation_signature than this "
            "run would produce. Without this, a signature mismatch aborts instead "
            "of silently replacing a possibly-frozen prior derivation."
        ),
    )
    args = parser.parse_args(argv)
    report = derive(
        args.run_root, args.output_root,
        declared_deterministic_decoder=args.declared_deterministic_decoder,
        allow_overwrite=args.allow_overwrite,
    )
    print(json.dumps(
        {
            "seed_groups": report["seed_groups"],
            "seed_evidence_method": report["seed_evidence_method"],
            "gate_reapplication": report.get("gate_reapplication"),
            "derivation_signature": report["derivation_signature"],
            "output_root": str(args.output_root.resolve()),
        },
        indent=2, sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

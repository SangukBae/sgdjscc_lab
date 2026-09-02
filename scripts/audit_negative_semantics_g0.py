#!/usr/bin/env python3
"""Dependency-free audit for the frozen negative-semantics G0 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SPLITS = ("pilot", "train", "development", "validation", "heldout_test")
EVENT_TYPES = {"ENTER", "EXIT", "OCCLUDE", "REAPPEAR", "SCENE_CUT"}
BASE = Path("data/negative_semantics/g0")
MANIFEST = BASE / "dataset_split_manifest.json"
MANIFEST_HASH = BASE / "dataset_split_manifest.sha256"
ONTOLOGY = BASE / "ontology_manifest.json"
EVENT_SCHEMA = BASE / "event_schema.schema.json"
PREREGISTRATION = Path(
    "configs/experiments/negative_semantics/g0_preregistered_gate.json"
)
PROTOCOL = Path("configs/experiments/negative_semantics/g0_protocol.yaml")
ENVIRONMENT = Path("environments/sgdjscc-py39-cu118.yml")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{number}")
            rows.append(value)
    return rows


def _check(check_id: str, passed: bool, **evidence: Any) -> Dict[str, Any]:
    return {"check_id": check_id, "passed": bool(passed), "evidence": evidence}


def _assigned_entries(manifest: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for split in SPLITS:
        for entry in manifest["splits"][split]:
            yield split, entry


def audit(repo_root: Path) -> Dict[str, Any]:
    root = repo_root.resolve()
    manifest = load_json(root / MANIFEST)
    prereg = load_json(root / PREREGISTRATION)
    ontology = load_json(root / ONTOLOGY)
    schema = load_json(root / EVENT_SCHEMA)
    protocol_text = (root / PROTOCOL).read_text(encoding="utf-8")
    expected_protocol_hash = prereg["frozen_artifacts"]["protocol_sha256"]
    actual_protocol_hash = sha256_file(root / PROTOCOL)

    checks: List[Dict[str, Any]] = []
    checks.append(_check(
        "protocol_frozen",
        "freeze_status: frozen" in protocol_text
        and "protocol_version: 1.0.0" in protocol_text
        and actual_protocol_hash == expected_protocol_hash,
        protocol=str(PROTOCOL),
        actual_sha256=actual_protocol_hash,
        preregistered_sha256=expected_protocol_hash,
    ))

    split_shape_ok = (
        manifest.get("split_lists_frozen") is True
        and isinstance(manifest.get("splits"), dict)
        and all(isinstance(manifest["splits"].get(split), list) for split in SPLITS)
    )
    checks.append(_check(
        "five_split_lists_frozen", split_shape_ok, splits=list(SPLITS)
    ))

    detached_tokens = (root / MANIFEST_HASH).read_text(encoding="utf-8").split()
    actual_manifest_hash = sha256_file(root / MANIFEST)
    detached_hash = detached_tokens[0] if detached_tokens else ""
    registered_hash = prereg["frozen_artifacts"]["split_manifest_sha256"]
    checks.append(_check(
        "manifest_detached_hash_matches",
        detached_hash == actual_manifest_hash == registered_hash,
        actual_sha256=actual_manifest_hash,
        detached_sha256=detached_hash,
        preregistered_sha256=registered_hash,
    ))

    hash_failures: List[Dict[str, str]] = []
    assigned = list(_assigned_entries(manifest))
    for split, entry in assigned:
        for path_key, hash_key in (("input_path", "input_sha256"), ("gt_path", "gt_sha256")):
            relative = entry.get(path_key)
            expected = entry.get(hash_key)
            if not relative or not expected:
                hash_failures.append({
                    "split": split,
                    "video_id": str(entry.get("video_id")),
                    "path": str(relative),
                    "reason": f"missing {path_key} or {hash_key}",
                })
                continue
            path = root / relative
            if not path.is_file():
                hash_failures.append({
                    "split": split,
                    "video_id": str(entry.get("video_id")),
                    "path": relative,
                    "reason": "file missing",
                })
            elif sha256_file(path) != expected:
                hash_failures.append({
                    "split": split,
                    "video_id": str(entry.get("video_id")),
                    "path": relative,
                    "reason": "sha256 mismatch",
                })
    checks.append(_check(
        "assigned_file_hashes_match",
        not hash_failures,
        assigned_video_count=len(assigned),
        failures=hash_failures,
    ))

    seen_video: Dict[str, str] = {}
    duplicate_video: List[Dict[str, str]] = []
    for split, entry in assigned:
        video_id = str(entry.get("video_id"))
        if video_id in seen_video:
            duplicate_video.append({
                "video_id": video_id,
                "first_split": seen_video[video_id],
                "second_split": split,
            })
        seen_video[video_id] = split
    checks.append(_check(
        "split_video_ids_disjoint", not duplicate_video, duplicates=duplicate_video
    ))

    annotation_hash_failures: List[str] = []
    event_rows: List[Dict[str, Any]] = []
    for item in prereg["frozen_artifacts"]["event_annotations"]:
        path = root / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            annotation_hash_failures.append(item["path"])
            continue
        event_rows.extend(load_jsonl(path))

    states = set(ontology.get("state_values", []))
    concepts = {item.get("concept_id") for item in ontology.get("concepts", [])}
    schema_types = set(schema.get("properties", {}).get("event_type", {}).get("enum", []))
    required_event_fields = set(schema.get("required", []))
    assigned_video_keys = {
        (split, entry.get("source_id"), entry.get("video_id"))
        for split, entry in assigned
    }
    event_errors: List[str] = []
    seen_event_ids = set()
    for index, row in enumerate(event_rows, start=1):
        missing = sorted(required_event_fields - set(row))
        if missing:
            event_errors.append(f"row {index}: missing fields {missing}")
        event_id = row.get("event_id")
        if event_id in seen_event_ids:
            event_errors.append(f"row {index}: duplicate event_id {event_id}")
        seen_event_ids.add(event_id)
        if row.get("event_type") not in EVENT_TYPES:
            event_errors.append(f"row {index}: invalid event_type")
        if row.get("state_before") not in states or row.get("state_after") not in states:
            event_errors.append(f"row {index}: invalid state")
        if row.get("event_type") != "SCENE_CUT" and row.get("concept_id") not in concepts:
            event_errors.append(f"row {index}: concept_id is not in ontology")
        if not isinstance(row.get("start_frame"), int) or not isinstance(row.get("end_frame"), int):
            event_errors.append(f"row {index}: frame boundaries must be integers")
        elif row["end_frame"] < row["start_frame"]:
            event_errors.append(f"row {index}: end_frame precedes start_frame")
        key = (row.get("split"), row.get("source_id"), row.get("video_id"))
        if key not in assigned_video_keys:
            event_errors.append(f"row {index}: event does not map to an assigned video")
        verification = row.get("human_verification", {})
        if row.get("human_verified") is True and verification.get("status") not in {"agreed", "adjudicated"}:
            event_errors.append(f"row {index}: human_verified conflicts with verification status")

    actual_ontology_hash = sha256_file(root / ONTOLOGY)
    actual_schema_hash = sha256_file(root / EVENT_SCHEMA)
    contract_ok = (
        states == {"present", "confirmed_absent", "unknown"}
        and schema_types == EVENT_TYPES
        and actual_ontology_hash == prereg["frozen_artifacts"]["ontology_sha256"]
        and actual_schema_hash == prereg["frozen_artifacts"]["event_schema_sha256"]
        and not annotation_hash_failures
        and not event_errors
    )
    checks.append(_check(
        "event_and_ontology_contract_present",
        contract_ok,
        ontology_states=sorted(states),
        event_types=sorted(schema_types),
        ontology_sha256=actual_ontology_hash,
        event_schema_sha256=actual_schema_hash,
        event_row_count=len(event_rows),
        event_errors=event_errors,
        annotation_hash_failures=annotation_hash_failures,
    ))

    qualifying = [
        row for row in event_rows
        if row.get("split") == "pilot"
        and row.get("event_type") in EVENT_TYPES
        and row.get("human_verified") is True
        and row.get("human_verification", {}).get("status") in {"agreed", "adjudicated"}
    ]
    pilot_clusters = sorted({row.get("independence_cluster_id") for row in qualifying})
    min_events = prereg["frozen_thresholds"]["pilot_independent_events_min"]
    max_events = prereg["frozen_thresholds"]["pilot_independent_events_max"]
    checks.append(_check(
        "pilot_has_30_to_50_human_verified_independent_events",
        not annotation_hash_failures and min_events <= len(pilot_clusters) <= max_events,
        human_verified_event_rows=len(qualifying),
        independent_event_clusters=len(pilot_clusters),
        required_range=[min_events, max_events],
        annotation_hash_failures=annotation_hash_failures,
    ))

    heldout = manifest["splits"]["heldout_test"]
    heldout_sources = {entry.get("source_id") for entry in heldout}
    earlier_sources = {
        entry.get("source_id")
        for split in SPLITS[:-1]
        for entry in manifest["splits"][split]
    }
    legacy_source = manifest["leakage_policy"]["legacy_source_id"]
    heldout_ok = bool(heldout) and not (
        heldout_sources & earlier_sources
    ) and legacy_source not in heldout_sources
    checks.append(_check(
        "heldout_is_nonempty_and_source_disjoint",
        heldout_ok,
        heldout_video_count=len(heldout),
        heldout_sources=sorted(str(item) for item in heldout_sources),
        conflicting_sources=sorted(str(item) for item in heldout_sources & earlier_sources),
    ))

    source_status = {
        item["source_id"]: item.get("license_status")
        for item in manifest.get("source_audit", [])
    }
    assigned_sources = sorted({str(entry.get("source_id")) for _, entry in assigned})
    unapproved = {
        source: source_status.get(source, "missing")
        for source in assigned_sources
        if source_status.get(source) != "approved"
    }
    checks.append(_check(
        "assigned_source_licenses_approved",
        bool(assigned_sources) and not unapproved,
        assigned_sources=assigned_sources,
        unapproved=unapproved,
    ))

    expected_environment_hash = prereg["frozen_artifacts"]["environment_sha256"]
    actual_environment_hash = (
        sha256_file(root / ENVIRONMENT) if (root / ENVIRONMENT).is_file() else None
    )
    reproducible = (
        actual_environment_hash == expected_environment_hash
        and not hash_failures
        and not annotation_hash_failures
        and all(entry.get("input_path") and entry.get("input_sha256") for _, entry in assigned)
    )
    checks.append(_check(
        "reproducible_acquisition_and_environment_recorded",
        reproducible,
        environment=str(ENVIRONMENT),
        actual_environment_sha256=actual_environment_hash,
        preregistered_environment_sha256=expected_environment_hash,
        assigned_video_count=len(assigned),
    ))

    required_ids = set(prereg["required_checks"])
    result_by_id = {item["check_id"]: item for item in checks}
    missing_required = sorted(required_ids - set(result_by_id))
    passed = not missing_required and all(result_by_id[item]["passed"] for item in required_ids)
    return {
        "gate_id": prereg["gate_id"],
        "protocol_version": prereg["protocol_version"],
        "audit_completed": True,
        "gate_status": "PASSED" if passed else "NOT_PASSED",
        "eligible_to_start_g1": passed,
        "required_check_count": len(required_ids),
        "passed_check_count": sum(
            1 for item in required_ids if result_by_id.get(item, {}).get("passed")
        ),
        "missing_required_checks": missing_required,
        "checks": checks,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", help="Optional path for the JSON audit report")
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit 4 unless every preregistered G0 condition passes",
    )
    args = parser.parse_args(argv)
    report = audit(Path(args.repo_root))
    rendered = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 4 if args.require_pass and report["gate_status"] != "PASSED" else 0


if __name__ == "__main__":
    raise SystemExit(main())

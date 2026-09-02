from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
REQUIRED_EXTERNAL_ARCHIVES = [
    ROOT / "data/external/youtube_vos_2019/archives/train.tar",
    ROOT / "data/external/youtube_vos_2019/archives/valid.tar",
    ROOT / "data/external/davis_2017/archives/DAVIS-2017-trainval-480p.zip",
]


def load_auditor():
    spec = importlib.util.spec_from_file_location(
        "_negative_semantics_g0_audit_test",
        ROOT / "scripts/audit_negative_semantics_g0.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not all(path.is_file() for path in REQUIRED_EXTERNAL_ARCHIVES),
    reason="official G0 external archives are intentionally git-ignored",
)
def test_frozen_g0_artifacts_are_valid_but_waiting_for_human_review():
    module = load_auditor()
    report = module.audit(ROOT)
    by_id = {item["check_id"]: item for item in report["checks"]}

    assert report["audit_completed"] is True
    assert report["gate_status"] == "NOT_PASSED"
    assert report["eligible_to_start_g1"] is False
    assert report["passed_check_count"] == 12
    assert report["required_check_count"] == 13
    assert by_id["protocol_frozen"]["passed"] is True
    assert by_id["manifest_detached_hash_matches"]["passed"] is True
    assert by_id["assigned_file_hashes_match"]["passed"] is True
    assert by_id["acquisition_archives_match"]["passed"] is True
    assert by_id[
        "pilot_has_30_to_50_human_verified_independent_events"
    ]["passed"] is False
    assert by_id["heldout_is_nonempty_and_source_disjoint"]["passed"] is True
    assert by_id["heldout_seal_matches"]["passed"] is True
    assert by_id["assigned_source_licenses_approved"]["passed"] is True
    assert by_id["data_use_approval_recorded"]["passed"] is True


def test_g0_json_and_jsonl_files_parse_without_optional_dependencies():
    paths = [
        ROOT / "configs/experiments/negative_semantics/g0_preregistered_gate.json",
        ROOT / "data/negative_semantics/g0/dataset_split_manifest.json",
        ROOT / "data/negative_semantics/g0/ontology_manifest.json",
        ROOT / "data/negative_semantics/g0/event_schema.schema.json",
        ROOT / "data/negative_semantics/g0/youtube_vos_category_mapping.json",
        ROOT / "data/negative_semantics/g0/data_use_approval.json",
        ROOT / "data/negative_semantics/g0/heldout_seal.json",
    ]
    for path in paths:
        assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)

    event_path = ROOT / "data/negative_semantics/g0/etri_legacy_event_annotations.jsonl"
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 5
    assert all(event["human_verified"] is False for event in events)

    candidate_path = (
        ROOT / "data/negative_semantics/g0/youtube_vos_pilot_event_candidates.jsonl"
    )
    candidates = [
        json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(candidates) == 33
    assert len({event["independence_cluster_id"] for event in candidates}) == 33
    assert sum(event["event_type"] == "ENTER" for event in candidates) == 17
    assert sum(event["event_type"] == "EXIT" for event in candidates) == 16
    assert all(event["human_verified"] is False for event in candidates)

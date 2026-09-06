"""Tests for scripts/audit_negative_semantics_g1.py's runner-vs-scientific
gate split (item 8). Uses a small synthetic run_root -- no dependency on the
real frozen outputs/negative_semantics_g1_pilot_rtx4080_v1_1 tree.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_test_{name}", ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _build_passing_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "run_root"
    video_ids = [f"ovis_valid_{i:02d}" for i in range(40)]
    _write_json(run_root / "run_spec.json", {
        "smoke": False, "heldout_accessed": False, "video_ids": video_ids,
        "policies": {"few10": 10, "full50": 50}, "seeds": [2025, 2026, 2027],
    })
    for policy in ("few10", "full50"):
        for seed in (2025, 2026, 2027):
            _write_json(
                run_root / "reconstruction" / policy / f"seed_{seed}" / "summary.json",
                {"n_videos": 40, "n_failed_pairs": 0, "run_status": "completed"},
            )
    detection_path = run_root / "evaluator" / "detection_rows.jsonl"
    detection_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for concept in ("person", "bird", "cat", "dog", "bicycle", "motorcycle", "vehicle"):
        for policy in ("few10", "full50"):
            for seed in (2025, 2026, 2027):
                rows.append({
                    "video_id": video_ids[0], "seed": seed, "policy": policy,
                    "concept_id": concept, "frame_index": 0, "score": 0.9, "gt_present": False,
                })
    with detection_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    _write_json(run_root / "g1_summary.json", {
        "evidence_scope": "OVIS_PILOT_G1", "heldout_accessed": False, "gate_status": "PASSED",
        "gate_checks": {"formal_not_smoke": True, "h_add_prevalence": True},
    })
    return run_root


def test_scientific_gate_reports_unknown_when_v1_2_not_derived(tmp_path):
    module = _load_script("audit_negative_semantics_g1.py")
    run_root = _build_passing_run_root(tmp_path)
    report = module.audit(run_root, v1_2_derived_summary_path=tmp_path / "does_not_exist.json")
    assert report["runner_gate_status"] == "PASSED"
    assert report["scientific_gate_status"] == "UNKNOWN_V1_2_NOT_DERIVED"


def test_scientific_gate_reads_effective_seed_status_from_derived_summary(tmp_path):
    module = _load_script("audit_negative_semantics_g1.py")
    run_root = _build_passing_run_root(tmp_path)
    derived_path = tmp_path / "derived" / "g1_summary_v1_2.json"
    _write_json(derived_path, {
        "gate_reapplication": {
            "effective_seed_status": "NOT_PASSED",
            "checks_that_flip_to_fail_under_effective_seeds": ["affected_seed_count"],
        },
        "seed_evidence_method": "score_and_hash_dual_verified",
    })
    report = module.audit(run_root, v1_2_derived_summary_path=derived_path)
    assert report["scientific_gate_status"] == "NOT_PASSED"
    assert report["seed_evidence_method"] == "score_and_hash_dual_verified"
    assert report["checks_that_flip_to_fail_under_effective_seeds"] == ["affected_seed_count"]


def test_require_runner_complete_flag_gates_only_on_runner_status(tmp_path, capsys):
    module = _load_script("audit_negative_semantics_g1.py")
    run_root = _build_passing_run_root(tmp_path)
    exit_code = module.main([
        "--run-root", str(run_root),
        "--v1-2-derived-summary", str(tmp_path / "missing.json"),
        "--require-runner-complete",
    ])
    assert exit_code == 0  # runner passed even though scientific gate is unknown


def test_require_scientific_gate_flag_fails_closed_when_not_derived(tmp_path):
    module = _load_script("audit_negative_semantics_g1.py")
    run_root = _build_passing_run_root(tmp_path)
    exit_code = module.main([
        "--run-root", str(run_root),
        "--v1-2-derived-summary", str(tmp_path / "missing.json"),
        "--require-scientific-gate",
    ])
    assert exit_code != 0


def test_deprecated_require_pass_behaves_like_require_runner_complete(tmp_path, capsys):
    module = _load_script("audit_negative_semantics_g1.py")
    run_root = _build_passing_run_root(tmp_path)
    exit_code = module.main([
        "--run-root", str(run_root),
        "--v1-2-derived-summary", str(tmp_path / "missing.json"),
        "--require-pass",
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "deprecated" in captured.err.lower()

#!/usr/bin/env python3
"""Read-only audit of a completed SV0/G2 Oracle ABSENT run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sgdjscc_lab.guidance.negative_conditioning import (  # noqa: E402
    G2_ARMS,
    validate_g2_condition_manifest,
)

PROTOCOL_PATH = ROOT / "configs/experiments/negative_semantics/g2_oracle_absent_protocol.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _check(check_id: str, passed: bool, **evidence: Any) -> Dict[str, Any]:
    return {"check_id": check_id, "passed": bool(passed), "evidence": evidence}


def audit(run_root: Path) -> Dict[str, Any]:
    from omegaconf import OmegaConf

    run_root = run_root.resolve()
    protocol = OmegaConf.to_container(OmegaConf.load(PROTOCOL_PATH), resolve=True)
    run_spec = _json(run_root / "run_spec.json")
    summary = _json(run_root / "g2_summary.json")
    matched = _json(run_root / "matched_compute_audit.json")
    checks: List[Dict[str, Any]] = []

    checks.append(_check(
        "protocol_hash_matches",
        run_spec.get("protocol_sha256") == _sha256(PROTOCOL_PATH),
        recorded=run_spec.get("protocol_sha256"), current=_sha256(PROTOCOL_PATH),
    ))
    checks.append(_check(
        "formal_pilot_not_smoke",
        run_spec.get("smoke") is False and len(run_spec.get("video_ids") or []) == 40,
        smoke=run_spec.get("smoke"), video_count=len(run_spec.get("video_ids") or []),
    ))
    checks.append(_check(
        "heldout_not_accessed",
        run_spec.get("heldout_accessed") is False and summary.get("heldout_accessed") is False,
        run_spec=run_spec.get("heldout_accessed"), summary=summary.get("heldout_accessed"),
    ))
    checks.append(_check(
        "four_arms_and_fixed_matrix",
        run_spec.get("arms") == list(G2_ARMS)
        and run_spec.get("policies") == {"few10": 10, "full50": 50}
        and run_spec.get("seeds") == [2025],
        arms=run_spec.get("arms"), policies=run_spec.get("policies"), seeds=run_spec.get("seeds"),
    ))
    checks.append(_check(
        "oracle_is_out_of_band_not_rate_accounted",
        run_spec.get("oracle_side_input") == "ORACLE_EVAL_ONLY"
        and run_spec.get("oracle_rate_accounted") is False
        and summary.get("oracle_rate_accounted") is False,
        oracle_side_input=run_spec.get("oracle_side_input"),
        oracle_rate_accounted=summary.get("oracle_rate_accounted"),
    ))

    gt_index = _json(ROOT / protocol["data"]["ground_truth_index"])
    expected_frames = {
        video_id: len(gt_index["videos"][video_id]["frame_names"])
        for video_id in run_spec.get("video_ids") or []
    }
    condition_results = {}
    for arm in G2_ARMS:
        path = run_root / "conditions" / f"{arm}.json"
        try:
            manifest = _json(path)
            rows = validate_g2_condition_manifest(manifest, expected_frames)
            valid = run_spec.get("condition_sha256", {}).get(arm) == _sha256(path)
            condition_results[arm] = {
                "valid": valid, "video_count": len(rows), "sha256": _sha256(path),
            }
        except Exception as error:  # noqa: BLE001 - audit records a failed check
            condition_results[arm] = {"valid": False, "error": str(error)}
    checks.append(_check(
        "condition_manifests_complete_and_hashed",
        all(value["valid"] for value in condition_results.values()),
        conditions=condition_results,
    ))

    child_count = completed_videos = failed_pairs = 0
    child_problems = []
    for arm in G2_ARMS:
        for policy in run_spec.get("policies") or {}:
            for seed in run_spec.get("seeds") or []:
                child = run_root / "reconstruction" / arm / policy / f"seed_{seed}"
                try:
                    child_summary = _json(child / "summary.json")
                    child_count += 1
                    completed_videos += int(child_summary.get("n_videos", 0))
                    failed_pairs += int(child_summary.get("n_failed_pairs", 0))
                    if child_summary.get("run_status") != "completed":
                        child_problems.append(f"{arm}/{policy}/seed_{seed}: {child_summary.get('run_status')}")
                except Exception as error:  # noqa: BLE001
                    child_problems.append(f"{arm}/{policy}/seed_{seed}: {error}")
    expected_children = len(G2_ARMS) * len(run_spec.get("policies") or {}) * len(run_spec.get("seeds") or [])
    expected_video_runs = expected_children * len(run_spec.get("video_ids") or [])
    checks.append(_check(
        "child_runs_complete_zero_failures",
        child_count == expected_children and completed_videos == expected_video_runs
        and failed_pairs == 0 and not child_problems,
        child_count=child_count, expected_children=expected_children,
        completed_videos=completed_videos, expected_video_runs=expected_video_runs,
        failed_pairs=failed_pairs, problems=child_problems,
    ))

    checks.append(_check(
        "matched_compute_and_rate_passed",
        matched.get("status") == "PASSED" and int(matched.get("mismatch_count", -1)) == 0,
        status=matched.get("status"), mismatch_count=matched.get("mismatch_count"),
        matched_fields=matched.get("matched_fields"),
    ))
    detection_rows = [
        json.loads(line) for line in
        (run_root / "evaluator/detection_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    concepts = set(protocol["evaluator"]["concepts"])
    expected_detection_count = (
        sum(expected_frames.values()) * len(concepts) * len(G2_ARMS)
        * len(run_spec.get("policies") or {}) * len(run_spec.get("seeds") or [])
    )
    detection_keys = {
        (row["arm"], row["policy"], int(row["seed"]), row["video_id"],
         int(row["frame_index"]), row["concept_id"])
        for row in detection_rows
    }
    checks.append(_check(
        "detection_grid_complete_unique",
        len(detection_rows) == expected_detection_count
        and len(detection_keys) == expected_detection_count
        and {row["concept_id"] for row in detection_rows} == concepts,
        rows=len(detection_rows), expected=expected_detection_count,
        unique_keys=len(detection_keys),
    ))
    accounting = _csv(run_root / "accounting_rows.csv")
    expected_accounting = (
        len(run_spec.get("video_ids") or []) * len(G2_ARMS)
        * len(run_spec.get("policies") or {}) * len(run_spec.get("seeds") or [])
    )
    checks.append(_check(
        "accounting_grid_complete",
        len(accounting) == expected_accounting
        and {row["arm"] for row in accounting} == set(G2_ARMS)
        and {row["bit_depth"] for row in accounting} == {"4"},
        rows=len(accounting), expected=expected_accounting,
        bit_depths=sorted({row["bit_depth"] for row in accounting}),
    ))

    checksum_report = _json(run_root / "artifact_checksums.json")
    checksum_failures = []
    for relative, expected_hash in checksum_report.items():
        path = run_root / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            checksum_failures.append(relative)
    checks.append(_check(
        "recorded_artifact_checksums_match", not checksum_failures,
        checked=len(checksum_report), failures=checksum_failures,
    ))

    required = {item["check_id"] for item in checks}
    runner_passed = all(item["passed"] for item in checks if item["check_id"] in required)
    provisional = summary.get("provisional_gate_status", "UNKNOWN")
    return {
        "audit_id": "negative_semantics_g2_oracle_absent_v1_0_audit",
        "run_root": str(run_root),
        "runner_gate_status": "PASSED" if runner_passed else "NOT_PASSED",
        "provisional_mechanism_gate_status": provisional,
        "scientific_gate_status": summary.get("gate_status", "UNKNOWN"),
        "g1_scientific_gate_status": summary.get("g1_scientific_gate_status", "UNKNOWN"),
        "checks": checks,
        "heldout_accessed": False,
    }


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--require-runner-complete", action="store_true")
    parser.add_argument("--require-provisional-gate", action="store_true")
    args = parser.parse_args(argv)
    report = audit(args.run_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_runner_complete and report["runner_gate_status"] != "PASSED":
        return 2
    if args.require_provisional_gate and report["provisional_mechanism_gate_status"] != "PASSED":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

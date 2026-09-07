#!/usr/bin/env python3
"""Read-only audit for prepared or evaluated G3 Oracle REVOKE artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sgdjscc_lab.guidance.saver_receiver_state import (  # noqa: E402
    G3_RECEIVER_ARMS,
    validate_g3_receiver_condition_manifest,
)


PROTOCOL_PATH = ROOT / "configs/experiments/negative_semantics/g3_oracle_revoke_protocol.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _check(check_id, passed, **evidence):
    return {"check_id": check_id, "passed": bool(passed), "evidence": evidence}


def audit(run_root: Path) -> dict:
    from omegaconf import OmegaConf
    run_root = run_root.resolve()
    protocol = OmegaConf.to_container(OmegaConf.load(PROTOCOL_PATH), resolve=True)
    spec = _json(run_root / "run_spec.json")
    checks = [
        _check(
            "protocol_hash_matches",
            spec.get("protocol_sha256") == _sha256(PROTOCOL_PATH),
            recorded=spec.get("protocol_sha256"), current=_sha256(PROTOCOL_PATH),
        ),
        _check(
            "heldout_not_accessed",
            spec.get("heldout_accessed") is False,
            heldout_accessed=spec.get("heldout_accessed"),
        ),
        _check(
            "oracle_not_rate_accounted",
            spec.get("oracle_side_input") == "ORACLE_EVAL_ONLY"
            and spec.get("serialized_in_packet") is False
            and spec.get("rate_accounted") is False,
            oracle_side_input=spec.get("oracle_side_input"),
            serialized_in_packet=spec.get("serialized_in_packet"),
            rate_accounted=spec.get("rate_accounted"),
        ),
        _check(
            "three_arm_contract",
            spec.get("arms") == list(G3_RECEIVER_ARMS),
            arms=spec.get("arms"),
        ),
    ]
    video_ids = spec.get("video_ids") or []
    expected = {}
    manifest_checks = {}
    for arm in G3_RECEIVER_ARMS:
        path = run_root / "conditions" / f"{arm}.json"
        try:
            manifest = _json(path)
            if not expected:
                expected = {
                    video_id: int(manifest["videos"][video_id]["frame_count"])
                    for video_id in video_ids
                }
            validate_g3_receiver_condition_manifest(manifest, expected)
            actual = _sha256(path)
            manifest_checks[arm] = {
                "passed": actual == (spec.get("condition_sha256") or {}).get(arm),
                "sha256": actual,
            }
        except Exception as exc:  # noqa: BLE001
            manifest_checks[arm] = {"passed": False, "error": str(exc)}
    checks.append(_check(
        "condition_manifests_valid_and_hashed",
        all(value["passed"] for value in manifest_checks.values()),
        arms=manifest_checks,
    ))
    is_smoke = spec.get("smoke") is True
    checks.append(_check(
        "formal_video_grid",
        is_smoke or len(video_ids) == int(protocol["data"]["expected_videos"]),
        smoke=is_smoke, video_count=len(video_ids),
    ))

    summary_path = run_root / "g3_summary.json"
    if summary_path.is_file():
        summary = _json(summary_path)
        rows_path = run_root / "evaluator" / "detection_rows.jsonl"
        evaluation_ok = (
            rows_path.is_file()
            and summary.get("protocol_sha256") == _sha256(PROTOCOL_PATH)
            and summary.get("detection_rows_sha256") == _sha256(rows_path)
            and summary.get("heldout_accessed") is False
        )
        checks.append(_check(
            "evaluation_summary_bound_to_rows", evaluation_ok,
            summary_path=str(summary_path), detection_rows_path=str(rows_path),
        ))
        provisional = summary.get("provisional_gate_status", "UNKNOWN")
    else:
        provisional = "NOT_RUN"
    preparation_complete = all(check["passed"] for check in checks)
    protocol_frozen = protocol.get("freeze_status") == "frozen"
    return {
        "audit_id": "negative_semantics_g3_oracle_revoke_v0_1_audit",
        "run_root": str(run_root),
        "preparation_status": "PASSED" if preparation_complete else "NOT_PASSED",
        "protocol_freeze_status": protocol.get("freeze_status"),
        "formal_evidence_status": (
            provisional if preparation_complete and protocol_frozen else "NOT_ELIGIBLE"
        ),
        "provisional_gate_status": provisional,
        "checks": checks,
    }


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--require-prepared", action="store_true")
    parser.add_argument("--require-provisional-gate", action="store_true")
    args = parser.parse_args(argv)
    report = audit(args.run_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_prepared and report["preparation_status"] != "PASSED":
        return 2
    if args.require_provisional_gate and report["formal_evidence_status"] != "PASSED":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

#!/usr/bin/env python3
"""Compile two independent Pilot reviews and optional adjudication into events."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


EVENT_TYPES = {"ENTER", "EXIT", "OCCLUDE", "REAPPEAR", "SCENE_CUT"}
ACCEPT = "accept"
REJECT = "reject"
UNCERTAIN = "uncertain"


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


def load_csv(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: Dict[str, Dict[str, str]] = {}
    for number, row in enumerate(rows, start=2):
        event_id = (row.get("event_id") or "").strip()
        if not event_id:
            raise ValueError(f"missing event_id at {path}:{number}")
        if event_id in result:
            raise ValueError(f"duplicate event_id {event_id} in {path}")
        result[event_id] = row
    return result


def text(row: Dict[str, str], key: str) -> str:
    return (row.get(key) or "").strip()


def yes(row: Dict[str, str], key: str) -> bool:
    value = text(row, key).lower()
    if value not in {"yes", "no"}:
        raise ValueError(f"{key} must be yes or no")
    return value == "yes"


def parse_review(row: Dict[str, str], event_id: str) -> Dict[str, Any]:
    decision = text(row, "decision").lower()
    if decision not in {ACCEPT, REJECT, UNCERTAIN}:
        raise ValueError(
            f"{event_id}: decision must be accept, reject, or uncertain"
        )
    parsed: Dict[str, Any] = {"decision": decision, "notes": text(row, "notes")}
    if decision != ACCEPT:
        return parsed
    event_type = text(row, "reviewed_event_type").upper()
    if event_type not in EVENT_TYPES:
        raise ValueError(f"{event_id}: invalid reviewed_event_type {event_type!r}")
    try:
        boundary = int(text(row, "reviewed_boundary_frame"))
    except ValueError as error:
        raise ValueError(f"{event_id}: reviewed_boundary_frame must be an integer") from error
    parsed.update({
        "event_type": event_type,
        "boundary": boundary,
        "absence_confirmed": yes(row, "absence_confirmed"),
        "identity_confirmed": yes(row, "identity_confirmed"),
    })
    return parsed


def parse_adjudication(row: Dict[str, str], event_id: str) -> Dict[str, Any]:
    decision = text(row, "final_decision").lower()
    if decision not in {ACCEPT, REJECT}:
        raise ValueError(f"{event_id}: adjudicator final_decision must be accept or reject")
    parsed: Dict[str, Any] = {"decision": decision, "notes": text(row, "notes")}
    if decision == REJECT:
        return parsed
    event_type = text(row, "final_event_type").upper()
    if event_type not in EVENT_TYPES:
        raise ValueError(f"{event_id}: invalid final_event_type {event_type!r}")
    try:
        boundary = int(text(row, "final_boundary_frame"))
    except ValueError as error:
        raise ValueError(f"{event_id}: final_boundary_frame must be an integer") from error
    parsed.update({
        "event_type": event_type,
        "boundary": boundary,
        "absence_confirmed": yes(row, "absence_confirmed"),
        "identity_confirmed": yes(row, "identity_confirmed"),
    })
    return parsed


def reviewers_agree(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if a["decision"] == b["decision"] == REJECT:
        return True
    if a["decision"] != ACCEPT or b["decision"] != ACCEPT:
        return False
    return (
        a["event_type"] == b["event_type"]
        and abs(a["boundary"] - b["boundary"]) <= 1
        and a["absence_confirmed"]
        and b["absence_confirmed"]
        and a["identity_confirmed"]
        and b["identity_confirmed"]
    )


def accepted_event(
    candidate: Dict[str, Any],
    decision: Dict[str, Any],
    status: str,
    annotator_ids: List[str],
    adjudicator_id: Optional[str],
) -> Dict[str, Any]:
    event = dict(candidate)
    event_type = decision["event_type"]
    boundary = decision["boundary"]
    event.update({
        "event_type": event_type,
        "start_frame": boundary,
        "end_frame": boundary,
        "human_verified": True,
        "human_verification": {
            "status": status,
            "annotator_ids": annotator_ids,
            "adjudicator_id": adjudicator_id,
        },
        "annotation_source": "two_independent_human_reviews",
    })
    if event_type == "ENTER":
        event.update({
            "state_before": "confirmed_absent",
            "state_after": "present",
            "visibility_before": "out_of_frame",
            "visibility_after": "visible",
        })
    elif event_type == "EXIT":
        event.update({
            "state_before": "present",
            "state_after": "confirmed_absent",
            "visibility_before": "visible",
            "visibility_after": "out_of_frame",
        })
    event["notes"] = (
        f"Human review status={status}. "
        f"Reviewer notes: {decision.get('notes', '')}"
    )
    return event


def compile_reviews(
    *,
    candidates_path: Path,
    annotator_a_path: Path,
    annotator_a_id: str,
    annotator_b_path: Path,
    annotator_b_id: str,
    adjudicator_path: Optional[Path],
    adjudicator_id: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not annotator_a_id or not annotator_b_id or annotator_a_id == annotator_b_id:
        raise ValueError("two distinct non-empty annotator IDs are required")
    if adjudicator_id in {annotator_a_id, annotator_b_id}:
        raise ValueError("adjudicator ID must differ from both annotator IDs")

    candidates = load_jsonl(candidates_path)
    review_a = load_csv(annotator_a_path)
    review_b = load_csv(annotator_b_path)
    adjudications = load_csv(adjudicator_path) if adjudicator_path else {}
    expected_ids = {row["event_id"] for row in candidates}
    for name, rows in (("annotator A", review_a), ("annotator B", review_b)):
        missing = sorted(expected_ids - set(rows))
        extra = sorted(set(rows) - expected_ids)
        if missing or extra:
            raise ValueError(f"{name} event IDs mismatch: missing={missing}, extra={extra}")

    accepted: List[Dict[str, Any]] = []
    rejected = 0
    agreed = 0
    adjudicated = 0
    for candidate in candidates:
        event_id = candidate["event_id"]
        a = parse_review(review_a[event_id], event_id)
        b = parse_review(review_b[event_id], event_id)
        if reviewers_agree(a, b):
            agreed += 1
            if a["decision"] == REJECT:
                rejected += 1
                continue
            decision = dict(a)
            decision["boundary"] = (a["boundary"] + b["boundary"] + 1) // 2
            decision["notes"] = f"A: {a['notes']} | B: {b['notes']}"
            accepted.append(accepted_event(
                candidate, decision, "agreed",
                [annotator_a_id, annotator_b_id], None,
            ))
            continue

        if not adjudicator_path or not adjudicator_id:
            raise ValueError(
                f"{event_id}: disagreement/uncertainty requires an independent adjudicator"
            )
        if event_id not in adjudications:
            raise ValueError(f"{event_id}: missing adjudication row")
        decision = parse_adjudication(adjudications[event_id], event_id)
        adjudicated += 1
        if decision["decision"] == REJECT:
            rejected += 1
            continue
        if not decision["absence_confirmed"] or not decision["identity_confirmed"]:
            raise ValueError(
                f"{event_id}: accepted adjudication requires confirmed absence and identity"
            )
        accepted.append(accepted_event(
            candidate, decision, "adjudicated",
            [annotator_a_id, annotator_b_id], adjudicator_id,
        ))

    clusters = {row["independence_cluster_id"] for row in accepted}
    summary = {
        "candidate_count": len(candidates),
        "accepted_event_count": len(accepted),
        "accepted_independent_cluster_count": len(clusters),
        "rejected_count": rejected,
        "agreed_count": agreed,
        "adjudicated_count": adjudicated,
        "pilot_gate_range": [30, 50],
        "pilot_event_gate_passed": 30 <= len(clusters) <= 50,
    }
    return accepted, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        default="data/negative_semantics/g0/youtube_vos_pilot_event_candidates.jsonl",
    )
    parser.add_argument("--annotator-a", required=True)
    parser.add_argument("--annotator-a-id", required=True)
    parser.add_argument("--annotator-b", required=True)
    parser.add_argument("--annotator-b-id", required=True)
    parser.add_argument("--adjudicator")
    parser.add_argument("--adjudicator-id")
    parser.add_argument(
        "--output",
        default="data/negative_semantics/g0/youtube_vos_pilot_event_annotations.jsonl",
    )
    args = parser.parse_args()
    if bool(args.adjudicator) != bool(args.adjudicator_id):
        parser.error("--adjudicator and --adjudicator-id must be provided together")

    accepted, summary = compile_reviews(
        candidates_path=Path(args.candidates),
        annotator_a_path=Path(args.annotator_a),
        annotator_a_id=args.annotator_a_id,
        annotator_b_path=Path(args.annotator_b),
        annotator_b_id=args.annotator_b_id,
        adjudicator_path=Path(args.adjudicator) if args.adjudicator else None,
        adjudicator_id=args.adjudicator_id,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in accepted),
        encoding="utf-8",
    )
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pilot_event_gate_passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())

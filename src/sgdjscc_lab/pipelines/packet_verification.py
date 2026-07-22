"""pipelines/packet_verification.py – Packet Verifier + Controller wiring (ETRI 2차 step 7).

Glues ``evaluators/packet_verifier.py`` (report + severity) and
``controllers/verifier_controller.py`` (error-type decision + candidate
actions) onto a :class:`~sgdjscc_lab.video.temporal_pipeline.TemporalPipeline`
run, and serialises the result to ``packet_match_report`` /
``controller_decisions`` JSON+CSV.

Design constraints (docs/etri_strategy.md 2차)
-----------------------------------------------
- This module never re-runs reconstruction and never mutates a packet or a
  reconstructed tensor — it only reads ``FrameRecord.orig_packet`` /
  ``recon_packet`` (already computed by the temporal pipeline) and appends
  verifier/controller columns.
- Everything here is behind a config gate (``use_packet_verifier`` +
  ``verifier.enabled``, both default False). :func:`maybe_run` returns
  ``None`` immediately when the gate is off and touches nothing, so the 1차
  outputs (``temporal_frames.csv``, ``segments.json``, …) are byte-for-byte
  unchanged unless a caller explicitly opts in.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Row fields every packet_match_report row is guaranteed to carry (ETRI 2차
# spec §4 minimum).
REQUIRED_REPORT_FIELDS = (
    "frame_index", "object_match_rate", "missing_objects", "additional_objects",
    "relation_errors", "attribute_errors", "scene_match", "severity",
    "controller_decision", "candidate_actions",
)
# The full, canonical packet_match_report.csv column schema — REQUIRED_REPORT_FIELDS
# plus extras (role, counts, triggered_modes, reason) kept for debuggability. This
# is the single source of truth for the CSV header: write_reports() uses it
# directly (via row.get(...)) instead of inferring columns from whatever a row
# happens to contain, so the on-disk schema can never silently drift between runs.
REPORT_FIELDS = REQUIRED_REPORT_FIELDS + (
    "role", "missing_object_count", "additional_object_count",
    "relation_error_count", "attribute_error_count", "triggered_modes", "reason",
)
DECISION_FIELDS = (
    "frame_index", "severity", "controller_decision", "candidate_actions",
    "triggered_modes", "reason",
)


def gate_enabled(cfg) -> bool:
    """Return True only when both ``use_packet_verifier`` and ``verifier.enabled``
    are true (and the phase4 master switch is on)."""
    from omegaconf import OmegaConf
    from sgdjscc_lab.phase_gates import effective_flag

    if not effective_flag(cfg, "use_packet_verifier", phase=4):
        return False
    return bool(OmegaConf.select(cfg, "verifier.enabled", default=False))


def build_verifier_and_controller(cfg):
    """Build a ``(PacketVerifier, VerifierController)`` pair from ``verifier.*`` cfg keys."""
    from omegaconf import OmegaConf
    from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
    from sgdjscc_lab.controllers.verifier_controller import (
        VerifierController, VerifierControllerConfig,
    )

    vcfg = VerifierControllerConfig(
        accept_severity=float(OmegaConf.select(cfg, "verifier.accept_severity", default=0.15)),
        fallback_severity=float(OmegaConf.select(cfg, "verifier.severity_threshold", default=0.6)),
        keyframe_fallback_severity=float(
            OmegaConf.select(cfg, "verifier.keyframe_fallback_severity", default=0.85)
        ),
        missing_object_threshold=int(
            OmegaConf.select(cfg, "verifier.missing_object_threshold", default=1)
        ),
        additional_object_threshold=int(
            OmegaConf.select(cfg, "verifier.additional_object_threshold", default=1)
        ),
        structural_error_threshold=int(
            OmegaConf.select(cfg, "verifier.structural_error_threshold", default=1)
        ),
    )

    verifier = PacketVerifier()
    return verifier, VerifierController(vcfg)


def _row_for_frame(index, role, verifier, controller, orig_packet, recon_packet) -> Dict:
    """Verify one frame and merge the report + controller decision into one row."""
    is_interframe = role != "keyframe"
    report = verifier.verify(orig_packet or {}, recon_packet or {}, item_id=index)
    decision = controller.decide(report, is_interframe=is_interframe)

    row = {
        "frame_index": index,
        "role": role,
        "object_match_rate": report["object_match_rate"],
        "missing_objects": report["missing_objects"],
        "missing_object_count": report["missing_object_count"],
        "additional_objects": report["additional_objects"],
        "additional_object_count": report["additional_object_count"],
        "relation_errors": report["relation_errors"],
        "relation_error_count": report["relation_error_count"],
        "attribute_errors": report["attribute_errors"],
        "attribute_error_count": report["attribute_error_count"],
        "scene_match": report["scene_match"],
        "severity": report["severity"],
    }
    row.update(decision.to_dict())
    return row


def verify_records(records, verifier=None, controller=None) -> List[Dict]:
    """Verify a list of ``FrameRecord``-like objects (need ``.index``, ``.role``,
    ``.orig_packet``, ``.recon_packet``) and return one report+decision row per frame."""
    from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
    from sgdjscc_lab.controllers.verifier_controller import VerifierController

    verifier = verifier or PacketVerifier()
    controller = controller or VerifierController()

    rows: List[Dict] = []
    for rec in records:
        index = getattr(rec, "index", None)
        role = getattr(rec, "role", None)
        orig_packet = getattr(rec, "orig_packet", None) or {}
        recon_packet = getattr(rec, "recon_packet", None) or {}
        rows.append(_row_for_frame(index, role, verifier, controller, orig_packet, recon_packet))
    return rows


def summarize_rows(rows: List[Dict]) -> Dict:
    """Aggregate a list of per-frame verifier rows into one segment-level summary."""
    if not rows:
        return {"mean_severity": None, "max_severity": None, "decision_counts": {}, "worst_decision": None}
    severities = [float(r["severity"]) for r in rows]
    counts = Counter(r["controller_decision"] for r in rows)
    worst = max(rows, key=lambda r: r["severity"])
    return {
        "mean_severity": float(sum(severities) / len(severities)),
        "max_severity": float(max(severities)),
        "decision_counts": dict(counts),
        "worst_decision": worst["controller_decision"],
    }


def attach_segment_summaries(segment_records: List[Dict], rows: List[Dict]) -> List[Dict]:
    """Attach a ``verifier_summary`` key to each segment record dict, in place.

    Purely additive: existing segment_record keys are untouched, so this is
    safe to call on ``segments.json`` records without disturbing 1차 fields.
    """
    rows_by_index = {r["frame_index"]: r for r in rows}
    for seg in segment_records:
        idxs = [seg["keyframe_index"]] + list(seg.get("inter_frame_indices") or [])
        seg_rows = [rows_by_index[i] for i in idxs if i in rows_by_index]
        seg["verifier_summary"] = summarize_rows(seg_rows)
    return segment_records


def _csv_safe_row(row: Dict, fields) -> Dict:
    """Flatten one row to CSV-writable scalars (lists/dicts → JSON strings)."""
    out = {}
    for k in fields:
        v = row.get(k)
        out[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
    return out


def write_reports(
    rows: List[Dict],
    report_json: Optional[str] = None,
    report_csv: Optional[str] = None,
    decisions_json: Optional[str] = None,
    decisions_csv: Optional[str] = None,
) -> None:
    """Write the packet_match_report and controller_decisions JSON/CSV files."""
    if report_json:
        p = Path(report_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, ensure_ascii=False)
        logger.info("Packet match report → %s", p)

    if report_csv and rows:
        p = Path(report_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(REPORT_FIELDS))
            w.writeheader()
            for row in rows:
                w.writerow(_csv_safe_row(row, REPORT_FIELDS))
        logger.info("Packet match report CSV → %s", p)

    decision_rows = [{k: r.get(k) for k in DECISION_FIELDS} for r in rows]
    if decisions_json:
        p = Path(decisions_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(decision_rows, fh, indent=2, ensure_ascii=False)
        logger.info("Controller decisions → %s", p)

    if decisions_csv and decision_rows:
        p = Path(decisions_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(DECISION_FIELDS))
            w.writeheader()
            for row in decision_rows:
                w.writerow(_csv_safe_row(row, DECISION_FIELDS))
        logger.info("Controller decisions CSV → %s", p)


def maybe_run(result: Dict, cfg) -> Optional[Dict]:
    """Run the packet verifier + controller over a ``TemporalPipeline.run()`` result.

    When the config gate is off, returns ``None`` and changes nothing (no
    reads of ``result`` beyond the gate check). When on:

    - computes one verifier+controller row per frame (``result["records"]``),
    - merges ``severity`` / ``controller_decision`` into each
      ``result["frame_records"]`` dict (so ``temporal_frames.csv`` gains the
      two columns when the gate is on),
    - attaches a ``verifier_summary`` to each ``result["segment_records"]`` dict,
    - optionally writes the report/decision JSON+CSV files (``verifier.save_reports``),
    - returns ``{"rows": rows}``.
    """
    if not gate_enabled(cfg):
        return None

    from omegaconf import OmegaConf

    verifier, controller = build_verifier_and_controller(cfg)
    rows = verify_records(result.get("records") or [], verifier, controller)
    rows_by_index = {r["frame_index"]: r for r in rows}

    for flog in result.get("frame_records") or []:
        extra = rows_by_index.get(flog.get("index"))
        if extra is not None:
            flog["severity"] = extra["severity"]
            flog["controller_decision"] = extra["controller_decision"]

    attach_segment_summaries(result.get("segment_records") or [], rows)

    if bool(OmegaConf.select(cfg, "verifier.save_reports", default=True)):
        write_reports(
            rows,
            report_json=OmegaConf.select(cfg, "verifier.report_json", default=None),
            report_csv=OmegaConf.select(cfg, "verifier.report_csv", default=None),
            decisions_json=OmegaConf.select(cfg, "verifier.decisions_json", default=None),
            decisions_csv=OmegaConf.select(cfg, "verifier.decisions_csv", default=None),
        )

    return {"rows": rows}

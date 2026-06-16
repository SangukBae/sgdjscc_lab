"""evaluators/temporal_consistency.py – Temporal/sequence metrics (Phase 4-B).

Aggregates per-frame reconstruction records of a sequence into temporal
reliability metrics.  These extend the single-image SRS into the time domain so
keyframe-reuse / semantic-delta strategies can be judged not only on per-frame
quality but on how *stable* the recovered semantics are across the sequence.

Reported metrics
----------------
    temporal_srs                 mean per-frame SRS over the sequence
    srs_flicker                  mean |SRS_t − SRS_{t-1}| (lower = steadier)
    object_identity_consistency  mean Jaccard of reconstructed object sets between
                                 consecutive frames (object permanence)
    temporal_segmentation_iou    mean segmentation-histogram overlap between
                                 consecutive reconstructed frames (None if absent)
    temporal_hallucination_rate  mean per-frame additional-object rate
                                 (recon objects absent from that frame's original)

Each *frame record* is a dict that may contain:
    ``srs`` (float), ``orig_packet`` (dict), ``recon_packet`` (dict).
Missing fields are skipped gracefully.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from sgdjscc_lab.evaluators.semantic_packet_matcher import compare, segmentation_consistency

logger = logging.getLogger(__name__)


def _field(record, name):
    """Read *name* from a dict-like record or a FrameRecord-like object."""
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def _mean(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def evaluate_sequence(frame_records: List[Dict]) -> Dict:
    """Aggregate temporal metrics over an ordered list of frame records.

    Parameters
    ----------
    frame_records:
        Ordered list (frame 0 … N-1) of dicts with optional ``srs``,
        ``orig_packet`` and ``recon_packet`` fields.

    Returns
    -------
    dict of temporal metrics (see module docstring).  ``n_frames`` is always set.
    """
    n = len(frame_records)
    out: Dict = {"n_frames": n}

    # ── temporal SRS + flicker ───────────────────────────────────────────────
    srs_vals = [_field(r, "srs") for r in frame_records if _field(r, "srs") is not None]
    out["temporal_srs"] = _mean([float(s) for s in srs_vals])
    flicker = [
        abs(float(srs_vals[i]) - float(srs_vals[i - 1]))
        for i in range(1, len(srs_vals))
    ]
    out["srs_flicker"] = _mean(flicker)

    # ── object identity consistency (consecutive recon frames) ───────────────
    recon_packets = [_field(r, "recon_packet") for r in frame_records]
    identity = []
    for i in range(1, n):
        p0, p1 = recon_packets[i - 1], recon_packets[i]
        if p0 is None or p1 is None:
            continue
        identity.append(_jaccard(set(p0.get("objects") or []), set(p1.get("objects") or [])))
    out["object_identity_consistency"] = _mean(identity)

    # ── temporal segmentation IoU (consecutive recon frames) ─────────────────
    seg_iou = []
    for i in range(1, n):
        p0, p1 = recon_packets[i - 1], recon_packets[i]
        if p0 is None or p1 is None:
            continue
        s = segmentation_consistency(
            p0.get("segmentation_summary"), p1.get("segmentation_summary")
        )
        if s is not None:
            seg_iou.append(s)
    out["temporal_segmentation_iou"] = _mean(seg_iou)

    # ── temporal hallucination rate (per-frame additional objects) ───────────
    hall = []
    for r in frame_records:
        op, rp = _field(r, "orig_packet"), _field(r, "recon_packet")
        if op is None or rp is None:
            continue
        rep = compare(op, rp)
        n_orig = max(len(op.get("objects") or []), 1)
        hall.append(rep["additional_object_count"] / n_orig)
    out["temporal_hallucination_rate"] = _mean(hall)

    return out


class TemporalConsistencyEvaluator:
    """OO wrapper around :func:`evaluate_sequence`."""

    def evaluate(self, frame_records: List[Dict]) -> Dict:
        return evaluate_sequence(frame_records)

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

ETRI 1차 time-axis semantic metrics (PROVISIONAL — CLIP/packet based)
---------------------------------------------------------------------
    ptc   Packet-Temporal Consistency: mean per-frame consistency between the
          reference (transmitted/original) packet and the reconstructed packet,
          i.e. how well packet agreement HOLDS over time.  1.0 = every frame's
          recon packet matches its reference packet.
    sfr   Semantic Flicker Rate: rate of *spurious* object births/deaths between
          consecutive reconstructed frames — recon-set changes that are NOT
          mirrored by the corresponding original-packet changes (so genuine
          scene changes are not counted as flicker).  0.0 = no flicker.
    sdi   Semantic Drift Index: least-squares slope of per-frame packet drift
          (1 − packet consistency) against the frame's distance from its GOP
          keyframe.  Positive = reconstruction drifts away from the reference
          the further a frame is from its keyframe.  Requires ``role`` fields
          ("keyframe"/"inter") on the records; None when unavailable or when
          distances are degenerate.

PROVISIONAL NOTE: ptc / sfr / sdi are first-pass metrics built on the CLIP
text-probe object judgements inside the semantic packets (see
``object_preservation.py``).  Per docs/etri_strategy.md they must be
re-measured after the OWLv2/VQA presence-verification reinforcement (5차 단계)
before being used for final (held-out) claims.

Each *frame record* is a dict (or FrameRecord) that may contain:
    ``srs`` (float), ``orig_packet`` (dict), ``recon_packet`` (dict),
    ``role`` ("keyframe" | "inter", used by sdi).
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


# Weights of the per-frame packet-consistency composite used by ptc / sdi.
# Mirrors the packet-composite structure in semantic_reliability.py (without the
# segmentation term, which is often absent in per-frame packets).
_PTC_WEIGHTS = {"w_obj": 0.5, "w_rel": 0.2, "w_attr": 0.2, "w_scene": 0.1}


def packet_consistency(orig_packet: Dict, recon_packet: Dict) -> float:
    """Scalar [0, 1] consistency between a reference and a reconstructed packet.

    Weighted combination of object match rate, relation consistency, attribute
    consistency and scene match from the packet matcher.  This is the per-frame
    quantity that ``ptc`` averages and ``sdi`` regresses over keyframe distance.
    """
    rep = compare(orig_packet, recon_packet)
    w = _PTC_WEIGHTS
    return float(
        w["w_obj"] * float(rep["object_match_rate"])
        + w["w_rel"] * float(rep["relation_consistency"])
        + w["w_attr"] * float(rep["attribute_consistency"])
        + w["w_scene"] * (1.0 if rep["scene_match"] else 0.0)
    )


def _keyframe_distances(frame_records: List[Dict]) -> Optional[List[Optional[int]]]:
    """Per-frame distance from the most recent keyframe, from ``role`` fields.

    Returns None when no record carries a "keyframe" role (e.g. plain metric
    dicts without pipeline structure) — sdi is then not computable.
    """
    roles = [_field(r, "role") for r in frame_records]
    if "keyframe" not in roles:
        return None
    distances: List[Optional[int]] = []
    last_key: Optional[int] = None
    for i, role in enumerate(roles):
        if role == "keyframe":
            last_key = i
        distances.append(None if last_key is None else i - last_key)
    return distances


def _slope(xs: List[float], ys: List[float]) -> Optional[float]:
    """Least-squares slope of ys over xs; None for < 2 points or zero variance."""
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    var = sum((x - mx) ** 2 for x in xs)
    if var <= 0.0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / var)


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

    # ── ETRI 1차 metrics: ptc / sfr / sdi (PROVISIONAL, CLIP/packet based) ────
    orig_packets = [_field(r, "orig_packet") for r in frame_records]

    # ptc — per-frame reference↔recon packet consistency held over time.
    cons: List[Optional[float]] = []
    for op, rp in zip(orig_packets, recon_packets):
        cons.append(packet_consistency(op, rp) if (op is not None and rp is not None) else None)
    out["ptc"] = _mean([c for c in cons if c is not None])

    # sfr — spurious object birth/death rate between consecutive recon frames.
    # Changes also present between the corresponding *original* packets are
    # legitimate scene evolution, not flicker, and are excluded.
    sfr_vals = []
    for i in range(1, n):
        r0, r1 = recon_packets[i - 1], recon_packets[i]
        if r0 is None or r1 is None:
            continue
        s0 = set(r0.get("objects") or [])
        s1 = set(r1.get("objects") or [])
        births, deaths = s1 - s0, s0 - s1
        o0, o1 = orig_packets[i - 1], orig_packets[i]
        if o0 is not None and o1 is not None:
            g0 = set(o0.get("objects") or [])
            g1 = set(o1.get("objects") or [])
            births -= (g1 - g0)     # object genuinely appeared in the source
            deaths -= (g0 - g1)     # object genuinely disappeared
        denom = max(len(s0 | s1), 1)
        sfr_vals.append((len(births) + len(deaths)) / denom)
    out["sfr"] = _mean(sfr_vals)

    # sdi — slope of packet drift (1 − consistency) over distance-from-keyframe.
    sdi = None
    distances = _keyframe_distances(frame_records)
    if distances is not None:
        xs, ys = [], []
        for d, c in zip(distances, cons):
            if d is None or c is None:
                continue
            xs.append(float(d))
            ys.append(1.0 - c)
        sdi = _slope(xs, ys)
    out["sdi"] = sdi

    return out


class TemporalConsistencyEvaluator:
    """OO wrapper around :func:`evaluate_sequence`."""

    def evaluate(self, frame_records: List[Dict]) -> Dict:
        return evaluate_sequence(frame_records)

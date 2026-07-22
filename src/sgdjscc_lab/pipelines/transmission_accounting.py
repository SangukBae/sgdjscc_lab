"""pipelines/transmission_accounting.py – Transmission accounting pipeline
(ETRI 6차, step 11).

Wraps — never alters — a ``video/temporal_pipeline.py::TemporalPipeline.run()``
result: reads each frame's already-made decision
(``keyframe``/``reuse``/``recompute_semantic``/``recompute_motion``/``generate``)
and accounts its bit/channel-symbol cost via
``accounting/bit_accounting.py``, then aggregates per-segment and per-run
summaries, including a diff against a naive baseline.

**PoC accounting, not a real bitstream/CBR** — see
``accounting/bit_accounting.py``'s module docstring and
docs/etri_strategy.md 6차 구현 결과 for exactly what is and is not measured.

Semantic-unit vs bit/channel-symbol reduction (ETRI's actual question)
--------------------------------------------------------------------------
1~4차 already track *semantic-unit* savings (``FrameRecord.transmitted_units``,
``TemporalPipeline`` summary's ``naive_units``/``overhead_reduction``). This
module does **not** recompute that — it copies it through unchanged as
``total_semantic_units``/``baseline_semantic_units``/``semantic_unit_reduction``
— and adds the genuinely new numbers this step is about:
``total_bits``/``total_channel_symbols`` and their reductions against a naive
baseline. Keeping the two separate (never blending them into one ratio) is
exactly what lets ETRI's "semantic unit 절감뿐 아니라 channel-symbol/bit 절감도
되는가" question be answered from one report.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from sgdjscc_lab.accounting.bit_accounting import (
    BASELINE_METADATA,
    BASELINE_NAIVE_FULL_FRAME,
    DEFAULT_EDGE_CR,
    DEFAULT_MOTION_BITS_PER_BLOCK,
    DEFAULT_SYMBOLS_PER_BIT,
    TransmissionAccountingRecord,
    account_frame,
    compute_baseline_record,
)

logger = logging.getLogger(__name__)


def _frame_kwargs(rec) -> Dict:
    """Pull the fields ``bit_accounting.account_frame``/``compute_baseline_record``
    need out of one ``FrameRecord``-like object."""
    packet = getattr(rec, "orig_packet", None) or {}
    return {
        "frame_index": getattr(rec, "index", None),
        "role": getattr(rec, "role", None),
        "frame": getattr(rec, "recon", None),
        "packet": packet,
        "caption": packet.get("caption") or None,
        "motion": getattr(rec, "motion", None),
        "transmitted_units": getattr(rec, "transmitted_units", 0) or 0,
    }


def account_transmission(
    result: Dict,
    baseline: str = BASELINE_NAIVE_FULL_FRAME,
    latent_symbols_per_frame: Optional[float] = None,
    edge_cr: float = DEFAULT_EDGE_CR,
    motion_bits_per_block: float = DEFAULT_MOTION_BITS_PER_BLOCK,
    symbols_per_bit_proxy: float = DEFAULT_SYMBOLS_PER_BIT,
) -> Dict:
    """Account a ``TemporalPipeline.run()`` result's transmission cost.

    Parameters
    ----------
    result:
        Return value of ``TemporalPipeline.run()`` — reads ``result["records"]``
        (``FrameRecord`` objects) and ``result["segment_records"]``
        (``SegmentRecord`` dicts) without mutating either.
    baseline:
        One of ``accounting.bit_accounting.BASELINES``.
    latent_symbols_per_frame / edge_cr / motion_bits_per_block / symbols_per_bit_proxy:
        Forwarded to ``bit_accounting.account_frame`` (see its docstring for
        what each proxy means).

    Returns
    -------
    dict with ``frame_records`` (list of dicts), ``segment_summaries`` (list
    of dicts) and ``summary`` (dict).
    """
    records = result.get("records") or []
    kw = dict(
        latent_symbols_override=latent_symbols_per_frame,
        edge_cr=edge_cr,
        motion_bits_per_block=motion_bits_per_block,
        symbols_per_bit=symbols_per_bit_proxy,
    )

    frame_recs: List[TransmissionAccountingRecord] = []
    baseline_recs: List[TransmissionAccountingRecord] = []
    for rec in records:
        fkw = _frame_kwargs(rec)
        frame_recs.append(account_frame(decision=getattr(rec, "decision", None), **fkw, **kw))
        baseline_recs.append(compute_baseline_record(baseline, **fkw, **kw))

    segment_summaries = _build_segment_summaries(
        result.get("segment_records") or [], frame_recs, baseline_recs,
    )
    summary = _build_summary(frame_recs, baseline_recs, baseline, result.get("summary") or {})

    return {
        "frame_records": [r.to_dict() for r in frame_recs],
        "segment_summaries": segment_summaries,
        "summary": summary,
    }


def _build_segment_summaries(
    segment_records: List[Dict],
    frame_recs: List[TransmissionAccountingRecord],
    baseline_recs: List[TransmissionAccountingRecord],
) -> List[Dict]:
    by_index = {r.frame_index: r for r in frame_recs}
    baseline_by_index = {r.frame_index: r for r in baseline_recs}

    out: List[Dict] = []
    for seg in segment_records:
        idxs = [seg.get("keyframe_index")] + list(seg.get("inter_frame_indices") or [])
        seg_frame_recs = [by_index[i] for i in idxs if i in by_index]
        seg_baseline_recs = [baseline_by_index[i] for i in idxs if i in baseline_by_index]

        decisions = [r.decision for r in seg_frame_recs]
        total_bits = sum(r.total_bits for r in seg_frame_recs)
        total_symbols = sum(r.total_channel_symbols for r in seg_frame_recs)
        baseline_bits = sum(r.total_bits for r in seg_baseline_recs)
        baseline_symbols = sum(r.total_channel_symbols for r in seg_baseline_recs)

        out.append({
            "segment_id": seg.get("segment_id"),
            "keyframe_index": seg.get("keyframe_index"),
            "n_frames": len(seg_frame_recs),
            "n_generate": sum(1 for d in decisions if d == "generate"),
            "n_reused": sum(1 for d in decisions if d == "reuse"),
            "n_recompute": sum(1 for d in decisions if d in ("recompute_semantic", "recompute_motion")),
            "total_bits": total_bits,
            "total_channel_symbols": total_symbols,
            "reduction_vs_naive_bits": _reduction(total_bits, baseline_bits),
            "reduction_vs_naive_symbols": _reduction(total_symbols, baseline_symbols),
        })
    return out


def _reduction(value: float, baseline_value: float) -> Optional[float]:
    if not baseline_value:
        return None
    return float(1.0 - (value / baseline_value))


def _build_summary(
    frame_recs: List[TransmissionAccountingRecord],
    baseline_recs: List[TransmissionAccountingRecord],
    baseline: str,
    temporal_summary: Dict,
) -> Dict:
    total_bits = sum(r.total_bits for r in frame_recs)
    total_symbols = sum(r.total_channel_symbols for r in frame_recs)
    total_semantic_units = sum(r.total_semantic_units for r in frame_recs)
    baseline_bits = sum(r.total_bits for r in baseline_recs)
    baseline_symbols = sum(r.total_channel_symbols for r in baseline_recs)

    decisions = [r.decision for r in frame_recs]
    n_frames = len(frame_recs)
    n_proxy_components = sum(1 for r in frame_recs for _ in r.proxy_notes)
    n_total_components = sum(len(r.components) for r in frame_recs) or 1

    return {
        "n_frames": n_frames,
        "n_keyframes": sum(1 for d in decisions if d == "keyframe"),
        "n_generate": sum(1 for d in decisions if d == "generate"),
        "n_reused": sum(1 for d in decisions if d == "reuse"),
        "n_recompute": sum(1 for d in decisions if d in ("recompute_semantic", "recompute_motion")),
        "total_bits": total_bits,
        "total_channel_symbols": total_symbols,
        "total_semantic_units": total_semantic_units,
        "baseline": baseline,
        "baseline_bits": baseline_bits,
        "baseline_channel_symbols": baseline_symbols,
        "bit_reduction": _reduction(total_bits, baseline_bits),
        "symbol_reduction": _reduction(total_symbols, baseline_symbols),
        # Copied through unchanged from TemporalPipeline's own summary — this
        # module does not recompute semantic-unit savings, only bit/symbol
        # savings, so the two stay independently verifiable (see module docstring).
        "baseline_semantic_units": temporal_summary.get("naive_units"),
        "semantic_unit_reduction": temporal_summary.get("overhead_reduction"),
        "proxy_fraction": float(n_proxy_components) / float(n_total_components),
        "baseline_metadata": BASELINE_METADATA.get(baseline),
        "note": (
            "PoC transmission accounting — not a real bitstream/CBR implementation. "
            "See docs/etri_strategy.md 6차 구현 결과 and accounting/bit_accounting.py "
            "for which components are exact vs proxy."
        ),
    }


_FRAME_CSV_FIELDS = (
    "frame_index", "decision", "role", "total_bits", "total_channel_symbols",
    "total_semantic_units", "components", "proxy_notes",
)
_SEGMENT_CSV_FIELDS = (
    "segment_id", "keyframe_index", "n_frames", "n_generate", "n_reused", "n_recompute",
    "total_bits", "total_channel_symbols", "reduction_vs_naive_bits", "reduction_vs_naive_symbols",
)


def _csv_safe(row: Dict, fields) -> Dict:
    out = {}
    for k in fields:
        v = row.get(k)
        out[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
    return out


def _write_rows_csv(rows: List[Dict], path, fields) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        for row in rows:
            w.writerow(_csv_safe(row, fields))


def write_accounting(
    result: Dict,
    frame_json: Optional[str] = None,
    frame_csv: Optional[str] = None,
    segment_json: Optional[str] = None,
    segment_csv: Optional[str] = None,
    summary_json: Optional[str] = None,
) -> None:
    """Persist :func:`account_transmission`'s output. Each path is optional
    and independently skipped when ``None``."""
    if frame_json:
        p = Path(frame_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(result["frame_records"], fh, indent=2, ensure_ascii=False)
        logger.info("Frame accounting → %s", p)
    if frame_csv:
        _write_rows_csv(result["frame_records"], frame_csv, _FRAME_CSV_FIELDS)
        logger.info("Frame accounting CSV → %s", frame_csv)

    if segment_json:
        p = Path(segment_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(result["segment_summaries"], fh, indent=2, ensure_ascii=False)
        logger.info("Segment accounting → %s", p)
    if segment_csv:
        _write_rows_csv(result["segment_summaries"], segment_csv, _SEGMENT_CSV_FIELDS)
        logger.info("Segment accounting CSV → %s", segment_csv)

    if summary_json:
        p = Path(summary_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(result["summary"], fh, indent=2, ensure_ascii=False)
        logger.info("Accounting summary → %s", p)

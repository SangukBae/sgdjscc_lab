"""video/segment.py – GOP/segment-level records (ETRI 1차 step 4).

Groups the per-frame :class:`~sgdjscc_lab.video.temporal_pipeline.FrameRecord`
outputs into GOP/segment units so downstream stages can reason about a segment
(keyframe + its dependent inter-frames) instead of individual frames.  This is
pure aggregation over already-computed frame records — it never re-runs
reconstruction and has no effect on the frame-wise numerics.

A :class:`SegmentRecord` summarises, per GOP:

- which frame is the keyframe and which inter-frames depend on it,
- the per-frame gate decisions (reuse / recompute_semantic / recompute_motion),
- transmitted semantic units,
- semantic-delta and motion summaries,
- temporal metrics computed over just this segment's frames.

Generate-branch interface (reserved, NOT implemented here)
----------------------------------------------------------
The 3차 start-only generate branch (docs/etri_strategy.md 순서 5,
docs/video_extension_lgvsc.md §6.3 4단계) will operate on the segment contract
``(start_keyframe, [end_keyframe], segment packets, side_info, length) →
frames``.  ``SegmentRecord.generation`` is the attachment point for its result;
it stays ``None`` in the 1차 implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _mean(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


@dataclass
class SegmentRecord:
    """One GOP/segment: a keyframe plus its dependent inter-frames."""

    segment_id: int
    keyframe_index: int
    inter_frame_indices: List[int] = field(default_factory=list)
    # Ordered per-frame decisions, e.g. [{"index": 3, "decision": "reuse",
    # "reused": True, "magnitude": 0.1, "motion_score": None}, ...]
    frame_decisions: List[Dict] = field(default_factory=list)
    transmitted_units: int = 0
    semantic_delta: Dict = field(default_factory=dict)
    motion: Dict = field(default_factory=dict)
    temporal_metrics: Dict = field(default_factory=dict)
    # Reserved for the 3차 generate branch (start-only segment generation);
    # always None in the 1차 implementation.
    generation: Optional[Dict] = None

    @property
    def frame_indices(self) -> List[int]:
        return [self.keyframe_index] + list(self.inter_frame_indices)

    def to_dict(self) -> Dict:
        return {
            "segment_id": self.segment_id,
            "keyframe_index": self.keyframe_index,
            "inter_frame_indices": list(self.inter_frame_indices),
            "frame_decisions": list(self.frame_decisions),
            "transmitted_units": self.transmitted_units,
            "semantic_delta": dict(self.semantic_delta),
            "motion": dict(self.motion),
            "temporal_metrics": dict(self.temporal_metrics),
            "generation": self.generation,
        }


def _delta_summary(records) -> Dict:
    """Aggregate the inter-frame semantic deltas of one segment."""
    mags = [float(r.delta["magnitude"]) for r in records
            if r.delta is not None and r.delta.get("magnitude") is not None]
    changes = [int(r.delta["num_changes"]) for r in records
               if r.delta is not None and r.delta.get("num_changes") is not None]
    return {
        "mean_magnitude": _mean(mags),
        "max_magnitude": max(mags) if mags else None,
        "total_changes": sum(changes) if changes else 0,
    }


def _motion_summary(records) -> Dict:
    """Aggregate the keyframe-anchored motion scores of one segment."""
    scores = [float(r.motion_score) for r in records if r.motion_score is not None]
    return {
        "mean_score": _mean(scores),
        "max_score": max(scores) if scores else None,
        "n_measured": len(scores),
        "n_motion_recompute": sum(1 for r in records if r.decision == "recompute_motion"),
    }


def build_segments(records, structure: Dict) -> List[SegmentRecord]:
    """Group frame records into :class:`SegmentRecord` objects, one per GOP.

    Parameters
    ----------
    records:
        Ordered list of ``FrameRecord`` objects from ``TemporalPipeline.run``.
    structure:
        The keyframe/GOP structure dict from the keyframe extractor
        (``{"gops": [{"keyframe": k, "inter_frames": [...]}, ...]}``).

    Returns
    -------
    list[SegmentRecord] — the union of all segments' frame indices equals the
    frame-wise record indices (segments are a regrouping, not a re-computation).
    """
    # Imported here (not at module top) to avoid a video ↔ evaluators import
    # cycle at package-import time.
    from sgdjscc_lab.evaluators.temporal_consistency import evaluate_sequence

    by_index = {r.index: r for r in records}
    segments: List[SegmentRecord] = []
    for sid, gop in enumerate(structure.get("gops") or []):
        key_idx = int(gop["keyframe"])
        inter = [int(i) for i in gop.get("inter_frames") or []]
        seg_records = [by_index[i] for i in [key_idx] + inter if i in by_index]

        decisions = [
            {
                "index": r.index,
                "role": r.role,
                "decision": r.decision,
                "reused": r.reused,
                "magnitude": (r.delta or {}).get("magnitude") if r.delta else None,
                "motion_score": r.motion_score,
            }
            for r in seg_records
        ]

        segments.append(SegmentRecord(
            segment_id=sid,
            keyframe_index=key_idx,
            inter_frame_indices=inter,
            frame_decisions=decisions,
            transmitted_units=sum(r.transmitted_units for r in seg_records),
            semantic_delta=_delta_summary(seg_records),
            motion=_motion_summary(seg_records),
            temporal_metrics=evaluate_sequence(seg_records),
        ))
    return segments

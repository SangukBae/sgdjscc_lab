"""guidance/importance_estimator.py – Semantic-unit importance estimator (Phase 4-A).

FAST-GSC (``paper/FAST-GSC/FAST_GSC.tex``) orders semantic units by their
intrinsic importance / temporal dependency before transmission so that the most
informative units arrive first.  Phase 4-A does not run reinforcement learning to
learn that order; instead it scores each object with a transparent heuristic and
exposes a transmission ordering that the adaptive controller and the (future)
delta-transmission logic can reuse.

Importance signals (all in [0, 1], linearly combined):

- **area**       – the object's region fraction from the segmentation summary, if
                   the segmentation class name matches the object name.
- **relational** – objects that participate in a relation triplet are more
                   central to the scene description.
- **caption**    – objects mentioned earlier in the caption are usually the
                   primary subject.
- **base**       – a small constant so every detected object has non-zero weight.

The result is a per-object score map plus an ``order`` list (descending score),
which is the "transmission order" of semantic units.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = {
    "w_area": 0.40,
    "w_relational": 0.30,
    "w_caption": 0.20,
    "w_base": 0.10,
}


class ImportanceEstimator:
    """Score and order the semantic units (objects) of a packet.

    Parameters
    ----------
    weights:
        Optional override for the four signal weights
        (``w_area``, ``w_relational``, ``w_caption``, ``w_base``).
    """

    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        w = dict(_DEFAULT_WEIGHTS)
        if weights:
            w.update(weights)
        self.weights = w

    def estimate(self, packet: Dict) -> Dict:
        """Return ``{"scores": {obj: float}, "order": [obj, ...]}`` for a packet.

        Parameters
        ----------
        packet:
            A semantic packet dict (see ``guidance/semantic_packet_extractor.py``).
            Only ``objects``, ``relations``, ``caption`` and
            ``segmentation_summary`` are consulted; all are optional.
        """
        objects: List[str] = list(packet.get("objects") or [])
        if not objects:
            return {"scores": {}, "order": []}

        relations = packet.get("relations") or []
        caption = (packet.get("caption") or "").lower()
        seg = packet.get("segmentation_summary") or {}
        class_hist: Dict[str, float] = dict(seg.get("class_histogram") or {})

        related_objs = set()
        for rel in relations:
            related_objs.add(rel.get("subject"))
            related_objs.add(rel.get("object"))

        w = self.weights
        scores: Dict[str, float] = {}
        for obj in objects:
            area = float(class_hist.get(obj, 0.0))
            area = min(max(area, 0.0), 1.0)
            relational = 1.0 if obj in related_objs else 0.0
            # Earlier caption mention → higher score (1.0 at start, →0 at end).
            pos = caption.find(obj.lower())
            caption_sig = 0.0
            if pos >= 0 and len(caption) > 0:
                caption_sig = 1.0 - (pos / len(caption))
            score = (
                w["w_area"] * area
                + w["w_relational"] * relational
                + w["w_caption"] * caption_sig
                + w["w_base"]
            )
            scores[obj] = float(round(min(score, 1.0), 6))

        order = sorted(objects, key=lambda o: scores[o], reverse=True)
        return {"scores": scores, "order": order}

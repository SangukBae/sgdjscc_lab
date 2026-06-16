"""guidance/relation_extractor.py – Relation-triplet extractor (Phase 4-A).

Produces the ``relations`` field of a semantic packet: a list of
``(subject, predicate, object)`` triplets describing how objects relate to one
another (spatial layout and simple interactions).

Approach
--------
Phase 4-A uses a lightweight, deterministic caption parser rather than a trained
scene-graph generator (which would add a heavy dependency outside the Phase-4
scope).  For each known relation keyword found in the caption, the nearest
preceding noun is taken as the *subject* and the nearest following noun as the
*object*.  Nouns are restricted to the supplied object vocabulary so the relation
endpoints always line up with the packet's ``objects`` field — which is exactly
what ``evaluators/relation_consistency.py`` compares.

This is intentionally simple but reproducible, and serves as the
"semantic unit" relation layer referenced by FAST-GSC's semantic-difference
calculation (``paper/FAST-GSC/FAST_GSC.tex``).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Ordered so multi-word predicates are matched before their single-word prefixes.
_RELATION_KEYWORDS: List[str] = [
    "in front of", "on top of", "next to", "close to", "left of", "right of",
    "attached to", "connected to", "standing on", "sitting on", "lying on",
    "resting on", "mounted on",
    "above", "below", "beside", "behind", "under", "near", "holding", "riding",
    "wearing", "carrying", "eating", "watching", "covering", "containing",
    "on", "in", "with", "over",
]


class RelationExtractor:
    """Extract ``(subject, predicate, object)`` relation triplets from a caption.

    Parameters
    ----------
    relation_keywords:
        Ordered list of predicate phrases to detect.  Longer phrases must precede
        their prefixes so greedy matching prefers the more specific relation.
    """

    def __init__(self, relation_keywords: Optional[List[str]] = None) -> None:
        self.relation_keywords = relation_keywords or list(_RELATION_KEYWORDS)

    def extract(self, caption: str, objects: List[str]) -> List[Dict[str, str]]:
        """Return relation triplets found in *caption* among *objects*.

        Parameters
        ----------
        caption:
            Free-text caption (e.g. from BLIP2).
        objects:
            Object vocabulary that triplet endpoints must belong to (typically the
            packet's ``objects`` list).

        Returns
        -------
        list of ``{"subject", "predicate", "object"}`` dicts, de-duplicated and
        order-stable.
        """
        if not caption or not objects:
            return []
        text = caption.lower()

        # Locate every object mention: list of (start_index, name).
        mentions: List[tuple] = []
        for obj in objects:
            for m in re.finditer(re.escape(obj.lower()), text):
                mentions.append((m.start(), obj))
        mentions.sort()
        if len(mentions) < 2:
            return []

        triplets: List[Dict[str, str]] = []
        seen = set()
        for pred in self.relation_keywords:
            for m in re.finditer(rf"\b{re.escape(pred)}\b", text):
                p_start, p_end = m.start(), m.end()
                subject = self._nearest_before(mentions, p_start)
                obj = self._nearest_after(mentions, p_end)
                if subject is None or obj is None or subject == obj:
                    continue
                key = (subject, pred, obj)
                if key in seen:
                    continue
                seen.add(key)
                triplets.append({"subject": subject, "predicate": pred, "object": obj})
        return triplets

    @staticmethod
    def _nearest_before(mentions: List[tuple], pos: int) -> Optional[str]:
        best = None
        for start, name in mentions:
            if start < pos:
                best = name
            else:
                break
        return best

    @staticmethod
    def _nearest_after(mentions: List[tuple], pos: int) -> Optional[str]:
        for start, name in mentions:
            if start >= pos:
                return name
        return None

"""controllers/adaptive_search_policy.py – Regeneration-search ordering (Phase 5-C).

Decides *which* regeneration strategies to try and *in what order*, from the
detected failure signals.  This generalises the Phase 4-A error-type-aware
regeneration policy into a multi-strategy search ordering consumed by
``evaluators/regeneration_search.py``.

Inputs (all optional): the packet-matcher error report, a hallucination score,
and the channel state.  Output: an ordered, de-duplicated list of strategy names.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Strategy names shared with regeneration_search.SEARCH_STRATEGIES.
STRONG_TEXT = "strong_text_weak_edge"
WEAK_TEXT = "weak_text_strong_edge"
UNCONDITIONAL = "unconditional"
CHANNEL_RETRY = "channel_conditioned_retry"

_DEFAULT_ORDER = [STRONG_TEXT, WEAK_TEXT, UNCONDITIONAL]


class AdaptiveSearchPolicy:
    """Order regeneration-search strategies by failure mode and channel state.

    Parameters
    ----------
    hallucination_threshold:
        Hallucination score above which hallucination-correcting strategies are
        prioritised.
    """

    def __init__(self, hallucination_threshold: float = 0.1) -> None:
        self.hallucination_threshold = float(hallucination_threshold)

    def order(
        self,
        error_report: Optional[Dict] = None,
        hallucination_score: Optional[float] = None,
        channel_state: Optional[Dict] = None,
        max_strategies: Optional[int] = None,
    ) -> List[str]:
        """Return an ordered list of strategy names.

        Parameters
        ----------
        error_report:
            Output of ``semantic_packet_matcher.compare`` (optional).
        hallucination_score:
            Scalar hallucination score (optional).
        channel_state:
            Dict with e.g. ``{"csi": "...", "snr_db": ..., "confidence": ...}``.
        max_strategies:
            Optional cap on the number of strategies returned.
        """
        report = error_report or {}
        ordered: List[str] = []

        # Blind / weak channel → try a channel-conditioned retry first.
        cs = channel_state or {}
        csi = str(cs.get("csi", "perfect"))
        snr = cs.get("snr_db")
        if csi in ("none", "unknown", "blind", "imperfect") or (snr is not None and snr < 0):
            ordered.append(CHANNEL_RETRY)

        missing = int(report.get("missing_object_count", 0) or 0)
        additional = int(report.get("additional_object_count", 0) or 0)
        rel_err = int(report.get("relation_error_count", 0) or 0)
        attr_err = int(report.get("attribute_error_count", 0) or 0)
        scene_bad = report.get("scene_match") is False
        hall = float(hallucination_score) if hallucination_score is not None else 0.0

        if missing > 0:
            ordered.append(STRONG_TEXT)
        if additional > 0 or hall > self.hallucination_threshold:
            ordered.append(WEAK_TEXT)
            ordered.append(UNCONDITIONAL)
        if rel_err > 0 or attr_err > 0 or scene_bad:
            ordered.append(WEAK_TEXT)

        # Fall back to the default sweep when nothing specific fired.
        if not ordered:
            ordered = list(_DEFAULT_ORDER)

        # De-duplicate, preserving order.
        seen = set()
        deduped = [s for s in ordered if not (s in seen or seen.add(s))]
        if max_strategies is not None:
            deduped = deduped[:max_strategies]
        return deduped

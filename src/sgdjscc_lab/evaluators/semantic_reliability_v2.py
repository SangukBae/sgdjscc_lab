"""evaluators/semantic_reliability_v2.py – SRS-v2 (Phase 5-C).

SRS-v2 extends the headline SRS into a single score that combines four
verification layers:

    base        Phase-3 SRS (CLIP image/text + object preservation)
    packet      Phase 4-A packet consistency composite (relation/attribute/scene)
    temporal    Phase 4-B temporal SRS (sequence stability), when available
    hallucination  a stronger (VQA) hallucination penalty, when available

The combination is a transparent weighted sum so it degrades gracefully: missing
layers are renormalised away rather than treated as zero.  ``combine_srs_v2`` is a
pure function (unit-testable offline); ``SemanticReliabilityV2Evaluator`` gathers
the layers from the existing evaluators.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

_DEFAULT_V2_WEIGHTS = {
    "w_base": 0.40,
    "w_packet": 0.30,
    "w_temporal": 0.20,
    "w_hall": 0.10,     # weight on (1 - hallucination_score)
}


def combine_srs_v2(components: Dict, weights: Optional[Dict] = None) -> Dict:
    """Combine available verification layers into ``srs_v2``.

    Parameters
    ----------
    components:
        Any subset of ``srs_base``, ``srs_packet``, ``temporal_srs``,
        ``hallucination_score``.  Missing layers are dropped and the remaining
        weights renormalised.
    weights:
        Optional override of the four layer weights.

    Returns
    -------
    dict with ``srs_v2`` and the per-layer values actually used.
    """
    w = dict(_DEFAULT_V2_WEIGHTS)
    if weights:
        w.update(weights)

    terms = []   # (weight, value)
    used: Dict = {}
    if components.get("srs_base") is not None:
        terms.append((w["w_base"], float(components["srs_base"])))
        used["srs_base"] = float(components["srs_base"])
    if components.get("srs_packet") is not None:
        terms.append((w["w_packet"], float(components["srs_packet"])))
        used["srs_packet"] = float(components["srs_packet"])
    if components.get("temporal_srs") is not None:
        terms.append((w["w_temporal"], float(components["temporal_srs"])))
        used["temporal_srs"] = float(components["temporal_srs"])
    if components.get("hallucination_score") is not None:
        hall = float(components["hallucination_score"])
        terms.append((w["w_hall"], 1.0 - hall))
        used["hallucination_score"] = hall

    total_w = sum(wt for wt, _ in terms)
    srs_v2 = sum(wt * v for wt, v in terms) / total_w if total_w > 0 else None
    used["srs_v2"] = None if srs_v2 is None else float(srs_v2)
    used["n_layers"] = len(terms)
    return used


class SemanticReliabilityV2Evaluator:
    """Gather verification layers and produce SRS-v2.

    Parameters
    ----------
    base_evaluator:
        ``SemanticReliabilityEvaluator`` (created lazily if None).
    vqa_evaluator:
        Optional ``VQAHallucinationEvaluator`` for the stronger hallucination layer.
    weights:
        Optional SRS-v2 layer weights.
    """

    def __init__(self, base_evaluator=None, vqa_evaluator=None,
                 weights: Optional[Dict] = None) -> None:
        self._base = base_evaluator
        self._vqa = vqa_evaluator
        self.weights = weights

    def _get_base(self):
        if self._base is None:
            from sgdjscc_lab.evaluators.semantic_reliability import SemanticReliabilityEvaluator
            self._base = SemanticReliabilityEvaluator()
        return self._base

    def evaluate(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        text_list=None,
        orig_packet: Optional[Dict] = None,
        recon_packet: Optional[Dict] = None,
        temporal_metrics: Optional[Dict] = None,
        base_result: Optional[Dict] = None,
    ) -> Dict:
        """Compute SRS-v2 from base SRS + packet + temporal + VQA hallucination.

        ``base_result`` may be supplied to reuse an already-computed base SRS dict
        (and avoid recomputing CLIP); otherwise the base evaluator is run.
        """
        if base_result is None:
            base_result = self._get_base().evaluate(
                original, reconstructed, text_list=text_list,
                orig_packet=orig_packet, recon_packet=recon_packet,
            )

        components = {
            "srs_base": base_result.get("srs_base", base_result.get("semantic_reliability_score")),
            "srs_packet": base_result.get("srs_packet"),
            "temporal_srs": (temporal_metrics or {}).get("temporal_srs"),
        }

        if self._vqa is not None:
            objs = (recon_packet or {}).get("objects") if recon_packet else None
            hall = self._vqa.evaluate(original, reconstructed, objects=objs)
            components["hallucination_score"] = hall.get("hallucination_score")
        elif base_result.get("hallucination_score") is not None:
            components["hallucination_score"] = base_result["hallucination_score"]

        out = combine_srs_v2(components, self.weights)
        out["base_result"] = base_result
        return out

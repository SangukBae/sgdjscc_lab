"""evaluators/relation_consistency.py – Relation-triplet consistency (Phase 4-A).

Compares the ``relations`` field of an original-frame packet against the
reconstructed-frame packet and reports how many ``(subject, predicate, object)``
triplets survived.  Used both as a stand-alone consistency term in the
packet-aware SRS and as a building block of
``evaluators/semantic_packet_matcher.py``.

A triplet matches when subject, predicate and object are all equal (case-folded).
The score is the Jaccard overlap of the two triplet sets; ``1.0`` when both sides
have no relations (vacuously consistent).
"""

from __future__ import annotations

from typing import Dict, List


def _key(triplet: Dict) -> tuple:
    return (
        str(triplet.get("subject", "")).lower(),
        str(triplet.get("predicate", "")).lower(),
        str(triplet.get("object", "")).lower(),
    )


def relation_consistency(
    orig_relations: List[Dict],
    recon_relations: List[Dict],
) -> Dict:
    """Compare two relation-triplet lists.

    Returns
    -------
    dict with keys:
        ``score``   – Jaccard overlap of triplet sets in [0, 1].
        ``matched`` – triplets present in both (list of dicts).
        ``missing`` – triplets in original but not reconstructed.
        ``extra``   – triplets in reconstructed but not original.
    """
    orig = {_key(t): t for t in (orig_relations or [])}
    recon = {_key(t): t for t in (recon_relations or [])}

    matched_keys = set(orig) & set(recon)
    missing_keys = set(orig) - set(recon)
    extra_keys = set(recon) - set(orig)

    union = set(orig) | set(recon)
    score = 1.0 if not union else len(matched_keys) / len(union)

    return {
        "score": float(score),
        "matched": [orig[k] for k in matched_keys],
        "missing": [orig[k] for k in missing_keys],
        "extra": [recon[k] for k in extra_keys],
    }


class RelationConsistencyEvaluator:
    """Thin OO wrapper around :func:`relation_consistency` for parity with the
    other evaluator classes."""

    def evaluate(self, orig_packet: Dict, recon_packet: Dict) -> Dict:
        return relation_consistency(
            orig_packet.get("relations") or [],
            recon_packet.get("relations") or [],
        )

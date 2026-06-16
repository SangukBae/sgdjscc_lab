"""evaluators/attribute_consistency.py – Object-attribute consistency (Phase 4-A).

Compares the ``attributes`` field (per-object colour / material / size adjectives)
of an original-frame packet against the reconstructed-frame packet.  Attribute
drift — e.g. a "red car" reconstructed as a "blue car" — is a subtle semantic
failure that object presence alone does not catch, so it is scored separately and
folded into the packet-aware SRS.

Only objects present in *both* packets' attribute maps are scored (an object that
was lost entirely is already penalised by the missing-object term).  Per-object
consistency is the Jaccard overlap of the two adjective sets; the overall score
is their mean (``1.0`` when there are no shared annotated objects).
"""

from __future__ import annotations

from typing import Dict, List


def attribute_consistency(
    orig_attributes: Dict[str, List[str]],
    recon_attributes: Dict[str, List[str]],
) -> Dict:
    """Compare two ``object -> [adjectives]`` maps.

    Returns
    -------
    dict with keys:
        ``score``  – mean per-object Jaccard overlap in [0, 1].
        ``errors`` – list of ``{object, original, reconstructed}`` for objects
                     whose attribute sets differ.
    """
    orig = orig_attributes or {}
    recon = recon_attributes or {}
    shared = set(orig) & set(recon)

    if not shared:
        return {"score": 1.0, "errors": []}

    scores: List[float] = []
    errors: List[Dict] = []
    for obj in sorted(shared):
        a = {x.lower() for x in orig[obj]}
        b = {x.lower() for x in recon[obj]}
        union = a | b
        jacc = 1.0 if not union else len(a & b) / len(union)
        scores.append(jacc)
        if a != b:
            errors.append({
                "object": obj,
                "original": sorted(a),
                "reconstructed": sorted(b),
            })

    return {"score": float(sum(scores) / len(scores)), "errors": errors}


class AttributeConsistencyEvaluator:
    """Thin OO wrapper around :func:`attribute_consistency`."""

    def evaluate(self, orig_packet: Dict, recon_packet: Dict) -> Dict:
        return attribute_consistency(
            orig_packet.get("attributes") or {},
            recon_packet.get("attributes") or {},
        )

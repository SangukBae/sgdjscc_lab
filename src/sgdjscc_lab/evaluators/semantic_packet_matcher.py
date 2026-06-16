"""evaluators/semantic_packet_matcher.py – Packet-aware verifier (Phase 4-A).

Compares the *original-frame* semantic packet against the *reconstructed-frame*
semantic packet and produces a structured error report.  This is the packet-aware
verifier referenced in the Phase 4 plan: it turns the diffuse "did the meaning
survive?" question into explicit, countable failure modes that both the
packet-aware SRS (``evaluators/semantic_reliability.py``) and the
error-type-aware regeneration policy (``controllers/regeneration_policy.py``)
consume.

Report fields
-------------
    missing_objects / missing_object_count        objects in orig not in recon
    additional_objects / additional_object_count  objects in recon not in orig
    object_match_rate                             |matched| / |orig objects|
    relation_errors / relation_error_count        missing+extra relation triplets
    relation_consistency                          Jaccard over relation triplets
    attribute_errors / attribute_error_count      objects with attribute drift
    attribute_consistency                         mean per-object attribute Jaccard
    scene_match / scene_mismatch                  coarse scene label agreement
    segmentation_consistency                      1 − ½·TV(class histograms)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from sgdjscc_lab.evaluators.attribute_consistency import attribute_consistency
from sgdjscc_lab.evaluators.relation_consistency import relation_consistency

logger = logging.getLogger(__name__)


def segmentation_consistency(orig_seg: Optional[Dict], recon_seg: Optional[Dict]) -> Optional[float]:
    """Histogram-overlap consistency of two segmentation summaries.

    Returns ``1 − ½·Σ|p−q|`` over the union of class names (1.0 = identical
    distributions).  Returns ``None`` when either summary is absent.
    """
    if not orig_seg or not recon_seg:
        return None
    p = orig_seg.get("class_histogram") or {}
    q = recon_seg.get("class_histogram") or {}
    if not p and not q:
        return 1.0
    classes = set(p) | set(q)
    tv = sum(abs(float(p.get(c, 0.0)) - float(q.get(c, 0.0))) for c in classes)
    return float(max(0.0, 1.0 - 0.5 * tv))


def compare(orig_packet: Dict, recon_packet: Dict) -> Dict:
    """Compare two semantic packets and return a structured error report.

    Parameters
    ----------
    orig_packet, recon_packet:
        Semantic packet dicts (see ``guidance/semantic_packet_extractor.py``).

    Returns
    -------
    dict – the error report described in the module docstring.
    """
    orig_objs = set(orig_packet.get("objects") or [])
    recon_objs = set(recon_packet.get("objects") or [])

    missing = sorted(orig_objs - recon_objs)
    additional = sorted(recon_objs - orig_objs)
    matched = orig_objs & recon_objs
    object_match_rate = 1.0 if not orig_objs else len(matched) / len(orig_objs)

    rel = relation_consistency(
        orig_packet.get("relations") or [], recon_packet.get("relations") or []
    )
    relation_errors = rel["missing"] + rel["extra"]

    attr = attribute_consistency(
        orig_packet.get("attributes") or {}, recon_packet.get("attributes") or {}
    )

    orig_scene = orig_packet.get("scene")
    recon_scene = recon_packet.get("scene")
    # Treat "both unknown" as a match; a known-vs-None as a mismatch only when the
    # original actually had a scene label.
    if orig_scene is None:
        scene_match = True
    else:
        scene_match = (orig_scene == recon_scene)

    seg_cons = segmentation_consistency(
        orig_packet.get("segmentation_summary"),
        recon_packet.get("segmentation_summary"),
    )

    return {
        "missing_objects": missing,
        "missing_object_count": len(missing),
        "additional_objects": additional,
        "additional_object_count": len(additional),
        "object_match_rate": float(object_match_rate),
        "relation_errors": relation_errors,
        "relation_error_count": len(relation_errors),
        "relation_consistency": rel["score"],
        "attribute_errors": attr["errors"],
        "attribute_error_count": len(attr["errors"]),
        "attribute_consistency": attr["score"],
        "scene_match": bool(scene_match),
        "scene_mismatch": (not scene_match),
        "original_scene": orig_scene,
        "reconstructed_scene": recon_scene,
        "segmentation_consistency": seg_cons,
    }


class SemanticPacketMatcher:
    """OO wrapper around :func:`compare` for parity with other evaluators."""

    def evaluate(self, orig_packet: Dict, recon_packet: Dict) -> Dict:
        return compare(orig_packet, recon_packet)

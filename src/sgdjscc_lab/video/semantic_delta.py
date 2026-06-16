"""video/semantic_delta.py – Packet-level temporal delta (Phase 4-B).

Computes the *semantic difference* between a frame's packet and a reference packet
(the previous keyframe or the previously transmitted frame).  This is the
``semantic difference calculation`` referenced by FAST-GSC
(``paper/FAST-GSC/FAST_GSC.tex``): instead of resending a full semantic
description every frame, only the changed semantic units need to be transmitted.

A delta contains:

    new_objects        objects present now but not in the reference
    removed_objects    objects in the reference but gone now
    changed_relations  relation triplets added/dropped vs the reference
    changed_attributes objects whose attribute set drifted
    scene_changed      coarse scene label differs
    magnitude          scalar change score in [0, 1]

``magnitude`` drives the temporal pipeline's reuse / attenuation policy: a small
delta means the reference reconstruction can be reused with light guidance, a
large delta means the frame is effectively a new keyframe.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from sgdjscc_lab.evaluators.semantic_packet_matcher import compare

logger = logging.getLogger(__name__)

_DEFAULT_MAGNITUDE_WEIGHTS = {
    "w_object": 0.5,
    "w_relation": 0.2,
    "w_attribute": 0.2,
    "w_scene": 0.1,
}


def compute_delta(
    reference_packet: Dict,
    current_packet: Dict,
    weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """Compute the semantic delta from *reference_packet* to *current_packet*.

    Parameters
    ----------
    reference_packet:
        The previous keyframe (or last transmitted) packet.
    current_packet:
        The current frame's packet.
    weights:
        Optional magnitude weights (``w_object``, ``w_relation``, ``w_attribute``,
        ``w_scene``).

    Returns
    -------
    dict – the delta described in the module docstring.
    """
    w = dict(_DEFAULT_MAGNITUDE_WEIGHTS)
    if weights:
        w.update(weights)

    # compare(orig=reference, recon=current): "missing" = removed, "additional" = new.
    report = compare(reference_packet, current_packet)

    new_objects = report["additional_objects"]
    removed_objects = report["missing_objects"]
    changed_relations = report["relation_errors"]
    changed_attributes = report["attribute_errors"]
    scene_changed = not report["scene_match"]

    n_ref_objs = max(len(reference_packet.get("objects") or []), 1)
    object_change = (len(new_objects) + len(removed_objects)) / n_ref_objs
    relation_change = 1.0 - float(report["relation_consistency"])
    attribute_change = 1.0 - float(report["attribute_consistency"])
    scene_change = 1.0 if scene_changed else 0.0

    magnitude = (
        w["w_object"] * min(object_change, 1.0)
        + w["w_relation"] * relation_change
        + w["w_attribute"] * attribute_change
        + w["w_scene"] * scene_change
    )
    magnitude = float(min(max(magnitude, 0.0), 1.0))

    num_changes = (
        len(new_objects) + len(removed_objects)
        + len(changed_relations) + len(changed_attributes)
        + (1 if scene_changed else 0)
    )

    return {
        "new_objects": new_objects,
        "removed_objects": removed_objects,
        "changed_relations": changed_relations,
        "changed_attributes": changed_attributes,
        "scene_changed": scene_changed,
        "magnitude": magnitude,
        "num_changes": num_changes,
        "is_empty": num_changes == 0,
    }


class SemanticDelta:
    """OO wrapper around :func:`compute_delta`."""

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights

    def compute(self, reference_packet: Dict, current_packet: Dict) -> Dict:
        return compute_delta(reference_packet, current_packet, self.weights)

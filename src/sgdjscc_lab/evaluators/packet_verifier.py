"""evaluators/packet_verifier.py – Packet Verifier service (ETRI 2차 step 7).

Rx-legal self-verification: compares the semantic packet that *would have been
transmitted* (built from the original/reference frame) against the packet
re-extracted from the *reconstructed* frame, and reduces the comparison to one
severity scalar the regeneration controller can threshold on.

This module is a thin wrapper/service, not a re-implementation: the actual
object / relation / attribute / scene comparison logic lives in
``evaluators/semantic_packet_matcher.compare()`` (Phase 4-A) and is reused
as-is. What this module adds:

- ``severity_score`` — a single, monotonic-in-error scalar folding the
  matcher's per-category terms into one number in ``[0, 1]`` (0 = perfect
  match), so ``controllers/verifier_controller.py`` has one signal to
  threshold on instead of five.
- ``PacketVerifier`` — an OO wrapper that returns a fully dict/JSON
  serialisable report (safe to write straight to ``packet_match_report.json``
  or a CSV row) tagged with an optional ``item_id`` (frame index / segment id).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from sgdjscc_lab.evaluators.semantic_packet_matcher import compare

logger = logging.getLogger(__name__)

# Default weights for folding the matcher report into one severity scalar.
# Mirrors the spirit of semantic_reliability.py's _DEFAULT_PACKET_WEIGHTS but
# expressed as an "error" (higher = worse) composite instead of a "quality"
# composite, since the controller wants a threshold that rises with damage.
DEFAULT_SEVERITY_WEIGHTS: Dict[str, float] = {
    "w_missing": 0.30,      # objects present in the reference but not the recon
    "w_additional": 0.25,   # hallucinated / extra objects in the recon
    "w_relation": 0.20,     # relation-triplet mismatch
    "w_attribute": 0.15,    # attribute (colour/material/size) drift
    "w_scene": 0.10,        # coarse scene-label mismatch
}


def severity_score(
    report: Dict,
    n_reference_objects: int,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Fold a ``semantic_packet_matcher.compare()`` report into one scalar.

    Each term is normalised to ``[0, 1]`` (0 = no error in that category) before
    being weighted, so the result is monotonically non-decreasing as any error
    category worsens. The final result is clamped to ``[0, 1]``: with the
    default weights (summing to 1.0) this clamp never engages, but a custom
    ``weights`` override whose values sum to more than 1.0 could otherwise push
    the composite above 1.0.

    Parameters
    ----------
    report:
        Output of ``semantic_packet_matcher.compare()``.
    n_reference_objects:
        Number of objects in the reference/transmitted packet — used to
        normalise the additional-object count (a raw count has no natural
        upper bound).
    weights:
        Optional override of ``DEFAULT_SEVERITY_WEIGHTS`` (missing keys keep
        their default). Values are not required to sum to 1.0; the result is
        clamped to ``[0, 1]`` regardless.
    """
    w = dict(DEFAULT_SEVERITY_WEIGHTS)
    if weights:
        w.update(weights)

    n_ref = max(int(n_reference_objects), 1)
    missing_term = 1.0 - float(report.get("object_match_rate", 1.0))
    additional_term = min(1.0, float(report.get("additional_object_count", 0) or 0) / n_ref)
    relation_term = 1.0 - float(report.get("relation_consistency", 1.0) or 0.0)
    attribute_term = 1.0 - float(report.get("attribute_consistency", 1.0) or 0.0)
    scene_term = 0.0 if report.get("scene_match", True) else 1.0

    severity = (
        w["w_missing"] * missing_term
        + w["w_additional"] * additional_term
        + w["w_relation"] * relation_term
        + w["w_attribute"] * attribute_term
        + w["w_scene"] * scene_term
    )
    return float(min(1.0, max(0.0, severity)))


class PacketVerifier:
    """Wrapper/service around ``semantic_packet_matcher.compare()``.

    Produces one dict per comparison containing the separated error-type
    fields (missing / additional / relation / attribute / scene) plus a single
    ``severity`` score. The returned dict is plain JSON-native types, so it can
    be written directly to ``packet_match_report.json`` or flattened into a CSV
    row without extra conversion.

    Parameters
    ----------
    severity_weights:
        Optional override of ``DEFAULT_SEVERITY_WEIGHTS``.
    """

    def __init__(
        self,
        severity_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.severity_weights = dict(DEFAULT_SEVERITY_WEIGHTS)
        if severity_weights:
            self.severity_weights.update(severity_weights)

    def verify(
        self,
        reference_packet: Dict,
        reconstructed_packet: Dict,
        item_id: Optional[object] = None,
    ) -> Dict:
        """Compare *reference_packet* (transmitted/original) vs *reconstructed_packet*.

        Returns the full ``compare()`` report plus ``severity`` and ``item_id``
        (e.g. a frame index or segment id, stored verbatim for the caller's
        convenience — this module does not interpret it).
        """
        reference_packet = reference_packet or {}
        reconstructed_packet = reconstructed_packet or {}
        report = compare(reference_packet, reconstructed_packet)
        n_ref = len(reference_packet.get("objects") or [])
        report = dict(report)
        report["severity"] = severity_score(report, n_ref, self.severity_weights)
        report["item_id"] = item_id
        return report

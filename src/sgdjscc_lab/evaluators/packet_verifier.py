"""evaluators/packet_verifier.py – Packet Verifier service (ETRI 2차 step 7 + 5차 step 8).

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

ETRI 5차 addition (presence-backend enhancement, scope note)
--------------------------------------------------------------
``PacketVerifier`` can optionally take a
``evaluators/presence_calibration.py::PresenceCalibrator`` and, when both a
calibrator AND a ``reconstructed_image`` tensor are supplied to ``verify()``,
re-check each ``missing``/``additional`` object against the calibrated
presence backend(s) instead of trusting the packet's (CLIP/caption-derived)
object list alone. **Default behaviour is unchanged**: with no calibrator (the
default) or no image passed, ``verify()`` produces byte-identical output to
2~4차. The enhanced report separates ``raw_clip_result`` (the original,
uncalibrated ``compare()`` numbers) from ``calibrated_presence_result``
(per-object calibration detail) so a consumer never confuses the two, and
tags every report with ``metric_role`` (``"loop_internal"`` by default —
this is the score reconstruction-time control loops may act on; use
``"held_out"`` only for a final, non-gaming evaluation report — see
``pipelines/heldout_remeasurement.py``). This is calibration *structure*, not
a verified accuracy improvement — see docs/etri_strategy.md 5차 구현 결과.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from sgdjscc_lab.evaluators.semantic_packet_matcher import compare

logger = logging.getLogger(__name__)

METRIC_ROLE_LOOP_INTERNAL = "loop_internal"
METRIC_ROLE_HELD_OUT = "held_out"

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
    presence_calibrator:
        Optional ``presence_calibration.PresenceCalibrator`` (ETRI 5차). When
        given, each ``missing``/``additional`` object from the raw comparison
        is re-checked against the calibrated presence backend(s) and the
        report is corrected accordingly (see module docstring).
        ``reconstructed_image``/``gt_metadata`` passed to ``verify()`` are
        forwarded to the calibrator but are NOT required — image-free
        backends (mock/gt) still run without an image (e.g. held-out
        remeasurement from saved packets with no pixels available); only
        image-based backends (clip/owlv2/vqa) then report themselves
        unavailable per object, per
        ``presence_backends.PresenceBackendUnavailableError``. Default
        ``None`` — no behaviour change from 2~4차.
    metric_role:
        Default value stamped onto every report's ``metric_role`` field
        (``"loop_internal"`` or ``"held_out"`` — see module docstring).
        Overridable per call via ``verify(..., metric_role=...)``.
    """

    def __init__(
        self,
        severity_weights: Optional[Dict[str, float]] = None,
        presence_calibrator=None,
        metric_role: str = METRIC_ROLE_LOOP_INTERNAL,
    ) -> None:
        self.severity_weights = dict(DEFAULT_SEVERITY_WEIGHTS)
        if severity_weights:
            self.severity_weights.update(severity_weights)
        self.presence_calibrator = presence_calibrator
        self.metric_role = metric_role

    def verify(
        self,
        reference_packet: Dict,
        reconstructed_packet: Dict,
        item_id: Optional[object] = None,
        reference_image=None,
        reconstructed_image=None,
        gt_metadata: Optional[Dict] = None,
        metric_role: Optional[str] = None,
    ) -> Dict:
        """Compare *reference_packet* (transmitted/original) vs *reconstructed_packet*.

        Returns the full ``compare()`` report plus ``severity``, ``item_id``
        (e.g. a frame index or segment id, stored verbatim for the caller's
        convenience — this module does not interpret it), ``metric_role``, a
        ``raw_clip_result`` snapshot of the uncalibrated numbers, and
        ``calibrated_presence_result`` (``None`` unless a presence calibrator
        was supplied AND at least one object could actually be calibrated —
        see class docstring). ``reconstructed_image``/``gt_metadata`` may both
        be ``None``; calibration still runs whenever a calibrator is
        configured, and each backend decides for itself whether it has enough
        to answer (image-free backends like mock/gt work fine with
        ``reconstructed_image=None``). ``reference_image`` is accepted for
        interface symmetry but not currently used by the calibration path
        (only the reconstructed frame/packet is re-checked).
        """
        reference_packet = reference_packet or {}
        reconstructed_packet = reconstructed_packet or {}
        report = compare(reference_packet, reconstructed_packet)
        n_ref = len(reference_packet.get("objects") or [])
        report = dict(report)
        report["severity"] = severity_score(report, n_ref, self.severity_weights)
        report["item_id"] = item_id
        report["metric_role"] = metric_role or self.metric_role
        report["raw_clip_result"] = {
            "missing_objects": list(report["missing_objects"]),
            "additional_objects": list(report["additional_objects"]),
            "object_match_rate": report["object_match_rate"],
            "severity": report["severity"],
        }
        report["calibrated_presence_result"] = None

        if self.presence_calibrator is not None:
            calibrated_report = self._calibrate_presence(
                reference_packet, reconstructed_packet, report, reconstructed_image, gt_metadata,
            )
            if calibrated_report is not None:
                report = calibrated_report

        return report

    def _calibrate_presence(
        self, reference_packet: Dict, reconstructed_packet: Dict, report: Dict,
        reconstructed_image=None, gt_metadata: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Re-check missing/additional objects with the presence calibrator.

        ``reconstructed_image``/``gt_metadata`` may be ``None`` — they are
        forwarded to the calibrator as-is; only backends that actually need
        them will report themselves unavailable when they're missing.

        Returns ``None`` (leaving the raw report untouched) when no object
        could actually be calibrated — e.g. every backend raised
        :class:`~sgdjscc_lab.evaluators.presence_backends.PresenceBackendUnavailableError`
        (missing dependency/weights/image/GT annotation). Otherwise returns a
        NEW report dict with
        ``missing_objects``/``additional_objects``/counts/``object_match_rate``/
        ``severity`` recomputed and ``calibrated_presence_result`` filled in.
        """
        from sgdjscc_lab.evaluators.presence_backends import PresenceBackendUnavailableError

        calibrator = self.presence_calibrator
        orig_objs = set(reference_packet.get("objects") or [])
        soft_bar = calibrator.threshold - calibrator.uncertain_band
        per_object = []

        still_missing = []
        for obj in report["missing_objects"]:
            try:
                calibrated = calibrator.calibrate(
                    obj, image=reconstructed_image, packet=reconstructed_packet, gt_metadata=gt_metadata,
                )
            except PresenceBackendUnavailableError as exc:
                logger.debug("Presence calibration unavailable for missing object %r: %s", obj, exc)
                still_missing.append(obj)
                continue
            per_object.append(calibrated.to_dict())
            # An object the packet calls "missing" stays missing unless the
            # calibrated backend(s) actually find it (hysteresis-compatible:
            # a confidence within the uncertain band of the threshold also
            # counts as "found", mirroring object_preservation.py's band).
            if calibrated.final_present or calibrated.final_confidence >= soft_bar:
                continue
            still_missing.append(obj)

        still_additional = []
        for obj in report["additional_objects"]:
            try:
                calibrated = calibrator.calibrate(
                    obj, image=reconstructed_image, packet=reconstructed_packet, gt_metadata=gt_metadata,
                )
            except PresenceBackendUnavailableError as exc:
                logger.debug("Presence calibration unavailable for additional object %r: %s", obj, exc)
                still_additional.append(obj)
                continue
            per_object.append(calibrated.to_dict())
            # An object the packet calls "additional"/hallucinated stays so
            # unless the calibrated backend(s) disagree it's actually there.
            if not calibrated.final_present:
                continue
            still_additional.append(obj)

        if not per_object:
            return None

        n_ref = max(len(orig_objs), 1)
        matched_count = len(orig_objs) - len(still_missing)
        object_match_rate = float(matched_count / n_ref) if orig_objs else 1.0

        calibrated_report = dict(report)
        calibrated_report["missing_objects"] = sorted(still_missing)
        calibrated_report["missing_object_count"] = len(still_missing)
        calibrated_report["additional_objects"] = sorted(still_additional)
        calibrated_report["additional_object_count"] = len(still_additional)
        calibrated_report["object_match_rate"] = object_match_rate
        calibrated_report["severity"] = severity_score(calibrated_report, n_ref, self.severity_weights)
        calibrated_report["calibrated_presence_result"] = per_object
        return calibrated_report

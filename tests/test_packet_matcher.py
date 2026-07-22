"""tests/test_packet_matcher.py – Phase 4-A packet-aware verifier tests (offline)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.guidance.semantic_packet_extractor import build_packet  # noqa: E402


def _orig():
    return build_packet(
        caption="a red car next to a black dog on a street",
        objects=["car", "dog"],
        scene="street scene",
        relations=[{"subject": "car", "predicate": "next to", "object": "dog"}],
        attributes={"car": ["red"], "dog": ["black"]},
    )


# ─────────────────────────────────────────────────────────────────────────────
# semantic_packet_matcher.compare
# ─────────────────────────────────────────────────────────────────────────────

class TestMatcherSchema:
    def test_report_keys(self):
        from sgdjscc_lab.evaluators.semantic_packet_matcher import compare
        rep = compare(_orig(), _orig())
        for k in ("missing_objects", "additional_objects", "missing_object_count",
                  "additional_object_count", "object_match_rate", "relation_errors",
                  "relation_error_count", "relation_consistency", "attribute_errors",
                  "attribute_error_count", "attribute_consistency", "scene_match",
                  "segmentation_consistency"):
            assert k in rep

    def test_identical_packets_perfect(self):
        from sgdjscc_lab.evaluators.semantic_packet_matcher import compare
        rep = compare(_orig(), _orig())
        assert rep["missing_object_count"] == 0
        assert rep["additional_object_count"] == 0
        assert rep["relation_consistency"] == 1.0
        assert rep["attribute_consistency"] == 1.0
        assert rep["scene_match"] is True
        assert rep["object_match_rate"] == 1.0

    def test_missing_and_additional(self):
        from sgdjscc_lab.evaluators.semantic_packet_matcher import compare
        recon = build_packet(objects=["car", "tree"], scene="street scene")
        rep = compare(_orig(), recon)
        assert "dog" in rep["missing_objects"]
        assert "tree" in rep["additional_objects"]

    def test_scene_mismatch(self):
        from sgdjscc_lab.evaluators.semantic_packet_matcher import compare
        recon = build_packet(objects=["car", "dog"], scene="beach")
        rep = compare(_orig(), recon)
        assert rep["scene_match"] is False


# ─────────────────────────────────────────────────────────────────────────────
# relation / attribute consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestRelationConsistency:
    def test_perfect(self):
        from sgdjscc_lab.evaluators.relation_consistency import relation_consistency
        rels = [{"subject": "car", "predicate": "near", "object": "tree"}]
        out = relation_consistency(rels, rels)
        assert out["score"] == 1.0
        assert out["missing"] == [] and out["extra"] == []

    def test_empty_is_vacuously_consistent(self):
        from sgdjscc_lab.evaluators.relation_consistency import relation_consistency
        assert relation_consistency([], [])["score"] == 1.0

    def test_partial_overlap(self):
        from sgdjscc_lab.evaluators.relation_consistency import relation_consistency
        a = [{"subject": "car", "predicate": "near", "object": "tree"}]
        b = [{"subject": "car", "predicate": "on", "object": "road"}]
        assert relation_consistency(a, b)["score"] == 0.0


class TestAttributeConsistency:
    def test_perfect(self):
        from sgdjscc_lab.evaluators.attribute_consistency import attribute_consistency
        a = {"car": ["red"]}
        assert attribute_consistency(a, a)["score"] == 1.0

    def test_drift_recorded(self):
        from sgdjscc_lab.evaluators.attribute_consistency import attribute_consistency
        out = attribute_consistency({"car": ["red"]}, {"car": ["blue"]})
        assert out["score"] == 0.0
        assert out["errors"][0]["object"] == "car"

    def test_no_shared_objects_vacuous(self):
        from sgdjscc_lab.evaluators.attribute_consistency import attribute_consistency
        assert attribute_consistency({"car": ["red"]}, {"dog": ["black"]})["score"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# segmentation consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestPacketVerifier:
    """evaluators/packet_verifier.py – wrapper/service + severity score (ETRI 2차)."""

    def test_report_separates_error_types(self):
        from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
        orig = _orig()
        recon = build_packet(
            objects=["car", "tree"], scene="beach",
            relations=[{"subject": "car", "predicate": "on", "object": "road"}],
            attributes={"car": ["blue"]},
        )
        rep = PacketVerifier().verify(orig, recon, item_id=3)
        assert rep["item_id"] == 3
        assert "dog" in rep["missing_objects"]
        assert "tree" in rep["additional_objects"]
        assert rep["relation_error_count"] > 0
        assert rep["attribute_error_count"] > 0
        assert rep["scene_match"] is False
        assert 0.0 <= rep["severity"] <= 1.0

    def test_perfect_match_zero_severity(self):
        from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
        rep = PacketVerifier().verify(_orig(), _orig())
        assert rep["severity"] == pytest.approx(0.0, abs=1e-9)

    def test_severity_increases_with_missing_objects(self):
        from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
        orig = build_packet(objects=["car", "dog", "tree", "bus"], scene="street scene")
        v = PacketVerifier()
        sev_none_missing = v.verify(orig, build_packet(objects=["car", "dog", "tree", "bus"], scene="street scene"))["severity"]
        sev_one_missing = v.verify(orig, build_packet(objects=["car", "dog", "tree"], scene="street scene"))["severity"]
        sev_two_missing = v.verify(orig, build_packet(objects=["car", "dog"], scene="street scene"))["severity"]
        assert sev_none_missing < sev_one_missing < sev_two_missing

    def test_severity_increases_with_additional_objects(self):
        # n_ref=3 gives the additional-object term room to grow before the
        # min(1.0, ...) normalisation cap saturates it.
        from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
        orig = build_packet(objects=["car", "dog", "tree"], scene="street scene")
        v = PacketVerifier()
        sev_0 = v.verify(orig, build_packet(objects=["car", "dog", "tree"], scene="street scene"))["severity"]
        sev_1 = v.verify(orig, build_packet(objects=["car", "dog", "tree", "bus"], scene="street scene"))["severity"]
        sev_2 = v.verify(orig, build_packet(objects=["car", "dog", "tree", "bus", "bike"], scene="street scene"))["severity"]
        assert sev_0 < sev_1 < sev_2

    def test_severity_increases_with_relation_and_attribute_errors(self):
        from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
        orig = _orig()
        v = PacketVerifier()
        sev_good = v.verify(orig, _orig())["severity"]
        recon_bad_attr = build_packet(
            caption="a red car next to a black dog on a street",
            objects=["car", "dog"], scene="street scene",
            relations=[{"subject": "car", "predicate": "next to", "object": "dog"}],
            attributes={"car": ["blue"], "dog": ["black"]},
        )
        sev_attr = v.verify(orig, recon_bad_attr)["severity"]
        recon_bad_rel = build_packet(
            caption="a red car next to a black dog on a street",
            objects=["car", "dog"], scene="street scene",
            relations=[{"subject": "dog", "predicate": "chases", "object": "car"}],
            attributes={"car": ["red"], "dog": ["black"]},
        )
        sev_rel = v.verify(orig, recon_bad_rel)["severity"]
        assert sev_good < sev_attr
        assert sev_good < sev_rel

    def test_severity_weight_override(self):
        from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
        orig = build_packet(objects=["car", "dog"], scene="street scene")
        recon = build_packet(objects=["car"], scene="street scene")
        default_sev = PacketVerifier().verify(orig, recon)["severity"]
        boosted_sev = PacketVerifier(severity_weights={"w_missing": 1.0}).verify(orig, recon)["severity"]
        assert boosted_sev > default_sev

    def test_severity_clamped_to_one_with_heavy_custom_weights(self):
        # Custom weights are not required to sum to 1.0; a total-mismatch packet
        # with weights summing well above 1.0 must still clamp to 1.0, not
        # silently exceed the documented [0, 1] range.
        from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
        orig = _orig()
        recon = build_packet(
            objects=["tree", "bus"], scene="beach",
            relations=[{"subject": "bus", "predicate": "near", "object": "tree"}],
            attributes={"tree": ["green"]},
        )
        heavy_weights = {
            "w_missing": 1.0, "w_additional": 1.0, "w_relation": 1.0,
            "w_attribute": 1.0, "w_scene": 1.0,
        }
        rep = PacketVerifier(severity_weights=heavy_weights).verify(orig, recon)
        assert rep["severity"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# PacketVerifier presence-backend enhancement (ETRI 5차, step 8)
# ─────────────────────────────────────────────────────────────────────────────

def _always_present_calibrator():
    """A PresenceCalibrator whose sole backend disagrees with every packet by
    always saying "present" — used to prove the calibration path can actually
    override the raw comparison (a plain MockPresenceBackend never can, since
    it derives its answer from the same packet being verified)."""
    from sgdjscc_lab.evaluators.presence_backends import PresenceBackend, PresenceResult
    from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator

    class _AlwaysPresent(PresenceBackend):
        backend_name = "clip"
        def check(self, object_name, image=None, packet=None, gt_metadata=None):
            return PresenceResult(object_name=object_name, present=True, confidence=0.95, backend="clip")

    return PresenceCalibrator({"clip": _AlwaysPresent()}, mode="clip_only")


def _always_absent_calibrator():
    from sgdjscc_lab.evaluators.presence_backends import PresenceBackend, PresenceResult
    from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator

    class _AlwaysAbsent(PresenceBackend):
        backend_name = "clip"
        def check(self, object_name, image=None, packet=None, gt_metadata=None):
            return PresenceResult(object_name=object_name, present=False, confidence=0.05, backend="clip")

    return PresenceCalibrator({"clip": _AlwaysAbsent()}, mode="clip_only")


class TestPacketVerifierPresenceCalibration:
    def test_default_unchanged_no_calibrator(self):
        """No presence_calibrator at all → byte-identical to 2~4차 (plus the
        new always-present metric_role/raw_clip_result/calibrated_presence_result
        bookkeeping fields, which mirror the top-level values / stay None)."""
        from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
        recon = build_packet(objects=["car"], scene="street scene")
        rep = PacketVerifier().verify(_orig(), recon)
        assert rep["missing_objects"] == ["dog"]
        assert rep["metric_role"] == "loop_internal"
        assert rep["calibrated_presence_result"] is None
        assert rep["raw_clip_result"]["missing_objects"] == ["dog"]
        assert rep["raw_clip_result"]["severity"] == rep["severity"]

    def test_image_required_backend_without_image_leaves_report_unchanged(self):
        """A calibrator whose only backend NEEDS an image (mirrors
        ClipPresenceBackend) reports itself unavailable per-object when no
        ``reconstructed_image`` is passed — the report must stay unchanged
        rather than crash or fabricate an answer."""
        from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
        from sgdjscc_lab.evaluators.presence_backends import PresenceBackend, PresenceBackendUnavailableError
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator

        class _NeedsImage(PresenceBackend):
            backend_name = "clip"
            def check(self, object_name, image=None, packet=None, gt_metadata=None):
                if image is None:
                    raise PresenceBackendUnavailableError("needs an image")
                raise AssertionError("should never be reached in this test")

        recon = build_packet(objects=["car"], scene="street scene")
        v = PacketVerifier(presence_calibrator=PresenceCalibrator({"clip": _NeedsImage()}, mode="clip_only"))
        rep = v.verify(_orig(), recon)   # reconstructed_image omitted
        assert rep["missing_objects"] == ["dog"]
        assert rep["calibrated_presence_result"] is None

    def test_image_free_backend_calibrates_without_image(self):
        """The whole point of image-free backends (mock/gt): calibration must
        still run — and be able to correct the report — even when
        ``reconstructed_image`` is never supplied (e.g. held-out
        remeasurement from saved packets with no pixels available)."""
        from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
        recon = build_packet(objects=["car"], scene="street scene")
        v = PacketVerifier(presence_calibrator=_always_present_calibrator())
        rep = v.verify(_orig(), recon)   # reconstructed_image omitted — backend doesn't need it
        assert rep["missing_objects"] == []
        assert rep["raw_clip_result"]["missing_objects"] == ["dog"]
        assert rep["calibrated_presence_result"] is not None

    def test_gt_metadata_forwarded_to_gt_backend(self):
        """PacketVerifier.verify(gt_metadata=...) must reach GtPresenceBackend,
        even without an image."""
        from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
        from sgdjscc_lab.evaluators.presence_backends import GtPresenceBackend
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator

        recon = build_packet(objects=["car"], scene="street scene")
        cal = PresenceCalibrator({"gt": GtPresenceBackend()}, mode="gt_only")   # no default metadata
        v = PacketVerifier(presence_calibrator=cal)
        rep = v.verify(_orig(), recon, gt_metadata={"dog": True})
        assert rep["missing_objects"] == []   # GT (per-call) says dog IS present → corrected
        assert rep["calibrated_presence_result"][0]["per_backend"][0]["backend"] == "gt"

    def test_calibration_removes_false_missing(self):
        import torch
        from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
        recon = build_packet(objects=["car"], scene="street scene")
        v = PacketVerifier(presence_calibrator=_always_present_calibrator())
        rep = v.verify(_orig(), recon, reconstructed_image=torch.rand(1, 3, 4, 4))
        assert rep["missing_objects"] == []                       # calibration disagreed → corrected
        assert rep["raw_clip_result"]["missing_objects"] == ["dog"]   # raw snapshot preserved
        assert rep["object_match_rate"] == pytest.approx(1.0)
        assert rep["severity"] < rep["raw_clip_result"]["severity"]
        assert rep["calibrated_presence_result"] is not None
        assert rep["calibrated_presence_result"][0]["object_name"] == "dog"

    def test_calibration_removes_false_additional(self):
        import torch
        from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
        orig = build_packet(objects=["car"], scene="street scene")
        recon = build_packet(objects=["car", "dog"], scene="street scene")
        v = PacketVerifier(presence_calibrator=_always_absent_calibrator())
        rep = v.verify(orig, recon, reconstructed_image=torch.rand(1, 3, 4, 4))
        assert rep["additional_objects"] == []                    # calibration says dog isn't really there
        assert rep["raw_clip_result"]["additional_objects"] == ["dog"]
        assert rep["severity"] < rep["raw_clip_result"]["severity"]

    def test_calibration_confirms_missing_when_backend_agrees(self):
        """A calibrator that agrees an object is absent must leave the report
        as-is (missing stays missing) — calibration should not fabricate
        false negatives just because it ran."""
        import torch
        from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
        recon = build_packet(objects=["car"], scene="street scene")
        v = PacketVerifier(presence_calibrator=_always_absent_calibrator())
        rep = v.verify(_orig(), recon, reconstructed_image=torch.rand(1, 3, 4, 4))
        assert rep["missing_objects"] == ["dog"]
        assert rep["severity"] == pytest.approx(rep["raw_clip_result"]["severity"])

    def test_metric_role_override_per_call(self):
        from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
        v = PacketVerifier(metric_role="loop_internal")
        rep = v.verify(_orig(), _orig(), metric_role="held_out")
        assert rep["metric_role"] == "held_out"

    def test_report_stays_json_serialisable_with_calibration(self):
        import json
        import torch
        from sgdjscc_lab.evaluators.packet_verifier import PacketVerifier
        recon = build_packet(objects=["car"], scene="street scene")
        v = PacketVerifier(presence_calibrator=_always_present_calibrator())
        rep = v.verify(_orig(), recon, reconstructed_image=torch.rand(1, 3, 4, 4))
        json.dumps(rep)   # must not raise (no stray tensors/objects)


class TestSegmentationConsistency:
    def test_none_when_absent(self):
        from sgdjscc_lab.evaluators.semantic_packet_matcher import segmentation_consistency
        assert segmentation_consistency(None, {"class_histogram": {}}) is None

    def test_identical_histograms(self):
        from sgdjscc_lab.evaluators.semantic_packet_matcher import segmentation_consistency
        seg = {"class_histogram": {"sky": 0.5, "ground": 0.5}}
        assert segmentation_consistency(seg, seg) == pytest.approx(1.0)

    def test_disjoint_histograms(self):
        from sgdjscc_lab.evaluators.semantic_packet_matcher import segmentation_consistency
        a = {"class_histogram": {"sky": 1.0}}
        b = {"class_histogram": {"road": 1.0}}
        assert segmentation_consistency(a, b) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# SRS packet-aware extension (score_packet, no CLIP needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestSRSPacket:
    def test_score_packet_keys(self):
        from sgdjscc_lab.evaluators.semantic_reliability import SemanticReliabilityEvaluator
        out = SemanticReliabilityEvaluator().score_packet(_orig(), _orig(), srs_base=0.8)
        for k in ("srs_packet", "packet_composite", "relation_consistency",
                  "attribute_consistency", "object_match_rate", "error_report"):
            assert k in out

    def test_identical_high_composite(self):
        from sgdjscc_lab.evaluators.semantic_reliability import SemanticReliabilityEvaluator
        out = SemanticReliabilityEvaluator().score_packet(_orig(), _orig(), srs_base=None)
        # All consistency terms are 1.0, no additional objects → composite ~ sum of weights.
        assert out["packet_composite"] > 0.9

    def test_degraded_lower_than_perfect(self):
        from sgdjscc_lab.evaluators.semantic_reliability import SemanticReliabilityEvaluator
        ev = SemanticReliabilityEvaluator()
        good = ev.score_packet(_orig(), _orig(), srs_base=None)["srs_packet"]
        bad_recon = build_packet(objects=["car"], scene="beach")
        bad = ev.score_packet(_orig(), bad_recon, srs_base=None)["srs_packet"]
        assert bad < good

    def test_blend_with_base(self):
        from sgdjscc_lab.evaluators.semantic_reliability import SemanticReliabilityEvaluator
        ev = SemanticReliabilityEvaluator(packet_blend=0.5)
        out = ev.score_packet(_orig(), _orig(), srs_base=0.0)
        # srs_packet = 0.5*0 + 0.5*composite
        assert out["srs_packet"] == pytest.approx(0.5 * out["packet_composite"])

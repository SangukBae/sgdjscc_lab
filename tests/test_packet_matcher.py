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

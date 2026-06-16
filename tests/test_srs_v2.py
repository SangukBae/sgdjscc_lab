"""tests/test_srs_v2.py – Phase 5-C SRS-v2 + VQA hallucination tests (offline)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.guidance.semantic_packet_extractor import build_packet  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# combine_srs_v2 (pure)
# ─────────────────────────────────────────────────────────────────────────────

class TestCombineSRSv2:
    def test_all_layers(self):
        from sgdjscc_lab.evaluators.semantic_reliability_v2 import combine_srs_v2
        out = combine_srs_v2({
            "srs_base": 0.8, "srs_packet": 0.7,
            "temporal_srs": 0.6, "hallucination_score": 0.2,
        })
        assert out["n_layers"] == 4
        assert 0.0 <= out["srs_v2"] <= 1.0

    def test_partial_layers_renormalised(self):
        from sgdjscc_lab.evaluators.semantic_reliability_v2 import combine_srs_v2
        out = combine_srs_v2({"srs_base": 0.8})
        # Only base present → srs_v2 == srs_base (weights renormalised).
        assert out["srs_v2"] == pytest.approx(0.8)
        assert out["n_layers"] == 1

    def test_no_layers_returns_none(self):
        from sgdjscc_lab.evaluators.semantic_reliability_v2 import combine_srs_v2
        out = combine_srs_v2({})
        assert out["srs_v2"] is None

    def test_hallucination_penalises(self):
        from sgdjscc_lab.evaluators.semantic_reliability_v2 import combine_srs_v2
        clean = combine_srs_v2({"srs_base": 0.8, "hallucination_score": 0.0})["srs_v2"]
        dirty = combine_srs_v2({"srs_base": 0.8, "hallucination_score": 0.9})["srs_v2"]
        assert dirty < clean


# ─────────────────────────────────────────────────────────────────────────────
# SemanticReliabilityV2Evaluator (with injected base_result, no CLIP)
# ─────────────────────────────────────────────────────────────────────────────

class TestSRSv2Evaluator:
    def test_evaluate_with_base_result(self):
        from sgdjscc_lab.evaluators.semantic_reliability_v2 import SemanticReliabilityV2Evaluator
        ev = SemanticReliabilityV2Evaluator()
        base = {"srs_base": 0.8, "srs_packet": 0.7, "hallucination_score": 0.1}
        out = ev.evaluate(
            torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8),
            base_result=base, temporal_metrics={"temporal_srs": 0.6},
        )
        assert out["srs_v2"] is not None
        assert "base_result" in out
        assert out["n_layers"] == 4

    def test_evaluate_with_vqa_injected(self):
        from sgdjscc_lab.evaluators.semantic_reliability_v2 import SemanticReliabilityV2Evaluator
        from sgdjscc_lab.evaluators.hallucination_vqa import VQAHallucinationEvaluator

        # VQA that always confirms 'dog' (hallucinated in recon).
        vqa = VQAHallucinationEvaluator(vqa_fn=lambda img, q: "yes" if "dog" in q else "no")
        ev = SemanticReliabilityV2Evaluator(vqa_evaluator=vqa)
        recon_packet = build_packet(objects=["dog"], scene="s")
        out = ev.evaluate(
            torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8),
            base_result={"srs_base": 0.8}, recon_packet=recon_packet,
        )
        assert "hallucination_score" in out


# ─────────────────────────────────────────────────────────────────────────────
# VQAHallucinationEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class TestVQAHallucination:
    def test_detects_hallucinated_object(self):
        from sgdjscc_lab.evaluators.hallucination_vqa import VQAHallucinationEvaluator

        # VQA says the object is present only in the reconstruction (id by mean).
        def vqa_fn(img, q):
            in_recon = float(img.mean()) > 0.5
            return "yes" if in_recon else "no"

        ev = VQAHallucinationEvaluator(vqa_fn=vqa_fn)
        original = torch.zeros(1, 3, 8, 8)        # mean 0 → object "absent"
        recon = torch.ones(1, 3, 8, 8)            # mean 1 → object "present"
        out = ev.evaluate(original, recon, objects=["dog"])
        assert out["method"] == "vqa"
        assert "dog" in out["hallucinated_objects"]
        assert out["hallucination_score"] > 0.0

    def test_no_vqa_uses_clip_fallback_schema(self):
        from sgdjscc_lab.evaluators.hallucination_vqa import VQAHallucinationEvaluator

        # Inject a mock CLIP-based fallback to avoid loading real CLIP.
        class _MockFallback:
            def evaluate(self, o, r):
                return {"hallucination_score": 0.0, "extra_objects": []}

        ev = VQAHallucinationEvaluator(vqa_fn=None)
        ev._fallback = _MockFallback()
        out = ev.evaluate(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8))
        assert out["method"] == "clip_fallback"
        assert "hallucination_score" in out


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 regression guards (contracts must remain intact)
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase4Regression:
    def test_packet_schema_intact(self):
        p = build_packet(caption="a cat", objects=["cat"], scene="indoor scene")
        for key in ("caption", "scene", "objects", "relations", "attributes",
                    "importance", "meta"):
            assert key in p

    def test_srs_still_reports_base_and_packet(self):
        from sgdjscc_lab.evaluators.semantic_reliability import SemanticReliabilityEvaluator
        ev = SemanticReliabilityEvaluator()
        out = ev.score_packet(
            build_packet(objects=["cat"], scene="s"),
            build_packet(objects=["cat"], scene="s"),
            srs_base=0.8,
        )
        assert "srs_packet" in out

    def test_temporal_summary_shape_intact(self):
        from sgdjscc_lab.evaluators.temporal_consistency import evaluate_sequence
        out = evaluate_sequence([
            {"srs": 0.8, "orig_packet": build_packet(objects=["cat"]),
             "recon_packet": build_packet(objects=["cat"])},
        ])
        for key in ("n_frames", "temporal_srs", "srs_flicker",
                    "object_identity_consistency", "temporal_hallucination_rate"):
            assert key in out

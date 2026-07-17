"""tests/test_evaluators.py – Unit tests for Phase 3 evaluators.

No GPU or SGDJSCC checkpoints required.
CLIP / LPIPS tests run on CPU (slow on first run due to model download).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def pair_1x3x64x64():
    """Return (original, reconstructed) as [1, 3, 64, 64] tensors in [0, 1]."""
    torch.manual_seed(42)
    original      = torch.rand(1, 3, 64, 64)
    reconstructed = torch.rand(1, 3, 64, 64)
    return original, reconstructed


@pytest.fixture()
def identical_pair_1x3x64x64():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    return x, x.clone()


@pytest.fixture()
def pair_batch_2():
    torch.manual_seed(7)
    original      = torch.rand(2, 3, 64, 64)
    reconstructed = torch.rand(2, 3, 64, 64)
    return original, reconstructed


# ─────────────────────────────────────────────────────────────────────────────
# PSNR
# ─────────────────────────────────────────────────────────────────────────────

class TestComputePSNR:
    def test_identical_images_returns_inf(self, identical_pair_1x3x64x64):
        from sgdjscc_lab.evaluators.quality import compute_psnr
        o, r = identical_pair_1x3x64x64
        assert compute_psnr(o, r) == float("inf")

    def test_returns_positive_dB(self, pair_1x3x64x64):
        from sgdjscc_lab.evaluators.quality import compute_psnr
        o, r = pair_1x3x64x64
        psnr = compute_psnr(o, r)
        assert psnr > 0

    def test_shape_mismatch_raises(self):
        from sgdjscc_lab.evaluators.quality import compute_psnr
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_psnr(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 32, 32))

    def test_accepts_3d_input(self):
        from sgdjscc_lab.evaluators.quality import compute_psnr
        o = torch.rand(3, 64, 64)
        r = torch.rand(3, 64, 64)
        psnr = compute_psnr(o, r)
        assert isinstance(psnr, float)

    def test_batch_of_two(self, pair_batch_2):
        from sgdjscc_lab.evaluators.quality import compute_psnr
        o, r = pair_batch_2
        psnr = compute_psnr(o, r)
        assert isinstance(psnr, float)

    def test_high_noise_lower_psnr_than_low_noise(self, identical_pair_1x3x64x64):
        from sgdjscc_lab.evaluators.quality import compute_psnr
        o, _ = identical_pair_1x3x64x64
        r_low_noise  = o + 0.01 * torch.randn_like(o)
        r_high_noise = o + 0.20 * torch.randn_like(o)
        assert compute_psnr(o, r_low_noise.clamp(0, 1)) > compute_psnr(o, r_high_noise.clamp(0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# SSIM
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSSIM:
    def test_identical_images_returns_near_one(self, identical_pair_1x3x64x64):
        from sgdjscc_lab.evaluators.quality import compute_ssim
        o, r = identical_pair_1x3x64x64
        ssim = compute_ssim(o, r)
        assert ssim > 0.99

    def test_returns_in_valid_range(self, pair_1x3x64x64):
        from sgdjscc_lab.evaluators.quality import compute_ssim
        o, r = pair_1x3x64x64
        ssim = compute_ssim(o, r)
        # SSIM is formally in [-1, 1]; for natural image pairs ≥ -0.1
        assert -1.0 <= ssim <= 1.0

    def test_shape_mismatch_raises(self):
        from sgdjscc_lab.evaluators.quality import compute_ssim
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_ssim(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 32, 32))


# ─────────────────────────────────────────────────────────────────────────────
# LPIPS
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeLPIPS:
    def test_returns_non_negative_float(self, pair_1x3x64x64):
        lpips = pytest.importorskip("lpips")
        from sgdjscc_lab.evaluators.quality import compute_lpips
        o, r = pair_1x3x64x64
        val = compute_lpips(o, r, net="vgg", device=torch.device("cpu"))
        assert isinstance(val, float)
        assert val >= 0.0

    def test_identical_images_near_zero(self, identical_pair_1x3x64x64):
        lpips = pytest.importorskip("lpips")
        from sgdjscc_lab.evaluators.quality import compute_lpips
        o, r = identical_pair_1x3x64x64
        val = compute_lpips(o, r, device=torch.device("cpu"))
        assert val < 0.1

    def test_shape_mismatch_raises(self):
        lpips = pytest.importorskip("lpips")
        from sgdjscc_lab.evaluators.quality import compute_lpips
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_lpips(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 32, 32))


# ─────────────────────────────────────────────────────────────────────────────
# QualityEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityEvaluator:
    def test_evaluate_returns_required_keys(self, pair_1x3x64x64):
        from sgdjscc_lab.evaluators.quality import QualityEvaluator
        o, r = pair_1x3x64x64
        ev = QualityEvaluator(use_lpips=False)
        result = ev.evaluate(o, r)
        assert "psnr" in result
        assert "ssim" in result
        assert "lpips" in result

    def test_evaluate_with_lpips(self, pair_1x3x64x64):
        lpips = pytest.importorskip("lpips")
        from sgdjscc_lab.evaluators.quality import QualityEvaluator
        o, r = pair_1x3x64x64
        ev = QualityEvaluator(use_lpips=True, device=torch.device("cpu"))
        result = ev.evaluate(o, r)
        assert result["lpips"] is not None
        assert isinstance(result["lpips"], float)

    def test_evaluate_without_lpips_returns_none(self, pair_1x3x64x64):
        from sgdjscc_lab.evaluators.quality import QualityEvaluator
        o, r = pair_1x3x64x64
        ev = QualityEvaluator(use_lpips=False)
        result = ev.evaluate(o, r)
        assert result["lpips"] is None


# ─────────────────────────────────────────────────────────────────────────────
# CLIPScoreEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIPScoreEvaluator:
    def test_image_image_score_range(self, pair_1x3x64x64):
        clip = pytest.importorskip("clip")
        from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator
        o, r = pair_1x3x64x64
        ev = CLIPScoreEvaluator(device=torch.device("cpu"))
        score = ev.image_image_score(o, r)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_identical_images_high_score(self, identical_pair_1x3x64x64):
        clip = pytest.importorskip("clip")
        from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator
        o, r = identical_pair_1x3x64x64
        ev = CLIPScoreEvaluator(device=torch.device("cpu"))
        score = ev.image_image_score(o, r)
        assert score > 0.95

    def test_text_image_score_range(self, pair_1x3x64x64):
        clip = pytest.importorskip("clip")
        from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator
        _, r = pair_1x3x64x64
        ev = CLIPScoreEvaluator(device=torch.device("cpu"))
        score = ev.text_image_score(["a photo of a cat"], r)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_text_image_length_mismatch_raises(self, pair_1x3x64x64):
        clip = pytest.importorskip("clip")
        from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator
        _, r = pair_1x3x64x64
        ev = CLIPScoreEvaluator(device=torch.device("cpu"))
        with pytest.raises(ValueError, match="text_list length"):
            ev.text_image_score(["cat", "dog"], r)   # 2 texts, batch=1

    def test_image_shape_mismatch_raises(self, pair_1x3x64x64):
        clip = pytest.importorskip("clip")
        from sgdjscc_lab.evaluators.clip_score import CLIPScoreEvaluator
        ev = CLIPScoreEvaluator(device=torch.device("cpu"))
        with pytest.raises(ValueError, match="Shape mismatch"):
            ev.image_image_score(torch.rand(1, 3, 64, 64), torch.rand(1, 3, 32, 32))


# ─────────────────────────────────────────────────────────────────────────────
# ObjectPreservationEvaluator (output schema only – no CLIP required)
# ─────────────────────────────────────────────────────────────────────────────

class TestObjectPreservationSchema:
    """Verify output dict schema using a mock CLIP evaluator."""

    def _make_mock_clip(self, sim_value: float = 0.3):
        """Return a CLIPScoreEvaluator-compatible mock."""
        import torch
        import torch.nn.functional as F

        class _MockCLIP:
            def _load(self): pass
            def _encode_images(self, tensor):
                n = tensor.shape[0]
                return F.normalize(torch.ones(n, 512), dim=-1)
            def _encode_texts(self, texts):
                n = len(texts)
                # Return uniform embeddings so all similarities equal sim_value
                feats = torch.ones(n, 512)
                return F.normalize(feats, dim=-1)

        return _MockCLIP()

    def test_evaluate_returns_required_keys(self):
        from sgdjscc_lab.evaluators.object_preservation import ObjectPreservationEvaluator
        o = torch.rand(1, 3, 64, 64)
        r = torch.rand(1, 3, 64, 64)
        ev = ObjectPreservationEvaluator(
            clip_evaluator=self._make_mock_clip(),
            vocabulary=["cat", "dog", "car"],
        )
        result = ev.evaluate(o, r)
        assert "preservation_rate" in result
        assert "matched_objects" in result
        assert "missing_objects" in result
        assert "original_count" in result
        assert "reconstructed_count" in result

    def test_preservation_rate_in_0_1(self):
        from sgdjscc_lab.evaluators.object_preservation import ObjectPreservationEvaluator
        o = torch.rand(1, 3, 64, 64)
        r = torch.rand(1, 3, 64, 64)
        ev = ObjectPreservationEvaluator(
            clip_evaluator=self._make_mock_clip(),
            vocabulary=["cat", "dog"],
        )
        result = ev.evaluate(o, r)
        assert 0.0 <= result["preservation_rate"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# HallucinationEvaluator (output schema)
# ─────────────────────────────────────────────────────────────────────────────

class TestHallucinationSchema:
    def _make_mock_clip(self):
        import torch
        import torch.nn.functional as F

        class _MockCLIP:
            def _load(self): pass
            def _encode_images(self, tensor):
                n = tensor.shape[0]
                return F.normalize(torch.ones(n, 512), dim=-1)
            def _encode_texts(self, texts):
                n = len(texts)
                return F.normalize(torch.ones(n, 512), dim=-1)

        return _MockCLIP()

    def test_evaluate_returns_required_keys(self):
        from sgdjscc_lab.evaluators.hallucination import HallucinationEvaluator
        from sgdjscc_lab.evaluators.object_preservation import ObjectPreservationEvaluator
        o = torch.rand(1, 3, 64, 64)
        r = torch.rand(1, 3, 64, 64)
        mock_clip = self._make_mock_clip()
        obj_ev = ObjectPreservationEvaluator(
            clip_evaluator=mock_clip, vocabulary=["cat", "dog"]
        )
        ev = HallucinationEvaluator(clip_evaluator=mock_clip, vocabulary=["cat", "dog"])
        ev._obj_eval = obj_ev
        result = ev.evaluate(o, r)
        assert "hallucination_score" in result
        assert "extra_objects" in result
        assert "notes" in result

    def test_hallucination_score_non_negative(self):
        from sgdjscc_lab.evaluators.hallucination import HallucinationEvaluator
        from sgdjscc_lab.evaluators.object_preservation import ObjectPreservationEvaluator
        o = torch.rand(1, 3, 64, 64)
        r = torch.rand(1, 3, 64, 64)
        mock_clip = self._make_mock_clip()
        obj_ev = ObjectPreservationEvaluator(
            clip_evaluator=mock_clip, vocabulary=["cat"]
        )
        ev = HallucinationEvaluator(clip_evaluator=mock_clip, vocabulary=["cat"])
        ev._obj_eval = obj_ev
        result = ev.evaluate(o, r)
        assert result["hallucination_score"] >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SemanticReliabilityEvaluator (output schema)
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticReliabilitySchema:
    def _make_mock_srs(self):
        """Return a SemanticReliabilityEvaluator with mocked sub-evaluators."""
        import torch
        import torch.nn.functional as F
        from sgdjscc_lab.evaluators.semantic_reliability import SemanticReliabilityEvaluator
        from sgdjscc_lab.evaluators.object_preservation import ObjectPreservationEvaluator
        from sgdjscc_lab.evaluators.hallucination import HallucinationEvaluator

        class _MockCLIP:
            def _load(self): pass
            def _encode_images(self, tensor):
                n = tensor.shape[0]
                return F.normalize(torch.ones(n, 512), dim=-1)
            def _encode_texts(self, texts):
                n = len(texts)
                return F.normalize(torch.ones(n, 512), dim=-1)
            def image_image_score(self, o, r): return 0.8
            def text_image_score(self, texts, r): return 0.7

        mock_clip = _MockCLIP()
        obj_ev = ObjectPreservationEvaluator(
            clip_evaluator=mock_clip, vocabulary=["cat", "dog"]
        )
        hall_ev = HallucinationEvaluator(clip_evaluator=mock_clip, vocabulary=["cat", "dog"])
        hall_ev._obj_eval = obj_ev

        srs = SemanticReliabilityEvaluator(
            clip_evaluator=mock_clip,
            obj_pres_evaluator=obj_ev,
            hallucination_evaluator=hall_ev,
        )
        return srs

    def test_evaluate_returns_required_keys(self):
        srs = self._make_mock_srs()
        o = torch.rand(1, 3, 64, 64)
        r = torch.rand(1, 3, 64, 64)
        result = srs.evaluate(o, r)
        required = {
            "semantic_reliability_score",
            "clip_image_image",
            "clip_text_image",
            "object_preservation_rate",
            "missing_object_rate",
            "additional_object_rate",
        }
        for k in required:
            assert k in result, f"Missing key: {k}"

    def test_score_is_float(self):
        srs = self._make_mock_srs()
        o = torch.rand(1, 3, 64, 64)
        r = torch.rand(1, 3, 64, 64)
        result = srs.evaluate(o, r)
        assert isinstance(result["semantic_reliability_score"], float)

    def test_evaluate_without_text_sets_clip_text_none(self):
        srs = self._make_mock_srs()
        o = torch.rand(1, 3, 64, 64)
        r = torch.rand(1, 3, 64, 64)
        result = srs.evaluate(o, r, text_list=None)
        assert result["clip_text_image"] is None

    def test_evaluate_with_text_sets_clip_text_float(self):
        srs = self._make_mock_srs()
        o = torch.rand(1, 3, 64, 64)
        r = torch.rand(1, 3, 64, 64)
        result = srs.evaluate(o, r, text_list=["a cat sitting on a mat"])
        assert isinstance(result["clip_text_image"], float)


# ─────────────────────────────────────────────────────────────────────────────
# Presence threshold / uncertain-band wiring (ETRI plan step 0)
# ─────────────────────────────────────────────────────────────────────────────

class _SimMockCLIP:
    """Mock CLIP whose per-vocabulary similarities are set explicitly.

    The original image must be all-zeros and the reconstruction all-ones; the
    image encoder maps them to basis vectors e0 / e1, and each vocabulary text
    feature row is ``[orig_sim, recon_sim, 0, ...]`` so that
    ``img_feat @ txt_feat.T`` reproduces the requested similarities exactly.
    """

    def __init__(self, orig_sims, recon_sims):
        self.orig_sims = orig_sims
        self.recon_sims = recon_sims

    def _load(self):
        pass

    def _encode_images(self, tensor):
        feat = torch.zeros(tensor.shape[0], 8)
        is_recon = bool(tensor.float().mean() > 0.5)
        feat[:, 1 if is_recon else 0] = 1.0
        return feat

    def _encode_texts(self, texts):
        n = len(texts)
        feats = torch.zeros(n, 8)
        for i in range(n):
            feats[i, 0] = self.orig_sims[i]
            feats[i, 1] = self.recon_sims[i]
        return feats


class TestPresenceThresholdWiring:
    """object_presence_threshold / uncertain band must actually reach the
    evaluators (previously the config key existed but was never wired)."""

    def _pair(self):
        return torch.zeros(1, 3, 8, 8), torch.ones(1, 3, 8, 8)

    def test_threshold_changes_detection(self):
        from sgdjscc_lab.evaluators.object_preservation import ObjectPreservationEvaluator
        o, r = self._pair()
        clip = _SimMockCLIP(orig_sims=[0.5, 0.1], recon_sims=[0.5, 0.5])
        low = ObjectPreservationEvaluator(
            clip_evaluator=clip, vocabulary=["cat", "dog"], presence_threshold=0.25)
        high = ObjectPreservationEvaluator(
            clip_evaluator=clip, vocabulary=["cat", "dog"], presence_threshold=0.6)
        assert low.evaluate(o, r)["original_count"] == 1.0     # cat only
        assert high.evaluate(o, r)["original_count"] == 0.0    # nothing clears 0.6

    def test_band_zero_matches_legacy(self):
        from sgdjscc_lab.evaluators.object_preservation import ObjectPreservationEvaluator
        o, r = self._pair()
        clip = _SimMockCLIP(orig_sims=[0.5], recon_sims=[0.22])
        legacy = ObjectPreservationEvaluator(
            clip_evaluator=clip, vocabulary=["cat"], presence_threshold=0.25)
        banded0 = ObjectPreservationEvaluator(
            clip_evaluator=clip, vocabulary=["cat"], presence_threshold=0.25,
            uncertain_band=0.0)
        assert legacy.evaluate(o, r) == banded0.evaluate(o, r)

    def test_uncertain_band_keeps_borderline_object(self):
        from sgdjscc_lab.evaluators.object_preservation import ObjectPreservationEvaluator
        o, r = self._pair()
        # cat: confident in orig (0.5) but borderline in recon (0.22 < 0.25).
        clip = _SimMockCLIP(orig_sims=[0.5], recon_sims=[0.22])
        no_band = ObjectPreservationEvaluator(
            clip_evaluator=clip, vocabulary=["cat"], presence_threshold=0.25)
        band = ObjectPreservationEvaluator(
            clip_evaluator=clip, vocabulary=["cat"], presence_threshold=0.25,
            uncertain_band=0.05)
        assert no_band.evaluate(o, r)["preservation_rate"] == pytest.approx(0.0)
        assert band.evaluate(o, r)["preservation_rate"] == pytest.approx(1.0)

    def test_uncertain_band_suppresses_borderline_hallucination(self):
        from sgdjscc_lab.evaluators.hallucination import HallucinationEvaluator
        o, r = self._pair()
        # dog: absent in orig (0.1), barely above threshold in recon (0.27).
        clip = _SimMockCLIP(orig_sims=[0.5, 0.1], recon_sims=[0.5, 0.27])
        no_band = HallucinationEvaluator(
            clip_evaluator=clip, vocabulary=["cat", "dog"], presence_threshold=0.25)
        band = HallucinationEvaluator(
            clip_evaluator=clip, vocabulary=["cat", "dog"], presence_threshold=0.25,
            uncertain_band=0.05)
        assert no_band.evaluate(o, r)["hallucination_score"] > 0.0
        assert band.evaluate(o, r)["hallucination_score"] == pytest.approx(0.0)

    def test_srs_evaluator_forwards_presence_settings(self):
        from sgdjscc_lab.evaluators.semantic_reliability import SemanticReliabilityEvaluator
        srs = SemanticReliabilityEvaluator(
            clip_evaluator=_SimMockCLIP([0.5], [0.5]),
            presence_threshold=0.4, presence_uncertain_band=0.1,
        )
        assert srs._get_obj_pres().presence_threshold == pytest.approx(0.4)
        assert srs._get_obj_pres().uncertain_band == pytest.approx(0.1)
        assert srs._get_hall().presence_threshold == pytest.approx(0.4)
        assert srs._get_hall().uncertain_band == pytest.approx(0.1)

    def test_eval_context_forwards_presence_settings(self):
        from sgdjscc_lab.pipelines.eval_pipeline import EvalContext
        ctx = EvalContext(
            clip_evaluator=_SimMockCLIP([0.5], [0.5]),
            presence_threshold=0.4, presence_uncertain_band=0.1,
        )
        srs = ctx._get_srs()
        assert srs.presence_threshold == pytest.approx(0.4)
        assert srs.presence_uncertain_band == pytest.approx(0.1)

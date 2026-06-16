"""tests/test_controllers.py – Phase 4-A controller tests (offline).

Covers the SNR guidance policy, the adaptive guidance controller, and the
error-type-aware regeneration policy branching.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ─────────────────────────────────────────────────────────────────────────────
# snr_guidance_policy
# ─────────────────────────────────────────────────────────────────────────────

class TestSNRGuidancePolicy:
    def test_classify_regime(self):
        from sgdjscc_lab.controllers.snr_guidance_policy import classify_regime
        assert classify_regime(-3) == "strong"
        assert classify_regime(4) == "moderate"
        assert classify_regime(12) == "weak"

    def test_low_snr_strengthens_guidance(self):
        from sgdjscc_lab.controllers.snr_guidance_policy import decide
        strong = decide(-3, 4.0, 0.3, 50, True)
        weak = decide(12, 4.0, 0.3, 50, True)
        assert strong.guidance_scale > weak.guidance_scale
        assert strong.diffusion_step >= weak.diffusion_step
        assert strong.regime == "strong"
        assert weak.regime == "weak"

    def test_diffusion_step_clamped(self):
        from sgdjscc_lab.controllers.snr_guidance_policy import decide
        d = decide(20, 4.0, 0.3, 1, True)
        assert d.diffusion_step >= 1

    def test_threshold_override(self):
        from sgdjscc_lab.controllers.snr_guidance_policy import classify_regime
        ov = {"thresholds": {"low": -5.0, "high": 20.0}}
        assert classify_regime(0, _merge(ov)) == "moderate"

    def test_weak_skip_override(self):
        from sgdjscc_lab.controllers.snr_guidance_policy import decide
        d = decide(15, 4.0, 0.3, 50, True, overrides={"weak": {"skip_diffusion": True}})
        assert d.skip_diffusion is True


def _merge(ov):
    from sgdjscc_lab.controllers.snr_guidance_policy import merge_policy
    return merge_policy(ov)


# ─────────────────────────────────────────────────────────────────────────────
# adaptive_guidance_controller
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(**kw):
    from omegaconf import OmegaConf
    base = {
        "snr_db": 10, "guidance_scale": 4.0, "controlnet_scale": 0.3,
        "diffusion_step": 50, "use_text": True, "use_semantic": True,
        "use_adaptive_guidance": True,
    }
    base.update(kw)
    return OmegaConf.create(base)


class TestAdaptiveGuidanceController:
    def test_apply_does_not_mutate_input(self):
        from sgdjscc_lab.controllers.adaptive_guidance_controller import AdaptiveGuidanceController
        cfg = _cfg(snr_db=-3)
        ctrl = AdaptiveGuidanceController()
        adapted, decision = ctrl.apply(cfg, snr_db=-3)
        assert cfg.guidance_scale == 4.0          # original unchanged
        assert adapted.guidance_scale != 4.0      # adapted changed
        assert decision.regime == "strong"

    def test_skip_diffusion_disables_semantic(self):
        from sgdjscc_lab.controllers.adaptive_guidance_controller import AdaptiveGuidanceController
        cfg = _cfg(snr_db=15, adaptive_guidance={"weak": {"skip_diffusion": True}})
        ctrl = AdaptiveGuidanceController.from_config(cfg)
        adapted, decision = ctrl.apply(cfg, snr_db=15)
        assert decision.skip_diffusion is True
        assert adapted.use_semantic is False

    def test_maybe_apply_off_returns_original(self):
        from sgdjscc_lab.controllers.adaptive_guidance_controller import maybe_apply_adaptive_guidance
        cfg = _cfg(use_adaptive_guidance=False)
        out_cfg, decision = maybe_apply_adaptive_guidance(cfg, snr_db=0)
        assert out_cfg is cfg
        assert decision is None


# ─────────────────────────────────────────────────────────────────────────────
# regeneration_policy branching
# ─────────────────────────────────────────────────────────────────────────────

class TestRegenerationPolicy:
    def test_missing_object_selects_strengthen_text(self):
        from sgdjscc_lab.controllers.regeneration_policy import RegenerationPolicy
        report = {"missing_object_count": 2, "additional_object_count": 0,
                  "relation_error_count": 0, "attribute_error_count": 0, "scene_match": True}
        strategies = RegenerationPolicy().select(error_report=report)
        assert [s.name for s in strategies] == ["strengthen_text"]

    def test_hallucination_selects_weaken_text(self):
        from sgdjscc_lab.controllers.regeneration_policy import RegenerationPolicy
        report = {"missing_object_count": 0, "additional_object_count": 3,
                  "relation_error_count": 0, "attribute_error_count": 0, "scene_match": True}
        strategies = RegenerationPolicy().select(error_report=report)
        assert [s.name for s in strategies] == ["weaken_text_strengthen_edge"]

    def test_structural_from_relation_error(self):
        from sgdjscc_lab.controllers.regeneration_policy import RegenerationPolicy
        report = {"missing_object_count": 0, "additional_object_count": 0,
                  "relation_error_count": 2, "attribute_error_count": 0, "scene_match": True}
        strategies = RegenerationPolicy().select(error_report=report)
        assert [s.name for s in strategies] == ["strengthen_structure"]

    def test_multiple_modes(self):
        from sgdjscc_lab.controllers.regeneration_policy import RegenerationPolicy
        report = {"missing_object_count": 1, "additional_object_count": 1,
                  "relation_error_count": 0, "attribute_error_count": 0, "scene_match": False}
        names = [s.name for s in RegenerationPolicy().select(error_report=report)]
        assert names == ["strengthen_text", "weaken_text_strengthen_edge", "strengthen_structure"]

    def test_no_failure_no_strategy(self):
        from sgdjscc_lab.controllers.regeneration_policy import RegenerationPolicy
        report = {"missing_object_count": 0, "additional_object_count": 0,
                  "relation_error_count": 0, "attribute_error_count": 0, "scene_match": True}
        assert RegenerationPolicy().select(error_report=report) == []

    def test_structural_fallback_from_low_clip(self):
        from sgdjscc_lab.controllers.regeneration_policy import RegenerationPolicy
        strategies = RegenerationPolicy().select(error_report={}, metrics={"clip_image_image": 0.3})
        assert [s.name for s in strategies] == ["strengthen_structure"]

    def test_apply_strategy_does_not_mutate(self):
        from sgdjscc_lab.controllers.regeneration_policy import apply_strategy, _STRATEGIES
        cfg = _cfg(guidance_scale=4.0, diffusion_step=50)
        out = apply_strategy(cfg, _STRATEGIES["missing_object"])
        assert cfg.guidance_scale == 4.0
        assert out.guidance_scale == pytest.approx(6.0)
        assert out.use_text is True

"""tests/test_phase_gates.py – Regression tests for Phase 4/5 master switches.

No GPU, no SGDJSCC checkpoints, no heavy model imports required.
All pipeline-level tests inject a lightweight mock reconstruct_fn.

Coverage
--------
1. phase_gates helpers — phase4_enabled, phase5_enabled, effective_flag
2. evaluate_dataset gate regression — Phase 4-A packet code does not run when
   use_phase4 is false even if use_packet_eval is explicitly true
3. evaluate_dataset gate regression — Phase 5 acceleration / channel-cond code
   does not run when use_phase5 is false
4. infer_pipeline._run_diffusion — early-exit blocked when use_phase5 is false
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(**kwargs):
    from omegaconf import OmegaConf
    return OmegaConf.create(kwargs)


def _image_dir(tmp_path):
    """Create two dummy PNG images and return the directory path string."""
    from PIL import Image as PILImage
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for name in ("a.png", "b.png"):
        PILImage.new("RGB", (128, 128), color=(80, 80, 80)).save(img_dir / name)
    return str(img_dir)


def _mock_reconstruct_fn():
    """Identity reconstruct function — returns cloned original; no GPU needed."""
    def fn(fpath, models, cfg):
        from sgdjscc_lab.io import load_image_as_tensor
        orig = load_image_as_tensor(fpath)
        return orig, orig.clone()
    return fn


def _quality_ctx():
    from sgdjscc_lab.pipelines.eval_pipeline import EvalContext
    return EvalContext(enabled_metrics={"psnr", "ssim"})


# ─────────────────────────────────────────────────────────────────────────────
# 1. phase_gates helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestPhaseGateHelpers:
    def test_phase4_disabled_by_default(self):
        from sgdjscc_lab.phase_gates import phase4_enabled
        assert phase4_enabled(_cfg()) is False

    def test_phase5_disabled_by_default(self):
        from sgdjscc_lab.phase_gates import phase5_enabled
        assert phase5_enabled(_cfg()) is False

    def test_phase4_enabled_when_true(self):
        from sgdjscc_lab.phase_gates import phase4_enabled
        assert phase4_enabled(_cfg(use_phase4=True)) is True

    def test_phase5_enabled_when_true(self):
        from sgdjscc_lab.phase_gates import phase5_enabled
        assert phase5_enabled(_cfg(use_phase5=True)) is True

    def test_phase4_explicit_false(self):
        from sgdjscc_lab.phase_gates import phase4_enabled
        assert phase4_enabled(_cfg(use_phase4=False)) is False

    def test_phase5_explicit_false(self):
        from sgdjscc_lab.phase_gates import phase5_enabled
        assert phase5_enabled(_cfg(use_phase5=False)) is False

    # effective_flag — Phase 4 ---------------------------------------------------

    def test_effective_flag_phase4_off_blocks_feature(self):
        from sgdjscc_lab.phase_gates import effective_flag
        cfg = _cfg(use_phase4=False, use_packet_eval=True)
        assert effective_flag(cfg, "use_packet_eval", phase=4) is False

    def test_effective_flag_phase4_on_passes_feature(self):
        from sgdjscc_lab.phase_gates import effective_flag
        cfg = _cfg(use_phase4=True, use_packet_eval=True)
        assert effective_flag(cfg, "use_packet_eval", phase=4) is True

    def test_effective_flag_phase4_on_feature_off(self):
        from sgdjscc_lab.phase_gates import effective_flag
        cfg = _cfg(use_phase4=True, use_packet_eval=False)
        assert effective_flag(cfg, "use_packet_eval", phase=4) is False

    # effective_flag — Phase 5 ---------------------------------------------------

    def test_effective_flag_phase5_off_blocks_feature(self):
        from sgdjscc_lab.phase_gates import effective_flag
        cfg = _cfg(use_phase5=False, use_channel_conditioning=True)
        assert effective_flag(cfg, "use_channel_conditioning", phase=5) is False

    def test_effective_flag_phase5_on_passes_feature(self):
        from sgdjscc_lab.phase_gates import effective_flag
        cfg = _cfg(use_phase5=True, use_channel_conditioning=True)
        assert effective_flag(cfg, "use_channel_conditioning", phase=5) is True

    def test_effective_flag_missing_feature_key_returns_false(self):
        from sgdjscc_lab.phase_gates import effective_flag
        cfg = _cfg(use_phase4=True)   # use_packet_eval not set
        assert effective_flag(cfg, "use_packet_eval", phase=4) is False

    # Independence — phase switches do not bleed into each other -----------------

    def test_phase5_true_does_not_enable_phase4(self):
        from sgdjscc_lab.phase_gates import effective_flag
        cfg = _cfg(use_phase4=False, use_phase5=True, use_packet_eval=True)
        assert effective_flag(cfg, "use_packet_eval", phase=4) is False

    def test_phase4_true_does_not_enable_phase5(self):
        from sgdjscc_lab.phase_gates import effective_flag
        cfg = _cfg(use_phase4=True, use_phase5=False, use_channel_conditioning=True)
        assert effective_flag(cfg, "use_channel_conditioning", phase=5) is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. evaluate_dataset — Phase 4-A packet gate regression
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateDatasetPhase4Gate:
    """use_phase4=false must prevent packet extraction even if use_packet_eval=true."""

    def _make_cfg(self, tmp_path, **extra):
        from omegaconf import OmegaConf
        base = {
            "input_path": _image_dir(tmp_path),
            "output_dir": str(tmp_path / "out"),
            "snr_db": 10,
            "device": "cpu",
        }
        base.update(extra)
        return OmegaConf.create(base)

    def test_packet_extractor_not_called_when_phase4_off(self, tmp_path):
        from sgdjscc_lab.pipelines.eval_pipeline import evaluate_dataset

        cfg = self._make_cfg(
            tmp_path,
            use_phase4=False,
            use_packet_eval=True,    # would activate packets if gate were absent
        )
        ctx = _quality_ctx()
        called = []

        with patch("sgdjscc_lab.pipelines.eval_pipeline.EvalContext._get_packet_extractor",
                   side_effect=lambda *a, **k: called.append(1) or MagicMock()):
            evaluate_dataset(cfg, models=None, eval_ctx=ctx, snr_db=10.0,
                             reconstruct_fn=_mock_reconstruct_fn())

        assert called == [], (
            "_get_packet_extractor was called despite use_phase4=false"
        )

    def test_packet_extractor_called_when_phase4_on(self, tmp_path):
        from sgdjscc_lab.pipelines.eval_pipeline import evaluate_dataset

        cfg = self._make_cfg(
            tmp_path,
            use_phase4=True,
            use_packet_eval=True,
        )
        ctx = _quality_ctx()
        called = []

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {"objects": [], "scene": ""}

        with patch("sgdjscc_lab.pipelines.eval_pipeline.EvalContext._get_packet_extractor",
                   side_effect=lambda *a, **k: called.append(1) or mock_extractor):
            evaluate_dataset(cfg, models=None, eval_ctx=ctx, snr_db=10.0,
                             reconstruct_fn=_mock_reconstruct_fn())

        assert len(called) > 0, (
            "_get_packet_extractor was NOT called despite use_phase4=true + use_packet_eval=true"
        )

    def test_adaptive_guidance_skipped_when_phase4_off(self, tmp_path):
        from sgdjscc_lab.pipelines.eval_pipeline import evaluate_dataset

        cfg = self._make_cfg(
            tmp_path,
            use_phase4=False,
            use_adaptive_guidance=True,
        )
        ctx = _quality_ctx()
        called = []

        with patch(
            "sgdjscc_lab.controllers.adaptive_guidance_controller.maybe_apply_adaptive_guidance",
            side_effect=lambda c, s: called.append(1) or (c, None),
        ):
            evaluate_dataset(cfg, models=None, eval_ctx=ctx, snr_db=10.0,
                             reconstruct_fn=_mock_reconstruct_fn())

        assert called == [], (
            "maybe_apply_adaptive_guidance was called despite use_phase4=false"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. evaluate_dataset — Phase 5 acceleration gate regression
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateDatasetPhase5Gate:
    """use_phase5=false must prevent build_sampler_cfg from running."""

    def _make_cfg(self, tmp_path, **extra):
        from omegaconf import OmegaConf
        base = {
            "input_path": _image_dir(tmp_path),
            "output_dir": str(tmp_path / "out"),
            "snr_db": 10,
            "device": "cpu",
        }
        base.update(extra)
        return OmegaConf.create(base)

    def test_build_sampler_not_called_when_phase5_off(self, tmp_path):
        from sgdjscc_lab.pipelines.eval_pipeline import evaluate_dataset

        cfg = self._make_cfg(
            tmp_path,
            use_phase5=False,
            use_channel_conditioning=True,  # would activate Phase 5-A if gate absent
        )
        ctx = _quality_ctx()
        called = []

        with patch("sgdjscc_lab.acceleration.build_sampler_cfg",
                   side_effect=lambda c: called.append(1) or (c, MagicMock(sampler_type="baseline", steps=50))):
            evaluate_dataset(cfg, models=None, eval_ctx=ctx, snr_db=10.0,
                             reconstruct_fn=_mock_reconstruct_fn())

        assert called == [], (
            "build_sampler_cfg was called despite use_phase5=false"
        )

    def test_build_sampler_called_when_phase5_on(self, tmp_path):
        from sgdjscc_lab.pipelines.eval_pipeline import evaluate_dataset

        cfg = self._make_cfg(tmp_path, use_phase5=True)
        ctx = _quality_ctx()
        called = []

        with patch("sgdjscc_lab.acceleration.build_sampler_cfg",
                   side_effect=lambda c: called.append(1) or (c, MagicMock(sampler_type="baseline", steps=50))):
            evaluate_dataset(cfg, models=None, eval_ctx=ctx, snr_db=10.0,
                             reconstruct_fn=_mock_reconstruct_fn())

        assert len(called) > 0, (
            "build_sampler_cfg was NOT called despite use_phase5=true"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. infer_pipeline._run_diffusion — early-exit blocked by use_phase5=false
# ─────────────────────────────────────────────────────────────────────────────

class TestRunDiffusionPhase5Gate:
    """early-exit must not activate when use_phase5 is false."""

    def _make_acc_cfg(self, use_phase5: bool):
        from omegaconf import OmegaConf
        return OmegaConf.create({
            "use_phase5": use_phase5,
            "acceleration": {
                "early_exit": True,
                "early_exit_mode": "intra_sampler",
                "srs_threshold": 0.8,
                "improvement_delta": 0.01,
                "min_steps": 1,
                "early_exit_check_interval": 5,
            },
        })

    def _call_run_diffusion(self, cfg, interruptible_called):
        """Call _run_diffusion with minimal mock args matching the actual signature."""
        import torch

        mock_pipe = MagicMock()
        mock_pipe.generate.return_value = (None, torch.zeros(1, 16, 16, 16))
        mock_pipe.alphas_cumprod = torch.ones(1000)

        patch_target = "sgdjscc_lab.models.diffusion_wrapper.generate_interruptible"
        with patch(
            patch_target,
            side_effect=lambda *a, **k: interruptible_called.append(1) or (torch.zeros(1, 16, 16, 16), {}),
        ):
            from sgdjscc_lab.pipelines.infer_pipeline import _run_diffusion
            _run_diffusion(
                pipe=mock_pipe,
                encode_features_hat=torch.zeros(1, 16, 16, 16),
                power_scalar=torch.tensor(1.0),
                semantic_text=["a test"],
                canny_latent=torch.zeros(1, 4, 32, 32),
                cur_step=50,
                cfg_method="constant",
                guidance_scale=7.5,
                ctrl_scale=1.0,
                not_control=[False],
                use_jscc_feat=False,
                use_controlnet=False,
                diffusion_step=50,
                step_style="linear",
                mask_token=torch.zeros(1, 16, 16, 16),
                cfg=cfg,
                early_exit_score_fn=None,
            )

    def test_early_exit_not_entered_when_phase5_false(self):
        """generate_interruptible must NOT be called when use_phase5 is false."""
        cfg = self._make_acc_cfg(use_phase5=False)
        called = []
        self._call_run_diffusion(cfg, called)
        assert called == [], (
            "generate_interruptible was called despite use_phase5=false"
        )

    def test_early_exit_entered_when_phase5_true(self):
        """generate_interruptible MUST be called when use_phase5=true + early_exit=true."""
        cfg = self._make_acc_cfg(use_phase5=True)
        called = []
        self._call_run_diffusion(cfg, called)
        assert len(called) > 0, (
            "generate_interruptible was NOT called despite use_phase5=true + early_exit=true"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. maybe_apply_adaptive_guidance — Phase 4 gate at the helper level
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveGuidanceHelperGate:
    """Direct callers of maybe_apply_adaptive_guidance must be blocked by use_phase4."""

    def test_returns_original_cfg_when_phase4_off(self):
        from sgdjscc_lab.controllers.adaptive_guidance_controller import (
            maybe_apply_adaptive_guidance,
        )
        cfg = _cfg(use_phase4=False, use_adaptive_guidance=True)
        out_cfg, decision = maybe_apply_adaptive_guidance(cfg, snr_db=0.0)
        assert decision is None, (
            "Guidance decision was produced despite use_phase4=false"
        )
        # Config must be returned unmodified (same object identity or equal values).
        assert out_cfg is cfg or out_cfg == cfg

    def test_returns_original_cfg_when_feature_flag_off(self):
        from sgdjscc_lab.controllers.adaptive_guidance_controller import (
            maybe_apply_adaptive_guidance,
        )
        cfg = _cfg(use_phase4=True, use_adaptive_guidance=False)
        out_cfg, decision = maybe_apply_adaptive_guidance(cfg, snr_db=0.0)
        assert decision is None

    def test_controller_called_when_both_on(self):
        from sgdjscc_lab.controllers.adaptive_guidance_controller import (
            maybe_apply_adaptive_guidance,
        )
        cfg = _cfg(
            use_phase4=True,
            use_adaptive_guidance=True,
            adaptive_guidance={"thresholds": {"low": 0.0, "high": 8.0}},
        )
        # At SNR=0 (strong regime) the controller should produce a decision.
        _, decision = maybe_apply_adaptive_guidance(cfg, snr_db=0.0)
        assert decision is not None, (
            "No guidance decision when use_phase4=true + use_adaptive_guidance=true"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. maybe_channel_conditioned_reconstruct — Phase 5 gate at the helper level
# ─────────────────────────────────────────────────────────────────────────────

class TestChannelConditionedHelperGate:
    """Direct callers of maybe_channel_conditioned_reconstruct must be blocked by use_phase5."""

    def test_returns_none_when_phase5_off(self):
        from sgdjscc_lab.pipelines.channel_conditioned_infer import (
            maybe_channel_conditioned_reconstruct,
        )
        import torch
        cfg = _cfg(use_phase5=False, use_channel_conditioning=True)
        frame = torch.zeros(1, 3, 128, 128)
        recon, info = maybe_channel_conditioned_reconstruct(frame, models=None, cfg=cfg)
        assert recon is None, (
            "Reconstruction was produced despite use_phase5=false"
        )
        assert info is None

    def test_returns_none_when_feature_flag_off(self):
        from sgdjscc_lab.pipelines.channel_conditioned_infer import (
            maybe_channel_conditioned_reconstruct,
        )
        import torch
        cfg = _cfg(use_phase5=True, use_channel_conditioning=False)
        frame = torch.zeros(1, 3, 128, 128)
        recon, info = maybe_channel_conditioned_reconstruct(frame, models=None, cfg=cfg)
        assert recon is None
        assert info is None

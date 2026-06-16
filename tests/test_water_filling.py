"""tests/test_water_filling.py – Fast-fading water-filling denoising (Algorithm 4).

Synthetic CPU tests: no checkpoints / GPU. A synthetic (oracle) f0-predictor lets
us verify the algorithm's correctness — the water-fill equalization, the selective
per-element update, reduction to the standard sampler under uniform noise, and
end-to-end recovery of f0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.training.noise_schedule import SigmoidNoiseScheduler
from sgdjscc_lab.acceleration.water_filling import (
    water_fill, _water_filling_step,
    water_filling_denoise, water_filling_denoise_from_bundle,
    build_mdt_f0_predictor, fast_fading_water_filling_decode,
)
import numpy as np


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler inverse  S^{-1}
# ─────────────────────────────────────────────────────────────────────────────

def test_inverse_beta_bar_round_trips():
    sch = SigmoidNoiseScheduler()
    for v in (0.02, 0.1, 0.3, 0.5, 0.8, 0.95):
        t = sch.inverse_beta_bar(torch.tensor(v))
        assert 0.0 <= float(t) <= 1.0
        assert abs(float(sch.beta_bar(t)) - v) < 1e-4   # S(S^{-1}(v)) == v


# ─────────────────────────────────────────────────────────────────────────────
# Water-fill (eq. 16)
# ─────────────────────────────────────────────────────────────────────────────

def test_water_fill_identity_when_already_at_target():
    # Uniform noise level already equal to β̄_t ⇒ zero added noise ⇒ g_t == f_t.
    f_t = torch.randn(2, 3, 4, 4)
    beta_t = 0.4
    b_t = torch.full_like(f_t, beta_t)
    g_t = water_fill(f_t, b_t, beta_t, generator=_gen())
    assert torch.allclose(g_t, f_t, atol=1e-6)


def test_water_fill_raises_clean_elements_to_target_variance():
    # f0 = 0 ⇒ g_t is pure noise at level β̄_t; its variance ≈ β̄_t for all elements
    # regardless of their (smaller) starting level b_t.
    torch.manual_seed(0)
    beta_t = 0.5
    f0 = torch.zeros(1, 1, 64, 64)
    b_t = torch.full_like(f0, 0.1)                 # clean start
    f_t = torch.sqrt(1 - b_t) * f0 + torch.sqrt(b_t) * torch.randn_like(f0)
    g_t = water_fill(f_t, b_t, beta_t, generator=_gen(1))
    assert abs(float(g_t.var()) - beta_t) < 0.1    # ≈ β̄_t (statistical)


# ─────────────────────────────────────────────────────────────────────────────
# Selective per-element update
# ─────────────────────────────────────────────────────────────────────────────

def test_selective_update_keeps_cleaner_elements():
    f_t = torch.randn(1, 1, 2, 2)
    # row 0 clean (b=0.05 < β̄_s), row 1 noisy (b=0.5 ≥ β̄_s)
    b_t = torch.tensor([[[[0.05, 0.05], [0.5, 0.5]]]])
    beta_t, beta_s = 0.6, 0.3
    f_s, b_s, _ = _water_filling_step(
        f_t, b_t, beta_t, beta_s,
        f0_predict_fn=lambda g, nl: torch.zeros_like(g), generator=_gen(2))
    # clean elements (b_t < β̄_s) are left unchanged
    assert torch.allclose(f_s[..., 0, :], f_t[..., 0, :])
    assert torch.allclose(b_s[..., 0, :], b_t[..., 0, :])
    # noisy elements (b_t ≥ β̄_s) are updated and their level drops to β̄_s
    assert not torch.allclose(f_s[..., 1, :], f_t[..., 1, :])
    assert torch.allclose(b_s[..., 1, :], torch.full_like(b_s[..., 1, :], beta_s))


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end with an oracle f0-predictor
# ─────────────────────────────────────────────────────────────────────────────

def _make_received(f0, d, seed=3):
    n = torch.randn(f0.shape, generator=_gen(seed))
    return torch.sqrt(1 - d) * f0 + torch.sqrt(d) * n   # paper eq. 12

def test_oracle_recovers_f0_heterogeneous_noise():
    torch.manual_seed(0)
    f0 = torch.randn(2, 4, 8, 8)
    d = torch.rand(2, 4, 8, 8) * 0.6 + 0.05            # heterogeneous d_i ∈ [0.05,0.65]
    f_tilde = _make_received(f0, d)
    out = water_filling_denoise(
        f_tilde, d, f0_predict_fn=lambda g, nl: f0,    # oracle
        scheduler=SigmoidNoiseScheduler(), steps=50, generator=_gen(4))
    assert torch.nn.functional.mse_loss(out, f0) < 1e-2

def test_oracle_recovers_f0_uniform_noise_reduces_to_standard():
    # Uniform d (flat fading / AWGN) — water-filling is identity each step, so this
    # reduces to the standard step-matched sampler and still recovers f0.
    torch.manual_seed(1)
    f0 = torch.randn(1, 4, 8, 8)
    d = torch.full_like(f0, 0.4)
    f_tilde = _make_received(f0, d, seed=5)
    out = water_filling_denoise(
        f_tilde, d, f0_predict_fn=lambda g, nl: f0, steps=40, generator=_gen(6))
    assert torch.nn.functional.mse_loss(out, f0) < 1e-2

def test_denoiser_receives_sqrt_beta_noise_level():
    # The f0-predictor must be conditioned on √β̄_t (the slow-fading convention).
    seen = []
    f0 = torch.randn(1, 2, 4, 4)
    d = torch.rand(1, 2, 4, 4) * 0.5 + 0.1
    f_tilde = _make_received(f0, d, seed=7)
    def rec(g, nl):
        seen.append(float(nl.flatten()[0]))
        return f0
    water_filling_denoise(f_tilde, d, rec, steps=20, generator=_gen(8))
    assert seen and all(0.0 <= v <= 1.0 for v in seen)
    # noise level decreases monotonically across the trajectory (t → 0)
    assert seen[-1] < seen[0]


# ─────────────────────────────────────────────────────────────────────────────
# Integration with the fast-fading MeasurementBundle
# ─────────────────────────────────────────────────────────────────────────────

def test_from_bundle_runs_on_fast_fading_observation():
    from sgdjscc_lab.channels import FastFadingChannel
    torch.manual_seed(0)
    latent = torch.randn(1, 4, 8, 8)
    bundle = FastFadingChannel(block_length=8, csi="perfect").observe(latent, 5.0)
    assert bundle.noise_level is not None and bundle.noise_level.shape == latent.shape
    out = water_filling_denoise_from_bundle(
        bundle, f0_predict_fn=lambda g, nl: g, steps=10, generator=_gen(9))
    assert out.shape == latent.shape and torch.isfinite(out).all()

def test_from_bundle_blind_raises():
    from sgdjscc_lab.channels import RayleighChannel
    bundle = RayleighChannel(csi="none").observe(torch.randn(1, 4, 8, 8), 5.0)
    # blind: no equalized latent → cannot run water-filling (message names equalized)
    with pytest.raises(ValueError, match="equalized"):
        water_filling_denoise_from_bundle(bundle, f0_predict_fn=lambda g, nl: g)


# ─────────────────────────────────────────────────────────────────────────────
# Real-DM adapter (matches public DiffusionGenerator.pred_image convention)
# ─────────────────────────────────────────────────────────────────────────────

def test_adapter_passes_sqrt_beta_noise_level_as_numpy_and_returns_f0():
    f0 = torch.randn(2, 4, 8, 8)
    seen = {}
    def stub_pred_image(noisy, labels, noise_level, class_guidance, c, controlnet,
                        mask_token, not_control):
        seen.update(noisy_shape=tuple(noisy.shape), nl=noise_level, cg=class_guidance,
                    labels=labels, c=c, controlnet=controlnet)
        return f0
    fn = build_mdt_f0_predictor(stub_pred_image, labels="LBL2B",
                                class_guidance=[3.0, 3.0], c="C", controlnet=True)
    out = fn(torch.randn(2, 4, 8, 8), torch.full((2, 1), 0.5))   # noise_level = √β̄ = 0.5
    # noise level is handed to pred_image as a numpy [B] array of √β̄ values
    assert isinstance(seen["nl"], np.ndarray) and seen["nl"].shape == (2,)
    assert np.allclose(seen["nl"], 0.5)
    assert isinstance(seen["cg"], np.ndarray) and np.allclose(seen["cg"], 3.0)
    assert seen["labels"] == "LBL2B" and seen["c"] == "C" and seen["controlnet"] is True
    assert seen["noisy_shape"] == (2, 4, 8, 8)   # NOT pre-doubled (pred_image dups internally)
    assert torch.allclose(out, f0)


def test_adapter_in_loop_recovers_f0():
    f0 = torch.randn(1, 4, 8, 8)
    d = torch.rand(1, 4, 8, 8) * 0.5 + 0.1
    f_tilde = _make_received(f0, d, seed=11)
    fn = build_mdt_f0_predictor(
        lambda noisy, labels, nl, cg, c, cn, mt, nc: f0,   # oracle pred_image
        labels=None, class_guidance=[3.0])
    out = water_filling_denoise(f_tilde, d, fn, steps=40, generator=_gen(12))
    assert torch.nn.functional.mse_loss(out, f0) < 1e-2


# ─────────────────────────────────────────────────────────────────────────────
# CSI policy (perfect runs / imperfect runs+warns / blind error|fallback)
# ─────────────────────────────────────────────────────────────────────────────

def test_csi_policy_perfect_runs():
    from sgdjscc_lab.channels import FastFadingChannel
    latent = torch.randn(1, 4, 8, 8)
    b = FastFadingChannel(block_length=8, csi="perfect").observe(latent, 5.0)
    out = fast_fading_water_filling_decode(
        b, lambda g, nl: g, steps=8, generator=_gen(1))
    assert out.shape == latent.shape and torch.isfinite(out).all()


def test_csi_policy_blind_errors_by_default_and_fallback_when_asked():
    from sgdjscc_lab.channels import RayleighChannel
    b = RayleighChannel(csi="none").observe(torch.randn(1, 4, 8, 8), 5.0)
    with pytest.raises(ValueError, match="blind"):
        fast_fading_water_filling_decode(b, lambda g, nl: g)
    called = {}
    def fb(bundle):
        called["x"] = True
        return bundle.received
    out = fast_fading_water_filling_decode(
        b, lambda g, nl: g, on_blind="fallback", fallback_fn=fb)
    assert called.get("x") and torch.equal(out, b.received)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime decode-swap wiring in infer_pipeline (stub pipe, no checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

class _StubPipe:
    """Minimal DiffusionGenerator stand-in for routing tests."""
    text_embed = object()
    alphas_cumprod = None

    def __init__(self):
        self.calls = {"pred": 0, "gen": 0}

    def encode_text(self, prompt, embed):
        return torch.zeros(len(prompt), 8)

    def pred_image(self, noisy, labels, noise_level, class_guidance, c, controlnet,
                   mask_token, not_control):
        self.calls["pred"] += 1
        return torch.zeros_like(noisy)            # f0 prediction

    def generate(self, **kw):
        self.calls["gen"] += 1
        return None, torch.zeros(1, 16, 16, 16)   # (image, latent)


def _call_run_diffusion(pipe, cfg, noise_level):
    from sgdjscc_lab.pipelines.infer_pipeline import _run_diffusion
    return _run_diffusion(
        pipe=pipe, encode_features_hat=torch.randn(1, 16, 16, 16),
        power_scalar=torch.ones(1, 1, 1, 1), semantic_text=[""], canny_latent=None,
        cur_step=0.5, cfg_method="constant", guidance_scale=3.0, ctrl_scale=1.0,
        not_control=[0, 0], use_jscc_feat=True, use_controlnet=False,
        diffusion_step=4, step_style="continuous", mask_token=None, cfg=cfg,
        noise_level=noise_level)


def test_water_filling_noise_level_prefers_per_patch_over_global_last_bundle():
    # Regression: the one-pass multi-patch path must use THIS patch's noise level,
    # not the channel's global last_bundle (which is the LAST patch's realisation).
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from sgdjscc_lab.pipelines.infer_pipeline import _water_filling_noise_level
    from sgdjscc_lab.channels.measurement import MeasurementBundle

    per_patch = torch.full((1, 16, 16, 16), 0.2)
    global_last = torch.full((1, 16, 16, 16), 0.9)
    art = SimpleNamespace(bundle=MeasurementBundle(
        received=torch.zeros(1, 16, 16, 16), noise_level=per_patch))
    jscc = SimpleNamespace(channel_model=SimpleNamespace(
        last_bundle=MeasurementBundle(received=torch.zeros(1, 16, 16, 16),
                                      noise_level=global_last)))
    cfg_on = OmegaConf.create({"use_water_filling": True})
    nl = _water_filling_noise_level(art, jscc, cfg_on)
    assert torch.equal(nl, per_patch)            # per-patch d, NOT the global last
    # no per-patch bundle → fall back to the channel's last_bundle (single-image path)
    nl_fb = _water_filling_noise_level(SimpleNamespace(bundle=None), jscc, cfg_on)
    assert torch.equal(nl_fb, global_last)
    # off → None
    assert _water_filling_noise_level(
        art, jscc, OmegaConf.create({"use_water_filling": False})) is None


def test_build_evidence_bundle_copies_noise_level():
    # Regression: per-patch evidence must carry noise_level so per-patch water-filling
    # has its own d (the field was previously dropped during bundle construction).
    from types import SimpleNamespace
    from sgdjscc_lab.pipelines.infer_pipeline import _build_evidence_bundle
    from sgdjscc_lab.channels.measurement import MeasurementBundle

    nl = torch.rand(1, 16, 16, 16)
    src = MeasurementBundle(received=torch.zeros(1, 16, 16, 16), noise_level=nl,
                            noise_var=torch.ones(1, 1, 1, 1),
                            meta={"channel": "fast_fading"})
    jscc = SimpleNamespace(channel_model=SimpleNamespace(last_bundle=src), snr=10.0)
    ef = torch.randn(1, 16, 16, 16)
    bundle = _build_evidence_bundle(jscc, {"encode_features_hat": ef, "decoder_init": ef})
    assert bundle.noise_level is not None and torch.equal(bundle.noise_level, nl)


def test_run_diffusion_routes_to_water_filling_when_enabled():
    from omegaconf import OmegaConf
    pipe = _StubPipe()
    cfg = OmegaConf.create({"use_phase5": True, "use_water_filling": True,
                            "water_filling": {"steps": 4}})
    nl = torch.rand(1, 16, 16, 16) * 0.4 + 0.1
    out = _call_run_diffusion(pipe, cfg, nl)
    assert pipe.calls["pred"] > 0 and pipe.calls["gen"] == 0   # water-filling, not generate
    assert out.shape == (1, 16, 16, 16)


def test_run_diffusion_standard_path_when_water_filling_off_or_no_noise_level():
    from omegaconf import OmegaConf
    # flag off → standard generate path
    p1 = _StubPipe()
    _call_run_diffusion(p1, OmegaConf.create({"use_phase5": True,
                                              "use_water_filling": False}), torch.rand(1, 16, 16, 16))
    assert p1.calls["gen"] == 1 and p1.calls["pred"] == 0
    # flag on but no per-element noise level (e.g. AWGN) → standard path
    p2 = _StubPipe()
    _call_run_diffusion(p2, OmegaConf.create({"use_phase5": True,
                                              "use_water_filling": True}), None)
    assert p2.calls["gen"] == 1 and p2.calls["pred"] == 0
    # phase5 off gates it even if the flag is on → standard path
    p3 = _StubPipe()
    _call_run_diffusion(p3, OmegaConf.create({"use_phase5": False,
                                              "use_water_filling": True}), torch.rand(1, 16, 16, 16))
    assert p3.calls["gen"] == 1 and p3.calls["pred"] == 0


def test_csi_policy_imperfect_runs_with_warning():
    import logging
    from sgdjscc_lab.channels import FastFadingChannel
    b = FastFadingChannel(block_length=8, csi="imperfect", csi_error_std=0.2).observe(
        torch.randn(1, 4, 8, 8), 5.0)
    msgs = []
    handler = logging.Handler()
    handler.emit = lambda r: msgs.append(r.getMessage())
    lg = logging.getLogger("sgdjscc_lab.acceleration.water_filling")
    lg.addHandler(handler)
    try:
        out = fast_fading_water_filling_decode(b, lambda g, nl: g, steps=6, generator=_gen(2))
    finally:
        lg.removeHandler(handler)
    assert out.shape == (1, 4, 8, 8)
    assert any("PERFECT CSI" in m for m in msgs)   # imperfect CSI → approximate warning

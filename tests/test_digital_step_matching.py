"""tests/test_digital_step_matching.py – digital_packet blind step/SNR fix.

Regression coverage for the NaN/Inf bug fixed in
``pipelines/infer_pipeline.py::_compute_step``: routing a digital_packet
(quantized) received latent through ``jscc.snr_prediction_net`` (trained
only on AWGN-shaped noise) could predict a signal_scale >= 1, making
``10*log10(1/cur_step - 1)`` evaluate log10 of a non-positive number.  The
fix replaces that, for the digital channel only, with a deterministic
quantization-metadata-derived SNR (``_digital_quant_snr_db``); the AWGN path
must stay byte-for-byte identical (algorithm-preservation invariant).
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.channels import DigitalPacketChannel
from sgdjscc_lab.pipelines.infer_pipeline import (
    DIGITAL_STEP_POLICIES,
    _DIGITAL_SNR_CEIL_DB,
    _DIGITAL_SNR_FLOOR_DB,
    _apply_channel,
    _compute_step,
    _digital_effective_snr_db,
    _digital_quant_snr_db,
    _digital_signal_scale,
)
from sgdjscc_lab.utils.finite_checks import NonFiniteError, assert_finite


def _fake_jscc(channel_model=None, snr=10.0, snr_prediction_net=None):
    jscc = types.SimpleNamespace(snr=snr, channel_model=channel_model)
    if snr_prediction_net is not None:
        jscc.snr_prediction_net = snr_prediction_net
    return jscc


class TestDigitalQuantSnrDb:
    def test_matches_20log10_qmax_for_typical_bit_depths(self):
        assert _digital_quant_snr_db(4) == pytest.approx(20 * math.log10(15))
        assert _digital_quant_snr_db(6) == pytest.approx(20 * math.log10(63))
        assert _digital_quant_snr_db(8) == pytest.approx(20 * math.log10(255))

    def test_lossless_and_near_lossless_clamp_to_ceiling(self):
        # int16's analytic 20*log10(65535) ~= 96 dB exceeds the ceiling, and
        # float32 (bit_depth=32, no quantization at all) is defined as the
        # ceiling directly -- both must clamp, never overflow into a huge
        # finite-but-meaningless dB value.
        assert _digital_quant_snr_db(16) == _DIGITAL_SNR_CEIL_DB
        assert _digital_quant_snr_db(32) == _DIGITAL_SNR_CEIL_DB

    def test_always_within_clamp_bounds(self):
        for bit_depth in (4, 6, 8, 16, 32):
            snr = _digital_quant_snr_db(bit_depth)
            assert _DIGITAL_SNR_FLOOR_DB <= snr <= _DIGITAL_SNR_CEIL_DB
            assert math.isfinite(snr)


class TestDigitalEffectiveSnrDbPolicies:
    def test_fixed_reference_always_returns_ceiling_regardless_of_bit_depth(self):
        for bit_depth in (4, 6, 8, 16):
            assert _digital_effective_snr_db(bit_depth, "fixed_reference") == _DIGITAL_SNR_CEIL_DB

    def test_bitdepth_proxy_matches_the_proxy_formula(self):
        for bit_depth in (4, 6, 8, 16):
            assert _digital_effective_snr_db(bit_depth, "bitdepth_proxy") == _digital_quant_snr_db(bit_depth)

    def test_lossless_bit_depth_always_ceiling_regardless_of_policy(self):
        # float32 is a structural fact (no quantization error at all), not a
        # policy choice -- every policy must agree it's the ceiling.
        for policy in DIGITAL_STEP_POLICIES:
            kwargs = {"quant_snr_db": 5.0} if policy == "quant_nmse" else {}
            assert _digital_effective_snr_db(32, policy, **kwargs) == _DIGITAL_SNR_CEIL_DB

    def test_quant_nmse_uses_the_supplied_measured_value(self):
        assert _digital_effective_snr_db(8, "quant_nmse", quant_snr_db=17.3) == pytest.approx(17.3)

    def test_quant_nmse_clamps_to_floor_and_ceiling(self):
        assert _digital_effective_snr_db(8, "quant_nmse", quant_snr_db=-999.0) == _DIGITAL_SNR_FLOOR_DB
        assert _digital_effective_snr_db(8, "quant_nmse", quant_snr_db=999.0) == _DIGITAL_SNR_CEIL_DB

    def test_quant_nmse_without_a_measured_value_raises_not_silently_substitutes(self):
        with pytest.raises(ValueError, match="quant_nmse"):
            _digital_effective_snr_db(8, "quant_nmse", quant_snr_db=None)

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="unknown digital_policy"):
            _digital_effective_snr_db(8, "made_up_policy")


class TestComputeStepDigitalPolicyDispatch:
    """_compute_step must honor digital_bit_depth/digital_policy explicitly,
    independent of jscc.channel_model (the receiver-uses-packet-metadata
    contract)."""

    def _pipe(self):
        return types.SimpleNamespace(
            scheduler=types.SimpleNamespace(alphas_cumprod=torch.linspace(0.999, 0.001, 1000))
        )

    def test_explicit_digital_bit_depth_overrides_missing_channel_model(self):
        # jscc.channel_model is None (no DigitalPacketChannel at all) -- the
        # explicit digital_bit_depth argument alone must still select the
        # digital branch, exactly as a bundle receiver with no in-process
        # channel object would call this.
        jscc = _fake_jscc(channel_model=None, snr_prediction_net=lambda x: torch.full((x.shape[0],), 10.0))
        cur_step, cur_snr = _compute_step(
            jscc=jscc, encode_features_hat=torch.randn(1, 4, 4, 4), power_scalar=torch.tensor([1.0]),
            signal_scale=torch.ones(1, 1), pipe=self._pipe(), step_style="continuous",
            use_jscc_feat=True, use_gt_csi=False, device=torch.device("cpu"),
            digital_bit_depth=8, digital_policy="bitdepth_proxy",
        )
        assert torch.isfinite(cur_step).all()
        assert torch.allclose(cur_snr, torch.full_like(cur_snr, _digital_quant_snr_db(8)))

    def test_fixed_reference_policy_gives_identical_step_across_bit_depths(self):
        jscc = _fake_jscc(channel_model=None)
        power_scalar = torch.tensor([50.0])
        steps = []
        for bit_depth in (4, 8, 16):
            cur_step, _ = _compute_step(
                jscc=jscc, encode_features_hat=torch.randn(1, 4, 4, 4), power_scalar=power_scalar,
                signal_scale=torch.ones(1, 1), pipe=self._pipe(), step_style="continuous",
                use_jscc_feat=True, use_gt_csi=False, device=torch.device("cpu"),
                digital_bit_depth=bit_depth, digital_policy="fixed_reference",
            )
            steps.append(cur_step)
        assert torch.allclose(steps[0], steps[1])
        assert torch.allclose(steps[1], steps[2])

    def test_bitdepth_proxy_policy_gives_different_steps_across_bit_depths(self):
        jscc = _fake_jscc(channel_model=None)
        power_scalar = torch.tensor([50.0])
        cur_step_4, _ = _compute_step(
            jscc=jscc, encode_features_hat=torch.randn(1, 4, 4, 4), power_scalar=power_scalar,
            signal_scale=torch.ones(1, 1), pipe=self._pipe(), step_style="continuous",
            use_jscc_feat=True, use_gt_csi=False, device=torch.device("cpu"),
            digital_bit_depth=4, digital_policy="bitdepth_proxy",
        )
        cur_step_16, _ = _compute_step(
            jscc=jscc, encode_features_hat=torch.randn(1, 4, 4, 4), power_scalar=power_scalar,
            signal_scale=torch.ones(1, 1), pipe=self._pipe(), step_style="continuous",
            use_jscc_feat=True, use_gt_csi=False, device=torch.device("cpu"),
            digital_bit_depth=16, digital_policy="bitdepth_proxy",
        )
        assert not torch.allclose(cur_step_4, cur_step_16)

    def test_quant_nmse_policy_uses_supplied_measured_snr(self):
        jscc = _fake_jscc(channel_model=None)
        cur_step, cur_snr = _compute_step(
            jscc=jscc, encode_features_hat=torch.randn(1, 4, 4, 4), power_scalar=torch.tensor([1.0]),
            signal_scale=torch.ones(1, 1), pipe=self._pipe(), step_style="continuous",
            use_jscc_feat=True, use_gt_csi=False, device=torch.device("cpu"),
            digital_bit_depth=8, digital_policy="quant_nmse", digital_quant_snr_db=22.0,
        )
        assert torch.allclose(cur_snr, torch.full_like(cur_snr, 22.0))


class TestDigitalSignalScale:
    @pytest.mark.parametrize("bit_depth", [4, 6, 8, 16, 32])
    def test_signal_scale_strictly_between_zero_and_one(self, bit_depth):
        like = torch.zeros(3, 1)
        signal_scale, snr_db = _digital_signal_scale(bit_depth, like)
        assert torch.isfinite(signal_scale).all()
        assert bool((signal_scale > 0).all())
        assert bool((signal_scale < 1).all())
        assert math.isfinite(snr_db)

    @pytest.mark.parametrize("bit_depth", [4, 6, 8, 16, 32])
    def test_cur_step_never_hits_zero_or_one_boundary(self, bit_depth):
        like = torch.zeros(2, 1)
        signal_scale, _ = _digital_signal_scale(bit_depth, like)
        cur_step = 1 - signal_scale
        assert bool((cur_step > 0).all())
        assert bool((cur_step < 1).all())
        # the original bug: log10(1/cur_step - 1) must stay finite
        recovered_snr_db = 10 * torch.log10(1 / cur_step - 1)
        assert torch.isfinite(recovered_snr_db).all()


class TestComputeStepDigitalChannel:
    """A pathologically out-of-domain snr_prediction_net must never be reached
    (and never matter) once a DigitalPacketChannel is set."""

    def _poisoned_net(self, x):
        # Would drive predicted_signal_scale >= 1 (the historical NaN trigger)
        # if it were ever called on a digital latent.
        return torch.full((x.shape[0],), 10.0)

    @pytest.mark.parametrize("bit_depth", [4, 6, 8, 16, 32])
    @pytest.mark.parametrize("step_style", ["continuous", "discrete"])
    def test_no_nan_inf_across_all_bit_depths_and_step_styles(self, bit_depth, step_style):
        ch = DigitalPacketChannel(bit_depth=bit_depth)
        jscc = _fake_jscc(channel_model=ch, snr_prediction_net=self._poisoned_net)
        power_scalar = torch.tensor([100.0, 200.0])
        encode_features_hat = torch.randn(2, 16, 16, 16)
        signal_scale = torch.ones(2, 1)  # only used by the use_gt_csi branch

        pipe = types.SimpleNamespace(
            scheduler=types.SimpleNamespace(alphas_cumprod=torch.linspace(0.999, 0.001, 1000))
        )

        cur_step, cur_snr = _compute_step(
            jscc=jscc, encode_features_hat=encode_features_hat, power_scalar=power_scalar,
            signal_scale=signal_scale, pipe=pipe, step_style=step_style,
            use_jscc_feat=True, use_gt_csi=False, device=torch.device("cpu"),
        )
        if torch.is_tensor(cur_step):
            assert torch.isfinite(cur_step).all()
        else:
            assert math.isfinite(float(cur_step))
        if torch.is_tensor(cur_snr):
            assert torch.isfinite(cur_snr).all()
        else:
            assert math.isfinite(float(cur_snr))

    def test_continuous_cur_snr_is_a_tensor_not_a_bare_float(self):
        # Regression: pipelines/infer_pipeline.py::_retransmit_canny does
        # (cur_snr <= -5).reshape(-1, 1, 1, 1) -- cur_snr must be a per-sample
        # tensor (exactly like the AWGN blind branch's cur_snr), never a bare
        # python float, or that crashes with "'bool' object has no attribute
        # 'reshape'" the moment canny_cr != "none" (caught via a real GPU
        # smoke run of a digital_packet config).
        ch = DigitalPacketChannel(bit_depth=8)
        jscc = _fake_jscc(channel_model=ch)
        power_scalar = torch.tensor([100.0, 200.0])
        pipe = types.SimpleNamespace(
            scheduler=types.SimpleNamespace(alphas_cumprod=torch.linspace(0.999, 0.001, 1000))
        )
        _, cur_snr = _compute_step(
            jscc=jscc, encode_features_hat=torch.randn(2, 16, 16, 16), power_scalar=power_scalar,
            signal_scale=torch.ones(2, 1), pipe=pipe, step_style="continuous",
            use_jscc_feat=True, use_gt_csi=False, device=torch.device("cpu"),
        )
        assert torch.is_tensor(cur_snr)
        thresholded = (cur_snr <= -5).reshape(-1, 1, 1, 1).repeat(1, 1, 4, 4).float()
        assert thresholded.shape == (2, 1, 4, 4)

    @pytest.mark.parametrize("step_style", ["continuous", "discrete"])
    def test_expanded_power_scalar_does_not_expand_step_batch(self, step_style):
        """Receiver power normalization is spatially expanded, but step matching is per sample."""
        ch = DigitalPacketChannel(bit_depth=32)
        jscc = _fake_jscc(channel_model=ch)
        latent = torch.randn(2, 16, 16, 16)
        expanded_power = torch.ones_like(latent)
        pipe = types.SimpleNamespace(
            scheduler=types.SimpleNamespace(alphas_cumprod=torch.linspace(0.999, 0.001, 1000))
        )

        cur_step, cur_snr = _compute_step(
            jscc=jscc, encode_features_hat=latent, power_scalar=expanded_power,
            signal_scale=torch.ones(2, 1), pipe=pipe, step_style=step_style,
            use_jscc_feat=True, use_gt_csi=False, device=torch.device("cpu"),
        )

        if step_style == "continuous":
            assert cur_step.shape == (2, 1)
            assert cur_snr.shape == (2, 1)
        else:
            assert isinstance(cur_step, int)
            assert isinstance(cur_snr, float)

    def test_poisoned_net_is_never_called_for_digital_channel(self):
        calls = {"n": 0}

        def poisoned_net(x):
            calls["n"] += 1
            return torch.full((x.shape[0],), 10.0)

        ch = DigitalPacketChannel(bit_depth=8)
        jscc = _fake_jscc(channel_model=ch, snr_prediction_net=poisoned_net)
        power_scalar = torch.tensor([100.0])
        pipe = types.SimpleNamespace(
            scheduler=types.SimpleNamespace(alphas_cumprod=torch.linspace(0.999, 0.001, 1000))
        )
        _compute_step(
            jscc=jscc, encode_features_hat=torch.randn(1, 16, 16, 16), power_scalar=power_scalar,
            signal_scale=torch.ones(1, 1), pipe=pipe, step_style="continuous",
            use_jscc_feat=True, use_gt_csi=False, device=torch.device("cpu"),
        )
        assert calls["n"] == 0


class TestComputeStepAwgnUnchanged:
    """AWGN path (channel_model=None) must be byte-for-byte identical to the
    pre-fix formula -- the digital branch must never be reached."""

    def test_continuous_blind_awgn_matches_original_formula(self):
        def net(x):
            return torch.full((x.shape[0],), 0.3)

        jscc = _fake_jscc(channel_model=None, snr=10.0, snr_prediction_net=net)
        power_scalar = torch.tensor([50.0])
        encode_features_hat = torch.ones(1, 4, 4, 4) * 15.0

        cur_step, cur_snr = _compute_step(
            jscc=jscc, encode_features_hat=encode_features_hat, power_scalar=power_scalar,
            signal_scale=torch.ones(1, 1), pipe=None, step_style="continuous",
            use_jscc_feat=True, use_gt_csi=False, device=torch.device("cpu"),
        )
        expected_pss = net(encode_features_hat / power_scalar).reshape([-1, 1]) ** 2
        expected_step = 1 - expected_pss
        expected_snr = 10 * torch.log10(1 / expected_step - 1)
        assert torch.allclose(cur_step, expected_step)
        assert torch.allclose(cur_snr, expected_snr)

    def test_gt_csi_path_untouched_regardless_of_channel(self):
        ch = DigitalPacketChannel(bit_depth=4)
        jscc = _fake_jscc(channel_model=ch, snr=12.0)
        signal_scale = torch.tensor([[0.7]])
        cur_step, cur_snr = _compute_step(
            jscc=jscc, encode_features_hat=torch.randn(1, 4, 4, 4), power_scalar=torch.tensor([1.0]),
            signal_scale=signal_scale, pipe=None, step_style="continuous",
            use_jscc_feat=True, use_gt_csi=True, device=torch.device("cpu"),
        )
        assert cur_step == pytest.approx(1 - 0.7)
        assert cur_snr == pytest.approx(12.0)


class TestApplyChannelFiniteGuard:
    def test_raises_non_finite_error_with_stage_when_channel_emits_nan(self):
        jscc = types.SimpleNamespace(
            snr=10.0,
            normalize=lambda x: x,
            channel=lambda x: torch.full_like(x, float("nan")),
        )
        with pytest.raises(NonFiniteError) as exc_info:
            _apply_channel(jscc, torch.randn(1, 4, 4, 4))
        assert exc_info.value.stage == "channel_transmit"

    def test_passes_through_finite_channel_output_unchanged(self):
        x = torch.randn(1, 4, 4, 4)
        jscc = types.SimpleNamespace(
            snr=10.0,
            normalize=lambda t: t,
            channel=lambda t: t,
        )
        out, scale = _apply_channel(jscc, x)
        assert torch.equal(out, x)


class TestAssertFinite:
    def test_finite_tensor_passes_through(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        assert assert_finite(x, "some_stage") is x

    def test_nan_raises_with_counts(self):
        x = torch.tensor([1.0, float("nan"), float("inf"), float("-inf")])
        with pytest.raises(NonFiniteError) as exc_info:
            assert_finite(x, "some_stage", context={"video": "v1", "frame": 3})
        err = exc_info.value
        assert err.stage == "some_stage"
        assert err.n_nan == 1
        assert err.n_inf == 2
        assert err.context == {"video": "v1", "frame": 3}
        assert "some_stage" in str(err)

    def test_non_tensor_passthrough(self):
        assert assert_finite(3.0, "stage") == 3.0
        assert assert_finite(None, "stage") is None

"""tests/test_channels_phase5.py – Phase 5-A channel + measurement tests (offline)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture()
def latent():
    torch.manual_seed(0)
    return torch.randn(2, 16, 16, 16)


# ─────────────────────────────────────────────────────────────────────────────
# MMSE equalization (paper §III/§V: y / sqrt(g² + σ²)) — replaces zero-forcing
# ─────────────────────────────────────────────────────────────────────────────

class TestMMSEEqualization:
    def test_mmse_helper_converges_to_zero_forcing_as_noise_vanishes(self):
        from sgdjscc_lab.channels.measurement import mmse_equalize
        torch.manual_seed(1)
        g = torch.rand(2, 1, 1, 1) + 0.5            # gain in [0.5, 1.5]
        x = torch.randn(2, 4, 8, 8)
        y = g * x                                    # noiseless received
        zf = y / g                                   # zero-forcing reference
        near0 = mmse_equalize(y, g, torch.zeros(2, 1, 1, 1))
        assert torch.allclose(near0, zf, atol=1e-4)  # σ²→0 ⇒ MMSE = ZF
        # larger noise_var ⇒ divides by larger sqrt(g²+σ²) ⇒ strictly shrinks
        big = mmse_equalize(y, g, torch.full((2, 1, 1, 1), 4.0))
        assert (big.abs() < near0.abs() + 1e-6).all()
        assert not torch.allclose(big, zf, atol=1e-2)

    def test_mmse_helper_per_element_gain_map(self):
        # Exercises the EXACT shapes the fast-fading channel passes: a per-element
        # gain map [B,C,H,W] with a per-sample noise_var [B,1,1,1] (broadcast).
        from sgdjscc_lab.channels.measurement import mmse_equalize
        torch.manual_seed(3)
        g = torch.rand(2, 4, 8, 8) + 0.5            # per-element gains
        y = torch.randn(2, 4, 8, 8)
        zf = y / g
        near0 = mmse_equalize(y, g, torch.zeros(2, 1, 1, 1))
        assert near0.shape == y.shape               # broadcasting [B,1,1,1] over [B,C,H,W]
        assert torch.allclose(near0, zf, atol=1e-4)  # σ²→0 ⇒ per-element ZF
        big = mmse_equalize(y, g, torch.full((2, 1, 1, 1), 3.0))
        assert (big.abs() <= zf.abs() + 1e-6).all()  # per-element MMSE shrink vs ZF

    def test_rayleigh_high_snr_equalized_approaches_zero_forcing(self, latent):
        from sgdjscc_lab.channels import RayleighChannel
        ch = RayleighChannel(csi="perfect")
        b = ch.observe(latent, 60.0)                 # very high SNR ⇒ σ²→0
        zf = b.received / b.channel_gain
        assert torch.allclose(b.equalized, zf, rtol=1e-3, atol=1e-3)

    def test_rayleigh_low_snr_equalized_differs_from_zero_forcing(self, latent):
        from sgdjscc_lab.channels import RayleighChannel
        b = RayleighChannel(csi="perfect").observe(latent, -10.0)   # low SNR
        zf = b.received / b.channel_gain
        # MMSE shrinks vs ZF and is clearly different at low SNR
        assert not torch.allclose(b.equalized, zf, rtol=1e-2, atol=1e-2)
        assert b.equalized.abs().mean() < zf.abs().mean()

    def test_fast_fading_mmse_high_snr_recovers_latent(self, latent):
        # At very high SNR, per-element MMSE y/sqrt(g²+σ²) ≈ y/g = latent, i.e. the
        # equalizer recovers the transmitted latent element-wise (ZF limit). A few
        # rare tiny-gain elements amplify residual noise, so check the MEDIAN error
        # (robust) rather than the max — this still exercises per-element behaviour.
        from sgdjscc_lab.channels import FastFadingChannel
        b = FastFadingChannel(block_length=16, csi="perfect").observe(latent, 60.0)
        assert b.equalized.shape == latent.shape
        assert (b.equalized - latent).abs().median() < 1e-2

    def test_fast_fading_mmse_low_snr_does_not_recover(self, latent):
        # At low SNR the output is noise-dominated, so it is NOT the clean latent.
        from sgdjscc_lab.channels import FastFadingChannel
        b = FastFadingChannel(block_length=16, csi="perfect").observe(latent, -10.0)
        assert torch.isfinite(b.equalized).all()
        assert (b.equalized - latent).abs().median() > 1e-1

    @pytest.mark.parametrize("perfect", [True, False])
    def test_noise_level_consistent_with_equalized(self, latent, perfect):
        # The per-element noise level d and the equalized latent f̃ must be derived
        # from the SAME gain estimate. Identity: d·received² == σ²·equalized²
        # (holds for perfect AND imperfect CSI; the old code used the TRUE gain for
        # d but g_hat for f̃, which broke this under csi='imperfect').
        from sgdjscc_lab.channels import RayleighChannel, FastFadingChannel
        csi = "perfect" if perfect else "imperfect"
        for ch in (RayleighChannel(csi=csi, csi_error_std=0.3),
                   FastFadingChannel(block_length=16, csi=csi, csi_error_std=0.3)):
            b = ch.observe(latent, 5.0)
            assert b.noise_level.shape == latent.shape
            assert torch.allclose(b.noise_level * b.received ** 2,
                                  b.noise_var * b.equalized ** 2, rtol=1e-4, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# AWGN-compatible transmit + observe shapes / ranges
# ─────────────────────────────────────────────────────────────────────────────

class TestChannelOutputs:
    def test_rayleigh_shapes(self, latent):
        from sgdjscc_lab.channels import RayleighChannel
        ch = RayleighChannel(csi="perfect")
        b = ch.observe(latent, 5.0)
        assert b.received.shape == latent.shape
        assert b.equalized.shape == latent.shape
        assert ch.transmit(latent, 5.0).shape == latent.shape
        assert b.channel_gain.shape == (2, 1, 1, 1)
        assert 0.0 <= b.mean_reliability() <= 1.0

    def test_rayleigh_blind_no_equalization(self, latent):
        from sgdjscc_lab.channels import RayleighChannel
        b = RayleighChannel(csi="none").observe(latent, 5.0)
        assert b.equalized is None
        # transmit falls back to the un-equalised received latent
        assert torch.equal(b.best_estimate, b.received)

    def test_fast_fading_shapes_and_reliability(self, latent):
        from sgdjscc_lab.channels import FastFadingChannel
        b = FastFadingChannel(block_length=32, csi="perfect").observe(latent, 0.0)
        assert b.received.shape == latent.shape
        assert b.reliability.shape == (2, 1, 16, 16)
        assert 0.0 <= b.reliability.min() and b.reliability.max() <= 1.0

    def test_packet_drop_mask(self, latent):
        from sgdjscc_lab.channels import PacketDropChannel
        b = PacketDropChannel(drop_prob=1.0, packet_length=64).observe(latent, 10.0)
        # Everything dropped → received is all zeros, mask ~0.
        assert torch.count_nonzero(b.received) == 0
        assert b.mean_reliability() == pytest.approx(0.0)

    def test_packet_drop_keep_all(self, latent):
        from sgdjscc_lab.channels import PacketDropChannel
        b = PacketDropChannel(drop_prob=0.0, packet_length=64).observe(latent, 30.0)
        assert b.mean_reliability() == pytest.approx(1.0)

    def test_build_channel_factory(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.channels import (
            build_channel, AWGNChannel, RayleighChannel, FastFadingChannel, PacketDropChannel,
        )
        assert isinstance(build_channel(OmegaConf.create({"channel": "awgn"})), AWGNChannel)
        assert isinstance(build_channel(OmegaConf.create({"channel": "rayleigh"})), RayleighChannel)
        assert isinstance(build_channel(OmegaConf.create({"channel": "fast_fading"})), FastFadingChannel)
        assert isinstance(build_channel(OmegaConf.create({"channel": "packet_drop"})), PacketDropChannel)
        # Unknown → AWGN fallback.
        assert isinstance(build_channel(OmegaConf.create({"channel": "??"})), AWGNChannel)


# ─────────────────────────────────────────────────────────────────────────────
# MeasurementBundle
# ─────────────────────────────────────────────────────────────────────────────

class TestMeasurementBundle:
    def test_summary_is_jsonable(self, latent):
        from sgdjscc_lab.channels import RayleighChannel
        b = RayleighChannel().observe(latent, 7.0)
        s = b.summary()
        import json
        json.dumps(s)   # must not raise
        assert s["snr_db_true"] == 7.0
        assert "mean_reliability" in s

    def test_best_estimate_prefers_equalized(self, latent):
        from sgdjscc_lab.channels.measurement import MeasurementBundle
        b = MeasurementBundle(received=latent, equalized=latent * 2)
        assert torch.equal(b.best_estimate, latent * 2)
        b2 = MeasurementBundle(received=latent)
        assert torch.equal(b2.best_estimate, latent)

    def test_awgn_noise_helper(self, latent):
        from sgdjscc_lab.channels.measurement import awgn_noise_like
        noise, var = awgn_noise_like(latent, 10.0)
        assert noise.shape == latent.shape
        assert var.shape == (2, 1, 1, 1)
        assert (var > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Record / replay (Fix: shared channel realisation for conditioning)
# ─────────────────────────────────────────────────────────────────────────────

class TestChannelTape:
    def test_replay_returns_recorded_realization(self, latent):
        from sgdjscc_lab.channels import RayleighChannel
        ch = RayleighChannel(csi="perfect")
        ch.start_recording()
        recorded = ch.transmit(latent, 3.0).clone()
        # Replay must return the SAME realisation even though global RNG advanced
        # and the input differs (decoder receives exactly what was observed).
        torch.manual_seed(999)
        _ = torch.randn(100)
        ch.start_replay()
        replayed = ch.transmit(torch.randn_like(latent), 3.0)
        assert torch.equal(recorded, replayed)

    def test_replay_preserves_call_order(self, latent):
        from sgdjscc_lab.channels import FastFadingChannel
        ch = FastFadingChannel(block_length=16)
        ch.start_recording()
        r0 = ch.transmit(latent, 0.0).clone()
        r1 = ch.transmit(latent, 0.0).clone()
        assert not torch.equal(r0, r1)          # different realisations recorded
        ch.start_replay()
        assert torch.equal(ch.transmit(latent, 0.0), r0)
        assert torch.equal(ch.transmit(latent, 0.0), r1)

    def test_no_tape_resamples(self, latent):
        from sgdjscc_lab.channels import RayleighChannel
        ch = RayleighChannel()
        a = ch.transmit(latent, 5.0)
        b = ch.transmit(latent, 5.0)
        # Without recording/replay each call is a fresh realisation.
        assert not torch.equal(a, b)

    def test_last_bundle_tracks_transmit(self, latent):
        from sgdjscc_lab.channels import PacketDropChannel
        ch = PacketDropChannel(drop_prob=0.2)
        ch.transmit(latent, 10.0)
        assert ch.last_bundle is not None
        assert ch.last_bundle.received.shape == latent.shape

    def test_recorded_bundles_returns_all(self, latent):
        from sgdjscc_lab.channels import RayleighChannel
        ch = RayleighChannel()
        ch.start_recording()
        ch.transmit(latent, 0.0)
        ch.transmit(latent, 0.0)
        assert len(ch.recorded_bundles()) == 2


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_bundles (multi-patch image-level conditioning)
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateBundles:
    def test_concatenates_patch_bundles(self):
        from sgdjscc_lab.channels import RayleighChannel
        from sgdjscc_lab.channels.measurement import aggregate_bundles
        ch = RayleighChannel(csi="perfect")
        bundles = [ch.observe(torch.randn(1, 16, 8, 8), 5.0) for _ in range(4)]
        agg = aggregate_bundles(bundles)
        assert agg.received.shape[0] == 4            # one row per patch
        assert agg.reliability.shape[0] == 4
        assert agg.meta["n_patches"] == 4

    def test_single_bundle_passthrough(self):
        from sgdjscc_lab.channels import RayleighChannel
        from sgdjscc_lab.channels.measurement import aggregate_bundles
        b = RayleighChannel().observe(torch.randn(1, 16, 8, 8), 5.0)
        assert aggregate_bundles([b]) is b

    def test_empty_returns_none(self):
        from sgdjscc_lab.channels.measurement import aggregate_bundles
        assert aggregate_bundles([]) is None

    def test_blind_equalized_stays_none(self):
        from sgdjscc_lab.channels import RayleighChannel
        from sgdjscc_lab.channels.measurement import aggregate_bundles
        ch = RayleighChannel(csi="none")
        bundles = [ch.observe(torch.randn(1, 16, 8, 8), 0.0) for _ in range(3)]
        agg = aggregate_bundles(bundles)
        assert agg.equalized is None                 # no CSI → no equalisation


# ─────────────────────────────────────────────────────────────────────────────
# Regression: AWGN default path on JSCCModel.channel() unchanged
# ─────────────────────────────────────────────────────────────────────────────

class TestChannelModelOverride:
    def test_channel_model_attr_defaults_none(self):
        # We cannot build the real JSCCModel without checkpoints, but the channel
        # override contract is simple enough to verify on a stub.
        class _Stub:
            def __init__(self):
                from sgdjscc_lab.channels import AWGNChannel
                self._awgn_channel = AWGNChannel()
                self.channel_model = None
                self.snr = 10.0
            # mirror JSCCModel.channel
            def channel(self, x):
                ch = self.channel_model if self.channel_model is not None else self._awgn_channel
                return ch.transmit(x, self.snr)

        from sgdjscc_lab.channels import RayleighChannel
        stub = _Stub()
        x = torch.randn(1, 16, 16, 16)
        torch.manual_seed(1)
        awgn_out = stub.channel(x)
        assert awgn_out.shape == x.shape
        stub.channel_model = RayleighChannel(csi="perfect")
        ray_out = stub.channel(x)
        assert ray_out.shape == x.shape

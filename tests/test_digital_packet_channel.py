"""tests/test_digital_packet_channel.py – DigitalPacketChannel + build_channel wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.channels import AWGNChannel, DigitalPacketChannel, build_channel


class TestBuildChannelDispatch:
    def test_digital_packet_selected_by_name(self):
        ch = build_channel({"channel": "digital_packet", "bit_depth": 4})
        assert isinstance(ch, DigitalPacketChannel)
        assert ch.bit_depth == 4

    def test_default_channel_still_awgn(self):
        ch = build_channel({})
        assert isinstance(ch, AWGNChannel)

    def test_unset_channel_key_falls_back_to_awgn_not_digital(self):
        # regression guard: analog path must stay the default when digital_packet
        # is not explicitly selected (algorithm-preservation invariant).
        ch = build_channel(None)
        assert isinstance(ch, AWGNChannel)


class TestDigitalPacketChannelTransmit:
    @pytest.fixture()
    def latent(self):
        torch.manual_seed(0)
        return torch.randn(1, 16, 16, 16)

    def test_transmit_returns_fresh_tensor_not_input_alias(self, latent):
        ch = DigitalPacketChannel(bit_depth=8)
        out = ch.transmit(latent, snr_db=10.0)
        assert out.shape == latent.shape
        assert out.data_ptr() != latent.data_ptr()

    def test_transmit_records_exact_packet_bytes(self, latent):
        ch = DigitalPacketChannel(bit_depth=4)
        ch.transmit(latent, snr_db=10.0)
        assert ch.last_packet_bytes is not None
        assert ch.last_breakdown is not None
        assert ch.last_breakdown.proxy is False
        assert ch.last_breakdown.total_bytes == len(ch.last_packet_bytes)

    def test_no_analog_noise_is_added_on_top_of_quantization(self, latent):
        # Two transmits of the identical latent at the same bit_depth must be
        # bit-identical (no random AWGN mixed into the digital path) — only the
        # deterministic quantization error should differ from the input.
        ch = DigitalPacketChannel(bit_depth=8)
        out1 = ch.transmit(latent, snr_db=10.0)
        out2 = ch.transmit(latent, snr_db=10.0)
        assert torch.equal(out1, out2)

    def test_batch_dimension_preserved(self):
        torch.manual_seed(0)
        x = torch.randn(3, 16, 16, 16)
        ch = DigitalPacketChannel(bit_depth=8)
        out = ch.transmit(x, snr_db=10.0)
        assert out.shape == x.shape

    def test_multi_patch_batch_records_every_patchs_packet_not_just_last(self):
        # Regression: a real video frame is tiled into multiple 128x128 patches
        # and JSCC batches all of them into one channel.transmit() call (bsz =
        # n_patches). The exact total bytes must sum every patch's packet, not
        # only the last one in the batch.
        torch.manual_seed(0)
        n_patches = 5
        x = torch.randn(n_patches, 16, 16, 16)
        ch = DigitalPacketChannel(bit_depth=8)
        ch.transmit(x, snr_db=10.0)

        assert len(ch.last_packets) == n_patches
        assert len(ch.last_breakdowns) == n_patches
        assert ch.last_total_bytes == sum(len(p) for p in ch.last_packets)
        assert ch.last_total_bytes == sum(b.total_bytes for b in ch.last_breakdowns)
        # a naive "only look at last_breakdown" total would undercount by ~5x
        assert ch.last_total_bytes > ch.last_breakdown.total_bytes * (n_patches - 1)

    def test_accumulator_sums_across_multiple_per_patch_transmit_calls(self):
        # Regression: pipelines/eval_pipeline.py::_reconstruct_with_cfg calls
        # transmit() once PER PATCH (bsz=1 each), not once per frame with all
        # patches batched. last_total_bytes alone would only reflect the last
        # such call; the cross-call accumulator must sum all of them.
        torch.manual_seed(0)
        ch = DigitalPacketChannel(bit_depth=8)
        ch.reset_accumulation()
        n_patches = 6
        expected_total = 0
        for _ in range(n_patches):
            x = torch.randn(1, 16, 16, 16)
            ch.transmit(x, snr_db=10.0)
            expected_total += ch.last_total_bytes
        assert len(ch.all_packets) == n_patches
        assert len(ch.all_breakdowns) == n_patches
        assert ch.all_total_bytes == expected_total

    def test_reset_accumulation_clears_prior_frame_state(self):
        torch.manual_seed(0)
        ch = DigitalPacketChannel(bit_depth=8)
        ch.transmit(torch.randn(1, 16, 16, 16), snr_db=10.0)
        assert ch.all_total_bytes > 0
        ch.reset_accumulation()
        assert ch.all_packets == [] and ch.all_breakdowns == [] and ch.all_total_bytes == 0

    def test_each_patch_in_batch_decodes_independently(self):
        torch.manual_seed(0)
        x = torch.randn(4, 16, 16, 16)
        ch = DigitalPacketChannel(bit_depth=8)
        out = ch.transmit(x, snr_db=10.0)
        from sgdjscc_lab.transmission.wire_packet import decode_latent_packet
        for i, data in enumerate(ch.last_packets):
            recon_i = decode_latent_packet(data)
            assert torch.allclose(recon_i, out[i:i + 1], atol=1e-5)

    def test_keyframe_index_round_trips_through_observe(self, latent):
        ch = DigitalPacketChannel(bit_depth=8)
        ch.keyframe_index = 42
        ch.transmit(latent, snr_db=10.0)
        assert ch.last_bundle.meta["keyframe_index"] == 42


class TestJsccModelChannelModelOverride:
    def test_channel_model_override_used_instead_of_default_awgn(self, monkeypatch):
        """Mirrors the Phase 5-A pattern: JSCCModel.channel() must dispatch to
        channel_model when set, so selecting digital_packet does not also run
        AWGN (no meaningless double application of both transports)."""
        import types

        calls = {"digital": 0, "awgn": 0}

        class FakeDigital:
            def transmit(self, latent, snr_db):
                calls["digital"] += 1
                return latent

        class FakeAwgn:
            def transmit(self, latent, snr_db):
                calls["awgn"] += 1
                return latent

        fake_model = types.SimpleNamespace(
            snr=10.0,
            channel_model=FakeDigital(),
            _awgn_channel=FakeAwgn(),
        )

        def channel(self, encode_features):
            ch = self.channel_model if self.channel_model is not None else self._awgn_channel
            return ch.transmit(encode_features, self.snr)

        fake_model.channel = types.MethodType(channel, fake_model)
        fake_model.channel(torch.zeros(1, 4, 4, 4))

        assert calls["digital"] == 1
        assert calls["awgn"] == 0

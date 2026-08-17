"""tests/test_wire_packet.py – Binary transmission-packet protocol tests.

Covers the "transmission reduction" feature's core correctness contract:
8/6/4-bit quantize/bit-pack round trip (incl. odd lengths + padding),
deterministic serialization, checksum/malformed-packet rejection, exact byte
accounting, and that a receiver decoding a packet never obtains a reference to
the sender's original tensor object.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.transmission.quantization import (
    QuantizationError,
    dequantize_tensor,
    pack_bits,
    quantize_tensor,
    unpack_bits,
)
from sgdjscc_lab.transmission.wire_packet import (
    PacketChecksumError,
    PacketLengthError,
    PacketMagicError,
    PacketVersionError,
    WirePacket,
    decode_latent_packet,
    encode_latent_packet,
    parse,
    serialize,
)
from sgdjscc_lab.transmission.byte_accounting import (
    estimate_channel_symbols,
    estimate_wire_bytes,
    packet_byte_breakdown,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bit packing round trip: 8/6/4-bit, incl. odd lengths and padding
# ─────────────────────────────────────────────────────────────────────────────

class TestBitPackingRoundTrip:
    @pytest.mark.parametrize("bit_depth", [8, 6, 4])
    @pytest.mark.parametrize("n", [0, 1, 3, 7, 8, 9, 16, 17, 4096])
    def test_pack_unpack_round_trip(self, bit_depth, n):
        import numpy as np

        rng = np.random.default_rng(0)
        qmax = (1 << bit_depth) - 1
        codes = rng.integers(0, qmax + 1, size=n, dtype=np.uint32)

        packed, pad_bits = pack_bits(codes, bit_depth)
        assert 0 <= pad_bits < 8
        # total bits packed must always land on a whole number of bytes
        assert (n * bit_depth + pad_bits) % 8 == 0
        assert len(packed) == (n * bit_depth + pad_bits) // 8

        recovered = unpack_bits(packed, bit_depth, n, pad_bits)
        assert np.array_equal(recovered, codes)

    def test_odd_length_padding_bits_are_deterministic_and_discarded(self):
        import numpy as np

        codes = np.array([1, 2, 3], dtype=np.uint32)  # 3 elements, 6-bit -> 18 bits -> pad 6
        packed, pad_bits = pack_bits(codes, 6)
        assert pad_bits == 6
        assert len(packed) == 3  # 24 bits total
        # the padding bits are always zero (not garbage) — re-packing gives identical bytes
        packed2, pad2 = pack_bits(codes, 6)
        assert packed == packed2 and pad_bits == pad2

    def test_unpack_rejects_length_pad_mismatch(self):
        import numpy as np

        codes = np.array([1, 2, 3, 4], dtype=np.uint32)
        packed, pad_bits = pack_bits(codes, 4)
        with pytest.raises(QuantizationError):
            unpack_bits(packed, 4, n_elements=5, pad_bits=pad_bits)  # wrong n_elements

    def test_pack_rejects_out_of_range_code(self):
        import numpy as np

        with pytest.raises(QuantizationError):
            pack_bits(np.array([16], dtype=np.uint32), bit_depth=4)  # 4-bit max is 15


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tensor quantize/dequantize round trip
# ─────────────────────────────────────────────────────────────────────────────

class TestTensorQuantization:
    @pytest.mark.parametrize("bit_depth", [8, 6, 4])
    @pytest.mark.parametrize("granularity", ["per_tensor", "per_channel"])
    def test_round_trip_error_bounded_by_quantization_step(self, bit_depth, granularity):
        torch.manual_seed(0)
        x = torch.randn(1, 16, 16, 16)
        q = quantize_tensor(x, bit_depth=bit_depth, granularity=granularity, channel_dim=1)
        recon = dequantize_tensor(q)
        assert recon.shape == x.shape

        qmax = (1 << bit_depth) - 1
        max_scale = max(q.scale)
        # affine quantization error is bounded by half the step size (+ fp slack)
        assert (recon - x).abs().max().item() <= 0.5 * max_scale + 1e-3

    def test_odd_shape_tensor_round_trips(self):
        torch.manual_seed(0)
        x = torch.randn(1, 3, 5, 7)  # not a multiple of 8 elements, exercises padding path
        q = quantize_tensor(x, bit_depth=4, granularity="per_tensor")
        recon = dequantize_tensor(q)
        assert recon.shape == x.shape

    def test_unsupported_bit_depth_rejected(self):
        with pytest.raises(QuantizationError):
            quantize_tensor(torch.randn(1, 4, 4, 4), bit_depth=5)

    def test_per_channel_uses_one_scale_per_channel(self):
        torch.manual_seed(0)
        x = torch.randn(1, 16, 16, 16) * torch.linspace(1, 5, 16).reshape(1, 16, 1, 1)
        q = quantize_tensor(x, bit_depth=8, granularity="per_channel", channel_dim=1)
        assert len(q.scale) == 16
        assert len(q.zero_point) == 16


# ─────────────────────────────────────────────────────────────────────────────
# 3. Deterministic serialization
# ─────────────────────────────────────────────────────────────────────────────

class TestSerializationDeterminism:
    def test_same_input_produces_byte_identical_packet(self):
        torch.manual_seed(0)
        x = torch.randn(1, 16, 16, 16)
        data1 = encode_latent_packet(x, bit_depth=8, keyframe_index=3, metadata={"b": 2, "a": 1})
        data2 = encode_latent_packet(x, bit_depth=8, keyframe_index=3, metadata={"a": 1, "b": 2})
        assert data1 == data2  # key order in metadata dict must not affect bytes

    def test_different_input_produces_different_packet(self):
        torch.manual_seed(0)
        x1 = torch.randn(1, 16, 16, 16)
        x2 = torch.randn(1, 16, 16, 16)
        assert encode_latent_packet(x1, bit_depth=8) != encode_latent_packet(x2, bit_depth=8)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Checksum + malformed-packet rejection
# ─────────────────────────────────────────────────────────────────────────────

class TestMalformedPacketRejection:
    @pytest.fixture()
    def valid_packet(self):
        torch.manual_seed(0)
        x = torch.randn(1, 16, 16, 16)
        return encode_latent_packet(x, bit_depth=8, keyframe_index=1)

    def test_valid_packet_parses(self, valid_packet):
        pkt = parse(valid_packet)
        assert pkt.bit_depth == 8
        assert pkt.keyframe_index == 1

    def test_bad_magic_rejected(self, valid_packet):
        corrupted = b"XXXX" + valid_packet[4:]
        with pytest.raises(PacketMagicError):
            parse(corrupted)

    def test_bad_version_rejected(self, valid_packet):
        # version is byte offset 4 (after 4-byte magic)
        corrupted = valid_packet[:4] + struct.pack(">B", 99) + valid_packet[5:]
        with pytest.raises(PacketVersionError):
            parse(corrupted)

    def test_severely_truncated_packet_rejected_as_length_error(self, valid_packet):
        # shorter than the fixed header -> can't even read declared lengths
        with pytest.raises(PacketLengthError):
            parse(valid_packet[:8])

    def test_mid_truncated_packet_rejected(self, valid_packet):
        # long enough to parse the header's declared lengths but body/checksum
        # no longer match -> rejected (as a checksum mismatch, since the trailing
        # 4 bytes of the truncated buffer essentially never satisfy crc32 by luck)
        from sgdjscc_lab.transmission.wire_packet import PacketError

        with pytest.raises(PacketError):
            parse(valid_packet[: len(valid_packet) // 2])

    def test_flipped_payload_byte_rejected_by_checksum(self, valid_packet):
        mid = len(valid_packet) // 2
        flipped = bytearray(valid_packet)
        flipped[mid] ^= 0xFF
        with pytest.raises(PacketChecksumError):
            parse(bytes(flipped))

    def test_corrupted_checksum_field_rejected(self, valid_packet):
        corrupted = valid_packet[:-4] + struct.pack(">I", 0xDEADBEEF)
        with pytest.raises(PacketChecksumError):
            parse(corrupted)

    def test_empty_bytes_rejected(self):
        with pytest.raises(PacketLengthError):
            parse(b"")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Receiver cannot bypass-reference the sender's original latent
# ─────────────────────────────────────────────────────────────────────────────

class TestReceiverCannotBypassOriginalLatent:
    def test_decoded_tensor_does_not_share_storage_with_original(self):
        torch.manual_seed(0)
        x = torch.randn(1, 16, 16, 16)
        data = encode_latent_packet(x, bit_depth=8)
        recon = decode_latent_packet(data)
        assert recon.data_ptr() != x.data_ptr()
        assert recon.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()

    def test_decode_only_needs_the_byte_buffer_not_the_original_object(self):
        torch.manual_seed(0)
        x = torch.randn(1, 16, 16, 16)
        data = encode_latent_packet(x, bit_depth=8)
        del x  # simulate the sender-side tensor being gone entirely
        recon = decode_latent_packet(data)  # must still work from bytes alone
        assert recon.shape == (1, 16, 16, 16)

    def test_wire_packet_object_carries_only_quantized_payload_not_float_tensor(self):
        torch.manual_seed(0)
        x = torch.randn(1, 16, 16, 16)
        data = encode_latent_packet(x, bit_depth=4)
        pkt = parse(data)
        assert isinstance(pkt.payload, bytes)
        # payload size is far smaller than a float32 tensor of the same shape,
        # proving no full-precision copy is smuggled through the packet
        assert len(pkt.payload) < x.numel() * 4


# ─────────────────────────────────────────────────────────────────────────────
# 6. Exact byte accounting
# ─────────────────────────────────────────────────────────────────────────────

class TestExactByteAccounting:
    def test_breakdown_reconciles_with_actual_serialized_length(self):
        torch.manual_seed(0)
        x = torch.randn(1, 16, 16, 16)
        q = quantize_tensor(x, bit_depth=6, granularity="per_tensor")
        pkt = WirePacket(
            bit_depth=q.bit_depth, granularity=q.granularity, channel_dim=q.channel_dim,
            shape=q.shape, scale=q.scale, zero_point=q.zero_point, pad_bits=q.pad_bits,
            n_elements=q.n_elements, payload=q.packed, keyframe_index=7, metadata={"k": "v"},
        )
        breakdown = packet_byte_breakdown(pkt, compress_metadata=True)
        serialized = serialize(pkt, compress_metadata=True)
        assert breakdown.total_bytes == len(serialized)
        assert breakdown.proxy is False

    def test_lower_bit_depth_yields_smaller_payload(self):
        torch.manual_seed(0)
        x = torch.randn(1, 16, 16, 16)
        sizes = {}
        for bd in (8, 6, 4):
            data = encode_latent_packet(x, bit_depth=bd, compress_metadata=False)
            sizes[bd] = len(data)
        assert sizes[4] < sizes[6] < sizes[8]

    def test_channel_symbol_and_wire_byte_estimates_are_labeled_proxy(self):
        sym = estimate_channel_symbols(1000, bits_per_symbol=2.0)
        assert sym["proxy"] is True
        wire_no_fec = estimate_wire_bytes(1000, code_rate=1.0)
        assert wire_no_fec["proxy"] is False
        assert wire_no_fec["estimated_wire_bytes"] == 1000
        wire_fec = estimate_wire_bytes(1000, code_rate=0.5)
        assert wire_fec["proxy"] is True
        assert wire_fec["estimated_wire_bytes"] == 2000

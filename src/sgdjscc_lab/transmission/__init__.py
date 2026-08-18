"""sgdjscc_lab.transmission – Real binary packet transport for JSCC latents.

Opt-in digital path alongside the original analog AWGN transport: quantizes
the JSCC latent to 8/6/4-bit codes (:mod:`quantization`), bit-packs and wraps
them in a deterministic binary packet with a checksum
(:mod:`wire_packet`), and exposes exact per-component byte accounting
(:mod:`byte_accounting`). Wired into the channel dispatch as
``channels.digital_packet.DigitalPacketChannel`` so it is selected the same
way Rayleigh/fast-fading/packet-drop are (``channel: digital_packet`` in
config) — the original analog AWGN path is untouched when this is not
selected.
"""

from sgdjscc_lab.transmission.quantization import (
    QuantizedTensor,
    QuantizationError,
    dequantize_tensor,
    pack_bits,
    quantize_tensor,
    unpack_bits,
)
from sgdjscc_lab.transmission.wire_packet import (
    PacketChecksumError,
    PacketError,
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
    PacketByteBreakdown,
    TransmissionMeasurement,
    estimate_channel_symbols,
    estimate_wire_bytes,
    measure_frame_transmission,
    packet_byte_breakdown,
)
from sgdjscc_lab.transmission.packet_bundle import (
    BundleChecksumError,
    BundleError,
    BundleItem,
    BundleLengthError,
    BundleMagicError,
    BundleVersionError,
    TransmissionBundle,
    build_frame_bundle,
    build_side_info_bundle,
    decode_frame_bundle,
    parse_bundle,
    serialize_bundle,
)
from sgdjscc_lab.transmission.receiver_runtime import (
    encode_frame_to_bundle_bytes,
    reconstruct_frame_from_bundle_bytes,
)

__all__ = [
    "QuantizedTensor",
    "QuantizationError",
    "dequantize_tensor",
    "pack_bits",
    "quantize_tensor",
    "unpack_bits",
    "PacketChecksumError",
    "PacketError",
    "PacketLengthError",
    "PacketMagicError",
    "PacketVersionError",
    "WirePacket",
    "decode_latent_packet",
    "encode_latent_packet",
    "parse",
    "serialize",
    "PacketByteBreakdown",
    "TransmissionMeasurement",
    "estimate_channel_symbols",
    "estimate_wire_bytes",
    "measure_frame_transmission",
    "packet_byte_breakdown",
    "BundleChecksumError",
    "BundleError",
    "BundleItem",
    "BundleLengthError",
    "BundleMagicError",
    "BundleVersionError",
    "TransmissionBundle",
    "build_frame_bundle",
    "build_side_info_bundle",
    "decode_frame_bundle",
    "parse_bundle",
    "serialize_bundle",
    "encode_frame_to_bundle_bytes",
    "reconstruct_frame_from_bundle_bytes",
]

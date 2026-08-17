"""transmission/wire_packet.py – Deterministic binary packet for a quantized JSCC latent.

Wire layout (all multi-byte integers big-endian ``>``, fixed field widths via
:mod:`struct`):

    magic                4s      b"SGDW"
    version               B      packet format version (currently 1)
    bit_depth             B      4 | 6 | 8
    granularity           B      0 = per_tensor, 1 = per_channel
    channel_dim           b      signed axis index used for per_channel quant
    ndim                  B      number of tensor dims
    pad_bits              B      trailing zero bits appended by bit-packing (0-7)
    metadata_compressed   B      0/1 — whether the metadata block is zlib-deflated
    n_elements            I      total tensor element count (redundant cross-check
                                  against product(shape), guards against a corrupted
                                  shape silently under/over-reading the payload)
    keyframe_index        I      frame index this packet anchors (0xFFFFFFFF = n/a)
    scale_count           I      number of (scale, zero_point) pairs
    shape[ndim]            I*    tensor shape, one uint32 per dim
    scale[scale_count]     f*    float32 per-tensor/per-channel scale
    zero_point[sc_count]   f*    float32 per-tensor/per-channel zero point
    metadata_length        I     byte length of the metadata block below
    metadata                     JSON (utf-8), optionally zlib-compressed
    payload_length          I    byte length of the packed-bits payload
    payload                      bit-packed quantization codes
    checksum                I    CRC32 (zlib.crc32) over every byte above

Serialization is fully deterministic: fixed struct formats, big-endian byte
order, and metadata JSON dumped with ``sort_keys=True, separators=(",", ":")``
so the same logical packet always produces byte-identical output — required
so a checksum/round-trip test is meaningful and so local/remote runs of the
same input produce byte-identical ``.sgpk`` files.

Parsing is defensive: a corrupted/truncated buffer, wrong magic, unsupported
version, inconsistent declared lengths, or a checksum mismatch all raise a
:class:`PacketError` subclass rather than silently returning wrong data —
required so a receiver can distinguish "no packet" from "hostile/garbled
packet" instead of decoding noise as a valid latent.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sgdjscc_lab.transmission.quantization import (
    QuantizedTensor,
    dequantize_tensor,
    quantize_tensor,
)

PACKET_MAGIC = b"SGDW"
PACKET_VERSION = 1
NO_KEYFRAME_INDEX = 0xFFFFFFFF

_GRANULARITY_CODE = {"per_tensor": 0, "per_channel": 1}
_GRANULARITY_NAME = {v: k for k, v in _GRANULARITY_CODE.items()}

# magic(4s) version(B) bit_depth(B) granularity(B) channel_dim(b) ndim(B)
# pad_bits(B) metadata_compressed(B) n_elements(I) keyframe_index(I) scale_count(I)
_HEADER_FMT = ">4sBBBbBBBIII"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


class PacketError(ValueError):
    """Base class for packet serialization/parsing failures."""


class PacketMagicError(PacketError):
    pass


class PacketVersionError(PacketError):
    pass


class PacketLengthError(PacketError):
    pass


class PacketChecksumError(PacketError):
    pass


@dataclass
class WirePacket:
    bit_depth: int
    granularity: str
    channel_dim: int
    shape: List[int]
    scale: List[float]
    zero_point: List[float]
    pad_bits: int
    n_elements: int
    payload: bytes
    keyframe_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = PACKET_VERSION

    @property
    def payload_bytes(self) -> int:
        return len(self.payload)

    def to_quantized_tensor(self) -> QuantizedTensor:
        return QuantizedTensor(
            packed=self.payload,
            shape=self.shape,
            bit_depth=self.bit_depth,
            granularity=self.granularity,
            channel_dim=self.channel_dim,
            scale=self.scale,
            zero_point=self.zero_point,
            pad_bits=self.pad_bits,
            n_elements=self.n_elements,
        )


def _dumps_metadata(metadata: Dict[str, Any]) -> bytes:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")


def serialize(packet: WirePacket, compress_metadata: bool = True) -> bytes:
    """Deterministically serialize a :class:`WirePacket` to bytes."""
    if packet.bit_depth not in (4, 6, 8):
        raise PacketError(f"unsupported bit_depth={packet.bit_depth!r}")
    granularity_code = _GRANULARITY_CODE.get(packet.granularity)
    if granularity_code is None:
        raise PacketError(f"unsupported granularity={packet.granularity!r}")
    if len(packet.scale) != len(packet.zero_point):
        raise PacketError("scale/zero_point length mismatch")

    metadata_raw = _dumps_metadata(packet.metadata)
    metadata_compressed = 0
    if compress_metadata and metadata_raw:
        deflated = zlib.compress(metadata_raw, level=9)
        if len(deflated) < len(metadata_raw):
            metadata_raw = deflated
            metadata_compressed = 1

    keyframe_index = NO_KEYFRAME_INDEX if packet.keyframe_index is None else int(packet.keyframe_index)

    header = struct.pack(
        _HEADER_FMT,
        PACKET_MAGIC,
        packet.version,
        packet.bit_depth,
        granularity_code,
        packet.channel_dim,
        len(packet.shape),
        packet.pad_bits,
        metadata_compressed,
        packet.n_elements,
        keyframe_index,
        len(packet.scale),
    )

    body = bytearray()
    body += struct.pack(f">{len(packet.shape)}I", *packet.shape)
    body += struct.pack(f">{len(packet.scale)}f", *packet.scale)
    body += struct.pack(f">{len(packet.zero_point)}f", *packet.zero_point)
    body += struct.pack(">I", len(metadata_raw))
    body += metadata_raw
    body += struct.pack(">I", len(packet.payload))
    body += packet.payload

    payload_no_checksum = header + bytes(body)
    checksum = zlib.crc32(payload_no_checksum) & 0xFFFFFFFF
    return payload_no_checksum + struct.pack(">I", checksum)


def parse(data: bytes) -> WirePacket:
    """Parse and validate bytes produced by :func:`serialize`.

    Raises :class:`PacketMagicError` / :class:`PacketVersionError` /
    :class:`PacketLengthError` / :class:`PacketChecksumError` on any
    corruption rather than returning a best-effort guess.
    """
    if len(data) < _HEADER_SIZE + 4:
        raise PacketLengthError(f"packet too short: {len(data)} bytes")

    (
        magic, version, bit_depth, granularity_code, channel_dim, ndim,
        pad_bits, metadata_compressed, n_elements, keyframe_index, scale_count,
    ) = struct.unpack(_HEADER_FMT, data[:_HEADER_SIZE])

    if magic != PACKET_MAGIC:
        raise PacketMagicError(f"bad magic {magic!r}, expected {PACKET_MAGIC!r}")
    if version != PACKET_VERSION:
        raise PacketVersionError(f"unsupported packet version {version}, expected {PACKET_VERSION}")
    if bit_depth not in (4, 6, 8):
        raise PacketError(f"unsupported bit_depth={bit_depth}")
    if granularity_code not in _GRANULARITY_NAME:
        raise PacketError(f"unsupported granularity code={granularity_code}")

    # checksum: verify before trusting any length-derived slicing below.
    body_and_header = data[:-4]
    declared_checksum = struct.unpack(">I", data[-4:])[0]
    actual_checksum = zlib.crc32(body_and_header) & 0xFFFFFFFF
    if actual_checksum != declared_checksum:
        raise PacketChecksumError(
            f"checksum mismatch: declared={declared_checksum:#010x} actual={actual_checksum:#010x}"
        )

    offset = _HEADER_SIZE
    shape_size = ndim * 4
    if offset + shape_size > len(body_and_header):
        raise PacketLengthError("truncated shape block")
    shape = list(struct.unpack(f">{ndim}I", data[offset:offset + shape_size]))
    offset += shape_size

    scale_block = scale_count * 4
    if offset + scale_block > len(body_and_header):
        raise PacketLengthError("truncated scale block")
    scale = list(struct.unpack(f">{scale_count}f", data[offset:offset + scale_block]))
    offset += scale_block

    if offset + scale_block > len(body_and_header):
        raise PacketLengthError("truncated zero_point block")
    zero_point = list(struct.unpack(f">{scale_count}f", data[offset:offset + scale_block]))
    offset += scale_block

    if offset + 4 > len(body_and_header):
        raise PacketLengthError("truncated metadata length field")
    (metadata_length,) = struct.unpack(">I", data[offset:offset + 4])
    offset += 4
    if offset + metadata_length > len(body_and_header):
        raise PacketLengthError("truncated metadata block")
    metadata_raw = data[offset:offset + metadata_length]
    offset += metadata_length

    if metadata_raw:
        if metadata_compressed:
            metadata_raw = zlib.decompress(metadata_raw)
        metadata = json.loads(metadata_raw.decode("utf-8"))
    else:
        metadata = {}

    if offset + 4 > len(body_and_header):
        raise PacketLengthError("truncated payload length field")
    (payload_length,) = struct.unpack(">I", data[offset:offset + 4])
    offset += 4
    if offset + payload_length > len(body_and_header):
        raise PacketLengthError("truncated payload block")
    payload = data[offset:offset + payload_length]
    offset += payload_length

    if offset != len(body_and_header):
        raise PacketLengthError(
            f"trailing/unexpected bytes after payload: {len(body_and_header) - offset} extra"
        )

    expected_n = 1
    for s in shape:
        expected_n *= s
    if expected_n != n_elements:
        raise PacketLengthError(
            f"n_elements={n_elements} inconsistent with shape product {expected_n} for shape={shape}"
        )

    granularity = _GRANULARITY_NAME[granularity_code]
    if granularity == "per_tensor" and scale_count != 1:
        raise PacketLengthError(f"per_tensor packet must have scale_count=1, got {scale_count}")
    if granularity == "per_channel" and channel_dim >= 0 and channel_dim < len(shape):
        if scale_count != shape[channel_dim]:
            raise PacketLengthError(
                f"per_channel packet scale_count={scale_count} != shape[channel_dim]={shape[channel_dim]}"
            )

    return WirePacket(
        bit_depth=bit_depth,
        granularity=granularity,
        channel_dim=channel_dim,
        shape=shape,
        scale=scale,
        zero_point=zero_point,
        pad_bits=pad_bits,
        n_elements=n_elements,
        payload=bytes(payload),
        keyframe_index=None if keyframe_index == NO_KEYFRAME_INDEX else keyframe_index,
        metadata=metadata,
        version=version,
    )


def encode_latent_packet(
    latent,
    bit_depth: int,
    granularity: str = "per_tensor",
    channel_dim: int = 1,
    keyframe_index: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    compress_metadata: bool = True,
) -> bytes:
    """Quantize a latent tensor and serialize it to a deterministic byte packet."""
    q = quantize_tensor(latent, bit_depth=bit_depth, granularity=granularity, channel_dim=channel_dim)
    packet = WirePacket(
        bit_depth=q.bit_depth,
        granularity=q.granularity,
        channel_dim=q.channel_dim,
        shape=q.shape,
        scale=q.scale,
        zero_point=q.zero_point,
        pad_bits=q.pad_bits,
        n_elements=q.n_elements,
        payload=q.packed,
        keyframe_index=keyframe_index,
        metadata=metadata or {},
    )
    return serialize(packet, compress_metadata=compress_metadata)


def decode_latent_packet(data: bytes, dtype=None, device="cpu"):
    """Parse bytes and dequantize back to a fresh latent tensor.

    The returned tensor is reconstructed entirely from ``data`` (via
    :func:`parse` + :func:`~sgdjscc_lab.transmission.quantization.dequantize_tensor`);
    it never references any sender-side tensor object, so a receiver holding
    only ``data`` cannot obtain the original full-precision latent.
    """
    import torch

    packet = parse(data)
    return dequantize_tensor(
        packet.to_quantized_tensor(),
        dtype=dtype or torch.float32,
        device=device,
    )

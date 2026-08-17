"""transmission/byte_accounting.py – Exact packet byte breakdown + labeled estimates.

Splits a serialized :class:`~sgdjscc_lab.transmission.wire_packet.WirePacket`
into its exact component byte counts (header / shape+scale side-info /
metadata / quantized payload / checksum — all counted directly from the real
serialized bytes, not inferred). Also provides *clearly-labeled* downstream
estimates (channel symbols under a given modulation order, wire bytes under a
given FEC code rate) — these carry ``proxy=True`` because no real
modulator/FEC exists in this codebase; only the packet byte count itself is
exact (``proxy=False``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from sgdjscc_lab.transmission.wire_packet import (
    _HEADER_SIZE,
    WirePacket,
    _dumps_metadata,
    serialize,
)


@dataclass
class PacketByteBreakdown:
    header_bytes: int
    shape_bytes: int
    scale_zp_bytes: int
    metadata_bytes: int
    payload_bytes: int
    checksum_bytes: int
    total_bytes: int
    proxy: bool = False  # every field here is an exact byte count of real serialized bytes

    def as_dict(self) -> Dict[str, int]:
        return {
            "header_bytes": self.header_bytes,
            "shape_bytes": self.shape_bytes,
            "scale_zp_bytes": self.scale_zp_bytes,
            "metadata_bytes": self.metadata_bytes,
            "payload_bytes": self.payload_bytes,
            "checksum_bytes": self.checksum_bytes,
            "total_bytes": self.total_bytes,
            "proxy": self.proxy,
        }


def packet_byte_breakdown(packet: WirePacket, compress_metadata: bool = True) -> PacketByteBreakdown:
    """Exact component byte counts derived from the real serialized packet.

    ``total_bytes`` equals ``len(serialize(packet, compress_metadata))`` — this
    function does not estimate; it decomposes the actual serialized output.
    """
    serialized = serialize(packet, compress_metadata=compress_metadata)

    metadata_raw = _dumps_metadata(packet.metadata)
    if compress_metadata and metadata_raw:
        import zlib

        deflated = zlib.compress(metadata_raw, level=9)
        if len(deflated) < len(metadata_raw):
            metadata_raw = deflated
    metadata_bytes = 4 + len(metadata_raw)  # length-prefix + block

    shape_bytes = 4 * len(packet.shape)
    scale_zp_bytes = 4 * len(packet.scale) * 2
    payload_bytes = 4 + len(packet.payload)  # length-prefix + block
    checksum_bytes = 4

    total = len(serialized)
    accounted = _HEADER_SIZE + shape_bytes + scale_zp_bytes + metadata_bytes + payload_bytes + checksum_bytes
    if accounted != total:
        raise AssertionError(
            f"packet byte breakdown does not reconcile: accounted={accounted} total={total}"
        )

    return PacketByteBreakdown(
        header_bytes=_HEADER_SIZE,
        shape_bytes=shape_bytes,
        scale_zp_bytes=scale_zp_bytes,
        metadata_bytes=metadata_bytes,
        payload_bytes=payload_bytes,
        checksum_bytes=checksum_bytes,
        total_bytes=total,
    )


def estimate_channel_symbols(total_bytes: int, bits_per_symbol: float = 1.0) -> Dict[str, object]:
    """Estimated real/complex channel-symbol count for *total_bytes* under a given modulation order.

    ``bits_per_symbol=1.0`` models one BPSK-like real symbol per bit (matches
    the JSCC analog convention of one channel use per latent element). This is
    a labeled estimate (``proxy=True``): no modulator is implemented here.
    """
    if bits_per_symbol <= 0:
        raise ValueError("bits_per_symbol must be > 0")
    symbols = (total_bytes * 8) / bits_per_symbol
    return {"channel_symbols": symbols, "bits_per_symbol": bits_per_symbol, "proxy": True}


def estimate_wire_bytes(total_bytes: int, code_rate: float = 1.0) -> Dict[str, object]:
    """Estimated on-air byte count after FEC overhead at *code_rate* (<=1.0).

    ``code_rate=1.0`` (default) means no FEC modeled — ``estimated_wire_bytes``
    equals ``total_bytes`` exactly in that case. For ``code_rate<1.0`` this is a
    labeled estimate (``proxy=True``): no real FEC coder is implemented here.
    """
    if not (0 < code_rate <= 1.0):
        raise ValueError("code_rate must be in (0, 1.0]")
    return {
        "estimated_wire_bytes": total_bytes / code_rate,
        "code_rate": code_rate,
        "proxy": code_rate != 1.0,
    }

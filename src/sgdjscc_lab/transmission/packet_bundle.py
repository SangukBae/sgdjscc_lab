"""transmission/packet_bundle.py – Full per-frame transmission bundle.

A single :class:`~sgdjscc_lab.transmission.wire_packet.WirePacket` covers only
one quantized tensor (the JSCC visual latent). What a transmitter actually
sends for one keyframe is more than that: a caption, an edge/structural guide
(canny), the selected-keyframe list, and a small manifest — this module
bundles all of them into one deterministic binary container so the *total*
transmission size is never understated as "visual latent bytes only".

Each component is one of:

  - **exact digital bytes** (``is_analog=False``): caption text (UTF-8),
    edge tensor (quantized via :mod:`quantization`, same real bit-packing as
    the visual latent), keyframe-list/manifest JSON. ``byte_len`` is an exact
    count of real serialized bytes.
  - **analog** (``is_analog=True``): used when the visual latent itself went
    over the analog AWGN channel rather than a digital packet. No bytes are
    claimed for it — only ``channel_symbols`` (the exact count of real-valued
    channel symbols transmitted), so an analog component can never be
    silently reported as if it were byte-exact wire data.

:func:`build_frame_bundle` / :func:`decode_frame_bundle` are the transmitter/
receiver pair: the receiver reconstructs the caption string, the edge tensor,
the visual latents (when digital), and the manifest **entirely from the
serialized bytes** — it never touches the sender's original frame/caption/
edge tensor objects (see ``tests/test_packet_bundle.py``'s
``TestReceiverBoundary`` for the proof: original objects are deleted before
decoding and the round trip still succeeds).
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sgdjscc_lab.transmission.quantization import (
    LOSSLESS_BIT_DEPTH,
    dequantize_tensor,
    quantize_tensor,
)
from sgdjscc_lab.transmission.wire_packet import (
    WirePacket,
    parse as parse_wire_packet,
    serialize as serialize_wire_packet,
)

BUNDLE_MAGIC = b"SGDB"
BUNDLE_VERSION = 1
NO_CHANNEL_SYMBOLS = 0xFFFFFFFF

_KIND_WIRE_PACKET = 0
_KIND_RAW_BYTES = 1
_KIND_JSON = 2
_KIND_NAME = {_KIND_WIRE_PACKET: "wire_packet", _KIND_RAW_BYTES: "raw_bytes", _KIND_JSON: "json"}
_NAME_KIND = {v: k for k, v in _KIND_NAME.items()}

# magic(4s) version(B) keyframe_index(I) item_count(I)
_HEADER_FMT = ">4sBII"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
# per item: kind(B) is_analog(B) channel_symbols(I) name_len(H) data_len(I)
_ITEM_FIXED_FMT = ">BBIHI"
_ITEM_FIXED_SIZE = struct.calcsize(_ITEM_FIXED_FMT)


class BundleError(ValueError):
    """Base class for bundle serialization/parsing failures."""


class BundleMagicError(BundleError):
    pass


class BundleVersionError(BundleError):
    pass


class BundleLengthError(BundleError):
    pass


class BundleChecksumError(BundleError):
    pass


@dataclass
class BundleItem:
    """One named component of a :class:`TransmissionBundle`."""

    name: str
    kind: str            # "wire_packet" | "raw_bytes" | "json"
    is_analog: bool = False
    channel_symbols: int = 0    # exact, only meaningful when is_analog=True
    data: bytes = b""            # exact bytes, empty when is_analog=True

    @property
    def byte_len(self) -> int:
        return len(self.data)


@dataclass
class TransmissionBundle:
    keyframe_index: int
    items: List[BundleItem] = field(default_factory=list)
    version: int = BUNDLE_VERSION

    def get(self, name: str) -> Optional[BundleItem]:
        return next((it for it in self.items if it.name == name), None)

    def total_exact_bytes(self) -> int:
        """Sum of every non-analog item's exact byte length (the real transmission size)."""
        return sum(it.byte_len for it in self.items if not it.is_analog)

    def total_analog_channel_symbols(self) -> int:
        return sum(it.channel_symbols for it in self.items if it.is_analog)


def serialize_bundle(bundle: TransmissionBundle) -> bytes:
    """Deterministically serialize a :class:`TransmissionBundle` to bytes."""
    keyframe_index = NO_CHANNEL_SYMBOLS if bundle.keyframe_index is None else int(bundle.keyframe_index)
    header = struct.pack(_HEADER_FMT, BUNDLE_MAGIC, bundle.version, keyframe_index, len(bundle.items))

    body = bytearray()
    for item in bundle.items:
        kind_code = _NAME_KIND.get(item.kind)
        if kind_code is None:
            raise BundleError(f"unsupported item kind={item.kind!r}")
        name_bytes = item.name.encode("utf-8")
        symbols = NO_CHANNEL_SYMBOLS if not item.is_analog else int(item.channel_symbols)
        body += struct.pack(
            _ITEM_FIXED_FMT, kind_code, 1 if item.is_analog else 0, symbols,
            len(name_bytes), len(item.data),
        )
        body += name_bytes
        body += item.data

    payload_no_checksum = header + bytes(body)
    checksum = zlib.crc32(payload_no_checksum) & 0xFFFFFFFF
    return payload_no_checksum + struct.pack(">I", checksum)


def parse_bundle(data: bytes) -> TransmissionBundle:
    """Parse and validate bytes produced by :func:`serialize_bundle`."""
    if len(data) < _HEADER_SIZE + 4:
        raise BundleLengthError(f"bundle too short: {len(data)} bytes")

    magic, version, keyframe_index, item_count = struct.unpack(_HEADER_FMT, data[:_HEADER_SIZE])
    if magic != BUNDLE_MAGIC:
        raise BundleMagicError(f"bad magic {magic!r}, expected {BUNDLE_MAGIC!r}")
    if version != BUNDLE_VERSION:
        raise BundleVersionError(f"unsupported bundle version {version}, expected {BUNDLE_VERSION}")

    body_and_header = data[:-4]
    declared_checksum = struct.unpack(">I", data[-4:])[0]
    actual_checksum = zlib.crc32(body_and_header) & 0xFFFFFFFF
    if actual_checksum != declared_checksum:
        raise BundleChecksumError(
            f"checksum mismatch: declared={declared_checksum:#010x} actual={actual_checksum:#010x}"
        )

    offset = _HEADER_SIZE
    items: List[BundleItem] = []
    for _ in range(item_count):
        if offset + _ITEM_FIXED_SIZE > len(body_and_header):
            raise BundleLengthError("truncated item header")
        kind_code, is_analog_flag, symbols, name_len, data_len = struct.unpack(
            _ITEM_FIXED_FMT, data[offset:offset + _ITEM_FIXED_SIZE]
        )
        offset += _ITEM_FIXED_SIZE
        if kind_code not in _KIND_NAME:
            raise BundleError(f"unsupported item kind code={kind_code}")

        if offset + name_len > len(body_and_header):
            raise BundleLengthError("truncated item name")
        name = data[offset:offset + name_len].decode("utf-8")
        offset += name_len

        if offset + data_len > len(body_and_header):
            raise BundleLengthError("truncated item data")
        item_data = data[offset:offset + data_len]
        offset += data_len

        items.append(BundleItem(
            name=name,
            kind=_KIND_NAME[kind_code],
            is_analog=bool(is_analog_flag),
            channel_symbols=0 if symbols == NO_CHANNEL_SYMBOLS else symbols,
            data=bytes(item_data),
        ))

    if offset != len(body_and_header):
        raise BundleLengthError(f"trailing/unexpected bytes: {len(body_and_header) - offset} extra")

    return TransmissionBundle(
        keyframe_index=None if keyframe_index == NO_CHANNEL_SYMBOLS else keyframe_index,
        items=items,
        version=version,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Transmitter: build a full frame bundle (visual + caption + edge + manifest)
# ─────────────────────────────────────────────────────────────────────────────

def _visual_elements(patches_shape) -> int:
    n_patches, c, h, w = patches_shape
    return int(n_patches) * int(c) * int(h) * int(w)


def build_frame_bundle(
    visual_latent_patches,          # torch.Tensor [N,C,H,W] or None
    visual_is_analog: bool,
    visual_bit_depth: Optional[int],
    visual_granularity: str,
    visual_channel_dim: int,
    visual_channel_symbols: int,     # exact latent element count (always known)
    caption: str,
    edge_tensor,                     # torch.Tensor or None
    edge_bit_depth: int,
    keyframe_index: int,
    manifest: Dict[str, Any],
    compress_metadata: bool = True,
) -> TransmissionBundle:
    """Build the full transmission bundle for one keyframe.

    ``visual_is_analog=True`` means the visual latent was sent over the
    analog AWGN channel: no visual wire bytes are recorded, only
    ``visual_channel_symbols`` (exact). ``visual_is_analog=False`` requires
    ``visual_latent_patches`` and ``visual_bit_depth`` to build a real
    per-patch :class:`WirePacket` for the visual component (exact bytes).

    Caption/edge/manifest are always digital (this repo's project convention
    never applies AWGN directly to caption/edge side-info — see CLAUDE.md's
    "Guide corruption rule"), so they always contribute exact bytes.
    """
    items: List[BundleItem] = []

    if visual_is_analog:
        items.append(BundleItem(
            name="visual", kind="raw_bytes", is_analog=True,
            channel_symbols=int(visual_channel_symbols), data=b"",
        ))
    else:
        if visual_latent_patches is None or visual_bit_depth is None:
            raise ValueError("visual_is_analog=False requires visual_latent_patches and visual_bit_depth")
        n_patches = visual_latent_patches.shape[0]
        for patch_idx in range(n_patches):
            sample = visual_latent_patches[patch_idx:patch_idx + 1]
            q = quantize_tensor(sample, bit_depth=visual_bit_depth,
                                 granularity=visual_granularity, channel_dim=visual_channel_dim)
            packet = WirePacket(
                bit_depth=q.bit_depth, granularity=q.granularity, channel_dim=q.channel_dim,
                shape=q.shape, scale=q.scale, zero_point=q.zero_point, pad_bits=q.pad_bits,
                n_elements=q.n_elements, payload=q.packed, keyframe_index=keyframe_index,
                metadata={"patch_index": patch_idx},
            )
            data = serialize_wire_packet(packet, compress_metadata=compress_metadata)
            items.append(BundleItem(name=f"visual_patch_{patch_idx:03d}", kind="wire_packet", data=data))

    caption_bytes = (caption or "").encode("utf-8")
    items.append(BundleItem(name="caption", kind="raw_bytes", data=caption_bytes))

    if edge_tensor is not None:
        q_edge = quantize_tensor(edge_tensor, bit_depth=edge_bit_depth, granularity="per_tensor")
        edge_packet = WirePacket(
            bit_depth=q_edge.bit_depth, granularity=q_edge.granularity, channel_dim=q_edge.channel_dim,
            shape=q_edge.shape, scale=q_edge.scale, zero_point=q_edge.zero_point, pad_bits=q_edge.pad_bits,
            n_elements=q_edge.n_elements, payload=q_edge.packed, keyframe_index=keyframe_index,
        )
        edge_data = serialize_wire_packet(edge_packet, compress_metadata=compress_metadata)
        items.append(BundleItem(name="edge", kind="wire_packet", data=edge_data))

    manifest_full = dict(manifest)
    manifest_full["keyframe_index"] = keyframe_index
    manifest_bytes = json.dumps(manifest_full, sort_keys=True, separators=(",", ":")).encode("utf-8")
    items.append(BundleItem(name="manifest", kind="json", data=manifest_bytes))

    return TransmissionBundle(keyframe_index=keyframe_index, items=items)


# ─────────────────────────────────────────────────────────────────────────────
# Receiver: reconstruct purely from bytes — never the sender's original objects
# ─────────────────────────────────────────────────────────────────────────────

def decode_frame_bundle(data: bytes, dtype=None, device: str = "cpu") -> Dict[str, Any]:
    """Parse *data* and reconstruct every component from the bytes alone.

    Returns a dict with:
      ``visual_latents``: list[Tensor] (one per patch) or None (analog visual)
      ``visual_channel_symbols``: int (exact, always present)
      ``visual_is_analog``: bool
      ``caption``: str
      ``edge``: Tensor or None
      ``manifest``: dict
      ``keyframe_index``: int or None

    Every field is derived only from ``data`` — the sender's original frame/
    caption/edge tensor objects are never referenced (this function's only
    input is a byte string).
    """
    import torch

    bundle = parse_bundle(data)

    visual_item = bundle.get("visual")
    visual_patch_items = sorted(
        (it for it in bundle.items if it.name.startswith("visual_patch_")),
        key=lambda it: it.name,
    )

    visual_latents = None
    visual_is_analog = False
    visual_channel_symbols = 0
    if visual_item is not None and visual_item.is_analog:
        visual_is_analog = True
        visual_channel_symbols = visual_item.channel_symbols
    elif visual_patch_items:
        visual_latents = []
        for it in visual_patch_items:
            packet = parse_wire_packet(it.data)
            tensor = dequantize_tensor(packet.to_quantized_tensor(), dtype=dtype or torch.float32, device=device)
            visual_latents.append(tensor)
            visual_channel_symbols += packet.n_elements

    caption_item = bundle.get("caption")
    caption = caption_item.data.decode("utf-8") if caption_item is not None else ""

    edge_item = bundle.get("edge")
    edge_tensor = None
    if edge_item is not None:
        edge_packet = parse_wire_packet(edge_item.data)
        edge_tensor = dequantize_tensor(edge_packet.to_quantized_tensor(), dtype=dtype or torch.float32, device=device)

    manifest_item = bundle.get("manifest")
    manifest = json.loads(manifest_item.data.decode("utf-8")) if manifest_item is not None else {}

    return {
        "visual_latents": visual_latents,
        "visual_is_analog": visual_is_analog,
        "visual_channel_symbols": visual_channel_symbols,
        "caption": caption,
        "edge": edge_tensor,
        "manifest": manifest,
        "keyframe_index": bundle.keyframe_index,
    }

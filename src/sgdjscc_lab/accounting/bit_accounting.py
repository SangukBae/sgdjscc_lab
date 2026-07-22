"""accounting/bit_accounting.py – Transmission bit / channel-symbol accounting
PoC (ETRI 6차, 구현 실행 순서 11-12).

Scope note (read before citing a number from this module)
-----------------------------------------------------------
ETRI's question behind 순서 11-12 is: does the temporal/generate pipeline
reduce not just *semantic units* (already tracked by
``video/temporal_pipeline.py``'s ``transmitted_units``/``naive_units``) but
also **channel symbols or bits**? This module answers that structurally by
assigning every frame's already-computed decision
(``keyframe``/``reuse``/``recompute_semantic``/``recompute_motion``/``generate``)
a per-component bit/channel-symbol cost, using real data wherever it exists in
this repo and a clearly-flagged proxy formula otherwise.

**This is not a real bitstream/CBR implementation.** There is no entropy
coder, no modulation/coding-rate model, no actual channel-coded side-info
stream in this codebase — every number here is either:

- an *exact* count of a real, already-produced artefact (e.g. the UTF-8 byte
  length of the actual semantic-packet JSON this repo already serialises via
  ``utils/packet_io.py``), or
- an explicit, documented *proxy* derived from real architecture constants or
  a configurable ratio (e.g. the VAE latent element count inferred from the
  frame's pixel shape + the fixed ``models/jscc_model.py`` architecture, or
  the edge side-info symbol count as a configurable fraction of the visual
  latent).

Every :class:`Component` carries a ``proxy: bool`` flag and a human-readable
``note`` so a report reader (or ETRI) never mistakes a proxy number for a
verified bitrate. See docs/etri_strategy.md 6차 구현 결과 for what is and is
not claimed.

Architecture constants used below (real, not invented)
----------------------------------------------------------
- ``models/jscc_model.py::DDCONFIG``: ``z_channels=16``, ``resolution=128``,
  ``ch_mult=[1,2,4,4]`` → spatial downsample factor ``2**(len(ch_mult)-1) =
  8`` → one 128×128 patch's VAE latent is ``16 × 16 × 16 = 4096`` real-valued
  elements. ``channels/awgn.py::AWGNChannel.transmit()`` adds noise
  element-wise to exactly this tensor, so "one latent element = one channel
  symbol" is the real transmission unit here (not a guess).
- ``SGDJSCC/inference_one.py``'s Canny transmission net
  (``in_feature=8192, size1=640, size2=320``): the channel-coded
  representation of the structural (edge) guide is ``size2=320`` real values
  per patch. ``DEFAULT_EDGE_CR = 320 / 4096`` uses this ratio as the default
  ``edge_cr`` proxy so it isn't an arbitrary guess, though the exact trigger
  condition for Canny retransmission (blind-SNR-driven) is not modelled here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# ── Real architecture constants (models/jscc_model.py DDCONFIG) ──────────────
Z_CHANNELS = 16
PATCH_SIZE = 128
DOWNSAMPLE_FACTOR = 8  # 2 ** (len(ch_mult) - 1), ch_mult = [1, 2, 4, 4]
LATENT_SPATIAL = PATCH_SIZE // DOWNSAMPLE_FACTOR  # 16
LATENT_ELEMENTS_PER_PATCH = Z_CHANNELS * LATENT_SPATIAL * LATENT_SPATIAL  # 4096

# Canny channel-encoder output dim per patch (SGDJSCC/inference_one.py: size2=320)
CANNY_CHANNEL_SYMBOLS_PER_PATCH = 320
DEFAULT_EDGE_CR = CANNY_CHANNEL_SYMBOLS_PER_PATCH / LATENT_ELEMENTS_PER_PATCH  # 0.078125

DEFAULT_MOTION_BITS_PER_BLOCK = 8.0   # quantized block-map proxy (256 levels/block)
DEFAULT_SYMBOLS_PER_BIT = 1.0         # bits->channel-symbol folding proxy (1.0 = no modulation gain assumed)

UNIT_BITS = "bits"
UNIT_SYMBOLS = "channel_symbols"
UNIT_SEMANTIC = "semantic_units"

BASELINE_NAIVE_FULL_FRAME = "naive_full_frame_packet"
BASELINE_KEYFRAME_ONLY_LGVSC = "keyframe_only_lgvsc_style"
BASELINES = (BASELINE_NAIVE_FULL_FRAME, BASELINE_KEYFRAME_ONLY_LGVSC)

_COMPONENT_KEYS = (
    "keyframe_visual_latent_symbols",
    "edge_side_info_symbols",
    "caption_bits",
    "semantic_packet_bits",
    "motion_side_info_bits",
    "generated_frame_symbols",
    "reused_frame_symbols",
    "recompute_frame_symbols",
)


@dataclass
class Component:
    """One transmission-payload component's accounted cost.

    ``unit`` is one of ``UNIT_BITS`` / ``UNIT_SYMBOLS`` / ``UNIT_SEMANTIC``.
    ``proxy=True`` means *value* is a documented approximation, not a
    measurement of an actual bitstream (see module docstring).
    """

    value: float
    unit: str
    proxy: bool
    note: str = ""

    def to_dict(self) -> Dict:
        return {"value": float(self.value), "unit": self.unit, "proxy": bool(self.proxy), "note": self.note}


_ZERO_SYMBOLS = Component(0.0, UNIT_SYMBOLS, False, "not applicable to this decision")
_ZERO_BITS = Component(0.0, UNIT_BITS, False, "not applicable to this decision")


def _zero_components() -> Dict[str, Component]:
    return {
        "keyframe_visual_latent_symbols": _ZERO_SYMBOLS,
        "edge_side_info_symbols": _ZERO_SYMBOLS,
        "caption_bits": _ZERO_BITS,
        "semantic_packet_bits": _ZERO_BITS,
        "motion_side_info_bits": _ZERO_BITS,
        "generated_frame_symbols": _ZERO_SYMBOLS,
        "reused_frame_symbols": _ZERO_SYMBOLS,
        "recompute_frame_symbols": _ZERO_SYMBOLS,
    }


# ── Per-component calculators ─────────────────────────────────────────────────

def n_patches(height: int, width: int, patch: int = PATCH_SIZE) -> int:
    """Number of ``patch``×``patch`` tiles covering a ``height``×``width`` frame
    (matches ``pipelines/infer_pipeline.py``'s 128×128 tiling convention)."""
    return max(1, math.ceil(height / patch)) * max(1, math.ceil(width / patch))


def frame_hw(frame) -> Optional[tuple]:
    """Extract ``(H, W)`` from a ``[..., H, W]`` tensor, or ``None``."""
    if frame is None:
        return None
    shape = tuple(frame.shape)
    if len(shape) < 2:
        return None
    return int(shape[-2]), int(shape[-1])


def visual_latent_symbols(frame=None, latent_symbols_override: Optional[float] = None) -> Component:
    """VAE-latent channel-symbol count for one full frame transmission.

    Always ``proxy=True``: even the frame-shape-based patch count is an
    *inference* from the reconstructed frame's pixel shape + the fixed VAE
    architecture constants (see module docstring) — this module never has
    access to the actual ``encode_features`` tensor JSCC_model.channel()
    transmits, since ``FrameRecord``/``TemporalPipeline`` do not retain it.
    """
    if latent_symbols_override is not None:
        return Component(
            float(latent_symbols_override), UNIT_SYMBOLS, True,
            "config override (accounting.latent_symbols_per_frame), not derived from this frame's shape",
        )
    hw = frame_hw(frame)
    if hw is None:
        return Component(0.0, UNIT_SYMBOLS, True, "no frame tensor available to infer patch count from")
    h, w = hw
    patches = n_patches(h, w)
    val = patches * LATENT_ELEMENTS_PER_PATCH
    return Component(
        float(val), UNIT_SYMBOLS, True,
        f"{patches} x {PATCH_SIZE}px patch(es) x {LATENT_ELEMENTS_PER_PATCH} latent elements/patch "
        f"(z_channels={Z_CHANNELS}, downsample={DOWNSAMPLE_FACTOR}) — inferred from frame shape + "
        "fixed VAE architecture constants, not measured from an actual encode_features tensor",
    )


def edge_side_info_symbols(latent_symbols_value: float, edge_cr: float = DEFAULT_EDGE_CR) -> Component:
    """Structural (edge/Canny) side-info channel-symbol proxy.

    Proxy: a configurable fraction of the visual latent symbol count. The
    default ratio (``DEFAULT_EDGE_CR``) is the real Canny channel-encoder
    output dimension (320) over the real per-patch VAE latent size (4096),
    not an arbitrary guess — but whether/when Canny retransmission actually
    fires (blind-SNR-driven, see CLAUDE.md) is not modelled here; this
    accounts for the structural guide as if it always accompanies a full
    visual-latent transmission.
    """
    val = float(latent_symbols_value) * float(edge_cr)
    return Component(
        val, UNIT_SYMBOLS, True,
        f"CR-based proxy: edge_cr={float(edge_cr):.4f} x visual latent symbols "
        f"(default derived from Canny channel-encoder size2={CANNY_CHANNEL_SYMBOLS_PER_PATCH} "
        f"/ {LATENT_ELEMENTS_PER_PATCH} latent elements per patch)",
    )


def caption_bits(caption: Optional[str]) -> Component:
    """UTF-8 byte length x 8 of the actual caption string — exact, not a proxy."""
    if not caption:
        return Component(0.0, UNIT_BITS, False, "no caption")
    n = len(str(caption).encode("utf-8")) * 8
    return Component(float(n), UNIT_BITS, False, "UTF-8 byte length x 8 of the actual caption string")


def semantic_packet_bits(packet: Optional[Dict]) -> Component:
    """UTF-8 JSON byte length x 8 of the actual semantic packet dict — exact,
    not a proxy (this repo already serialises packets to JSON this way, see
    ``utils/packet_io.py``). This is a real byte count of an artefact this
    repo actually produces, though a production bitstream would use a denser
    binary/entropy-coded representation instead of JSON text.

    Note: the packet's ``caption`` field is included here (JSON text), so
    :func:`caption_bits` is reported *separately for visibility only* and
    must not be added again on top of this when computing totals.
    """
    if not packet:
        return Component(0.0, UNIT_BITS, False, "no packet")
    s = json.dumps(packet, ensure_ascii=False, sort_keys=True, default=str)
    n = len(s.encode("utf-8")) * 8
    return Component(
        float(n), UNIT_BITS, False,
        "UTF-8 JSON byte length x 8 of the actual semantic packet dict (includes caption text; "
        "caption_bits is reported separately for visibility and is NOT double-counted in totals)",
    )


def motion_side_info_bits(
    motion: Optional[Dict], bits_per_block: float = DEFAULT_MOTION_BITS_PER_BLOCK,
) -> Component:
    """Quantized keyframe-anchored motion block-map bit proxy.

    Proxy: ``video/motion_residual.py::estimate()``'s real ``block_map``
    (grid x grid list of block residual energies) x a configurable
    bits-per-block quantization (default 8 bits = 256 levels/block). There is
    no actual quantizer/entropy coder for this map in this repo.
    """
    block_map = (motion or {}).get("block_map") if motion else None
    if not block_map:
        return Component(0.0, UNIT_BITS, True, "no motion block map available (motion gate off or not generate decision)")
    if block_map and isinstance(block_map[0], (list, tuple)):
        n_blocks = sum(len(row) for row in block_map)
    else:
        n_blocks = len(block_map)
    val = float(n_blocks) * float(bits_per_block)
    return Component(
        val, UNIT_BITS, True,
        f"quantized block-map proxy: {n_blocks} block(s) x {bits_per_block:g} bits/block",
    )


@dataclass
class TransmissionAccountingRecord:
    """One frame's transmission-cost accounting (ETRI 6차, step 11)."""

    frame_index: object
    decision: Optional[str]
    role: Optional[str]
    components: Dict[str, Dict] = field(default_factory=dict)
    total_bits: float = 0.0
    total_channel_symbols: float = 0.0
    total_semantic_units: float = 0.0
    proxy_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "frame_index": self.frame_index,
            "decision": self.decision,
            "role": self.role,
            "components": dict(self.components),
            "total_bits": self.total_bits,
            "total_channel_symbols": self.total_channel_symbols,
            "total_semantic_units": self.total_semantic_units,
            "proxy_notes": list(self.proxy_notes),
        }


def account_frame(
    frame_index,
    decision: Optional[str],
    role: Optional[str] = None,
    frame=None,
    packet: Optional[Dict] = None,
    caption: Optional[str] = None,
    motion: Optional[Dict] = None,
    transmitted_units: float = 0.0,
    latent_symbols_override: Optional[float] = None,
    edge_cr: float = DEFAULT_EDGE_CR,
    motion_bits_per_block: float = DEFAULT_MOTION_BITS_PER_BLOCK,
    symbols_per_bit: float = DEFAULT_SYMBOLS_PER_BIT,
) -> TransmissionAccountingRecord:
    """Account one frame's transmission cost given its already-made
    ``TemporalPipeline`` decision.

    Per-decision payload model (see module docstring for the reasoning):

    - ``keyframe``: full visual latent + edge side-info + the full semantic
      packet (needed to establish the GOP reference).
    - ``reuse``: nothing new is transmitted (all components zero).
    - ``recompute_semantic`` / ``recompute_motion``: same payload as a
      keyframe (a fresh full JSCC forward pass runs), but attributed to
      ``recompute_frame_symbols`` instead of ``keyframe_visual_latent_symbols``
      so per-decision totals stay unambiguous.
    - ``generate``: only caption + motion/side-info are modelled as sent (the
      mock ``video_generator`` backends condition on the already-received
      keyframe reconstruction + caption/side-info, per
      ``TemporalPipeline._generate_frame()`` — no fresh visual latent is
      transmitted in this PoC's accounting).

    An unrecognised/``None`` decision leaves every component at zero.
    """
    components = _zero_components()

    if decision == "keyframe":
        lat = visual_latent_symbols(frame, latent_symbols_override)
        components["keyframe_visual_latent_symbols"] = lat
        components["edge_side_info_symbols"] = edge_side_info_symbols(lat.value, edge_cr)
        components["semantic_packet_bits"] = semantic_packet_bits(packet)
    elif decision == "reuse":
        components["reused_frame_symbols"] = Component(
            0.0, UNIT_SYMBOLS, False, "reuse transmits nothing new (real, not a proxy)",
        )
    elif decision in ("recompute_semantic", "recompute_motion"):
        lat = visual_latent_symbols(frame, latent_symbols_override)
        components["recompute_frame_symbols"] = Component(
            lat.value, UNIT_SYMBOLS, lat.proxy,
            "same visual-latent cost as a keyframe (fresh JSCC forward pass); " + lat.note,
        )
        components["edge_side_info_symbols"] = edge_side_info_symbols(lat.value, edge_cr)
        components["semantic_packet_bits"] = semantic_packet_bits(packet)
    elif decision == "generate":
        components["generated_frame_symbols"] = Component(
            0.0, UNIT_SYMBOLS, True,
            "generate branch conditions on the already-received keyframe recon + caption/side-info; "
            "no additional visual latent is modelled as transmitted (PoC assumption, see "
            "video/video_generator.py's Rx-legal boundary note)",
        )
        components["caption_bits"] = caption_bits(caption)
        components["motion_side_info_bits"] = motion_side_info_bits(motion, motion_bits_per_block)

    total_bits = sum(c.value for c in components.values() if c.unit == UNIT_BITS)
    total_symbols_direct = sum(c.value for c in components.values() if c.unit == UNIT_SYMBOLS)
    total_channel_symbols = total_symbols_direct + total_bits * float(symbols_per_bit)
    proxy_notes = [f"{k}: {v.note}" for k, v in components.items() if v.proxy and v.value]

    return TransmissionAccountingRecord(
        frame_index=frame_index,
        decision=decision,
        role=role,
        components={k: v.to_dict() for k, v in components.items()},
        total_bits=total_bits,
        total_channel_symbols=total_channel_symbols,
        total_semantic_units=float(transmitted_units or 0.0),
        proxy_notes=proxy_notes,
    )


# ── Naive baselines (ETRI 6차, step 11 comparison protocol) ───────────────────

def account_frame_as_full(
    frame_index, role: Optional[str] = None, frame=None, packet: Optional[Dict] = None, **kw,
) -> TransmissionAccountingRecord:
    """Hypothetical "as if this frame were fully transmitted" cost — the
    per-frame unit the ``naive_full_frame_packet`` baseline sums over every
    frame regardless of what its real decision was. Uses this frame's own
    real packet/shape data (not a fixed global constant)."""
    return account_frame(frame_index, decision="keyframe", role=role, frame=frame, packet=packet, **kw)


def account_frame_as_side_info_only(
    frame_index, role: Optional[str] = None, packet: Optional[Dict] = None, motion: Optional[Dict] = None, **kw,
) -> TransmissionAccountingRecord:
    """Hypothetical "as if this inter-frame were always side-info-only
    generated" cost — the per-inter-frame unit the
    ``keyframe_only_lgvsc_style`` baseline uses."""
    caption = (packet or {}).get("caption") if packet else None
    return account_frame(frame_index, decision="generate", role=role, caption=caption, motion=motion, **kw)


def compute_baseline_record(
    baseline: str, frame_index, role: Optional[str] = None, frame=None,
    packet: Optional[Dict] = None, motion: Optional[Dict] = None, **kw,
) -> TransmissionAccountingRecord:
    """Compute one frame's cost under *baseline* (see :data:`BASELINES`).

    ``naive_full_frame_packet``: every frame assumed fully transmitted
    (visual latent + edge + packet), regardless of real role/decision.

    ``keyframe_only_lgvsc_style``: only real keyframes are fully transmitted;
    every other frame is assumed side-info-only (caption + motion), mirroring
    an LGVSC-style "only keyframes carry heavy visual payload" design.
    """
    if baseline == BASELINE_NAIVE_FULL_FRAME:
        return account_frame_as_full(frame_index, role=role, frame=frame, packet=packet, **kw)
    if baseline == BASELINE_KEYFRAME_ONLY_LGVSC:
        if role == "keyframe":
            return account_frame_as_full(frame_index, role=role, frame=frame, packet=packet, **kw)
        return account_frame_as_side_info_only(frame_index, role=role, packet=packet, motion=motion, **kw)
    raise NotImplementedError(f"Unknown baseline {baseline!r}; expected one of {BASELINES}")


BASELINE_METADATA: Dict[str, Dict] = {
    BASELINE_NAIVE_FULL_FRAME: {
        "assumption": "Every frame (keyframe or inter) transmits a full visual latent + edge side-info + "
                       "the full semantic packet, as if no reuse/generate/temporal reasoning existed.",
        "proxy": True,
        "not_a_real_cbr": True,
    },
    BASELINE_KEYFRAME_ONLY_LGVSC: {
        "assumption": "Only real keyframes transmit a full visual latent + edge side-info + packet; every "
                       "inter-frame is assumed side-info-only (caption + quantized motion block map), "
                       "mirroring an LGVSC-style generate-everything-between-keyframes design.",
        "proxy": True,
        "not_a_real_cbr": True,
    },
}

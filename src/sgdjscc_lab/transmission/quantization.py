"""transmission/quantization.py – Uniform affine quantization + real bit packing.

Quantizes a float32 JSCC latent tensor to 8/6/4-bit unsigned integer codes
(uniform affine: ``code = round((x - zero_point) / scale)``) and packs the
codes into a dense bitstream (no byte padding per element — 4-bit and 6-bit
codes are bit-packed, not stored one-per-byte). This is real compression: a
4-bit tensor occupies ~4/32 of the original float32 storage, not merely a
narrowed-but-still-byte-aligned representation.

Two quantization granularities:
  - ``per_tensor``:  one (scale, zero_point) pair for the whole tensor.
  - ``per_channel``: one (scale, zero_point) pair per slice along
    ``channel_dim`` (dim 1 for a JSCC latent ``[B, C, H, W]``).

Padding: bit-packing works on a flat 1-D array of codes. When
``n_elements * bit_depth`` is not a multiple of 8, the final byte is padded
with zero bits; ``pad_bits`` (0-7) records how many trailing bits to discard
on unpack so round-trip is exact regardless of tensor size/oddness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import torch

SUPPORTED_BIT_DEPTHS = (8, 6, 4)


class QuantizationError(ValueError):
    pass


@dataclass
class QuantizedTensor:
    """Result of quantizing a tensor: packed codes + everything needed to invert it."""

    packed: bytes
    shape: List[int]
    bit_depth: int
    granularity: str          # "per_tensor" | "per_channel"
    channel_dim: int
    scale: List[float]
    zero_point: List[float]
    pad_bits: int
    n_elements: int

    @property
    def payload_bytes(self) -> int:
        return len(self.packed)


def _validate_bit_depth(bit_depth: int) -> None:
    if bit_depth not in SUPPORTED_BIT_DEPTHS:
        raise QuantizationError(
            f"unsupported bit_depth={bit_depth!r}; must be one of {SUPPORTED_BIT_DEPTHS}"
        )


def pack_bits(codes: np.ndarray, bit_depth: int) -> "tuple[bytes, int]":
    """Pack an array of non-negative integer codes (< 2**bit_depth) into bytes.

    Vectorized (no Python-level per-element loop): each code is expanded to
    ``bit_depth`` bits (MSB first), concatenated, right-padded with zero bits
    to a byte boundary, then packed via ``numpy.packbits``. Returns
    ``(packed_bytes, pad_bits)`` where ``pad_bits`` is how many zero bits were
    appended (0-7).
    """
    codes = np.asarray(codes, dtype=np.uint32).reshape(-1)
    qmax = (1 << bit_depth) - 1
    if codes.size and (codes.max(initial=0) > qmax or codes.min(initial=0) < 0):
        raise QuantizationError(f"code out of range for bit_depth={bit_depth}")

    if codes.size == 0:
        return b"", 0

    shifts = np.arange(bit_depth - 1, -1, -1, dtype=np.uint32)
    bits = ((codes[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)

    total_bits = bits.size
    pad_bits = (-total_bits) % 8
    if pad_bits:
        bits = np.concatenate([bits, np.zeros(pad_bits, dtype=np.uint8)])

    packed = np.packbits(bits)
    return packed.tobytes(), pad_bits


def unpack_bits(data: bytes, bit_depth: int, n_elements: int, pad_bits: int) -> np.ndarray:
    """Inverse of :func:`pack_bits`. Returns a ``uint32`` array of length ``n_elements``."""
    if n_elements == 0:
        return np.zeros(0, dtype=np.uint32)

    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    total_bits = n_elements * bit_depth
    expected_len = total_bits + pad_bits
    if bits.size != expected_len:
        raise QuantizationError(
            f"packed payload has {bits.size} bits, expected {expected_len} "
            f"(n_elements={n_elements}, bit_depth={bit_depth}, pad_bits={pad_bits})"
        )
    bits = bits[:total_bits].reshape(n_elements, bit_depth)
    weights = (1 << np.arange(bit_depth - 1, -1, -1, dtype=np.uint32))
    codes = (bits.astype(np.uint32) * weights).sum(axis=1)
    return codes


def _per_tensor_scale_zp(tensor: torch.Tensor, qmax: int) -> "tuple[List[float], List[float]]":
    tmin = float(tensor.min().item()) if tensor.numel() else 0.0
    tmax = float(tensor.max().item()) if tensor.numel() else 0.0
    scale = (tmax - tmin) / qmax if tmax > tmin else 1.0
    return [scale], [tmin]


def _per_channel_scale_zp(
    tensor: torch.Tensor, channel_dim: int, qmax: int
) -> "tuple[List[float], List[float]]":
    other_dims = [d for d in range(tensor.dim()) if d != channel_dim]
    if other_dims:
        tmin = tensor.amin(dim=other_dims)
        tmax = tensor.amax(dim=other_dims)
    else:
        tmin = tensor.clone()
        tmax = tensor.clone()
    span = (tmax - tmin)
    scale = torch.where(span > 0, span / qmax, torch.ones_like(span))
    return scale.reshape(-1).tolist(), tmin.reshape(-1).tolist()


def quantize_tensor(
    tensor: torch.Tensor,
    bit_depth: int,
    granularity: str = "per_tensor",
    channel_dim: int = 1,
) -> QuantizedTensor:
    """Quantize a float tensor to ``bit_depth``-bit unsigned codes and bit-pack them."""
    _validate_bit_depth(bit_depth)
    if granularity not in ("per_tensor", "per_channel"):
        raise QuantizationError(f"unsupported granularity={granularity!r}")

    qmax = (1 << bit_depth) - 1
    tensor = tensor.detach()

    if granularity == "per_tensor":
        scale, zero_point = _per_tensor_scale_zp(tensor, qmax)
        scale_t = torch.tensor(scale[0], dtype=tensor.dtype, device=tensor.device)
        zp_t = torch.tensor(zero_point[0], dtype=tensor.dtype, device=tensor.device)
        codes = torch.round((tensor - zp_t) / scale_t)
    else:
        if channel_dim < 0 or channel_dim >= tensor.dim():
            raise QuantizationError(f"channel_dim={channel_dim} out of range for shape {tuple(tensor.shape)}")
        scale, zero_point = _per_channel_scale_zp(tensor, channel_dim, qmax)
        view_shape = [1] * tensor.dim()
        view_shape[channel_dim] = tensor.shape[channel_dim]
        scale_t = torch.tensor(scale, dtype=tensor.dtype, device=tensor.device).reshape(view_shape)
        zp_t = torch.tensor(zero_point, dtype=tensor.dtype, device=tensor.device).reshape(view_shape)
        codes = torch.round((tensor - zp_t) / scale_t)

    codes = codes.clamp(0, qmax).to(torch.int64).cpu().numpy().reshape(-1)
    packed, pad_bits = pack_bits(codes, bit_depth)

    return QuantizedTensor(
        packed=packed,
        shape=list(tensor.shape),
        bit_depth=bit_depth,
        granularity=granularity,
        channel_dim=channel_dim,
        scale=scale,
        zero_point=zero_point,
        pad_bits=pad_bits,
        n_elements=int(tensor.numel()),
    )


def dequantize_tensor(
    q: QuantizedTensor,
    dtype: torch.dtype = torch.float32,
    device: "torch.device | str" = "cpu",
) -> torch.Tensor:
    """Invert :func:`quantize_tensor`. Reconstructs a fresh tensor from packed bytes only."""
    _validate_bit_depth(q.bit_depth)
    codes = unpack_bits(q.packed, q.bit_depth, q.n_elements, q.pad_bits)
    codes_t = torch.from_numpy(codes.astype(np.float32)).to(dtype=dtype, device=device)

    if q.granularity == "per_tensor":
        scale_t = torch.tensor(q.scale[0], dtype=dtype, device=device)
        zp_t = torch.tensor(q.zero_point[0], dtype=dtype, device=device)
        flat = codes_t * scale_t + zp_t
        return flat.reshape(q.shape)

    view_shape = [1] * len(q.shape)
    view_shape[q.channel_dim] = q.shape[q.channel_dim]
    scale_t = torch.tensor(q.scale, dtype=dtype, device=device).reshape(view_shape)
    zp_t = torch.tensor(q.zero_point, dtype=dtype, device=device).reshape(view_shape)
    codes_full = codes_t.reshape(q.shape)
    return codes_full * scale_t + zp_t

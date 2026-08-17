"""channels/digital_packet.py – Real binary-packet digital transport channel.

Alternative to :class:`~sgdjscc_lab.channels.awgn.AWGNChannel` selected via
``channel: digital_packet`` (same extension point Rayleigh/fast-fading/
packet-drop use, see ``channels/__init__.py::build_channel``). Instead of
adding analog Gaussian noise to the float latent, it:

    1. quantizes the latent to N-bit codes (8/6/4),
    2. bit-packs + serializes them into a deterministic
       :mod:`~sgdjscc_lab.transmission.wire_packet` binary packet,
    3. (the packet bytes are the actual "on the wire" artifact — optionally
       written to disk by the caller for byte accounting),
    4. parses the packet back and dequantizes to a *fresh* tensor.

Step 4 never returns (a view of) the tensor passed into ``transmit()`` — the
returned tensor is reconstructed purely from the serialized bytes, so a
receiver holding only the packet cannot obtain the original full-precision
latent object. This is intentionally **not** combined with AWGN noise: this
channel models "reliable digital delivery of a lossily-quantized source",
not the analog SGD-JSCC channel — combining both would double-apply
degradation without a defined model for it, so no noise is injected here
(mirrors ``PacketDropChannel``'s separate erasure-vs-noise-floor split, but
this channel intentionally omits any noise floor).
"""

from __future__ import annotations

from typing import Optional

import torch

from sgdjscc_lab.channels.measurement import ChannelTape, MeasurementBundle
from sgdjscc_lab.transmission.byte_accounting import packet_byte_breakdown
from sgdjscc_lab.transmission.wire_packet import (
    WirePacket,
    decode_latent_packet,
    encode_latent_packet,
    parse,
    serialize,
)


class DigitalPacketChannel(ChannelTape):
    """Quantize → binary-pack → (transport) → parse → dequantize channel.

    Parameters
    ----------
    bit_depth:
        8, 6, or 4.
    granularity:
        "per_tensor" or "per_channel".
    channel_dim:
        Tensor axis used for per-channel scale/zero-point (default 1, the
        JSCC latent's channel axis in ``[B, C, H, W]``).
    compress_metadata:
        Whether the packet's JSON metadata block is zlib-deflated.
    """

    def __init__(
        self,
        bit_depth: int = 8,
        granularity: str = "per_tensor",
        channel_dim: int = 1,
        compress_metadata: bool = True,
    ) -> None:
        self.bit_depth = int(bit_depth)
        self.granularity = str(granularity)
        self.channel_dim = int(channel_dim)
        self.compress_metadata = bool(compress_metadata)
        self.keyframe_index: Optional[int] = None
        self.last_packet_bytes: Optional[bytes] = None
        self.last_breakdown = None
        self._init_tape()

    def transmit(self, latent: torch.Tensor, snr_db: float) -> torch.Tensor:
        return self._taped_transmit(latent, snr_db)

    def observe(self, latent: torch.Tensor, snr_db: float) -> MeasurementBundle:
        bsz = latent.shape[0]
        # Quantize/serialize/parse/dequantize per-sample so scale/zero_point stay
        # meaningful per image (matches how the JSCC forward pass calls transmit()
        # once per patch, batch size 1 in practice, but kept general).
        recon_samples = []
        total_bytes = 0
        last_packet: Optional[WirePacket] = None
        last_data: Optional[bytes] = None
        for i in range(bsz):
            sample = latent[i:i + 1].detach().cpu()
            data = encode_latent_packet(
                sample,
                bit_depth=self.bit_depth,
                granularity=self.granularity,
                channel_dim=self.channel_dim,
                keyframe_index=self.keyframe_index,
                metadata={"snr_db": float(snr_db)},
                compress_metadata=self.compress_metadata,
            )
            total_bytes += len(data)
            last_data = data
            last_packet = parse(data)
            recon = decode_latent_packet(data, dtype=latent.dtype, device=latent.device)
            recon_samples.append(recon)

        received = torch.cat(recon_samples, dim=0)

        self.last_packet_bytes = last_data
        self.last_breakdown = (
            packet_byte_breakdown(last_packet, compress_metadata=self.compress_metadata)
            if last_packet is not None
            else None
        )

        return MeasurementBundle(
            received=received,
            equalized=received,
            channel_gain=None,
            noise_var=torch.zeros(bsz, 1, 1, 1, device=latent.device),
            mask=torch.ones(bsz, 1, 1, 1, device=latent.device),
            snr_db_true=None,  # digital delivery: SNR is not the degradation source
            reliability=torch.ones(bsz, 1, 1, 1, device=latent.device),
            meta={
                "channel": "digital_packet",
                "bit_depth": self.bit_depth,
                "granularity": self.granularity,
                "channel_dim": self.channel_dim,
                "total_packet_bytes": total_bytes,
                "keyframe_index": self.keyframe_index,
            },
        )

"""channels/packet_drop.py – Erasure / packet-drop channel (Phase 5-A).

Models semantic-packet loss at the symbol-block level: the flattened latent is
split into ``packet_length``-element packets and each packet is erased (zeroed)
independently with probability ``drop_prob``.  An AWGN floor is applied to the
surviving symbols at *snr_db*.

This is the channel used for the Phase 4 "optional packet-drop simulation for
semantic delta experiments" and gives the channel-condition encoder an explicit
``mask`` (1 = delivered, 0 = erased) as its reliability signal.

API-compatible with ``AWGNChannel`` and exposes ``observe()`` →
:class:`MeasurementBundle`.
"""

from __future__ import annotations

import torch

from sgdjscc_lab.channels.measurement import ChannelTape, MeasurementBundle, awgn_noise_like


class PacketDropChannel(ChannelTape):
    """Block-erasure channel with an AWGN floor.

    Parameters
    ----------
    drop_prob:
        Per-packet erasure probability in [0, 1].
    packet_length:
        Number of consecutive latent elements per packet (>=1).
    """

    def __init__(self, drop_prob: float = 0.1, packet_length: int = 256) -> None:
        self.drop_prob = float(drop_prob)
        self.packet_length = max(int(packet_length), 1)
        self._init_tape()

    def transmit(self, latent: torch.Tensor, snr_db: float) -> torch.Tensor:
        return self._taped_transmit(latent, snr_db)

    def observe(self, latent: torch.Tensor, snr_db: float) -> MeasurementBundle:
        bsz, c, h, w = latent.shape
        flat = latent.reshape(bsz, -1)
        n = flat.shape[1]
        n_packets = (n + self.packet_length - 1) // self.packet_length

        keep_pkt = (torch.rand(bsz, n_packets, device=latent.device) >= self.drop_prob).float()
        keep = keep_pkt.repeat_interleave(self.packet_length, dim=1)[:, :n]   # [B, n]
        mask_map = keep.reshape(bsz, c, h, w)

        noise, noise_var = awgn_noise_like(latent, snr_db)
        received = (latent + noise) * mask_map

        reliability = mask_map.mean(dim=1, keepdim=True)        # [B,1,H,W]

        return MeasurementBundle(
            received=received,
            equalized=received,                                 # erasure: no inverse
            channel_gain=None,
            noise_var=noise_var,
            mask=reliability,
            snr_db_true=float(snr_db),
            reliability=reliability,
            meta={"channel": "packet_drop", "drop_prob": self.drop_prob,
                  "packet_length": self.packet_length},
        )

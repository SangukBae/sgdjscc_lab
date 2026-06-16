"""sgdjscc_lab.channels – Channel simulation modules.

Phase 1-4: AWGN.  Phase 5-A adds Rayleigh / fast fading / packet-drop channels
that share the AWGN ``transmit(latent, snr_db) -> Tensor`` contract and add a
richer ``observe() -> MeasurementBundle`` (DiffCom-style receiver evidence).
"""

from .awgn import AWGNChannel
from .rayleigh import RayleighChannel
from .fast_fading import FastFadingChannel
from .packet_drop import PacketDropChannel
from .measurement import MeasurementBundle, awgn_noise_like

__all__ = [
    "AWGNChannel",
    "RayleighChannel",
    "FastFadingChannel",
    "PacketDropChannel",
    "MeasurementBundle",
    "awgn_noise_like",
    "build_channel",
]


def build_channel(cfg=None, **kwargs):
    """Build a channel from a config block (or kwargs).

    Recognised keys (all optional):
        ``channel`` / ``channel_type`` : "awgn" | "rayleigh" | "fast_fading" | "packet_drop"
        ``csi``                        : "perfect" | "imperfect" | "none"
        ``csi_error_std``              : float (imperfect CSI)
        ``block_length``               : int (fast fading)
        ``drop_prob`` / ``packet_length`` : packet drop params

    A ``cfg`` mapping (e.g. OmegaConf) is read via ``.get``; explicit *kwargs*
    override it.  Unknown channel names fall back to AWGN.
    """
    def _get(key, default=None):
        if key in kwargs:
            return kwargs[key]
        if cfg is not None:
            try:
                return cfg.get(key, default)
            except AttributeError:
                return getattr(cfg, key, default)
        return default

    name = str(_get("channel", _get("channel_type", "awgn"))).lower()
    csi = str(_get("csi", "perfect"))
    csi_error_std = float(_get("csi_error_std", 0.1))

    if name in ("awgn", "none", ""):
        return AWGNChannel()
    if name == "rayleigh":
        return RayleighChannel(csi=csi, csi_error_std=csi_error_std)
    if name in ("fast_fading", "fast", "block_fading"):
        return FastFadingChannel(
            block_length=int(_get("block_length", 64)),
            csi=csi, csi_error_std=csi_error_std,
        )
    if name in ("packet_drop", "erasure", "drop"):
        return PacketDropChannel(
            drop_prob=float(_get("drop_prob", 0.1)),
            packet_length=int(_get("packet_length", 256)),
        )
    return AWGNChannel()

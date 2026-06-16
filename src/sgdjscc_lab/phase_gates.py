"""phase_gates.py – Phase master switch helpers.

Each ``phaseN_enabled`` function reads the corresponding ``use_phaseN`` flag
from cfg.  ``effective_flag`` combines the master-switch check with the
per-feature flag so call sites remain one-liners.

Rule: if the master switch is off, the per-feature flag is ignored and the
function returns False — even when the per-feature flag is explicitly True.
"""
from __future__ import annotations

from omegaconf import OmegaConf


def phase4_enabled(cfg) -> bool:
    """Return True only when ``use_phase4: true`` is set in cfg."""
    return bool(OmegaConf.select(cfg, "use_phase4", default=False))


def phase5_enabled(cfg) -> bool:
    """Return True only when ``use_phase5: true`` is set in cfg."""
    return bool(OmegaConf.select(cfg, "use_phase5", default=False))


def effective_flag(cfg, flag_key: str, phase: int) -> bool:
    """Return the per-feature flag value, gated by the phase master switch.

    Returns False when the master switch for *phase* (4 or 5) is off,
    regardless of the per-feature flag's own value in cfg.

    Parameters
    ----------
    cfg:
        OmegaConf DictConfig (or any object that OmegaConf.select can handle).
    flag_key:
        Dotted key path understood by OmegaConf.select (e.g. ``"use_packet_eval"``).
    phase:
        Which phase master switch to check (4 or 5).
    """
    if phase == 4 and not phase4_enabled(cfg):
        return False
    if phase == 5 and not phase5_enabled(cfg):
        return False
    return bool(OmegaConf.select(cfg, flag_key, default=False))

"""acceleration/ddim_sampler.py – Sampler / step-budget selection (Phase 5-B).

Off-by-default helpers that let experiments switch the denoising **step budget**
(50 / 20 / 10 / 5 …) and choose a sampler type without modifying the SGD-JSCC
sampler.  The SGD-JSCC ``pipe.generate`` already accepts ``diffusion_step``, so a
"DDIM step ablation" is realised by overriding ``cfg.diffusion_step`` — this
module centralises that choice and the dynamic SNR→budget routing.

It also ports the Karras noise schedule from the LDM-enabled SemCom code
(``paper/LDM-enabled-SemCom-system/.../t_calculate.py``) for use by the
consistency decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

VALID_SAMPLERS = ("baseline", "ddim", "fewstep", "distilled_placeholder")


@dataclass
class SamplerSpec:
    """Resolved sampling configuration."""

    sampler_type: str = "baseline"
    steps: int = 50

    def as_dict(self):
        return {"sampler_type": self.sampler_type, "steps": self.steps}


def build_sampler_cfg(base_cfg: DictConfig) -> Tuple[DictConfig, SamplerSpec]:
    """Return ``(cfg, spec)`` with ``cfg.diffusion_step`` set per the sampler.

    Reads the ``acceleration`` config block (or top-level keys):
        ``sampler``/``sampler_type`` : one of VALID_SAMPLERS (default "baseline")
        ``sampler_steps``            : explicit step budget (overrides default)

    When acceleration is not configured, the base cfg is returned unchanged
    (off-by-default) with a baseline spec.
    """
    acc = OmegaConf.select(base_cfg, "acceleration", default=None)
    acc = OmegaConf.to_container(acc, resolve=True) if acc is not None else {}

    sampler = str(acc.get("sampler", acc.get("sampler_type",
                  base_cfg.get("sampler_type", "baseline")))).lower()
    if sampler not in VALID_SAMPLERS:
        sampler = "baseline"

    default_steps = int(base_cfg.get("diffusion_step", 50))
    steps = int(acc.get("sampler_steps", default_steps))
    steps = max(steps, 1)

    spec = SamplerSpec(sampler_type=sampler, steps=steps)

    if sampler == "baseline" and "sampler_steps" not in acc:
        return base_cfg, spec   # untouched legacy path

    out = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    out.diffusion_step = steps
    out.sampler_type = sampler
    return out, spec


def dynamic_step_budget(
    snr_db: Optional[float],
    confidence: Optional[float] = None,
    blind: bool = False,
    high: int = 5,
    mid: int = 15,
    low: int = 40,
) -> int:
    """Pick a denoising step budget from channel state (Phase 5-B routing).

    Policy:
        high SNR (>= 12 dB) AND high confidence (>= 0.6) → ``high`` (1–5 steps)
        mid  SNR (>= 4 dB)                               → ``mid``  (10–20 steps)
        low SNR / blind / low confidence                → ``low``  (30–50 steps)
    """
    if blind or snr_db is None:
        return low
    conf = 1.0 if confidence is None else float(confidence)
    if snr_db >= 12.0 and conf >= 0.6:
        return high
    if snr_db >= 4.0:
        return mid
    return low


def karras_schedule(
    num_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 2.0,
    rho: float = 7.0,
    device=None,
) -> torch.Tensor:
    """Karras et al. noise schedule (ported from the LDM SemCom code).

    Returns ``num_steps`` decreasing sigmas from ``sigma_max`` to ``sigma_min``.
    """
    ramp = torch.linspace(0, 1, num_steps, device=device)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return sigmas

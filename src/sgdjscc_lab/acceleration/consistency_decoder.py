"""acceleration/consistency_decoder.py – Few-step / distilled decoder interface (Phase 5-B).

Provides a clean, model-agnostic few-step decoding interface inspired by
``ConsistencySamplingAndEditing`` in the LDM-enabled SemCom code
(``paper/LDM-enabled-SemCom-system/train_DIV2K/consistency_models.py``).

Three experimental modes share one ``decode`` entry point:

- ``"baseline"``            : single full denoise call (delegates entirely to the
                              injected ``denoise_fn``; the caller supplies the
                              normal multi-step pipeline).
- ``"fewstep"``             : deterministic consistency-style few-step sampling on
                              a Karras schedule (1 / 2 / 5 steps).  Works with any
                              ``denoise_fn(x, sigma) -> x0`` — testable with a mock.
- ``"distilled_placeholder"``: hook for a distilled student decoder.  If no
                              distilled model is registered it falls back to
                              ``fewstep`` and logs a warning (so the API and
                              evaluation path are complete without trained weights).

``denoise_fn(x, sigma) -> x0_hat`` is the single-step denoiser the caller injects;
the SGD-JSCC integration would wrap its denoiser here, but the math is exercised
in tests with a trivial mock.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

from sgdjscc_lab.acceleration.ddim_sampler import karras_schedule

logger = logging.getLogger(__name__)


def skip_scaling(sigma, sigma_data: float = 0.5, sigma_min: float = 0.002):
    return sigma_data ** 2 / ((sigma - sigma_min) ** 2 + sigma_data ** 2)


def output_scaling(sigma, sigma_data: float = 0.5, sigma_min: float = 0.002):
    return (sigma_data * (sigma - sigma_min)) / (sigma_data ** 2 + sigma ** 2) ** 0.5


class ConsistencyDecoder:
    """Few-step decoder with baseline / fewstep / distilled_placeholder modes.

    Parameters
    ----------
    mode:
        One of ``"baseline"`` | ``"fewstep"`` | ``"distilled_placeholder"``.
    steps:
        Number of consistency steps for the fewstep / distilled modes.
    sigma_min, sigma_max, rho:
        Karras schedule parameters.
    distilled_model:
        Optional callable ``(x, sigma) -> x0`` standing in for a distilled student.
    """

    def __init__(
        self,
        mode: str = "fewstep",
        steps: int = 2,
        sigma_min: float = 0.002,
        sigma_max: float = 2.0,
        rho: float = 7.0,
        distilled_model: Optional[Callable] = None,
    ) -> None:
        self.mode = mode
        self.steps = max(int(steps), 1)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.distilled_model = distilled_model

    def decode(
        self,
        denoise_fn: Callable,
        init: Optional[torch.Tensor] = None,
        shape=None,
        device=None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Run the configured decoder.

        Parameters
        ----------
        denoise_fn:
            ``(x, sigma) -> x0_hat`` single-step denoiser.
        init:
            Optional initial latent.  If None, sampled as ``N(0, sigma_max²)`` of
            ``shape`` (which must then be provided).
        """
        if self.mode == "baseline":
            # One delegated denoise at the max sigma (caller owns the full loop).
            x = self._init(init, shape, device, generator)
            return denoise_fn(x, self.sigma_max)

        if self.mode == "distilled_placeholder":
            if self.distilled_model is not None:
                x = self._init(init, shape, device, generator)
                return self.distilled_model(x, self.sigma_max)
            logger.warning(
                "distilled_placeholder: no distilled model registered; "
                "falling back to fewstep consistency sampling."
            )
        # fewstep (and distilled fallback)
        return self._fewstep(denoise_fn, init, shape, device, generator)

    # ── internals ────────────────────────────────────────────────────────────
    def _init(self, init, shape, device, generator):
        if init is not None:
            return init
        if shape is None:
            raise ValueError("Either `init` or `shape` must be provided.")
        return torch.randn(*shape, device=device, generator=generator) * self.sigma_max

    def _fewstep(self, denoise_fn, init, shape, device, generator):
        dev = device or (init.device if init is not None else None)
        sigmas = karras_schedule(self.steps + 1, self.sigma_min, self.sigma_max,
                                 self.rho, device=dev)
        x = self._init(init, shape, dev, generator)
        # Start at the largest sigma, refine toward sigma_min (consistency style).
        x0 = denoise_fn(x, float(sigmas[0]))
        for i in range(1, self.steps):
            sigma = float(sigmas[i])
            noise = torch.randn(x0.shape, device=x0.device, generator=generator)
            x = x0 + noise * sigma
            x0 = denoise_fn(x, sigma)
        return x0

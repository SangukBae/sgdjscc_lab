"""training/noise_schedule.py – Continuous sigmoid noise schedule for the DM stages.

Implements the paper's continuous-timestep forward diffusion (Sec. V-B,
Algorithm 1) used to train the text-guided DM (stage 2) and the ControlNet
branch (stage 3):

    t ~ Uniform(0, 1)
    β̄_t = S(t)                                          (sigmoid schedule, eq. 13)
    n  ~ N(0, I)
    f_t = sqrt(1 - β̄_t) · f_0 + sqrt(β̄_t) · n           (forward trajectory, eq. 8)

The sigmoid schedule ``S(t)`` mirrors ``DiffusionGenerator.sigmoid_schedule`` in
``SGDJSCC/models/test_advanced_network/diffusion_element_wise.py`` so the noise
level fed to the denoiser at training time matches the one used at inference
(algorithm-preservation invariant).  The denoiser is conditioned on the
**noise level** ``sqrt(β̄_t)`` exactly as ``pred_image`` does at inference.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


class SigmoidNoiseScheduler:
    """Continuous variance-preserving sigmoid noise schedule.

    Parameters
    ----------
    start, end, tau:
        Shape hyperparameters of the sigmoid schedule (paper defaults
        ``start=0, end=3, tau=0.7``; the inference code uses the same).
    clip_min:
        Numerical clamp keeping ``β̄_t`` inside ``(clip_min, 1-clip_min)``.
    """

    def __init__(
        self,
        start: float = 0.0,
        end: float = 3.0,
        tau: float = 0.7,
        clip_min: float = 1e-6,
    ) -> None:
        self.start = float(start)
        self.end = float(end)
        self.tau = float(tau)
        self.clip_min = float(clip_min)

    # ── schedule ──────────────────────────────────────────────────────────────
    def beta_bar(self, t: torch.Tensor) -> torch.Tensor:
        """β̄_t = S(t) for ``t`` in ``[0, 1]`` (variance of the noise at step t)."""
        sig = torch.sigmoid
        v_start = sig(torch.tensor(self.start / self.tau))
        v_end = sig(torch.tensor(self.end / self.tau))
        out = sig((t * (self.end - self.start) + self.start) / self.tau)
        out = (out - v_start) / (v_end - v_start)
        return out.clamp(self.clip_min, 1.0 - self.clip_min)

    def inverse_beta_bar(self, value: torch.Tensor) -> torch.Tensor:
        """``S^{-1}(value)``: the timestep ``t`` whose noise variance is ``value``.

        Analytic inverse of :meth:`beta_bar` (the sigmoid schedule is monotonic in
        ``t``). Used by the fast-fading water-filling denoiser (paper Algorithm 4)
        to initialise ``t = S^{-1}(max_i d_i)``.
        """
        if not torch.is_tensor(value):
            value = torch.tensor(float(value))
        v = value.clamp(self.clip_min, 1.0 - self.clip_min)
        v_start = torch.sigmoid(torch.tensor(self.start / self.tau))
        v_end = torch.sigmoid(torch.tensor(self.end / self.tau))
        # beta_bar(t) = (sigmoid(arg) - v_start)/(v_end - v_start), arg=(t(e-s)+s)/tau
        sig_val = (v * (v_end - v_start) + v_start).clamp(self.clip_min, 1.0 - self.clip_min)
        arg = torch.log(sig_val / (1.0 - sig_val))            # logit(sig_val)
        t = (arg * self.tau - self.start) / (self.end - self.start)
        return t.clamp(0.0, 1.0)

    def sample_t(self, batch_size: int, device, generator=None) -> torch.Tensor:
        """Sample ``t ~ Uniform(0, 1)`` with shape ``[batch_size]``."""
        return torch.rand(batch_size, device=device, generator=generator)

    # ── forward trajectory ────────────────────────────────────────────────────
    def add_noise(
        self,
        f0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        generator=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the forward trajectory to clean latent ``f0`` ``[B,C,H,W]``.

        Returns
        -------
        ft:
            Noised latent ``sqrt(1-β̄_t)·f0 + sqrt(β̄_t)·n``.
        noise_level:
            ``sqrt(β̄_t)`` shaped ``[B, 1]`` — the conditioning value the denoiser
            expects (matches inference ``pred_image``).
        noise:
            The Gaussian noise instance ``n`` used.
        t:
            The sampled (or supplied) timesteps ``[B]``.
        """
        b = f0.shape[0]
        if t is None:
            t = self.sample_t(b, f0.device, generator=generator)
        beta = self.beta_bar(t)                       # [B]
        beta_v = beta.reshape(b, *([1] * (f0.dim() - 1)))
        if noise is None:
            noise = torch.randn(f0.shape, device=f0.device, generator=generator, dtype=f0.dtype)
        ft = torch.sqrt(1.0 - beta_v) * f0 + torch.sqrt(beta_v) * noise
        noise_level = torch.sqrt(beta).reshape(b, 1)  # [B, 1] for the DiT model
        return ft, noise_level, noise, t

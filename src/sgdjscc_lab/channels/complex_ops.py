"""channels/complex_ops.py – Complex channel + joint phase/SNR CSI (paper Alg. 3).

The SGD-JSCC paper transmits **complex** symbols ``z ∈ ℂ^M`` over a channel
``h = |h| e^{jφ}`` with complex AWGN ``n ∼ CN(0, 2σ²)`` and equalizes the output
in **two steps** (paper Sec. III):

    1) phase removal     ŷ  = e^{-jφ̂} · y
    2) magnitude norm.   y_eq = ŷ / √(|h|² + σ²)

with the real↔complex maps ``C: ℝ^{2L}→ℂ^L`` and ``R: ℂ^L→ℝ^{2L}`` (the first
half of a real vector is the real part, the second half the imaginary part).
This module implements that complex path on a **2-channel (real, imag)**
representation of a latent, plus an **alternating** phase/SNR estimation loop.

Fidelity (see docs/paper_gap_closure.md)
----------------------------------------
* **paper-faithful** : the C/R maps, complex channel, and two-step (phase then
  magnitude) equalization implemented here.
* **unsupported (end-to-end)** : the *public* SGDJSCC JSCC encoder/decoder used
  by ``sgdjscc_lab`` emits a **real** latent and its channels model a **real**
  gain (``channels/measurement.py::mmse_equalize`` documents this). Routing this
  complex path through the *frozen public* JSCC forward would require RE-TRAINING
  the JSCC encoder to emit complex symbols (non-public weights/data). So this is a
  correct, smoke-tested **complex-channel layer + estimators**, NOT a drop-in for
  the real-gain inference forward. ``paper_mode`` therefore treats end-to-end
  complex transport as *unsupported*; this module is the building block + a
  forward-shape-validated estimation loop for a future complex-JSCC path.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


# ── real ↔ complex maps (paper C / R) ─────────────────────────────────────────

def real_to_complex(v_r: torch.Tensor) -> torch.Tensor:
    """``C: ℝ^{2L} → ℂ^L`` along the last dim (first L = real, last L = imag)."""
    if v_r.shape[-1] % 2 != 0:
        raise ValueError(f"real_to_complex expects an even last dim, got {v_r.shape[-1]}")
    L = v_r.shape[-1] // 2
    return torch.complex(v_r[..., :L].contiguous(), v_r[..., L:].contiguous())


def complex_to_real(v: torch.Tensor) -> torch.Tensor:
    """``R: ℂ^L → ℝ^{2L}`` (concatenate real then imaginary along the last dim)."""
    return torch.cat([v.real, v.imag], dim=-1)


def two_channel_to_complex(x: torch.Tensor) -> torch.Tensor:
    """``[B, 2, H, W]`` (real, imag channels) → complex ``[B, 1, H, W]``."""
    if x.shape[1] % 2 != 0:
        raise ValueError(f"two_channel_to_complex expects even channels, got {x.shape[1]}")
    half = x.shape[1] // 2
    return torch.complex(x[:, :half], x[:, half:])


def complex_to_two_channel(z: torch.Tensor) -> torch.Tensor:
    """complex ``[B, C, H, W]`` → real ``[B, 2C, H, W]`` (real channels then imag)."""
    return torch.cat([z.real, z.imag], dim=1)


# ── complex channel + two-step equalization ───────────────────────────────────

def sigma_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """Noise std σ from signal level ``α = |h|²/(|h|²+σ²)`` assuming ``|h|=1``.

    With unit gain, ``α = 1/(1+σ²)`` ⇒ ``σ = √((1-α)/α)`` (clamped).
    """
    a = alpha.clamp(1e-6, 1.0)
    return torch.sqrt((1.0 - a) / a)


def apply_complex_channel(z: torch.Tensor, h: torch.Tensor,
                          sigma: float) -> torch.Tensor:
    """Complex channel ``y = h·z + n``, ``n ∼ CN(0, 2σ²)`` (real/imag each ``N(0,σ²)``)."""
    n = torch.randn_like(z.real) + 1j * torch.randn_like(z.imag)
    return h * z + sigma * n


def two_step_equalize(y: torch.Tensor, h: torch.Tensor,
                      sigma: float) -> torch.Tensor:
    """Paper two-step equalization: phase removal then magnitude normalization.

    ``y_eq = e^{-jφ̂} · y / √(|h|² + σ²)`` with ``φ̂ = ∠h``, ``|h| = |h|``.
    """
    phi = torch.angle(h)
    mag = torch.abs(h)
    return torch.exp(-1j * phi) * y / torch.sqrt(mag ** 2 + sigma ** 2)


# ── alternating phase/SNR estimation (paper Algorithm 3) ──────────────────────

def alternating_phase_snr_equalize(
    y_2ch: torch.Tensor,
    snr_estimator,
    phase_estimator=None,
    max_iter: int = 3,
    snr_is_amplitude: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alternating phase/SNR estimation + phase removal on a complex latent.

    Unlike ``models/csi_estimation.joint_csi_estimate`` (which is a *real-gain*
    scaffold whose phase-removal is a no-op), this operates on a genuinely
    **complex** latent (2-channel real/imag), so the phase-removal step is a real
    rotation. Per iteration:

      1) estimate signal level ``α`` from the current real representation;
      2) if a phase estimator is given, estimate ``φ̂`` and apply ``e^{-jφ̂}``.

    Parameters
    ----------
    y_2ch:
        ``[B, 2, H, W]`` channel output (real, imag channels).
    snr_estimator / phase_estimator:
        Networks from ``models/csi_estimation`` (or any callable with the same
        interface). ``phase_estimator=None`` → SNR-only (no phase removal).

    Returns ``(y_eq_2ch, alpha, phi)`` — equalized 2-channel latent, ``α [B,1]``,
    ``φ [B,1]`` (radians; zeros when no phase estimator).
    """
    if snr_is_amplitude is None:
        snr_is_amplitude = bool(getattr(snr_estimator, "output_is_amplitude", True))

    def _alpha(feat: torch.Tensor) -> torch.Tensor:
        out = snr_estimator(feat).clamp(0.0, 1.0)
        return out ** 2 if snr_is_amplitude else out

    z = two_channel_to_complex(y_2ch)                       # [B, C, H, W] complex
    b = y_2ch.shape[0]
    phi = torch.zeros(b, 1, device=y_2ch.device, dtype=y_2ch.dtype)
    alpha = _alpha(y_2ch)
    for _ in range(max(1, int(max_iter))):
        feat = complex_to_two_channel(z)
        alpha = _alpha(feat)
        if phase_estimator is not None:
            dphi = phase_estimator(feat, alpha) * math.pi   # φ/π → radians, [B,1]
            phi = phi + dphi
            rot = torch.exp(-1j * dphi).view(b, 1, 1, 1)
            z = z * rot                                     # phase removal
    return complex_to_two_channel(z), alpha, phi

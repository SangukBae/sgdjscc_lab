"""models/csi_estimation.py – Pilot-free blind CSI estimation (paper Sec. IV-C).

Paper definitions (primary reference)
-------------------------------------
The receiver estimates the channel directly from the normalized noisy feature
``f̄ = √α·R(e^{jφ}C(f)) + √(1-α)·n``, where ``α = |h|²/(|h|²+σ²)`` is the signal
level (so ``1-α`` is the noise variance):

* **SNR estimation** (eq. 15): assuming the phase is removed,
  ``min_P E[‖ζ_P(√α f + √(1-α) n) − α‖²]`` — a network ``ζ_P`` regresses ``α``.
* **Phase estimation** (eq. after 15): assuming the SNR is known,
  ``min_Q E[‖ξ_Q(√α R(e^{jφ}C(f)) + √(1-α) n, α) − φ/π‖²]`` — "a similar network
  with attention feature (AF) modules after each residual block to project the
  SNR information".
* **Joint estimation** (Algorithm 3): init ``φ=0``; loop {phase removal; ``α=ζ_P``;
  ``φ=ξ_Q``} until convergence / max iters.

Public SGDJSCC code (secondary reference)
-----------------------------------------
``models/test_advanced_network/snr_prediction_net.py::Prediction_Model`` is the
**SNR estimator**: 16-ch latent → 4 residual ``BasicBlock`` (32/64/128/256) →
adaptive-avg-pool → ``Linear(256,1)`` → sigmoid → ``α∈[0,1]``. This is reused by
``sgdjscc_lab`` at inference for blind SNR (``jscc.snr_prediction_net``). There is
**no phase estimator and no joint loop** in the public code.

Fidelity labelling (no overstatement)
-------------------------------------
* :class:`SNREstimator` — **paper-like** (mirrors the public ``Prediction_Model``
  architecture / the paper's Table; weights here are untrained). NOTE the runtime
  contract: the inference path squares the predictor (``net² = signal level α``),
  so the ``csi_estimation`` stage trains it to output the **amplitude √α** by
  default (``target="amplitude"``); ``target="alpha"`` regresses ``α`` literally
  (paper eq. 15) and is √-wrapped at load time by :func:`load_snr_estimator_into`.
* :class:`PhaseEstimator`, :func:`joint_csi_estimate` — **paper-inspired scaffold**.
  The public code has no phase net, AND ``sgdjscc_lab``'s channels model a *real*
  gain ``g=|h|`` (no complex phase ``e^{jφ}`` in the real-valued latent path), so
  the phase estimate / phase removal are structural placeholders provided so a
  complex-valued extension can drop in. They are NOT a trained, paper-faithful CSI
  estimator.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Residual backbone (mirrors the public snr_prediction_net.BasicBlock)
# ─────────────────────────────────────────────────────────────────────────────

class _BasicBlock(nn.Module):
    """3×3-conv residual block with optional stride (public ``BasicBlock`` shape)."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class AFModule(nn.Module):
    """Attention-feature module (ADJSCC, Xu et al. 2021): project a scalar
    signal-level ``α ∈ [0,1]`` into per-channel scales applied to the feature map.

    Used by :class:`PhaseEstimator` to inject the (estimated) signal level ``α``
    after each residual block — the paper's "AF modules to project the SNR
    information". ``snr`` here is the signal **level** ``α`` (e.g. the converted
    output of :func:`joint_csi_estimate`), not the amplitude ``√α`` or dB.
    """

    def __init__(self, channels: int, snr_dim: int = 1) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels + snr_dim, channels), nn.ReLU(inplace=True),
            nn.Linear(channels, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
        ctx = x.mean(dim=(2, 3))                       # global pooled context [B,C]
        scale = self.fc(torch.cat([ctx, snr], dim=1))  # [B,C]
        return x * scale[:, :, None, None]


# ─────────────────────────────────────────────────────────────────────────────
# SNR estimator — paper-like (mirrors public Prediction_Model)
# ─────────────────────────────────────────────────────────────────────────────

class SNREstimator(nn.Module):
    """Regress a scalar in ``[0,1]`` from a noisy latent (blind CSI).

    Architecture mirrors the public ``Prediction_Model`` (4 residual blocks
    16→32→64→128→256, avg-pool, FC→1, sigmoid). **paper-like** (untrained here).

    Output semantics depend on the training target
    (``train.csi_estimation.target``, see ``training/stage_runners.py``):
    by **default** the ``csi_estimation`` stage trains it to output the signal
    **amplitude** ``√α`` (so the inference path's ``net²`` recovers the signal
    level ``α = |h|²/(|h|²+σ²)`` — the public runtime contract); with
    ``target="alpha"`` it regresses ``α`` directly (paper eq. 15).
    """

    def __init__(self, latent_ch: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _BasicBlock(latent_ch, 32, 1),
            _BasicBlock(32, 64, 1),
            _BasicBlock(64, 128, 1),
            _BasicBlock(128, 256, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, 1)
        # Output convention (set by the trainer/loader): True → the net outputs the
        # signal AMPLITUDE √α (stage default), False → the signal LEVEL α directly.
        # Consumers that need α (e.g. joint_csi_estimate) read this to convert.
        self.output_is_amplitude = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x).flatten(1)
        # Raw scalar ∈ [0,1], shape [B,1]. By default (output_is_amplitude=True)
        # this is the signal amplitude √α; with output_is_amplitude=False it is α.
        return torch.sigmoid(self.fc(h))


# ─────────────────────────────────────────────────────────────────────────────
# Phase estimator — paper-inspired SCAFFOLD (no phase net in the public code)
# ─────────────────────────────────────────────────────────────────────────────

class PhaseEstimator(nn.Module):
    """Regress the normalized phase ``φ/π ∈ [-1,1]`` from a noisy latent + the
    signal level ``α`` (``forward(x, snr=α)``; ``α ∈ [0,1]``, NOT amplitude/dB).

    Mirrors the SNR-estimator backbone with :class:`AFModule` α-projection after
    each residual block (the paper's design), output ``tanh`` → ``φ/π``.

    SCAFFOLD: there is no phase net in the public SGDJSCC code, and the real-gain
    channels carry no complex phase, so this is a structural placeholder — useful
    as an interface / a drop-in point for a complex-valued extension, NOT a trained
    paper-faithful estimator.
    """

    def __init__(self, latent_ch: int = 16) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            _BasicBlock(latent_ch, 32, 1),
            _BasicBlock(32, 64, 1),
            _BasicBlock(64, 128, 1),
            _BasicBlock(128, 256, 1),
        ])
        self.afs = nn.ModuleList([AFModule(c) for c in (32, 64, 128, 256)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
        for block, af in zip(self.blocks, self.afs):
            x = af(block(x), snr)                       # SNR projected after each block
        h = self.pool(x).flatten(1)
        return torch.tanh(self.fc(h))                  # φ/π ∈ [-1,1], shape [B,1]


# ─────────────────────────────────────────────────────────────────────────────
# Joint estimation (paper Algorithm 3) — scaffold for the real-gain path
# ─────────────────────────────────────────────────────────────────────────────

def joint_csi_estimate(
    noisy_feature: torch.Tensor,
    snr_estimator: SNREstimator,
    phase_estimator: Optional[PhaseEstimator] = None,
    max_iter: int = 3,
    complex_phase: bool = False,
    snr_is_amplitude: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Alternating SNR/phase estimation (paper Algorithm 3).

    Returns ``(alpha, phi_over_pi)`` (each ``[B,1]``), where ``alpha`` is always the
    **signal level** ``α = |h|²/(|h|²+σ²) ∈ [0,1]`` (NOT the amplitude). Because the
    stage-default SNR estimator outputs the amplitude ``√α``, this converts the
    estimator output to ``α`` before using it / feeding the phase estimator:

    * ``snr_is_amplitude=None`` (default) → read ``snr_estimator.output_is_amplitude``
      (True for stage-default / loaded estimators) and convert accordingly;
    * ``True`` → ``α = snr_out²``;  ``False`` → ``α = snr_out``.

    With ``phase_estimator=None`` (or ``complex_phase=False`` for the real-gain
    channels) only the SNR is estimated and ``phi_over_pi`` is zeros — the
    phase-removal step is a structural no-op because the real-valued latent path has
    no complex phase to recover (scaffold; see module docstring).
    """
    if snr_is_amplitude is None:
        snr_is_amplitude = bool(getattr(snr_estimator, "output_is_amplitude", True))

    def _alpha(feat: torch.Tensor) -> torch.Tensor:
        out = snr_estimator(feat).clamp(0.0, 1.0)
        return out ** 2 if snr_is_amplitude else out      # → signal LEVEL α

    b = noisy_feature.shape[0]
    phi = torch.zeros(b, 1, device=noisy_feature.device, dtype=noisy_feature.dtype)
    feat = noisy_feature
    alpha = _alpha(feat)
    if phase_estimator is None or not complex_phase:
        return alpha, phi
    for _ in range(max(1, int(max_iter))):
        # Phase removal would apply e^{-jφ} here for a COMPLEX latent; the real path
        # has no phase, so this is a structural no-op (scaffold).
        alpha = _alpha(feat)
        phi = phase_estimator(feat, alpha)                # phase net takes α (level)
    return alpha, phi


def build_csi_estimators(latent_ch: int = 16, with_phase: bool = False, device=None):
    """Construct ``(snr_estimator, phase_estimator|None)`` on *device*."""
    dev = device or torch.device("cpu")
    snr = SNREstimator(latent_ch).to(dev)
    phase = PhaseEstimator(latent_ch).to(dev) if with_phase else None
    logger.info("CSI estimators: SNREstimator(paper-like) + Phase=%s (scaffold).",
                "on" if with_phase else "off")
    return snr, phase


class _SqrtSNRAdapter(nn.Module):
    """Wrap an α-regressing estimator so its output is ``√α``.

    The inference path squares the predictor (``net² = signal level``), so an
    estimator trained with ``target="alpha"`` (outputs ``α``) must be wrapped to
    output ``√α`` for ``net² = α`` to hold.
    """

    def __init__(self, inner: "SNREstimator") -> None:
        super().__init__()
        self.inner = inner
        self.output_is_amplitude = True       # wrapper output is √α

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.inner(x).clamp(0.0, 1.0))


def _extract_snr_state_and_target(ckpt) -> Tuple[Optional[dict], str]:
    """Return ``(snr_estimator_state_dict, target)`` from a checkpoint dict.

    ``target`` defaults to ``"amplitude"`` (the stage default) when the checkpoint
    carries no metadata (legacy / external state dicts).
    """
    sd, target = None, "amplitude"
    if isinstance(ckpt, dict):
        rs = ckpt.get("runner_state")
        if isinstance(rs, dict):
            sd = (rs.get("modules") or {}).get("snr_estimator")
            target = str((rs.get("meta") or {}).get("csi_target", target)).lower()
        if sd is None:
            sd = (ckpt.get("model_state") or {}).get("snr_estimator")
        if sd is None and all(torch.is_tensor(v) for v in ckpt.values()):
            sd = ckpt
    return sd, target


def load_snr_estimator(checkpoint, latent_ch: int = 16, device=None) -> "SNREstimator":
    """Load the raw :class:`SNREstimator` from a ``csi_estimation`` checkpoint
    (no target adaptation — see :func:`load_snr_estimator_into`)."""
    from pathlib import Path
    p = Path(checkpoint)
    if not p.exists():
        raise FileNotFoundError(f"SNR-estimator checkpoint not found: {p}")
    sd, target = _extract_snr_state_and_target(torch.load(p, map_location="cpu"))
    if sd is None:
        raise KeyError(f"No SNR-estimator weights found in {p}.")
    est = SNREstimator(latent_ch).to(device or torch.device("cpu"))
    est.load_state_dict(sd, strict=True)
    est.output_is_amplitude = (target != "alpha")    # convention for downstream reuse
    est.eval()
    return est


def load_snr_estimator_into(jscc, checkpoint, latent_ch: int = 16, device=None) -> None:
    """Replace ``jscc.snr_prediction_net`` with a trained SNR estimator, adapting
    to the checkpoint's regression target so ``net² = α`` always holds.

    The inference blind step-matching path calls
    ``jscc.snr_prediction_net(latent).reshape([-1,1]) ** 2`` (inference_one.py:102),
    so the predictor must output the signal **amplitude** ``√α``.
    * ``target="amplitude"`` (default) → load the estimator directly (it already
      outputs ``√α``) — a drop-in for the public ``Prediction_Model``.
    * ``target="alpha"`` (paper eq. 15 literal, outputs ``α``) → wrap with
      :class:`_SqrtSNRAdapter` so the squaring runtime still recovers ``α`` (a
      warning is logged). This prevents silently loading an α-target net into the
      ``net²=α`` path (which would feed ``α²``).
    """
    from pathlib import Path
    p = Path(checkpoint)
    if not p.exists():
        raise FileNotFoundError(f"SNR-estimator checkpoint not found: {p}")
    sd, target = _extract_snr_state_and_target(torch.load(p, map_location="cpu"))
    if sd is None:
        raise KeyError(f"No SNR-estimator weights found in {p}.")
    dev = device or torch.device("cpu")
    est = SNREstimator(latent_ch).to(dev)
    est.load_state_dict(sd, strict=True)
    est.output_is_amplitude = (target != "alpha")
    est.eval()
    if target == "alpha":
        logger.warning("csi target='alpha' checkpoint → wrapping with √ so the "
                       "squaring inference path (net²) still yields α, not α².")
        net: nn.Module = _SqrtSNRAdapter(est).to(dev).eval()
    else:
        net = est
    jscc.snr_prediction_net = net
    logger.info("Loaded trained SNR estimator (target=%s) into "
                "jscc.snr_prediction_net from %s.", target, checkpoint)

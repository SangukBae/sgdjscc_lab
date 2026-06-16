"""channels/measurement.py – Receiver-evidence bundle (Phase 5-A).

Mirrors DiffCom's ``operator.observe()`` idea
(``paper/diffcom/guided_diffusion/measurement.py``): every channel experiment
should return not only a reconstruction but the *receiver evidence* that a
channel-conditioned decoder can be conditioned on.

A :class:`MeasurementBundle` collects, for one transmitted latent:

    received            noisy received latent (post-channel, pre-equalisation)
    equalized           optional equalised latent (channel inverse applied)
    channel_gain        per-sample / per-block gain magnitude(s)
    noise_var           per-sample noise variance
    mask                packet/erasure mask or reliability-like signal in [0,1]
    snr_db_true         the channel's true SNR (None when blind)
    snr_db_est          receiver-estimated SNR (blind predictor or proxy)
    phase_est           optional phase estimate
    reliability         per-token reliability proxy in [0,1] (shape [B,1,H,W])
    encode_features_hat optional real receiver feature from the JSCC forward pass
    mask_token          optional SGD-JSCC mask token from the forward pass
    power_scalar        optional SGD-JSCC normalisation scalar
    meta                free-form dict (channel name, params, csi mode, …)

The channel-level fields are produced by the new channels' ``observe()``; the
``encode_features_hat`` / ``mask_token`` / ``power_scalar`` fields are filled in
(optionally) by the JSCC forward pass when measurement collection is requested,
so the bundle works both with and without the heavy models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch


class ChannelTape:
    """Record/replay mixin so one channel realisation is shared across passes.

    DiffCom-style conditioning requires that the channel sample the *condition*
    is derived from equals the sample the decoder actually receives.  Because the
    SGD-JSCC forward pass re-enters ``channel.transmit`` per 128×128 patch and
    re-samples noise/fading each call, a cheap "measurement" pass and the final
    reconstruction would otherwise see *different* realisations.

    This mixin lets a channel **record** the per-call realisations during the
    measurement pass and **replay** them (in call order) during reconstruction,
    making the observed and decoded realisations identical regardless of any RNG
    divergence from a different diffusion budget.  Channels route their
    ``transmit`` through :meth:`_taped_transmit` and implement ``observe``.
    """

    def _init_tape(self) -> None:
        self._record = False
        self._replay_idx = None
        self._tape: list = []
        self.last_bundle = None

    def start_recording(self) -> None:
        self._record = True
        self._replay_idx = None
        self._tape = []

    def start_replay(self) -> None:
        self._record = False
        self._replay_idx = 0

    def stop_tape(self) -> None:
        self._record = False
        self._replay_idx = None

    def recorded_bundles(self) -> list:
        """Return the per-call bundles captured during the last recording pass."""
        return list(getattr(self, "_tape", []))

    def _taped_transmit(self, latent: torch.Tensor, snr_db: float) -> torch.Tensor:
        if self._replay_idx is not None and self._replay_idx < len(self._tape):
            bundle = self._tape[self._replay_idx]
            self._replay_idx += 1
        else:
            bundle = self.observe(latent, snr_db)
            if self._record:
                self._tape.append(bundle)
        self.last_bundle = bundle
        return bundle.best_estimate


def aggregate_bundles(bundles: list) -> Optional["MeasurementBundle"]:
    """Combine per-patch bundles into one image-level :class:`MeasurementBundle`.

    Tensor fields are concatenated along the batch dimension (so the condition
    encoder / reliability head pool over *all* patches, not just the last one);
    scalar / meta fields are taken from the first bundle. A field is kept only if
    present in *every* bundle (so e.g. blind ``equalized=None`` stays None).
    """
    if not bundles:
        return None
    if len(bundles) == 1:
        return bundles[0]

    tensor_fields = ("received", "equalized", "channel_gain", "noise_var",
                     "noise_level", "mask", "reliability", "phase_est",
                     "encode_features_hat", "mask_token", "power_scalar")
    kwargs: Dict = {}
    for f in tensor_fields:
        vals = [getattr(b, f) for b in bundles]
        if all(v is not None for v in vals):
            try:
                kwargs[f] = torch.cat([v for v in vals], dim=0)
            except Exception:  # noqa: BLE001 – shape/agg mismatch → drop field
                kwargs[f] = vals[0]
        else:
            kwargs[f] = None

    first = bundles[0]
    kwargs["snr_db_true"] = first.snr_db_true
    # Average the per-patch SNR estimates when available.
    ests = [b.snr_db_est for b in bundles if b.snr_db_est is not None]
    kwargs["snr_db_est"] = (sum(ests) / len(ests)) if ests else first.snr_db_est
    kwargs["meta"] = dict(first.meta)
    kwargs["meta"]["n_patches"] = len(bundles)
    return MeasurementBundle(**kwargs)


def awgn_noise_like(signal: torch.Tensor, snr_db: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(noise, noise_var)`` matched to *signal* power at *snr_db*.

    Uses the same per-sample power normalisation as ``channels/awgn.py`` so the
    fading channels reduce to the AWGN baseline when the gain is unity.
    Returns the additive noise tensor and the per-sample variance ``[B,1,1,1]``.
    """
    bsz = signal.shape[0]
    norm_2 = torch.linalg.norm(signal.reshape([bsz, -1]), ord=2, dim=1)
    noise_var = ((norm_2 ** 2 / (signal.numel() / bsz)) / (10 ** (snr_db / 10)))
    noise_var = noise_var.reshape([-1, 1, 1, 1])
    noise = torch.randn_like(signal) * torch.sqrt(noise_var)
    return noise, noise_var


def mmse_equalize(
    received: torch.Tensor,
    gain: torch.Tensor,
    noise_var: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Paper-faithful "MMSE equalization + normalization" for a *real-gain* model.

    The SGD-JSCC paper equalizes the channel output by phase removal followed by
    power normalization (Sec. III receiver; Sec. V, eq. for fast fading):

        y_eq = e^{-jφ} y / sqrt(|h|² + σ²)   ⇒  real gain g, φ=0:  y_eq = y / sqrt(g² + σ²)

    giving the variance-preserving form ``y_eq = sqrt(α)·x + sqrt(1-α)·n`` with
    ``α = g²/(g²+σ²)`` — exactly the diffusion intermediate state the DM expects.

    Behaviour vs the previous zero-forcing (``y/g``):
      * ``noise_var → 0``  →  ``y / sqrt(g²) = y / g``  (converges to zero-forcing);
      * ``noise_var ↑``    →  divides by a larger ``sqrt(g²+σ²)`` (MMSE-style shrink).

    Real-gain limitation
    --------------------
    These channels model a **real, non-negative** gain ``g = |h|`` and add real
    AWGN, so there is no complex phase ``e^{-jφ}`` to remove. This routine
    therefore reproduces the paper's magnitude/normalization behaviour but **not**
    the complex phase rotation of a true ``h ∈ ℂ`` channel (that would need a
    complex-valued latent path, which the SGD-JSCC forward pass does not expose).
    """
    return received / torch.sqrt(gain ** 2 + noise_var + eps)


@dataclass
class MeasurementBundle:
    """Container for per-latent receiver evidence (see module docstring)."""

    received: torch.Tensor
    equalized: Optional[torch.Tensor] = None
    channel_gain: Optional[torch.Tensor] = None
    noise_var: Optional[torch.Tensor] = None
    # Per-element noise level d_i = σ²/(|h_i|²+σ²) ∈ [0,1) of the EQUALIZED latent
    # (paper eq. 12: f̃_i = √(1-d_i) f0_i + √d_i n_i). Feeds the fast-fading
    # water-filling denoiser (Algorithm 4). Shape broadcasts to the latent.
    noise_level: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    snr_db_true: Optional[float] = None
    snr_db_est: Optional[float] = None
    phase_est: Optional[torch.Tensor] = None
    reliability: Optional[torch.Tensor] = None
    # Optional real SGD-JSCC receiver evidence (filled by the forward pass).
    encode_features_hat: Optional[torch.Tensor] = None
    mask_token: Optional[torch.Tensor] = None
    power_scalar: Optional[torch.Tensor] = None
    meta: Dict = field(default_factory=dict)

    @property
    def best_estimate(self) -> torch.Tensor:
        """The most decoder-ready latent: equalised if present, else received."""
        return self.equalized if self.equalized is not None else self.received

    def mean_reliability(self) -> float:
        """Scalar mean reliability in [0,1] (1.0 when no reliability signal)."""
        r = self.reliability if self.reliability is not None else self.mask
        if r is None:
            return 1.0
        return float(r.float().mean().item())

    def summary(self) -> Dict:
        """JSON-friendly scalar summary (no tensors) for logging."""
        def _scal(t):
            return None if t is None else float(t.float().mean().item())
        return {
            "snr_db_true": self.snr_db_true,
            "snr_db_est": self.snr_db_est,
            "channel_gain_mean": _scal(self.channel_gain),
            "noise_var_mean": _scal(self.noise_var),
            "mask_keep_fraction": _scal(self.mask),
            "mean_reliability": self.mean_reliability(),
            **{k: v for k, v in self.meta.items() if isinstance(v, (int, float, str, bool, type(None)))},
        }

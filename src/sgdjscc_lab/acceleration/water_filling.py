"""acceleration/water_filling.py – Fast-fading water-filling denoising (paper Algorithm 4).

Implements the SGD-JSCC paper's *training-free* fast-fading denoising
(Sec. V "Extension to Fast Fading Case", Algorithm 4). It lets a DM trained on a
slow-fading (uniform-noise) channel denoise a fast-fading channel output whose
elements carry **heterogeneous** per-element noise levels.

Setup (paper eqs. 11–12)
-----------------------
After MMSE equalization + normalization, the per-element equalized latent is

    f̃_i = √(1 − d_i) · f0_i + √(d_i) · n_i ,   d_i = σ² / (|h_i|² + σ²) ∈ [0,1)

i.e. element ``i`` sits at noise level ``d_i`` along the diffusion trajectory.

Algorithm 4 (per step, current level t → next level s = t − 1/T)
---------------------------------------------------------------
1. **Water filling** (eq. 16): raise *every* element to the common target
   ``β̄_t`` by adding noise (elements already at ``β̄_t`` get none)::

       g_t,i = √((1−β̄_t)/(1−b_t,i))·f_t,i + √(β̄_t − b_t,i·(1−β̄_t)/(1−b_t,i))·ε

   so ``p(g_t|f0) = N(√(1−β̄_t) f0, β̄_t I)`` — a valid uniform-noise diffusion state.
2. **One DM step** (eq. 17, same form as the slow-fading sampler / Algorithm 2)::

       ĝ_s = √(β̄_s/β̄_t)·g_t + (√(1−β̄_s) − √(β̄_s(1−β̄_t)/β̄_t))·ε_Ω(g_t, β̄_t)

   where ``ε_Ω`` is the DM's **f0 prediction** conditioned on noise level ``√β̄_t``.
3. **Selective update**: element ``i`` adopts ``ĝ_s`` (and ``b←β̄_s``) only if
   ``b_t,i ≥ β̄_s`` (still noisier than the next target); cleaner elements are kept.

Initialisation: ``t = S^{-1}(max_i d_i)``, ``f_t = f̃``, ``b_t = d`` — the footnote's
"set the initial noise level to the maximum channel noise level" guarantees the
invariant ``b_t,i ≤ β̄_t`` throughout.

This module implements the **algorithm** with an *injectable* f0-predictor so it is
verifiable on CPU with a synthetic denoiser. The real-DM adapter
(:func:`build_mdt_f0_predictor`, over ``DiffusionGenerator.pred_image``) and the
runtime decode-swap are wired in ``pipelines/infer_pipeline.py``
(``_run_water_filling_diffusion``, gated by ``cfg.use_water_filling``); only the
numeric run needs the MDTv2 checkpoint. See docs/phase5.md.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

from sgdjscc_lab.training.noise_schedule import SigmoidNoiseScheduler

logger = logging.getLogger(__name__)

# An f0-predictor: (latent [B,C,H,W], noise_level [B,1]) -> predicted f0 [B,C,H,W].
F0Predictor = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def water_fill(f_t: torch.Tensor, b_t: torch.Tensor, beta_t: float,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Eq. 16: raise every element from its level ``b_t`` to the common ``β̄_t``.

    ``g_t,i = √((1−β̄_t)/(1−b_t,i))·f_t,i + √(β̄_t − b_t,i·(1−β̄_t)/(1−b_t,i))·ε``.
    Elements already at ``β̄_t`` receive zero added variance (identity).
    """
    ratio = (1.0 - beta_t) / (1.0 - b_t)
    sig_coef = torch.sqrt(ratio.clamp(min=0.0))
    add_var = (beta_t - b_t * ratio).clamp(min=0.0)
    eps = torch.randn(f_t.shape, device=f_t.device, dtype=f_t.dtype, generator=generator)
    return sig_coef * f_t + torch.sqrt(add_var) * eps


def _water_filling_step(
    f_t: torch.Tensor, b_t: torch.Tensor, beta_t: float, beta_s: float,
    f0_predict_fn: F0Predictor, generator: Optional[torch.Generator] = None,
):
    """One Algorithm-4 iteration: water-fill → DM step → selective update.

    Returns ``(f_s, b_s, g_t)`` where ``g_t`` is the water-filled (uniform-level)
    latent (exposed for testing/inspection).
    """
    g_t = water_fill(f_t, b_t, beta_t, generator=generator)
    nl_t = torch.full((f_t.shape[0], 1), beta_t ** 0.5, device=f_t.device, dtype=f_t.dtype)
    f0_hat = f0_predict_fn(g_t, nl_t)
    a = (beta_s / beta_t) ** 0.5
    c = (1.0 - beta_s) ** 0.5 - (beta_s * (1.0 - beta_t) / beta_t) ** 0.5
    g_s = a * g_t + c * f0_hat
    update = b_t >= beta_s                                   # still noisier than next target
    f_s = torch.where(update, g_s, f_t)
    b_s = torch.where(update, torch.full_like(b_t, beta_s), b_t)
    return f_s, b_s, g_t


def water_filling_denoise(
    f_tilde: torch.Tensor,
    noise_level: torch.Tensor,
    f0_predict_fn: F0Predictor,
    scheduler: Optional[SigmoidNoiseScheduler] = None,
    steps: int = 50,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Run the fast-fading water-filling denoising loop (paper Algorithm 4).

    Parameters
    ----------
    f_tilde:
        Equalized noisy latent ``f̃`` ``[B,C,H,W]`` (paper eq. 12).
    noise_level:
        Per-element noise level ``d`` (same shape as ``f_tilde``, or broadcastable),
        ``d_i = σ²/(|h_i|²+σ²) ∈ [0,1)``.
    f0_predict_fn:
        The DM's f0 predictor ``(g_t, √β̄_t) -> f̂0``. For verification a synthetic
        predictor can be injected; in production this wraps the MDTv2 denoiser.
    scheduler:
        :class:`SigmoidNoiseScheduler` (defaults to the paper's e=3, τ=0.7).
    steps:
        ``T`` in ``s = t − 1/T`` — the (fixed) timestep resolution.

    Returns
    -------
    ``f̂0`` ``[B,C,H,W]`` — the denoised latent.
    """
    if scheduler is None:
        scheduler = SigmoidNoiseScheduler()
    cmin = scheduler.clip_min

    device = f_tilde.device
    f_t = f_tilde.clone()
    # Per-element current noise level b_t (clamped into the open interval).
    b_t = noise_level.to(device, dtype=f_t.dtype).expand_as(f_t).clone()
    b_t = b_t.clamp(cmin, 1.0 - cmin)

    # Initialise t at the MAX noise level over the batch (footnote invariant).
    d_max = float(b_t.max().item())
    t = float(scheduler.inverse_beta_bar(torch.tensor(d_max)).item())
    t = min(max(t, 0.0), 1.0)
    dt = 1.0 / float(max(1, steps))

    def _bb(x: float) -> float:
        return float(scheduler.beta_bar(torch.tensor(x)).item())

    n_iter = 0
    while t > cmin:
        s = max(t - dt, 0.0)
        f_t, b_t, _ = _water_filling_step(
            f_t, b_t, _bb(t), _bb(s), f0_predict_fn, generator=generator)
        t = s
        n_iter += 1
        if n_iter > 4 * max(1, steps) + 4:   # safety guard against pathological loops
            logger.warning("water_filling_denoise: iteration guard hit (t=%.4g).", t)
            break

    return f_t


def water_filling_denoise_from_bundle(
    bundle,
    f0_predict_fn: F0Predictor,
    scheduler: Optional[SigmoidNoiseScheduler] = None,
    steps: int = 50,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Run Algorithm 4 directly from a fast-fading :class:`MeasurementBundle`.

    Reads ``bundle.equalized`` (``f̃``) and ``bundle.noise_level`` (per-element
    ``d``). Raises if the bundle lacks per-element noise levels (e.g. a blind
    ``csi="none"`` bundle has no equalized latent) — the caller should fall back to
    the standard sampler in that case.

    Real-DM wiring (connected)
    --------------------------
    In production ``f0_predict_fn`` wraps the SGD-JSCC diffusion denoiser via
    :func:`build_mdt_f0_predictor` (over ``DiffusionGenerator.pred_image``). The
    fast-fading branch of ``pipelines/infer_pipeline.py`` calls this loop
    (``_run_water_filling_diffusion``) when ``cfg.use_water_filling`` is set and the
    channel supplies a per-element ``noise_level``; only producing numbers needs the
    MDTv2 checkpoint (the routing/adapter are CPU-tested with a stub ``pipe``).
    """
    f_tilde = bundle.equalized
    d = bundle.noise_level
    if f_tilde is None:
        # A blind (csi="none") bundle is NOT equalized — it carries `noise_level`
        # but no `equalized` latent, so Algorithm 4 cannot start.
        raise ValueError(
            "water_filling_denoise_from_bundle needs bundle.equalized (the "
            "equalized latent f̃). This bundle has none — it is blind "
            "(csi='none'); fall back to the standard step-matched sampler.")
    if d is None:
        raise ValueError(
            "water_filling_denoise_from_bundle needs bundle.noise_level (the "
            "per-element noise level d). Use a fading channel that populates it "
            "(channels/fast_fading.py or rayleigh.py).")
    # Paper assumption: Algorithm 4 assumes PERFECT CSI at the receiver. With an
    # imperfect estimate, (f̃, d) are both derived from the same noisy gain so they
    # stay mutually consistent, but neither matches the true channel state — treat
    # such runs as approximate.
    csi = (bundle.meta or {}).get("csi") if hasattr(bundle, "meta") else None
    if csi is not None and str(csi) != "perfect":
        logger.warning("water_filling_denoise: bundle csi=%r but Algorithm 4 assumes "
                       "PERFECT CSI — (f̃, d) are self-consistent (same estimate) but "
                       "approximate vs the true channel.", csi)
    return water_filling_denoise(
        f_tilde, d, f0_predict_fn, scheduler=scheduler, steps=steps, generator=generator)


# ─────────────────────────────────────────────────────────────────────────────
# Real-DM adapter + CSI-policy decode
# ─────────────────────────────────────────────────────────────────────────────

def build_mdt_f0_predictor(
    pred_image_fn,
    labels,
    class_guidance,
    *,
    c=None,
    controlnet: bool = False,
    mask_token=None,
    not_control=None,
):
    """Wrap the SGD-JSCC DM's ``pred_image`` into ``f0_predict_fn(g_t, √β̄_t)``.

    Mirrors the public ``DiffusionGenerator.pred_image`` call convention
    (``SGDJSCC/models/test_advanced_network/diffusion_element_wise.py``):

        pred_image(noisy[B,C,H,W], labels[2B], noise_level_np[B]=σ=√β̄,
                   class_guidance_np[B], c, controlnet, mask_token, not_control)
                   -> f0_pred[B,C,H,W]   (already CFG-combined)

    so this adapter only re-shapes our ``noise_level`` tensor ``[B,1]`` to the
    numpy ``[B]`` that ``pred_image`` expects (it internally duplicates the latent
    and noise level for classifier-free guidance). ``labels`` MUST already be the
    concatenated ``[pos; neg]`` ``[2B]`` embedding the public code builds before
    its loop, and ``class_guidance`` a per-sample array ``[B]``.

    paper-like: the algorithm (variance space) is the paper's; the call wiring
    follows the public code (std space, internal CFG duplication).
    """
    import numpy as np

    cg = np.asarray(class_guidance, dtype=np.float64).reshape(-1)

    def f0_predict_fn(g_t: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        # noise_level is √β̄_t, shape [B,1]; pred_image wants numpy [B].
        nl = noise_level.detach().to("cpu").numpy().reshape(-1)
        out = pred_image_fn(g_t, labels, nl, cg, c, controlnet, mask_token, not_control)
        return out.to(g_t.device, g_t.dtype)

    return f0_predict_fn


def fast_fading_water_filling_decode(
    bundle,
    f0_predict_fn: F0Predictor,
    *,
    scheduler: Optional[SigmoidNoiseScheduler] = None,
    steps: int = 50,
    on_blind: str = "error",
    fallback_fn=None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Run Algorithm-4 decode from a bundle, applying the **CSI policy**.

    Policy (matches the paper's perfect-CSI assumption):
      * ``perfect``   → run water-filling (default supported case);
      * ``imperfect`` → run, but ``water_filling_denoise_from_bundle`` warns that
        ``(f̃, d)`` are self-consistent yet approximate vs the true channel;
      * blind (``csi="none"`` / no ``equalized``) → ``on_blind``:
          ``"error"``    raise (default — Algorithm 4 cannot start);
          ``"fallback"`` call ``fallback_fn(bundle)`` (e.g. the standard
                         step-matched sampler) and return its result.

    Returns the denoised latent ``f̂0``.
    """
    is_blind = bundle.equalized is None
    if is_blind:
        if str(on_blind) == "fallback":
            if fallback_fn is None:
                raise ValueError(
                    "fast_fading_water_filling_decode: on_blind='fallback' but no "
                    "fallback_fn was provided (e.g. the standard sampler).")
            logger.warning("blind CSI (no equalized latent) → water-filling skipped; "
                           "using fallback decode.")
            return fallback_fn(bundle)
        raise ValueError(
            "fast_fading_water_filling_decode: blind bundle (csi='none', no "
            "equalized latent) — Algorithm 4 needs the equalized f̃. Provide "
            "on_blind='fallback' with a fallback_fn, or use perfect/imperfect CSI.")
    return water_filling_denoise_from_bundle(
        bundle, f0_predict_fn, scheduler=scheduler, steps=steps, generator=generator)

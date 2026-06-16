"""training/losses.py – Config-driven loss scaffold for sgdjscc_lab training.

Designed to be extended without breaking existing inference/evaluation paths.
All loss classes accept ``(pred, target)`` float32 tensors ``[B, C, H, W]``.

Config schema (train/default.yaml → loss block)
------------------------------------------------
loss:
  use_reconstruction: true
  reconstruction_type: "l1"    # "l1" | "mse" | "huber"
  reconstruction_weight: 1.0

  use_semantic: false           # CLIP-based semantic similarity (placeholder)
  semantic_weight: 0.0

Extension points
----------------
- Subclass SemanticLoss and plug in a trained CLIP/BLIP evaluator.
- Add more loss terms (e.g. perceptual / SSIM) by adding a block to TotalLoss.
- Keep weights config-driven so ablations need only a YAML change.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Individual loss modules
# ─────────────────────────────────────────────────────────────────────────────

class ReconstructionLoss(nn.Module):
    """Pixel-level reconstruction loss between reconstructed and original images.

    Parameters
    ----------
    loss_type:
        ``"l1"`` (default), ``"mse"``, or ``"huber"``.
    """

    def __init__(self, loss_type: str = "l1") -> None:
        super().__init__()
        self.loss_type = loss_type.lower()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        if self.loss_type == "huber":
            return F.huber_loss(pred, target)
        return F.l1_loss(pred, target)   # default: l1


class SemanticLoss(nn.Module):
    """Semantic similarity loss placeholder.

    Currently returns a zero tensor.  Replace ``_compute`` with a real
    CLIP / SRS-based loss once the training objective is finalised.

    Extension example::

        class CLIPSemanticLoss(SemanticLoss):
            def __init__(self, clip_evaluator):
                super().__init__()
                self.clip = clip_evaluator

            def _compute(self, pred, target):
                sim = self.clip.image_similarity(pred, target)
                return 1.0 - sim.mean()
    """

    def __init__(self) -> None:
        super().__init__()

    def _compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=pred.device, requires_grad=pred.requires_grad)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._compute(pred, target)


class TotalLoss(nn.Module):
    """Weighted sum of active loss terms.

    Parameters
    ----------
    reconstruction_loss:
        ReconstructionLoss instance (or any nn.Module with the same signature).
    semantic_loss:
        SemanticLoss instance (or any nn.Module with the same signature).
    reconstruction_weight:
        Scalar weight for the reconstruction term.
    semantic_weight:
        Scalar weight for the semantic term.
    """

    def __init__(
        self,
        reconstruction_loss: Optional[nn.Module] = None,
        semantic_loss: Optional[nn.Module] = None,
        reconstruction_weight: float = 1.0,
        semantic_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.recon_loss   = reconstruction_loss
        self.sem_loss     = semantic_loss
        self.recon_weight = reconstruction_weight
        self.sem_weight   = semantic_weight

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Return a dict with individual and total losses."""
        total = torch.tensor(0.0, device=pred.device)
        out: Dict[str, torch.Tensor] = {}

        if self.recon_loss is not None and self.recon_weight > 0:
            rl = self.recon_loss(pred, target)
            out["loss_recon"] = rl
            total = total + self.recon_weight * rl

        if self.sem_loss is not None and self.sem_weight > 0:
            sl = self.sem_loss(pred, target)
            out["loss_semantic"] = sl
            total = total + self.sem_weight * sl

        out["loss"] = total
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_loss(cfg: DictConfig) -> TotalLoss:
    """Build a TotalLoss from the ``loss`` block of the training config.

    Gracefully handles a missing ``loss`` block by returning an L1
    reconstruction loss with weight 1.0.
    """
    loss_cfg = OmegaConf.select(cfg, "loss", default=None)

    use_recon = True
    recon_type = "l1"
    recon_w = 1.0
    use_sem = False
    sem_w = 0.0

    if loss_cfg is not None:
        use_recon  = bool(OmegaConf.select(loss_cfg, "use_reconstruction", default=True))
        recon_type = str(OmegaConf.select(loss_cfg, "reconstruction_type", default="l1"))
        recon_w    = float(OmegaConf.select(loss_cfg, "reconstruction_weight", default=1.0))
        use_sem    = bool(OmegaConf.select(loss_cfg, "use_semantic", default=False))
        sem_w      = float(OmegaConf.select(loss_cfg, "semantic_weight", default=0.0))

    recon = ReconstructionLoss(recon_type) if use_recon else None
    sem   = SemanticLoss()                 if use_sem   else None

    logger.info(
        "Loss: recon=%s (w=%.2f, type=%s)  semantic=%s (w=%.2f)",
        use_recon, recon_w, recon_type, use_sem, sem_w,
    )
    return TotalLoss(recon, sem, recon_w, sem_w)


# ═════════════════════════════════════════════════════════════════════════════
# Stage-aware losses (paper-faithful 3-stage training)
# ═════════════════════════════════════════════════════════════════════════════
#
# Stage 1 (JSCC):      L = ||x - x̂||²  +  λ · L_GAN          (eq. 7)
# Stage 2 (text DM):   L = ||f0 - ε(f_t)||² + ||f0 - ε(f̂_t)||²  (Algorithm 1, MDT)
# Stage 3 (ControlNet) reuses the stage-2 DiffusionF0Loss; only the trainable
#                      module set differs (enforced by training/freeze.py).


def _disc_norm(kind: str, ch: int) -> nn.Module:
    """Normalisation layer for the discriminator (batch | instance | none)."""
    kind = (kind or "batch").lower()
    if kind == "instance":
        return nn.InstanceNorm2d(ch, affine=False)
    if kind in ("none", "spectral"):  # spectral norm wraps convs, not a layer
        return nn.Identity()
    return nn.BatchNorm2d(ch)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator (Pix2Pix/LDM ``NLayerDiscriminator`` style).

    This mirrors the standard ``NLayerDiscriminator`` used by the LDM/taming
    autoencoders the paper builds on — a stack of stride-2 4×4 convs followed by
    a 1-stride head producing a patch logit map.  All structural knobs
    (``ndf``, ``n_layers``, ``norm``) are config-driven.  The weighting/LPIPS
    interplay from the original repo is not reproduced, so GAN-on perceptual
    numbers remain a structural approximation, not a numeric match.

    Parameters
    ----------
    in_channels:
        Input image channels (3).
    ndf:
        Base feature width.
    n_layers:
        Number of stride-2 downsampling blocks.
    norm:
        ``"batch"`` (default), ``"instance"`` or ``"none"``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=2, padding=1),
                _disc_norm(norm, ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=1, padding=1),
            _disc_norm(norm, ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, 4, stride=1, padding=1),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


def build_discriminator(cfg: DictConfig) -> "PatchDiscriminator":
    """Build a :class:`PatchDiscriminator` from ``train.jscc.gan`` config."""
    gan = OmegaConf.select(cfg, "train.jscc.gan", default=None)
    ndf = int(OmegaConf.select(gan, "ndf", default=64)) if gan else 64
    n_layers = int(OmegaConf.select(gan, "n_layers", default=3)) if gan else 3
    norm = str(OmegaConf.select(gan, "norm", default="batch")) if gan else "batch"
    in_ch = int(OmegaConf.select(gan, "in_channels", default=3)) if gan else 3
    logger.info("PatchDiscriminator: in=%d ndf=%d n_layers=%d norm=%s",
                in_ch, ndf, n_layers, norm)
    return PatchDiscriminator(in_ch, ndf, n_layers, norm)


class GANLoss(nn.Module):
    """Hinge GAN loss helper exposing generator and discriminator objectives."""

    def __init__(self, mode: str = "hinge") -> None:
        super().__init__()
        self.mode = mode.lower()

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        if self.mode == "hinge":
            return -fake_logits.mean()
        return F.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits)
        )

    def discriminator_loss(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor
    ) -> torch.Tensor:
        if self.mode == "hinge":
            return (F.relu(1.0 - real_logits).mean()
                    + F.relu(1.0 + fake_logits).mean())
        real = F.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits))
        fake = F.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits))
        return real + fake


class JSCCStageLoss(nn.Module):
    """Stage-1 JSCC objective: MSE + optional patch-GAN + optional LPIPS.

    ``forward(recon, target, disc=None)`` returns a dict with ``loss_mse``,
    ``loss_gan`` (generator term, only when *disc* is given and ``gan_weight>0``),
    ``loss_lpips`` (when ``lpips_weight>0`` and an LPIPS backend is available) and
    the combined ``loss``.  The discriminator's own update is driven separately by
    the stage runner via :class:`GANLoss`.

    Perceptual term
    ---------------
    The paper's Stage-1 loss is ``‖x−x̂‖² + λ·L_GAN`` (patch discriminator). The
    public SGD-JSCC code (``utils/addtional_loss.py::MSE_LPIPS``) couples MSE with
    LPIPS (``mse + w·lpips(2x−1, 2x̂−1)``). We expose **both** config-driven and
    default-off: LPIPS aligns with the public code's perceptual term, the patch-GAN
    with the paper. ``lpips_fn`` is injectable (tests); otherwise it is lazily built
    from the ``lpips`` package and degrades gracefully (term skipped) if absent.
    This is **structural alignment**, not exact numeric reproduction.
    """

    def __init__(self, gan_weight: float = 0.0, gan_mode: str = "hinge",
                 lpips_weight: float = 0.0, lpips_net: str = "alex",
                 lpips_fn=None) -> None:
        super().__init__()
        self.gan_weight = float(gan_weight)
        self.gan = GANLoss(gan_mode)
        self.lpips_weight = float(lpips_weight)
        self.lpips_net = str(lpips_net)
        self._lpips_fn = lpips_fn
        self._lpips_unavailable = False

    def _lpips(self, recon: torch.Tensor, target: torch.Tensor):
        """Return the LPIPS perceptual distance (frozen net), or None if absent."""
        if self._lpips_fn is None and not self._lpips_unavailable:
            try:
                import lpips
                fn = lpips.LPIPS(net=self.lpips_net)
                for p in fn.parameters():
                    p.requires_grad_(False)          # loss net is fixed (public code)
                self._lpips_fn = fn.to(recon.device).eval()
                logger.info("Stage-1 LPIPS perceptual term enabled (net=%s, w=%.3f).",
                            self.lpips_net, self.lpips_weight)
            except Exception as exc:  # pragma: no cover - env-dependent
                logger.warning("LPIPS unavailable (%s) — perceptual term skipped. "
                               "Install the 'lpips' package to enable it.", exc)
                self._lpips_unavailable = True
        if self._lpips_fn is None:
            return None
        # LPIPS expects inputs in [-1, 1] (public MSE_LPIPS uses 2x-1).
        return self._lpips_fn(recon * 2 - 1, target * 2 - 1).mean()

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        disc: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        mse = F.mse_loss(recon, target)
        out["loss_mse"] = mse
        total = mse
        if disc is not None and self.gan_weight > 0:
            gan = self.gan.generator_loss(disc(recon * 2 - 1))
            out["loss_gan"] = gan
            total = total + self.gan_weight * gan
        if self.lpips_weight > 0:
            lp = self._lpips(recon, target)
            if lp is not None:
                out["loss_lpips"] = lp
                total = total + self.lpips_weight * lp
        out["loss"] = total
        return out


class DiffusionF0Loss(nn.Module):
    """Stage 2/3 DM objective: f0-prediction MSE over masked + unmasked branches.

    Mirrors the MDT training objective the paper adopts (Algorithm 1 plus the
    masked-latent modeling of MDTv2):

        loss = ||f0 - ε(f_t)||²  +  mask_weight · ||f0 - ε(f̂_t)||²

    where ``ε(f_t)`` is the unmasked prediction and ``ε(f̂_t)`` the prediction of
    the masked-latent branch.  When ``pred_masked`` is None only the unmasked
    term is used (and a warning-free single-term loss is returned).
    """

    def __init__(self, mask_weight: float = 1.0) -> None:
        super().__init__()
        self.mask_weight = float(mask_weight)

    def forward(
        self,
        f0: torch.Tensor,
        pred_unmasked: torch.Tensor,
        pred_masked: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        l_unmasked = F.mse_loss(pred_unmasked, f0)
        out["loss_dm_unmasked"] = l_unmasked
        total = l_unmasked
        if pred_masked is not None:
            l_masked = F.mse_loss(pred_masked, f0)
            out["loss_dm_masked"] = l_masked
            total = total + self.mask_weight * l_masked
        out["loss"] = total
        return out


class EndToEndFTLoss(nn.Module):
    """End-to-end fine-tuning objective (paper appendix extension).

    Combines an image-space reconstruction term (the goal of joint JSCC↔DM
    fine-tuning is to minimise the distortion of the *final* reconstructed image)
    with a diffusion f0-prediction term that preserves the DM's denoising ability:

        loss = recon_weight · ||x - x̂||²  +  diff_weight · ||f0 - ε(f_t)||²

    ``forward(recon, target, f0, pred_f0)`` returns the individual + total terms.
    The ``diff`` term is optional (set ``diff_weight=0`` or pass ``pred_f0=None``).
    """

    def __init__(self, recon_weight: float = 1.0, diff_weight: float = 1.0) -> None:
        super().__init__()
        self.recon_weight = float(recon_weight)
        self.diff_weight = float(diff_weight)

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        pred_f0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        l_recon = F.mse_loss(recon, target)
        out["loss_recon"] = l_recon
        total = self.recon_weight * l_recon
        if pred_f0 is not None and f0 is not None and self.diff_weight > 0:
            l_diff = F.mse_loss(pred_f0, f0)
            out["loss_diff"] = l_diff
            total = total + self.diff_weight * l_diff
        out["loss"] = total
        return out


class EdgeCodecLoss(nn.Module):
    """Edge-codec objective: BCE-with-logits + soft-Dice on the reconstructed edge.

    Trains the dedicated edge JSCC (``models/edge_jscc.py``) as a real edge
    reconstruction codec, matching the paper's edge-transmission objective
    (Sec. V): a per-pixel binary term (BCE) plus a region-overlap term (Dice)
    that copes with the heavy foreground/background imbalance of thin edge maps.

        loss = bce_weight · BCE(logits, edge)  +  dice_weight · (1 − Dice(σ(logits), edge))

    ``forward(logits, target)`` takes raw logits ``[B, 1, H, W]`` and a target
    edge map in ``[0, 1]`` and returns a dict with ``loss_bce``, ``loss_dice``
    and the combined ``loss``.
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 eps: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.eps = float(eps)

    def _dice(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b = prob.shape[0]
        p = prob.reshape(b, -1)
        t = target.reshape(b, -1)
        inter = (p * t).sum(dim=1)
        denom = p.sum(dim=1) + t.sum(dim=1)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        target = target.to(logits.dtype)
        out: Dict[str, torch.Tensor] = {}
        bce = F.binary_cross_entropy_with_logits(logits, target)
        out["loss_bce"] = bce
        total = self.bce_weight * bce
        if self.dice_weight > 0:
            dice = self._dice(torch.sigmoid(logits), target)
            out["loss_dice"] = dice
            total = total + self.dice_weight * dice
        out["loss"] = total
        return out


class SNREstimationLoss(nn.Module):
    """Paper eq. 15: ``min E[‖ζ_P(√α f + √(1-α) n) − α‖²]`` — MSE between the
    predicted signal level ``α̂`` and the true ``α``. (paper-like objective)."""

    def forward(self, alpha_hat: torch.Tensor, alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss = F.mse_loss(alpha_hat, alpha.to(alpha_hat.dtype))
        return {"loss_snr": loss, "loss": loss}


class PhaseEstimationLoss(nn.Module):
    """Paper phase objective: ``min E[‖ξ_Q(…) − φ/π‖²]`` — MSE on the normalized
    phase. (paper-inspired SCAFFOLD: no phase net exists in the public code and the
    real-gain channels carry no complex phase — see models/csi_estimation.py)."""

    def forward(self, phi_hat: torch.Tensor, phi_over_pi: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss = F.mse_loss(phi_hat, phi_over_pi.to(phi_hat.dtype))
        return {"loss_phase": loss, "loss": loss}


def synthesize_noisy_latent(f0: torch.Tensor, alpha: torch.Tensor,
                            generator=None) -> torch.Tensor:
    """Self-supervised CSI-estimation training pair (paper eq. 15 form):
    ``f̄ = √α·f0 + √(1-α)·n``, ``n~N(0,I)``. ``alpha`` is ``[B,1]`` (broadcast)."""
    a = alpha.reshape(f0.shape[0], *([1] * (f0.dim() - 1))).to(f0.dtype)
    n = torch.randn(f0.shape, device=f0.device, dtype=f0.dtype, generator=generator)
    return torch.sqrt(a.clamp(0, 1)) * f0 + torch.sqrt((1 - a).clamp(0, 1)) * n


def build_stage_loss(cfg: DictConfig, stage: str):
    """Build the loss module for *stage*.

    ``jscc`` → :class:`JSCCStageLoss`, ``text_dm`` / ``controlnet`` →
    :class:`DiffusionF0Loss`, ``edge_codec`` → :class:`EdgeCodecLoss`,
    ``end_to_end_ft`` → :class:`EndToEndFTLoss`.
    """
    from sgdjscc_lab.training.stages import (
        STAGE_JSCC, STAGE_TEXT_DM, STAGE_CONTROLNET, STAGE_EDGE_CODEC,
        STAGE_CSI_ESTIMATION, STAGE_END_TO_END_FT,
    )

    if stage == STAGE_CSI_ESTIMATION:
        logger.info("CSI-estimation loss: MSE on α (paper eq. 15).")
        return SNREstimationLoss()

    if stage == STAGE_EDGE_CODEC:
        bw = float(OmegaConf.select(cfg, "train.edge_codec.bce_weight", default=1.0))
        dw = float(OmegaConf.select(cfg, "train.edge_codec.dice_weight", default=1.0))
        logger.info("Edge-codec loss: %.2f·BCE + %.2f·Dice", bw, dw)
        return EdgeCodecLoss(bce_weight=bw, dice_weight=dw)

    if stage == STAGE_JSCC:
        gan_enabled = bool(OmegaConf.select(cfg, "train.jscc.gan.enabled", default=False))
        gan_weight = float(OmegaConf.select(cfg, "train.jscc.gan.weight", default=0.0))
        gan_mode = str(OmegaConf.select(cfg, "train.jscc.gan.mode", default="hinge"))
        lpips_enabled = bool(OmegaConf.select(cfg, "train.jscc.lpips.enabled", default=False))
        lpips_weight = float(OmegaConf.select(cfg, "train.jscc.lpips.weight", default=0.0))
        lpips_net = str(OmegaConf.select(cfg, "train.jscc.lpips.net", default="alex"))
        logger.info("Stage-1 JSCC loss: MSE + GAN(enabled=%s, w=%.3f) + LPIPS(enabled=%s, w=%.3f)",
                    gan_enabled, gan_weight if gan_enabled else 0.0,
                    lpips_enabled, lpips_weight if lpips_enabled else 0.0)
        return JSCCStageLoss(
            gan_weight=gan_weight if gan_enabled else 0.0, gan_mode=gan_mode,
            lpips_weight=lpips_weight if lpips_enabled else 0.0, lpips_net=lpips_net,
        )

    if stage in (STAGE_TEXT_DM, STAGE_CONTROLNET):
        mask_weight = float(OmegaConf.select(cfg, "train.dm.mask_weight", default=1.0))
        logger.info("Stage-2/3 DM loss: f0-MSE (unmasked + %.2f·masked)", mask_weight)
        return DiffusionF0Loss(mask_weight=mask_weight)

    if stage == STAGE_END_TO_END_FT:
        rw = float(OmegaConf.select(cfg, "train.end_to_end_ft.recon_weight", default=1.0))
        dw = float(OmegaConf.select(cfg, "train.end_to_end_ft.diff_weight", default=1.0))
        logger.info("End-to-end FT loss: %.2f·recon + %.2f·diff", rw, dw)
        return EndToEndFTLoss(recon_weight=rw, diff_weight=dw)

    raise ValueError(f"build_stage_loss: unknown stage {stage!r}")

"""training/val_images.py – periodic validation-sample image logging.

Scalar validation loss tells you *that* a run is improving; it does not show
*what* the model produces.  This logger periodically writes a small panel of
actual samples so a human can eyeball reconstruction quality during training —
most useful for Stage 3 (ControlNet, edge-conditioned) and Stage 2 (text_dm).

Design constraints honoured
---------------------------
* **Real validation data + eval mode.** The panel is rendered from a FIXED
  validation batch (the first batch of ``val_loader``, so the same samples are
  tracked across time) with the runner switched to ``set_mode(False)`` under
  ``torch.no_grad`` — a genuine validation view, not a training-batch snapshot.
  When no ``val_loader`` is available it falls back to the passed training batch
  (clearly the degraded case) but still renders in eval mode.
* **Reuse, don't reinvent.** It calls the *runner's own* wired callables
  (``encode_latent_fn`` / ``encode_text_fn`` / ``encode_edge_fn`` / ``scheduler`` /
  the trainable denoiser) plus the shared VAE decode — no new inference stack.
  The saved reconstruction is the runner's one-step ``f0`` prediction decoded to
  image space (labelled as such), which mirrors exactly the objective the stage
  optimizes.
* **Off by default.** Controlled entirely by ``train.val_images.*``; with the
  shipped defaults (``enabled: false``) nothing is written and there is zero
  overhead, so existing runs / reproducibility are unaffected.
* **Rank 0 only.** Under DDP only the main rank materialises and writes images.
* **Priority: controlnet > text_dm.** Other stages (jscc) are supported as a
  bonus when trivially expressible; unsupported stages are skipped with a log.

Panel layout (one row per sample): ``input | edge(if any) | recon``.
Files land in ``<checkpoint_dir>/<subdir>/step_XXXXXXX.png`` (or ``epoch_XXXX.png``);
using ``checkpoint_dir`` as the base is unambiguous regardless of nesting depth.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch

from sgdjscc_lab import distributed as ddp
from sgdjscc_lab.training.stages import (
    STAGE_CONTROLNET, STAGE_JSCC, STAGE_TEXT_DM, STAGE_END_TO_END_FT,
)

logger = logging.getLogger(__name__)

# Stages we can visualise, in priority order (controlnet first, per the task).
_SUPPORTED = (STAGE_CONTROLNET, STAGE_TEXT_DM, STAGE_JSCC, STAGE_END_TO_END_FT)


class ValImageLogger:
    """Config-driven periodic sample writer (rank-0 only).

    Parameters
    ----------
    cfg, stage, device:
        The active config / training stage / compute device.
    vae:
        A VAE module with ``.decode(latent) -> (sample, ...)`` used to map a
        predicted latent back to image space (the JSCC/diffusion shared VAE).
    output_dir:
        Base output directory — the caller passes ``checkpoint_dir``; panels go
        under ``<output_dir>/<output_subdir>``.
    val_loader:
        The validation DataLoader. The first batch is cached once and reused as
        the fixed sample set so progress is comparable across log points. When
        None, the logger falls back to the (training) batch handed to it.
    """

    def __init__(self, cfg, stage: str, device, vae, output_dir: str | Path,
                 val_loader=None) -> None:
        from omegaconf import OmegaConf
        self.cfg = cfg
        self.stage = stage
        self.device = device
        self.vae = vae
        self.val_loader = val_loader
        self._fixed_batch = None          # cached first val batch (lazy)
        self._fixed_batch_ready = False
        vi = OmegaConf.select(cfg, "train.val_images", default=None)

        def _get(key, default):
            return OmegaConf.select(vi, key, default=default) if vi is not None else default

        self.enabled       = bool(_get("enabled", False))
        self.every_steps   = int(_get("every_steps", 0) or 0)
        self.every_epochs  = int(_get("every_epochs", 0) or 0)
        self.num_samples   = max(1, int(_get("num_samples", 2)))
        subdir             = str(_get("output_subdir", "val_images"))
        self.out_dir       = Path(output_dir) / subdir

        # Only rank 0 ever writes; disable entirely off-rank so callers need no guard.
        if self.enabled and not ddp.is_rank0():
            self.enabled = False
        if self.enabled and stage not in _SUPPORTED:
            logger.info("val_images: stage '%s' has no sample visualiser — disabled.", stage)
            self.enabled = False
        if self.enabled and vae is None:
            logger.warning("val_images: no VAE available to decode latents — disabled.")
            self.enabled = False
        if self.enabled:
            logger.info("val_images: ON (stage=%s, every_steps=%d, every_epochs=%d, "
                        "num_samples=%d) → %s",
                        stage, self.every_steps, self.every_epochs, self.num_samples,
                        self.out_dir)

    # ── cadence ────────────────────────────────────────────────────────────────
    def should_log_step(self, global_step: int) -> bool:
        return (self.enabled and self.every_steps > 0
                and global_step > 0 and global_step % self.every_steps == 0)

    def should_log_epoch(self, epoch: int) -> bool:
        return (self.enabled and self.every_epochs > 0
                and epoch > 0 and epoch % self.every_epochs == 0)

    # ── entry points ────────────────────────────────────────────────────────────
    def maybe_log_step(self, runner, fallback_batch, global_step: int, epoch: int) -> None:
        if self.should_log_step(global_step):
            self._log(runner, fallback_batch, tag=f"step_{global_step:07d}")

    def maybe_log_epoch(self, runner, fallback_batch, epoch: int) -> None:
        if self.should_log_epoch(epoch):
            self._log(runner, fallback_batch, tag=f"epoch_{epoch:04d}")

    # ── validation batch (fixed, cached) ─────────────────────────────────────────
    def _val_batch(self, fallback_batch):
        """Return the fixed validation batch (first of val_loader), cached once.

        Falls back to *fallback_batch* (the training batch) only when no
        val_loader is available — that degraded case is logged once.
        """
        if self.val_loader is None:
            return fallback_batch
        if not self._fixed_batch_ready:
            self._fixed_batch_ready = True
            try:
                self._fixed_batch = next(iter(self.val_loader))
            except Exception as exc:      # empty / unreadable val loader
                logger.warning("val_images: could not read a validation batch (%s) — "
                               "falling back to the training batch.", exc)
                self._fixed_batch = None
        return self._fixed_batch if self._fixed_batch is not None else fallback_batch

    # ── core ─────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _log(self, runner, fallback_batch, *, tag: str) -> None:
        batch = self._val_batch(fallback_batch)
        if batch is None:
            return
        # Switch to EVAL mode for the whole reconstruction (train/eval-sensitive
        # behaviour such as dropout / CFG label-dropout must be OFF here), then
        # restore the runner's prior mode so the training loop is unaffected.
        was_training = bool(getattr(runner, "_training", True))
        runner.set_mode(False)
        try:
            panel = self._reconstruct(runner, batch)
        except Exception as exc:  # never let visualisation break training
            logger.warning("val_images: sample generation failed (%s) — skipping %s.",
                           exc, tag)
            return
        finally:
            runner.set_mode(was_training)
        if not panel:
            return
        self._save_panel(panel, tag)

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Latent → image in [0,1] via the shared VAE (mirrors the stage decode)."""
        out = self.vae.decode(latent)
        sample = out[0] if isinstance(out, (tuple, list)) else out
        return ((sample + 1) / 2).clamp(0, 1)

    def _reconstruct(self, runner, batch) -> Dict[str, torch.Tensor]:
        """Return image-space tensors {input, [edge], recon}, each [N,3,H,W] in [0,1]."""
        n = self.num_samples
        images = batch["image"][:n].to(self.device)

        # JSCC stage: the runner reconstructs directly through channel + VAE.
        if self.stage == STAGE_JSCC and hasattr(runner, "_reconstruct"):
            recon = runner._reconstruct(images).clamp(0, 1)
            return {"input": images.clamp(0, 1), "recon": recon}

        # Diffusion stages (text_dm / controlnet / end_to_end_ft): one-step f0
        # prediction through the runner's wired denoiser, decoded to image space.
        encode_latent = getattr(runner, "encode_latent_fn", None)
        encode_text = getattr(runner, "encode_text_fn", None)
        scheduler = getattr(runner, "scheduler", None)
        denoiser = getattr(runner, "denoiser", None) or getattr(runner, "_denoiser_core", None)
        if encode_latent is None or encode_text is None or scheduler is None or denoiser is None:
            logger.info("val_images: runner for stage '%s' lacks the callables needed "
                        "to visualise — skipping.", self.stage)
            return {}

        captions = batch.get("caption", [""] * images.shape[0])[:n]
        f0 = encode_latent(images)
        labels = encode_text(captions).to(self.device)

        panel: Dict[str, torch.Tensor] = {"input": images.clamp(0, 1)}
        c = None
        encode_edge = getattr(runner, "encode_edge_fn", None)
        if "edge" in batch and encode_edge is not None:
            edges = batch["edge"][:n].to(self.device)
            panel["edge"] = _to_rgb(edges).clamp(0, 1)
            c = encode_edge(edges)

        ft, noise_level, _noise, _t = scheduler.add_noise(f0)
        noise_level = noise_level.to(self.device)
        if c is not None:
            pred = denoiser(ft, noise_level, labels, c=c, enable_mask=False)
        else:
            pred = denoiser(ft, noise_level, labels, enable_mask=False)
        panel["recon"] = self._decode(pred)
        return panel

    def _save_panel(self, panel: Dict[str, torch.Tensor], tag: str) -> None:
        from torchvision.utils import make_grid, save_image
        # Column order: input, edge (if present), recon.
        cols: List[torch.Tensor] = [panel["input"]]
        if "edge" in panel:
            cols.append(panel["edge"])
        cols.append(panel["recon"])
        ncol = len(cols)
        n = cols[0].shape[0]
        # Interleave row-major: for each sample emit its columns back-to-back so
        # make_grid(nrow=ncol) lays out one sample per row.
        tiles: List[torch.Tensor] = []
        for i in range(n):
            for col in cols:
                tiles.append(col[i].detach().float().cpu())
        grid = make_grid(torch.stack(tiles, dim=0), nrow=ncol, padding=2)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{tag}.png"
        save_image(grid, str(path))
        logger.info("val_images: wrote %d-sample panel → %s", n, path)


def _to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Map an edge/condition tensor to a 3-channel image for display.

    Edge maps may be 1-, 2- or many-channel (MuGE). Take channel 0 (the mean-edge
    channel by convention) and broadcast to RGB.
    """
    if x.dim() == 4 and x.shape[1] == 3:
        return x
    if x.dim() == 4 and x.shape[1] >= 1:
        return x[:, :1].repeat(1, 3, 1, 1)
    return x

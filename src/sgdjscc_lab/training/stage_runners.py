"""training/stage_runners.py – Stage-aware forward + optimization for training.

Each runner expresses **one** training stage of the paper's 3-stage procedure
and owns its forward pass, loss wiring and optimizer step(s).  The epoch loop in
``pipelines/train_pipeline.py`` is therefore stage-agnostic: it just calls
``runner.training_step(batch)`` / ``runner.validation_step(batch)``.

  :class:`JSCCStageRunner`        stage 1 — VAE encode → AWGN(SNR=10) → decode,
                                  MSE (+ optional patch-GAN, two-optimizer step).
  :class:`TextDMStageRunner`      stage 2 — f0 = VAE latent; t~U(0,1); f_t via the
                                  sigmoid schedule; predict f0 on the masked and
                                  unmasked branches; f0-MSE on both.
  :class:`ControlNetStageRunner`  stage 3 — same DM objective with an edge-map
                                  condition fed to the (only-trainable) ControlNet
                                  branches; the base DM is frozen by the freeze
                                  policy.

The runners depend on small injected callables (``encode_latent_fn``,
``encode_text_fn``, ``denoiser`` …) so they can be unit-tested with stubs without
loading the multi-GB checkpoints.  :func:`build_stage_runner` wires the real
implementations from a ``ModelBundle``.

Algorithm-preservation note
---------------------------
The JSCC encode/decode here mirror ``pipelines/infer_pipeline.py`` exactly
(``_SCALING_FACTOR = 15.45``, ``x*2-1`` encode, L2-normalise, ``(decode+1)/2``)
so a JSCC model trained with this runner stays numerically compatible with the
inference path.  Do not "improve" these constants.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from sgdjscc_lab.training.noise_schedule import SigmoidNoiseScheduler
from sgdjscc_lab.training.losses import (
    DiffusionF0Loss,
    EndToEndFTLoss,
    GANLoss,
    JSCCStageLoss,
    build_discriminator,
    build_stage_loss,
)
from sgdjscc_lab.training.freeze import apply_stage_freeze_policy
from sgdjscc_lab.training.stages import (
    STAGE_CONTROLNET,
    STAGE_CSI_ESTIMATION,
    STAGE_EDGE_CODEC,
    STAGE_END_TO_END_FT,
    STAGE_JSCC,
    STAGE_TEXT_DM,
    resolve_dataset_type,
)

logger = logging.getLogger(__name__)

# Identical to pipelines/infer_pipeline.py — algorithm-preservation invariant.
_SCALING_FACTOR = 15.45


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class StageRunner:
    """Base class: optimizer creation + generic train/val steps.

    Supports gradient accumulation (``train.grad_accum_steps``) and mixed
    precision (``train.mixed_precision``, auto-disabled off-CUDA).  After each
    ``training_step`` the attribute ``last_step_did_update`` tells the caller
    whether an optimizer update actually fired (i.e. an accumulation window
    closed) so the pipeline can count *global optimizer steps*.
    """

    stage: str = "base"

    def __init__(self, cfg: DictConfig, device, param_groups: List[Dict]) -> None:
        self.cfg = cfg
        self.device = device
        lr = float(OmegaConf.select(cfg, "train.lr", default=1e-4))
        wd = float(OmegaConf.select(cfg, "train.weight_decay", default=1e-5))
        # Optional 8-bit AdamW (train.use_8bit_adam) with graceful fallback; the
        # default is a plain torch.optim.AdamW (unchanged behaviour).
        from sgdjscc_lab.training.perf import build_optimizer
        self.optimizer = build_optimizer(
            param_groups, cfg, lr=lr, weight_decay=wd, name="optimizer")
        self._init_step_controls()

    # Whether the runner is in a training step (set by set_mode). Gates effects
    # that must only apply during training, e.g. CFG label-dropout.
    _training: bool = True

    def _init_step_controls(self) -> None:
        cfg, device = self.cfg, self.device
        self.grad_accum = max(1, int(OmegaConf.select(cfg, "train.grad_accum_steps", default=1)))
        want_amp = bool(OmegaConf.select(cfg, "train.mixed_precision", default=False))
        self.use_amp = want_amp and torch.cuda.is_available() and "cuda" in str(device)
        if want_amp and not self.use_amp:
            logger.info("mixed_precision requested but CUDA unavailable → AMP disabled.")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self._accum = 0
        self.last_step_did_update = False
        # DDP-wrapped modules trained by this runner (empty for single-process).
        # Used for grad-accum no_sync and the epoch-boundary grad sync.
        self._ddp_modules: List = []

    def apply_perf_toggles(self) -> None:
        """Apply opt-in memory toggles (gradient checkpointing / xformers).

        Operates on this runner's trainable modules (``state_modules()``). Off by
        default; a requested-but-inapplicable toggle is logged, never silently
        dropped. Called once by :func:`build_stage_runner` after construction so
        both CLI and programmatic runs get the same behaviour.
        """
        from sgdjscc_lab.training.perf import apply_memory_optimizations
        apply_memory_optimizations(self.state_modules(), self.cfg, stage=self.stage)

    def register_ddp_modules(self, modules: List) -> None:
        """Record the DDP-wrapped trainable modules (for no_sync / flush sync)."""
        from sgdjscc_lab import distributed as _ddp
        self._ddp_modules = [m for m in modules
                             if m is not None and _ddp.is_distributed()
                             and hasattr(m, "no_sync")]

    def _ddp_find_unused(self) -> bool:
        """Whether the stage needs DDP find_unused_parameters (override per stage)."""
        return False

    def _autocast(self):
        return torch.cuda.amp.autocast(enabled=self.use_amp)

    # ── subclasses must implement ─────────────────────────────────────────────
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def state_modules(self) -> Dict[str, nn.Module]:
        """Modules whose state_dict should be checkpointed."""
        return {}

    # ── train/val ─────────────────────────────────────────────────────────────
    def set_mode(self, training: bool) -> None:
        self._training = bool(training)
        for m in self.state_modules().values():
            if hasattr(m, "train"):
                m.train(training)

    def _backward_and_maybe_step(self, loss: torch.Tensor) -> None:
        """Scale-backward with grad accumulation; step at window boundaries."""
        self.last_step_did_update = False
        if self.optimizer is None:
            return
        if not loss.requires_grad:
            raise RuntimeError(
                f"stage={self.stage}: loss has requires_grad=False — no gradient "
                "path to any trainable parameter. Check that the stage's module is "
                "unfrozen (freeze policy) and that real models are loaded "
                "(not --no-models)."
            )
        # DDP grad-accumulation: skip the gradient all-reduce on the NON-boundary
        # micro-steps (no_sync) so it only fires on the step that actually updates.
        will_step = (self._accum + 1) % self.grad_accum == 0
        if self._ddp_modules and not will_step:
            import contextlib
            with contextlib.ExitStack() as stack:
                for m in self._ddp_modules:
                    stack.enter_context(m.no_sync())
                self.scaler.scale(loss / self.grad_accum).backward()
        else:
            self.scaler.scale(loss / self.grad_accum).backward()
        self._accum += 1
        if will_step:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.last_step_did_update = True

    def training_step(self, batch: Dict) -> Dict[str, float]:
        self.set_mode(True)
        with self._autocast():
            out = self.forward(batch)
        self._backward_and_maybe_step(out["loss"])
        return _to_floats(out)

    @torch.no_grad()
    def validation_step(self, batch: Dict) -> Dict[str, float]:
        self.set_mode(False)
        with self._autocast():
            out = self.forward(batch)
        return _to_floats(out)

    # ── CFG null-token helpers (text-guided DM runners opt in) ───────────────
    def _setup_cfg_null(self, sample_labels: Optional[torch.Tensor] = None) -> None:
        """Init the CFG null mode. ``learned`` creates a trainable null token.

        DDP-safe design: when *sample_labels* is given (the DM runners pass a probe
        from ``encode_text_fn``) the token is created **eagerly** with the right
        shape, DDP-wrapped, registered in the optimizer, and recorded as a DDP
        module — so every rank has an identical, gradient-synced token (no rank-
        local lazy parameter). Without a probe it falls back to the lazy path
        (single-process only; the param is created on first use).
        """
        from sgdjscc_lab import distributed as _ddp
        self.cfg_null_mode = str(OmegaConf.select(
            self.cfg, "train.dm.cfg_null_mode", default="zero")).lower()
        self._null_core = None       # the unwrapped LearnedNullToken (for state_dict)
        self.null_module = None      # the (possibly DDP-wrapped) callable module
        if self.cfg_null_mode != "learned":
            return
        if sample_labels is None:
            # lazy fallback — NOT DDP-safe (single-process only).
            self._null_core = LearnedNullToken().to(self.device)
            self.null_module = self._null_core
            return
        core = LearnedNullToken(shape=(1, *tuple(sample_labels.shape[1:]))).to(self.device)
        self._null_core = core
        self.null_module = _ddp.maybe_wrap_ddp(core, find_unused_parameters=False)
        if self.optimizer is not None:
            self.optimizer.add_param_group(
                {"params": list(core.parameters()), "name": "cfg_null_token"})
        if _ddp.is_distributed() and hasattr(self.null_module, "no_sync"):
            self._ddp_modules.append(self.null_module)

    def _cfg_null_token(self, labels: torch.Tensor):
        """Null embedding tensor for CFG dropout (None → zero-vector path).

        Eager path: returns the token THROUGH the module's ``forward`` so a DDP
        wrapper's gradient-sync hook fires. Lazy fallback (single-process): create
        + register the token on first use.
        """
        core = getattr(self, "_null_core", None)
        if core is None:
            return None
        if core.token is None:                          # lazy fallback path
            core.materialize(labels)
            if self.optimizer is not None and not core._opt_registered:
                self.optimizer.add_param_group(
                    {"params": [core.token], "name": "cfg_null_token"})
                core._opt_registered = True
            return core.token
        # Pass labels as a scatter anchor so the DDP-wrapped null module's forward
        # works on GPU (device_ids scatter needs a positional input).
        return self.null_module(labels)                 # eager (DDP hook fires)

    def _cfg_null_state(self) -> Dict[str, nn.Module]:
        """Checkpoint the learned null token (UNWRAPPED; empty for zero mode)."""
        core = getattr(self, "_null_core", None)
        return {"cfg_null_token": core} if core is not None else {}

    # ── optimizers / scalers (subclasses override to add e.g. the GAN pair) ───
    def optimizers(self) -> Dict[str, object]:
        return {"optimizer": self.optimizer}

    def scalers(self) -> Dict[str, object]:
        return {"scaler": self.scaler}

    def _optimizer_scaler_pairs(self):
        """(optimizer, scaler) pairs to step together (base: the single pair)."""
        return [(self.optimizer, self.scaler)]

    def flush_pending(self) -> bool:
        """Apply any partially-accumulated gradients (grad_accum > 1 leftover).

        Called at epoch boundaries so the last micro-batches of an epoch are not
        silently dropped. No-op when the accumulation window is already closed.
        Returns True if an optimizer update was performed.
        """
        if self._accum % self.grad_accum == 0:
            return False  # window already flushed / nothing pending
        # The trailing micro-steps were accumulated under no_sync (rank-local), so
        # sync the grads across ranks BEFORE this forced step to keep ranks identical.
        if self._ddp_modules:
            from sgdjscc_lab import distributed as _ddp
            _ddp.all_reduce_grads(self._ddp_modules)
        stepped = False
        for opt, sc in self._optimizer_scaler_pairs():
            if opt is not None:
                sc.step(opt)
                sc.update()
                opt.zero_grad()
                stepped = True
        self._accum = 0
        self.last_step_did_update = stepped
        return stepped

    # ── full train-state checkpoint (modules + ALL optimizers/scalers + accum) ─
    def get_train_state(self) -> Dict:
        """Snapshot everything needed to resume *exactly*: module weights, every
        optimizer, every GradScaler, and the accumulation counter."""
        return {
            "modules": {n: m.state_dict() for n, m in self.state_modules().items()
                        if m is not None and hasattr(m, "state_dict")},
            "optimizers": {n: o.state_dict() for n, o in self.optimizers().items()
                           if o is not None},
            "scalers": {n: s.state_dict() for n, s in self.scalers().items()
                        if s is not None},
            "accum": int(self._accum),
        }

    def load_train_state(self, state: Dict) -> None:
        """Restore from :meth:`get_train_state` output (or a legacy checkpoint
        with top-level ``model_state`` / ``optimizer_state``)."""
        modules = state.get("modules")
        if modules is None:
            modules = state.get("model_state", {})  # legacy
        targets = self.state_modules()
        for name, sd in (modules or {}).items():
            m = targets.get(name)
            if m is not None and hasattr(m, "load_state_dict"):
                try:
                    m.load_state_dict(sd, strict=False)
                    logger.info("  Restored module %s", name)
                except Exception as exc:  # pragma: no cover
                    logger.warning("  Skipped module %s: %s", name, exc)

        opts = state.get("optimizers")
        if opts is None and state.get("optimizer_state"):
            opts = {"optimizer": state["optimizer_state"]}  # legacy
        my_opts = self.optimizers()
        for name, sd in (opts or {}).items():
            o = my_opts.get(name)
            if o is not None:
                try:
                    o.load_state_dict(sd)
                    logger.info("  Restored optimizer %s", name)
                except Exception as exc:  # pragma: no cover
                    logger.warning("  Could not restore optimizer %s: %s", name, exc)

        my_scalers = self.scalers()
        for name, sd in (state.get("scalers") or {}).items():
            s = my_scalers.get(name)
            if s is not None and sd:
                try:
                    s.load_state_dict(sd)
                except Exception as exc:  # pragma: no cover
                    logger.warning("  Could not restore scaler %s: %s", name, exc)

        if "accum" in state:
            self._accum = int(state["accum"])

    def optimizer_state(self) -> Dict:
        """Backward-compatible single-optimizer state (superseded by
        :meth:`get_train_state`)."""
        return self.optimizer.state_dict() if self.optimizer is not None else {}


def _to_floats(out: Dict) -> Dict[str, float]:
    return {k: float(v.detach()) for k, v in out.items() if torch.is_tensor(v)}


class LearnedNullToken(nn.Module):
    """Trainable CFG null token used by ``cfg_null_mode='learned'``.

    Replaces the dropped text condition with a *trainable* unconditional token
    (closer to the conditional/unconditional branch the paper relies on for CFG
    scale 4.5) instead of the zero vector. ``forward()`` returns the token so it
    can be called THROUGH a DDP wrapper (the gradient-sync hook fires).

    Construction
    ------------
    * ``LearnedNullToken(shape=(1, D))`` — **eager** (DDP-safe): the parameter
      exists at construction, so the module can be DDP-wrapped and added to the
      optimizer deterministically across ranks. Used by the DM runners.
    * ``LearnedNullToken()`` — **lazy** (single-process fallback): created on the
      first ``materialize`` call. Kept for backward compatibility / non-DDP use.
    """

    def __init__(self, shape=None) -> None:
        super().__init__()
        self.token: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(tuple(shape))) if shape is not None else None)
        # Whether the token has been added to the runner's optimizer yet (lazy path).
        self._opt_registered: bool = False

    def materialize(self, like: torch.Tensor) -> bool:
        """Create the parameter to match ``like``'s feature shape. True if created."""
        if self.token is None:
            self.token = nn.Parameter(
                torch.zeros((1, *like.shape[1:]), dtype=like.dtype, device=like.device))
            return True
        return False

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Materialise ``token`` from a checkpoint BEFORE copying values in.

        The token is created lazily on the first forward, but resume restores
        state *before* any forward — so without this, a saved ``token`` would be
        dropped and re-initialised to zeros. Here we create the Parameter with
        the checkpointed shape so ``load_state_dict`` copies the saved values.
        """
        key = prefix + "token"
        if self.token is None and key in state_dict:
            self.token = nn.Parameter(torch.zeros_like(state_dict[key]))
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Accepts (and ignores) a dummy input so the module can be called THROUGH
        # a DDP wrapper: GPU DDP scatters the positional input across device_ids,
        # and a no-input forward would raise IndexError on the empty scatter. The
        # caller passes ``labels`` purely as that scatter anchor.
        return self.token


def apply_cfg_label_dropout(
    labels: torch.Tensor,
    prob: float,
    training: bool,
    null_token: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Classifier-free guidance (CFG) label-dropout for the text-guided DM.

    CFG inference (the paper uses a CFG scale of 4.5, Table III) requires the DM
    to have been trained to *also* denoise without the conditioning, by randomly
    replacing the text condition with a NULL embedding during training. This
    drops each sample's label embedding to the null embedding with probability
    *prob* (only while *training*).

    null embedding (``train.dm.cfg_null_mode``)
    -------------------------------------------
    * ``null_token is None`` → **zero** vector (``cfg_null_mode=zero``): simple,
      common, paper-LIKE (the paper does not publish its null token).
    * ``null_token`` given → a **learned** unconditional token
      (``cfg_null_mode=learned``): trainable, closer to a true unconditional
      branch — the paper-mode default.

    The paper does not state the dropout probability; we expose it config-driven
    (``train.dm.cfg_dropout_prob``, default 0.1, the PixArt convention).
    """
    if prob <= 0.0 or not training:
        return labels
    keep = (torch.rand(labels.shape[0], device=labels.device) >= float(prob))
    view = [labels.shape[0]] + [1] * (labels.dim() - 1)
    keep_f = keep.view(view).to(labels.dtype)
    if null_token is None:
        return labels * keep_f                                   # zero null (paper-like)
    null = null_token.to(device=labels.device, dtype=labels.dtype)
    return labels * keep_f + null * (1.0 - keep_f)               # learned null token


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — JSCC
# ─────────────────────────────────────────────────────────────────────────────

class JSCCStageRunner(StageRunner):
    """Stage 1: train the JSCC autoencoder under a fixed AWGN SNR.

    ``forward`` reconstructs the image through VAE-encode → channel → VAE-decode
    and applies :class:`JSCCStageLoss`.  When the GAN term is enabled, a separate
    discriminator optimizer is updated before the generator step.
    """

    stage = STAGE_JSCC

    def __init__(
        self,
        jscc,
        cfg: DictConfig,
        device,
        param_groups: List[Dict],
        loss: Optional[JSCCStageLoss] = None,
        discriminator: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(cfg, device, param_groups)
        self.jscc = jscc
        self.snr_db = float(OmegaConf.select(cfg, "train.jscc.snr_db", default=10.0))
        self.loss = loss if loss is not None else build_stage_loss(cfg, STAGE_JSCC)

        gan_enabled = bool(OmegaConf.select(cfg, "train.jscc.gan.enabled", default=False))
        self.gan_weight = self.loss.gan_weight if isinstance(self.loss, JSCCStageLoss) else 0.0
        self.disc = None
        self.d_optimizer = None
        self.d_scaler = None
        if gan_enabled:
            self.disc = (discriminator if discriminator is not None
                         else build_discriminator(cfg).to(device))
            self.gan = GANLoss(str(OmegaConf.select(cfg, "train.jscc.gan.mode", default="hinge")))
            d_lr = float(OmegaConf.select(cfg, "train.jscc.gan.lr", default=1e-4))
            self.d_optimizer = torch.optim.AdamW(self.disc.parameters(), lr=d_lr)
            self.d_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
            logger.info("Stage-1 GAN enabled (patch discriminator, w=%.3f, lr=%.2e)",
                        self.gan_weight, d_lr)

    # ── JSCC encode/decode (mirrors infer_pipeline) ───────────────────────────
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        latent_dist = self.jscc.vae.encode(x * 2 - 1).latent_dist
        return self.jscc.normalize(latent_dist.mean / _SCALING_FACTOR)

    def _reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        self.jscc.snr = self.snr_db                       # fixed-SNR training
        f0 = self._encode(x)
        hat = self.jscc.normalize(self.jscc.channel(f0))
        return (self.jscc.vae.decode(self.jscc.normalize(hat))[0] + 1) / 2

    def state_modules(self) -> Dict[str, nn.Module]:
        mods = {"jscc_model": self.jscc}
        if self.disc is not None:
            mods["jscc_discriminator"] = self.disc
        return mods

    def optimizers(self) -> Dict[str, object]:
        opts = {"optimizer": self.optimizer}
        if self.d_optimizer is not None:
            opts["d_optimizer"] = self.d_optimizer
        return opts

    def scalers(self) -> Dict[str, object]:
        scs = {"scaler": self.scaler}
        if self.d_scaler is not None:
            scs["d_scaler"] = self.d_scaler
        return scs

    def _optimizer_scaler_pairs(self):
        pairs = [(self.optimizer, self.scaler)]
        if self.d_optimizer is not None:
            pairs.append((self.d_optimizer, self.d_scaler))
        return pairs

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        x = batch["image"].to(self.device)
        recon = self._reconstruct(x)
        return self.loss(recon, x, disc=self.disc)

    def training_step(self, batch: Dict) -> Dict[str, float]:
        # Non-GAN path: reuse the base step (AMP + grad-accum handled there).
        if self.disc is None or self.gan_weight <= 0 or self.d_optimizer is None:
            return super().training_step(batch)

        # GAN path: alternating D / G updates with the same AMP + grad-accum
        # window as the generator. logs loss_mse, loss_gan, loss_disc.
        self.set_mode(True)
        x = batch["image"].to(self.device)
        with self._autocast():
            recon = self._reconstruct(x)

        out: Dict[str, torch.Tensor] = {}

        # ── Discriminator (real vs detached fake) ─────────────────────────────
        with self._autocast():
            d_loss = self.gan.discriminator_loss(
                self.disc(x * 2 - 1), self.disc(recon.detach() * 2 - 1)
            )
        out["loss_disc"] = d_loss
        self.d_scaler.scale(d_loss / self.grad_accum).backward()

        # ── Generator (MSE + GAN) ─────────────────────────────────────────────
        with self._autocast():
            g_out = self.loss(recon, x, disc=self.disc)
        out.update(g_out)
        self.last_step_did_update = False
        if self.optimizer is not None:
            self.scaler.scale(g_out["loss"] / self.grad_accum).backward()
            self._accum += 1
            if self._accum % self.grad_accum == 0:
                self.d_scaler.step(self.d_optimizer)
                self.d_scaler.update()
                self.d_optimizer.zero_grad()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.last_step_did_update = True
        return _to_floats(out)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — text-guided DM
# ─────────────────────────────────────────────────────────────────────────────

class TextDMStageRunner(StageRunner):
    """Stage 2: train the text-guided DM to predict f0 (masked + unmasked).

    The denoiser callable must accept
    ``denoiser(ft, noise_level, labels, enable_mask=bool)`` and return a tensor
    shaped like ``f0`` (MDTv2's signature).  ``encode_latent_fn`` maps an image
    batch → clean latent ``f0``; ``encode_text_fn`` maps captions → label
    embeddings on ``device``.
    """

    stage = STAGE_TEXT_DM

    def __init__(
        self,
        denoiser: nn.Module,
        encode_latent_fn: Callable[[torch.Tensor], torch.Tensor],
        encode_text_fn: Callable[[List[str]], torch.Tensor],
        cfg: DictConfig,
        device,
        param_groups: List[Dict],
        scheduler: Optional[SigmoidNoiseScheduler] = None,
        loss: Optional[DiffusionF0Loss] = None,
    ) -> None:
        super().__init__(cfg, device, param_groups)
        # DDP: wrap the trainable denoiser; the runner calls the WRAPPED module so
        # the gradient-sync hooks fire, and keeps the unwrapped core for clean
        # state_dict keys (checkpoints stay compatible with single-process runs).
        # Disable DDP buffer broadcasts here: Stage-2 does two forwards through
        # the same denoiser per step (unmasked + masked), and rebroadcasting the
        # denoiser's fixed relative-position buffers between those forwards trips
        # autograd's version counter on backward.
        from sgdjscc_lab import distributed as _ddp
        self._denoiser_core = denoiser
        self.denoiser = _ddp.maybe_wrap_ddp(
            denoiser,
            find_unused_parameters=self._ddp_find_unused(),
            broadcast_buffers=False,
        )
        self.encode_latent_fn = encode_latent_fn
        self.encode_text_fn = encode_text_fn
        self.scheduler = scheduler if scheduler is not None else _build_scheduler(cfg)
        self.loss = loss if loss is not None else build_stage_loss(cfg, self.stage)
        self.use_masked = bool(OmegaConf.select(cfg, "train.dm.use_masked_branch", default=True))
        self.cfg_dropout_prob = float(OmegaConf.select(cfg, "train.dm.cfg_dropout_prob", default=0.0))
        # Probe the label-embedding shape so the learned CFG null token can be
        # created EAGERLY (DDP-safe) instead of lazily on first forward.
        probe = None
        if str(OmegaConf.select(cfg, "train.dm.cfg_null_mode", default="zero")).lower() == "learned":
            with torch.no_grad():
                probe = self.encode_text_fn([""]).to(self.device)
        self._setup_cfg_null(probe)
        self.register_ddp_modules([self.denoiser, self.null_module])

    def state_modules(self) -> Dict[str, nn.Module]:
        return {"diffusion": self._denoiser_core, **self._cfg_null_state()}

    def _denoise(self, ft, noise_level, labels, *, enable_mask: bool) -> torch.Tensor:
        return self.denoiser(ft, noise_level, labels, enable_mask=enable_mask)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        images = batch["image"].to(self.device)
        captions = batch["caption"]

        # f0 (clean latent) and label embeddings come from frozen modules (VAE,
        # CLIP). Compute both under no_grad + detach so neither the VAE nor the
        # CLIP text encoder joins the autograd graph (no wasted grad/memory).
        with torch.no_grad():
            f0 = self.encode_latent_fn(images).detach()
            labels = self.encode_text_fn(captions).to(self.device).detach()
        # CFG: drop labels to the null embedding (training only) so the DM learns
        # the unconditional branch needed for classifier-free guidance at inference.
        labels = apply_cfg_label_dropout(labels, self.cfg_dropout_prob,
                                         self._training, self._cfg_null_token(labels))

        ft, noise_level, _noise, _t = self.scheduler.add_noise(f0)
        noise_level = noise_level.to(self.device)

        pred_unmasked = self._denoise(ft, noise_level, labels, enable_mask=False)
        pred_masked = (
            self._denoise(ft, noise_level, labels, enable_mask=True)
            if self.use_masked else None
        )
        return self.loss(f0, pred_unmasked, pred_masked)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — edge-guided ControlNet
# ─────────────────────────────────────────────────────────────────────────────

class ControlNetStageRunner(StageRunner):
    """Stage 3: train only the ControlNet branches with edge-map guidance.

    The base text-guided DM is frozen by the freeze policy; this runner feeds an
    encoded edge condition ``c`` through the ControlNet path
    (``denoiser(ft, noise_level, labels, c=c, enable_mask=bool)``).

    Edge-conditioning caveat: the paper transmits a Canny edge map via a
    dedicated edge-JSCC and aligns it to the latent space.  Here the edge map is
    encoded into the shared latent space via ``encode_edge_fn`` (a structural
    stand-in).  See docs/training_scaffold.md.
    """

    stage = STAGE_CONTROLNET

    def __init__(
        self,
        denoiser: nn.Module,
        encode_latent_fn: Callable[[torch.Tensor], torch.Tensor],
        encode_text_fn: Callable[[List[str]], torch.Tensor],
        encode_edge_fn: Callable[[torch.Tensor], torch.Tensor],
        cfg: DictConfig,
        device,
        param_groups: List[Dict],
        scheduler: Optional[SigmoidNoiseScheduler] = None,
        loss: Optional[DiffusionF0Loss] = None,
    ) -> None:
        super().__init__(cfg, device, param_groups)
        # DDP: wrap the denoiser (only the ControlNet branches train; see
        # _ddp_find_unused). edge_transport stays UNWRAPPED — it is fixed side
        # info computed under no_grad, so it needs no gradient sync. Disable DDP
        # buffer broadcasts here for the same reason as Stage-2: the denoiser can
        # be called multiple times per step, and its registered fixed buffers do
        # not need per-forward rebroadcasts.
        from sgdjscc_lab import distributed as _ddp
        self._denoiser_core = denoiser
        self.denoiser = _ddp.maybe_wrap_ddp(
            denoiser,
            find_unused_parameters=self._ddp_find_unused(),
            broadcast_buffers=False,
        )
        self.encode_latent_fn = encode_latent_fn
        self.encode_text_fn = encode_text_fn
        self.encode_edge_fn = encode_edge_fn
        # When edge_transport=edge_jscc the callable carries its dedicated codec
        # module — checkpoint it so a resumed run reproduces the same `c`.
        self.edge_module = getattr(encode_edge_fn, "module", None)
        self.scheduler = scheduler if scheduler is not None else _build_scheduler(cfg)
        self.loss = loss if loss is not None else build_stage_loss(cfg, self.stage)
        self.use_masked = bool(OmegaConf.select(cfg, "train.dm.use_masked_branch", default=True))
        self.cfg_dropout_prob = float(OmegaConf.select(cfg, "train.dm.cfg_dropout_prob", default=0.0))
        probe = None
        if str(OmegaConf.select(cfg, "train.dm.cfg_null_mode", default="zero")).lower() == "learned":
            with torch.no_grad():
                probe = self.encode_text_fn([""]).to(self.device)
        self._setup_cfg_null(probe)
        self.register_ddp_modules([self.denoiser, self.null_module])

    def _ddp_find_unused(self) -> bool:
        # Stage 3 freezes the base DM and trains only the ControlNet branches, so
        # some WRAPPED-module parameters may not receive a gradient on a given
        # step (only the structural branches do). DDP needs find_unused_parameters
        # to handle that without hanging. Enabled CONSERVATIVELY (correctness >
        # the small overhead); see docs/paper_gap_closure.md "DDP" for the honest
        # rationale — it can be set False if profiling shows all trainable params
        # always receive grad.
        return True

    def state_modules(self) -> Dict[str, nn.Module]:
        mods = {"diffusion": self._denoiser_core, **self._cfg_null_state()}
        if self.edge_module is not None:
            mods["edge_jscc"] = self.edge_module
        return mods

    def _denoise(self, ft, noise_level, labels, c, *, enable_mask: bool) -> torch.Tensor:
        return self.denoiser(ft, noise_level, labels, c=c, enable_mask=enable_mask)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        images = batch["image"].to(self.device)
        edges = batch["edge"].to(self.device)
        captions = batch["caption"]

        # f0, edge condition and labels all come from frozen modules — compute
        # under no_grad + detach to keep them out of the autograd graph.
        with torch.no_grad():
            f0 = self.encode_latent_fn(images).detach()
            c = self.encode_edge_fn(edges).detach()
            labels = self.encode_text_fn(captions).to(self.device).detach()
        # CFG text-label dropout (training only). The edge condition `c` is kept;
        # only the text label is dropped, matching the paper's text-CFG scale.
        labels = apply_cfg_label_dropout(labels, self.cfg_dropout_prob,
                                         self._training, self._cfg_null_token(labels))

        ft, noise_level, _n, _t = self.scheduler.add_noise(f0)
        noise_level = noise_level.to(self.device)

        pred_unmasked = self._denoise(ft, noise_level, labels, c, enable_mask=False)
        pred_masked = (
            self._denoise(ft, noise_level, labels, c, enable_mask=True)
            if self.use_masked else None
        )
        return self.loss(f0, pred_unmasked, pred_masked)


# ─────────────────────────────────────────────────────────────────────────────
# Extension stage — end-to-end JSCC↔DM fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def _noise_level_from_snr(snr_db: float, batch: int, device) -> torch.Tensor:
    """Map a channel SNR to a DM noise level ``sqrt(noise_var)`` ``[B, 1]``.

    Uses the variance-preserving relation ``noise_var = 1/(1+snr_lin)`` so a
    higher SNR maps to a smaller noise level (weaker denoising) — the same
    monotonic intuition as the inference step-matching.
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_var = 1.0 / (1.0 + snr_lin)
    nl = torch.full((batch, 1), float(noise_var) ** 0.5, device=device)
    return nl


class EndToEndFTStageRunner(StageRunner):
    """Extension stage: joint end-to-end fine-tuning of JSCC + DM.

    Paper appendix note: the authors describe a sequential DM-then-JSCC-decoder
    fine-tuning that shifts the objective from noise removal to minimising the
    final image distortion.  This runner is a **tractable structural** version of
    that idea — one forward wires the full path and optimises both objectives
    jointly:

      x → encode(x)=f0 → channel(SNR) → 1-step DM denoise → decode → x̂   (recon)
      f0(detached) → f_t → DM predict f0                                  (diff)

      loss = recon_weight·‖x−x̂‖² + diff_weight·‖f0−ε(f_t)‖²

    Differences from the paper are documented in docs/training_scaffold.md
    (single-step denoise instead of the full reverse process; joint rather than
    strictly sequential fine-tuning).
    """

    stage = STAGE_END_TO_END_FT

    def __init__(
        self,
        jscc,
        denoiser: nn.Module,
        encode_text_fn: Callable[[List[str]], torch.Tensor],
        cfg: DictConfig,
        device,
        param_groups: List[Dict],
        edge_transport: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        scheduler: Optional[SigmoidNoiseScheduler] = None,
        loss: Optional[EndToEndFTLoss] = None,
    ) -> None:
        super().__init__(cfg, device, param_groups)
        self.jscc = jscc
        self.denoiser = denoiser
        self.encode_text_fn = encode_text_fn
        self.edge_transport = edge_transport
        self.edge_module = getattr(edge_transport, "module", None)
        self.scheduler = scheduler if scheduler is not None else _build_scheduler(cfg)
        self.loss = loss if loss is not None else build_stage_loss(cfg, self.stage)
        self.snr_db = float(OmegaConf.select(cfg, "train.end_to_end_ft.snr_db", default=10.0))

    def state_modules(self) -> Dict[str, nn.Module]:
        mods = {"jscc_model": self.jscc, "diffusion": self.denoiser}
        if self.edge_module is not None:
            mods["edge_jscc"] = self.edge_module
        return mods

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        latent_dist = self.jscc.vae.encode(x * 2 - 1).latent_dist
        return self.jscc.normalize(latent_dist.mean / _SCALING_FACTOR)

    def _denoise(self, latent, noise_level, labels, c, *, enable_mask=False):
        if c is not None:
            return self.denoiser(latent, noise_level, labels, c=c, enable_mask=enable_mask)
        return self.denoiser(latent, noise_level, labels, enable_mask=enable_mask)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        x = batch["image"].to(self.device)
        captions = batch["caption"]

        # frozen side information (labels, edge condition)
        with torch.no_grad():
            labels = self.encode_text_fn(captions).to(self.device).detach()
            c = None
            if self.edge_transport is not None and "edge" in batch:
                c = self.edge_transport(batch["edge"].to(self.device)).detach()

        # ── recon branch (full path, grad to JSCC encoder/decoder + DM) ───────
        f0 = self._encode(x)
        self.jscc.snr = self.snr_db
        hat = self.jscc.normalize(self.jscc.channel(f0))
        nl_chan = _noise_level_from_snr(self.snr_db, x.shape[0], self.device)
        f0_pred = self._denoise(hat, nl_chan, labels, c, enable_mask=False)
        recon = (self.jscc.vae.decode(self.jscc.normalize(f0_pred))[0] + 1) / 2

        # ── diffusion branch (preserve DM denoising; f0 target detached) ──────
        ft, nl_t, _n, _t = self.scheduler.add_noise(f0.detach())
        nl_t = nl_t.to(self.device)
        pred_f0 = self._denoise(ft, nl_t, labels, c, enable_mask=False)

        return self.loss(recon, x, f0.detach(), pred_f0)


# ─────────────────────────────────────────────────────────────────────────────
# Supporting stage — dedicated edge codec (BCE + Dice)
# ─────────────────────────────────────────────────────────────────────────────

class EdgeCodecStageRunner(StageRunner):
    """Train the dedicated edge JSCC as a real edge-reconstruction codec.

    This is the supporting stage behind Stage-3's ``edge_jscc`` transport: it
    trains :class:`~sgdjscc_lab.models.edge_jscc.EdgeJSCC` (encoder → channel →
    projector → decoder) self-supervised on edge maps with a BCE + Dice loss
    (:class:`~sgdjscc_lab.training.losses.EdgeCodecLoss`):

        edge ─▶ encode_latent (the Stage-3 condition latent ``c``) ─▶ decode ─▶ ê
        loss = BCE(ê_logits, edge) + Dice(σ(ê_logits), edge)

    The trained checkpoint is then loaded by ``build_edge_transport`` so Stage 3
    conditions on a *trained* edge codec rather than a random stand-in.  The
    codec is self-contained (it owns its AWGN channel), so this stage needs no
    pretrained JSCC/diffusion checkpoints.
    """

    stage = STAGE_EDGE_CODEC

    def __init__(self, edge_codec, cfg: DictConfig, device, param_groups, loss=None) -> None:
        super().__init__(cfg, device, param_groups)
        self.edge_codec = edge_codec
        self.loss = loss if loss is not None else build_stage_loss(cfg, STAGE_EDGE_CODEC)
        # Multi-SNR training: sample the edge-link SNR per step from
        # [min,max] dB so the SNR conditioning is actually exercised (otherwise
        # the codec trains at a single fixed SNR and the conditioning is a
        # constant). Paper-like: WITT-style codecs train across SNRs.
        ms = OmegaConf.select(cfg, "train.edge_codec.multi_snr", default=None)
        self.multi_snr = bool(OmegaConf.select(ms, "enabled", default=False)) if ms else False
        self.snr_min = float(OmegaConf.select(ms, "min_db", default=0.0)) if ms else 0.0
        self.snr_max = float(OmegaConf.select(ms, "max_db", default=20.0)) if ms else 20.0

    def state_modules(self) -> Dict[str, nn.Module]:
        return {"edge_jscc": self.edge_codec}

    def _sample_snr(self) -> Optional[float]:
        """Uniform SNR in [min,max] dB for multi-SNR training, else None (fixed)."""
        if not (self.multi_snr and self._training):
            return None
        return float(self.snr_min + (self.snr_max - self.snr_min) * torch.rand(()).item())

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        edge = batch["edge"].to(self.device)
        logits = self.edge_codec.reconstruct(edge, snr_db=self._sample_snr())
        return self.loss(logits, edge)


# ─────────────────────────────────────────────────────────────────────────────
# Supporting stage — blind SNR estimator (paper Sec. IV-C, eq. 15)
# ─────────────────────────────────────────────────────────────────────────────

class CSIEstimationStageRunner(StageRunner):
    """Train the blind SNR estimator self-supervised on image latents.

    For each batch: ``f0 = VAE(image)`` (frozen), sample the signal level
    ``α ~ U(0,1)``, form ``f̄ = √α·f0 + √(1-α)·n`` (paper eq. 12 form) and regress a
    target with :class:`~sgdjscc_lab.models.csi_estimation.SNREstimator`.

    Target semantics (``train.csi_estimation.target``)
    --------------------------------------------------
    The inference blind step-matching path computes
    ``signal_scale = jscc.snr_prediction_net(f̄) ** 2`` (inference_one.py:102), i.e.
    the public predictor outputs the signal **amplitude** ``√α`` whose square is the
    signal level ``α``.

    * ``"amplitude"`` (default) — regress ``√α``, so the trained net is a runtime
      **drop-in** for ``jscc.snr_prediction_net`` (loaded via
      ``models/csi_estimation.py::load_snr_estimator_into``).
    * ``"alpha"`` — regress ``α`` literally (paper eq. 15); NOT directly consumable
      by the squaring runtime (would need a √ at load time).

    Fidelity: the SNR estimator is **paper-like** (mirrors the public
    ``Prediction_Model``). The phase estimator / joint loop are scaffolds and are
    NOT trained here (no complex phase in the real-gain latent path).
    """

    stage = STAGE_CSI_ESTIMATION

    def __init__(self, encode_latent_fn, snr_estimator, cfg, device, param_groups, loss=None):
        super().__init__(cfg, device, param_groups)
        self.encode_latent_fn = encode_latent_fn
        self.snr_estimator = snr_estimator
        self.target = str(OmegaConf.select(
            cfg, "train.csi_estimation.target", default="amplitude")).lower()
        self.loss = loss if loss is not None else build_stage_loss(cfg, STAGE_CSI_ESTIMATION)

    def state_modules(self) -> Dict[str, nn.Module]:
        return {"snr_estimator": self.snr_estimator}

    def get_train_state(self) -> Dict:
        # Record the regression target so the inference loader knows whether the
        # net outputs √α (amplitude, runtime drop-in) or α (paper eq. 15) and can
        # adapt accordingly — prevents loading an α-target net into the net²=α path.
        state = super().get_train_state()
        state["meta"] = {"csi_target": self.target}
        return state

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        from sgdjscc_lab.training.losses import synthesize_noisy_latent
        images = batch["image"].to(self.device)
        with torch.no_grad():
            f0 = self.encode_latent_fn(images).detach()
        alpha = torch.rand(f0.shape[0], 1, device=self.device, dtype=f0.dtype)
        noisy = synthesize_noisy_latent(f0, alpha)
        # default: regress √α (amplitude) so net² = α matches the inference runtime.
        target = torch.sqrt(alpha) if self.target == "amplitude" else alpha
        return self.loss(self.snr_estimator(noisy), target)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_scheduler(cfg: DictConfig) -> SigmoidNoiseScheduler:
    sch = OmegaConf.select(cfg, "train.dm.scheduler", default=None)
    if sch is None:
        return SigmoidNoiseScheduler()
    return SigmoidNoiseScheduler(
        start=float(OmegaConf.select(sch, "start", default=0.0)),
        end=float(OmegaConf.select(sch, "end", default=3.0)),
        tau=float(OmegaConf.select(sch, "tau", default=0.7)),
    )


def _jscc_latent_encoder(jscc) -> Callable[[torch.Tensor], torch.Tensor]:
    def _encode(x: torch.Tensor) -> torch.Tensor:
        latent_dist = jscc.vae.encode(x * 2 - 1).latent_dist
        return jscc.normalize(latent_dist.mean / _SCALING_FACTOR)
    return _encode


def _clip_text_encoder(sem_pipeline, device) -> Callable[[List[str]], torch.Tensor]:
    model_dtype = next(sem_pipeline.model.parameters()).dtype

    def _encode(captions: List[str]) -> torch.Tensor:
        labels = sem_pipeline.encode_text(captions, sem_pipeline.text_embed)
        # CLIP may emit fp16 embeddings on CUDA while the training denoiser stays
        # in fp32. Align the label dtype to the denoiser so label_proj never sees
        # Half x Float and stage-2/3 training matches the inference wrapper.
        return labels.to(device=device, dtype=model_dtype)

    return _encode


def _edge_latent_encoder(jscc) -> Callable[[torch.Tensor], torch.Tensor]:
    """Encode a 1-channel edge map into the shared latent space.

    Structural stand-in for the paper's transmitted-edge conditioning: the edge
    map is broadcast to 3 channels and pushed through the (frozen) VAE encoder so
    it lands in the same latent geometry as ``f0``.
    """
    base = _jscc_latent_encoder(jscc)

    def _encode(edge: torch.Tensor) -> torch.Tensor:
        if edge.shape[1] == 1:
            edge = edge.repeat(1, 3, 1, 1)
        return base(edge)
    return _encode


def build_stage_runner(stage: str, models, cfg: DictConfig, device) -> StageRunner:
    """Build the runner for *stage*, applying the freeze policy and wiring models.

    Raises a clear error if a module required by the stage is missing from the
    bundle (e.g. ControlNet stage without a loaded diffusion pipeline). After
    construction the opt-in memory toggles (gradient checkpointing / xformers)
    are applied to the runner's trainable modules (no-op unless enabled in cfg).
    """
    runner = _build_stage_runner_impl(stage, models, cfg, device)
    runner.apply_perf_toggles()
    return runner


def _build_stage_runner_impl(stage: str, models, cfg: DictConfig, device) -> StageRunner:
    param_groups, report = apply_stage_freeze_policy(models, cfg, stage)

    jscc = getattr(models, "jscc_model", None) if models is not None else None
    sem = getattr(models, "sem_pipeline", None) if models is not None else None
    denoiser = getattr(sem, "model", None) if sem is not None else None

    if stage == STAGE_EDGE_CODEC:
        # Self-contained: build a fresh trainable edge codec (with decoder head)
        # and its own AWGN channel — no pretrained JSCC/diffusion needed.
        from sgdjscc_lab.training.edge_transport import build_edge_codec
        edge_codec = build_edge_codec(cfg, device)
        cg = [p for p in edge_codec.parameters() if p.requires_grad]
        param_groups = param_groups + ([{"params": cg, "name": "edge_jscc"}] if cg else [])
        return EdgeCodecStageRunner(edge_codec, cfg, device, param_groups)

    if stage == STAGE_CSI_ESTIMATION:
        if jscc is None:
            raise RuntimeError(
                "stage='csi_estimation' needs the JSCC VAE for image latents "
                "(use real models, not --no-models).")
        from sgdjscc_lab.models.csi_estimation import SNREstimator
        latent_ch = int(OmegaConf.select(cfg, "train.csi_estimation.latent_ch", default=16))
        snr_est = SNREstimator(latent_ch).to(device)
        cg = [p for p in snr_est.parameters() if p.requires_grad]
        param_groups = param_groups + ([{"params": cg, "name": "snr_estimator"}] if cg else [])
        return CSIEstimationStageRunner(
            _jscc_latent_encoder(jscc), snr_est, cfg, device, param_groups)

    if stage == STAGE_JSCC:
        if jscc is None:
            raise RuntimeError(
                "stage='jscc' needs a JSCC model; none is loaded (use real models, "
                "not --no-models)."
            )
        return JSCCStageRunner(jscc, cfg, device, param_groups)

    if stage == STAGE_TEXT_DM:
        if denoiser is None or jscc is None:
            raise RuntimeError(
                "stage='text_dm' needs both the JSCC VAE (for latents) and the "
                "diffusion denoiser; one is missing (use_semantic:true, real models)."
            )
        return TextDMStageRunner(
            denoiser,
            _jscc_latent_encoder(jscc),
            _clip_text_encoder(sem, device),
            cfg, device, param_groups,
        )

    if stage == STAGE_CONTROLNET:
        if denoiser is None or jscc is None:
            raise RuntimeError(
                "stage='controlnet' needs the diffusion denoiser and JSCC VAE "
                "(use_semantic:true, use_controlnet:true, real models)."
            )
        from sgdjscc_lab.training.edge_transport import build_edge_transport
        return ControlNetStageRunner(
            denoiser,
            _jscc_latent_encoder(jscc),
            _clip_text_encoder(sem, device),
            build_edge_transport(cfg, jscc, device),
            cfg, device, param_groups,
        )

    if stage == STAGE_END_TO_END_FT:
        if denoiser is None or jscc is None:
            raise RuntimeError(
                "stage='end_to_end_ft' needs both the JSCC model and the diffusion "
                "denoiser (use_semantic:true, real models)."
            )
        # Build an edge transport only when the dataset carries edges.
        edge_transport = None
        ds_type = resolve_dataset_type(cfg, stage)
        train_ctrl = bool(OmegaConf.select(cfg, "train.end_to_end_ft.train_controlnet", default=False))
        if ds_type == "text_image_edge":
            from sgdjscc_lab.training.edge_transport import build_edge_transport
            edge_transport = build_edge_transport(cfg, jscc, device)
        elif train_ctrl:
            # Should be caught by validate_stage_config; guard against direct calls.
            raise RuntimeError(
                "stage='end_to_end_ft' train_controlnet=true but the dataset has no "
                "edges (dataset.type != text_image_edge) → ControlNet would train "
                "with c=None. Set dataset.type=text_image_edge (or auto) + edge_source."
            )
        return EndToEndFTStageRunner(
            jscc, denoiser, _clip_text_encoder(sem, device),
            cfg, device, param_groups, edge_transport=edge_transport,
        )

    raise ValueError(f"build_stage_runner: unknown stage {stage!r}")

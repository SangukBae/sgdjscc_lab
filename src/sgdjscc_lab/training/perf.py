"""training/perf.py – optional memory / performance toggles for training.

These are **opt-in** and default OFF so they never perturb the reproducible
baseline (mixed precision + grad accumulation are configured elsewhere and are
untouched here).  Each toggle degrades gracefully: when a dependency is missing
or a module does not support the feature we log a clear, single line rather than
silently ignoring the request or crashing.

Toggles (all under ``train.*``)
-------------------------------
``use_8bit_adam``          Build the optimizer with bitsandbytes' 8-bit AdamW
                           (large VRAM saving on the optimizer state). Falls back
                           to ``torch.optim.AdamW`` with a warning if bitsandbytes
                           is unavailable.
``gradient_checkpointing`` Trade compute for activation memory on the trainable
                           module(s). Applied via the module's own
                           ``gradient_checkpointing_enable()`` /
                           ``enable_gradient_checkpointing()`` /
                           ``gradient_checkpointing`` flag when present.
``use_xformers``           Memory-efficient attention. NOTE: the SGD-JSCC MDTv2
                           backbone already calls
                           ``xformers.ops.memory_efficient_attention`` in its
                           attention forward, so for that model this toggle only
                           *verifies* xformers is importable (and calls a
                           diffusers-style enable hook if the module exposes one).

Compatibility
-------------
* 8-bit AdamW composes with ``torch.cuda.amp`` (GradScaler) exactly like the
  fp32 AdamW it replaces — the scaler unscales the fp32 master grads before the
  optimizer's 8-bit state update.
* Gradient checkpointing composes with AMP and grad accumulation; it only affects
  how activations are stored for the backward pass.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer builder (optional 8-bit AdamW)
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(
    param_groups: List[Dict],
    cfg: DictConfig,
    *,
    lr: float,
    weight_decay: float,
    name: str = "optimizer",
):
    """Build the AdamW optimizer for *param_groups*, honouring ``train.use_8bit_adam``.

    Returns a ``torch.optim.AdamW`` by default.  When ``train.use_8bit_adam`` is
    true AND bitsandbytes is importable, returns ``bitsandbytes.optim.AdamW8bit``
    instead; otherwise logs a clear warning and falls back to fp32 AdamW so a
    missing dependency never aborts the run.
    """
    if not param_groups:
        return None

    want_8bit = bool(OmegaConf.select(cfg, "train.use_8bit_adam", default=False))
    if want_8bit:
        try:
            import bitsandbytes as bnb  # type: ignore
        except Exception as exc:  # ImportError, or a broken CUDA-less build
            logger.warning(
                "train.use_8bit_adam=true but bitsandbytes is unavailable (%s) — "
                "falling back to torch.optim.AdamW (fp32 optimizer state).", exc)
        else:
            logger.info("[%s] using bitsandbytes 8-bit AdamW (lr=%.2e, wd=%.2e).",
                        name, lr, weight_decay)
            return bnb.optim.AdamW8bit(param_groups, lr=lr, weight_decay=weight_decay)

    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


# ─────────────────────────────────────────────────────────────────────────────
# Model-level memory toggles (gradient checkpointing, xformers)
# ─────────────────────────────────────────────────────────────────────────────

def _enable_gradient_checkpointing(module: nn.Module, tag: str) -> bool:
    """Try to turn on gradient checkpointing for *module*. Returns True on success."""
    # 1) HF/diffusers convention.
    for meth in ("gradient_checkpointing_enable", "enable_gradient_checkpointing"):
        fn = getattr(module, meth, None)
        if callable(fn):
            try:
                fn()
                logger.info("[perf] gradient checkpointing enabled on '%s' via %s().",
                            tag, meth)
                return True
            except Exception as exc:  # pragma: no cover
                logger.warning("[perf] %s() failed on '%s': %s", meth, tag, exc)
    # 2) Plain boolean flag some custom modules read in their forward.
    if hasattr(module, "gradient_checkpointing"):
        try:
            module.gradient_checkpointing = True  # type: ignore[attr-defined]
            logger.info("[perf] gradient checkpointing flag set on '%s'.", tag)
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("[perf] could not set gradient_checkpointing on '%s': %s", tag, exc)
    logger.warning(
        "[perf] train.gradient_checkpointing=true but module '%s' (%s) exposes no "
        "gradient-checkpointing hook — NOT applied for this module.",
        tag, type(module).__name__)
    return False


# Module class-name prefixes whose attention already routes through
# xformers.ops.memory_efficient_attention natively (SGD-JSCC diffusion backbones).
# For anything else we must NOT claim the optimization was applied.
_NATIVE_XFORMERS_PREFIXES = ("MDTv2",)


def _enable_xformers(module: nn.Module, tag: str) -> bool:
    """Best-effort enable of memory-efficient attention for *module*.

    Returns True only when the feature is genuinely active for *module*: either a
    diffusers-style ``enable_xformers_memory_efficient_attention()`` hook was
    called, or *module* is a known SGD-JSCC diffusion backbone (MDTv2 /
    MDTv2_ControlNet) whose attention already uses xformers natively. For any
    other module (e.g. ``jscc_model`` / ``edge_jscc``) there is no xformers path,
    so we log that it was NOT applied rather than over-reporting success.
    """
    try:
        import xformers  # noqa: F401  (verify importability)
    except Exception as exc:
        logger.warning(
            "[perf] train.use_xformers=true but xformers is not importable (%s) — "
            "NOT applied for '%s'.", exc, tag)
        return False
    # 1) diffusers-style hook, if the module exposes one.
    fn = getattr(module, "enable_xformers_memory_efficient_attention", None)
    if callable(fn):
        try:
            fn()
            logger.info("[perf] xformers memory-efficient attention enabled on '%s' "
                        "via enable_xformers_memory_efficient_attention().", tag)
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("[perf] enabling xformers on '%s' failed: %s", tag, exc)
            return False
    # 2) Known SGD-JSCC diffusion backbone: attention is already xformers-native.
    cls_name = type(module).__name__
    if any(cls_name.startswith(p) for p in _NATIVE_XFORMERS_PREFIXES):
        logger.info(
            "[perf] '%s' (%s) attention already uses xformers.ops."
            "memory_efficient_attention natively (verified importable) — active.",
            tag, cls_name)
        return True
    # 3) Anything else: no xformers path — do not over-report.
    logger.warning(
        "[perf] train.use_xformers=true but module '%s' (%s) has no xformers hook "
        "and is not an xformers-native backbone — NOT applied for this module.",
        tag, cls_name)
    return False


def apply_memory_optimizations(modules: Dict[str, nn.Module], cfg: DictConfig, *,
                               stage: str) -> None:
    """Apply the opt-in memory toggles to the trainable *modules* dict.

    *modules* maps a checkpoint name → nn.Module (typically ``runner.state_modules()``).
    Only real ``nn.Module`` values are considered; non-module state entries are
    skipped.  Every requested-but-inapplicable case is logged (never silently
    ignored), per the task's requirement.
    """
    grad_ckpt = bool(OmegaConf.select(cfg, "train.gradient_checkpointing", default=False))
    use_xformers = bool(OmegaConf.select(cfg, "train.use_xformers", default=False))
    if not (grad_ckpt or use_xformers):
        return

    real = {n: m for n, m in modules.items()
            if isinstance(m, nn.Module) and any(p.requires_grad for p in m.parameters())}
    if not real:
        logger.info("[perf] no trainable module to apply memory toggles to (stage=%s).", stage)
        return

    for name, module in real.items():
        if grad_ckpt:
            _enable_gradient_checkpointing(module, name)
        if use_xformers:
            _enable_xformers(module, name)

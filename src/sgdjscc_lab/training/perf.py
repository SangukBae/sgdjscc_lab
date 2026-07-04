"""training/perf.py – optional memory / performance toggles for training.

These are **opt-in** and default OFF so they never perturb the reproducible
baseline (mixed precision + grad accumulation are configured elsewhere and are
untouched here).  Each toggle degrades gracefully: when a dependency is missing
or a module does not support the feature we log a clear, single line rather than
silently ignoring the request or crashing.

Which toggles actually do something in THIS repo
-------------------------------------------------
* ``use_8bit_adam`` — **effective** when bitsandbytes is installed: swaps
  ``torch.optim.AdamW`` for ``bitsandbytes.optim.AdamW8bit`` (large optimizer-
  state VRAM saving). Falls back to fp32 AdamW with a warning otherwise.
* ``gradient_checkpointing`` — **effectively a NO-OP today.** The trainable core
  modules here (``MDTv2`` / ``MDTv2_ControlNet`` / ``AutoencoderKL``, all from
  read-only ``SGDJSCC/``) expose NO ``gradient_checkpointing_enable`` /
  ``enable_gradient_checkpointing`` / ``gradient_checkpointing`` hook and use no
  ``torch.utils.checkpoint``. Enabling this toggle therefore applies nothing and
  only logs that it was not applied. It is kept wired for forward-compatibility
  (Selection A in docs/training_scaffold.md); for DM-stage memory pressure use
  ``grad_accum_steps`` / a smaller per-rank batch instead.
* ``use_xformers`` — **NOT an incremental optimization here.** The ``MDTv2`` /
  ``MDTv2_ControlNet`` attention forward already calls
  ``xformers.ops.memory_efficient_attention`` unconditionally, so this toggle
  merely VERIFIES xformers is importable and REPORTS status (plus calls a
  diffusers-style enable hook if some module exposes one). For non-DM modules
  (``jscc_model`` / ``edge_jscc``) there is no xformers path → logged NOT applied.

The genuinely useful operational knobs live elsewhere and are honest about it:
``train.mixed_precision`` (AMP) and ``train.num_workers`` (input pipeline).

Compatibility
-------------
* 8-bit AdamW composes with ``torch.cuda.amp`` (GradScaler) exactly like the
  fp32 AdamW it replaces — the scaler unscales the fp32 master grads before the
  optimizer's 8-bit state update.
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

# Class-name prefixes of the SGD-JSCC diffusion core modules (read-only SGDJSCC/).
# Their attention already routes through xformers.ops.memory_efficient_attention
# natively, and they expose NO gradient-checkpointing hook — used by both helpers
# below to emit a precise, honest message for the DM stages (text_dm / controlnet).
_DM_CORE_PREFIXES = ("MDTv2",)      # matches MDTv2 and MDTv2_ControlNet
_NATIVE_XFORMERS_PREFIXES = _DM_CORE_PREFIXES


def _is_dm_core(module: nn.Module) -> bool:
    return type(module).__name__.startswith(_DM_CORE_PREFIXES)


def _enable_gradient_checkpointing(module: nn.Module, tag: str, stage: str) -> bool:
    """Try to turn on gradient checkpointing for *module*. Returns True on success.

    In THIS repo no trainable core module (MDTv2 / MDTv2_ControlNet /
    AutoencoderKL) exposes a hook, so this returns False and logs that nothing was
    applied — the DM-core case gets a more direct message so the no-op is obvious.
    """
    cls_name = type(module).__name__
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
    # 3) No hook. Be explicit that this is a NO-OP — extra-direct for the DM core.
    if _is_dm_core(module):
        logger.warning(
            "[perf] train.gradient_checkpointing=true but the core DM module '%s' "
            "(%s, stage=%s) has NO gradient-checkpointing hook in this repo "
            "(SGDJSCC MDTv2 uses no torch.utils.checkpoint) — this toggle is a "
            "NO-OP, nothing applied. For DM-stage memory use grad_accum_steps / a "
            "smaller per-rank batch instead.", tag, cls_name, stage)
    else:
        logger.warning(
            "[perf] train.gradient_checkpointing=true but module '%s' (%s) exposes "
            "no gradient-checkpointing hook — NOT applied for this module.",
            tag, cls_name)
    return False


def _enable_xformers(module: nn.Module, tag: str, stage: str) -> bool:
    """Report/verify memory-efficient attention status for *module*.

    This is NOT an incremental optimization in this repo: the MDTv2 /
    MDTv2_ControlNet attention forward already calls
    ``xformers.ops.memory_efficient_attention`` unconditionally. Returns True only
    when the feature is genuinely active (a diffusers-style hook fired, OR *module*
    is an MDTv2-family backbone that is already xformers-native). For any other
    module (``jscc_model`` / ``edge_jscc``) there is no xformers path, so we log
    NOT applied rather than over-reporting success.
    """
    try:
        import xformers  # noqa: F401  (verify importability)
    except Exception as exc:
        logger.warning(
            "[perf] train.use_xformers=true but xformers is not importable (%s) — "
            "NOT applied for '%s'.", exc, tag)
        return False
    cls_name = type(module).__name__
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
    # 2) MDTv2-family DM core: already xformers-native — no incremental optimization.
    if _is_dm_core(module):
        logger.info(
            "[perf] train.use_xformers=true: core DM module '%s' (%s, stage=%s) "
            "attention ALREADY uses xformers.ops.memory_efficient_attention "
            "natively — this toggle adds NO incremental optimization "
            "(xformers importability verified).", tag, cls_name, stage)
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
            _enable_gradient_checkpointing(module, name, stage)
        if use_xformers:
            _enable_xformers(module, name, stage)

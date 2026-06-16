"""training/freeze.py – Stage-enforced freeze policy for 3-stage training.

The paper trains the three stages with strict freeze rules:

  stage 1 ``jscc``        train the JSCC encoder/decoder.
  stage 2 ``text_dm``     train the (base) text-guided DM denoiser; the JSCC
                          model is *frozen* (it was finalised in stage 1).
  stage 3 ``controlnet``  "the parameters of the original text-guided DM are
                          frozen, and only the parameters of the DiT blocks
                          handling the structural semantic features are
                          updated" (Sec. V-C).

This module makes those rules a **hard policy**, not a config suggestion:

* The stage decides which modules *may* train.  The legacy
  ``trainable_modules.freeze_*`` flags can only freeze *more* than the stage
  allows — they can never unfreeze something the stage forbids.
* In the ``controlnet`` stage the base DM is frozen and only the ControlNet
  branches (``en_inblocks_controlnet`` / ``en_outblocks_controlnet``) train.
  Overriding this requires the explicit, deliberately-named danger flag
  ``train.controlnet.allow_unfrozen_base_dm: true``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from sgdjscc_lab.training.stages import (
    STAGE_CONTROLNET,
    STAGE_CSI_ESTIMATION,
    STAGE_EDGE_CODEC,
    STAGE_END_TO_END_FT,
    STAGE_JSCC,
    STAGE_TEXT_DM,
)

logger = logging.getLogger(__name__)

# nn.ModuleList attributes on MDTv2_ControlNet that hold the trainable
# structural-semantic branches (see SGDJSCC mask_diffusion_controlnet.py).
CONTROLNET_BRANCH_ATTRS = ("en_inblocks_controlnet", "en_outblocks_controlnet")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap_module(module):
    """Return a real nn.Module from a wrapper object when possible.

    Training bundles may carry lightweight wrappers such as TextExtractor /
    EdgeExtractor that expose the underlying PyTorch module via ``model`` or
    ``_model``. Freeze policy operates on parameters, so unwrap those wrappers
    before touching ``requires_grad``.
    """
    if module is None or isinstance(module, nn.Module):
        return module
    inner = getattr(module, "model", None)
    if isinstance(inner, nn.Module):
        return inner
    inner = getattr(module, "_model", None)
    if isinstance(inner, nn.Module):
        return inner
    return None


def _set_requires_grad(module, flag: bool) -> None:
    module = _unwrap_module(module)
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad_(flag)


def _controlnet_branches(denoiser) -> List[nn.Module]:
    branches = []
    for attr in CONTROLNET_BRANCH_ATTRS:
        m = getattr(denoiser, attr, None)
        if m is not None:
            branches.append(m)
    return branches


def _trainable_param_count(module) -> int:
    module = _unwrap_module(module)
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _frozen(cfg, key: str) -> bool:
    """Read a ``trainable_modules.freeze_*`` flag (default True = frozen)."""
    return bool(OmegaConf.select(cfg, f"trainable_modules.{key}", default=True))


# ─────────────────────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────────────────────

def apply_stage_freeze_policy(
    models, cfg: DictConfig, stage: str
) -> Tuple[List[Dict], Dict]:
    """Freeze/unfreeze modules per *stage* and return optimizer param groups.

    Returns
    -------
    param_groups:
        ``[{"params": [...], "name": "..."}]`` for the optimizer (may be empty
        in dry-runs where *models* is None).
    report:
        Dict describing what trained / froze (for logging & tests):
        ``{"stage", "trainable", "frozen", "forced", "n_trainable_params"}``.
    """
    report: Dict = {
        "stage": stage,
        "trainable": [],
        "frozen": [],
        "forced": [],
        "n_trainable_params": 0,
    }

    if models is None:
        logger.warning("apply_stage_freeze_policy: models is None (dry-run) — "
                       "no parameters to (un)freeze.")
        return [], report

    jscc = getattr(models, "jscc_model", None)
    sem = getattr(models, "sem_pipeline", None)
    denoiser = getattr(sem, "model", None) if sem is not None else None
    text_ex = getattr(models, "text_extractor", None)
    edge_ex = getattr(models, "edge_extractor", None)

    # 1) Freeze EVERYTHING first — the stage explicitly opts modules back in.
    for m in (jscc, denoiser, text_ex, edge_ex):
        _set_requires_grad(m, False)

    param_groups: List[Dict] = []

    def _add_group(module, name: str) -> None:
        if module is None:
            return
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            param_groups.append({"params": params, "name": name})
            report["trainable"].append(name)
        else:
            report["frozen"].append(name)

    # 2) Stage-specific unfreezing.
    if stage in (STAGE_EDGE_CODEC, STAGE_CSI_ESTIMATION):
        # The trainable module (edge codec / SNR estimator) is built fresh by the
        # runner (not part of the bundle); all bundle modules stay frozen here. The
        # JSCC VAE is used (frozen) only for image latents in csi_estimation. The
        # runner appends the new module's own (trainable) param group.
        for name, m in (("jscc_model", jscc), ("diffusion", denoiser)):
            if m is not None:
                report["frozen"].append(name)

    elif stage == STAGE_JSCC:
        if denoiser is not None:
            report["frozen"].append("diffusion")
        if not _frozen(cfg, "freeze_jscc"):
            logger.debug("freeze_jscc=false honoured (stage default trainable).")
        # Stage policy: JSCC trains. (The freeze_jscc flag may only *freeze* it.)
        if _frozen(cfg, "freeze_jscc"):
            # User asked to freeze JSCC even in the JSCC stage → respect, but warn.
            logger.warning(
                "stage='jscc' but trainable_modules.freeze_jscc=true → JSCC frozen; "
                "no parameters will train. Set freeze_jscc:false (default for this stage)."
            )
            report["frozen"].append("jscc_model")
        else:
            _set_requires_grad(jscc, True)
            _add_group(jscc, "jscc_model")

    elif stage == STAGE_TEXT_DM:
        if jscc is not None:
            report["frozen"].append("jscc_model")  # finalised after stage 1
        if denoiser is None:
            logger.warning("stage='text_dm' but no diffusion denoiser is loaded "
                            "(use_semantic may be false).")
        else:
            _set_requires_grad(denoiser, True)
            # Stage 2 trains the *base* DM; ControlNet branches belong to stage 3.
            for branch in _controlnet_branches(denoiser):
                _set_requires_grad(branch, False)
            if _controlnet_branches(denoiser):
                report["frozen"].append("controlnet_branches")
            if _frozen(cfg, "freeze_diffusion"):
                _set_requires_grad(denoiser, False)
                logger.warning(
                    "stage='text_dm' but freeze_diffusion=true → DM frozen; nothing trains."
                )
                report["frozen"].append("diffusion")
            else:
                _add_group(denoiser, "diffusion")

    elif stage == STAGE_CONTROLNET:
        if jscc is not None:
            report["frozen"].append("jscc_model")
        if denoiser is None:
            logger.warning("stage='controlnet' but no diffusion denoiser is loaded.")
        else:
            branches = _controlnet_branches(denoiser)
            allow_unfrozen = bool(OmegaConf.select(
                cfg, "train.controlnet.allow_unfrozen_base_dm", default=False))

            # Start from a fully-frozen denoiser; decide the trainable set BEFORE
            # building any param groups so no parameter lands in two groups.
            _set_requires_grad(denoiser, False)

            if not branches:
                logger.error(
                    "stage='controlnet' but the denoiser has no ControlNet branches "
                    "(%s). Did you set use_controlnet:true so MDTv2_ControlNet loads?",
                    ", ".join(CONTROLNET_BRANCH_ATTRS),
                )

            if allow_unfrozen:
                logger.warning(
                    "⚠ allow_unfrozen_base_dm=true: UNFREEZING the base text-guided DM "
                    "in the ControlNet stage. This DEVIATES from the paper (which freezes "
                    "the base DM) and risks catastrophic forgetting of the stage-2 prior."
                )
                # Single group over the whole denoiser (control branches included).
                _set_requires_grad(denoiser, True)
                report["forced"].append("base_diffusion_unfrozen")
                _add_group(denoiser, "diffusion_full")
            else:
                # HARD POLICY: base DM frozen, only control branches train.
                report["frozen"].append("base_diffusion")
                for i, branch in enumerate(branches):
                    _set_requires_grad(branch, True)
                    _add_group(branch, CONTROLNET_BRANCH_ATTRS[i]
                               if i < len(CONTROLNET_BRANCH_ATTRS)
                               else f"controlnet_branch_{i}")

    elif stage == STAGE_END_TO_END_FT:
        # Joint JSCC↔DM fine-tuning. Trainable set is config-driven:
        #   train.end_to_end_ft.{train_jscc, train_diffusion, train_controlnet}
        train_jscc = bool(OmegaConf.select(cfg, "train.end_to_end_ft.train_jscc", default=True))
        train_dm = bool(OmegaConf.select(cfg, "train.end_to_end_ft.train_diffusion", default=True))
        train_ctrl = bool(OmegaConf.select(cfg, "train.end_to_end_ft.train_controlnet", default=False))

        if train_jscc and jscc is not None:
            _set_requires_grad(jscc, True)
            _add_group(jscc, "jscc_model")
        elif jscc is not None:
            report["frozen"].append("jscc_model")

        if denoiser is not None:
            if train_dm:
                # Whole denoiser (base + any ControlNet branches) trainable.
                _set_requires_grad(denoiser, True)
                _add_group(denoiser, "diffusion_full")
            elif train_ctrl:
                # Only the ControlNet branches; base DM frozen.
                _set_requires_grad(denoiser, False)
                report["frozen"].append("base_diffusion")
                for i, branch in enumerate(_controlnet_branches(denoiser)):
                    _set_requires_grad(branch, True)
                    _add_group(branch, CONTROLNET_BRANCH_ATTRS[i]
                               if i < len(CONTROLNET_BRANCH_ATTRS)
                               else f"controlnet_branch_{i}")
            else:
                report["frozen"].append("diffusion")

    else:
        raise ValueError(f"apply_stage_freeze_policy: unknown stage {stage!r}")

    report["n_trainable_params"] = sum(
        p.numel() for g in param_groups for p in g["params"]
    )
    logger.info(
        "Freeze policy [stage=%s]: trainable=%s  frozen=%s  forced=%s  params=%d",
        stage, report["trainable"] or "(none)", report["frozen"] or "(none)",
        report["forced"] or "(none)", report["n_trainable_params"],
    )
    if not param_groups:
        logger.warning(
            "No trainable parameters after applying the stage freeze policy — "
            "optimizer will be disabled (dry-run / fully-frozen)."
        )
    return param_groups, report

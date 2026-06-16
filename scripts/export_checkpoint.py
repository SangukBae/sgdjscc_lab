#!/usr/bin/env python
"""export_checkpoint.py – Convert a training-stage checkpoint into the format the
inference/runtime loaders expect.

A training run writes ``outputs/checkpoints/<stage>/best.pth`` whose module
weights live under ``runner_state.modules.<name>`` (see
``pipelines/train_pipeline.py``).  The inference loaders, however, expect a
*different* on-disk layout:

  * ``checkpoints/JSCC_model.pth``         → a **raw** ``state_dict``
    (``models/jscc_model.py``: ``model.load_state_dict(torch.load(...))``)
  * ``checkpoints/diffusion_backbone.pth`` → a dict with a top-level ``model_ema``
    (``models/diffusion_wrapper.py``: ``ckpt["model_ema"]``)
  * ``checkpoints/diffusion_controlnet.pth`` → likewise top-level ``model_ema``

This script does the *format conversion* (not a manual copy) for those three
stages:

    stage        training module        export format
    -----------  --------------------   ------------------------------
    jscc         jscc_model             raw state_dict
    text_dm      diffusion              {"model_ema": state_dict}
    controlnet   diffusion              {"model_ema": state_dict}

NOT handled here (by design — they need no format conversion):

  * ``edge_codec`` → ``controlnet`` : the Stage-3 config field
    ``train.controlnet.edge_jscc.checkpoint`` reads
    ``outputs/checkpoints/edge_codec/best.pth`` **directly** (nested extraction
    is done by the loader). Just point the config at the stage output.
  * ``csi_estimation`` → inference : this is a *path wiring* target, not a format
    conversion. Put the stage output path in ``snr_estimator_checkpoint``;
    ``models/csi_estimation.py`` extracts the nested weights itself.

Usage
-----
python scripts/export_checkpoint.py --stage jscc \\
    --input outputs/checkpoints/jscc/best.pth \\
    --output checkpoints/JSCC_model.pth

python scripts/export_checkpoint.py --stage text_dm \\
    --input outputs/checkpoints/text_dm_coco_json/best.pth \\
    --output checkpoints/diffusion_backbone.pth

python scripts/export_checkpoint.py --stage controlnet \\
    --input outputs/checkpoints/controlnet/best.pth \\
    --output checkpoints/diffusion_controlnet.pth --force

# Inspect what would be extracted without writing anything:
python scripts/export_checkpoint.py --stage controlnet \\
    --input outputs/checkpoints/controlnet/best.pth \\
    --output checkpoints/diffusion_controlnet.pth --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("sgdjscc_lab.export_checkpoint")


# stage → (training module name, export wrapper).
# stage → (training module name, export wrapper, allow_bare).
#   wrapper "raw"       : save the extracted state_dict as-is.
#   wrapper "model_ema" : save {"model_ema": extracted_state_dict}.
#   allow_bare          : whether a *bare* state_dict input (no runner_state /
#                         model_state nesting, hence no module identity) is an
#                         acceptable source.
#
# A bare state_dict carries no module name, so it cannot be checked against the
# stage's expected module. We therefore allow it ONLY for ``jscc``, whose export
# is itself a raw state_dict — a 1:1 pass-through where wrapping cannot silently
# mislabel the payload. For the diffusion stages a bare input is rejected, so a
# raw (e.g. JSCC) state_dict can never be silently wrapped as ``model_ema``.
_STAGE_SPEC: Dict[str, Tuple[str, str, bool]] = {
    "jscc": ("jscc_model", "raw", True),
    "text_dm": ("diffusion", "model_ema", False),
    "controlnet": ("diffusion", "model_ema", False),
}


def _looks_like_state_dict(obj) -> bool:
    """A bare state_dict is a non-empty dict whose values are all tensors."""
    return (
        isinstance(obj, dict)
        and len(obj) > 0
        and all(torch.is_tensor(v) for v in obj.values())
    )


def _extract_module_state(ckpt, module_name: str, allow_bare: bool) -> Tuple[dict, str]:
    """Find *module_name*'s state_dict inside a training checkpoint.

    Searches, in order:
      1. ``runner_state.modules.<module_name>``  (current format)
      2. ``model_state.<module_name>``           (legacy format)
      3. the checkpoint itself, if it is already a bare state_dict *and*
         *allow_bare* is set for this stage

    Returns ``(state_dict, resolved_source_key)``. Raises ``KeyError`` when the
    module cannot be located so callers fail early on a stage/checkpoint mismatch.

    A bare state_dict has no module identity, so it cannot be verified against
    *module_name*; *allow_bare* (false for the diffusion stages) prevents an
    unrelated raw checkpoint from being silently accepted and mis-wrapped.
    """
    if not isinstance(ckpt, dict):
        raise KeyError(
            f"Checkpoint is not a dict (got {type(ckpt).__name__}); cannot extract "
            f"module {module_name!r}."
        )

    rs = ckpt.get("runner_state")
    if isinstance(rs, dict):
        modules = rs.get("modules") or {}
        if module_name in modules:
            return modules[module_name], f"runner_state.modules.{module_name}"

    model_state = ckpt.get("model_state")
    if isinstance(model_state, dict) and module_name in model_state:
        return model_state[module_name], f"model_state.{module_name}"

    is_bare = _looks_like_state_dict(ckpt)
    if is_bare and allow_bare:
        # Already-extracted raw checkpoint (only stages whose export is itself a
        # raw state_dict opt into this — see _STAGE_SPEC).
        return ckpt, "<bare state_dict>"

    # Build a helpful message about what *was* available / why we refused.
    if is_bare:  # bare but this stage forbids it
        raise KeyError(
            f"Refusing a bare state_dict for stage's module {module_name!r}: a bare "
            "checkpoint has no module identity and cannot be verified against this "
            "stage, so it will not be wrapped (it could be an unrelated raw "
            "checkpoint). Pass a training checkpoint with "
            f"runner_state.modules.{module_name} (e.g. .../<stage>/best.pth)."
        )
    available = []
    if isinstance(rs, dict):
        available.append(f"runner_state.modules={list((rs.get('modules') or {}).keys())}")
    if isinstance(model_state, dict):
        available.append(f"model_state={list(model_state.keys())}")
    if not available:
        available.append(f"top-level keys={list(ckpt.keys())}")
    raise KeyError(
        f"Module {module_name!r} not found in checkpoint. Available: "
        + "; ".join(available)
        + ". Did you pass the right --stage for this checkpoint?"
    )


def export_checkpoint(stage: str, ckpt) -> Tuple[object, str]:
    """Convert a loaded training checkpoint to the inference payload for *stage*.

    Returns ``(payload, resolved_source_key)`` where *payload* is the object to be
    ``torch.save``-d (raw state_dict for ``jscc``; ``{"model_ema": ...}`` for the
    diffusion stages).
    """
    if stage not in _STAGE_SPEC:
        raise ValueError(
            f"Unknown stage {stage!r}; expected one of {sorted(_STAGE_SPEC)}."
        )
    module_name, wrapper, allow_bare = _STAGE_SPEC[stage]
    state_dict, source_key = _extract_module_state(ckpt, module_name, allow_bare)
    if not _looks_like_state_dict(state_dict):
        raise KeyError(
            f"Extracted object for stage={stage!r} (source {source_key}) is not a "
            f"valid state_dict (expected a non-empty tensor mapping)."
        )
    if wrapper == "raw":
        return state_dict, source_key
    if wrapper == "model_ema":
        return {"model_ema": state_dict}, source_key
    raise AssertionError(f"unhandled wrapper {wrapper!r}")  # pragma: no cover


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a training-stage checkpoint to inference format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="edge_codec / csi_estimation need no export — see module docstring.",
    )
    p.add_argument("--stage", required=True, choices=sorted(_STAGE_SPEC),
                   help="Training stage that produced --input.")
    p.add_argument("--input", "-i", required=True,
                   help="Training checkpoint (e.g. outputs/checkpoints/<stage>/best.pth)")
    p.add_argument("--output", "-o", required=True,
                   help="Destination inference checkpoint path.")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve and report the extracted keys without writing.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite --output if it already exists.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {in_path}")
    if out_path.exists() and not args.force and not args.dry_run:
        raise FileExistsError(
            f"Output already exists: {out_path} (use --force to overwrite)."
        )

    ckpt = torch.load(in_path, map_location="cpu")
    payload, source_key = export_checkpoint(args.stage, ckpt)

    # Describe the payload concisely.
    if isinstance(payload, dict) and "model_ema" in payload:
        n_tensors = len(payload["model_ema"])
        out_format = "{'model_ema': state_dict}"
    else:
        n_tensors = len(payload)
        out_format = "raw state_dict"

    logger.info("stage        : %s", args.stage)
    logger.info("input        : %s", in_path)
    logger.info("source key   : %s", source_key)
    logger.info("output       : %s", out_path)
    logger.info("format       : %s (%d tensors)", out_format, n_tensors)

    if args.dry_run:
        logger.info("[dry-run] nothing written.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    logger.info("Exported → %s", out_path)


if __name__ == "__main__":
    sys.exit(main())

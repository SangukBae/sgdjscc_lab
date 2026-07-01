#!/usr/bin/env python
"""train.py – Stage-aware training CLI for sgdjscc_lab.

Entry point for the paper's 3-stage SGD-JSCC training procedure.  Config-driven
and OmegaConf-based, mirroring evaluate.py; the existing inference/evaluation
paths are untouched.

Stages (``--stage`` / ``train.stage``)
--------------------------------------
  jscc        Stage 1 — train the JSCC encoder/decoder under a fixed AWGN SNR
              (default 10 dB). MSE (+ optional patch-GAN). Image-only data.
  text_dm     Stage 2 — train the text-guided latent DM to predict f0 from a
              noised latent (masked + unmasked branches). Needs captions.
  controlnet  Stage 3 — train ONLY the ControlNet branches with edge guidance;
              the base DM is frozen. Needs captions + edges.

Each stage validates its required inputs up-front and fails with an explicit
message if a caption/edge source is missing.

Usage examples
--------------
# Stage 1 (JSCC) full run on GPU
python scripts/train.py --config configs/composed_train_jscc.yaml \\
    --train-list /data/imagenet/train/ --val-list /data/imagenet/val/ \\
    --device cuda:0 --epochs 20

# Stage 2 (text DM) — captions via per-image .txt sidecars
python scripts/train.py --config configs/composed_train_text_dm.yaml \\
    --train-list /data/pairs/train/ --device cuda:0

# Stage 3 (ControlNet) — edges via on-the-fly Canny
python scripts/train.py --config configs/composed_train_controlnet.yaml \\
    --train-list /data/pairs/train/ --device cuda:0

# Dry-run (no checkpoints/GPU): exercises config/stage/dataset wiring only
python scripts/train.py --config configs/composed_train_jscc.yaml \\
    --train-list /path/to/images/ --no-models --epochs 1

Options
-------
--config        Path to YAML config file (required)
--stage         Training stage: jscc | text_dm | controlnet (overrides config)
--train-list    Override config.train_input_path
--val-list      Override config.val_input_path
--output-dir    Override config.checkpoint_dir
--epochs        Override train.epochs
--resume        Override train.resume (checkpoint path)
--device        Override config.device
--no-models     Skip model loading (dry-run: no trainable params, no weight updates)
--seed          Override train.seed
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Make src/ importable without editable install ────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sgdjscc_lab.config import load_config, merge_cli_overrides
from omegaconf import OmegaConf

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sgdjscc_lab.train")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="sgdjscc_lab Training CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", "-c", required=True,
                   help="Path to YAML config (e.g. configs/composed_train.yaml)")
    p.add_argument("--stage", default=None,
                   choices=["jscc", "text_dm", "controlnet", "edge_codec",
                            "csi_estimation", "end_to_end_ft"],
                   help="Training stage (overrides train.stage): jscc | text_dm | "
                        "controlnet | edge_codec | csi_estimation | end_to_end_ft")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override train.max_steps (>0 → step-based training)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override train.batch_size (useful for smoke/OOM recovery)")
    p.add_argument("--log-every-steps", type=int, default=None,
                   help="Override train.log_every_steps (step mode; set 1 to see "
                        "per-step loss — useful for smoke runs)")
    p.add_argument("--save-every-steps", type=int, default=None,
                   help="Override train.save_every_steps (step mode checkpoint cadence)")
    p.add_argument("--train-list", default=None,
                   help="Override config.train_input_path — folder of training images")
    p.add_argument("--val-list", default=None,
                   help="Override config.val_input_path — folder of validation images")
    p.add_argument("--output-dir", default=None,
                   help="Override config.checkpoint_dir")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override train.epochs")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--device", default=None,
                   help="Compute device, e.g. 'cuda:0' or 'cpu'")
    p.add_argument("--no-models", action="store_true",
                   help="Skip model loading; use identity reconstruction (dry-run)")
    p.add_argument("--seed", type=int, default=None,
                   help="Override train.seed for reproducibility")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Rank-0-only logging helper: under `torchrun --nproc_per_node=N` every rank
    # runs this script, so informational startup lines would otherwise print N
    # times. Gate them to rank 0 (RANK is set by torchrun before the process
    # group is initialised, so is_rank0() is already correct here).
    from sgdjscc_lab import distributed as ddp
    ddp.configure_worker_logging()
    def _log0(msg, *a):
        if ddp.is_rank0():
            logger.info(msg, *a)

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, device=args.device)

    # Apply training-specific CLI overrides.  Accumulate all train.* overrides
    # into a single plain dict so multiple flags compose (e.g. --epochs --stage).
    overrides: dict = {}
    if args.train_list is not None:
        overrides["train_input_path"] = args.train_list
    if args.val_list is not None:
        overrides["val_input_path"]   = args.val_list
    if args.output_dir is not None:
        overrides["checkpoint_dir"]   = args.output_dir

    train_overrides: dict = {}
    if args.epochs is not None:
        train_overrides["epochs"] = args.epochs
    if args.resume is not None:
        train_overrides["resume"] = args.resume
    if args.seed is not None:
        train_overrides["seed"] = args.seed
    if args.stage is not None:
        train_overrides["stage"] = args.stage
    if args.max_steps is not None:
        train_overrides["max_steps"] = args.max_steps
    if args.batch_size is not None:
        train_overrides["batch_size"] = args.batch_size
    if args.log_every_steps is not None:
        train_overrides["log_every_steps"] = args.log_every_steps
    if args.save_every_steps is not None:
        train_overrides["save_every_steps"] = args.save_every_steps
    if train_overrides:
        overrides["train"] = train_overrides

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))

    # ── Respect train.enabled flag (check before any expensive work) ─────────
    if not bool(OmegaConf.select(cfg, "train.enabled", default=True)):
        logger.info("train.enabled=false — training disabled in config. Exiting.")
        return

    # ── Resolve + validate the training stage BEFORE loading any checkpoints ──
    # A misconfigured stage (missing caption/edge source, etc.) fails here with
    # an explicit message rather than silently doing the wrong thing later.
    from sgdjscc_lab.training.stages import (
        resolve_stage, validate_stage_config, StageConfigError,
    )
    from sgdjscc_lab import paper_mode as _paper_mode
    try:
        stage = resolve_stage(cfg)
        validate_stage_config(cfg, stage)
        # paper_mode (default off): enforce paper-faithful guardrails — blocks
        # auto-captions, Canny edges, shared_vae transport, zero CFG null, etc.
        _paper_mode.enforce(cfg, stage)
    except StageConfigError as exc:   # PaperModeError subclasses StageConfigError
        sys.exit(f"Error: invalid training config for stage.\n  {exc}")
    if _paper_mode.is_enabled(cfg):
        _log0("paper_mode: ON — %s", _paper_mode.summary())

    # ── Log key settings (rank 0 only) ─────────────────────────────────────────
    _log0("Config:           %s", args.config)
    _log0("stage:            %s", stage)
    _log0("train_input_path: %s", OmegaConf.select(cfg, "train_input_path", default=None))
    _log0("val_input_path:   %s", OmegaConf.select(cfg, "val_input_path",   default=None))
    _log0("checkpoint_dir:   %s", OmegaConf.select(cfg, "checkpoint_dir",   default="outputs/checkpoints"))
    _log0("device:           %s", cfg.get("device", "cpu"))
    _log0("epochs:           %d", OmegaConf.select(cfg, "train.epochs", default=10))
    _log0("no_models:        %s", args.no_models)

    # ── Seed ──────────────────────────────────────────────────────────────────
    from sgdjscc_lab.utils.seed import set_global_seed
    seed = int(OmegaConf.select(cfg, "train.seed", default=2025))
    set_global_seed(seed)
    _log0("Seed: %d", seed)

    # ── Device (DDP-aware) ─────────────────────────────────────────────────────
    # Under `torchrun --nproc_per_node=N` setup_distributed() inits the process
    # group and returns cuda:{LOCAL_RANK}; single-process runs fall back to the
    # configured device (identical to before).
    from sgdjscc_lab.runtime import resolve_device
    rank, world_size, local_rank, ddp_device = ddp.setup_distributed()
    device = ddp_device if ddp_device is not None else resolve_device(str(cfg.get("device", "cpu")))
    if world_size > 1:
        # rank-0-only summary (previously printed once per rank).
        _log0("DDP: world_size=%d  (rank0 local_rank=%d device=%s)", world_size, local_rank, device)
    _log0("Resolved device: %s", device)

    # ── Models ────────────────────────────────────────────────────────────────
    # The edge_codec stage is self-contained (it builds its own trainable edge
    # codec + AWGN channel), so it needs NO pretrained JSCC/diffusion bundle.
    models = None
    if stage == "edge_codec" and not args.no_models:
        _log0("stage='edge_codec' is self-contained — skipping the "
              "JSCC/diffusion model bundle (no checkpoints needed).")
    elif not args.no_models:
        from sgdjscc_lab.runtime import build_models
        _log0("Building models…")
        models = build_models(cfg, device)
        _log0("Models loaded.")
    else:
        _log0("--no-models: skipping model loading (dry-run mode).")

    # ── Run training ──────────────────────────────────────────────────────────
    from sgdjscc_lab.pipelines.train_pipeline import run_training
    try:
        run_training(cfg, models, device)
    finally:
        # Always tear down the process group (barrier + destroy) so torchrun
        # exits cleanly; no-op for single-process runs.
        ddp.cleanup_distributed()

    _log0("train.py complete.")


if __name__ == "__main__":
    main()

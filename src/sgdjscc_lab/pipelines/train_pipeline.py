"""pipelines/train_pipeline.py – Stage-aware training loop for sgdjscc_lab.

Orchestrates the paper's 3-stage SGD-JSCC training (``train.stage`` ∈
{``jscc``, ``text_dm``, ``controlnet``}).  This module stays stage-agnostic: the
forward pass, loss and optimization for each stage live in
``training/stage_runners.py``; dataset selection in ``data/datasets.py``; the
freeze policy in ``training/freeze.py``.

Entry points
------------
run_training(cfg, models, device)
    Resolve + validate the stage, build stage dataloaders + runner, run epochs.

run_epoch(runner, loader, epoch, log_every, training)
    Iterate one epoch, driving ``runner.training_step`` / ``validation_step``
    and aggregating the per-batch metric dicts.

save_checkpoint / restore_runner_state
    Persist / restore the runner's trained modules + optimizer (stage-tagged).

collect_trainable_params(models, cfg)
    Legacy flag-based param collection (superseded by the stage freeze policy in
    ``training/freeze.py``; retained for backward compatibility).

Notes
-----
- All existing inference/evaluation paths are unmodified; the JSCC encode/decode
  in the stage-1 runner mirror ``infer_pipeline.py`` numerically.
- A run with ``--no-models`` has no trainable parameters (optimizer disabled);
  it exercises config/dataset/freeze wiring but does not update weights.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module freeze / trainable-param collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_trainable_params(
    models, cfg: DictConfig
) -> Tuple[List[Dict], List[str]]:
    """Return (optimizer_param_groups, frozen_module_names).

    Reads ``trainable_modules.*`` flags from *cfg*.  By default all modules
    are frozen; set ``freeze_*: false`` to make them trainable.

    Returns
    -------
    param_groups:
        List of ``{"params": [...], "name": "..."}`` dicts for the optimizer.
        Empty when all modules are frozen.
    frozen_names:
        Names of modules that were frozen.
    """
    tm_cfg = OmegaConf.select(cfg, "trainable_modules", default=None)

    def _frozen(key: str) -> bool:
        if tm_cfg is None:
            return True
        return bool(OmegaConf.select(tm_cfg, key, default=True))

    freeze_jscc       = _frozen("freeze_jscc")
    freeze_diffusion  = _frozen("freeze_diffusion")
    freeze_controlnet = _frozen("freeze_controlnet")
    freeze_guidance   = _frozen("freeze_guidance")

    param_groups: List[Dict] = []
    frozen_names: List[str]  = []

    def _maybe_add(module, name: str, do_freeze: bool) -> None:
        if module is None:
            return
        if do_freeze:
            for p in module.parameters():
                p.requires_grad_(False)
            frozen_names.append(name)
        else:
            params = [p for p in module.parameters() if p.requires_grad]
            if params:
                param_groups.append({"params": params, "name": name})

    if models is not None:
        _maybe_add(getattr(models, "jscc_model",     None), "jscc_model",     freeze_jscc)
        _maybe_add(getattr(models, "sem_pipeline",   None), "sem_pipeline",   freeze_diffusion)
        _maybe_add(getattr(models, "text_extractor", None), "text_extractor", freeze_guidance)
        _maybe_add(getattr(models, "edge_extractor", None), "edge_extractor", freeze_guidance)

        # freeze_controlnet: ControlNet branches are embedded inside
        # sem_pipeline.model (MDTv2_ControlNet) as en_inblocks_controlnet /
        # en_outblocks_controlnet.  Only meaningful when freeze_diffusion=False;
        # when freeze_diffusion=True the whole sem_pipeline is already frozen.
        if not freeze_diffusion:
            sem = getattr(models, "sem_pipeline", None)
            denoiser = getattr(sem, "model", None) if sem is not None else None
            ctrl_branches = []
            for attr in ("en_inblocks_controlnet", "en_outblocks_controlnet"):
                m = getattr(denoiser, attr, None) if denoiser is not None else None
                if m is not None:
                    ctrl_branches.append(m)

            if ctrl_branches:
                if freeze_controlnet:
                    # Re-freeze just the ControlNet branches that _maybe_add
                    # made trainable a moment ago.
                    frozen_ctrl_ids: set = set()
                    for branch in ctrl_branches:
                        for p in branch.parameters():
                            p.requires_grad_(False)
                            frozen_ctrl_ids.add(id(p))
                    # Prune those params from the sem_pipeline param_group.
                    for g in param_groups:
                        if g.get("name") == "sem_pipeline":
                            g["params"] = [
                                p for p in g["params"]
                                if id(p) not in frozen_ctrl_ids
                            ]
                    frozen_names.append("sem_pipeline.controlnet_branches")
                    logger.info(
                        "freeze_controlnet=true: ControlNet branches frozen within sem_pipeline"
                    )
                else:
                    logger.info(
                        "freeze_controlnet=false: ControlNet branches trainable within sem_pipeline"
                    )
            else:
                if freeze_controlnet:
                    logger.warning(
                        "freeze_controlnet=true but no ControlNet branches found in "
                        "sem_pipeline.model (use_controlnet may be false in config). "
                        "ControlNet freezing falls back to freeze_diffusion."
                    )

    n_trainable = sum(p.numel() for g in param_groups for p in g["params"])
    logger.info(
        "Trainable params: %d  |  Frozen modules: %s",
        n_trainable, frozen_names or "(none)",
    )
    if not param_groups:
        logger.warning(
            "All modules are frozen (default).  Set 'trainable_modules.freeze_*: false' "
            "in the config to enable gradient updates."
        )
    return param_groups, frozen_names


# ─────────────────────────────────────────────────────────────────────────────
# Stage-aware epoch loop
# ─────────────────────────────────────────────────────────────────────────────
#
# The forward/loss/optimization are owned by the stage runner
# (training/stage_runners.py); this loop only iterates batches and aggregates
# the per-batch metric dicts the runner returns.

def _mean_metrics(accum: Dict[str, float], n: int) -> Dict[str, float]:
    return {k: (v / max(n, 1)) for k, v in accum.items()}


def run_epoch(runner, loader, epoch: int, log_every: int = 10, training: bool = True) -> Dict:
    """Iterate one epoch over *loader*, driving *runner* per batch.

    Returns aggregated mean metrics plus ``n_batches`` / ``epoch_s``.
    """
    t0 = time.time()
    accum: Dict[str, float] = {}
    n_batches = 0
    step_fn = runner.training_step if training else runner.validation_step

    for batch_idx, batch in enumerate(loader):
        metrics = step_fn(batch)
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + float(v)
        n_batches += 1

        if training and (batch_idx + 1) % log_every == 0:
            logger.info("  [epoch %d | %d/%d]  loss=%.6f",
                        epoch, batch_idx + 1, len(loader), float(metrics.get("loss", 0.0)))

    out = _mean_metrics(accum, n_batches)
    out["n_batches"] = n_batches
    out["epoch_s"] = time.time() - t0
    tag = "Epoch" if training else "Validation"
    logger.info("%s %d done — loss=%.6f  batches=%d  time=%.1fs",
                tag, epoch, out.get("loss", 0.0), n_batches, out["epoch_s"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    state: Dict,
    checkpoint_dir: str | Path,
    epoch: int,
    is_best: bool = False,
    is_latest: bool = True,
    save_every: int = 0,
) -> None:
    """Save training state to *checkpoint_dir*.

    Saves:
    - ``latest.pth``            always (when *is_latest* is True)
    - ``epoch_{N:04d}.pth``     when *save_every* > 0 and epoch % save_every == 0
    - ``best.pth``              when *is_best* is True

    State dict keys: ``epoch``, ``model_state`` (dict), ``optimizer_state``,
    ``best_metric``, ``cfg`` (serialised).
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _save(name: str) -> None:
        path = ckpt_dir / name
        torch.save(state, path)
        logger.info("Checkpoint → %s", path)

    if is_latest:
        _save("latest.pth")
    if is_best:
        _save("best.pth")
    if save_every > 0 and epoch % save_every == 0:
        _save(f"epoch_{epoch:04d}.pth")


def load_checkpoint(
    path: str | Path,
    models=None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    """Load a checkpoint and restore model / optimizer states.

    Returns the raw state dict so callers can read ``epoch``, ``best_metric``,
    etc.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location="cpu")
    logger.info("Loaded checkpoint from %s (epoch %s)", path, state.get("epoch", "?"))

    if models is not None and "model_state" in state:
        model_state = state["model_state"]
        for attr, sd in model_state.items():
            module = getattr(models, attr, None)
            if module is not None and hasattr(module, "load_state_dict"):
                try:
                    module.load_state_dict(sd, strict=False)
                    logger.info("  Restored %s", attr)
                except Exception as exc:
                    logger.warning("  Skipped %s: %s", attr, exc)

    if optimizer is not None and "optimizer_state" in state:
        try:
            optimizer.load_state_dict(state["optimizer_state"])
            logger.info("  Restored optimizer state")
        except Exception as exc:
            logger.warning("  Could not restore optimizer state: %s", exc)

    return state


def _collect_model_state(models) -> Dict:
    """Snapshot the state_dict of each module in the bundle (legacy helper)."""
    state: Dict = {}
    if models is None:
        return state
    for attr in ("jscc_model", "sem_pipeline", "text_extractor", "edge_extractor"):
        m = getattr(models, attr, None)
        if m is not None and hasattr(m, "state_dict"):
            state[attr] = m.state_dict()
    return state


def _collect_runner_state(runner) -> Dict:
    """Snapshot the state_dict of each module the *runner* trains."""
    state: Dict = {}
    for name, module in runner.state_modules().items():
        if module is not None and hasattr(module, "state_dict"):
            state[name] = module.state_dict()
    return state


def _runner_save_state(runner) -> Dict:
    """Full train-state for *runner* (all modules/optimizers/scalers/accum).

    Uses ``runner.get_train_state()`` when available; falls back to the legacy
    single-optimizer snapshot otherwise (e.g. for lightweight stub runners)."""
    if hasattr(runner, "get_train_state"):
        return {"runner_state": runner.get_train_state()}
    return {
        "model_state": _collect_runner_state(runner),
        "optimizer_state": runner.optimizer_state() if hasattr(runner, "optimizer_state") else {},
    }


def restore_runner_state(path, runner) -> Dict:
    """Restore a stage checkpoint into the runner (modules + every optimizer +
    every scaler + the grad-accumulation counter)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    state = torch.load(p, map_location="cpu")
    logger.info("Loaded checkpoint from %s (epoch %s, global_step %s, stage %s)",
                p, state.get("epoch", "?"), state.get("global_step", "?"),
                state.get("stage", "?"))
    if hasattr(runner, "load_train_state"):
        # new format nests under "runner_state"; legacy keeps keys at top level.
        runner.load_train_state(state.get("runner_state", state))
    else:  # pragma: no cover - legacy fallback for non-StageRunner runners
        modules = runner.state_modules()
        for name, sd in state.get("model_state", {}).items():
            m = modules.get(name)
            if m is not None and hasattr(m, "load_state_dict"):
                m.load_state_dict(sd, strict=False)
        if getattr(runner, "optimizer", None) is not None and state.get("optimizer_state"):
            runner.optimizer.load_state_dict(state["optimizer_state"])
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Training log (JSONL)
# ─────────────────────────────────────────────────────────────────────────────

class TrainingLogger:
    """Append-mode JSONL logger for per-epoch training metrics."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    cfg: DictConfig,
    models,
    device: torch.device,
    train_loader=None,
    val_loader=None,
    runner=None,
) -> None:
    """Stage-aware training loop.

    Resolves ``train.stage`` (jscc | text_dm | controlnet), validates the
    stage's required config inputs, builds the stage-appropriate dataloaders and
    runner (which applies the freeze policy and owns its optimizer), then runs a
    shared epoch loop.

    Parameters
    ----------
    cfg:
        Loaded OmegaConf config (must contain a ``train`` block with ``stage``).
    models:
        ModelBundle from runtime.build_models(), or None for a dry-run.
    device:
        Compute device.
    train_loader / val_loader:
        Pre-built DataLoaders.  When None they are built from cfg paths
        (``train_input_path`` / ``val_input_path``) for the active stage.
    """
    from sgdjscc_lab.training.stages import validate_stage_config, STAGE_EDGE_CODEC
    from sgdjscc_lab.training.stage_runners import build_stage_runner
    from sgdjscc_lab.data.datasets import build_dataloader_for_stage

    # ── Respect train.enabled flag ────────────────────────────────────────────
    if not bool(OmegaConf.select(cfg, "train.enabled", default=True)):
        logger.info("train.enabled=false: training skipped.")
        return

    # ── Resolve + validate stage (fails early on bad config) ──────────────────
    stage = validate_stage_config(cfg)
    logger.info("Training stage: %s", stage)

    # ── Config (epoch- and step-based) ────────────────────────────────────────
    epochs      = int(OmegaConf.select(cfg, "train.epochs",     default=10))
    save_every  = int(OmegaConf.select(cfg, "train.save_every", default=5))
    val_every   = int(OmegaConf.select(cfg, "train.val_every",  default=1))
    log_every   = int(OmegaConf.select(cfg, "train.log_every",  default=10))
    ckpt_dir    = str(OmegaConf.select(cfg, "checkpoint_dir",   default="outputs/checkpoints"))
    log_path    = str(OmegaConf.select(cfg, "train_log_path",   default="outputs/train_log.jsonl"))
    resume_path = OmegaConf.select(cfg, "train.resume", default=None)

    # Step-based controls (0/None = disabled). When max_steps > 0 the run is
    # driven by *global optimizer steps* rather than epochs.
    max_steps        = int(OmegaConf.select(cfg, "train.max_steps",        default=0) or 0)
    save_every_steps = int(OmegaConf.select(cfg, "train.save_every_steps", default=0) or 0)
    val_every_steps  = int(OmegaConf.select(cfg, "train.val_every_steps",  default=0) or 0)
    log_every_steps  = int(OmegaConf.select(cfg, "train.log_every_steps",  default=0) or 0)
    step_mode = max_steps > 0

    # ── DataLoaders (stage-aware) ─────────────────────────────────────────────
    train_input = OmegaConf.select(cfg, "train_input_path", default=None)
    val_input   = OmegaConf.select(cfg, "val_input_path",   default=None)

    # File-list mode drives the dataset from an explicit path list, so a loader
    # can be built even without train_input_path / val_input_path (the list path
    # itself provides the images). Folder mode keeps requiring the directory.
    file_list_mode = str(OmegaConf.select(
        cfg, "train.dataset.input_mode", default="folder")).lower() == "file_list"
    val_file_list = OmegaConf.select(cfg, "train.dataset.val_file_list_path", default=None)

    if train_loader is None and (train_input or file_list_mode):
        train_loader = build_dataloader_for_stage(
            train_input, cfg, shuffle=True, training=True, stage=stage)
    if val_loader is None and (val_input or (file_list_mode and val_file_list)):
        val_loader = build_dataloader_for_stage(
            val_input, cfg, shuffle=False, training=False, stage=stage)

    if train_loader is None:
        raise RuntimeError(
            "No training data: set 'train_input_path' (or pass --train-list), or use "
            "train.dataset.input_mode=file_list + train.dataset.file_list_path."
        )

    # ── Dry-run (no models): validate the stage + data wiring, then stop ──────
    # This is an explicit smoke test of config/stage/dataset selection — it does
    # NOT train (there are no parameters to update without real models).
    # Skipped when a pre-built runner is injected (tests / programmatic use), and
    # for the self-contained edge_codec stage (it owns its trainable codec, so it
    # trains without a JSCC/diffusion bundle).
    if models is None and runner is None and stage != STAGE_EDGE_CODEC:
        logger.warning(
            "Dry-run: models not loaded (--no-models). Validating stage '%s' "
            "data pipeline, then exiting WITHOUT training.", stage)
        try:
            batch = next(iter(train_loader))
            shapes = {k: (tuple(v.shape) if hasattr(v, "shape") else f"list[{len(v)}]")
                      for k, v in batch.items()}
            logger.info("Dry-run batch OK — keys/shapes: %s", shapes)
        except StopIteration:
            logger.error("Dry-run: training loader yielded no batches "
                         "(empty dataset / batch_size too large).")
        logger.info("Dry-run complete. Re-run without --no-models to train stage '%s'.",
                    stage)
        return

    # ── Stage runner (applies freeze policy + builds optimizer) ───────────────
    if runner is None:
        runner = build_stage_runner(stage, models, cfg, device)

    # ── Resume (epoch AND global step) ─────────────────────────────────────────
    start_epoch = 1
    best_metric = float("inf")
    global_step = 0
    if resume_path:
        state = restore_runner_state(resume_path, runner)
        start_epoch = int(state.get("epoch", 0)) + 1
        best_metric = float(state.get("best_metric", float("inf")))
        global_step = int(state.get("global_step", 0))
        logger.info("Resuming from epoch %d / global_step %d (best_metric=%.6f)",
                    start_epoch, global_step, best_metric)

    # ── Training log ──────────────────────────────────────────────────────────
    train_log = TrainingLogger(log_path)
    lr = float(OmegaConf.select(cfg, "train.lr", default=1e-4))

    def _lr_now() -> float:
        return (runner.optimizer.param_groups[0]["lr"]
                if runner.optimizer is not None else 0.0)

    def _save(epoch: int, monitor: float) -> None:
        nonlocal best_metric
        is_best = monitor < best_metric
        if is_best:
            best_metric = monitor
        ckpt_state = {
            "epoch":       epoch,
            "global_step": global_step,
            "stage":       stage,
            "best_metric": best_metric,
            **_runner_save_state(runner),   # all modules/optimizers/scalers/accum
        }
        save_checkpoint(ckpt_state, ckpt_dir, epoch=epoch,
                        is_best=is_best, is_latest=True, save_every=save_every)

    def _validate(epoch: int) -> Dict:
        if val_loader is None:
            return {}
        vm = run_epoch(runner, val_loader, epoch, log_every, training=False)
        return vm

    logger.info(
        "Starting training [stage=%s]: %s  lr=%.2e  grad_accum=%d  amp=%s  device=%s",
        stage,
        (f"max_steps={max_steps} (step mode)" if step_mode else f"epochs {start_epoch}→{epochs}"),
        lr, getattr(runner, "grad_accum", 1), getattr(runner, "use_amp", False), device,
    )

    # ── Unified loop: epoch-paced, but counting GLOBAL OPTIMIZER STEPS ─────────
    # Step-based events (log/val/save_every_steps, max_steps termination) fire on
    # completed optimizer updates; epoch-based events fire at epoch boundaries
    # when not in step mode.
    epoch = start_epoch
    done = False
    run_acc: Dict[str, float] = {}     # running metric sums since last step-log
    run_n = 0
    last_metrics: Dict[str, float] = {}

    def _on_global_step(monitor_loss: float) -> bool:
        """Run log/val/save events for the just-completed global step.

        Shared by normal optimizer updates AND epoch-boundary flush steps so the
        step count, the step-based events, and max_steps termination stay
        consistent regardless of how the step was produced.  Returns True when
        max_steps is reached (step mode).
        """
        nonlocal run_acc, run_n
        if log_every_steps and global_step % log_every_steps == 0:
            avg = {k: run_acc[k] / max(run_n, 1) for k in run_acc}
            logger.info("  [step %d] loss=%.6f", global_step, avg.get("loss", 0.0))
            train_log.log({"global_step": global_step, "epoch": epoch, "stage": stage,
                           "lr": _lr_now(), **avg})
            run_acc, run_n = {}, 0
        if val_every_steps and val_loader is not None and global_step % val_every_steps == 0:
            vm = _validate(epoch)
            train_log.log({"global_step": global_step, "epoch": epoch, "stage": stage,
                           **{f"val_{k}": v for k, v in vm.items()}})
            runner.set_mode(True)
        if save_every_steps and global_step % save_every_steps == 0:
            _save(epoch, float(monitor_loss))
        return bool(step_mode and global_step >= max_steps)

    while not done:
        runner.set_mode(True)
        logger.info("─── Epoch %d%s  (stage=%s) ───",
                    epoch, "" if step_mode else f" / {epochs}", stage)
        ep_acc: Dict[str, float] = {}
        ep_n = 0

        for batch in train_loader:
            metrics = runner.training_step(batch)
            for k, v in metrics.items():
                ep_acc[k] = ep_acc.get(k, 0.0) + float(v)
                run_acc[k] = run_acc.get(k, 0.0) + float(v)
            ep_n += 1
            run_n += 1
            last_metrics = metrics

            if not getattr(runner, "last_step_did_update", True):
                continue  # mid-accumulation micro-step: no global step yet
            global_step += 1
            if _on_global_step(metrics.get("loss", float("inf"))):
                logger.info("Reached max_steps=%d — stopping.", max_steps)
                done = True
                break

        # ── Flush a partial grad-accumulation window at the epoch boundary ────
        # so the epoch's last micro-batches are not dropped. The flush produces a
        # real optimizer step, so it goes through the SAME step-event path (and
        # can itself reach max_steps).
        if not done and hasattr(runner, "flush_pending") and runner.flush_pending():
            global_step += 1
            if _on_global_step(last_metrics.get("loss", float("inf"))):
                logger.info("Reached max_steps=%d (epoch-boundary flush) — stopping.",
                            max_steps)
                done = True

        # ── Epoch-boundary events (epoch mode) ────────────────────────────────
        ep_mean = {k: ep_acc[k] / max(ep_n, 1) for k in ep_acc}
        if not step_mode:
            record: Dict = {"epoch": epoch, "global_step": global_step,
                            "stage": stage, **ep_mean}
            val_metrics = _validate(epoch) if (epoch % val_every == 0) else {}
            if val_metrics:
                record.update({f"val_{k}": v for k, v in val_metrics.items()})
                runner.set_mode(True)
            record["lr"] = _lr_now()
            train_log.log(record)
            monitor = float(val_metrics.get("loss", ep_mean.get("loss", float("inf"))))
            _save(epoch, monitor)
            if epoch >= epochs:
                done = True

        epoch += 1

    # Always persist a final checkpoint in step mode.
    if step_mode:
        _save(epoch - 1, best_metric if best_metric != float("inf") else 0.0)

    logger.info("Training complete [stage=%s]. global_step=%d  best_metric=%.6f",
                stage, global_step, best_metric)
    logger.info("Checkpoints → %s", ckpt_dir)
    logger.info("Training log → %s", log_path)

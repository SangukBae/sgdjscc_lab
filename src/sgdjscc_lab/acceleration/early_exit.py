"""acceleration/early_exit.py – Early stopping for the denoising budget (Phase 5-B).

Two modes:

- **intra_sampler** (Phase 5 upgrade): :func:`run_interruptible_sampling` drives a
  single denoising run, evaluating a score every ``check_interval`` steps and
  terminating the loop the moment the controller says so — so unused steps are
  never executed.  The matching real-model loop is
  ``models.diffusion_wrapper.generate_interruptible``.  The score is supplied by
  the caller (an increasing "quality-like" score), so heuristic / SRS / SRS-v2
  metrics all share one mechanism.

- **checkpoint_legacy**: :func:`evaluate_checkpoints` renders at a few candidate
  step counts and picks the earliest "good enough" one. Kept as a fallback.

``EarlyExitController.should_stop`` is shared by both: it stops when the score
reaches ``srs_threshold`` or stops improving by at least ``improvement_delta``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class EarlyExitConfig:
    srs_threshold: float = 0.8
    improvement_delta: float = 0.01
    min_steps: int = 1


class EarlyExitController:
    """Decide when to stop spending denoising steps.

    Parameters
    ----------
    srs_threshold:
        Stop once the verified SRS reaches this value.
    improvement_delta:
        Stop when SRS gains less than this between consecutive checkpoints.
    min_steps:
        Never stop before this checkpoint index/step.
    """

    def __init__(self, srs_threshold: float = 0.8, improvement_delta: float = 0.01,
                 min_steps: int = 1) -> None:
        self.cfg = EarlyExitConfig(srs_threshold, improvement_delta, min_steps)

    def should_stop(self, history: Sequence[float], step: int) -> Dict:
        """Given an SRS *history* (one per checkpoint) decide whether to stop now.

        Returns ``{"stop": bool, "reason": str}``.
        """
        if not history:
            return {"stop": False, "reason": "no_history"}
        if step < self.cfg.min_steps:
            return {"stop": False, "reason": "below_min_steps"}
        last = float(history[-1])
        if last >= self.cfg.srs_threshold:
            return {"stop": True, "reason": "threshold_reached"}
        if len(history) >= 2:
            improvement = last - float(history[-2])
            if improvement < self.cfg.improvement_delta:
                return {"stop": True, "reason": "diminishing_returns"}
        return {"stop": False, "reason": "continue"}


def run_interruptible_sampling(
    init_state: Any,
    step_fn: Callable[[Any, int, int], Any],
    total_steps: int,
    score_fn: Optional[Callable[[Any, int, int], float]] = None,
    controller: Optional[EarlyExitController] = None,
    check_interval: int = 5,
    min_steps: int = 1,
) -> Dict:
    """Drive a single denoising loop with mid-loop early-exit (intra-sampler).

    Parameters
    ----------
    init_state:
        Opaque loop state (e.g. a dict holding the current latent).
    step_fn:
        ``(state, i, total) -> new_state`` performs one denoising step.
    total_steps:
        Number of steps in the schedule.
    score_fn:
        Optional ``(state, i, total) -> float`` increasing "quality-like" score,
        evaluated at the check points.  When None the loop never exits early.
    controller:
        ``EarlyExitController`` deciding when to stop.
    check_interval:
        Evaluate the score every ``check_interval`` steps (and on the last step).
    min_steps:
        Never exit before this many steps have run.

    Returns
    -------
    dict with ``state`` (final), ``stopped_at`` (steps actually run),
    ``completed`` (bool), ``reason`` and ``history`` (list of ``{step, score}``).
    """
    state = init_state
    log: List[Dict] = []
    history: List[float] = []
    interval = max(int(check_interval), 1)

    for i in range(total_steps):
        state = step_fn(state, i, total_steps)
        step_no = i + 1
        is_check = (step_no % interval == 0) or (step_no == total_steps)
        if score_fn is not None and is_check:
            score = float(score_fn(state, i, total_steps))
            history.append(score)
            log.append({"step": step_no, "score": score})
            if controller is not None and step_no >= min_steps and step_no < total_steps:
                decision = controller.should_stop(history, step_no)
                if decision.get("stop"):
                    return {"state": state, "stopped_at": step_no, "completed": False,
                            "reason": decision.get("reason"), "history": log}

    return {"state": state, "stopped_at": total_steps, "completed": True,
            "reason": "completed", "history": log}


def evaluate_checkpoints(
    render_fn: Callable[[int], object],
    eval_fn: Callable[[object], float],
    checkpoints: Sequence[int],
    controller: Optional[EarlyExitController] = None,
) -> Dict:
    """Render + score increasing step checkpoints, stopping early when allowed.

    Parameters
    ----------
    render_fn:
        ``step -> reconstruction`` (e.g. run the pipeline at that step count).
    eval_fn:
        ``reconstruction -> srs`` verified score.
    checkpoints:
        Increasing list of candidate step counts (e.g. ``[5, 10, 20, 50]``).
    controller:
        ``EarlyExitController`` (created with defaults if None).

    Returns
    -------
    dict with ``chosen_step``, ``chosen_recon``, ``history`` (list of
    ``{step, srs}``) and ``reason``.
    """
    controller = controller or EarlyExitController()
    history: List[float] = []
    log: List[Dict] = []
    chosen_step = checkpoints[-1] if checkpoints else 0
    chosen_recon = None
    reason = "exhausted"

    for idx, step in enumerate(checkpoints):
        recon = render_fn(step)
        srs = float(eval_fn(recon))
        history.append(srs)
        log.append({"step": step, "srs": srs})
        chosen_step, chosen_recon = step, recon
        decision = controller.should_stop(history, step=idx + 1)
        if decision["stop"]:
            reason = decision["reason"]
            break

    return {"chosen_step": chosen_step, "chosen_recon": chosen_recon,
            "history": log, "reason": reason}

"""acceleration/latency_profiler.py – Wall-clock latency measurement (Phase 5-B).

Ports the timing idea from the LDM SemCom ``t_calculate.py`` into a small,
dependency-free profiler for the SGD-JSCC pipeline.  Reports end-to-end latency,
named-section latency (e.g. decoder), per-step average and effective step count,
so DDIM / few-step ablations can be compared on a quality-vs-latency curve.

CUDA timings are synchronised when a CUDA device is active so GPU work is not
under-counted.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

try:
    import torch
    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    _HAS_TORCH = False


def _sync():
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


class LatencyProfiler:
    """Accumulate named-section wall-clock timings.

    Usage
    -----
    >>> prof = LatencyProfiler()
    >>> with prof.section("decoder"):
    ...     run_decoder()
    >>> prof.report()
    """

    def __init__(self) -> None:
        self._sections: Dict[str, List[float]] = {}

    @contextmanager
    def section(self, name: str):
        _sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            dt = time.perf_counter() - t0
            self._sections.setdefault(name, []).append(dt)

    def add(self, name: str, seconds: float) -> None:
        self._sections.setdefault(name, []).append(float(seconds))

    def report(self, steps: Optional[int] = None) -> Dict:
        """Return a latency summary dict.

        Parameters
        ----------
        steps:
            Effective denoising step count, used to compute ``per_step`` from the
            total latency.
        """
        out: Dict = {}
        total = 0.0
        for name, vals in self._sections.items():
            s = sum(vals)
            out[f"{name}_total_s"] = round(s, 6)
            out[f"{name}_mean_s"] = round(s / max(len(vals), 1), 6)
            out[f"{name}_calls"] = len(vals)
            total += s
        out["total_latency_s"] = round(total, 6)
        if steps:
            out["effective_steps"] = int(steps)
            out["per_step_s"] = round(total / max(int(steps), 1), 6)
        return out

    def reset(self) -> None:
        self._sections.clear()


def profile_callable(
    fn: Callable,
    *args,
    n_warmup: int = 1,
    n_runs: int = 3,
    steps: Optional[int] = None,
    **kwargs,
) -> Dict:
    """Time ``fn(*args, **kwargs)`` over warmup + measured runs.

    Returns ``{mean_s, min_s, max_s, runs, per_step_s?}`` and the last result
    under ``result``.
    """
    for _ in range(max(n_warmup, 0)):
        fn(*args, **kwargs)

    times: List[float] = []
    result = None
    for _ in range(max(n_runs, 1)):
        _sync()
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        _sync()
        times.append(time.perf_counter() - t0)

    mean_s = sum(times) / len(times)
    out = {
        "mean_s": round(mean_s, 6),
        "min_s": round(min(times), 6),
        "max_s": round(max(times), 6),
        "runs": len(times),
        "result": result,
    }
    if steps:
        out["per_step_s"] = round(mean_s / max(int(steps), 1), 6)
        out["effective_steps"] = int(steps)
    return out

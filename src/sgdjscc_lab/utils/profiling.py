"""utils/profiling.py – Lightweight call-count / progress profiler (opt-in).

Built for diagnosing why real-model video evaluation runs are slow: it counts
how many times the expensive model calls (diffusion sampling, BLIP2 caption,
CLIP encode) actually happen, and streams a progress file to disk during a
long run so it can be tailed without waiting for the process to finish.

Design constraints
-------------------
- Zero sgdjscc_lab imports (stdlib only) so it can be imported from any leaf
  module (infer_pipeline, text_extractor, clip_score, ...) without touching
  the import graph or risking circular imports.
- No-op by default: call sites call the free functions
  (``record_diffusion_call`` / ``record_blip2_call`` / ``record_clip_call``),
  which do nothing unless a profiler has been installed via ``set_active``.
  This means the instrumentation is inert for every existing script/test that
  never calls ``set_active`` — no numeric or behavioural change.
- Per-process only. Each evaluate_video.py invocation (including each worker
  in a parallel batch) gets its own profiler instance; there is no cross-
  process aggregation here (batch-level aggregation is done by summarizing
  each run's profiling.json after the fact — see
  scripts/run_speed_experiment.py).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_lock = threading.Lock()
_active: "RunProfiler | None" = None


def set_active(profiler: "Optional[RunProfiler]") -> None:
    """Install (or clear, with None) the process-wide active profiler."""
    global _active
    with _lock:
        _active = profiler


def get_active() -> "Optional[RunProfiler]":
    with _lock:
        return _active


def record_diffusion_call(steps: int = 0) -> None:
    """One diffusion sampling call (one patch's denoising loop). ``steps`` is
    the per-call denoising step count, tallied separately as "diffusion_steps"
    — the *call* count is what a frame's diffusion_calls column reports."""
    p = get_active()
    if p is not None:
        p.record("diffusion_calls", steps=1)
        p.record("diffusion_steps", steps=steps)


def record_blip2_call(n: int = 1) -> None:
    """One BLIP2 caption-generation call over a batch of ``n`` image(s)."""
    p = get_active()
    if p is not None:
        p.record("blip2_calls", steps=1)
        p.record("blip2_images", steps=n)


def record_clip_call(kind: str = "image", n: int = 1) -> None:
    """One CLIP encode call over ``n`` item(s). ``kind``: 'image' or 'text' —
    tallied separately (text calls are the ones the embedding cache in
    evaluators/clip_score.py can eliminate)."""
    p = get_active()
    if p is not None:
        p.record(f"clip_{kind}_calls", steps=1)
        p.record(f"clip_{kind}_n", steps=n)


@dataclass
class FrameRecordLite:
    index: int
    decision: Optional[str]
    elapsed_sec: float
    diffusion_calls: int
    blip2_calls: int
    clip_calls: int


@dataclass
class RunProfiler:
    """Accumulates call counts + per-frame timing for one evaluate_video.py run.

    Usage::

        prof = RunProfiler(video="01_person_walk", total_frames=100,
                            progress_path="outputs/.../progress.json")
        profiling.set_active(prof)
        with prof.frame(index=i) as ctx:
            ... run reconstruct_fn / packet_fn ...
            ctx.decision = "keyframe"
        prof.write_summary("outputs/.../profiling_summary.json")
    """

    video: str = ""
    total_frames: int = 0
    progress_path: Optional[str] = None
    progress_every: int = 1          # flush progress.json every N frames
    progress_every_sec: float = 5.0  # ...or at least this often, whichever first

    counters: Dict[str, int] = field(default_factory=dict)
    frame_records: List[FrameRecordLite] = field(default_factory=list)
    _t_start: float = field(default_factory=time.monotonic)
    _last_flush_t: float = field(default_factory=time.monotonic)
    _frame_counters_snapshot: Dict[str, int] = field(default_factory=dict)

    def record(self, name: str, steps: int = 1) -> None:
        with _lock:
            self.counters[name] = self.counters.get(name, 0) + steps

    class _FrameCtx:
        def __init__(self, outer: "RunProfiler", index: int):
            self._outer = outer
            self.index = index
            self.decision: Optional[str] = None
            self._t0 = 0.0
            self._before: Dict[str, int] = {}

        def __enter__(self):
            self._t0 = time.monotonic()
            self._before = dict(self._outer.counters)
            return self

        def __exit__(self, exc_type, exc, tb):
            elapsed = time.monotonic() - self._t0
            after = self._outer.counters
            def _delta(key):
                return after.get(key, 0) - self._before.get(key, 0)

            rec = FrameRecordLite(
                index=self.index,
                decision=self.decision,
                elapsed_sec=round(elapsed, 4),
                diffusion_calls=_delta("diffusion_calls"),
                blip2_calls=_delta("blip2_calls"),
                clip_calls=_delta("clip_image_calls") + _delta("clip_text_calls"),
            )
            self._outer.frame_records.append(rec)
            self._outer._maybe_flush()
            return False

    def frame(self, index: int) -> "RunProfiler._FrameCtx":
        return RunProfiler._FrameCtx(self, index)

    def _maybe_flush(self) -> None:
        if self.progress_path is None:
            return
        n = len(self.frame_records)
        now = time.monotonic()
        due_count = (n % max(1, self.progress_every)) == 0
        due_time = (now - self._last_flush_t) >= self.progress_every_sec
        if not (due_count or due_time or n == self.total_frames):
            return
        self._last_flush_t = now
        self.write_progress(self.progress_path)

    def _snapshot(self) -> dict:
        n = len(self.frame_records)
        elapsed = time.monotonic() - self._t_start
        last = self.frame_records[-1] if self.frame_records else None
        avg_frame_sec = (elapsed / n) if n else None
        eta_sec = (
            avg_frame_sec * max(0, self.total_frames - n)
            if avg_frame_sec is not None and self.total_frames
            else None
        )
        return {
            "video": self.video,
            "pid": os.getpid(),
            "frames_done": n,
            "total_frames": self.total_frames,
            "elapsed_sec": round(elapsed, 2),
            "avg_frame_sec": round(avg_frame_sec, 4) if avg_frame_sec is not None else None,
            "eta_sec": round(eta_sec, 1) if eta_sec is not None else None,
            "last_frame": (
                {
                    "index": last.index, "decision": last.decision,
                    "elapsed_sec": last.elapsed_sec,
                } if last is not None else None
            ),
            "counters": dict(self.counters),
            "updated_at": time.time(),
        }

    def write_progress(self, path) -> None:
        """Atomically write a small JSON progress snapshot (safe to tail mid-run)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(self._snapshot(), indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def write_summary(self, path) -> None:
        """Final per-run summary: totals + full per-frame call/timing table."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = self._snapshot()
        summary["frame_records"] = [
            {
                "index": r.index, "decision": r.decision, "elapsed_sec": r.elapsed_sec,
                "diffusion_calls": r.diffusion_calls, "blip2_calls": r.blip2_calls,
                "clip_calls": r.clip_calls,
            }
            for r in self.frame_records
        ]
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

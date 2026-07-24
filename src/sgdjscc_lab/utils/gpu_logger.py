"""utils/gpu_logger.py – Background nvidia-smi sampler for batch GPU runs.

Polls ``nvidia-smi --query-gpu=...`` on an interval from a background thread
and appends rows to a CSV, so a multi-hour/multi-GPU batch run has a
utilization/memory timeline to look at afterwards without needing pynvml
(not installed in the sgdjscc docker image — see remote-server-docker memory
notes) or any GPU-bound Python dependency. Silently degrades to "not
sampled" (writes nothing, logs one warning) when ``nvidia-smi`` is not on
PATH — e.g. on a CPU-only dev machine — so callers can unconditionally start
it without an availability check first.
"""

from __future__ import annotations

import csv
import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_FIELDS = ["timestamp", "index", "name", "utilization_gpu_pct", "memory_used_mib", "memory_total_mib"]


def _query_once() -> Optional[List[dict]]:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    ts = time.time()
    rows = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        idx, name, util, mem_used, mem_total = parts
        rows.append({
            "timestamp": ts, "index": idx, "name": name,
            "utilization_gpu_pct": util, "memory_used_mib": mem_used,
            "memory_total_mib": mem_total,
        })
    return rows


class GPULogger:
    """``start()`` / ``stop()`` a background nvidia-smi sampling thread.

    Usage::

        gl = GPULogger(csv_path, interval_sec=10)
        gl.start()
        ... run the batch ...
        gl.stop()
    """

    def __init__(self, csv_path, interval_sec: float = 10.0) -> None:
        self.csv_path = Path(csv_path)
        self.interval_sec = float(interval_sec)
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._available = shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if not self._available:
            logger.warning("nvidia-smi not found on PATH — GPU utilization logging disabled.")
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        wrote_header = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_FIELDS)
            if not wrote_header:
                writer.writeheader()
                fh.flush()
            while not self._stop_evt.is_set():
                try:
                    rows = _query_once()
                except Exception as exc:  # noqa: BLE001 — a telemetry thread must never
                    # take the main run down (e.g. nvidia-smi behaving unexpectedly under
                    # something that intercepts subprocess.run, such as a test harness).
                    logger.debug("GPU sampling iteration failed: %s", exc)
                    rows = None
                if rows:
                    for r in rows:
                        writer.writerow(r)
                    fh.flush()
                self._stop_evt.wait(self.interval_sec)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 5)

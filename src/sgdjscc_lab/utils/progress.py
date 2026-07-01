"""utils/progress.py – rank-0 tqdm console progress for the training loop.

The JSONL ``TrainingLogger`` (machine record) and checkpoint policy are unchanged;
this module only improves the *human-facing* console output:

- a single ``tqdm`` progress bar (ASCII, SSH/tmux/nohup friendly — chosen over
  ``rich`` for stability under redirected / non-tty output),
- **rank 0 only** under DDP (other ranks get a silent no-op object),
- step mode (``global_step / max_steps``) and epoch mode
  (``batch_idx / len(loader)``),
- a compact postfix (loss / lr / step_time / it/s / img/s / ETA / val_in /
  save_in / stage-specific losses) plus live GPU util+mem via ``pynvml`` with a
  graceful fallback (``na`` / torch memory) when it is unavailable,
- progress-safe messages (``write`` → ``tqdm.write``) so checkpoint / validation
  notices do not corrupt the bar.

Everything degrades to a no-op when tqdm is missing or when not on rank 0, so the
training loop can call these unconditionally.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from sgdjscc_lab import distributed as ddp

logger = logging.getLogger(__name__)

# Stage-specific extra loss keys surfaced in the postfix (core "loss" always
# shown first).  Keep the lists short; the structure is the extension point for
# new stages — add a key here and it appears automatically when present.
_STAGE_EXTRA_KEYS: Dict[str, List[str]] = {
    "jscc":       ["loss_mse", "loss_gan", "loss_disc"],
    "text_dm":    ["loss_masked", "loss_unmasked"],
    "edge_codec": ["loss_bce", "loss_dice"],
    "controlnet": ["loss_mse"],
}

_BAR_FORMAT = "{desc} |{bar}| {n_fmt}/{total_fmt} {percentage:5.2f}% {postfix}"


def _stage_extra_keys(stage: str) -> List[str]:
    return _STAGE_EXTRA_KEYS.get(str(stage), [])


def _fmt_eta(seconds: Optional[float]) -> str:
    """Compact ETA like ``2d03h`` / ``3h04m`` / ``5m10s`` / ``42s``."""
    if seconds is None or seconds != seconds or seconds < 0 or seconds == float("inf"):
        return "?"
    s = int(seconds)
    d, r = divmod(s, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    if d:
        return f"{d}d{h:02d}h"
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ─────────────────────────────────────────────────────────────────────────────
# GPU stats (pynvml first, torch fallback, cached/periodic refresh)
# ─────────────────────────────────────────────────────────────────────────────

class GpuStatsSampler:
    """Sample GPU utilization + used memory cheaply.

    Utilization needs ``pynvml`` (torch cannot report it); memory falls back to
    ``torch.cuda`` when pynvml is absent.  To avoid per-step overhead the NVML
    query only runs every *refresh_every* calls; cached values are reused in
    between.  ``peak`` is intentionally not tracked (per the display spec).

    Multi-GPU note: this samples the rank's *local* device only; the constructor
    signature (device) keeps room to extend to a per-rank aggregate later.
    """

    def __init__(self, device, refresh_every: int = 10) -> None:
        self._refresh = max(1, int(refresh_every))
        self._i = 0
        self._util: Optional[int] = None
        self._mem: Optional[float] = None      # GiB used
        self._nvml = None
        self._handle = None
        # torch fallback index
        self._torch_idx = getattr(device, "index", None)
        self._torch_ok = False
        try:
            import torch
            self._torch = torch
            self._torch_ok = torch.cuda.is_available() and str(getattr(device, "type", device)) != "cpu"
            if self._torch_idx is None and self._torch_ok:
                self._torch_idx = torch.cuda.current_device()
        except Exception:
            self._torch = None
        try:
            import pynvml
            pynvml.nvmlInit()
            idx = self._torch_idx if self._torch_idx is not None else 0
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(int(idx))
            self._nvml = pynvml
        except Exception:
            self._nvml = None  # graceful fallback: util=na, mem from torch

    def sample(self) -> Dict[str, str]:
        """Return ``{"gpu": "91%"|"na", "mem": "18.4G"|"na"}`` (refresh-cached)."""
        if self._nvml is not None:
            if self._i % self._refresh == 0:
                try:
                    u = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                    m = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                    self._util = int(u.gpu)
                    self._mem = float(m.used) / (1024 ** 3)
                except Exception:
                    pass
            self._i += 1
            util = f"{self._util}%" if self._util is not None else "na"
            mem = f"{self._mem:.1f}G" if self._mem is not None else "na"
            return {"gpu": util, "mem": mem}
        # Fallback: no pynvml → utilization unavailable, memory via torch.
        mem = "na"
        if self._torch_ok:
            try:
                used = self._torch.cuda.memory_reserved(self._torch_idx) / (1024 ** 3)
                mem = f"{used:.1f}G"
            except Exception:
                pass
        return {"gpu": "na", "mem": mem}

    def shutdown(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None


# ─────────────────────────────────────────────────────────────────────────────
# Training progress (rank-0 tqdm)
# ─────────────────────────────────────────────────────────────────────────────

class TrainProgress:
    """Console progress for the training loop; no-op off rank 0 / without tqdm.

    Step mode → one persistent bar over ``max_steps`` (advanced per global
    optimizer step).  Epoch mode → a fresh bar per epoch over the batches.
    """

    def __init__(
        self,
        *,
        stage: str,
        step_mode: bool,
        max_steps: int,
        epochs: int,
        grad_accum: int,
        world_size: int,
        device,
        refresh_gpu_every: int = 10,
    ) -> None:
        self.stage = str(stage)
        self.step_mode = bool(step_mode)
        self.max_steps = int(max_steps)
        self.epochs = int(epochs)
        self.grad_accum = max(1, int(grad_accum))
        self.world_size = max(1, int(world_size))
        self.is_rank0 = ddp.is_rank0()

        self._tqdm = None
        if self.is_rank0:
            try:
                from tqdm import tqdm
                self._tqdm = tqdm
            except Exception:
                self._tqdm = None
        self.enabled = self._tqdm is not None

        self._bar = None
        self._epoch = 0
        self._t_prev: Optional[float] = None
        self._ema: Optional[float] = None
        self._alpha = 0.3
        self._batch_samples = self.world_size
        self._gpu = GpuStatsSampler(device, refresh_gpu_every) if self.enabled else None

    # -- lifecycle ------------------------------------------------------------
    def _new_bar(self, total: int, position: int = 0):
        return self._tqdm(
            total=max(int(total), 0),
            desc=self._desc(),
            position=position,
            leave=True,
            ascii=True,
            dynamic_ncols=True,
            mininterval=1.0,
            bar_format=_BAR_FORMAT,
        )

    def _desc(self) -> str:
        return f"stage={self.stage} ep={self._epoch} ga={self.grad_accum}"

    def begin_run(self) -> None:
        """Create the persistent step-mode bar (no-op in epoch mode)."""
        if self.enabled and self.step_mode:
            self._bar = self._new_bar(self.max_steps)
            self._t_prev = time.time()

    def begin_epoch(self, epoch: int, n_batches: int) -> None:
        self._epoch = int(epoch)
        if not self.enabled:
            return
        if self.step_mode:
            if self._bar is not None:
                self._bar.set_description_str(self._desc())
        else:
            self._bar = self._new_bar(n_batches)
            self._t_prev = time.time()

    def end_epoch(self) -> None:
        if self.enabled and not self.step_mode and self._bar is not None:
            self._bar.close()
            self._bar = None

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        if self._gpu is not None:
            self._gpu.shutdown()

    # -- updates --------------------------------------------------------------
    def _tick(self) -> float:
        now = time.time()
        dt = (now - self._t_prev) if self._t_prev is not None else 0.0
        self._t_prev = now
        if dt > 0:
            self._ema = dt if self._ema is None else (self._alpha * dt + (1 - self._alpha) * self._ema)
        return dt

    def after_batch(
        self,
        *,
        batch_idx: int,
        n_batches: int,
        global_step: int,
        did_update: bool,
        metrics: Dict[str, float],
        lr: float,
        val_in: Optional[int] = None,
        save_in: Optional[int] = None,
        batch_samples: Optional[int] = None,
    ) -> None:
        """Advance the bar: per global step (step mode) or per batch (epoch mode)."""
        if not self.enabled or self._bar is None:
            return
        if batch_samples:
            self._batch_samples = int(batch_samples)
        advance = did_update if self.step_mode else True
        if not advance:
            return
        dt = self._tick()
        self._bar.update(1)
        self._render(global_step=global_step, batch_idx=batch_idx, n_batches=n_batches,
                     metrics=metrics, lr=lr, dt=dt, val_in=val_in, save_in=save_in)

    def _render(self, *, global_step, batch_idx, n_batches, metrics, lr, dt, val_in, save_in) -> None:
        step_time = self._ema if self._ema else dt
        its = (1.0 / step_time) if step_time and step_time > 0 else 0.0
        if self.step_mode:
            samples = self._batch_samples * self.grad_accum
            remaining = max(self.max_steps - global_step, 0)
        else:
            samples = self._batch_samples
            rem_epoch = max(n_batches - (batch_idx + 1), 0)
            rem_full = max(self.epochs - self._epoch, 0) * max(n_batches, 0)
            remaining = rem_epoch + rem_full
        sps = (samples / step_time) if step_time and step_time > 0 else 0.0
        eta = _fmt_eta(remaining * step_time) if step_time and step_time > 0 else "?"

        parts = [
            f"loss={float(metrics.get('loss', 0.0)):.4f}",
            f"lr={float(lr):.1e}",
            f"step={step_time:.2f}s",
            f"{its:.1f}it/s",
            f"{sps:.0f}img/s",
            f"ETA={eta}",
        ]
        if val_in is not None:
            parts.append(f"val_in={val_in}")
        if save_in is not None:
            parts.append(f"save_in={save_in}")
        for k in _stage_extra_keys(self.stage):
            if k in metrics:
                parts.append(f"{k[5:] if k.startswith('loss_') else k}={float(metrics[k]):.3f}")
        if self._gpu is not None:
            g = self._gpu.sample()
            parts.append(f"gpu={g['gpu']}")
            parts.append(f"mem={g['mem']}")
        self._bar.set_postfix_str(" ".join(parts))

    # -- messages & validation ------------------------------------------------
    def write(self, msg: str) -> None:
        """Progress-safe, rank-0-only message (bar-aware)."""
        if not self.is_rank0:
            return
        if self._tqdm is not None:
            self._tqdm.write(msg)
        else:
            logger.info(msg)

    def open_val_bar(self, total: int, epoch: int):
        """Transient validation bar (rank 0 only); returns None when disabled."""
        if not self.enabled:
            return None
        return self._tqdm(
            total=max(int(total), 0),
            desc=f"val ep={epoch}",
            position=1,
            leave=False,
            ascii=True,
            dynamic_ncols=True,
            mininterval=1.0,
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} {postfix}",
        )

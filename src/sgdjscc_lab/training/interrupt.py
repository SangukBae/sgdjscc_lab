"""training/interrupt.py – SIGINT/SIGTERM-safe checkpointing for training.

Long training runs are frequently stopped by Ctrl-C (SIGINT) or by a scheduler /
container runtime sending SIGTERM.  Without handling, the process dies between
checkpoint cadences and the trailing progress since the last ``latest.pth`` /
``epoch_*.pth`` save is lost.

:class:`InterruptHandler` is a tiny, correctness-first helper that makes the stop
*graceful*:

* On the first SIGINT/SIGTERM it only **sets a flag** (async-signal-safe — no
  torch.save inside the handler). The training loop polls :meth:`requested` at a
  safe point (batch / optimizer-step boundary) and saves an interrupt checkpoint
  there, on the main thread, when no backward/optimizer step is in flight.
* On a **second** signal it restores the previous handler and re-raises so an
  impatient operator can still force-kill.

DDP note: every rank installs the handler (``torchrun`` forwards the signal to
the whole process group, so all ranks observe it and break together), but only
rank 0 writes the checkpoint file — matching ``save_checkpoint`` / the rest of
the training I/O.  The actual file write is done by the caller, not here, so this
module stays free of any checkpoint-format knowledge.
"""

from __future__ import annotations

import logging
import signal
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Signals we treat as a graceful-stop request. SIGTERM covers scheduler / docker
# stop; SIGINT covers Ctrl-C. Both exist on POSIX; SIGTERM may be absent on some
# platforms, so we install defensively.
_STOP_SIGNALS = ("SIGINT", "SIGTERM")


class InterruptHandler:
    """Context manager that turns SIGINT/SIGTERM into a pollable stop flag.

    Usage::

        with InterruptHandler() as stop:
            for batch in loader:
                train_step(batch)
                if stop.requested:
                    save_interrupt_checkpoint(...)
                    break

    The handler is only armed while the ``with`` block is active; on exit the
    previous handlers are restored, so nested / repeated training calls in the
    same process (e.g. tests) do not leak handler state.
    """

    def __init__(self) -> None:
        self._requested = False
        self._signum: Optional[int] = None
        self._prev = {}          # signalnum -> previous handler
        self._installed = False

    # ── public state ──────────────────────────────────────────────────────────
    @property
    def requested(self) -> bool:
        """True once a stop signal has been received."""
        return self._requested

    @property
    def signal_name(self) -> str:
        if self._signum is None:
            return "?"
        try:
            return signal.Signals(self._signum).name
        except (ValueError, AttributeError):    # pragma: no cover
            return str(self._signum)

    # ── handler ────────────────────────────────────────────────────────────────
    def _handle(self, signum, _frame) -> None:
        if self._requested:
            # Second signal: the operator is insisting. Restore the default and
            # re-raise so the process can be force-killed immediately.
            logger.warning("Second %s received — aborting without a further save.",
                           signal.Signals(signum).name)
            self._restore()
            raise KeyboardInterrupt
        self._requested = True
        self._signum = signum
        # Logging from a signal handler is not strictly async-signal-safe, but is
        # the pragmatic norm and only fires once per run at a stop request.
        logger.warning(
            "Received %s — will save an interrupt checkpoint at the next safe "
            "point and stop. Send it again to force-quit.",
            signal.Signals(signum).name,
        )

    def install(self) -> "InterruptHandler":
        """Install the handlers (main thread only; no-op elsewhere)."""
        if self._installed:
            return self
        # signal.signal() only works on the main thread of the main interpreter.
        if threading.current_thread() is not threading.main_thread():
            logger.info("InterruptHandler: not on the main thread — signal-based "
                        "graceful stop disabled for this run.")
            return self
        for name in _STOP_SIGNALS:
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                self._prev[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle)
            except (ValueError, OSError) as exc:    # pragma: no cover
                logger.info("InterruptHandler: could not install %s handler: %s", name, exc)
        self._installed = bool(self._prev)
        return self

    def uninstall(self) -> None:
        """Restore the previous signal handlers (safe to call more than once)."""
        self._restore()

    def _restore(self) -> None:
        for sig, prev in self._prev.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):            # pragma: no cover
                pass
        self._prev.clear()
        self._installed = False

    # ── context manager ────────────────────────────────────────────────────────
    def __enter__(self) -> "InterruptHandler":
        return self.install()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._restore()
        return False

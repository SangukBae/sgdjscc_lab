"""utils/text_progress.py – reusable single-line text progress bar.

A dependency-free ``|████----| 42.3% (n/total)`` renderer shared by the offline
data-prep CLIs (``scripts/generate_captions.py``, ``scripts/prepare_muge_edges.py``)
so their per-unit progress reads identically.

Kept deliberately separate from ``utils/progress.py`` (the tqdm-based training
``TrainProgress``, which imports torch/DDP): this module imports nothing heavy,
so the CPU-only caption modes (``fixed`` / ``filename``) never pull torch in just
to format a log line.

Line-update strategy
--------------------
``UnitProgress`` renders one work unit's progress on a *single* terminal line:

* **TTY** – the line is overwritten in place with carriage-return + ANSI
  clear-to-end-of-line (``\\r … \\033[K``) and **no** per-tick newline, so a busy
  unit no longer stacks a fresh log row for every update. ``close()`` commits the
  final state with a trailing newline, leaving a permanent 100% line on screen.
* **non-TTY** (pipe / file / CI) – falls back to the historical line-by-line
  logging (one ``INFO`` row per throttled tick via the caller's logger), which is
  what redirected output / log files want and which survives parallel-worker
  interleaving.

Each update is written with a single ``write()`` call carrying the full
``[stage][gpu=..][label][i/N]`` prefix, so when several workers share one
terminal the writes stay atomic per line (last writer wins the live line) and
every completed unit is committed as its own labelled row — no per-tick spam.
"""

from __future__ import annotations

import os
import sys

BAR_FILL = "█"
BAR_EMPTY = "-"
DEFAULT_WIDTH = 30

# CR returns to column 0; ESC[K clears from the cursor to end of line so a
# shorter new line never leaves stale tail characters from a longer old one.
_CR = "\r"
_CLEAR_EOL = "\033[K"


def render_bar(processed: int, total: int, *, width: int = DEFAULT_WIDTH,
               fill: str = BAR_FILL, empty: str = BAR_EMPTY) -> str:
    """``|████----| 42.3% (n/total)`` for *processed* of *total* (total 0 → 100%)."""
    processed = max(0, int(processed))
    total = max(0, int(total))
    frac = (processed / total) if total else 1.0
    frac = min(1.0, max(0.0, frac))
    width = max(1, int(width))
    filled = min(width, max(0, int(round(frac * width))))
    bar = fill * filled + empty * (width - filled)
    return f"|{bar}| {100.0 * frac:5.1f}% ({processed}/{total})"


def format_unit_progress(*, stage: str, gpu, label: str, unit_index, unit_total,
                         processed: int, total: int, width: int = DEFAULT_WIDTH) -> str:
    """One identifiable per-unit progress line, e.g.::

        [caption][gpu=0][sa1b_images/train][5/60] |████----| 42.3% (4732/11186)

    ``gpu`` is omitted when None/empty so single-GPU / standalone runs stay clean.
    """
    prefix = f"[{stage}]"
    if gpu is not None and str(gpu) != "":
        prefix += f"[gpu={gpu}]"
    prefix += f"[{label}][{unit_index}/{unit_total}]"
    return f"{prefix} {render_bar(processed, total, width=width)}"


def stream_is_tty(stream) -> bool:
    """True when *stream* is an interactive terminal we may draw ANSI onto.

    Only ``TERM=dumb`` (emacs shells, some CI) disables in-place drawing; an
    *unset* TERM is common on real ptys (e.g. a non-login ``docker exec``) and
    still supports CR / clear-to-EOL, so it stays enabled. Non-ttys (pipes,
    files) always fall the caller back to plain line-by-line logging.
    """
    try:
        if not stream.isatty():
            return False
    except Exception:
        return False
    return os.environ.get("TERM", "") != "dumb"


class UnitProgress:
    """In-place single-line progress reporter for one work unit.

    Parameters mirror :func:`format_unit_progress`.  Call :meth:`update` on your
    throttled cadence and :meth:`close` once when the unit finishes.

    * TTY: :meth:`update` overwrites one line in place; :meth:`close` commits the
      final line with a newline so the finished bar stays visible.
    * non-TTY: :meth:`update` logs one line per call through *logger* (historical
      behaviour); :meth:`close` is a no-op (the caller already logs the final
      tick), so redirected output is byte-for-byte the old format.

    ``label`` may be ``None`` for standalone runs → a plain ``  42.3% (n/total)``
    line (matching the pre-existing no-label logging).
    """

    def __init__(self, *, stage, total, label=None, gpu=None, unit_index=1,
                 unit_total=1, width: int = DEFAULT_WIDTH, stream=None, logger=None):
        self.stage = stage
        self.total = max(0, int(total))
        self.label = label
        self.gpu = gpu
        self.unit_index = unit_index
        self.unit_total = unit_total
        self.width = width
        self.stream = stream if stream is not None else sys.stderr
        self.logger = logger
        self.tty = stream_is_tty(self.stream)

    def _line(self, processed: int) -> str:
        if self.label:
            return format_unit_progress(
                stage=self.stage, gpu=self.gpu, label=self.label,
                unit_index=self.unit_index, unit_total=self.unit_total,
                processed=processed, total=self.total, width=self.width)
        pct = (100.0 * processed / self.total) if self.total else 100.0
        return f"  {pct:.1f}% ({processed}/{self.total})"

    def update(self, processed: int) -> None:
        """Render *processed*/total — overwrite in place (TTY) or log a row (non-TTY)."""
        line = self._line(processed)
        if self.tty:
            self.stream.write(_CR + line + _CLEAR_EOL)
            self.stream.flush()
        elif self.logger is not None:
            self.logger.info("%s", line)
        else:
            print(line, file=self.stream, flush=True)

    def close(self, processed: int | None = None) -> None:
        """Commit the final state. TTY: rewrite + newline; non-TTY: no-op."""
        if not self.tty:
            return
        if processed is None:
            processed = self.total
        self.stream.write(_CR + self._line(processed) + _CLEAR_EOL + "\n")
        self.stream.flush()

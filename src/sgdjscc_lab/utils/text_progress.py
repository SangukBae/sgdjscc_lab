"""utils/text_progress.py – reusable single-line text progress bar.

A dependency-free ``|████----| 42.3% (n/total)`` renderer shared by the offline
data-prep CLIs (``scripts/generate_captions.py``, ``scripts/prepare_muge_edges.py``)
so their per-unit progress reads identically.

Kept deliberately separate from ``utils/progress.py`` (the tqdm-based training
``TrainProgress``, which imports torch/DDP): this module imports nothing heavy,
so the CPU-only caption modes (``fixed`` / ``filename``) never pull torch in just
to format a log line. Plain strings (no ANSI overwrite) also survive
parallel-worker interleaving and redirected / non-tty output.
"""

from __future__ import annotations

BAR_FILL = "█"
BAR_EMPTY = "-"
DEFAULT_WIDTH = 30


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

"""Helpers for quarantining unreadable images during dataset preparation."""

from __future__ import annotations

import shutil
from pathlib import Path


def quarantine_bad_image(path: str | Path, bad_root: str | Path | None = None) -> Path:
    """Move *path* into a ``_bad_images`` tree and return the destination path.

    Destination policy:
    - when *bad_root* is given, preserve the filename under that root;
    - otherwise, if the file lives somewhere under a ``data/`` directory, move it
      under ``data/_bad_images/<original-relative-path>``;
    - else fall back to ``<parent>/_bad_images/<filename>``.
    """
    src = Path(path)
    if bad_root is not None:
        dst = Path(bad_root) / src.name
    else:
        data_root = next((p for p in (src.parent, *src.parents) if p.name == "data"), None)
        if data_root is not None:
            dst = data_root / "_bad_images" / src.relative_to(data_root)
        else:
            dst = src.parent / "_bad_images" / src.name

    dst = _unique_destination(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return dst


def _unique_destination(dst: Path) -> Path:
    """Return a non-existing path by appending ``.N`` before the suffix."""
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    idx = 1
    while True:
        candidate = dst.with_name(f"{stem}.{idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1

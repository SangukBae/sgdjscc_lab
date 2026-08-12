"""_sgdjscc.py – Shared helper for injecting SGDJSCC onto sys.path.

All sub-modules that import from SGDJSCC should call ensure_sgdjscc_on_path()
instead of repeating the four-level .parent chain inline.

Resolution order for the *external*, read-only SGDJSCC repo (never modified
by sgdjscc_lab):

1. ``SGDJSCC_ROOT`` env var, if set — an explicit override, useful when the
   two repos aren't checked out as siblings (containers, alternate layouts).
2. The legacy sibling layout: ``.../sgdjscc_lab/src/sgdjscc_lab/_sgdjscc.py``
   climbs four levels to ``Semantic/`` and looks for ``SGDJSCC/`` there.
3. Auto-discovery: walk up from this file (and from the current working
   directory) looking for a directory containing both an ``SGDJSCC/`` folder
   and a ``sgdjscc_lab/`` folder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _looks_like_sgdjscc_repo(path: Path) -> bool:
    # A cheap existence check — SGDJSCC/models/ is present in every checkout.
    return path.is_dir() and (path / "models").is_dir()


def _auto_discover_sgdjscc_root() -> Path | None:
    candidates = [Path(__file__).resolve(), Path.cwd()]
    seen = set()
    for start in candidates:
        for parent in [start, *start.parents]:
            if parent in seen:
                continue
            seen.add(parent)
            candidate = parent / "SGDJSCC"
            if _looks_like_sgdjscc_repo(candidate):
                return candidate
    return None


def _resolve_sgdjscc_root() -> Path:
    env = os.environ.get("SGDJSCC_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    # Legacy layout: .../sgdjscc_lab/src/sgdjscc_lab/_sgdjscc.py
    #   Climbing up: sgdjscc_lab/ (pkg) → src/ → sgdjscc_lab/ (repo) → Semantic/
    legacy = Path(__file__).resolve().parent.parent.parent.parent / "SGDJSCC"
    if _looks_like_sgdjscc_repo(legacy):
        return legacy

    discovered = _auto_discover_sgdjscc_root()
    if discovered is not None:
        return discovered

    # Nothing found — fall back to the legacy guess so the error message
    # callers get (missing models/, etc.) points at the expected location.
    return legacy


SGDJSCC_ROOT: Path = _resolve_sgdjscc_root()


def ensure_sgdjscc_on_path() -> Path:
    """Insert SGDJSCC_ROOT at the front of sys.path if not already present."""
    if str(SGDJSCC_ROOT) not in sys.path:
        sys.path.insert(0, str(SGDJSCC_ROOT))
    return SGDJSCC_ROOT

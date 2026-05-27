"""_sgdjscc.py – Shared helper for injecting SGDJSCC onto sys.path.

All sub-modules that import from SGDJSCC should call ensure_sgdjscc_on_path()
instead of repeating the four-level .parent chain inline.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _resolve_sgdjscc_root() -> Path:
    # This file lives at:  .../sgdjscc_lab/src/sgdjscc_lab/_sgdjscc.py
    # Climbing up:         sgdjscc_lab/ (pkg) → src/ → sgdjscc_lab/ (repo) → Semantic/
    return Path(__file__).resolve().parent.parent.parent.parent / "SGDJSCC"


SGDJSCC_ROOT: Path = _resolve_sgdjscc_root()


def ensure_sgdjscc_on_path() -> Path:
    """Insert SGDJSCC_ROOT at the front of sys.path if not already present."""
    if str(SGDJSCC_ROOT) not in sys.path:
        sys.path.insert(0, str(SGDJSCC_ROOT))
    return SGDJSCC_ROOT

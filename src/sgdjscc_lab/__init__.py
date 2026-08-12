"""sgdjscc_lab – SGDJSCC research fork.

Phase 1: AWGN single-image / folder inference (complete).
Phase 2: Modular structure – channels / guidance / models / pipelines (complete).
Phase 3: Full metric evaluation (planned).
"""

from .paths import configure_external_cache_env as _configure_external_cache_env

__version__ = "0.2.0"

# No-op unless SGDJSCC_CACHE_ROOT is set; see paths.py.
_configure_external_cache_env()

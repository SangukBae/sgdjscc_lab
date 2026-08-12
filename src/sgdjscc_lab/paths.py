"""paths.py – Workspace root resolution (code/data separation).

Large working areas (datasets, checkpoints, run outputs, caches) do not have
to live inside the git checkout. This module resolves each root from an
environment variable first, falling back to the legacy in-repo location
(``<lab_repo_root>/data``, ``.../checkpoints``, ``.../outputs``, ...) so
existing configs and commands keep working unchanged when no environment
variable is set.

Env vars
--------
``SGDJSCC_DATA_ROOT``    – dataset root (legacy: ``<lab_repo_root>/data``)
``SGDJSCC_MODEL_ROOT``   – checkpoint workspace root, holding ``checkpoints/``,
                           ``checkpoints_custom/`` and ``remote_weights/``
                           subfolders (legacy: ``<lab_repo_root>/`` itself)
``SGDJSCC_RUN_ROOT``     – run/output root (legacy: ``<lab_repo_root>/outputs``)
``SGDJSCC_CACHE_ROOT``   – scratch/cache root (legacy: ``<lab_repo_root>/.cache``)

Note: ``SGDJSCC_ROOT`` (the *external* original SGDJSCC repo used for model
code) is a separate concept handled by :mod:`sgdjscc_lab._sgdjscc`.

Usage from configs
-------------------
An OmegaConf resolver named ``sgdjscc`` is available once
:func:`register_omegaconf_resolver` has run (``config.py`` calls it at import
time)::

    model_root: "${sgdjscc:model}/"
    input_path: "${sgdjscc:data}/imagenet/train"
"""

from __future__ import annotations

import os
from pathlib import Path

_MARKER_SUBPATH = Path("src") / "sgdjscc_lab"


def _lab_repo_root() -> Path:
    """Auto-discover the sgdjscc_lab checkout root (contains pyproject.toml +
    src/sgdjscc_lab/), independent of how deep this file's own path is nested
    (e.g. still correct inside a git worktree under .claude/worktrees/...).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / _MARKER_SUBPATH).is_dir():
            return parent
    # Fallback: this file lives at <root>/src/sgdjscc_lab/paths.py
    return here.parent.parent.parent


def _env_path(var: str) -> Path | None:
    val = os.environ.get(var)
    if not val:
        return None
    return Path(val).expanduser().resolve()


def lab_repo_root() -> Path:
    """The sgdjscc_lab checkout root (not configurable; auto-discovered)."""
    return _lab_repo_root()


def data_root() -> Path:
    return _env_path("SGDJSCC_DATA_ROOT") or (_lab_repo_root() / "data")


def _model_workspace_root() -> Path | None:
    return _env_path("SGDJSCC_MODEL_ROOT")


def model_root() -> Path:
    ws = _model_workspace_root()
    return (ws / "checkpoints") if ws is not None else (_lab_repo_root() / "checkpoints")


def checkpoints_custom_root() -> Path:
    ws = _model_workspace_root()
    return (ws / "checkpoints_custom") if ws is not None else (_lab_repo_root() / "checkpoints_custom")


def remote_weights_root() -> Path:
    ws = _model_workspace_root()
    return (ws / "remote_weights") if ws is not None else (_lab_repo_root() / "remote_weights")


def run_root() -> Path:
    return _env_path("SGDJSCC_RUN_ROOT") or (_lab_repo_root() / "outputs")


def cache_root() -> Path:
    return _env_path("SGDJSCC_CACHE_ROOT") or (_lab_repo_root() / ".cache")


_RESOLVER_MAP = {
    "root": lab_repo_root,
    "data": data_root,
    "model": model_root,
    "checkpoints_custom": checkpoints_custom_root,
    "remote_weights": remote_weights_root,
    "run": run_root,
    "cache": cache_root,
}


def resolve(kind: str) -> str:
    try:
        fn = _RESOLVER_MAP[kind]
    except KeyError:
        raise KeyError(
            f"unknown sgdjscc path kind {kind!r}; expected one of {sorted(_RESOLVER_MAP)}"
        ) from None
    return str(fn())


def register_omegaconf_resolver() -> None:
    """Register the ``${sgdjscc:<kind>}`` OmegaConf resolver (idempotent)."""
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("sgdjscc"):
        OmegaConf.register_new_resolver("sgdjscc", resolve)

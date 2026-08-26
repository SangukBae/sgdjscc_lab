"""utils/run_manifest.py – Reproducibility manifest for a saved run.

Builds a JSON-serializable record of everything needed to judge whether a
result in `results/` can be trusted or reproduced: git commit/dirty state,
the exact command invoked (and whether it was actually captured or only
reconstructed after the fact), the resolved config (or, failing that, the
unmerged config source it was built from), seed, dataset/checkpoint hashes,
the Python/CUDA/GPU environment, evaluator versions, the original (untracked)
artifact location plus a copied-vs-original hash comparison per artifact,
which numbers are exact vs. proxy/estimated, and NaN/failure counts.

Never guesses. Any field that cannot be verified is recorded as the literal
string ``"unknown"`` rather than inferred or left blank, so a manifest always
distinguishes "checked and empty" from "not checked". `None` is never used as
a stand-in for "unknown" or "not applicable" — see `seed` below — because a
bare `null` in the JSON output is ambiguous about which of those it means.

Kept import-light (no hard `torch` dependency) so it stays usable from
report-only tooling that must not pull in torch — see CLAUDE.md's note on
`summarize_etri_video_eval.py` / `generate_etri_final_report.py`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

UNKNOWN = "unknown"
"""The value could not be verified. Never inferred/guessed — always literal."""

NOT_SET = "not_set"
"""The value was positively confirmed to not exist/not apply (e.g. a script
with no `--seed` argument). Distinct from `UNKNOWN`: this means "checked, and
there is nothing there", not "not checked"."""

SCHEMA_VERSION = 2

VALID_COMMAND_SOURCES = {"captured", "reconstructed", UNKNOWN}
"""How `command` was obtained: `"captured"` — recorded live from sys.argv (or
equivalent) at run time; `"reconstructed"` — pieced together afterward from
docs/logs and may not match the exact flags/order actually used;
`"unknown"` — provenance of the command text itself is not known."""


def _run_git(args: list, cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")


def _git_dir(root: Path) -> Optional[Path]:
    """Resolve a normal or linked-worktree ``.git`` directory without git."""
    marker = root / ".git"
    if marker.is_dir():
        return marker
    if marker.is_file():
        try:
            line = marker.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if line.startswith("gitdir:"):
            path = Path(line.split(":", 1)[1].strip())
            return path if path.is_absolute() else (root / path).resolve()
    return None


def _read_git_metadata(root: Path) -> Dict[str, Any]:
    """Read HEAD/refs directly when the container has no ``git`` binary.

    This deliberately leaves ``dirty`` unknown: determining worktree/index
    differences correctly requires git.  The commit and branch, however, are
    exact checkout metadata and are sufficient to reject a resume after a
    code-version change.
    """
    git_dir = _git_dir(root)
    if git_dir is None:
        return {"commit": UNKNOWN, "dirty": UNKNOWN, "branch": UNKNOWN}
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return {"commit": UNKNOWN, "dirty": UNKNOWN, "branch": UNKNOWN}

    branch: Any = UNKNOWN
    commit: Any = UNKNOWN
    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        branch = ref.removeprefix("refs/heads/") if ref.startswith("refs/heads/") else ref
        ref_path = git_dir / ref
        try:
            value = ref_path.read_text(encoding="utf-8").strip()
        except OSError:
            value = ""
            try:
                for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
                    if not line.startswith(("#", "^")):
                        candidate, _, candidate_ref = line.partition(" ")
                        if candidate_ref == ref:
                            value = candidate
                            break
            except OSError:
                pass
        if _COMMIT_RE.fullmatch(value):
            commit = value.lower()
    elif _COMMIT_RE.fullmatch(head):
        commit = head.lower()
        branch = "HEAD"
    return {"commit": commit, "dirty": UNKNOWN, "branch": branch}


def _git_env_override() -> Optional[Dict[str, Any]]:
    """Return explicitly injected host provenance, if provided and valid."""
    commit = os.environ.get("SGDJSCC_GIT_COMMIT")
    if commit is None:
        return None
    if not _COMMIT_RE.fullmatch(commit.strip()):
        raise ValueError("SGDJSCC_GIT_COMMIT must be a 40-64 character hexadecimal commit")
    dirty_text = os.environ.get("SGDJSCC_GIT_DIRTY")
    if dirty_text is None:
        dirty: Any = UNKNOWN
    elif dirty_text.strip().lower() in {"1", "true", "yes"}:
        dirty = True
    elif dirty_text.strip().lower() in {"0", "false", "no"}:
        dirty = False
    else:
        raise ValueError("SGDJSCC_GIT_DIRTY must be true/false (or 1/0) when set")
    return {
        "commit": commit.strip().lower(),
        "dirty": dirty,
        "branch": os.environ.get("SGDJSCC_GIT_BRANCH", UNKNOWN),
    }


def get_git_state(repo_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Return ``{"commit", "dirty", "branch"}`` for the repo at `repo_root`.

    Any field that can't be determined (not a git repo, `git` missing, repo
    root doesn't exist) is `"unknown"` — never guessed from context.
    """
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    if not root.exists():
        return {"commit": UNKNOWN, "dirty": UNKNOWN, "branch": UNKNOWN}

    override = _git_env_override()
    if override is not None:
        return override

    commit = _run_git(["rev-parse", "HEAD"], root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    # Untracked reports/assets cannot change the committed Python checkout and
    # are intentionally excluded. ``dirty`` means tracked/indexed code or
    # configuration differs from HEAD.
    status = _run_git(["status", "--porcelain", "--untracked-files=no"], root)

    state = {
        "commit": commit if commit else UNKNOWN,
        "dirty": (len(status) > 0) if status is not None else UNKNOWN,
        "branch": branch if branch else UNKNOWN,
    }
    if state["commit"] == UNKNOWN:
        fallback = _read_git_metadata(root)
        state["commit"] = fallback["commit"]
        state["branch"] = fallback["branch"]
    return state


def get_python_env() -> Dict[str, Any]:
    """Python interpreter / OS platform info. Always determinable."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _try_import_torch() -> Any:
    """Import and return the `torch` module, or `None` on *any* failure.

    A broken CUDA/driver install can fail import with `OSError` (missing
    shared library), `RuntimeError`, or other exotic exceptions — not just
    `ImportError` for a genuinely absent package — so this deliberately
    catches broadly rather than just `ImportError`. Factored out of
    `get_cuda_env` so tests can substitute this one seam instead of having
    to fake a real broken torch install.
    """
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:
        return None


def get_cuda_env() -> Dict[str, Any]:
    """Torch/CUDA/GPU info, `"unknown"` per-field if torch/CUDA unavailable.

    Never raises: a missing torch package, a broken/incompatible torch
    install, or a broken CUDA runtime all degrade to `"unknown"` fields
    rather than propagating an exception to the caller.
    """
    info: Dict[str, Any] = {
        "torch_version": UNKNOWN,
        "cuda_available": UNKNOWN,
        "cuda_version": UNKNOWN,
        "gpu_name": UNKNOWN,
    }
    torch = _try_import_torch()
    if torch is None:
        return info

    info["torch_version"] = getattr(torch, "__version__", UNKNOWN)
    try:
        available = bool(torch.cuda.is_available())
    except Exception:
        return info
    info["cuda_available"] = available
    try:
        info["cuda_version"] = getattr(torch.version, "cuda", None) or UNKNOWN
    except Exception:
        info["cuda_version"] = UNKNOWN
    if available:
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            info["gpu_name"] = UNKNOWN
    return info


def sha256_file(path: Optional[Union[str, Path]]) -> str:
    """SHA-256 hex digest of a file, or `"unknown"` if missing/unreadable."""
    if path is None:
        return UNKNOWN
    p = Path(path)
    if not p.is_file():
        return UNKNOWN
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return UNKNOWN
    return h.hexdigest()


def _require_file_hash(path: Union[str, Path], field_name: str) -> str:
    """Return a file hash or reject a path that cannot support a verified claim."""
    digest = sha256_file(path)
    if digest == UNKNOWN:
        raise ValueError(f"{field_name} must point to an existing readable file: {path}")
    return digest


def hash_artifact_pair(
    copied_path: Union[str, Path],
    original_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Hash a tracked `results/` copy and (if given) its untracked `outputs/`
    original, and report whether they match.

    `matches` is the boolean comparison only when both hashes were
    computable; if either side is unreadable/missing, `matches` is
    `"unknown"` rather than a guessed `True`/`False`.
    """
    copied = Path(copied_path)
    copied_sha = sha256_file(copied)
    if original_path is None:
        return {
            "copied_path": str(copied),
            "copied_sha256": copied_sha,
            "original_path": UNKNOWN,
            "original_sha256": UNKNOWN,
            "matches": UNKNOWN,
        }
    original = Path(original_path)
    original_sha = sha256_file(original)
    if copied_sha == UNKNOWN or original_sha == UNKNOWN:
        matches: Any = UNKNOWN
    else:
        matches = copied_sha == original_sha
    return {
        "copied_path": str(copied),
        "copied_sha256": copied_sha,
        "original_path": str(original),
        "original_sha256": original_sha,
        "matches": matches,
    }


def build_run_manifest(
    *,
    run_id: str,
    command: Optional[str] = None,
    command_argv: Optional[Sequence[str]] = None,
    command_source: str = UNKNOWN,
    seed: Any = UNKNOWN,
    resolved_config: Optional[Dict[str, Any]] = None,
    resolved_config_path: Optional[Union[str, Path]] = None,
    config_source_path: Optional[Union[str, Path]] = None,
    config_source_note: Optional[str] = None,
    dataset_ref: Any = UNKNOWN,
    dataset_hash: Any = UNKNOWN,
    checkpoints: Optional[Dict[str, Union[str, Path]]] = None,
    evaluator_versions: Optional[Dict[str, Any]] = None,
    original_artifact_paths: Optional[Dict[str, str]] = None,
    artifacts: Optional[Dict[str, Dict[str, Union[str, Path, None]]]] = None,
    exact_fields: Optional[list] = None,
    proxy_fields: Optional[list] = None,
    nan_or_failure_counts: Optional[Dict[str, Any]] = None,
    repo_root: Optional[Union[str, Path]] = None,
    include_environment: bool = True,
    include_git: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a full reproducibility manifest dict for one saved run.

    Every argument left as `None`/unset is recorded as `"unknown"` (or an
    empty collection where a collection is expected) — callers must pass
    real values to get anything else; nothing here is inferred.

    `include_git=False` skips probing the current repo entirely and records
    `git` as `"unknown"` — use this for a historical/backfilled run whose
    actual commit was never captured at run time. Leaving it `True` (default)
    for such a run would silently record *this session's* current HEAD/dirty
    state instead of the run's, which is exactly the guessing this module
    exists to prevent.

    `seed` must never be Python `None` — pass the literal seed value, or one
    of the `UNKNOWN` / `NOT_SET` sentinels, so the manifest always
    distinguishes "not verified" from "confirmed there is no seed". A bare
    `null` collapses that distinction, so passing `None` raises `ValueError`.

    `command` or `command_argv`, if given, must be paired with
    `command_source` (one of `"captured"`, `"reconstructed"`, `"unknown"`)
    so a manifest never presents a reconstructed-from-docs command as if it
    were captured live. Prefer `command_argv=sys.argv` for live capture: the
    argv array is lossless and its display text is rendered with
    `shlex.join()`, including arguments containing whitespace.

    Exactly one of `resolved_config` / `resolved_config_path` /
    `config_source_path` should normally be given:
    - `resolved_config` / `resolved_config_path`: use ONLY when the final,
      fully-merged config actually used for this run (after `_defaults_`
      composition and any CLI overrides) can be reconstructed with
      confidence. Recorded with `status: "resolved"`.
    - `config_source_path`: use when only the pre-merge root config file is
      available and the true merged/overridden config cannot be verified.
      Recorded with `status: "config_source_only"` and `resolved` stays
      `"unknown"` — this must not be presented as the resolved config.
    - If none are given: `status: "unknown"`.

    `artifacts`: optional `{name: {"copied_path": ..., "original_path": ...}}`
    — each entry is hashed via `hash_artifact_pair` to record the copied
    (tracked) file's path/sha256, the original (untracked) file's
    path/sha256, and whether they match.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if seed is None:
        raise ValueError(
            "seed must not be None — pass run_manifest.UNKNOWN (not verified) "
            "or run_manifest.NOT_SET (confirmed no seed applies), or an "
            "explicit seed value"
        )
    if command_source not in VALID_COMMAND_SOURCES:
        raise ValueError(f"command_source must be one of {sorted(VALID_COMMAND_SOURCES)}, got {command_source!r}")
    if command is not None and command_argv is not None:
        raise ValueError("pass at most one of command / command_argv")
    if command is not None and not command.strip():
        raise ValueError("command must be a non-empty string when provided")
    if command_argv is not None and not command_argv:
        raise ValueError("command_argv must contain at least one argument")
    if command is None and command_argv is None and command_source != UNKNOWN:
        raise ValueError("command_source requires command or command_argv")

    git_state = get_git_state(repo_root) if include_git else {
        "commit": UNKNOWN, "dirty": UNKNOWN, "branch": UNKNOWN,
    }

    if checkpoints is None:
        checkpoint_record: Dict[str, Any] = {"status": UNKNOWN, "items": {}}
    elif not checkpoints:
        checkpoint_record = {"status": NOT_SET, "items": {}}
    else:
        checkpoint_items = {
            name: {
                "path": str(path),
                "sha256": _require_file_hash(path, f"checkpoints[{name!r}]")
            }
            for name, path in checkpoints.items()
        }
        checkpoint_record = {"status": "recorded", "items": checkpoint_items}

    given = [x is not None for x in (resolved_config, resolved_config_path, config_source_path)]
    if sum(given) > 1:
        raise ValueError(
            "pass at most one of resolved_config / resolved_config_path / config_source_path"
        )

    resolved_config_record: Dict[str, Any]
    if resolved_config is not None:
        resolved_config_record = {
            "status": "resolved",
            "resolved": {"inline": resolved_config},
            "config_source": None,
        }
    elif resolved_config_path is not None:
        resolved_config_sha = _require_file_hash(resolved_config_path, "resolved_config_path")
        resolved_config_record = {
            "status": "resolved",
            "resolved": {
                "path": str(resolved_config_path),
                "sha256": resolved_config_sha,
            },
            "config_source": None,
        }
    elif config_source_path is not None:
        config_source_sha = _require_file_hash(config_source_path, "config_source_path")
        resolved_config_record = {
            "status": "config_source_only",
            "resolved": UNKNOWN,
            "config_source": {
                "path": str(config_source_path),
                "sha256": config_source_sha,
                "note": config_source_note or (
                    "unmerged root config only — _defaults_ fragment composition "
                    "and any CLI overrides applied at run time are not captured; "
                    "do not treat this as the resolved config actually used"
                ),
            },
        }
    else:
        resolved_config_record = {"status": "unknown", "resolved": UNKNOWN, "config_source": None}

    if command_argv is not None:
        argv = [str(arg) for arg in command_argv]
        command_record = {
            "text": shlex.join(argv),
            "argv": argv,
            "source": command_source,
        }
    elif command is not None:
        command_record = {"text": command, "argv": UNKNOWN, "source": command_source}
    else:
        command_record = {"text": UNKNOWN, "argv": UNKNOWN, "source": UNKNOWN}

    artifact_records = {
        name: hash_artifact_pair(spec.get("copied_path"), spec.get("original_path"))
        for name, spec in (artifacts or {}).items()
    }

    if include_environment:
        environment = {**get_python_env(), **get_cuda_env()}
    else:
        environment = {
            "python_version": UNKNOWN,
            "platform": UNKNOWN,
            "torch_version": UNKNOWN,
            "cuda_available": UNKNOWN,
            "cuda_version": UNKNOWN,
            "gpu_name": UNKNOWN,
        }

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": git_state,
        "command": command_record,
        "seed": seed,
        "resolved_config": resolved_config_record,
        "dataset": {"ref": dataset_ref, "hash": dataset_hash},
        "checkpoints": checkpoint_record,
        "environment": environment,
        "evaluator_versions": evaluator_versions or {},
        "original_artifact_paths": original_artifact_paths or {},
        "artifacts": artifact_records,
        "accounting": {
            "exact_fields": exact_fields or [],
            "proxy_fields": proxy_fields or [],
        },
        "nan_or_failure_counts": nan_or_failure_counts or {},
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_run_manifest(path: Union[str, Path], manifest: Dict[str, Any]) -> Path:
    """Write `manifest` as pretty-printed JSON to `path`, creating parents."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")
    return out

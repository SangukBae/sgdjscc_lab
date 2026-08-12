"""cli.py – console_scripts entry points (``sgdjscc-infer``, ``sgdjscc-train``,
``sgdjscc-evaluate``, ``sgdjscc-evaluate-video``).

The argparse-based ``main()`` implementations still live in ``scripts/*.py``
(unchanged, so ``python scripts/infer_images.py ...`` keeps working exactly
as before). These wrappers load that same module by file path — the pattern
already used by ``summarize_etri_video_eval.py`` to reach into ``pipelines/``
without a package-level import — and call its ``main()``, so both entry
points execute identical code.

This only works when ``scripts/`` sits next to the installed ``sgdjscc_lab``
checkout (true for an editable install, ``pip install -e .``, which is how
this project is always run — see CLAUDE.md). It is not meant to survive a
relocated, non-editable wheel install.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .paths import lab_repo_root


def _load_script_main(script_name: str):
    script_path = lab_repo_root() / "scripts" / script_name
    if not script_path.exists():
        raise FileNotFoundError(
            f"{script_path} not found — sgdjscc-* console scripts require an "
            "editable install (pip install -e .) run from within the "
            "sgdjscc_lab checkout, so scripts/ is discoverable next to src/."
        )
    module_name = f"sgdjscc_lab._cli_scripts.{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.main


def infer_main() -> None:
    _load_script_main("infer_images.py")()


def train_main() -> None:
    _load_script_main("train.py")()


def evaluate_main() -> None:
    _load_script_main("evaluate.py")()


def evaluate_video_main() -> None:
    _load_script_main("evaluate_video.py")()

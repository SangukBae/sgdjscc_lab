"""config.py – OmegaConf-based config loader with CLI override support."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str | Path) -> DictConfig:
    """Load YAML config from *config_path*.

    Paths inside the config are resolved relative to the **config file's
    directory**, not the current working directory.  This lets users run
    ``python scripts/infer_images.py --config configs/default.yaml`` from any
    working directory and still have ``input_path`` / ``output_dir`` /
    ``model_root`` resolve sensibly.
    """
    config_path = Path(config_path).resolve()
    cfg = OmegaConf.load(config_path)

    cfg_dir = config_path.parent

    # Resolve relative paths anchored to the config file's directory.
    for key in ("input_path", "output_dir", "model_root"):
        val = cfg.get(key, None)
        if val is not None:
            p = Path(val)
            if not p.is_absolute():
                cfg[key] = str(cfg_dir / p)

    return cfg


def merge_cli_overrides(
    cfg: DictConfig,
    input_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    snr_db: Optional[float] = None,
    device: Optional[str] = None,
) -> DictConfig:
    """Apply CLI flag overrides on top of a loaded config.

    Only non-None arguments override the corresponding config field.
    """
    overrides: dict = {}
    if input_path is not None:
        overrides["input_path"] = input_path
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if snr_db is not None:
        overrides["snr_db"] = snr_db
    if device is not None:
        overrides["device"] = device

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))

    return cfg

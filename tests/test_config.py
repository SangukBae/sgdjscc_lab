"""tests/test_config.py – Unit tests for config loading and CLI override merge.

No GPU, no checkpoints, no SGDJSCC imports required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make sure the package is importable even without editable install
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.config import load_config, merge_cli_overrides


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def minimal_yaml(tmp_path):
    """Write a minimal YAML config and return its path."""
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(
        "input_path: '../inputs/'\n"
        "output_dir: '../outputs/'\n"
        "model_root: '../checkpoints/'\n"
        "snr_db: 10\n"
        "device: 'cpu'\n"
        "use_semantic: true\n"
        "use_text: true\n"
        "use_controlnet: false\n"
        "use_jscc_feature: true\n"
        "use_gt_csi: false\n"
        "canny_cr: '0.2'\n"
        "diffusion_step: 50\n"
        "step_style: 'continuous'\n"
        "guidance_scale: 4.0\n"
        "controlnet_scale: 0.3\n"
        "cfg_method: 'pcs_1.0'\n"
        "mask_method: 'none'\n"
        "th: 0.25\n"
    )
    return cfg_file


# ─────────────────────────────────────────────────────────────────────────────
# load_config tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_returns_dictconfig(self, minimal_yaml):
        from omegaconf import DictConfig
        cfg = load_config(minimal_yaml)
        assert isinstance(cfg, DictConfig)

    def test_relative_input_path_resolved(self, minimal_yaml):
        """input_path '../inputs/' should resolve to an absolute path."""
        cfg = load_config(minimal_yaml)
        assert Path(cfg.input_path).is_absolute()

    def test_relative_output_dir_resolved(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        assert Path(cfg.output_dir).is_absolute()

    def test_relative_model_root_resolved(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        assert Path(cfg.model_root).is_absolute()

    def test_scalar_fields_preserved(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        assert cfg.snr_db == 10
        assert cfg.device == "cpu"
        assert cfg.diffusion_step == 50

    def test_path_resolved_relative_to_config_dir(self, tmp_path):
        """Paths should be resolved relative to the config file, not cwd."""
        cfg_file = tmp_path / "subdir" / "config.yaml"
        cfg_file.parent.mkdir()
        cfg_file.write_text("input_path: '../data/'\noutput_dir: '../out/'\nmodel_root: '../ckpt/'\n")
        cfg = load_config(cfg_file)
        # '../data/' relative to subdir/ resolves to tmp_path/data/
        # resolve() collapses the '..' segments for comparison
        assert Path(cfg.input_path).resolve() == (tmp_path / "data").resolve()


# ─────────────────────────────────────────────────────────────────────────────
# merge_cli_overrides tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeCliOverrides:
    def test_no_overrides_leaves_config_unchanged(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        original_snr = cfg.snr_db
        cfg2 = merge_cli_overrides(cfg)
        assert cfg2.snr_db == original_snr

    def test_snr_override(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        cfg = merge_cli_overrides(cfg, snr_db=5.0)
        assert cfg.snr_db == 5.0

    def test_device_override(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        cfg = merge_cli_overrides(cfg, device="cuda:1")
        assert cfg.device == "cuda:1"

    def test_input_path_override(self, minimal_yaml, tmp_path):
        cfg = load_config(minimal_yaml)
        new_path = str(tmp_path / "images")
        cfg = merge_cli_overrides(cfg, input_path=new_path)
        assert cfg.input_path == new_path

    def test_output_dir_override(self, minimal_yaml, tmp_path):
        cfg = load_config(minimal_yaml)
        new_dir = str(tmp_path / "out")
        cfg = merge_cli_overrides(cfg, output_dir=new_dir)
        assert cfg.output_dir == new_dir

    def test_none_overrides_are_ignored(self, minimal_yaml):
        cfg = load_config(minimal_yaml)
        snr_before = cfg.snr_db
        cfg = merge_cli_overrides(cfg, snr_db=None, device=None)
        assert cfg.snr_db == snr_before

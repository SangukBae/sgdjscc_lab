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


# ─────────────────────────────────────────────────────────────────────────────
# _defaults_ fragment composition tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFragmentComposition:
    def _make_fragment(self, directory: Path, name: str, content: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        f = directory / f"{name}.yaml"
        f.write_text(content)
        return f

    def test_fragments_merged_into_cfg(self, tmp_path):
        """_defaults_ entries are loaded and merged; root keys override."""
        self._make_fragment(tmp_path / "channel", "awgn", "channel: awgn\nsnr_db: 10\n")
        self._make_fragment(tmp_path / "model", "base", "use_semantic: true\ndiffusion_step: 50\n")

        root = tmp_path / "composed.yaml"
        root.write_text("_defaults_:\n  - channel/awgn\n  - model/base\n")

        cfg = load_config(root)
        assert cfg.channel == "awgn"
        assert cfg.snr_db == 10
        assert cfg.use_semantic is True
        assert cfg.diffusion_step == 50

    def test_root_key_overrides_fragment(self, tmp_path):
        """An explicit key in the root config wins over a fragment value."""
        self._make_fragment(tmp_path / "channel", "awgn", "channel: awgn\nsnr_db: 10\n")

        root = tmp_path / "composed.yaml"
        root.write_text("_defaults_:\n  - channel/awgn\nsnr_db: 5\n")

        cfg = load_config(root)
        assert cfg.snr_db == 5

    def test_later_fragment_overrides_earlier(self, tmp_path):
        """Later entries in _defaults_ override earlier ones."""
        self._make_fragment(tmp_path / "a", "first", "value: 1\n")
        self._make_fragment(tmp_path / "a", "second", "value: 2\n")

        root = tmp_path / "composed.yaml"
        root.write_text("_defaults_:\n  - a/first\n  - a/second\n")

        cfg = load_config(root)
        assert cfg.value == 2

    def test_path_fields_resolved_after_composition(self, tmp_path):
        """input_path from a fragment is resolved relative to the root config dir.

        Root config is at tmp_path/composed.yaml → cfg_dir = tmp_path.
        Fragment path 'data/' resolves to tmp_path/data (not the fragment's dir).
        """
        self._make_fragment(tmp_path / "infer", "io", "input_path: 'data/'\noutput_dir: 'out/'\n")

        root = tmp_path / "composed.yaml"
        root.write_text("_defaults_:\n  - infer/io\nmodel_root: 'ckpt/'\n")

        cfg = load_config(root)
        assert Path(cfg.input_path).is_absolute()
        assert Path(cfg.output_dir).is_absolute()
        assert Path(cfg.model_root).is_absolute()
        assert Path(cfg.input_path).resolve() == (tmp_path / "data").resolve()

    def test_no_defaults_key_unchanged(self, minimal_yaml):
        """A config without _defaults_ is loaded unchanged."""
        cfg = load_config(minimal_yaml)
        assert "_defaults_" not in cfg
        assert cfg.snr_db == 10

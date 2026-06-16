"""tests/test_export_checkpoint.py – Unit tests for scripts/export_checkpoint.py.

No GPU, no real checkpoints: synthetic training-checkpoint dicts exercise the
per-stage format conversion (raw state_dict for ``jscc``; ``model_ema`` for the
diffusion stages), the legacy-layout fallbacks, and the early-failure paths.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# Load scripts/export_checkpoint.py as a module (scripts/ is not a package).
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "export_checkpoint.py"
_spec = importlib.util.spec_from_file_location("export_checkpoint", _SCRIPT)
export_checkpoint = importlib.util.module_from_spec(_spec)
sys.modules["export_checkpoint"] = export_checkpoint
_spec.loader.exec_module(export_checkpoint)


def _sd(prefix: str) -> dict:
    """A tiny but valid state_dict (all-tensor mapping)."""
    return {f"{prefix}.w": torch.zeros(2, 2), f"{prefix}.b": torch.ones(2)}


def _runner_ckpt(module_name: str) -> dict:
    """Synthetic checkpoint in the current ``runner_state.modules`` layout."""
    return {
        "epoch": 3,
        "global_step": 100,
        "stage": "synthetic",
        "runner_state": {
            "modules": {module_name: _sd(module_name)},
            "optimizers": {},
            "scalers": {},
            "accum": 0,
        },
    }


# ── per-stage export format ──────────────────────────────────────────────────

def test_export_jscc_raw_state_dict():
    ckpt = _runner_ckpt("jscc_model")
    payload, source = export_checkpoint.export_checkpoint("jscc", ckpt)
    assert source == "runner_state.modules.jscc_model"
    # raw: no model_ema wrapper, identical tensors.
    assert "model_ema" not in payload
    assert set(payload) == set(_sd("jscc_model"))
    assert torch.equal(payload["jscc_model.b"], torch.ones(2))


def test_export_text_dm_model_ema():
    payload, source = export_checkpoint.export_checkpoint("text_dm", _runner_ckpt("diffusion"))
    assert source == "runner_state.modules.diffusion"
    assert set(payload) == {"model_ema"}
    assert set(payload["model_ema"]) == set(_sd("diffusion"))


def test_export_controlnet_model_ema():
    payload, source = export_checkpoint.export_checkpoint("controlnet", _runner_ckpt("diffusion"))
    assert source == "runner_state.modules.diffusion"
    assert set(payload) == {"model_ema"}


# ── legacy / bare fallbacks ──────────────────────────────────────────────────

def test_export_legacy_model_state():
    ckpt = {"model_state": {"jscc_model": _sd("jscc_model")}}
    payload, source = export_checkpoint.export_checkpoint("jscc", ckpt)
    assert source == "model_state.jscc_model"
    assert "model_ema" not in payload


def test_export_bare_state_dict_jscc_allowed():
    # jscc export is itself a raw state_dict → a bare input is a safe 1:1 source.
    payload, source = export_checkpoint.export_checkpoint("jscc", _sd("jscc_model"))
    assert source == "<bare state_dict>"
    assert set(payload) == set(_sd("jscc_model"))


@pytest.mark.parametrize("stage", ["text_dm", "controlnet"])
def test_export_bare_state_dict_diffusion_rejected(stage):
    # A bare state_dict (e.g. a raw JSCC_model.pth) has no module identity, so the
    # diffusion stages must refuse it rather than silently wrap it as model_ema.
    with pytest.raises(KeyError):
        export_checkpoint.export_checkpoint(stage, _sd("jscc_model"))


# ── early-failure paths ──────────────────────────────────────────────────────

def test_unknown_stage_raises():
    with pytest.raises(ValueError):
        export_checkpoint.export_checkpoint("nope", _runner_ckpt("diffusion"))


def test_stage_module_mismatch_raises():
    # jscc stage against a diffusion-only checkpoint → jscc_model missing.
    with pytest.raises(KeyError):
        export_checkpoint.export_checkpoint("jscc", _runner_ckpt("diffusion"))


def test_text_dm_against_jscc_ckpt_raises():
    with pytest.raises(KeyError):
        export_checkpoint.export_checkpoint("text_dm", _runner_ckpt("jscc_model"))


def test_empty_module_state_raises():
    ckpt = {"runner_state": {"modules": {"jscc_model": {}}}}
    with pytest.raises(KeyError):
        export_checkpoint.export_checkpoint("jscc", ckpt)


# ── end-to-end file round-trip (covers main() write + --force semantics) ─────

def test_file_roundtrip_and_force(tmp_path, monkeypatch):
    in_path = tmp_path / "best.pth"
    out_path = tmp_path / "diffusion_backbone.pth"
    torch.save(_runner_ckpt("diffusion"), in_path)

    argv = ["export_checkpoint.py", "--stage", "text_dm",
            "--input", str(in_path), "--output", str(out_path)]
    monkeypatch.setattr(sys, "argv", argv)
    export_checkpoint.main()

    loaded = torch.load(out_path, map_location="cpu")
    assert "model_ema" in loaded

    # Second run without --force must refuse to overwrite.
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileExistsError):
        export_checkpoint.main()

    # --force overwrites.
    monkeypatch.setattr(sys, "argv", argv + ["--force"])
    export_checkpoint.main()  # no raise


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    in_path = tmp_path / "best.pth"
    out_path = tmp_path / "out.pth"
    torch.save(_runner_ckpt("jscc_model"), in_path)
    argv = ["export_checkpoint.py", "--stage", "jscc",
            "--input", str(in_path), "--output", str(out_path), "--dry-run"]
    monkeypatch.setattr(sys, "argv", argv)
    export_checkpoint.main()
    assert not out_path.exists()


def test_missing_input_raises(tmp_path, monkeypatch):
    argv = ["export_checkpoint.py", "--stage", "jscc",
            "--input", str(tmp_path / "nope.pth"), "--output", str(tmp_path / "o.pth")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileNotFoundError):
        export_checkpoint.main()

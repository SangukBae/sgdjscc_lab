"""tests/test_train_ops.py – operational-stability & memory-toggle features.

Covers the additions in training/interrupt.py, training/perf.py,
training/val_images.py and the resume/interrupt wiring in
pipelines/train_pipeline.py.  No GPU, no checkpoints, no SGDJSCC imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.pipelines.train_pipeline import (
    resolve_resume_path, save_interrupt_checkpoint, restore_runner_state,
    INTERRUPT_CKPT_NAME,
)
from sgdjscc_lab.training.perf import build_optimizer, apply_memory_optimizations
from sgdjscc_lab.training.val_images import ValImageLogger


# ── auto-resume path resolution ──────────────────────────────────────────────

def test_resolve_resume_none_and_null(tmp_path):
    assert resolve_resume_path(None, tmp_path) == (None, False)
    assert resolve_resume_path("null", tmp_path) == (None, False)
    assert resolve_resume_path("", tmp_path) == (None, False)


def test_resolve_resume_explicit_path_preserved(tmp_path):
    path, is_auto = resolve_resume_path("/abs/ckpt.pth", tmp_path)
    assert path == Path("/abs/ckpt.pth") and is_auto is False


def test_resolve_resume_latest_missing_is_fresh(tmp_path):
    # auto request with nothing to resume → fresh run (safe default), not an error.
    path, is_auto = resolve_resume_path("latest", tmp_path)
    assert path is None and is_auto is True


def test_resolve_resume_latest_and_interrupt_discovery(tmp_path):
    (tmp_path / INTERRUPT_CKPT_NAME).write_bytes(b"x")
    # interrupt checkpoint is discovered when latest.pth is absent…
    assert resolve_resume_path("auto", tmp_path)[0] == tmp_path / INTERRUPT_CKPT_NAME
    # …but latest.pth takes precedence when both exist.
    (tmp_path / "latest.pth").write_bytes(b"y")
    assert resolve_resume_path("latest", tmp_path)[0] == tmp_path / "latest.pth"


# ── interrupt checkpoint round-trips through the normal resume path ───────────

class _StateRunner:
    def get_train_state(self):
        return {"modules": {}, "optimizers": {}, "scalers": {}, "accum": 0}
    def load_train_state(self, s):
        self.loaded = s


def test_interrupt_checkpoint_writes_both_and_restores(tmp_path):
    state = {"epoch": 2, "global_step": 7, "stage": "jscc", "best_metric": 0.3,
             "runner_state": {"modules": {}, "optimizers": {}, "scalers": {}, "accum": 0}}
    save_interrupt_checkpoint(state, tmp_path)
    assert (tmp_path / INTERRUPT_CKPT_NAME).exists()
    assert (tmp_path / "latest.pth").exists()      # refreshed for --resume latest
    # Restorable by the SAME code path as a normal checkpoint.
    restored = restore_runner_state(tmp_path / "latest.pth", _StateRunner())
    assert restored["global_step"] == 7 and restored["epoch"] == 2


# ── 8-bit optimizer graceful fallback ────────────────────────────────────────

def test_build_optimizer_default_is_adamw():
    lin = nn.Linear(4, 4)
    pg = [{"params": list(lin.parameters()), "name": "x"}]
    opt = build_optimizer(pg, OmegaConf.create({"train": {}}), lr=1e-4, weight_decay=1e-5)
    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_8bit_falls_back_when_missing():
    lin = nn.Linear(4, 4)
    pg = [{"params": list(lin.parameters()), "name": "x"}]
    cfg = OmegaConf.create({"train": {"use_8bit_adam": True}})
    opt = build_optimizer(pg, cfg, lr=1e-4, weight_decay=1e-5)
    # bitsandbytes is not a hard dep → must not raise; falls back to AdamW.
    assert opt is not None
    assert opt.__class__.__name__ in {"AdamW", "AdamW8bit"}


def test_build_optimizer_empty_groups_is_none():
    assert build_optimizer([], OmegaConf.create({"train": {}}), lr=1e-4, weight_decay=1e-5) is None


# ── memory toggles never crash / never silently ignore ───────────────────────

def test_apply_memory_optimizations_unsupported_module_logs():
    import logging
    from sgdjscc_lab.training import perf
    records = []
    handler = logging.Handler()
    handler.emit = lambda rec: records.append(rec.getMessage())
    perf.logger.addHandler(handler)
    old_level = perf.logger.level
    perf.logger.setLevel(logging.INFO)
    try:
        m = nn.Sequential(nn.Linear(4, 4))
        cfg = OmegaConf.create({"train": {"gradient_checkpointing": True, "use_xformers": True}})
        apply_memory_optimizations({"diffusion": m}, cfg, stage="text_dm")
    finally:
        perf.logger.removeHandler(handler)
        perf.logger.setLevel(old_level)
    text = " ".join(records)
    assert "NOT applied" in text          # grad-ckpt unsupported → logged, not silent
    assert "xformers" in text


def test_apply_memory_optimizations_noop_when_all_off():
    m = nn.Sequential(nn.Linear(4, 4))
    # Should return immediately without touching the module.
    apply_memory_optimizations({"diffusion": m}, OmegaConf.create({"train": {}}), stage="jscc")


# ── validation image logger ──────────────────────────────────────────────────

class _FakeVAE:
    def decode(self, z):
        return (torch.rand(z.shape[0], 3, 32, 32) * 2 - 1,)


class _JSCCRunner:
    _training = True
    def set_mode(self, t): self._training = t
    def _reconstruct(self, x): return torch.rand_like(x)


def _val_cfg(**vi):
    base = {"enabled": True, "every_steps": 2, "num_samples": 2}
    base.update(vi)
    return OmegaConf.create({"train": {"val_images": base}})


def test_val_logger_disabled_by_default(tmp_path):
    vl = ValImageLogger(OmegaConf.create({"train": {}}), "controlnet",
                        torch.device("cpu"), _FakeVAE(), tmp_path)
    assert vl.enabled is False


def test_val_logger_cadence_and_write(tmp_path):
    vl = ValImageLogger(_val_cfg(), "jscc", torch.device("cpu"), _FakeVAE(), tmp_path)
    batch = {"image": torch.rand(4, 3, 32, 32)}
    vl.maybe_log_step(_JSCCRunner(), batch, global_step=1, epoch=1)   # 1 % 2 != 0 → skip
    assert not list((tmp_path / "val_images").glob("*.png"))
    vl.maybe_log_step(_JSCCRunner(), batch, global_step=2, epoch=1)   # write
    files = list((tmp_path / "val_images").glob("*.png"))
    assert [f.name for f in files] == ["step_0000002.png"]


class _DMRunner:
    _training = True
    def set_mode(self, t): self._training = t
    encode_latent_fn = staticmethod(lambda imgs: torch.rand(imgs.shape[0], 16, 4, 4))
    encode_text_fn = staticmethod(lambda caps: torch.rand(len(caps), 8))
    encode_edge_fn = staticmethod(lambda e: torch.rand(e.shape[0], 16, 4, 4))

    class _Sched:
        def add_noise(self, f0):
            return f0, torch.rand(f0.shape[0], 1), None, None
    scheduler = _Sched()

    def denoiser(self, ft, nl, labels, c=None, enable_mask=False):
        return torch.rand(ft.shape[0], 16, 4, 4)


def test_val_logger_controlnet_panel_with_edge(tmp_path):
    vl = ValImageLogger(_val_cfg(every_epochs=1, every_steps=0), "controlnet",
                        torch.device("cpu"), _FakeVAE(), tmp_path)
    batch = {"image": torch.rand(4, 3, 32, 32), "edge": torch.rand(4, 1, 32, 32),
             "caption": ["a", "b", "c", "d"]}
    vl.maybe_log_epoch(_DMRunner(), batch, epoch=1)
    assert (tmp_path / "val_images" / "epoch_0001.png").exists()


def test_val_logger_uses_val_batch_and_eval_mode(tmp_path):
    """Panel must render from the val_loader batch, in eval mode, restoring mode."""
    seen = {}

    class _ModeRunner:
        _training = True
        def set_mode(self, t): self._training = t
        def _reconstruct(self, x):
            seen["mode_during"] = self._training     # must be False (eval)
            seen["input_sum"] = float(x.sum())       # must come from the val batch
            return torch.rand_like(x)

    val_batch = {"image": torch.ones(2, 3, 32, 32)}      # nonzero → distinguishable
    train_batch = {"image": torch.zeros(2, 3, 32, 32)}   # zero
    vl = ValImageLogger(_val_cfg(), "jscc", torch.device("cpu"), _FakeVAE(),
                        tmp_path, val_loader=[val_batch])
    r = _ModeRunner(); r.set_mode(True)
    vl.maybe_log_step(r, train_batch, global_step=2, epoch=1)
    assert seen["mode_during"] is False              # ran in eval mode
    assert seen["input_sum"] > 0                      # used the val batch, not train
    assert r._training is True                        # prior (training) mode restored


def test_val_logger_falls_back_to_training_batch_without_val_loader(tmp_path):
    seen = {}

    class _R:
        _training = True
        def set_mode(self, t): self._training = t
        def _reconstruct(self, x):
            seen["input_sum"] = float(x.sum()); return torch.rand_like(x)

    train_batch = {"image": torch.ones(2, 3, 32, 32)}
    vl = ValImageLogger(_val_cfg(), "jscc", torch.device("cpu"), _FakeVAE(),
                        tmp_path, val_loader=None)
    vl.maybe_log_step(_R() if False else _R(), train_batch, global_step=2, epoch=1)
    assert seen["input_sum"] > 0                      # used the fallback training batch


def test_enable_xformers_not_claimed_on_non_diffusion_module():
    """xformers must NOT be reported active for arbitrary trainable modules."""
    from sgdjscc_lab.training.perf import _enable_xformers
    pytest.importorskip("xformers")
    assert _enable_xformers(nn.Linear(4, 4), "jscc_model") is False


def test_enable_xformers_claimed_for_mdtv2_like():
    from sgdjscc_lab.training.perf import _enable_xformers
    pytest.importorskip("xformers")

    class MDTv2(nn.Module):    # class name prefix marks it xformers-native
        def forward(self, x): return x
    assert _enable_xformers(MDTv2(), "diffusion") is True

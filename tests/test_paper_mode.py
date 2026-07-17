"""tests/test_paper_mode.py – CPU tests for the paper-faithful (paper_mode) path.

Covers the paper-gap-closure work (docs/paper_gap_closure.md):
  [1] MuGE edge source (muge_sidecar load + muge_runtime guard)
  [2] paper_mode blocks auto-generated captions
  [3] learned CFG null token
  [4] edge_jscc arch='paper' guardrail
  [5] multi-SNR edge codec forward (snr override)
  [7] complex CSI ops (roundtrip + two-step equalization + alternating loop)
  [8] paper_* config load + stage/paper-mode validation

All tests run on CPU with tiny synthetic tensors (no checkpoints / GPU).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab import paper_mode  # noqa: E402
from sgdjscc_lab.paper_mode import PaperModeError  # noqa: E402

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _img(p: Path, size=16):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), (123, 80, 200)).save(p)


# ── [8] paper config load + paper_mode validation ─────────────────────────────

@pytest.mark.parametrize("name", [
    "paper_train_jscc", "paper_train_text_dm", "paper_train_edge_codec",
    "paper_train_controlnet", "paper_eval_awgn",
])
def test_paper_configs_load(name):
    from sgdjscc_lab.config import load_config
    cfg = load_config(str(_CONFIGS / f"{name}.yaml"))
    assert bool(OmegaConf.select(cfg, "paper_mode", default=False)) is True


def test_paper_text_dm_passes_validation_and_paper_mode():
    """paper_train_text_dm validates and passes paper_mode guardrails (coco_json)."""
    from sgdjscc_lab.config import load_config
    from sgdjscc_lab.training.stages import validate_stage_config
    cfg = load_config(str(_CONFIGS / "paper_train_text_dm.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"train_input_path": "/tmp/x"}))
    validate_stage_config(cfg, "text_dm")
    # No auto-caption sentinel + coco_json source + learned null → passes.
    paper_mode.enforce(cfg, "text_dm", input_dirs=[])


def test_paper_mode_requires_jscc_gan_and_assumed_match():
    """Stage-1 paper mode rejects MSE-only and stale assumed-hparam wiring."""
    cfg = OmegaConf.create({
        "paper_mode": True,
        "paper_assumed_hparams": {
            "optimizer": {"lr": 1e-4, "weight_decay": 1e-5},
            "jscc_gan": {"weight": 0.5, "mode": "hinge", "lr": 1e-4,
                          "ndf": 64, "n_layers": 3, "norm": "batch"},
        },
        "train": {
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "jscc": {"gan": {"enabled": False, "weight": 0.5, "mode": "hinge",
                              "lr": 1e-4, "ndf": 64, "n_layers": 3, "norm": "batch"}},
        },
    })
    with pytest.raises(PaperModeError, match="MSE \\+ patch-GAN"):
        paper_mode.enforce(cfg, "jscc", input_dirs=[])

    cfg.train.jscc.gan.enabled = True
    paper_mode.enforce(cfg, "jscc", input_dirs=[])

    cfg.train.jscc.gan.weight = 0.25
    with pytest.raises(PaperModeError, match="paper_assumed_hparams"):
        paper_mode.enforce(cfg, "jscc", input_dirs=[])


def test_paper_mode_requires_trained_edge_checkpoint(tmp_path):
    """Stage-3 paper mode refuses null/missing edge_jscc checkpoints."""
    base = {
        "paper_mode": True,
        "paper_assumed_hparams": {
            "optimizer": {"lr": 1e-4, "weight_decay": 1e-5},
            "dm": {"cfg_dropout_prob": 0.1, "cfg_null_mode": "learned"},
        },
        "train": {
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "dataset": {"caption_source": "coco_json", "edge_source": "muge_sidecar"},
            "dm": {"cfg_dropout_prob": 0.1, "cfg_null_mode": "learned"},
            "controlnet": {
                "edge_transport": "edge_jscc",
                "edge_jscc": {"checkpoint": None},
            },
        },
    }
    cfg = OmegaConf.create(base)
    with pytest.raises(PaperModeError, match="TRAINED edge_codec"):
        paper_mode.enforce(cfg, "controlnet", input_dirs=[])

    cfg.train.controlnet.edge_jscc.checkpoint = str(tmp_path / "missing.pth")
    with pytest.raises(PaperModeError, match="missing edge_codec"):
        paper_mode.enforce(cfg, "controlnet", input_dirs=[])

    ckpt = tmp_path / "best.pth"
    ckpt.write_bytes(b"placeholder")
    cfg.train.controlnet.edge_jscc.checkpoint = str(ckpt)
    paper_mode.enforce(cfg, "controlnet", input_dirs=[])


# ── [2] caption guardrails ────────────────────────────────────────────────────

def test_paper_mode_blocks_filename_caption():
    cfg = OmegaConf.create({"paper_mode": True,
                            "train": {"dataset": {"caption_source": "filename"}}})
    with pytest.raises(PaperModeError):
        paper_mode.enforce_caption_policy(cfg, "text_dm", input_dirs=[])


def test_paper_mode_blocks_autocaption_sentinel(tmp_path):
    (tmp_path / paper_mode.AUTOCAPTION_SENTINEL).write_text("{}")
    cfg = OmegaConf.create({"paper_mode": True,
                            "train": {"dataset": {"caption_source": "sidecar"}}})
    with pytest.raises(PaperModeError, match="auto-generated captions"):
        paper_mode.enforce_caption_policy(cfg, "text_dm", input_dirs=[tmp_path])


def test_generate_captions_writes_sentinel_and_paper_mode_blocks(tmp_path):
    """generate_captions writes the provenance sentinel → paper_mode then blocks it."""
    import importlib.util
    d = tmp_path / "celeba" / "train"
    _img(d / "img0.png")
    spec = importlib.util.spec_from_file_location(
        "gen_caps", str(Path(__file__).resolve().parents[1] / "scripts" / "generate_captions.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    gen.generate_captions(str(d), mode="fixed", text="a photo of a person")
    assert (d / paper_mode.AUTOCAPTION_SENTINEL).is_file()
    cfg = OmegaConf.create({"paper_mode": True,
                            "train": {"dataset": {"caption_source": "sidecar"}}})
    with pytest.raises(PaperModeError):
        paper_mode.enforce_caption_policy(cfg, "text_dm", input_dirs=[d])


# ── [1] MuGE edge sources ─────────────────────────────────────────────────────

def test_muge_sidecar_dataset_reads_precomputed(tmp_path):
    """text_image_edge with edge_source=muge_sidecar loads <stem>_muge.png."""
    from sgdjscc_lab.data.datasets import build_dataset_for_stage
    d = tmp_path / "train"
    _img(d / "a.png")
    (d / "a.txt").write_text("a caption")
    _img(d / "a_muge.png")          # precomputed MuGE soft edge
    cfg = OmegaConf.create({"train": {"stage": "controlnet", "image_size": 16,
                                      "dataset": {"type": "text_image_edge",
                                                  "caption_source": "sidecar",
                                                  "edge_source": "muge_sidecar"}}})
    ds = build_dataset_for_stage(str(d), cfg, training=True, stage="controlnet")
    item = ds[0]
    assert item["edge"].shape[0] == 1 and "caption" in item


def test_muge_runtime_guard_without_extractor():
    """_load_edge_map raises a clear error if muge_runtime has no extractor."""
    from sgdjscc_lab.data.datasets import _load_edge_map
    from sgdjscc_lab.training.stages import StageConfigError
    with pytest.raises(StageConfigError):
        _load_edge_map(Path("x.png"), torch.rand(3, 16, 16), "muge_runtime", None, None)


def test_muge_reduce_shapes():
    from sgdjscc_lab.data.datasets import muge_reduce
    out = muge_reduce(torch.rand(1, 11, 8, 8))
    assert out.shape == (1, 8, 8) and out.min() >= 0 and out.max() <= 1


def test_muge_sidecar_edge_uncertainty_npy_resizes(tmp_path):
    """muge_sidecar loads a 2-channel .npy sidecar and resizes it to the image transform."""
    import numpy as np
    from sgdjscc_lab.data.datasets import build_dataset_for_stage
    d = tmp_path / "train"
    _img(d / "a.png", size=24)
    (d / "a.txt").write_text("a caption")
    np.save(d / "a_muge.npy", np.random.rand(2, 10, 12).astype("float32"))
    cfg = OmegaConf.create({
        "train": {
            "stage": "controlnet",
            "dataset": {
                "type": "text_image_edge",
                "caption_source": "sidecar",
                "edge_source": "muge_sidecar",
                "muge_repr": "edge_uncertainty",
            },
            "transforms": {"resize_to": 16, "crop_mode": "center"},
        }
    })
    ds = build_dataset_for_stage(str(d), cfg, training=True, stage="controlnet")
    item = ds[0]
    assert item["edge"].shape == (2, 16, 16)


def test_muge_sidecar_requires_npy_for_multichannel_repr(tmp_path):
    """edge_uncertainty/multi should fail early if only a legacy png sidecar exists."""
    from sgdjscc_lab.data.datasets import build_dataset_for_stage
    d = tmp_path / "train"
    _img(d / "a.png")
    (d / "a.txt").write_text("a caption")
    _img(d / "a_muge.png")
    cfg = OmegaConf.create({
        "train": {
            "stage": "controlnet",
            "dataset": {
                "type": "text_image_edge",
                "caption_source": "sidecar",
                "edge_source": "muge_sidecar",
                "muge_repr": "edge_uncertainty",
            },
        }
    })
    ds = build_dataset_for_stage(str(d), cfg, training=True, stage="controlnet")
    with pytest.raises(FileNotFoundError, match="_muge.npy"):
        _ = ds[0]


def test_multichannel_muge_edge_codec_runner_path(tmp_path):
    """Synthetic multi-channel MuGE sidecar flows dataset -> batch -> edge_codec runner."""
    import numpy as np
    from sgdjscc_lab.data.datasets import build_dataset_for_stage, collate_stage_batch
    from sgdjscc_lab.models.edge_jscc import EdgeJSCC
    from sgdjscc_lab.training.stage_runners import EdgeCodecStageRunner
    d = tmp_path / "train"
    _img(d / "a.png", size=24)
    np.save(d / "a_muge.npy", np.random.rand(2, 10, 12).astype("float32"))
    cfg = OmegaConf.create({
        "train": {
            "stage": "edge_codec",
            "batch_size": 1,
            "dataset": {
                "type": "edge",
                "edge_source": "muge_sidecar",
                "muge_repr": "edge_uncertainty",
            },
            "transforms": {"resize_to": 16, "crop_mode": "center"},
            "edge_codec": {"bce_weight": 1.0, "dice_weight": 1.0, "multi_snr": {"enabled": False}},
            "lr": 1e-4,
            "weight_decay": 0.0,
        }
    })
    ds = build_dataset_for_stage(str(d), cfg, training=True, stage="edge_codec")
    batch = collate_stage_batch([ds[0]])
    codec = EdgeJSCC(latent_ch=16, base_ch=8, with_decoder=True, channel=None, in_ch=2)
    runner = EdgeCodecStageRunner(codec, cfg, "cpu", [{"params": list(codec.parameters())}])
    out = runner.forward(batch)
    assert "loss" in out and float(out["loss"]) >= 0.0


# ── [4] edge_jscc paper guardrail + [5] multi-SNR ─────────────────────────────

def test_edge_jscc_arch_paper_guardrail():
    from sgdjscc_lab.models.edge_jscc import EdgeJSCC
    with pytest.raises(NotImplementedError, match="UNSUPPORTED"):
        EdgeJSCC(latent_ch=16, arch="paper")


def test_edge_jscc_snr_override_forward():
    from sgdjscc_lab.models.edge_jscc import EdgeJSCC
    ej = EdgeJSCC(latent_ch=16, base_ch=16, with_decoder=True, channel=None)
    e = torch.rand(2, 1, 64, 64)
    assert ej.reconstruct(e, snr_db=3.0).shape == (2, 1, 64, 64)
    assert ej.encode(e, snr_db=15.0).shape[1] == 16


# ── [3] learned CFG null token ────────────────────────────────────────────────

def test_learned_null_token_dropout():
    from sgdjscc_lab.training.stage_runners import (
        LearnedNullToken, apply_cfg_label_dropout)
    labels = torch.randn(8, 32)
    tok = LearnedNullToken()
    assert tok.materialize(labels) and tok.token.shape == (1, 32)
    # prob=1 → every row replaced by the null token.
    out = apply_cfg_label_dropout(labels, 1.0, True, tok.token)
    assert torch.allclose(out, tok.token.expand_as(out))
    # zero-mode (null_token=None) → dropped rows are zero.
    out0 = apply_cfg_label_dropout(labels, 1.0, True, None)
    assert torch.allclose(out0, torch.zeros_like(out0))


# ── [7] complex CSI ───────────────────────────────────────────────────────────

def test_complex_roundtrip_and_equalize():
    from sgdjscc_lab.channels import complex_ops as C
    x = torch.randn(2, 2, 8, 8)                      # 2-channel real/imag latent
    z = C.two_channel_to_complex(x)
    assert torch.allclose(C.complex_to_two_channel(z), x)
    # known channel, zero noise → two-step equalize recovers z exactly.
    h = torch.polar(torch.ones(2, 1, 1, 1), torch.full((2, 1, 1, 1), 0.7))
    y = C.apply_complex_channel(z, h, sigma=0.0)
    z_eq = C.two_step_equalize(y, h, sigma=0.0)
    assert torch.allclose(z_eq, z, atol=1e-5)


def test_alternating_phase_snr_shapes():
    from sgdjscc_lab.channels import complex_ops as C
    from sgdjscc_lab.models.csi_estimation import build_csi_estimators
    snr, phase = build_csi_estimators(latent_ch=2, with_phase=True)
    y = torch.randn(3, 2, 16, 16)
    y_eq, alpha, phi = C.alternating_phase_snr_equalize(y, snr, phase, max_iter=2)
    assert y_eq.shape == y.shape and alpha.shape == (3, 1) and phi.shape == (3, 1)


# ── eval guardrail (paper_mode now wired into evaluate.py) ────────────────────

def test_enforce_eval_blocks_extensions():
    cfg = OmegaConf.create({"paper_mode": True, "use_phase5": True})
    with pytest.raises(PaperModeError, match="extensions"):
        paper_mode.enforce_eval(cfg)
    cfg2 = OmegaConf.create({"paper_mode": True,
                             "train": {"controlnet": {"edge_transport": "shared_vae"}}})
    with pytest.raises(PaperModeError, match="shared_vae"):
        paper_mode.enforce_eval(cfg2)


def test_paper_eval_config_passes_enforce_eval():
    from sgdjscc_lab.config import load_config
    cfg = load_config(str(_CONFIGS / "paper_eval_awgn.yaml"))
    paper_mode.enforce_eval(cfg)            # must NOT raise (extensions are off)
    # off by default → enforce_eval is a no-op.
    paper_mode.enforce_eval(OmegaConf.create({"use_phase5": True}))


def test_enforce_eval_metrics_requires_full_paper_set():
    from sgdjscc_lab.utils.metric_profiles import resolve_profile
    on = OmegaConf.create({"paper_mode": True})
    paper_set = resolve_profile("paper")           # PSNR/LPIPS/CLIP(x2)/FID
    extended_set = resolve_profile("extended")     # has SSIM/SRS/object… (non-paper)
    # --profile extended under paper_mode → rejected (non-paper metrics enabled).
    with pytest.raises(PaperModeError, match="non-paper"):
        paper_mode.enforce_eval_metrics(on, extended_set, no_clip=False)
    # --no-clip under paper_mode → rejected (CLIP is a paper metric).
    with pytest.raises(PaperModeError, match="no-clip"):
        paper_mode.enforce_eval_metrics(on, paper_set, no_clip=True)
    # a REDUCED custom set (missing CLIP/FID) → rejected (not the FULL paper set).
    with pytest.raises(PaperModeError, match="missing"):
        paper_mode.enforce_eval_metrics(on, {"psnr", "lpips"}, no_clip=False)
    # the EXACT full paper metric set → passes.
    paper_mode.enforce_eval_metrics(on, set(paper_set), no_clip=False)
    # paper_mode off → no-op even with a reduced set / no_clip.
    paper_mode.enforce_eval_metrics(OmegaConf.create({}), {"psnr"}, no_clip=True)


# ── [3] learned CFG null token — save/resume ──────────────────────────────────

def test_learned_null_token_save_resume():
    """A trained null token round-trips through state_dict (resume restores it)."""
    from sgdjscc_lab.training.stage_runners import LearnedNullToken
    src = LearnedNullToken()
    src.materialize(torch.randn(4, 32))
    with torch.no_grad():
        src.token.add_(1.5)                 # "trained" values
    sd = src.state_dict()
    assert "token" in sd
    dst = LearnedNullToken()                # fresh: token is None
    dst.load_state_dict(sd)                 # _load_from_state_dict materialises + copies
    assert dst.token is not None and torch.allclose(dst.token, src.token)


def test_cfg_null_token_eager_setup_registers_with_optimizer():
    """DDP-safe eager null token: created + registered with the optimizer at setup
    (no rank-local lazy parameter), and returned via the module forward."""
    from sgdjscc_lab.training.stage_runners import StageRunner
    r = StageRunner.__new__(StageRunner)    # bypass heavy __init__
    r.cfg = OmegaConf.create({"train": {"dm": {"cfg_null_mode": "learned"}}})
    r.device = torch.device("cpu")
    r._ddp_modules = []
    base = torch.nn.Parameter(torch.zeros(3))
    r.optimizer = torch.optim.AdamW([base], lr=1e-3)
    n0 = len(r.optimizer.param_groups)
    r._setup_cfg_null(torch.randn(2, 8))             # eager: token created + registered
    assert r._null_core.token is not None and tuple(r._null_core.token.shape) == (1, 8)
    assert len(r.optimizer.param_groups) == n0 + 1   # registered exactly once at setup
    tok = r._cfg_null_token(torch.randn(2, 8))       # returned via module forward
    assert tuple(tok.shape) == (1, 8)
    # zero mode → no token, no extra param group.
    r2 = StageRunner.__new__(StageRunner)
    r2.cfg = OmegaConf.create({"train": {"dm": {"cfg_null_mode": "zero"}}})
    r2.device = torch.device("cpu"); r2._ddp_modules = []; r2.optimizer = None
    r2._setup_cfg_null(torch.randn(2, 8))
    assert r2._null_core is None and r2._cfg_null_token(torch.randn(2, 8)) is None

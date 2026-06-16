"""tests/test_train_stages.py – Unit tests for the stage-aware training framework.

No GPU, no checkpoints, no SGDJSCC imports required.  Real model modules are
replaced with lightweight stubs so the stage wiring (config validation, dataset
selection, freeze policy, fixed-SNR, masked/unmasked DM loss) is exercised in
isolation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Make the package importable without an editable install.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.training.stages import (
    STAGE_JSCC, STAGE_TEXT_DM, STAGE_CONTROLNET, STAGE_EDGE_CODEC,
    STAGE_CSI_ESTIMATION, STAGE_END_TO_END_FT,
    StageConfigError, resolve_stage, resolve_dataset_type, validate_stage_config,
)
from sgdjscc_lab.training.noise_schedule import SigmoidNoiseScheduler
from sgdjscc_lab.training.losses import (
    DiffusionF0Loss, JSCCStageLoss, PatchDiscriminator, build_discriminator,
    EndToEndFTLoss, EdgeCodecLoss, build_stage_loss,
)
from sgdjscc_lab.training.freeze import apply_stage_freeze_policy
from sgdjscc_lab.training.stage_runners import (
    JSCCStageRunner, TextDMStageRunner, ControlNetStageRunner, EndToEndFTStageRunner,
    EdgeCodecStageRunner, CSIEstimationStageRunner,
)
from sgdjscc_lab.training.edge_transport import (
    resolve_edge_transport, build_edge_transport, build_edge_codec,
    EDGE_TRANSPORT_SHARED_VAE, EDGE_TRANSPORT_EDGE_JSCC,
)
from sgdjscc_lab.models.edge_jscc import EdgeJSCC, EdgeJSCCViTEncoder, EdgeJSCCViTDecoder
from sgdjscc_lab.data.datasets import (
    ImageOnlyDataset, TextImageDataset, TextImageEdgeDataset, EdgeOnlyDataset,
    build_dataset_for_stage,
)
from sgdjscc_lab.io import save_tensor_as_image


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_images(root: Path, n: int = 3, size: int = 64) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        save_tensor_as_image(torch.rand(3, size, size), root / f"img_{i}.png")
    return root


def _cfg(**train_kw) -> OmegaConf:
    base = {
        "train_input_path": "/tmp/does_not_matter",
        "train": {
            "stage": STAGE_JSCC,
            "jscc": {"snr_db": 10.0},
            "dataset": {"type": "auto"},
            "transforms": {"resize_to": 32, "crop_mode": "center"},
        },
    }
    cfg = OmegaConf.create(base)
    if train_kw:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"train": train_kw}))
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 1) Stage config validation
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_stage_default_and_unknown():
    assert resolve_stage(OmegaConf.create({})) == STAGE_JSCC
    with pytest.raises(StageConfigError):
        resolve_stage(OmegaConf.create({"train": {"stage": "bogus"}}))


def test_validate_jscc_ok_and_requires_snr():
    cfg = _cfg()
    assert validate_stage_config(cfg) == STAGE_JSCC
    # remove snr → error
    cfg2 = OmegaConf.create({"train_input_path": "x", "train": {"stage": "jscc"}})
    with pytest.raises(StageConfigError):
        validate_stage_config(cfg2)


def test_validate_text_dm_requires_caption_source():
    cfg = OmegaConf.create({"train_input_path": "x", "train": {"stage": "text_dm"}})
    with pytest.raises(StageConfigError, match="caption"):
        validate_stage_config(cfg)
    cfg.train.dataset = {"caption_source": "sidecar"}
    assert validate_stage_config(cfg) == STAGE_TEXT_DM


def test_validate_controlnet_requires_edge_source():
    cfg = OmegaConf.create({
        "train_input_path": "x",
        "train": {"stage": "controlnet", "dataset": {"caption_source": "sidecar"}},
    })
    with pytest.raises(StageConfigError, match="edge"):
        validate_stage_config(cfg)
    cfg.train.dataset.edge_source = "canny"
    assert validate_stage_config(cfg) == STAGE_CONTROLNET


def test_validate_controlnet_caption_source_completeness():
    # unknown caption_source rejected (same completeness as text_dm)
    cfg = OmegaConf.create({
        "train_input_path": "x",
        "train": {"stage": "controlnet",
                  "dataset": {"caption_source": "bogus", "edge_source": "canny"}},
    })
    with pytest.raises(StageConfigError, match="caption_source"):
        validate_stage_config(cfg)
    # manifest caption_source without caption_path rejected
    cfg.train.dataset.caption_source = "manifest"
    with pytest.raises(StageConfigError, match="caption_path"):
        validate_stage_config(cfg)


def test_validate_text_dm_caption_source_value_and_manifest():
    cfg = OmegaConf.create({
        "train_input_path": "x",
        "train": {"stage": "text_dm", "dataset": {"caption_source": "manifest"}},
    })
    with pytest.raises(StageConfigError, match="caption_path"):
        validate_stage_config(cfg)
    cfg.train.dataset.caption_source = "nope"
    with pytest.raises(StageConfigError, match="caption_source"):
        validate_stage_config(cfg)


def test_validate_missing_train_input():
    # No train_input_path and no file_list mode → must fail (message now also
    # mentions the file_list alternative).
    with pytest.raises(StageConfigError, match="training images"):
        validate_stage_config(OmegaConf.create({"train": {"stage": "jscc"}}))


# ─────────────────────────────────────────────────────────────────────────────
# 2) Dataset builder selection per stage
# ─────────────────────────────────────────────────────────────────────────────

def test_dataset_type_resolution():
    assert resolve_dataset_type(_cfg(), STAGE_JSCC) == "image"
    assert resolve_dataset_type(_cfg(stage="text_dm"), STAGE_TEXT_DM) == "text_image"
    assert resolve_dataset_type(
        _cfg(stage="controlnet"), STAGE_CONTROLNET) == "text_image_edge"


def test_build_dataset_image_stage(tmp_path):
    img_dir = _make_images(tmp_path / "imgs")
    ds = build_dataset_for_stage(str(img_dir), _cfg(), stage=STAGE_JSCC)
    assert isinstance(ds, ImageOnlyDataset)
    item = ds[0]
    assert set(item.keys()) == {"image", "path"}
    assert item["image"].shape == (3, 32, 32)


def test_build_dataset_text_image_stage(tmp_path):
    img_dir = _make_images(tmp_path / "imgs")
    cfg = _cfg(stage="text_dm", dataset={"type": "auto", "caption_source": "filename"})
    ds = build_dataset_for_stage(str(img_dir), cfg, stage=STAGE_TEXT_DM)
    assert isinstance(ds, TextImageDataset)
    item = ds[0]
    assert set(item.keys()) == {"image", "caption", "path"}
    assert isinstance(item["caption"], str) and item["caption"]


def test_build_dataset_text_image_edge_stage(tmp_path):
    img_dir = _make_images(tmp_path / "imgs")
    cfg = _cfg(stage="controlnet",
               dataset={"type": "auto", "caption_source": "filename",
                        "edge_source": "canny"})
    ds = build_dataset_for_stage(str(img_dir), cfg, stage=STAGE_CONTROLNET)
    assert isinstance(ds, TextImageEdgeDataset)
    item = ds[0]
    assert set(item.keys()) == {"image", "caption", "edge", "path"}
    assert item["edge"].shape == (1, 32, 32)


# ─────────────────────────────────────────────────────────────────────────────
# 3) ControlNet freeze policy enforcement
# ─────────────────────────────────────────────────────────────────────────────

class _StubDenoiser(nn.Module):
    """Mimics MDTv2_ControlNet: a frozen-able base + ControlNet branches."""

    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)                       # base DM params
        self.en_inblocks_controlnet = nn.ModuleList([nn.Linear(4, 4)])
        self.en_outblocks_controlnet = nn.ModuleList([nn.Linear(4, 4)])


def _models_with_denoiser(denoiser):
    return SimpleNamespace(
        jscc_model=nn.Linear(2, 2),
        sem_pipeline=SimpleNamespace(model=denoiser),
        text_extractor=None,
        edge_extractor=None,
    )


def test_controlnet_freeze_policy_default():
    denoiser = _StubDenoiser()
    models = _models_with_denoiser(denoiser)
    cfg = OmegaConf.create({"train": {"controlnet": {"allow_unfrozen_base_dm": False}}})

    groups, report = apply_stage_freeze_policy(models, cfg, STAGE_CONTROLNET)

    # base DM frozen, control branches trainable
    assert all(not p.requires_grad for p in denoiser.base.parameters())
    assert all(p.requires_grad for p in denoiser.en_inblocks_controlnet.parameters())
    assert all(p.requires_grad for p in denoiser.en_outblocks_controlnet.parameters())
    names = {g["name"] for g in groups}
    assert "en_inblocks_controlnet" in names and "en_outblocks_controlnet" in names
    assert "base_diffusion" in report["frozen"]


def test_controlnet_freeze_policy_override_danger_flag():
    denoiser = _StubDenoiser()
    models = _models_with_denoiser(denoiser)
    cfg = OmegaConf.create({"train": {"controlnet": {"allow_unfrozen_base_dm": True}}})

    groups, report = apply_stage_freeze_policy(models, cfg, STAGE_CONTROLNET)
    assert all(p.requires_grad for p in denoiser.base.parameters())
    assert "base_diffusion_unfrozen" in report["forced"]

    # No parameter may appear in more than one optimizer group (else AdamW errors).
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "duplicate params across optimizer groups"
    # AdamW must construct without raising on the override path.
    torch.optim.AdamW(groups, lr=1e-4)


def test_jscc_stage_freeze_trains_jscc_only():
    denoiser = _StubDenoiser()
    models = _models_with_denoiser(denoiser)
    cfg = OmegaConf.create({"trainable_modules": {"freeze_jscc": False}})
    groups, report = apply_stage_freeze_policy(models, cfg, STAGE_JSCC)
    assert "jscc_model" in report["trainable"]
    # the diffusion denoiser stays frozen in the JSCC stage
    assert all(not p.requires_grad for p in denoiser.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# 4) JSCC stage fixed-SNR
# ─────────────────────────────────────────────────────────────────────────────

class _FakeVAE:
    def encode(self, x):
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=x))

    def decode(self, z):
        return (z,)


class _FakeJSCC:
    def __init__(self):
        self.snr = 0.0
        self.vae = _FakeVAE()
        self.channel_snr_seen = None

    def normalize(self, x):
        return x

    def channel(self, x):
        self.channel_snr_seen = self.snr
        return x


def test_jscc_runner_applies_fixed_snr():
    jscc = _FakeJSCC()
    cfg = OmegaConf.create({"train": {"jscc": {"snr_db": 10.0, "gan": {"enabled": False}}}})
    runner = JSCCStageRunner(jscc, cfg, torch.device("cpu"), param_groups=[])
    assert runner.snr_db == 10.0
    runner._reconstruct(torch.rand(2, 3, 16, 16))
    assert jscc.snr == 10.0
    assert jscc.channel_snr_seen == 10.0


def test_jscc_loss_gan_term_toggles():
    loss = JSCCStageLoss(gan_weight=0.5)
    recon = torch.rand(2, 3, 64, 64, requires_grad=True)
    target = torch.rand(2, 3, 64, 64)
    out_no_gan = loss(recon, target, disc=None)
    assert "loss_gan" not in out_no_gan and "loss_mse" in out_no_gan
    disc = PatchDiscriminator()
    out_gan = loss(recon, target, disc=disc)
    assert "loss_gan" in out_gan


def test_jscc_loss_lpips_term_toggles_and_backprops():
    # stub LPIPS (avoids the alexnet download): a differentiable perceptual proxy
    calls = {"n": 0}
    def stub_lpips(a, b):                     # a,b in [-1,1]
        calls["n"] += 1
        return ((a - b) ** 2).mean()
    loss = JSCCStageLoss(lpips_weight=0.5, lpips_fn=stub_lpips)
    recon = torch.rand(2, 3, 16, 16, requires_grad=True)
    target = torch.rand(2, 3, 16, 16)
    out = loss(recon, target)
    assert "loss_lpips" in out and calls["n"] == 1
    out["loss"].backward()                    # perceptual term is in the graph
    assert recon.grad is not None
    # weight 0 → no LPIPS term, no call
    out0 = JSCCStageLoss(lpips_weight=0.0, lpips_fn=stub_lpips)(recon, target)
    assert "loss_lpips" not in out0


def test_jscc_loss_gan_and_lpips_combined():
    stub_lpips = lambda a, b: ((a - b) ** 2).mean()
    loss = JSCCStageLoss(gan_weight=0.5, lpips_weight=0.3, lpips_fn=stub_lpips)
    recon = torch.rand(2, 3, 64, 64, requires_grad=True)
    target = torch.rand(2, 3, 64, 64)
    out = loss(recon, target, disc=PatchDiscriminator())
    assert {"loss_mse", "loss_gan", "loss_lpips", "loss"} <= set(out)


def test_build_stage_loss_jscc_reads_lpips_config():
    cfg = OmegaConf.create({"train": {"jscc": {
        "gan": {"enabled": False},
        "lpips": {"enabled": True, "weight": 0.25}}}})
    loss = build_stage_loss(cfg, STAGE_JSCC)
    assert isinstance(loss, JSCCStageLoss) and loss.lpips_weight == 0.25


# ─────────────────────────────────────────────────────────────────────────────
# 5) DM stage: masked + unmasked loss terms
# ─────────────────────────────────────────────────────────────────────────────

class _StubDMDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, ft, noise_level, labels, c=None, enable_mask=False):
        # Return a tensor shaped like f0 with a grad-bearing parameter.
        return ft * self.scale


def _dm_runner(cls, with_edge=False):
    denoiser = _StubDMDenoiser()
    cfg = OmegaConf.create({"train": {"dm": {"use_masked_branch": True, "mask_weight": 1.0}}})
    pg = [{"params": list(denoiser.parameters()), "name": "diffusion"}]

    def enc_latent(x):
        return torch.randn(x.shape[0], 4, 8, 8)

    def enc_text(caps):
        return torch.zeros(len(caps), 8)

    if with_edge:
        def enc_edge(e):
            return torch.randn(e.shape[0], 4, 8, 8)
        return ControlNetStageRunner(
            denoiser, enc_latent, enc_text, enc_edge,
            cfg, torch.device("cpu"), pg,
        )
    return TextDMStageRunner(
        denoiser, enc_latent, enc_text, cfg, torch.device("cpu"), pg,
    )


def test_text_dm_computes_masked_and_unmasked_terms():
    runner = _dm_runner(TextDMStageRunner)
    batch = {"image": torch.rand(2, 3, 16, 16), "caption": ["a", "b"]}
    out = runner.forward(batch)
    assert "loss_dm_unmasked" in out and "loss_dm_masked" in out
    assert out["loss"].requires_grad
    # training_step runs backward + step without error
    metrics = runner.training_step(batch)
    assert "loss_dm_unmasked" in metrics and "loss_dm_masked" in metrics


def test_controlnet_runner_uses_edge_and_both_terms():
    runner = _dm_runner(ControlNetStageRunner, with_edge=True)
    batch = {
        "image": torch.rand(2, 3, 16, 16),
        "edge": torch.rand(2, 1, 16, 16),
        "caption": ["a", "b"],
    }
    out = runner.forward(batch)
    assert "loss_dm_unmasked" in out and "loss_dm_masked" in out


def test_diffusion_f0_loss_single_term_when_unmasked_only():
    loss = DiffusionF0Loss(mask_weight=1.0)
    f0 = torch.randn(2, 4, 8, 8)
    pred = torch.randn(2, 4, 8, 8, requires_grad=True)
    out = loss(f0, pred, pred_masked=None)
    assert "loss_dm_unmasked" in out and "loss_dm_masked" not in out


# ─────────────────────────────────────────────────────────────────────────────
# 6) Noise scheduler
# ─────────────────────────────────────────────────────────────────────────────

def test_sigmoid_scheduler_shapes_and_range():
    sch = SigmoidNoiseScheduler()
    f0 = torch.randn(4, 4, 8, 8)
    ft, noise_level, noise, t = sch.add_noise(f0)
    assert ft.shape == f0.shape
    assert noise_level.shape == (4, 1)
    assert t.shape == (4,)
    assert torch.all((noise_level >= 0) & (noise_level <= 1))
    # β̄_t monotonic-ish: larger t → larger noise variance
    b_lo = sch.beta_bar(torch.tensor([0.1]))
    b_hi = sch.beta_bar(torch.tensor([0.9]))
    assert float(b_hi) > float(b_lo)


# ─────────────────────────────────────────────────────────────────────────────
# 7) Edge transport modes (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────

def test_edge_transport_resolution_and_validation():
    assert resolve_edge_transport(OmegaConf.create({})) == EDGE_TRANSPORT_SHARED_VAE
    cfg = OmegaConf.create({"train": {"controlnet": {"edge_transport": "edge_jscc"}}})
    assert resolve_edge_transport(cfg) == EDGE_TRANSPORT_EDGE_JSCC
    bad = OmegaConf.create({"train": {"controlnet": {"edge_transport": "nope"}}})
    with pytest.raises(StageConfigError):
        resolve_edge_transport(bad)


def test_edge_transport_shared_vae_vs_edge_jscc_shapes():
    jscc = _FakeJSCC()
    edges = torch.rand(2, 1, 128, 128)

    # shared_vae: identity-ish VAE → latent has 3 channels (broadcast edge)
    cfg_sv = OmegaConf.create({"train": {"controlnet": {"edge_transport": "shared_vae"}}})
    t_sv = build_edge_transport(cfg_sv, jscc, torch.device("cpu"))
    c_sv = t_sv(edges)
    assert c_sv.shape[0] == 2

    # edge_jscc: dedicated edge encoder → [B, 16, 16, 16] (no channel for the test)
    cfg_ej = OmegaConf.create({"train": {"controlnet": {
        "edge_transport": "edge_jscc",
        "edge_jscc": {"base_ch": 16, "use_channel": False}}}})
    t_ej = build_edge_transport(cfg_ej, jscc, torch.device("cpu"))
    c_ej = t_ej(edges)
    assert c_ej.shape == (2, 16, 16, 16)
    assert hasattr(t_ej, "module")  # the dedicated edge codec is exposed


# ─────────────────────────────────────────────────────────────────────────────
# 8) Discriminator config-driven
# ─────────────────────────────────────────────────────────────────────────────

def test_build_discriminator_config_driven():
    cfg = OmegaConf.create({"train": {"jscc": {"gan": {
        "ndf": 32, "n_layers": 2, "norm": "instance"}}}})
    disc = build_discriminator(cfg)
    assert isinstance(disc, PatchDiscriminator)
    out = disc(torch.rand(1, 3, 64, 64))
    assert out.shape[1] == 1  # patch logit map


# ─────────────────────────────────────────────────────────────────────────────
# 9) Gradient accumulation
# ─────────────────────────────────────────────────────────────────────────────

def test_grad_accumulation_steps_pattern():
    runner = _dm_runner(TextDMStageRunner)
    # force grad_accum=3 and re-init step controls
    runner.cfg = OmegaConf.merge(runner.cfg, OmegaConf.create(
        {"train": {"grad_accum_steps": 3}}))
    runner._init_step_controls()
    batch = {"image": torch.rand(2, 3, 16, 16), "caption": ["a", "b"]}
    fired = []
    for _ in range(3):
        runner.training_step(batch)
        fired.append(runner.last_step_did_update)
    assert fired == [False, False, True], fired


# ─────────────────────────────────────────────────────────────────────────────
# 10) end_to_end_ft stage validation + freeze policy
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_end_to_end_ft_requires_caption_and_trainable():
    # missing caption_source → error
    cfg = OmegaConf.create({"train_input_path": "x",
                            "train": {"stage": "end_to_end_ft"}})
    with pytest.raises(StageConfigError, match="caption"):
        validate_stage_config(cfg)
    # caption present but nothing trainable → error
    cfg = OmegaConf.create({"train_input_path": "x", "train": {
        "stage": "end_to_end_ft",
        "dataset": {"caption_source": "filename"},
        "end_to_end_ft": {"train_jscc": False, "train_diffusion": False,
                          "train_controlnet": False}}})
    with pytest.raises(StageConfigError, match="nothing to train"):
        validate_stage_config(cfg)
    # valid
    cfg.train.end_to_end_ft.train_jscc = True
    assert validate_stage_config(cfg) == STAGE_END_TO_END_FT


def test_end_to_end_ft_freeze_policy_jscc_and_dm():
    denoiser = _StubDenoiser()
    models = _models_with_denoiser(denoiser)
    cfg = OmegaConf.create({"train": {"end_to_end_ft": {
        "train_jscc": True, "train_diffusion": True, "train_controlnet": False}}})
    groups, report = apply_stage_freeze_policy(models, cfg, STAGE_END_TO_END_FT)
    assert "jscc_model" in report["trainable"]
    assert "diffusion_full" in report["trainable"]
    assert all(p.requires_grad for p in denoiser.parameters())
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))


def test_end_to_end_ft_freeze_policy_controlnet_only():
    denoiser = _StubDenoiser()
    models = _models_with_denoiser(denoiser)
    cfg = OmegaConf.create({"train": {"end_to_end_ft": {
        "train_jscc": True, "train_diffusion": False, "train_controlnet": True}}})
    groups, report = apply_stage_freeze_policy(models, cfg, STAGE_END_TO_END_FT)
    # base DM frozen, control branches trainable
    assert all(not p.requires_grad for p in denoiser.base.parameters())
    assert all(p.requires_grad for p in denoiser.en_inblocks_controlnet.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# 11) Step-based training: global-step save + stop
# ─────────────────────────────────────────────────────────────────────────────

class _StubStepRunner:
    """Minimal runner driving the step-based loop in run_training."""

    def __init__(self):
        self.optimizer = SimpleNamespace(param_groups=[{"lr": 1e-4}])
        self.grad_accum = 1
        self.use_amp = False
        self.last_step_did_update = True
        self.calls = 0

    def set_mode(self, training):  # noqa: D401
        pass

    def training_step(self, batch):
        self.calls += 1
        self.last_step_did_update = True
        return {"loss": 1.0 / self.calls}

    def validation_step(self, batch):
        return {"loss": 0.5}

    def state_modules(self):
        return {}

    def optimizer_state(self):
        return {}


def test_step_based_training_stops_and_saves(tmp_path):
    from sgdjscc_lab.pipelines.train_pipeline import run_training

    ckpt_dir = tmp_path / "ckpt"
    cfg = OmegaConf.create({
        "train_input_path": "x",
        "checkpoint_dir": str(ckpt_dir),
        "train_log_path": str(tmp_path / "log.jsonl"),
        "train": {
            "stage": "jscc", "jscc": {"snr_db": 10.0},
            "max_steps": 5, "save_every_steps": 2, "log_every_steps": 2,
            "lr": 1e-4,
        },
    })
    loader = [{"image": torch.rand(1, 3, 8, 8)} for _ in range(50)]
    runner = _StubStepRunner()
    run_training(cfg, models=None, device=torch.device("cpu"),
                 train_loader=loader, val_loader=None, runner=runner)

    assert runner.calls == 5, "must stop exactly at max_steps"
    assert (ckpt_dir / "latest.pth").exists()


def test_step_based_resume_restores_global_step(tmp_path):
    import torch as _torch
    ckpt = tmp_path / "latest.pth"
    _torch.save({"epoch": 1, "global_step": 3, "stage": "jscc",
                 "model_state": {}, "optimizer_state": {}, "best_metric": 0.1}, ckpt)

    from sgdjscc_lab.pipelines.train_pipeline import run_training
    cfg = OmegaConf.create({
        "train_input_path": "x",
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "train_log_path": str(tmp_path / "log.jsonl"),
        "train": {"stage": "jscc", "jscc": {"snr_db": 10.0},
                  "max_steps": 5, "resume": str(ckpt), "lr": 1e-4},
    })
    loader = [{"image": torch.rand(1, 3, 8, 8)} for _ in range(50)]
    runner = _StubStepRunner()
    run_training(cfg, models=None, device=torch.device("cpu"),
                 train_loader=loader, val_loader=None, runner=runner)
    # resumed at step 3 → only 2 more steps to reach max_steps=5
    assert runner.calls == 2, runner.calls


# ─────────────────────────────────────────────────────────────────────────────
# 12) end_to_end_ft loss
# ─────────────────────────────────────────────────────────────────────────────

def test_end_to_end_ft_loss_terms():
    loss = EndToEndFTLoss(recon_weight=1.0, diff_weight=0.5)
    recon = torch.rand(2, 3, 16, 16, requires_grad=True)
    target = torch.rand(2, 3, 16, 16)
    f0 = torch.randn(2, 4, 8, 8)
    pred = torch.randn(2, 4, 8, 8, requires_grad=True)
    out = loss(recon, target, f0, pred)
    assert "loss_recon" in out and "loss_diff" in out and "loss" in out
    # diff-free variant
    out2 = loss(recon, target, None, None)
    assert "loss_diff" not in out2


def test_end_to_end_ft_runner_forward_both_terms():
    jscc = _FakeJSCC()
    denoiser = _StubDMDenoiser()
    cfg = OmegaConf.create({"train": {"end_to_end_ft": {
        "snr_db": 10.0, "recon_weight": 1.0, "diff_weight": 1.0}}})
    pg = [{"params": list(denoiser.parameters()), "name": "diffusion_full"}]
    runner = EndToEndFTStageRunner(
        jscc, denoiser, lambda caps: torch.zeros(len(caps), 8),
        cfg, torch.device("cpu"), pg,
    )
    batch = {"image": torch.rand(2, 3, 16, 16), "caption": ["a", "b"]}
    out = runner.forward(batch)
    assert "loss_recon" in out and "loss_diff" in out
    assert out["loss"].requires_grad
    # training_step runs an optimizer update without error
    metrics = runner.training_step(batch)
    assert "loss_recon" in metrics and "loss_diff" in metrics
    assert runner.last_step_did_update is True


# ─────────────────────────────────────────────────────────────────────────────
# 13) Checkpoint completeness (resume reproducibility)
# ─────────────────────────────────────────────────────────────────────────────

def test_gan_runner_exposes_disc_optimizer_and_scaler_in_state():
    jscc = _FakeJSCC()
    cfg = OmegaConf.create({"train": {"jscc": {
        "snr_db": 10.0, "gan": {"enabled": True, "weight": 0.5, "mode": "hinge",
                                "ndf": 16, "n_layers": 2}}}})
    runner = JSCCStageRunner(jscc, cfg, torch.device("cpu"), param_groups=[])
    assert "d_optimizer" in runner.optimizers()
    assert "d_scaler" in runner.scalers()
    state = runner.get_train_state()
    # discriminator module + both optimizers/scalers captured
    assert "jscc_discriminator" in state["modules"]
    assert "d_optimizer" in state["optimizers"]
    assert "d_scaler" in state["scalers"]


def test_edge_jscc_module_checkpointed_and_round_trips():
    jscc = _FakeJSCC()
    cfg = OmegaConf.create({"train": {"dm": {"use_masked_branch": False},
        "controlnet": {"edge_transport": "edge_jscc",
                       "edge_jscc": {"base_ch": 16, "use_channel": False}}}})
    transport = build_edge_transport(cfg, jscc, torch.device("cpu"))
    runner = ControlNetStageRunner(
        _StubDMDenoiser(), lambda x: torch.randn(x.shape[0], 4, 8, 8),
        lambda caps: torch.zeros(len(caps), 8), transport,
        cfg, torch.device("cpu"),
        [{"params": [], "name": "diffusion"}],
    )
    # the dedicated edge codec is part of the checkpoint
    assert "edge_jscc" in runner.state_modules()
    assert "edge_jscc" in runner.get_train_state()["modules"]

    # Serialize like a real checkpoint (torch.save copies storage), mutate the
    # codec, then restore → weights come back (reproducible `c`).
    import io
    codec = runner.edge_module
    first = list(codec.parameters())[0].clone()
    buf = io.BytesIO()
    torch.save(runner.get_train_state(), buf)
    with torch.no_grad():
        list(codec.parameters())[0].add_(1.0)
    buf.seek(0)
    runner.load_train_state(torch.load(buf))
    assert torch.allclose(list(codec.parameters())[0], first)


def test_get_load_train_state_round_trip_optimizer():
    runner = _dm_runner(TextDMStageRunner)
    batch = {"image": torch.rand(2, 3, 16, 16), "caption": ["a", "b"]}
    runner.training_step(batch)            # populate optimizer moments + accum
    state = runner.get_train_state()
    assert "accum" in state and "optimizer" in state["optimizers"]
    # fresh runner restores cleanly
    runner2 = _dm_runner(TextDMStageRunner)
    runner2.load_train_state(state)
    assert runner2._accum == runner._accum


def test_flush_pending_applies_partial_window():
    runner = _dm_runner(TextDMStageRunner)
    runner.cfg = OmegaConf.merge(runner.cfg, OmegaConf.create(
        {"train": {"grad_accum_steps": 3}}))
    runner._init_step_controls()
    batch = {"image": torch.rand(2, 3, 16, 16), "caption": ["a", "b"]}
    runner.training_step(batch)
    runner.training_step(batch)            # 2 micro-steps: window not full
    assert runner.last_step_did_update is False
    flushed = runner.flush_pending()       # apply the partial window
    assert flushed is True
    assert runner.last_step_did_update is True
    assert runner._accum == 0
    # no-op when nothing pending
    assert runner.flush_pending() is False


def test_e2e_dataset_promoted_to_edge_when_controlnet_trained():
    cfg = OmegaConf.create({"train": {"stage": "end_to_end_ft", "dataset": {"type": "auto"},
                                      "end_to_end_ft": {"train_controlnet": True}}})
    assert resolve_dataset_type(cfg, STAGE_END_TO_END_FT) == "text_image_edge"
    cfg.train.end_to_end_ft.train_controlnet = False
    assert resolve_dataset_type(cfg, STAGE_END_TO_END_FT) == "text_image"


def test_run_training_writes_complete_runner_state(tmp_path):
    from sgdjscc_lab.pipelines.train_pipeline import run_training
    runner = _dm_runner(TextDMStageRunner)
    ckpt_dir = tmp_path / "ckpt"
    cfg = OmegaConf.create({
        "train_input_path": "x",
        "checkpoint_dir": str(ckpt_dir),
        "train_log_path": str(tmp_path / "log.jsonl"),
        "train": {"stage": "text_dm",
                  "dataset": {"caption_source": "filename"},
                  "max_steps": 4, "save_every_steps": 2, "lr": 1e-4},
    })
    loader = [{"image": torch.rand(2, 3, 16, 16), "caption": ["a", "b"]} for _ in range(20)]
    run_training(cfg, models=None, device=torch.device("cpu"),
                 train_loader=loader, val_loader=None, runner=runner)

    state = torch.load(ckpt_dir / "latest.pth")
    assert state["global_step"] == 4
    # full train-state present: modules + optimizer + scaler + accum
    rs = state["runner_state"]
    assert "diffusion" in rs["modules"]
    assert "optimizer" in rs["optimizers"]
    assert "scaler" in rs["scalers"]
    assert "accum" in rs


# ─────────────────────────────────────────────────────────────────────────────
# 14) Step-mode flush consistency + e2e explicit-dataset guard
# ─────────────────────────────────────────────────────────────────────────────

def test_step_mode_flush_reaches_max_steps_and_fires_events(tmp_path):
    from sgdjscc_lab.pipelines.train_pipeline import run_training
    denoiser = _StubDMDenoiser()
    cfg = OmegaConf.create({
        "train_input_path": "x",
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "train_log_path": str(tmp_path / "log.jsonl"),
        "train": {"stage": "text_dm", "dataset": {"caption_source": "filename"},
                  "dm": {"use_masked_branch": False},
                  "grad_accum_steps": 2, "max_steps": 2, "save_every_steps": 2,
                  "lr": 1e-3},
    })
    pg = [{"params": list(denoiser.parameters()), "name": "diffusion"}]
    runner = TextDMStageRunner(
        denoiser, lambda x: torch.randn(x.shape[0], 4, 8, 8),
        lambda caps: torch.zeros(len(caps), 8), cfg, torch.device("cpu"), pg)
    # 3 batches/epoch with grad_accum=2 → 1 update + a partial window flushed at
    # the epoch boundary, which itself reaches max_steps=2 and must STOP there.
    loader = [{"image": torch.rand(2, 3, 16, 16), "caption": ["a", "b"]} for _ in range(3)]
    run_training(cfg, models=None, device=torch.device("cpu"),
                 train_loader=loader, val_loader=None, runner=runner)
    state = torch.load(tmp_path / "ckpt" / "latest.pth")
    assert state["global_step"] == 2, state["global_step"]  # no extra epoch ran


# ─────────────────────────────────────────────────────────────────────────────
# 15) Edge-codec stage (real trainable BCE+Dice edge codec)
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_edge_codec_requires_edge_source_no_caption():
    # missing edge_source → error
    cfg = OmegaConf.create({"train_input_path": "x",
                            "train": {"stage": "edge_codec"}})
    with pytest.raises(StageConfigError, match="edge"):
        validate_stage_config(cfg)
    # edge_source present, NO caption needed → ok
    cfg = OmegaConf.create({"train_input_path": "x", "train": {
        "stage": "edge_codec", "dataset": {"edge_source": "canny"}}})
    assert validate_stage_config(cfg) == STAGE_EDGE_CODEC
    # sidecar without edge_dir → error
    cfg.train.dataset.edge_source = "sidecar"
    with pytest.raises(StageConfigError, match="edge_dir"):
        validate_stage_config(cfg)


def test_edge_codec_dataset_type_and_builder(tmp_path):
    assert resolve_dataset_type(_cfg(stage="edge_codec"), STAGE_EDGE_CODEC) == "edge"
    img_dir = _make_images(tmp_path / "imgs")
    cfg = _cfg(stage="edge_codec", dataset={"type": "auto", "edge_source": "canny"})
    ds = build_dataset_for_stage(str(img_dir), cfg, stage=STAGE_EDGE_CODEC)
    assert isinstance(ds, EdgeOnlyDataset)
    item = ds[0]
    assert set(item.keys()) == {"edge", "path"}
    assert item["edge"].shape == (1, 32, 32)


def test_edge_codec_loss_bce_and_dice():
    loss = EdgeCodecLoss(bce_weight=1.0, dice_weight=0.5)
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    out = loss(logits, target)
    assert "loss_bce" in out and "loss_dice" in out and "loss" in out
    assert out["loss"].requires_grad
    # dice-free variant
    out2 = EdgeCodecLoss(dice_weight=0.0)(logits, target)
    assert "loss_dice" not in out2
    # build_stage_loss routes edge_codec → EdgeCodecLoss
    built = build_stage_loss(
        OmegaConf.create({"train": {"edge_codec": {"bce_weight": 2.0}}}),
        STAGE_EDGE_CODEC)
    assert isinstance(built, EdgeCodecLoss) and built.bce_weight == 2.0


def test_edge_jscc_reconstruct_shape_and_grad():
    codec = EdgeJSCC(latent_ch=16, base_ch=16, downsample_factor=8,
                     channel=None, with_decoder=True)
    edge = torch.rand(2, 1, 128, 128)
    logits = codec.reconstruct(edge)
    assert logits.shape == edge.shape           # back to input resolution
    logits.sum().backward()                     # gradient flows to the codec
    assert any(p.grad is not None for p in codec.parameters())
    # no-decoder codec raises on reconstruct
    enc_only = EdgeJSCC(base_ch=16, with_decoder=False)
    with pytest.raises(RuntimeError, match="decoder"):
        enc_only.reconstruct(edge)


_VIT = {"embed_dim": 32, "depth": 2, "num_heads": 4}


def test_edge_jscc_vit_reconstruct_shape_and_grad():
    codec = EdgeJSCC(latent_ch=16, downsample_factor=8, channel=None,
                     with_decoder=True, arch="vit", vit_cfg=_VIT)
    assert codec.arch == "vit"
    assert isinstance(codec.encoder, EdgeJSCCViTEncoder)
    assert isinstance(codec.decoder, EdgeJSCCViTDecoder)
    edge = torch.rand(2, 1, 64, 64)
    logits = codec.reconstruct(edge)
    assert logits.shape == edge.shape           # back to input resolution
    logits.sum().backward()
    assert any(p.grad is not None for p in codec.parameters())
    # encode aligns the condition latent to the requested (VAE) grid
    c = codec.encode(edge, target_hw=(8, 8))
    assert c.shape == (2, 16, 8, 8)


def test_edge_jscc_vit_snr_cond_adaln_forward_and_grad():
    # adaLN SNR conditioning (WITT-style) is wired and differentiable; the SNR
    # embedder + adaLN blocks actually run in the forward.
    from sgdjscc_lab.models.edge_jscc import SNREmbedder, _AdaLNTransformer
    vit = {**_VIT, "snr_cond": True}
    codec = EdgeJSCC(latent_ch=16, downsample_factor=8, channel=None,
                     with_decoder=True, arch="vit", vit_cfg=vit, snr_db=12.0)
    assert codec._snr_cond is True
    assert isinstance(codec.encoder.blocks, _AdaLNTransformer)
    assert isinstance(codec.encoder.blocks.snr_embed, SNREmbedder)
    edge = torch.rand(2, 1, 64, 64)
    logits = codec.reconstruct(edge)              # uses the fixed snr_db internally
    assert logits.shape == edge.shape
    logits.sum().backward()
    # adaLN params receive gradient (the SNR pathway is in the graph)
    assert any(p.grad is not None for p in codec.encoder.blocks.parameters())
    c = codec.encode(edge, target_hw=(8, 8))
    assert c.shape == (2, 16, 8, 8)


def test_snr_embedder_shapes():
    from sgdjscc_lab.models.edge_jscc import SNREmbedder
    emb = SNREmbedder(hidden_size=32)
    out = emb(torch.tensor([5.0, 10.0, 15.0]))
    assert out.shape == (3, 32)


def test_edge_jscc_vit_snr_cond_checkpoint_round_trips():
    vit = {**_VIT, "snr_cond": True}
    a = EdgeJSCC(latent_ch=16, downsample_factor=8, with_decoder=True,
                 arch="vit", vit_cfg=vit)
    sd = a.state_dict()
    b = EdgeJSCC(latent_ch=16, downsample_factor=8, with_decoder=True,
                 arch="vit", vit_cfg=vit)
    b.load_state_dict(sd)                          # adaLN params load cleanly
    edge = torch.rand(1, 1, 64, 64)
    assert torch.allclose(a.reconstruct(edge), b.reconstruct(edge), atol=1e-5)


def test_edge_jscc_arch_invalid_raises():
    with pytest.raises(ValueError, match="conv.*vit"):
        EdgeJSCC(arch="bogus")


def test_build_edge_codec_arch_vit_selection():
    cfg = OmegaConf.create({"train": {"edge_codec": {
        "arch": "vit", "use_channel": False, "vit": _VIT}}})
    codec = build_edge_codec(cfg, torch.device("cpu"))
    assert isinstance(codec.encoder, EdgeJSCCViTEncoder) and codec.decoder is not None
    # default arch is still conv (backward-compatible)
    conv = build_edge_codec(OmegaConf.create({"train": {"edge_codec": {
        "base_ch": 16, "use_channel": False}}}), torch.device("cpu"))
    assert conv.arch == "conv"


def test_edge_codec_vit_checkpoint_loads_into_stage3_transport(tmp_path):
    cfg_codec = OmegaConf.create({"train": {"edge_codec": {
        "arch": "vit", "use_channel": False, "vit": _VIT}}})
    codec = build_edge_codec(cfg_codec, torch.device("cpu"))
    pg = [{"params": list(codec.parameters()), "name": "edge_jscc"}]
    runner = EdgeCodecStageRunner(codec, cfg_codec, torch.device("cpu"), pg)
    ckpt = tmp_path / "best.pth"
    torch.save({"epoch": 1, "global_step": 1, "stage": "edge_codec",
                "runner_state": runner.get_train_state()}, ckpt)

    jscc = _FakeJSCC()
    cfg_s3 = OmegaConf.create({"train": {"controlnet": {
        "edge_transport": "edge_jscc",
        "edge_jscc": {"arch": "vit", "use_channel": False, "vit": _VIT,
                      "checkpoint": str(ckpt)}}}})
    transport = build_edge_transport(cfg_s3, jscc, torch.device("cpu"))
    tcodec = transport.module
    assert tcodec.arch == "vit" and tcodec.decoder is None   # transport: no decoder
    for k, v in codec.encoder.state_dict().items():          # encoder weights loaded
        assert torch.allclose(tcodec.encoder.state_dict()[k], v)
    c = transport(torch.rand(2, 1, 128, 128))
    assert c.shape == (2, 16, 16, 16)                        # condition on VAE grid


def test_edge_codec_runner_trains_and_checkpoints():
    cfg = OmegaConf.create({"train": {"edge_codec": {
        "base_ch": 16, "use_channel": False, "bce_weight": 1.0, "dice_weight": 1.0}}})
    codec = build_edge_codec(cfg, torch.device("cpu"))
    assert codec.decoder is not None
    pg = [{"params": list(codec.parameters()), "name": "edge_jscc"}]
    runner = EdgeCodecStageRunner(codec, cfg, torch.device("cpu"), pg)
    batch = {"edge": (torch.rand(2, 1, 64, 64) > 0.5).float()}
    out = runner.forward(batch)
    assert "loss_bce" in out and "loss_dice" in out
    metrics = runner.training_step(batch)
    assert runner.last_step_did_update is True
    assert "edge_jscc" in runner.state_modules()
    assert "edge_jscc" in runner.get_train_state()["modules"]


def test_edge_codec_checkpoint_loads_into_stage3_transport(tmp_path):
    # Train-side codec (with decoder) → save a runner-style checkpoint.
    cfg_codec = OmegaConf.create({"train": {"edge_codec": {
        "base_ch": 16, "use_channel": False}}})
    codec = build_edge_codec(cfg_codec, torch.device("cpu"))
    pg = [{"params": list(codec.parameters()), "name": "edge_jscc"}]
    runner = EdgeCodecStageRunner(codec, cfg_codec, torch.device("cpu"), pg)
    ckpt = tmp_path / "best.pth"
    torch.save({"epoch": 1, "global_step": 1, "stage": "edge_codec",
                **{"runner_state": runner.get_train_state()}}, ckpt)

    # Stage-3 transport (no decoder) loads encoder/projector weights strict=False.
    jscc = _FakeJSCC()
    cfg_s3 = OmegaConf.create({"train": {"controlnet": {
        "edge_transport": "edge_jscc",
        "edge_jscc": {"base_ch": 16, "use_channel": False,
                      "checkpoint": str(ckpt)}}}})
    transport = build_edge_transport(cfg_s3, jscc, torch.device("cpu"))
    # encoder weights match the trained codec (decoder keys ignored).
    tcodec = transport.module
    for k, v in codec.encoder.state_dict().items():
        assert torch.allclose(tcodec.encoder.state_dict()[k], v)
    # produces a condition latent on the VAE grid
    c = transport(torch.rand(2, 1, 128, 128))
    assert c.shape == (2, 16, 16, 16)


def test_csi_estimation_stage_validation_and_dataset_type():
    # image-only; no caption/edge required
    cfg = OmegaConf.create({"train_input_path": "x",
                            "train": {"stage": "csi_estimation"}})
    assert validate_stage_config(cfg) == STAGE_CSI_ESTIMATION
    assert resolve_dataset_type(_cfg(stage="csi_estimation"), STAGE_CSI_ESTIMATION) == "image"


def test_csi_estimation_runner_trains_and_checkpoints():
    cfg = OmegaConf.create({"train": {"stage": "csi_estimation"}})
    from sgdjscc_lab.models.csi_estimation import SNREstimator
    snr_est = SNREstimator(latent_ch=16)
    enc = lambda imgs: torch.randn(imgs.shape[0], 16, 16, 16)   # stub VAE latents
    pg = [{"params": list(snr_est.parameters()), "name": "snr_estimator"}]
    runner = CSIEstimationStageRunner(enc, snr_est, cfg, torch.device("cpu"), pg)
    batch = {"image": torch.rand(4, 3, 128, 128)}
    out = runner.forward(batch)
    assert "loss_snr" in out and out["loss"].requires_grad
    metrics = runner.training_step(batch)
    assert runner.last_step_did_update is True
    assert "snr_estimator" in runner.state_modules()
    assert "snr_estimator" in runner.get_train_state()["modules"]


def test_csi_estimation_freeze_policy_freezes_bundle():
    denoiser = _StubDenoiser()
    models = _models_with_denoiser(denoiser)
    cfg = OmegaConf.create({"train": {"stage": "csi_estimation"}})
    groups, report = apply_stage_freeze_policy(models, cfg, STAGE_CSI_ESTIMATION)
    assert all(not p.requires_grad for p in denoiser.parameters())
    assert groups == []                       # SNR estimator group added by the runner
    assert "diffusion" in report["frozen"] and "jscc_model" in report["frozen"]


def test_cfg_label_dropout_helper():
    from sgdjscc_lab.training.stage_runners import apply_cfg_label_dropout
    labels = torch.ones(8, 4)
    # prob=0 → unchanged; not-training → unchanged
    assert torch.equal(apply_cfg_label_dropout(labels, 0.0, True), labels)
    assert torch.equal(apply_cfg_label_dropout(labels, 1.0, False), labels)
    # prob=1 + training → all dropped to null (zeros)
    out = apply_cfg_label_dropout(labels, 1.0, True)
    assert torch.count_nonzero(out) == 0


class _RecordingDenoiser(nn.Module):
    """Captures the label tensor it last received (to inspect CFG dropout)."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.last_labels = None

    def forward(self, ft, noise_level, labels, c=None, enable_mask=False):
        self.last_labels = labels
        return ft * self.scale


def _cfg_dropout_runner(prob, with_edge=False):
    denoiser = _RecordingDenoiser()
    cfg = OmegaConf.create({"train": {"dm": {
        "use_masked_branch": False, "cfg_dropout_prob": prob}}})
    pg = [{"params": list(denoiser.parameters()), "name": "diffusion"}]
    enc_latent = lambda x: torch.randn(x.shape[0], 4, 8, 8)
    enc_text = lambda caps: torch.ones(len(caps), 8)      # non-zero labels
    if with_edge:
        enc_edge = lambda e: torch.randn(e.shape[0], 4, 8, 8)
        r = ControlNetStageRunner(denoiser, enc_latent, enc_text, enc_edge,
                                  cfg, torch.device("cpu"), pg)
    else:
        r = TextDMStageRunner(denoiser, enc_latent, enc_text,
                              cfg, torch.device("cpu"), pg)
    return r, denoiser


def test_cfg_dropout_applied_in_training_not_validation():
    runner, denoiser = _cfg_dropout_runner(prob=1.0)
    batch = {"image": torch.rand(4, 3, 16, 16), "caption": ["a", "b", "c", "d"]}
    runner.training_step(batch)                       # set_mode(True) → dropout fires
    assert torch.count_nonzero(denoiser.last_labels) == 0   # all dropped (p=1)
    runner.validation_step(batch)                     # set_mode(False) → no dropout
    assert torch.count_nonzero(denoiser.last_labels) > 0    # labels preserved


def test_cfg_dropout_controlnet_runner_path():
    runner, denoiser = _cfg_dropout_runner(prob=1.0, with_edge=True)
    batch = {"image": torch.rand(2, 3, 16, 16), "edge": torch.rand(2, 1, 16, 16),
             "caption": ["a", "b"]}
    runner.training_step(batch)
    assert torch.count_nonzero(denoiser.last_labels) == 0
    # default (no cfg_dropout_prob) → no dropout
    r2, d2 = _cfg_dropout_runner(prob=0.0, with_edge=True)
    r2.training_step(batch)
    assert torch.count_nonzero(d2.last_labels) > 0


def test_edge_jscc_transport_missing_checkpoint_fails_fast(tmp_path):
    jscc = _FakeJSCC()
    cfg = OmegaConf.create({"train": {"controlnet": {
        "edge_transport": "edge_jscc",
        "edge_jscc": {"base_ch": 16, "use_channel": False,
                      "checkpoint": str(tmp_path / "does_not_exist.pth")}}}})
    with pytest.raises(FileNotFoundError, match="edge_codec"):
        build_edge_transport(cfg, jscc, torch.device("cpu"))


def test_edge_sidecar_supports_stem_edge_naming_in_edge_dir(tmp_path):
    # images in one dir, "<stem>_edge.png" maps in a SEPARATE edge_dir
    img_dir = _make_images(tmp_path / "imgs", n=2, size=32)
    edge_dir = tmp_path / "edges"
    edge_dir.mkdir()
    for f in sorted(img_dir.glob("*.png")):
        save_tensor_as_image(torch.rand(3, 32, 32), edge_dir / f"{f.stem}_edge.png")
    cfg = _cfg(stage="controlnet",
               dataset={"type": "auto", "caption_source": "filename",
                        "edge_source": "sidecar", "edge_dir": str(edge_dir)})
    ds = build_dataset_for_stage(str(img_dir), cfg, stage=STAGE_CONTROLNET)
    item = ds[0]                       # must resolve <stem>_edge.png inside edge_dir
    assert item["edge"].shape == (1, 32, 32)


def test_edge_codec_freeze_policy_freezes_bundle():
    denoiser = _StubDenoiser()
    models = _models_with_denoiser(denoiser)
    cfg = OmegaConf.create({"train": {"edge_codec": {}}})
    groups, report = apply_stage_freeze_policy(models, cfg, STAGE_EDGE_CODEC)
    # bundle modules all frozen; the codec param group is added by the runner.
    assert all(not p.requires_grad for p in denoiser.parameters())
    assert groups == []
    assert "diffusion" in report["frozen"] and "jscc_model" in report["frozen"]


def test_edge_codec_end_to_end_run_training(tmp_path):
    from sgdjscc_lab.pipelines.train_pipeline import run_training
    img_dir = _make_images(tmp_path / "imgs", n=4, size=64)
    ckpt_dir = tmp_path / "ckpt"
    cfg = OmegaConf.create({
        "train_input_path": str(img_dir),
        "checkpoint_dir": str(ckpt_dir),
        "train_log_path": str(tmp_path / "log.jsonl"),
        "train": {
            "stage": "edge_codec",
            "dataset": {"type": "edge", "edge_source": "canny"},
            "transforms": {"resize_to": 64, "crop_mode": "center"},
            "edge_codec": {"base_ch": 16, "use_channel": False},
            "batch_size": 2, "max_steps": 2, "save_every_steps": 2, "lr": 1e-3,
        },
    })
    # models=None: edge_codec is self-contained and must STILL train (not dry-run).
    run_training(cfg, models=None, device=torch.device("cpu"))
    state = torch.load(ckpt_dir / "latest.pth")
    assert state["global_step"] == 2
    assert "edge_jscc" in state["runner_state"]["modules"]


def test_e2e_explicit_text_image_with_controlnet_rejected():
    cfg = OmegaConf.create({"train_input_path": "x", "train": {
        "stage": "end_to_end_ft",
        "dataset": {"type": "text_image", "caption_source": "filename",
                    "edge_source": "canny"},
        "end_to_end_ft": {"train_controlnet": True}}})
    with pytest.raises(StageConfigError, match="train_controlnet"):
        validate_stage_config(cfg)
    # auto type is promoted to text_image_edge and passes
    cfg.train.dataset.type = "auto"
    assert validate_stage_config(cfg) == STAGE_END_TO_END_FT

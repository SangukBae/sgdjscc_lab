"""tests/test_channel_conditioning.py – Phase 5-A channel-conditioning tests (offline)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.channels import RayleighChannel, PacketDropChannel  # noqa: E402


def _bundle(snr=5.0):
    return RayleighChannel(csi="perfect").observe(torch.randn(2, 16, 16, 16), snr)


# ─────────────────────────────────────────────────────────────────────────────
# ChannelConditionEncoder
# ─────────────────────────────────────────────────────────────────────────────

class TestChannelConditionEncoder:
    def test_stats_mode_shapes(self):
        from sgdjscc_lab.models.channel_condition_encoder import ChannelConditionEncoder, NUM_SCALARS
        enc = ChannelConditionEncoder(token_grid=4, token_dim=8, mode="stats")
        cond = enc.encode(_bundle())
        assert cond["scalars"].shape == (2, NUM_SCALARS)
        assert cond["tokens"].shape == (2, 16, 8)
        assert cond["reliability_map"].shape == (2, 1, 4, 4)

    def test_linear_mode_shapes(self):
        from sgdjscc_lab.models.channel_condition_encoder import ChannelConditionEncoder
        enc = ChannelConditionEncoder(token_grid=2, token_dim=6, mode="linear")
        cond = enc.encode(_bundle())
        assert cond["tokens"].shape == (2, 4, 6)

    def test_packet_drop_bundle_encodes(self):
        from sgdjscc_lab.models.channel_condition_encoder import ChannelConditionEncoder
        b = PacketDropChannel(drop_prob=0.5, packet_length=64).observe(torch.randn(1, 16, 16, 16), 10.0)
        cond = ChannelConditionEncoder().encode(b)
        assert cond["tokens"].shape[0] == 1


# ─────────────────────────────────────────────────────────────────────────────
# ReliabilityHead
# ─────────────────────────────────────────────────────────────────────────────

class TestReliabilityHead:
    def test_confidence_in_range(self):
        from sgdjscc_lab.models.reliability_head import ReliabilityHead
        out = ReliabilityHead().predict(_bundle(snr=5.0))
        assert out["confidence"].shape == (2,)
        assert (0.0 <= out["confidence"]).all() and (out["confidence"] <= 1.0).all()

    def test_higher_snr_higher_confidence(self):
        from sgdjscc_lab.models.reliability_head import ReliabilityHead
        head = ReliabilityHead()
        lo = head.predict(_bundle(snr=-10.0))["confidence"].mean()
        hi = head.predict(_bundle(snr=20.0))["confidence"].mean()
        assert hi >= lo


# ─────────────────────────────────────────────────────────────────────────────
# ChannelConditionPolicy
# ─────────────────────────────────────────────────────────────────────────────

class TestChannelConditionPolicy:
    def test_blind_csi_selects_blind_mode(self):
        from sgdjscc_lab.controllers.channel_condition_policy import ChannelConditionPolicy
        d = ChannelConditionPolicy().decide(csi="none", confidence=0.9)
        assert d.mode == "blind_conditioned"
        assert d.blind_snr is True

    def test_known_high_conf_latent_mode(self):
        from sgdjscc_lab.controllers.channel_condition_policy import ChannelConditionPolicy
        d = ChannelConditionPolicy().decide(csi="perfect", confidence=0.9)
        assert d.mode == "latent_conditioned"

    def test_known_low_conf_joint_mode(self):
        from sgdjscc_lab.controllers.channel_condition_policy import ChannelConditionPolicy
        d = ChannelConditionPolicy().decide(csi="perfect", confidence=0.1)
        assert d.mode == "joint_conditioned"

    def test_forced_mode_overrides(self):
        from sgdjscc_lab.controllers.channel_condition_policy import ChannelConditionPolicy
        d = ChannelConditionPolicy().decide(csi="none", confidence=0.9,
                                            forced_mode="latent_conditioned")
        assert d.mode == "latent_conditioned"


# ─────────────────────────────────────────────────────────────────────────────
# ChannelConditionedDiffusion wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TestChannelConditionedDiffusion:
    def _cfg(self):
        return OmegaConf.create({
            "guidance_scale": 4.0, "controlnet_scale": 0.3, "diffusion_step": 50,
            "use_jscc_feature": True, "use_gt_csi": True, "use_text": True,
        })

    def test_does_not_mutate_base_cfg(self):
        from sgdjscc_lab.models.diffusion_wrapper_channel import ChannelConditionedDiffusion
        cfg = self._cfg()
        ChannelConditionedDiffusion().build_conditioned_cfg(cfg, _bundle(), csi="none")
        assert cfg.guidance_scale == 4.0   # original untouched

    def test_blind_disables_gt_csi_and_adds_tokens(self):
        from sgdjscc_lab.models.diffusion_wrapper_channel import ChannelConditionedDiffusion
        out, info = ChannelConditionedDiffusion().build_conditioned_cfg(
            self._cfg(), _bundle(), mode="blind_conditioned", csi="none")
        assert out.use_gt_csi is False
        assert info["mode"] == "blind_conditioned"
        assert "channel_condition_tokens" in out

    def test_preserves_prompt_override(self):
        from sgdjscc_lab.models.diffusion_wrapper_channel import ChannelConditionedDiffusion
        out, _ = ChannelConditionedDiffusion().build_conditioned_cfg(
            self._cfg(), _bundle(), base_prompt="a red car")
        assert out.prompt_override == "a red car"


# ─────────────────────────────────────────────────────────────────────────────
# ChannelConditionedInference coexists with Phase 4 (injected fns)
# ─────────────────────────────────────────────────────────────────────────────

class TestChannelConditionedInference:
    def test_run_with_injected_fns(self):
        from sgdjscc_lab.pipelines.channel_conditioned_infer import ChannelConditionedInference

        captured = {}

        def reconstruct_fn(frame, cfg):
            captured["guidance_scale"] = cfg.guidance_scale
            captured["use_gt_csi"] = cfg.get("use_gt_csi")
            return frame.clone()

        def measure_fn(frame, snr_db):
            return RayleighChannel(csi="none").observe(frame, snr_db or 0.0)

        cfg = OmegaConf.create({
            "guidance_scale": 4.0, "controlnet_scale": 0.3, "diffusion_step": 50,
            "use_jscc_feature": True, "use_gt_csi": True, "use_text": True,
        })
        cci = ChannelConditionedInference(
            reconstruct_fn=reconstruct_fn, measure_fn=measure_fn,
            base_cfg=cfg, csi="none", condition_mode="auto",
        )
        out = cci.run(torch.randn(1, 16, 16, 16), snr_db=-5.0)
        assert out["reconstruction"].shape == (1, 16, 16, 16)
        assert out["info"]["mode"] == "blind_conditioned"
        # The conditioned cfg actually reached reconstruction (blind → gt_csi off).
        assert captured["use_gt_csi"] is False
        assert "measurement" in out["info"]


# ─────────────────────────────────────────────────────────────────────────────
# build_channel_conditioned_inference – import / config-wiring regression
# (catches the missing OmegaConf import that broke the real builder path)
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildChannelConditionedInference:
    class _FakeJSCC:
        def __init__(self):
            self.channel_model = None
            self.snr = 10.0

    class _FakeModels:
        def __init__(self):
            self.jscc_model = TestBuildChannelConditionedInference._FakeJSCC()
            self.device = torch.device("cpu")

    def test_builder_runs_and_wires_config(self):
        from sgdjscc_lab.pipelines.channel_conditioned_infer import (
            build_channel_conditioned_inference, OnePassChannelConditionedInference,
        )
        cfg = OmegaConf.create({
            "channel": "rayleigh", "csi": "none", "snr_db": 0, "condition_mode": "auto",
            "channel_condition": {
                "token_grid": 3, "token_dim": 5,
                "encoder_mode": "stats", "confidence_threshold": 0.7,
            },
        })
        models = self._FakeModels()
        cci = build_channel_conditioned_inference(models, cfg)
        # One-pass builder (no separate measurement forward).
        assert isinstance(cci, OnePassChannelConditionedInference)
        # channel_condition block is consumed (no dead config).
        assert cci.wrapper._encoder.token_grid == 3
        assert cci.wrapper._encoder.token_dim == 5
        assert cci.wrapper._policy.confidence_threshold == 0.7
        # Chosen channel is installed on the JSCC model for the forward pass.
        assert models.jscc_model.channel_model is not None

    def test_builder_defaults_when_no_block(self):
        from sgdjscc_lab.pipelines.channel_conditioned_infer import build_channel_conditioned_inference
        cfg = OmegaConf.create({"channel": "awgn", "snr_db": 10})
        cci = build_channel_conditioned_inference(self._FakeModels(), cfg)
        assert cci.wrapper._encoder.token_grid == 4    # default


# ─────────────────────────────────────────────────────────────────────────────
# One-pass conditioning: channel sampled once per patch (no extra measure forward)
# and the measurement source == the decoder's diffusion-init source.
# ─────────────────────────────────────────────────────────────────────────────

class TestOnePassConditioning:
    def test_encode_and_transmit_source_sharing(self):
        """The measurement bundle's received feature is the SAME tensor the decoder
        uses as the diffusion init (non-semantic path exercises the shared field)."""
        from sgdjscc_lab.pipelines.infer_pipeline import _encode_and_transmit

        class _LD:
            def __init__(self, m): self.mean = m; self.std = torch.ones_like(m)

        class _Enc:
            def __init__(self, m): self.latent_dist = _LD(m)

        class _VAE:
            def encode(self, x): return _Enc(torch.ones(1, 16, 16, 16))

        class _Jscc:
            def __init__(self): self.vae = _VAE(); self.snr = 10.0; self.channel_model = None; self.calls = 0
            def normalize(self, x): return x
            def channel(self, x): self.calls += 1; return x * 0.9

        cfg = OmegaConf.create({
            "use_semantic": False, "use_text": False, "mask_method": "none",
            "step_style": "continuous", "use_jscc_feature": False, "use_gt_csi": False,
        })
        j = _Jscc()
        art = _encode_and_transmit(torch.rand(1, 3, 128, 128), j, None, None, None,
                                   cfg, torch.device("cpu"), build_bundle=True)
        assert j.calls == 1                                        # channel sampled once
        assert art.bundle.encode_features_hat is art.encode_features_hat

    def test_no_extra_measurement_forward(self, monkeypatch):
        """run_image_channel_conditioned must encode+transmit and decode each patch
        EXACTLY once (regression: the old path ran a throwaway measurement forward,
        doubling the encode+channel work)."""
        import sgdjscc_lab.pipelines.infer_pipeline as ip
        from sgdjscc_lab.pipelines.infer_pipeline import ForwardArtifacts, run_image_channel_conditioned
        from sgdjscc_lab.channels import RayleighChannel
        from sgdjscc_lab.models.diffusion_wrapper_channel import ChannelConditionedDiffusion

        counters = {"encode": 0, "decode": 0}
        ray = RayleighChannel(csi="perfect")

        def _fake_encode(x, jscc, pipe, cd, cu, cfg, device, measurement_out=None, build_bundle=False):
            counters["encode"] += 1
            bundle = ray.observe(torch.randn(1, 16, 16, 16), 5.0) if build_bundle else None
            return ForwardArtifacts(
                use_semantic=True, encode_features_hat=torch.zeros(1, 16, 16, 16),
                signal_scale=torch.ones(1, 1, 1, 1), device=device, batch_size=1,
                bundle=bundle,
            )

        def _fake_decode(art, jscc, pipe, gt_text, cfg, device, original_image=None):
            counters["decode"] += 1
            return torch.zeros(1, 3, 128, 128)

        monkeypatch.setattr(ip, "_encode_and_transmit", _fake_encode)
        monkeypatch.setattr(ip, "_decode_diffusion", _fake_decode)

        class _Models:
            def __init__(self):
                self.jscc_model = type("J", (), {"snr": 10.0, "channel_model": None})()
                self.sem_pipeline = None
                self.device = torch.device("cpu")
                self.text_extractor = None
                self.edge_extractor = None

        base_cfg = OmegaConf.create({
            "use_semantic": False, "use_text": False, "guidance_scale": 4.0,
            "controlnet_scale": 0.3, "diffusion_step": 50, "use_jscc_feature": True,
            "use_gt_csi": True,
        })
        wrapper = ChannelConditionedDiffusion()
        patches = torch.randn(4, 3, 128, 128)
        out, info = run_image_channel_conditioned(patches, _Models(), base_cfg, wrapper)
        assert out.shape == (4, 3, 128, 128)
        # 4 patches → 4 encode+transmit and 4 decode (NOT 8 / NOT a second pass).
        assert counters["encode"] == 4
        assert counters["decode"] == 4
        assert "mode" in info
        # The actually-applied conditioned cfg is exposed (not the base cfg).
        assert "resolved_cfg" in info
        assert hasattr(info["resolved_cfg"], "guidance_scale")


# ─────────────────────────────────────────────────────────────────────────────
# Finding-1 regression: the condition source (bundle.best_estimate) must be the
# decoder's diffusion-init latent, not the raw pre-mask channel output.
# ─────────────────────────────────────────────────────────────────────────────

class TestEvidenceBundleSource:
    def test_best_estimate_is_decoder_init(self):
        from sgdjscc_lab.pipelines.infer_pipeline import _build_evidence_bundle
        from sgdjscc_lab.channels import RayleighChannel

        ch = RayleighChannel(csi="perfect")
        ch.transmit(torch.randn(1, 16, 16, 16), 5.0)   # populates last_bundle

        class _J:
            def __init__(self, c): self.channel_model = c; self.snr = 10.0

        ef = torch.randn(1, 16, 16, 16)          # post-mask received feature
        di = torch.randn(1, 16, 16, 16)          # decoder diffusion-init latent
        b = _build_evidence_bundle(_J(ch), {
            "encode_features_hat": ef, "decoder_init": di,
            "mask_token": None, "power_scalar": None,
        })
        # The encoder reads best_estimate → must be the decoder init, not raw channel out.
        assert b.best_estimate is di
        assert b.encode_features_hat is ef
        # Channel-level descriptors come from the realisation…
        assert b.reliability is ch.last_bundle.reliability
        # …without mutating the shared last_bundle (its received stays the raw output).
        assert ch.last_bundle.received is not di
        assert ch.last_bundle.equalized is not di

    def test_aggregate_preserves_decoder_init_source(self):
        from sgdjscc_lab.pipelines.infer_pipeline import _build_evidence_bundle
        from sgdjscc_lab.channels import RayleighChannel
        from sgdjscc_lab.channels.measurement import aggregate_bundles

        class _J:
            def __init__(self, c): self.channel_model = c; self.snr = 10.0

        bundles = []
        for _ in range(3):
            ch = RayleighChannel(csi="perfect")
            ch.transmit(torch.randn(1, 16, 16, 16), 5.0)
            di = torch.randn(1, 16, 16, 16)
            bundles.append(_build_evidence_bundle(_J(ch), {
                "encode_features_hat": di, "decoder_init": di}))
        agg = aggregate_bundles(bundles)
        # Aggregated best_estimate is the concatenation of the per-patch decoder inits.
        assert agg.best_estimate.shape[0] == 3

    def test_disabled_returns_none(self):
        from sgdjscc_lab.pipelines.channel_conditioned_infer import maybe_channel_conditioned_reconstruct
        cfg = OmegaConf.create({"use_channel_conditioning": False})
        recon, info = maybe_channel_conditioned_reconstruct(torch.rand(1, 3, 8, 8), None, cfg)
        assert recon is None and info is None

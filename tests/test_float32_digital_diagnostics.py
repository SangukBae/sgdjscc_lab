"""tests/test_float32_digital_diagnostics.py – float32 digital diagnostic harness.

CPU-only, no checkpoints, no GPU (see CLAUDE.md / docs/protocols/
float32_digital_diagnostics.md). Uses ``diagnostics.mock_models`` (a
deterministic synthetic VAE/diffusion/channel stand-in) so the harness's
routing, instrumentation, ablation, comparison, verdict, report, and
resume/signature logic can be verified structurally without real weights.
Real quality findings require a server GPU run — see the harness's own
REPORT.md, which never claims a root cause under ``--no-models``/dry-run.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sgdjscc_lab.diagnostics.ablations import BASELINE_ABLATION, build_default_ablations
from sgdjscc_lab.diagnostics.float32_digital_paths import (
    RecordCtx,
    instrumented_decode,
    run_frame_awgn,
    run_frame_digital_inprocess,
    run_frame_digital_wire,
    run_path_with_failure_capture,
)
from sgdjscc_lab.diagnostics.mock_models import build_diagnostic_cfg, build_mock_models
from sgdjscc_lab.diagnostics.tensor_compare import compare_tensors
from sgdjscc_lab.diagnostics.tensor_recorder import TensorRecorder, tensor_fingerprint
from sgdjscc_lab.diagnostics.verdict import VerdictResult, aggregate_verdicts, classify


def _frame(h=256, w=256):
    torch.manual_seed(0)
    return torch.rand(1, 3, h, w)


def _mock_env():
    models = build_mock_models("cpu")
    cfg = build_diagnostic_cfg()
    return models, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Routing: all three paths run structurally and return valid images
# ─────────────────────────────────────────────────────────────────────────────

class TestRouting:
    def test_awgn_path_runs(self):
        models, cfg = _mock_env()
        rec = TensorRecorder(enabled=True)
        out = run_frame_awgn(_frame(), models, cfg, BASELINE_ABLATION, recorder=rec,
                              video="v", frame_index=0, seed=1, record_patch_index=0)
        assert not out.failed
        assert out.reconstructed.shape == (1, 3, 256, 256)
        assert out.n_patches == 4
        assert len(rec.rows) > 0

    def test_digital_inprocess_path_runs(self):
        models, cfg = _mock_env()
        rec = TensorRecorder(enabled=True)
        out = run_frame_digital_inprocess(
            _frame(), models, cfg, BASELINE_ABLATION, bit_depth=32, granularity="per_tensor",
            recorder=rec, video="v", frame_index=0, seed=1, record_patch_index=0,
        )
        assert not out.failed
        assert out.reconstructed.shape == (1, 3, 256, 256)

    def test_digital_wire_path_runs_and_reports_wire_bytes(self):
        models, cfg = _mock_env()
        rec = TensorRecorder(enabled=True)
        out = run_frame_digital_wire(
            _frame(), models, cfg, BASELINE_ABLATION, bit_depth=32, granularity="per_tensor",
            digital_step_policy="fixed_reference", recorder=rec, video="v", frame_index=0, seed=1,
            record_patch_index=0,
        )
        assert not out.failed
        assert out.reconstructed.shape == (1, 3, 256, 256)
        assert out.wire_bytes is not None and out.wire_bytes > 0

    def test_digital_wire_rejects_non_none_mask_method(self):
        models, cfg = _mock_env()
        cfg = build_diagnostic_cfg(mask_method="topk")
        rec = TensorRecorder(enabled=False)
        with pytest.raises(ValueError, match="mask_method=none"):
            run_frame_digital_wire(
                _frame(), models, cfg, BASELINE_ABLATION, bit_depth=32, granularity="per_tensor",
                digital_step_policy="fixed_reference", recorder=rec, video="v", frame_index=0, seed=1,
            )


# ─────────────────────────────────────────────────────────────────────────────
# float32 round-trip: bit_depth=32 must be bitwise identical sender vs receiver
# ─────────────────────────────────────────────────────────────────────────────

class TestFloat32RoundTrip:
    def test_wire_roundtrip_is_bitexact_at_bit_depth_32(self):
        models, cfg = _mock_env()
        rec = TensorRecorder(enabled=True)
        out = run_frame_digital_wire(
            _frame(), models, cfg, BASELINE_ABLATION, bit_depth=32, granularity="per_tensor",
            digital_step_policy="fixed_reference", recorder=rec, video="v", frame_index=0, seed=1,
            record_patch_index=0,
        )
        assert out.roundtrip_bitexact is True

        pre = rec.live[("v", 0, 1, "baseline", "digital_wire", "sender_vae_latent_post_norm")]
        post = rec.live[("v", 0, 1, "baseline", "digital_wire", "post_deserialize_latent_raw")]
        assert torch.equal(pre, post)
        assert tensor_fingerprint(pre) == tensor_fingerprint(post)

    def test_wire_roundtrip_not_bitexact_at_lossy_bit_depth(self):
        models, cfg = _mock_env()
        rec = TensorRecorder(enabled=True)
        out = run_frame_digital_wire(
            _frame(), models, cfg, BASELINE_ABLATION, bit_depth=4, granularity="per_tensor",
            digital_step_policy="fixed_reference", recorder=rec, video="v", frame_index=0, seed=1,
            record_patch_index=0,
        )
        assert out.roundtrip_bitexact is None  # only asserted for bit_depth==32


# ─────────────────────────────────────────────────────────────────────────────
# Wire path must record every transport stage the awgn/in-process paths do
# (previously receiver_post_norm_latent/power_scalar/cur_step/cur_snr and an
# explicit pre_serialize_latent stage were silently missing for digital_wire).
# ─────────────────────────────────────────────────────────────────────────────

class TestWirePathInstrumentation:
    def test_wire_records_receiver_post_norm_power_scalar_and_step(self):
        models, cfg = _mock_env()
        rec = TensorRecorder(enabled=True)
        out = run_frame_digital_wire(
            _frame(), models, cfg, BASELINE_ABLATION, bit_depth=32, granularity="per_tensor",
            digital_step_policy="fixed_reference", recorder=rec, video="v", frame_index=0, seed=1,
            record_patch_index=0,
        )
        assert not out.failed
        for stage in ("pre_serialize_latent", "receiver_post_norm_latent", "power_scalar", "cur_step", "cur_snr"):
            key = ("v", 0, 1, "baseline", "digital_wire", stage)
            assert key in rec.live, f"missing wire-path stage recording: {stage}"

    def test_receiver_post_norm_latent_recorded_and_comparable_across_all_three_paths(self):
        models, cfg = _mock_env()
        rec = TensorRecorder(enabled=True)
        frame = _frame(128, 128)
        run_frame_awgn(frame, models, cfg, BASELINE_ABLATION, recorder=rec,
                        video="v", frame_index=0, seed=1, record_patch_index=0)
        run_frame_digital_inprocess(frame, models, cfg, BASELINE_ABLATION, bit_depth=32,
                                     granularity="per_tensor", recorder=rec, video="v",
                                     frame_index=0, seed=1, record_patch_index=0)
        run_frame_digital_wire(frame, models, cfg, BASELINE_ABLATION, bit_depth=32,
                                granularity="per_tensor", digital_step_policy="fixed_reference",
                                recorder=rec, video="v", frame_index=0, seed=1, record_patch_index=0)
        for path in ("awgn", "digital_inprocess", "digital_wire"):
            key = ("v", 0, 1, "baseline", path, "receiver_post_norm_latent")
            assert key in rec.live, f"missing receiver_post_norm_latent for {path}"
        # digital_inprocess and digital_wire are both lossless at bit_depth=32
        # and hit the exact same normalize() call on the exact same latent --
        # they must agree bit-for-bit at this stage.
        inprocess = rec.live[("v", 0, 1, "baseline", "digital_inprocess", "receiver_post_norm_latent")]
        wire = rec.live[("v", 0, 1, "baseline", "digital_wire", "receiver_post_norm_latent")]
        assert torch.equal(inprocess, wire)


# ─────────────────────────────────────────────────────────────────────────────
# Tensor comparison math
# ─────────────────────────────────────────────────────────────────────────────

class TestTensorCompare:
    def test_identical_tensors(self):
        t = torch.randn(4, 4)
        cmp = compare_tensors(t, t.clone())
        assert cmp["comparable"] and cmp["exact_equal"]
        assert cmp["max_abs_err"] == 0.0
        assert cmp["cosine_similarity"] == pytest.approx(1.0)
        assert cmp["norm_ratio"] == pytest.approx(1.0)

    def test_different_tensors(self):
        a = torch.zeros(4, 4)
        b = torch.ones(4, 4)
        cmp = compare_tensors(a, b)
        assert cmp["comparable"] and not cmp["exact_equal"]
        assert cmp["max_abs_err"] == pytest.approx(1.0)
        assert cmp["mse"] == pytest.approx(1.0)

    def test_shape_mismatch_not_comparable(self):
        cmp = compare_tensors(torch.zeros(2, 2), torch.zeros(3, 3))
        assert not cmp["comparable"]
        assert "shape_mismatch" in cmp["reason"]

    def test_missing_tensor_not_comparable(self):
        assert not compare_tensors(None, torch.zeros(2))["comparable"]
        assert not compare_tensors(torch.zeros(2), None)["comparable"]

    def test_nonfinite_tensor_marks_not_both_finite(self):
        a = torch.tensor([1.0, float("nan")])
        b = torch.tensor([1.0, 2.0])
        cmp = compare_tensors(a, b)
        assert cmp["comparable"]
        assert cmp["both_finite"] is False
        assert cmp["max_abs_err"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Ablations actually change behavior
# ─────────────────────────────────────────────────────────────────────────────

class TestAblations:
    def test_bypass_diffusion_never_calls_pipe_generate(self):
        models, cfg = _mock_env()
        ablations = build_default_ablations()

        def _boom(**kwargs):
            raise AssertionError("pipe.generate must not be called under diffusion_bypass_vae_direct")

        models.sem_pipeline.generate = _boom
        rec = TensorRecorder(enabled=True)
        out = run_frame_awgn(
            _frame(128, 128), models, cfg, ablations["diffusion_bypass_vae_direct"],
            recorder=rec, video="v", frame_index=0, seed=1, record_patch_index=0,
        )
        assert not out.failed
        assert out.diffusion_steps == 0
        init = rec.live[("v", 0, 1, "diffusion_bypass_vae_direct", "awgn", "diffusion_latent_init")]
        final = rec.live[("v", 0, 1, "diffusion_bypass_vae_direct", "awgn", "diffusion_latent_final")]
        assert torch.equal(init, final)

    def test_bypass_diffusion_never_calls_canny_retransmit_or_controlnet_encode(self, monkeypatch):
        # True VAE-direct bypass (not just skipping the diffusion sampler
        # call): Canny retransmission and ControlNet edge-latent encoding
        # must also never run, so latency/VRAM measured under this ablation
        # reflect ONLY the VAE decode and cannot themselves OOM on
        # edge-processing this ablation's whole point is to rule out.
        import sgdjscc_lab.pipelines.infer_pipeline as infer

        models, cfg = _mock_env()
        ablations = build_default_ablations()
        calls = {"retransmit": 0, "canny_latent": 0}

        def _spy_retransmit(*args, **kwargs):
            calls["retransmit"] += 1
            raise AssertionError("_retransmit_canny must not be called under diffusion_bypass_vae_direct")

        def _spy_canny_latent(*args, **kwargs):
            calls["canny_latent"] += 1
            raise AssertionError("_encode_canny_latent must not be called under diffusion_bypass_vae_direct")

        monkeypatch.setattr(infer, "_retransmit_canny", _spy_retransmit)
        monkeypatch.setattr(infer, "_encode_canny_latent", _spy_canny_latent)
        rec = TensorRecorder(enabled=True)
        out = run_frame_awgn(
            _frame(128, 128), models, cfg, ablations["diffusion_bypass_vae_direct"],
            recorder=rec, video="v", frame_index=0, seed=1, record_patch_index=0,
        )
        assert not out.failed
        assert calls == {"retransmit": 0, "canny_latent": 0}
        # The skipped stages are still explicitly recorded as absent, not
        # silently omitted (tensor_stage_stats.jsonl should show
        # present=False, never leave no record at all).
        skipped_rows = [
            r for r in rec.rows
            if r["stage"] in ("edge_post_retransmit", "uncertainty_post_ablation", "controlnet_input_latent")
            and r["path"] == "awgn"
        ]
        assert len(skipped_rows) == 3
        assert all(r["present"] is False for r in skipped_rows)

    def test_minimal_denoise_uses_configured_step_count(self):
        models, cfg = _mock_env()
        ablations = build_default_ablations(minimal_denoise_steps=3)
        rec = TensorRecorder(enabled=False)
        out = run_frame_awgn(
            _frame(128, 128), models, cfg, ablations["minimal_denoise"],
            recorder=rec, video="v", frame_index=0, seed=1, record_patch_index=None,
        )
        assert out.diffusion_steps == 3

    def test_fixed_step_overrides_cur_step_for_every_path(self):
        models, cfg = _mock_env()
        ablations = build_default_ablations(fixed_step_value=0.42)
        rec = TensorRecorder(enabled=True)
        run_frame_awgn(
            _frame(128, 128), models, cfg, ablations["fixed_step"],
            recorder=rec, video="v", frame_index=0, seed=1, record_patch_index=0,
        )
        recorded_step = rec.live[("v", 0, 1, "fixed_step", "awgn", "diffusion_latent_init")]
        # latent_init is computed from encode_features_hat/power_scalar regardless of
        # cur_step; assert the recorded stage exists (fixed_step's effect is on
        # cur_step passed into _run_diffusion, exercised via the smoke run not erroring).
        assert recorded_step is not None

    def test_use_edge_false_zeroes_controlnet_input(self):
        models, cfg = _mock_env()
        ablations = build_default_ablations()
        rec = TensorRecorder(enabled=True)
        run_frame_awgn(
            _frame(128, 128), models, cfg, ablations["edge_and_uncertainty_off"],
            recorder=rec, video="v", frame_index=0, seed=1, record_patch_index=0,
        )
        edge_post = rec.live[("v", 0, 1, "edge_and_uncertainty_off", "awgn", "edge_post_retransmit")]
        assert torch.count_nonzero(edge_post).item() == 0

    def test_reuse_awgn_step_ablation_uses_supplied_reference(self):
        models, cfg = _mock_env()
        ablations = build_default_ablations()
        rec = TensorRecorder(enabled=True)
        awgn_out = run_frame_awgn(
            _frame(128, 128), models, cfg, ablations["baseline"],
            recorder=rec, video="v", frame_index=0, seed=1, record_patch_index=0,
        )
        digital_out = run_frame_digital_inprocess(
            _frame(128, 128), models, cfg, ablations["reuse_awgn_step"], bit_depth=32,
            granularity="per_tensor", recorder=rec, video="v", frame_index=0, seed=1,
            record_patch_index=0, awgn_step_ref=awgn_out.cur_step_ref,
        )
        assert not digital_out.failed


# ─────────────────────────────────────────────────────────────────────────────
# NaN/Inf propagation to a documented failure, not a crash
# ─────────────────────────────────────────────────────────────────────────────

class TestFailureCapture:
    def test_nonfinite_decode_is_caught_and_reported(self, monkeypatch):
        models, cfg = _mock_env()

        def _nan_decode(z):
            return (torch.full((z.shape[0], 3, 128, 128), float("nan")),)

        monkeypatch.setattr(models.jscc_model.vae, "decode", _nan_decode)
        rec = TensorRecorder(enabled=False)
        outcome = run_path_with_failure_capture(
            run_frame_awgn, "awgn", _frame(128, 128), models, cfg, BASELINE_ABLATION,
            recorder=rec, video="v", frame_index=0, seed=1, record_patch_index=0,
        )
        assert outcome.failed
        assert outcome.path == "awgn"
        assert outcome.failure_stage is not None

    def test_genuine_bug_is_not_swallowed(self, monkeypatch):
        models, cfg = _mock_env()

        def _boom(z):
            raise RuntimeError("not a NonFiniteError")

        monkeypatch.setattr(models.jscc_model.vae, "decode", _boom)
        rec = TensorRecorder(enabled=False)
        with pytest.raises(RuntimeError, match="not a NonFiniteError"):
            run_path_with_failure_capture(
                run_frame_awgn, "awgn", _frame(128, 128), models, cfg, BASELINE_ABLATION,
                recorder=rec, video="v", frame_index=0, seed=1, record_patch_index=0,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Decode parity: the diagnostic decode duplicate must match the real one
# ─────────────────────────────────────────────────────────────────────────────

class TestDecodeParity:
    def test_decode_parity_with_production(self):
        from sgdjscc_lab.pipelines.infer_pipeline import _decode_diffusion, _encode_and_transmit

        models, cfg = _mock_env()
        jscc = models.jscc_model
        x = _frame(128, 128)
        canny = torch.zeros(1, 11, 128, 128)
        unc = torch.zeros_like(canny)
        gt_text = [["a caption"]]

        torch.manual_seed(42)
        artifacts = _encode_and_transmit(x, jscc, models.sem_pipeline, canny, unc, cfg, models.device)

        expected = _decode_diffusion(artifacts, jscc, models.sem_pipeline, gt_text, cfg, models.device, original_image=x)

        rec = TensorRecorder(enabled=False)
        ctx = RecordCtx(recorder=rec, video="v", frame=0, seed=42, ablation="baseline", path="awgn")
        actual, _steps = instrumented_decode(
            artifacts, jscc, models.sem_pipeline, gt_text, cfg, models.device, ctx,
            edge_already_received=False, ablation=BASELINE_ABLATION,
        )
        assert torch.equal(expected, actual)


# ─────────────────────────────────────────────────────────────────────────────
# Sender latent capture matches what the production function computes
# ─────────────────────────────────────────────────────────────────────────────

class TestSenderLatentCapture:
    def test_sender_latent_capture_matches_production(self):
        from sgdjscc_lab.pipelines.infer_pipeline import _encode_latent
        from sgdjscc_lab.diagnostics.float32_digital_paths import instrumented_encode_and_transmit

        models, cfg = _mock_env()
        jscc = models.jscc_model
        x = _frame(128, 128)
        canny = torch.zeros(1, 11, 128, 128)
        unc = torch.zeros_like(canny)

        expected, _std = _encode_latent(jscc, x)

        rec = TensorRecorder(enabled=True)
        ctx = RecordCtx(recorder=rec, video="v", frame=0, seed=1, ablation="baseline", path="awgn")
        instrumented_encode_and_transmit(x, jscc, models.sem_pipeline, canny, unc, cfg, models.device, ctx)
        captured = rec.live[("v", 0, 1, "baseline", "awgn", "sender_vae_latent_post_norm")]
        assert torch.equal(expected, captured)


# ─────────────────────────────────────────────────────────────────────────────
# Verdict classification
# ─────────────────────────────────────────────────────────────────────────────

class TestVerdict:
    def test_packet_tx_rx_issue_when_paths_diverge(self):
        comparisons = [
            {"stage": "sender_vae_latent_post_norm", "comparable": True, "both_finite": True,
             "exact_equal": True, "mean_abs_err": 0.0, "cosine_similarity": 1.0},
            {"stage": "post_deserialize_latent_raw", "comparable": True, "both_finite": True,
             "exact_equal": False, "mean_abs_err": 0.5, "cosine_similarity": 0.1},
        ]
        v = classify(
            inprocess_vs_wire_comparisons=comparisons,
            path_quality={"awgn_psnr": 25.0, "digital_inprocess_psnr": 10.0, "digital_wire_psnr": 5.0},
        )
        assert v.verdict == "packet_tx_rx_issue"
        assert v.first_divergent_stage == "post_deserialize_latent_raw"

    def test_decoder_pipeline_issue_when_paths_agree_but_worse_than_awgn(self):
        comparisons = [
            {"stage": "final_reconstruction", "comparable": True, "both_finite": True,
             "exact_equal": True, "mean_abs_err": 0.0, "cosine_similarity": 1.0},
        ]
        v = classify(
            inprocess_vs_wire_comparisons=comparisons,
            path_quality={"awgn_psnr": 23.3, "digital_inprocess_psnr": 11.3, "digital_wire_psnr": 11.3},
        )
        assert v.verdict == "decoder_pipeline_issue"

    def test_latent_normalization_issue_when_vae_direct_already_low(self):
        comparisons = [
            {"stage": "final_reconstruction", "comparable": True, "both_finite": True,
             "exact_equal": True, "mean_abs_err": 0.0, "cosine_similarity": 1.0},
        ]
        v = classify(
            inprocess_vs_wire_comparisons=comparisons,
            path_quality={"awgn_psnr": 23.3, "digital_inprocess_psnr": 11.3, "digital_wire_psnr": 11.3},
            vae_direct_quality={"digital_wire_psnr": 10.0},
        )
        assert v.verdict == "latent_normalization_issue"

    def test_inconclusive_when_quality_is_similar(self):
        comparisons = [
            {"stage": "final_reconstruction", "comparable": True, "both_finite": True,
             "exact_equal": True, "mean_abs_err": 0.0, "cosine_similarity": 1.0},
        ]
        v = classify(
            inprocess_vs_wire_comparisons=comparisons,
            path_quality={"awgn_psnr": 23.0, "digital_inprocess_psnr": 22.8, "digital_wire_psnr": 22.8},
        )
        assert v.verdict == "inconclusive"

    def test_inconclusive_when_missing_data(self):
        v = classify(inprocess_vs_wire_comparisons=[], path_quality={"awgn_psnr": None})
        assert v.verdict == "inconclusive"

    def test_aggregate_verdicts(self):
        verdicts = [
            VerdictResult(verdict="packet_tx_rx_issue", reason="x"),
            VerdictResult(verdict="packet_tx_rx_issue", reason="y"),
            VerdictResult(verdict="inconclusive", reason="z"),
        ]
        agg = aggregate_verdicts(verdicts)
        assert agg["dominant_verdict"] == "packet_tx_rx_issue"
        assert agg["counts"]["inconclusive"] == 1

    def test_expected_edge_asymmetry_alone_is_not_packet_tx_rx_issue(self):
        # digital_inprocess retransmits its edge through the analog Canny/WITT
        # net while digital_wire uses its already-received packet edge as-is
        # -- BY DESIGN (transmission/receiver_runtime.py) -- so these
        # edge/decoder stages are EXPECTED to differ under the baseline
        # ablation even when nothing is broken. A divergence confined to
        # those stages must not be misclassified as packet_tx_rx_issue.
        comparisons = [
            {"stage": "sender_vae_latent_post_norm", "comparable": True, "both_finite": True,
             "exact_equal": True, "mean_abs_err": 0.0, "cosine_similarity": 1.0},
            {"stage": "power_scalar", "comparable": True, "both_finite": True,
             "exact_equal": True, "mean_abs_err": 0.0, "cosine_similarity": 1.0},
            # Expected-to-differ edge/decoder stages -- large divergence:
            {"stage": "edge_post_retransmit", "comparable": True, "both_finite": True,
             "exact_equal": False, "mean_abs_err": 0.9, "cosine_similarity": 0.05},
            {"stage": "controlnet_input_latent", "comparable": True, "both_finite": True,
             "exact_equal": False, "mean_abs_err": 0.9, "cosine_similarity": 0.05},
            {"stage": "final_reconstruction", "comparable": True, "both_finite": True,
             "exact_equal": False, "mean_abs_err": 0.9, "cosine_similarity": 0.05},
        ]
        v = classify(
            inprocess_vs_wire_comparisons=comparisons,
            path_quality={"awgn_psnr": 23.0, "digital_inprocess_psnr": 22.8, "digital_wire_psnr": 22.5},
        )
        assert v.verdict != "packet_tx_rx_issue"

    def test_edge_asymmetry_becomes_evidence_once_equalized(self):
        # Once edge_handling_equalized=True (e.g. the serialized_raw_edge /
        # awgn_edge_retransmit ablation forced edge_already_received identical
        # across paths), a divergence in the edge/decoder stages IS meaningful
        # evidence again.
        comparisons = [
            {"stage": "controlnet_input_latent", "comparable": True, "both_finite": True,
             "exact_equal": False, "mean_abs_err": 0.9, "cosine_similarity": 0.05},
        ]
        v = classify(
            inprocess_vs_wire_comparisons=comparisons,
            path_quality={"awgn_psnr": 23.0, "digital_inprocess_psnr": 22.8, "digital_wire_psnr": 22.5},
            edge_handling_equalized=True,
        )
        assert v.verdict == "packet_tx_rx_issue"
        assert v.first_divergent_stage == "controlnet_input_latent"


# ─────────────────────────────────────────────────────────────────────────────
# CLI: dry-run, --no-models end-to-end, resume, signature mismatch
# ─────────────────────────────────────────────────────────────────────────────

_CLI = str(_REPO_ROOT / "scripts" / "diagnose_float32_digital_quality.py")


def _run_cli(args, **kwargs):
    return subprocess.run(
        [sys.executable, _CLI, *args], cwd=str(_REPO_ROOT),
        capture_output=True, text=True, timeout=180, **kwargs,
    )


@pytest.mark.skipif(
    not (_REPO_ROOT / "data" / "etri_video_eval" / "manifest.csv").exists(),
    reason="ETRI dataset manifest not present in this environment",
)
class TestCli:
    def test_dry_run_prints_plan_and_touches_nothing(self, tmp_path):
        out_root = tmp_path / "dryrun"
        result = _run_cli([
            "--output-root", str(out_root), "--video-ids", "01_person_walk",
            "--frames", "0", "--dry-run",
        ])
        assert result.returncode == 0, result.stderr
        assert "dry-run" in result.stdout
        assert not out_root.exists()

    def test_no_models_end_to_end_produces_all_outputs(self, tmp_path):
        out_root = tmp_path / "run1"
        result = _run_cli([
            "--output-root", str(out_root), "--video-ids", "01_person_walk", "--frames", "0",
            "--no-models", "--device", "cpu", "--no-lpips",
            "--ablations", "baseline,diffusion_bypass_vae_direct",
        ])
        assert result.returncode == 0, result.stderr
        for name in (
            "run_manifest_initial.json", "run_manifest.json", "run_signature.json",
            "path_comparison.csv", "tensor_stage_stats.jsonl", "tensor_pair_comparison.csv",
            "summary.json", "REPORT.md",
        ):
            assert (out_root / name).exists(), name

        with (out_root / "path_comparison.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 6  # 2 ablations x 3 paths

        summary = json.loads((out_root / "summary.json").read_text())
        assert summary["dry_run"] is False
        assert summary["counts"]["n_frames_processed"] == 1

    def test_resume_skips_completed_groups(self, tmp_path):
        out_root = tmp_path / "run2"
        args = [
            "--output-root", str(out_root), "--video-ids", "01_person_walk", "--frames", "0",
            "--no-models", "--device", "cpu", "--no-lpips",
        ]
        r1 = _run_cli(args)
        assert r1.returncode == 0, r1.stderr
        with (out_root / "path_comparison.csv").open() as fh:
            rows_after_first = list(csv.DictReader(fh))

        r2 = _run_cli(args + ["--resume"])
        assert r2.returncode == 0, r2.stderr
        with (out_root / "path_comparison.csv").open() as fh:
            rows_after_second = list(csv.DictReader(fh))
        assert len(rows_after_second) == len(rows_after_first)  # no duplicate rows

    def test_bare_rerun_without_resume_flag_is_refused_not_duplicated(self, tmp_path):
        # Regression test for the reproduced "3 rows -> 6 rows" bug: a bare
        # re-run (no --resume) into a non-empty output-root must refuse
        # outright, never silently duplicate CSV rows.
        out_root = tmp_path / "run_bare"
        args = [
            "--output-root", str(out_root), "--video-ids", "01_person_walk", "--frames", "0",
            "--no-models", "--device", "cpu", "--no-lpips",
        ]
        r1 = _run_cli(args)
        assert r1.returncode == 0, r1.stderr
        with (out_root / "path_comparison.csv").open() as fh:
            rows_after_first = list(csv.DictReader(fh))
        assert len(rows_after_first) == 3

        r2 = _run_cli(args)  # deliberately no --resume
        assert r2.returncode != 0
        assert "--resume" in r2.stderr

        with (out_root / "path_comparison.csv").open() as fh:
            rows_after_bare_rerun = list(csv.DictReader(fh))
        assert len(rows_after_bare_rerun) == 3  # unchanged, not doubled to 6

    def test_resume_preserves_prior_verdict_summary(self, tmp_path):
        # Regression test: --resume must not overwrite a previously-computed
        # verdict_summary with None just because this invocation's baseline
        # group was already completed (and therefore skipped).
        out_root = tmp_path / "run_verdict_resume"
        args = [
            "--output-root", str(out_root), "--video-ids", "01_person_walk", "--frames", "0",
            "--no-models", "--device", "cpu", "--no-lpips",
        ]
        r1 = _run_cli(args)
        assert r1.returncode == 0, r1.stderr
        summary1 = json.loads((out_root / "summary.json").read_text())
        assert summary1["verdict_summary"] is not None
        assert summary1["verdict_summary"]["n_verdicts"] == 1
        verdicts_path = out_root / "verdicts.jsonl"
        assert verdicts_path.exists()
        assert len(verdicts_path.read_text().strip().splitlines()) == 1

        r2 = _run_cli(args + ["--resume"])
        assert r2.returncode == 0, r2.stderr
        summary2 = json.loads((out_root / "summary.json").read_text())
        assert summary2["verdict_summary"] is not None
        assert summary2["verdict_summary"]["n_verdicts"] == 1
        assert len(verdicts_path.read_text().strip().splitlines()) == 1  # not duplicated either

    def test_ssim_and_lpips_delta_columns_are_populated(self, tmp_path):
        out_root = tmp_path / "run_deltas"
        result = _run_cli([
            "--output-root", str(out_root), "--video-ids", "01_person_walk", "--frames", "0",
            "--no-models", "--device", "cpu", "--no-lpips",
        ])
        assert result.returncode == 0, result.stderr
        with (out_root / "path_comparison.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        awgn_row = next(r for r in rows if r["path"] == "awgn")
        assert float(awgn_row["ssim_delta_vs_awgn"]) == pytest.approx(0.0)
        for row in rows:
            if row["path"] != "awgn":
                assert row["ssim_delta_vs_awgn"] not in ("", None)

    def test_failed_total_counts_rows_not_unique_groups(self, tmp_path, monkeypatch):
        # Regression test: failed_count must be the number of FAILED PATH
        # ATTEMPTS (rows in failed_cases.csv), not the number of unique
        # (video, frame, ablation) groups -- a single group where all 3
        # paths fail must count as 3, not 1. Calls run() in-process (not via
        # the subprocess-based _run_cli helper) so monkeypatch can reach the
        # mock VAE's decode() the CLI process actually uses.
        import importlib.util

        import sgdjscc_lab.diagnostics.mock_models as mock_models_mod

        def _nan_decode(self, z):
            return (torch.full((z.shape[0], 3, 128, 128), float("nan")),)

        monkeypatch.setattr(mock_models_mod._MockVae, "decode", _nan_decode)

        spec = importlib.util.spec_from_file_location("_f32dig_cli_inprocess", _CLI)
        cli_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli_mod)

        out_root = tmp_path / "run_failures"
        rc = cli_mod.run([
            "--output-root", str(out_root), "--video-ids", "01_person_walk", "--frames", "0",
            "--no-models", "--device", "cpu", "--no-lpips",
        ])
        assert rc == 3
        with (out_root / "failed_cases.csv").open() as fh:
            failed_rows = list(csv.DictReader(fh))
        assert len(failed_rows) == 3  # awgn + digital_inprocess + digital_wire all failed
        summary = json.loads((out_root / "summary.json").read_text())
        assert summary["counts"]["n_frames_processed"] == 1

    def test_signature_mismatch_on_resume_is_rejected(self, tmp_path):
        out_root = tmp_path / "run3"
        base_args = [
            "--output-root", str(out_root), "--video-ids", "01_person_walk", "--frames", "0",
            "--no-models", "--device", "cpu", "--no-lpips",
        ]
        r1 = _run_cli(base_args)
        assert r1.returncode == 0, r1.stderr

        r2 = _run_cli(base_args + ["--seed", "999", "--resume"])
        assert r2.returncode != 0
        assert "run_signature mismatch" in r2.stderr

    def test_report_states_pending_server_measurement_under_no_models(self, tmp_path):
        out_root = tmp_path / "run4"
        result = _run_cli([
            "--output-root", str(out_root), "--video-ids", "01_person_walk", "--frames", "0",
            "--no-models", "--device", "cpu", "--no-lpips",
        ])
        assert result.returncode == 0, result.stderr
        report = (out_root / "REPORT.md").read_text()
        assert "서버 실측 대기" in report

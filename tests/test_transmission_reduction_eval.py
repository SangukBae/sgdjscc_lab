"""tests/test_transmission_reduction_eval.py – CPU-only tests for the
transmission-reduction sweep driver's non-GPU logic (aggregation, pareto
selection, CSV writing, keyframe-reason normalization). Loaded via importlib
since it's a scripts/ file, not an installed package — same pattern as
test_video_rate_benchmark.py.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_txred_mod", _REPO_ROOT / "scripts" / "run_transmission_reduction_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


def _default_args(**overrides):
    ns = mod._parse_args(["--output-root", "/tmp/x"])
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class TestArgParsing:
    def test_default_configs_match_fair_baseline_grid(self):
        args = mod._parse_args(["--output-root", "/tmp/x"])
        configs = args.configs.split(",")
        # Keep baselines in the default run until a digital-baseline and
        # semantic-reliability validation confirms an operating point.
        assert configs == [
            "fixed_awgn",
            "fixed_int16", "fixed_int8", "fixed_int6", "fixed_int4",
            "skem_int16", "skem_int8", "skem_int6", "skem_int4",
        ]

    def test_unknown_config_rejected_by_run(self, tmp_path):
        import pytest
        with pytest.raises(ValueError):
            mod.run(["--output-root", str(tmp_path / "out"), "--configs", "bogus_config",
                     "--dataset-root", str(tmp_path)])

    def test_real_psss_backend_without_model_id_rejected(self, tmp_path):
        import pytest
        with pytest.raises(ValueError):
            mod.run(["--output-root", str(tmp_path / "out"), "--psss-backend", "real",
                     "--dataset-root", str(tmp_path)])

    def test_all_configs_includes_reliable_digital_and_lossless(self):
        assert "fixed_int16" in mod.ALL_CONFIGS
        assert "fixed_float32" in mod.ALL_CONFIGS
        assert "skem_int16" in mod.ALL_CONFIGS

    def test_retry_failed_is_explicit_opt_in(self):
        assert mod._parse_args(["--output-root", "/tmp/x"]).retry_failed is False
        assert mod._parse_args(["--output-root", "/tmp/x", "--retry-failed"]).retry_failed is True


def _pv_row(**overrides):
    """A complete, valid per_video_metrics.csv-shaped row (all fields
    _aggregate/_pareto_frontier read), with sane defaults so each test only
    needs to override what's relevant to it."""
    row = {
        "video": "v1", "config": "fixed_int16", "selector": "fixed", "channel": "int16",
        "bit_depth": 16, "psss_backend_kind": None, "digital_step_policy": "fixed_reference",
        "n_frames_total": 12, "n_transmitting_frames": 3, "n_keyframes_selected": 3,
        "n_nan_or_inf_frames": 0, "nonfinite_stages": "",
        "fixed_selector_kind": "", "fixed_count_target": "", "fixed_max_gop_used": "",
        "keyframe_count_matched": "",
        "n_quality_frames": 12, "valid_frame_ratio": 1.0,
        "mean_psnr": 24.0, "mean_ssim": 0.79, "mean_lpips": 0.40,
        "latent_elements_total": 294912,
        "analog_channel_symbols_total": "", "source_packet_bits_total": 640000,
        "digital_side_information_bytes_total": "",
        "total_bundle_bytes": 80000, "total_bundle_bytes_per_frame": 80000 / 12,
        "analog_no_wire_bytes": False, "visual_transport_complete": True,
        "total_elapsed_s": 1.0,
    }
    row.update(overrides)
    return row


class TestAggregateAndPareto:
    def _rows(self):
        return [
            _pv_row(config="fixed_awgn", channel="awgn", bit_depth="",
                    mean_psnr=24.0, mean_ssim=0.79, mean_lpips=0.40,
                    total_bundle_bytes="", total_bundle_bytes_per_frame="",
                    analog_no_wire_bytes=True),
            _pv_row(config="fixed_int16", channel="int16", bit_depth=16,
                    mean_psnr=23.99, mean_ssim=0.7899, mean_lpips=0.401,
                    total_bundle_bytes=80000, total_bundle_bytes_per_frame=80000 / 12),
            _pv_row(config="fixed_int8", channel="int8", bit_depth=8,
                    mean_psnr=23.8, mean_ssim=0.785, mean_lpips=0.41,
                    total_bundle_bytes=40000, total_bundle_bytes_per_frame=40000 / 12),
            _pv_row(config="fixed_int4", channel="int4", bit_depth=4,
                    mean_psnr=20.0, mean_ssim=0.70, mean_lpips=0.55,  # way out of budget
                    total_bundle_bytes=20000, total_bundle_bytes_per_frame=20000 / 12),
        ]

    def test_aggregate_groups_by_config_and_includes_lpips(self):
        rows = self._rows()
        agg = mod._aggregate(rows)
        assert {r["config"] for r in agg} == {"fixed_awgn", "fixed_int16", "fixed_int8", "fixed_int4"}
        awgn_row = next(r for r in agg if r["config"] == "fixed_awgn")
        assert awgn_row["mean_total_bundle_bytes_per_video"] == ""
        assert awgn_row["mean_total_bundle_bytes_per_frame"] == ""
        assert awgn_row["analog_no_wire_bytes"] is True
        assert awgn_row["mean_lpips"] == 0.40

    def test_aggregate_computes_bytes_per_frame_separately_from_bytes_per_video(self):
        agg = mod._aggregate(self._rows())
        int16_row = next(r for r in agg if r["config"] == "fixed_int16")
        assert int16_row["mean_total_bundle_bytes_per_video"] == 80000
        assert abs(int16_row["mean_total_bundle_bytes_per_frame"] - 80000 / 12) < 1e-9

    def test_pareto_baseline_prefers_reliable_digital_over_awgn(self):
        rows = self._rows()
        agg = mod._aggregate(rows)
        pareto, baseline_info = mod._pareto_frontier(agg)
        assert baseline_info["baseline_config"] == "fixed_int16"
        assert baseline_info["baseline_is_analog"] is False
        assert not any(r["config"] == "fixed_awgn" for r in pareto)  # AWGN never a Pareto candidate here

    def test_awgn_with_side_information_bytes_is_still_not_a_pareto_candidate(self):
        rows = self._rows()
        awgn = rows[0]
        awgn["total_bundle_bytes"] = 1
        awgn["total_bundle_bytes_per_frame"] = 1 / 12
        awgn["visual_transport_complete"] = False
        pareto, _ = mod._pareto_frontier(mod._aggregate(rows))
        assert not any(r["channel"] == "awgn" for r in pareto)

    def test_pareto_never_falls_back_to_awgn_when_no_digital_baseline_present(self):
        # float32/int16 (or a "정상" i.e. zero-NaN version of either) are the
        # ONLY eligible baselines; AWGN mixes analog noise with quantization
        # loss and is never a meaningful "reliable digital" reference, so with
        # no digital reliable config present the baseline must be unavailable,
        # never silently fall back to AWGN.
        rows = [r for r in self._rows() if r["config"] in ("fixed_awgn", "fixed_int8")]
        agg = mod._aggregate(rows)
        pareto, baseline_info = mod._pareto_frontier(agg)
        assert baseline_info["baseline_valid"] is False
        assert baseline_info["baseline_config"] is None
        assert pareto == []

    def test_baseline_preference_excludes_awgn_entirely(self):
        assert "fixed_awgn" not in mod.BASELINE_PREFERENCE
        assert "skem_awgn" not in mod.BASELINE_PREFERENCE
        assert set(mod.BASELINE_PREFERENCE) == {
            "fixed_float32", "fixed_int16", "skem_float32", "skem_int16",
        }

    def test_pareto_baseline_ignored_even_if_awgn_has_zero_nan_frames(self):
        # An AWGN row with n_nan_or_inf_frames == 0 must still never become the
        # baseline -- only its presence in BASELINE_PREFERENCE would allow that.
        rows = [r for r in self._rows() if r["config"] in ("fixed_awgn", "fixed_int8")]
        agg = mod._aggregate(rows)
        _, baseline_info = mod._pareto_frontier(agg)
        assert baseline_info["baseline_config"] != "fixed_awgn"
        assert baseline_info["baseline_valid"] is False

    def test_pareto_prefers_smallest_bytes_within_quality_gate(self):
        rows = self._rows()
        agg = mod._aggregate(rows)
        pareto, _ = mod._pareto_frontier(agg)
        in_budget = [r for r in pareto if r["within_quality_gate"]]
        assert any(r["config"] == "fixed_int8" for r in in_budget)
        assert not any(r["config"] == "fixed_int4" for r in in_budget)
        selected = next(r for r in pareto if r.get("selected_as_smallest_in_budget"))
        assert selected["config"] == "fixed_int8"

    def test_pareto_gate_checks_lpips_too(self):
        # int8 passes PSNR/SSIM vs the int16 baseline but LPIPS rise exceeds
        # the gate -> must be excluded from "within_quality_gate".
        rows = [
            _pv_row(config="fixed_int16", channel="int16", bit_depth=16,
                    mean_psnr=30.0, mean_ssim=0.90, mean_lpips=0.10,
                    total_bundle_bytes=8000, total_bundle_bytes_per_frame=8000 / 12),
            _pv_row(config="fixed_int4", channel="int4", bit_depth=4,
                    mean_psnr=29.8, mean_ssim=0.895, mean_lpips=0.20,  # rise = 0.10 > 0.02 gate
                    total_bundle_bytes=2000, total_bundle_bytes_per_frame=2000 / 12),
        ]
        agg = mod._aggregate(rows)
        pareto, _ = mod._pareto_frontier(agg)
        int4_row = next(r for r in pareto if r["config"] == "fixed_int4")
        assert int4_row["within_quality_gate"] is False
        assert abs(int4_row["lpips_rise"] - 0.10) < 1e-9

    def test_non_finite_frames_always_fail_gate(self):
        rows = self._rows()
        broken = next(r for r in rows if r["config"] == "fixed_int8")
        # Even excellent finite-frame averages cannot make an incomplete video
        # a valid Pareto candidate.
        broken.update({
            "mean_psnr": 99.0, "mean_ssim": 0.999, "mean_lpips": 0.001,
            "n_quality_frames": 1, "n_transmitting_frames": 3, "n_nan_or_inf_frames": 2,
        })
        pareto, _ = mod._pareto_frontier(mod._aggregate(rows))
        int8_row = next(r for r in pareto if r["config"] == "fixed_int8")
        assert int8_row["within_quality_gate"] is False
        assert int8_row["quality_gate_failure_reason"] == "non_finite_frames"

    def test_incomplete_quality_coverage_fails_gate_even_with_zero_nan_count(self):
        # valid_frame_ratio < 1 (fewer quality frames than transmitting
        # frames) must fail the gate on its own, independent of the
        # n_nan_or_inf_frames counter.
        rows = self._rows()
        broken = next(r for r in rows if r["config"] == "fixed_int8")
        broken.update({"n_quality_frames": 1, "n_transmitting_frames": 3, "n_nan_or_inf_frames": 0})
        pareto, _ = mod._pareto_frontier(mod._aggregate(rows))
        int8_row = next(r for r in pareto if r["config"] == "fixed_int8")
        assert int8_row["within_quality_gate"] is False
        assert int8_row["quality_gate_failure_reason"] == "incomplete_quality_coverage"

    def test_non_finite_metric_fails_gate(self):
        rows = self._rows()
        broken = next(r for r in rows if r["config"] == "fixed_int8")
        broken["mean_psnr"] = float("nan")
        pareto, _ = mod._pareto_frontier(mod._aggregate(rows))
        int8_row = next(r for r in pareto if r["config"] == "fixed_int8")
        assert int8_row["within_quality_gate"] is False
        assert int8_row["quality_gate_failure_reason"] == "non_finite_metrics"

    def test_missing_expected_video_fails_gate(self):
        rows = self._rows()  # every row is "v1" only
        agg = mod._aggregate(rows, expected_video_keys={"v1", "v2"})
        pareto, baseline_info = mod._pareto_frontier(agg)
        # baseline itself is missing an expected video -> unavailable
        assert baseline_info["baseline_valid"] is False

    def test_video_set_mismatch_vs_baseline_fails_gate(self):
        # expected_video_keys left unset (None) so "all_expected_videos_present"
        # is trivially true for every row -- isolates the video-SET-differs-
        # from-baseline check specifically (baseline covers {v1, v2}, the
        # candidate covers {v1, v3}: same count, different videos).
        rows = [
            _pv_row(video="v1", config="fixed_int16"),
            _pv_row(video="v2", config="fixed_int16",
                    total_bundle_bytes=80000, total_bundle_bytes_per_frame=80000 / 12),
            _pv_row(video="v1", config="fixed_int8", channel="int8", bit_depth=8,
                    total_bundle_bytes=40000, total_bundle_bytes_per_frame=40000 / 12),
            _pv_row(video="v3", config="fixed_int8", channel="int8", bit_depth=8,
                    total_bundle_bytes=40000, total_bundle_bytes_per_frame=40000 / 12),
        ]
        agg = mod._aggregate(rows)
        pareto, baseline_info = mod._pareto_frontier(agg)
        assert baseline_info["baseline_config"] == "fixed_int16"
        int8_row = next(r for r in pareto if r["config"] == "fixed_int8")
        assert int8_row["within_quality_gate"] is False
        assert int8_row["quality_gate_failure_reason"] == "video_set_mismatch_vs_baseline"

    def test_pareto_reports_nearest_when_nothing_qualifies(self):
        rows = [
            _pv_row(config="fixed_int16", channel="int16", bit_depth=16,
                    mean_psnr=24.0, mean_ssim=0.79, mean_lpips=0.40,
                    total_bundle_bytes="", total_bundle_bytes_per_frame=""),
            _pv_row(config="fixed_int4", channel="int4", bit_depth=4,
                    mean_psnr=10.0, mean_ssim=0.3, mean_lpips=0.9,
                    total_bundle_bytes=500, total_bundle_bytes_per_frame=500 / 12),
        ]
        agg = mod._aggregate(rows)
        pareto, baseline_info = mod._pareto_frontier(agg)
        assert baseline_info["baseline_config"] == "fixed_int16"
        assert len(pareto) == 1  # nearest candidate still reported, never hidden
        assert pareto[0]["within_quality_gate"] is False
        assert not any(r.get("selected_as_smallest_in_budget") for r in pareto)


class TestKeyframeCountMatchedAggregation:
    def test_all_applicable_rows_matched_is_true(self):
        rows = [
            _pv_row(video="v1", keyframe_count_matched=True),
            _pv_row(video="v2", keyframe_count_matched=True),
        ]
        agg = mod._aggregate(rows)
        assert agg[0]["keyframe_count_matched"] is True

    def test_any_applicable_row_unmatched_is_false(self):
        rows = [
            _pv_row(video="v1", keyframe_count_matched=True),
            _pv_row(video="v2", keyframe_count_matched=False),
        ]
        agg = mod._aggregate(rows)
        assert agg[0]["keyframe_count_matched"] is False

    def test_not_applicable_rows_never_read_as_false(self):
        # "" means matching wasn't requested/relevant for this row -- must
        # never be silently coerced into "not matched".
        rows = [_pv_row(video="v1", keyframe_count_matched="")]
        agg = mod._aggregate(rows)
        assert agg[0]["keyframe_count_matched"] == ""


class TestKeyframeReasonAndForceReasonHandling:
    def test_select_keyframes_uses_real_structured_force_reason(self, monkeypatch):
        # video/skem_selector.py returns a real, structured force_reason dict
        # (categorical: first_frame/scene_change/max_segment_length/psss) —
        # _select_keyframes must surface it as-is, never re-derive "forced"
        # status by pattern-matching the human-readable reason prose.
        class FakeSelector:
            def extract(self, frames):
                return {
                    "keyframes": [0, 2, 3],
                    "keyframe_reasons": {
                        "0": "first frame (K_1 = 1)",
                        "2": "scene_change detected by scene_detector",
                        "3": "S_rel=0.9 > threshold=0.35",
                    },
                    "force_reason": {"0": "first_frame", "2": "scene_change", "3": "psss"},
                    "psss_scores": [{"index": 3, "s_abs": 0.1, "s_rel": 0.9}],
                    "psss_backend_kind": "real",
                }

        monkeypatch.setattr(mod, "_build_selector", lambda *a, **kw: FakeSelector())
        args = _default_args()
        sel = mod._select_keyframes("v", [None] * 4, None, "skem", 0.35, 16, args)
        assert sel.force_reason == {0: "first_frame", 2: "scene_change", 3: "psss"}
        assert sel.psss_backend_kind == "real"

    def test_fixed_selector_without_force_reason_falls_back_to_selected(self, monkeypatch):
        # KeyframeExtractor (the "fixed" selector) has no force_reason concept
        # at all — _select_keyframes must not crash, and must not fabricate a
        # scene_change/psss category it cannot actually attribute.
        class FakeFixedSelector:
            def extract(self, frames):
                return {"keyframes": [0, 5], "keyframe_reasons": {}, "psss_scores": []}

        monkeypatch.setattr(mod, "_build_selector", lambda *a, **kw: FakeFixedSelector())
        args = _default_args()
        sel = mod._select_keyframes("v", [None] * 6, None, "fixed", 0.35, 16, args)
        assert sel.force_reason == {0: "first_frame", 5: "selected"}


class TestPsssCliPassthrough:
    def test_psss_backend_cfg_carries_model_id_device_dtype(self):
        args = _default_args(
            psss_backend="real", psss_model_id="org/some-causal-lm",
            psss_device="cuda:1", psss_dtype="bf16",
        )
        cfg = mod._psss_backend_cfg(args)
        assert cfg["real"]["model_id"] == "org/some-causal-lm"
        assert cfg["real"]["device"] == "cuda:1"
        assert cfg["real"]["dtype"] == "bf16"

    def test_build_selector_skem_passes_real_backend_request_to_build_psss_backend(self, monkeypatch):
        captured = {}

        def fake_build_psss_backend(name, cfg=None, **kwargs):
            captured["name"] = name
            captured["model_id"] = cfg.real.model_id if cfg is not None else None
            class _Stub:
                backend_kind = "real"
                backend_name = "stub"
                model_id = cfg.real.model_id if cfg is not None else None
            return _Stub()

        monkeypatch.setattr(
            "sgdjscc_lab.video.psss.build_psss_backend", fake_build_psss_backend,
        )
        args = _default_args(psss_backend="real", psss_model_id="org/some-causal-lm")
        mod._build_selector("skem", ["caption"] * 3, 0.35, 16, args)
        assert captured["name"] == "real"
        assert captured["model_id"] == "org/some-causal-lm"

    def test_mock_and_proxy_backend_kind_never_reported_as_real(self):
        args = _default_args(psss_backend="mock")
        sel = mod._build_selector("skem", ["caption"] * 3, 0.35, 16, args)
        assert sel.psss_backend.backend_kind == "mock"
        assert sel.psss_backend.backend_kind != "real"


class TestCsvWriting:
    def test_write_csv_round_trips(self, tmp_path):
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        path = tmp_path / "out.csv"
        mod._write_csv(path, rows)
        with open(path, newline="", encoding="utf-8") as fh:
            read_back = list(csv.DictReader(fh))
        assert read_back == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]

    def test_write_csv_handles_empty_rows(self, tmp_path):
        path = tmp_path / "empty.csv"
        mod._write_csv(path, [])
        assert path.exists()


class TestResumeRoundTrip:
    """--resume reloads a prior (possibly interrupted) run's per_video_metrics.csv
    via _read_csv_dicts + _coerce_per_video_row; _aggregate/_pareto_frontier must
    treat a reloaded row identically to a freshly-computed one."""

    def _fresh_rows(self):
        return [
            _pv_row(config="fixed_int16", channel="int16", bit_depth=16,
                    mean_psnr=23.99, mean_ssim=0.7899, mean_lpips=0.401,
                    total_bundle_bytes=80000, total_bundle_bytes_per_frame=80000 / 12,
                    keyframe_count_matched=True),
            _pv_row(config="fixed_int8", channel="int8", bit_depth=8,
                    mean_psnr=23.8, mean_ssim=0.785, mean_lpips=0.41,
                    total_bundle_bytes=40000, total_bundle_bytes_per_frame=40000 / 12,
                    keyframe_count_matched=False),
        ]

    def test_reloaded_rows_produce_identical_aggregate_and_pareto(self, tmp_path):
        fresh = self._fresh_rows()
        path = tmp_path / "per_video_metrics.csv"
        mod._write_csv(path, fresh)

        reloaded = [mod._coerce_per_video_row(r) for r in mod._read_csv_dicts(path)]

        agg_fresh = mod._aggregate(fresh)
        agg_reloaded = mod._aggregate(reloaded)
        assert agg_fresh == agg_reloaded

        pareto_fresh, base_fresh = mod._pareto_frontier(agg_fresh)
        pareto_reloaded, base_reloaded = mod._pareto_frontier(agg_reloaded)
        assert pareto_fresh == pareto_reloaded
        assert base_fresh == base_reloaded

    def test_reloaded_keyframe_count_matched_stays_a_real_bool(self, tmp_path):
        fresh = self._fresh_rows()
        path = tmp_path / "per_video_metrics.csv"
        mod._write_csv(path, fresh)
        reloaded = [mod._coerce_per_video_row(r) for r in mod._read_csv_dicts(path)]
        matched = {r["config"]: r["keyframe_count_matched"] for r in reloaded}
        assert matched["fixed_int16"] is True
        assert matched["fixed_int8"] is False

    def test_reloaded_not_applicable_keyframe_count_matched_stays_blank(self, tmp_path):
        fresh = [_pv_row(config="fixed_int16", keyframe_count_matched="")]
        path = tmp_path / "per_video_metrics.csv"
        mod._write_csv(path, fresh)
        reloaded = [mod._coerce_per_video_row(r) for r in mod._read_csv_dicts(path)]
        assert reloaded[0]["keyframe_count_matched"] == ""

    def test_read_csv_dicts_returns_empty_for_missing_or_empty_file(self, tmp_path):
        assert mod._read_csv_dicts(tmp_path / "does_not_exist.csv") == []
        empty = tmp_path / "empty.csv"
        empty.write_text("", encoding="utf-8")
        assert mod._read_csv_dicts(empty) == []

    def test_done_pairs_set_built_from_reloaded_rows(self, tmp_path):
        fresh = self._fresh_rows()
        path = tmp_path / "per_video_metrics.csv"
        mod._write_csv(path, fresh)
        reloaded = [mod._coerce_per_video_row(r) for r in mod._read_csv_dicts(path)]
        done_pairs = {(r["video"], r["config"]) for r in reloaded}
        assert done_pairs == {("v1", "fixed_int16"), ("v1", "fixed_int8")}


class TestWriteManifest:
    def test_writes_manifest_json_with_captured_command_and_real_seed(self, tmp_path, monkeypatch):
        from omegaconf import OmegaConf

        args = _default_args(dataset_root=str(tmp_path / "no_such_dataset"), seed=777)
        cfg = OmegaConf.create({"snr_db": 10.0, "use_phase4": True})
        output_root = tmp_path / "out"
        output_root.mkdir()

        monkeypatch.setattr(sys, "argv", [
            "run_transmission_reduction_eval.py", "--output-root", str(output_root),
        ])
        signature = {"seed": 777, "dataset_manifest_sha256": "unknown"}
        mod._write_manifest(
            args, output_root, cfg,
            per_video_rows=[_pv_row(n_nan_or_inf_frames=0)],
            failed_pairs=[],
            signature=signature, phase="final",
        )

        manifest_path = output_root / "run_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["command"]["source"] == "captured"
        assert "--output-root" in manifest["command"]["argv"]
        assert manifest["seed"] == 777
        assert manifest["resolved_config"]["status"] == "resolved"
        assert manifest["checkpoints"]["status"] in ("not_set", "recorded")
        assert manifest["extra"]["phase"] == "final"
        assert manifest["extra"]["run_signature"] == signature
        assert "output_artifact_sha256" in manifest["extra"]

    def test_manifest_counts_failed_pairs_and_stages(self, tmp_path, monkeypatch):
        from omegaconf import OmegaConf

        args = _default_args(dataset_root=str(tmp_path), seed=1)
        output_root = tmp_path / "out"
        output_root.mkdir()
        monkeypatch.setattr(sys, "argv", ["prog", "--output-root", str(output_root)])
        mod._write_manifest(
            args, output_root, OmegaConf.create({"snr_db": 10}), [],
            [{"failure_stage": "diffusion_latent", "n_nan": "2", "n_inf": "1"}],
            {}, phase="final",
        )
        manifest = json.loads((output_root / "run_manifest.json").read_text())
        counts = manifest["nan_or_failure_counts"]
        assert counts["total_nan_or_inf_frames"] == 1
        assert counts["n_failed_pairs"] == 1
        assert counts["failed_pair_nan_values"] == 2
        assert counts["failure_stages"] == {"diffusion_latent": 1}
        assert manifest["extra"]["run_status"] == "completed_with_failures"

    def test_initial_phase_has_no_artifact_hashes(self, tmp_path, monkeypatch):
        from omegaconf import OmegaConf

        args = _default_args(dataset_root=str(tmp_path / "no_such_dataset"))
        cfg = OmegaConf.create({"snr_db": 10.0})
        output_root = tmp_path / "out"
        output_root.mkdir()
        monkeypatch.setattr(sys, "argv", ["prog", "--output-root", str(output_root)])

        mod._write_manifest(
            args, output_root, cfg, per_video_rows=[], failed_pairs=[],
            signature={}, phase="initial",
        )
        manifest = json.loads((output_root / "run_manifest_initial.json").read_text(encoding="utf-8"))
        assert "output_artifact_sha256" not in manifest["extra"]
        assert not (output_root / "run_manifest.json").exists()  # only the final phase writes this name

    def test_run_manifest_module_is_a_hard_dependency(self):
        # No try/except ImportError anywhere around the module-level import --
        # it must be a plain top-level import that fails loudly if missing.
        import inspect
        source = inspect.getsource(mod)
        assert "from sgdjscc_lab.utils import run_manifest as rm" in source
        assert '"status": "unavailable"' not in source


class TestRunSignatureAndResumeSafety:
    def _entries(self):
        return [
            {"key": "01_person_walk", "row": {"n_frames": "100"}},
            {"key": "02_car_pass", "row": {"n_frames": "100"}},
        ]

    def test_signature_captures_seed_granularity_psss_and_configs(self, tmp_path):
        from omegaconf import OmegaConf

        args = _default_args(seed=123, granularity="per_channel",
                              psss_threshold=0.5, configs="fixed_int8,skem_int8")
        cfg = OmegaConf.create({"a": 1})
        sig = mod._build_run_signature(args, cfg, self._entries(), tmp_path)
        assert sig["seed"] == 123
        assert sig["granularity"] == "per_channel"
        assert sig["psss"]["threshold"] == 0.5
        assert sig["configs"] == ["fixed_int8", "skem_int8"]
        assert sig["video_keys"] == ["01_person_walk", "02_car_pass"]
        assert sig["video_frame_counts"] == {"01_person_walk": 100, "02_car_pass": 100}
        assert "ablation_label" in sig

    def test_check_resume_signature_writes_on_first_run(self, tmp_path):
        output_root = tmp_path / "out"
        output_root.mkdir()
        sig = {"seed": 1}
        mod._check_resume_signature(output_root, sig)
        assert json.loads((output_root / "run_signature.json").read_text()) == sig

    def test_check_resume_signature_passes_when_identical(self, tmp_path):
        output_root = tmp_path / "out"
        output_root.mkdir()
        sig = {"seed": 1, "configs": ["fixed_int8"]}
        mod._check_resume_signature(output_root, sig)
        mod._check_resume_signature(output_root, dict(sig))  # identical, different dict object

    def test_check_resume_signature_refuses_on_seed_mismatch(self, tmp_path):
        import pytest
        output_root = tmp_path / "out"
        output_root.mkdir()
        mod._check_resume_signature(output_root, {"seed": 1})
        with pytest.raises(SystemExit, match="resume signature mismatch"):
            mod._check_resume_signature(output_root, {"seed": 2})

    def test_diff_signature_reports_only_changed_keys(self):
        diff = mod._diff_signature({"seed": 1, "same": "x"}, {"seed": 2, "same": "x"})
        assert "seed" in diff
        assert "same" not in diff


class TestHashOutputArtifacts:
    def test_hashes_only_files_that_exist(self, tmp_path):
        (tmp_path / "aggregate.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        hashes = mod._hash_output_artifacts(tmp_path)
        assert "aggregate.csv" in hashes
        assert "pareto_frontier.csv" not in hashes  # never fabricated for a file that doesn't exist

    def test_hash_is_real_sha256(self, tmp_path):
        (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
        hashes = mod._hash_output_artifacts(tmp_path)
        assert len(hashes["summary.json"]) == 64

    def test_hashes_post_summarizer_artifacts(self, tmp_path):
        for name in (
            "quantization_effect.csv", "selector_effect.csv",
            "normalization_effect_summary.json",
        ):
            (tmp_path / name).write_text("x", encoding="utf-8")
        hashes = mod._hash_output_artifacts(tmp_path)
        assert set(hashes) == {
            "quantization_effect.csv", "selector_effect.csv",
            "normalization_effect_summary.json",
        }


class TestRateMatching:
    def test_close_bytes_and_matched_keyframes_is_rate_matched(self):
        rows = [
            _pv_row(video="v1", config="fixed_int8", selector="fixed", channel="int8",
                    total_bundle_bytes=10000, total_bundle_bytes_per_frame=1000,
                    n_keyframes_selected=5, keyframe_count_matched=True),
            _pv_row(video="v1", config="skem_int8", selector="skem", channel="int8",
                    total_bundle_bytes=10200, total_bundle_bytes_per_frame=1020,
                    n_keyframes_selected=5, keyframe_count_matched=""),
        ]
        out = mod._compute_rate_matching(rows)
        assert len(out) == 1
        assert out[0]["rate_matched"] is True
        assert out[0]["byte_diff_ratio"] < mod.RATE_MATCH_BYTE_TOLERANCE

    def test_matched_keyframes_but_dissimilar_bytes_is_not_rate_matched(self):
        # Task requirement: never call it rate-matched just because keyframe
        # counts matched -- bytes must actually be close.
        rows = [
            _pv_row(video="v1", config="fixed_int8", selector="fixed", channel="int8",
                    total_bundle_bytes=10000, total_bundle_bytes_per_frame=1000,
                    n_keyframes_selected=5, keyframe_count_matched=True),
            _pv_row(video="v1", config="skem_int8", selector="skem", channel="int8",
                    total_bundle_bytes=50000, total_bundle_bytes_per_frame=5000,
                    n_keyframes_selected=5, keyframe_count_matched=""),
        ]
        out = mod._compute_rate_matching(rows)
        assert out[0]["rate_matched"] is False
        assert out[0]["byte_diff_ratio"] > mod.RATE_MATCH_BYTE_TOLERANCE

    def test_unmatched_keyframe_count_is_never_rate_matched_even_if_bytes_close(self):
        rows = [
            _pv_row(video="v1", config="fixed_int8", selector="fixed", channel="int8",
                    total_bundle_bytes=10000, total_bundle_bytes_per_frame=1000,
                    n_keyframes_selected=4, keyframe_count_matched=False),
            _pv_row(video="v1", config="skem_int8", selector="skem", channel="int8",
                    total_bundle_bytes=10100, total_bundle_bytes_per_frame=1010,
                    n_keyframes_selected=5, keyframe_count_matched=""),
        ]
        out = mod._compute_rate_matching(rows)
        assert out[0]["rate_matched"] is False

    def test_actual_equal_counts_override_stale_fixed_boolean(self):
        rows = [
            _pv_row(video="v1", config="fixed_int8", selector="fixed", channel="int8",
                    total_bundle_bytes=10000, n_keyframes_selected=5,
                    keyframe_count_matched=False),
            _pv_row(video="v1", config="skem_int8", selector="skem", channel="int8",
                    total_bundle_bytes=10100, n_keyframes_selected=5),
        ]
        assert mod._compute_rate_matching(rows)[0]["rate_matched"] is True

    def test_awgn_channel_excluded(self):
        rows = [
            _pv_row(video="v1", config="fixed_awgn", selector="fixed", channel="awgn"),
            _pv_row(video="v1", config="skem_awgn", selector="skem", channel="awgn"),
        ]
        assert mod._compute_rate_matching(rows) == []

    def test_missing_counterpart_selector_skipped(self):
        rows = [_pv_row(video="v1", config="fixed_int8", selector="fixed", channel="int8")]
        assert mod._compute_rate_matching(rows) == []

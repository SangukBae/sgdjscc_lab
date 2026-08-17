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
        # includes int16 reliable-digital baselines, not just fixed_awgn +
        # lossy int8/6/4 — the "fair baseline" fix.
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


class TestAggregateAndPareto:
    def _rows(self):
        return [
            {"video": "v1", "config": "fixed_awgn", "selector": "fixed", "channel": "awgn",
             "bit_depth": "", "psss_backend_kind": None,
             "mean_psnr": 24.0, "mean_ssim": 0.79, "mean_lpips": 0.40,
             "latent_elements_total": 294912, "total_bundle_bytes": "",
             "analog_no_wire_bytes": True, "total_elapsed_s": 1.0},
            {"video": "v1", "config": "fixed_int16", "selector": "fixed", "channel": "int16",
             "bit_depth": 16, "psss_backend_kind": None,
             "mean_psnr": 23.99, "mean_ssim": 0.7899, "mean_lpips": 0.401,
             "latent_elements_total": 294912, "total_bundle_bytes": 80000,
             "analog_no_wire_bytes": False, "total_elapsed_s": 1.05},
            {"video": "v1", "config": "fixed_int8", "selector": "fixed", "channel": "int8",
             "bit_depth": 8, "psss_backend_kind": None,
             "mean_psnr": 23.8, "mean_ssim": 0.785, "mean_lpips": 0.41,
             "latent_elements_total": 294912, "total_bundle_bytes": 40000,
             "analog_no_wire_bytes": False, "total_elapsed_s": 1.1},
            {"video": "v1", "config": "fixed_int4", "selector": "fixed", "channel": "int4",
             "bit_depth": 4, "psss_backend_kind": None,
             "mean_psnr": 20.0, "mean_ssim": 0.70, "mean_lpips": 0.55,  # way out of budget
             "latent_elements_total": 294912, "total_bundle_bytes": 20000,
             "analog_no_wire_bytes": False, "total_elapsed_s": 1.0},
        ]

    def test_aggregate_groups_by_config_and_includes_lpips(self):
        rows = self._rows()
        agg = mod._aggregate(rows)
        assert {r["config"] for r in agg} == {"fixed_awgn", "fixed_int16", "fixed_int8", "fixed_int4"}
        awgn_row = next(r for r in agg if r["config"] == "fixed_awgn")
        assert awgn_row["mean_total_bundle_bytes"] == ""
        assert awgn_row["analog_no_wire_bytes"] is True
        assert awgn_row["mean_lpips"] == 0.40

    def test_pareto_baseline_prefers_reliable_digital_over_awgn(self):
        rows = self._rows()
        agg = mod._aggregate(rows)
        pareto, baseline_info = mod._pareto_frontier(agg)
        assert baseline_info["baseline_config"] == "fixed_int16"
        assert baseline_info["baseline_is_analog"] is False
        assert not any(r["config"] == "fixed_awgn" for r in pareto)  # AWGN never a Pareto candidate here

    def test_pareto_falls_back_to_awgn_when_no_digital_baseline_present(self):
        rows = [r for r in self._rows() if r["config"] in ("fixed_awgn", "fixed_int8")]
        agg = mod._aggregate(rows)
        pareto, baseline_info = mod._pareto_frontier(agg)
        assert baseline_info["baseline_config"] == "fixed_awgn"
        assert baseline_info["baseline_is_analog"] is True

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
            {"video": "v1", "config": "fixed_int16", "selector": "fixed", "channel": "int16",
             "bit_depth": 16, "psss_backend_kind": None,
             "mean_psnr": 30.0, "mean_ssim": 0.90, "mean_lpips": 0.10,
             "latent_elements_total": 1000, "total_bundle_bytes": 8000,
             "analog_no_wire_bytes": False, "total_elapsed_s": 1.0},
            {"video": "v1", "config": "fixed_int4", "selector": "fixed", "channel": "int4",
             "bit_depth": 4, "psss_backend_kind": None,
             "mean_psnr": 29.8, "mean_ssim": 0.895, "mean_lpips": 0.20,  # lpips rise = 0.10 > 0.02 gate
             "latent_elements_total": 1000, "total_bundle_bytes": 2000,
             "analog_no_wire_bytes": False, "total_elapsed_s": 1.0},
        ]
        agg = mod._aggregate(rows)
        pareto, _ = mod._pareto_frontier(agg)
        int4_row = next(r for r in pareto if r["config"] == "fixed_int4")
        assert int4_row["within_quality_gate"] is False
        assert abs(int4_row["lpips_rise"] - 0.10) < 1e-9

    def test_pareto_reports_nearest_when_nothing_qualifies(self):
        rows = [
            {"config": "fixed_int16", "selector": "fixed", "channel": "int16", "bit_depth": 16,
             "psss_backend_kind": None, "mean_psnr": 24.0, "mean_ssim": 0.79, "mean_lpips": 0.40,
             "mean_latent_elements": 100, "mean_total_bundle_bytes": "", "analog_no_wire_bytes": False,
             "n_videos": 1},
            {"config": "fixed_int4", "selector": "fixed", "channel": "int4", "bit_depth": 4,
             "psss_backend_kind": None, "mean_psnr": 10.0, "mean_ssim": 0.3, "mean_lpips": 0.9,
             "mean_latent_elements": 100, "mean_total_bundle_bytes": 500, "analog_no_wire_bytes": False,
             "n_videos": 1},
        ]
        pareto, baseline_info = mod._pareto_frontier(rows)
        assert baseline_info["baseline_config"] == "fixed_int16"
        assert len(pareto) == 1  # nearest candidate still reported, never hidden
        assert pareto[0]["within_quality_gate"] is False
        assert not any(r.get("selected_as_smallest_in_budget") for r in pareto)


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

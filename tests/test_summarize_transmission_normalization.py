"""tests/test_summarize_transmission_normalization.py – quantization vs.
selector effect-table builders (scripts/summarize_transmission_normalization.py).

Pure CSV-in/CSV-out logic, no torch — loaded via importlib since it's a
scripts/ file, same pattern as test_transmission_reduction_eval.py.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_txnorm_summarize_mod", _REPO_ROOT / "scripts" / "summarize_transmission_normalization.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


def _row(selector, channel, bit_depth, psnr, ssim, lpips, bytes_, n_nan=0, n_kf=3):
    bytes_per_frame = (bytes_ / 12) if bytes_ != "" else ""
    return {
        "config": f"{selector}_{channel}", "selector": selector, "channel": channel,
        "bit_depth": bit_depth, "psss_backend_kind": "", "digital_step_policy": "fixed_reference",
        "ablation_label": "",
        "n_videos": 10,
        "total_frames": 120, "total_quality_frames": 120, "valid_frame_ratio": 1.0,
        "all_finite_metrics": True, "all_expected_videos_present": True,
        "mean_psnr": psnr, "mean_ssim": ssim, "mean_lpips": lpips,
        "mean_latent_elements": 294912, "mean_total_bundle_bytes_per_video": bytes_,
        "mean_total_bundle_bytes_per_frame": bytes_per_frame,
        "mean_n_keyframes_selected": n_kf, "keyframe_count_matched": "",
        "analog_no_wire_bytes": (channel == "awgn"),
        "total_nan_or_inf_frames": n_nan, "nonfinite_stages": "step_match" if n_nan else "",
    }


class TestQuantizationEffect:
    def test_isolates_bit_depth_within_one_selector(self):
        rows = [
            _row("fixed", "awgn", "", 24.0, 0.79, 0.40, ""),
            _row("fixed", "float32", 32, 24.5, 0.80, 0.38, 100000),
            _row("fixed", "int16", 16, 24.49, 0.7999, 0.381, 50000),
            _row("fixed", "int8", 8, 24.0, 0.79, 0.40, 25000),
            _row("fixed", "int4", 4, 18.0, 0.60, 0.60, 12500),
        ]
        out = mod.build_quantization_effect(rows)
        channels = {r["channel"] for r in out}
        assert "awgn" not in channels  # AWGN excluded entirely
        assert channels == {"float32", "int16", "int8", "int4"}

        ref_row = next(r for r in out if r["channel"] == "float32")
        assert ref_row["is_reference"] is True
        assert ref_row["psnr_drop_db"] == ""  # reference has no delta vs itself

        int8_row = next(r for r in out if r["channel"] == "int8")
        assert int8_row["reference_channel"] == "float32"
        assert abs(int8_row["psnr_drop_db"] - 0.5) < 1e-9
        assert int8_row["byte_ratio_vs_reference"] == 0.25

    def test_nan_config_excluded_from_being_a_comparison_but_still_listed(self):
        rows = [
            _row("fixed", "float32", 32, 24.5, 0.80, 0.38, 100000),
            _row("fixed", "int8", 8, 22.0, 0.75, 0.45, 25000, n_nan=3),
        ]
        out = mod.build_quantization_effect(rows)
        int8_row = next(r for r in out if r["channel"] == "int8")
        assert int8_row["valid"] is False
        assert int8_row["psnr_drop_db"] == ""
        assert "strict validity" in int8_row["note"]

    def test_falls_back_to_int16_reference_when_float32_absent_or_invalid(self):
        rows = [
            _row("fixed", "float32", 32, 24.5, 0.80, 0.38, 100000, n_nan=1),  # invalid reference candidate
            _row("fixed", "int16", 16, 24.49, 0.7999, 0.381, 50000),
            _row("fixed", "int8", 8, 24.0, 0.79, 0.40, 25000),
        ]
        out = mod.build_quantization_effect(rows)
        int8_row = next(r for r in out if r["channel"] == "int8")
        assert int8_row["reference_channel"] == "int16"

    def test_no_valid_reference_leaves_deltas_blank_but_lists_row(self):
        rows = [_row("fixed", "int8", 8, 24.0, 0.79, 0.40, 25000)]  # no float32/int16 at all
        out = mod.build_quantization_effect(rows)
        assert len(out) == 1
        assert out[0]["reference_channel"] == ""
        assert out[0]["psnr_drop_db"] == ""
        assert "no valid float32/int16 reference" in out[0]["note"]

    def test_selectors_are_isolated_from_each_other(self):
        rows = [
            _row("fixed", "float32", 32, 24.5, 0.80, 0.38, 100000),
            _row("fixed", "int8", 8, 24.0, 0.79, 0.40, 25000),
            _row("skem", "float32", 32, 26.0, 0.85, 0.30, 60000),
            _row("skem", "int8", 8, 25.5, 0.84, 0.32, 15000),
        ]
        out = mod.build_quantization_effect(rows)
        skem_int8 = next(r for r in out if r["selector"] == "skem" and r["channel"] == "int8")
        # Must compare against skem's OWN float32, never fixed's.
        assert abs(skem_int8["psnr_drop_db"] - 0.5) < 1e-9


class TestSelectorEffect:
    def test_isolates_selector_at_fixed_bit_depth(self):
        rows = [
            _row("fixed", "int8", 8, 24.0, 0.79, 0.40, 40000, n_kf=8),
            _row("skem", "int8", 8, 24.3, 0.80, 0.38, 24000, n_kf=5),
        ]
        out = mod.build_selector_effect(rows)
        assert len(out) == 1
        row = out[0]
        assert row["channel"] == "int8"
        assert row["valid"] is True
        assert abs(row["psnr_delta_skem_minus_fixed"] - 0.3) < 1e-9
        assert row["byte_ratio_skem_over_fixed"] == 0.6
        assert row["keyframe_count_delta_skem_minus_fixed"] == -3

    def test_awgn_excluded(self):
        rows = [
            _row("fixed", "awgn", "", 24.0, 0.79, 0.40, ""),
            _row("skem", "awgn", "", 24.3, 0.80, 0.38, ""),
        ]
        out = mod.build_selector_effect(rows)
        assert out == []

    def test_missing_counterpart_skipped_not_crashed(self):
        rows = [_row("fixed", "int4", 4, 18.0, 0.60, 0.60, 12500)]  # no skem_int4
        out = mod.build_selector_effect(rows)
        assert out == []

    def test_nan_on_either_side_marks_invalid_and_blanks_deltas(self):
        rows = [
            _row("fixed", "int4", 4, 18.0, 0.60, 0.60, 12500),
            _row("skem", "int4", 4, 5.0, 0.10, 0.90, 8000, n_nan=2),
        ]
        out = mod.build_selector_effect(rows)
        assert out[0]["valid"] is False
        assert out[0]["psnr_delta_skem_minus_fixed"] == ""
        assert "strict validity" in out[0]["note"]

    def test_skem_backend_kind_reported_never_labeled_real_when_proxy(self):
        rows = [
            _row("fixed", "int8", 8, 24.0, 0.79, 0.40, 40000),
            {**_row("skem", "int8", 8, 24.3, 0.80, 0.38, 24000), "psss_backend_kind": "proxy"},
        ]
        out = mod.build_selector_effect(rows)
        assert out[0]["skem_psss_backend_kind"] == "proxy"

    def test_keyframe_count_matched_field_passed_through(self):
        rows = [
            {**_row("fixed", "int8", 8, 24.0, 0.79, 0.40, 40000), "keyframe_count_matched": "True"},
            _row("skem", "int8", 8, 24.3, 0.80, 0.38, 24000),
        ]
        out = mod.build_selector_effect(rows)
        assert out[0]["keyframe_count_matched"] == "True"


class TestValidityConditions:
    def test_non_finite_metrics_makes_row_invalid(self):
        row = _row("fixed", "int8", 8, 24.0, 0.79, 0.40, 40000)
        row["all_finite_metrics"] = "False"
        assert mod._is_valid(row) is False

    def test_incomplete_valid_frame_ratio_makes_row_invalid(self):
        row = _row("fixed", "int8", 8, 24.0, 0.79, 0.40, 40000)
        row["valid_frame_ratio"] = "0.8"
        assert mod._is_valid(row) is False

    def test_missing_expected_video_makes_row_invalid(self):
        row = _row("fixed", "int8", 8, 24.0, 0.79, 0.40, 40000)
        row["all_expected_videos_present"] = "False"
        assert mod._is_valid(row) is False

    def test_all_conditions_satisfied_is_valid(self):
        row = _row("fixed", "int8", 8, 24.0, 0.79, 0.40, 40000)
        assert mod._is_valid(row) is True

    def test_missing_columns_fail_closed(self):
        row = {"total_nan_or_inf_frames": 0}
        assert mod._is_valid(row) is False

    def test_nan_quality_value_fails_closed(self):
        row = _row("fixed", "int8", 8, float("nan"), 0.79, 0.40, 40000)
        assert mod._is_valid(row) is False


class TestRunCli:
    def test_run_writes_both_csvs_and_summary(self, tmp_path):
        rows = [
            _row("fixed", "float32", 32, 24.5, 0.80, 0.38, 100000),
            _row("fixed", "int8", 8, 24.0, 0.79, 0.40, 25000),
            _row("skem", "float32", 32, 26.0, 0.85, 0.30, 60000),
            _row("skem", "int8", 8, 25.5, 0.84, 0.32, 15000),
        ]
        agg_path = tmp_path / "aggregate.csv"
        with open(agg_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        rc = mod.run(["--run-root", str(tmp_path)])
        assert rc == 0
        assert (tmp_path / "quantization_effect.csv").exists()
        assert (tmp_path / "selector_effect.csv").exists()
        assert (tmp_path / "normalization_effect_summary.json").exists()

    def test_run_raises_clear_error_when_aggregate_csv_missing(self, tmp_path):
        import pytest
        with pytest.raises(FileNotFoundError):
            mod.run(["--run-root", str(tmp_path)])

    def test_run_reports_ablation_policies_in_summary(self, tmp_path, capsys):
        rows = [
            {**_row("fixed", "float32", 32, 24.5, 0.80, 0.38, 100000), "digital_step_policy": "fixed_reference"},
            {**_row("fixed", "int8", 8, 24.0, 0.79, 0.40, 25000),
             "digital_step_policy": "bitdepth_proxy", "ablation_label": "bp_v1"},
        ]
        agg_path = tmp_path / "aggregate.csv"
        with open(agg_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        mod.run(["--run-root", str(tmp_path)])
        summary = json.loads((tmp_path / "normalization_effect_summary.json").read_text())
        assert summary["ablation_policies_present"] == ["bitdepth_proxy"]
        assert "ABLATION" in capsys.readouterr().err
        assert (tmp_path / "quantization_effect_ablation.csv").exists()
        assert (tmp_path / "selector_effect_ablation.csv").exists()
        assert not (tmp_path / "quantization_effect.csv").exists()
        with open(tmp_path / "quantization_effect_ablation.csv", newline="", encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        assert any(r["ablation_label"] == "bp_v1" for r in written)

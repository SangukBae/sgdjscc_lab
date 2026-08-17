"""tests/test_transmission_reduction_eval.py – CPU-only tests for the
transmission-reduction sweep driver's non-GPU logic (aggregation, pareto
selection, CSV writing). Loaded via importlib since it's a scripts/ file, not
an installed package — same pattern as test_video_rate_benchmark.py.
"""

from __future__ import annotations

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


class TestArgParsing:
    def test_default_configs_match_task_spec_grid(self):
        args = mod._parse_args(["--output-root", "/tmp/x"])
        configs = args.configs.split(",")
        assert configs == ["fixed_awgn", "fixed_int8", "fixed_int6", "fixed_int4",
                            "skem_int8", "skem_int6", "skem_int4"]

    def test_unknown_config_rejected_by_run(self, tmp_path):
        import pytest
        with pytest.raises(ValueError):
            mod.run(["--output-root", str(tmp_path / "out"), "--configs", "bogus_config",
                     "--dataset-root", str(tmp_path)])


class TestAggregateAndPareto:
    def _rows(self):
        return [
            {"video": "v1", "config": "fixed_awgn", "selector": "fixed", "channel": "awgn",
             "bit_depth": "", "n_keyframes": 9, "n_frames_total": 100,
             "mean_psnr": 24.0, "mean_ssim": 0.79, "mean_lpips": 0.40,
             "total_channel_symbols": 294912, "total_packet_bytes": "",
             "analog_no_wire_bytes": True, "total_elapsed_s": 1.0},
            {"video": "v1", "config": "fixed_int8", "selector": "fixed", "channel": "int8",
             "bit_depth": 8, "n_keyframes": 9, "n_frames_total": 100,
             "mean_psnr": 23.8, "mean_ssim": 0.785, "mean_lpips": 0.41,
             "total_channel_symbols": 294912, "total_packet_bytes": 40000,
             "analog_no_wire_bytes": False, "total_elapsed_s": 1.1},
            {"video": "v1", "config": "fixed_int4", "selector": "fixed", "channel": "int4",
             "bit_depth": 4, "n_keyframes": 9, "n_frames_total": 100,
             "mean_psnr": 20.0, "mean_ssim": 0.70, "mean_lpips": 0.55,  # way out of budget
             "total_channel_symbols": 294912, "total_packet_bytes": 20000,
             "analog_no_wire_bytes": False, "total_elapsed_s": 1.0},
        ]

    def test_aggregate_groups_by_config(self):
        rows = self._rows()
        agg = mod._aggregate(rows)
        assert {r["config"] for r in agg} == {"fixed_awgn", "fixed_int8", "fixed_int4"}
        awgn_row = next(r for r in agg if r["config"] == "fixed_awgn")
        assert awgn_row["mean_total_packet_bytes"] == ""
        assert awgn_row["analog_no_wire_bytes"] is True

    def test_pareto_prefers_smallest_bytes_within_quality_gate(self):
        rows = self._rows()
        agg = mod._aggregate(rows)
        pareto = mod._pareto_frontier(agg)
        in_budget = [r for r in pareto if r["within_quality_gate"]]
        assert any(r["config"] == "fixed_int8" for r in in_budget)
        assert not any(r["config"] == "fixed_int4" for r in in_budget)
        selected = next(r for r in pareto if r.get("selected_as_smallest_in_budget"))
        assert selected["config"] == "fixed_int8"

    def test_pareto_reports_nearest_when_nothing_qualifies(self):
        rows = [
            {"config": "fixed_awgn", "selector": "fixed", "channel": "awgn", "bit_depth": "",
             "mean_psnr": 24.0, "mean_ssim": 0.79, "mean_total_channel_symbols": 100,
             "mean_total_packet_bytes": "", "analog_no_wire_bytes": True, "n_videos": 1},
            {"config": "fixed_int4", "selector": "fixed", "channel": "int4", "bit_depth": 4,
             "mean_psnr": 10.0, "mean_ssim": 0.3, "mean_total_channel_symbols": 100,
             "mean_total_packet_bytes": 500, "analog_no_wire_bytes": False, "n_videos": 1},
        ]
        pareto = mod._pareto_frontier(rows)
        assert len(pareto) == 1  # nearest candidate still reported, never hidden
        assert pareto[0]["within_quality_gate"] is False
        assert not any(r.get("selected_as_smallest_in_budget") for r in pareto)


class TestKeyframeReasonKeyNormalization:
    def test_select_keyframes_normalizes_string_keyed_reasons(self, monkeypatch):
        # Regression: video/skem_selector.py::PsssKeyframeSelector.extract()
        # returns keyframe_reasons keyed by str(int) (a deliberate JSON-
        # compatibility convention shared with keyframes.json elsewhere in this
        # codebase), while psss_scores' "index" field stays a plain int.
        # _select_keyframes must normalize keyframe_reasons back to int keys or
        # every sel.reasons.get(kf_idx, "") lookup silently misses and reports
        # an empty reason/forced-flag for every real keyframe.
        class FakeSelector:
            def extract(self, frames):
                return {
                    "keyframes": [0, 3],
                    "keyframe_reasons": {"0": "first frame (K_1 = 1)", "3": "max_segment_length forced"},
                    "psss_scores": [{"index": 3, "s_abs": 0.1, "s_rel": 0.9}],
                }

        monkeypatch.setattr(mod, "_build_selector", lambda *a, **kw: FakeSelector())
        sel = mod._select_keyframes("v", [None, None, None, None], None, "skem", 0.35, 16)
        assert sel.reasons == {0: "first frame (K_1 = 1)", 3: "max_segment_length forced"}
        assert sel.forced_flags == [True, True]  # frame 0 always forced; frame 3 via max_segment_length


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

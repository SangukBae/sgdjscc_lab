"""tests/test_eval_pipeline.py – Unit tests for eval_pipeline and utilities.

No GPU, no SGDJSCC imports, no model checkpoints required.
All heavy components (models, inference) are replaced with lightweight mocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ─────────────────────────────────────────────────────────────────────────────
# utils/metrics_io
# ─────────────────────────────────────────────────────────────────────────────

class TestSummarizeMetrics:
    def test_empty_returns_n_zero(self):
        from sgdjscc_lab.utils.metrics_io import summarize_metrics
        result = summarize_metrics([])
        assert result == {"n": 0}

    def test_single_row_std_zero(self):
        from sgdjscc_lab.utils.metrics_io import summarize_metrics
        rows = [{"psnr": 30.0, "ssim": 0.9}]
        result = summarize_metrics(rows)
        assert result["psnr_mean"] == pytest.approx(30.0)
        assert result["psnr_std"]  == pytest.approx(0.0)

    def test_two_rows_correct_mean(self):
        from sgdjscc_lab.utils.metrics_io import summarize_metrics
        rows = [{"psnr": 30.0}, {"psnr": 20.0}]
        result = summarize_metrics(rows)
        assert result["psnr_mean"] == pytest.approx(25.0)

    def test_none_values_skipped(self):
        from sgdjscc_lab.utils.metrics_io import summarize_metrics
        rows = [{"psnr": 30.0, "lpips": None}, {"psnr": 28.0, "lpips": None}]
        result = summarize_metrics(rows)
        assert "psnr_mean" in result
        assert "lpips_mean" not in result

    def test_non_numeric_skipped(self):
        from sgdjscc_lab.utils.metrics_io import summarize_metrics
        rows = [{"filename": "a.png", "psnr": 30.0}]
        result = summarize_metrics(rows)
        assert "filename_mean" not in result
        assert "psnr_mean" in result

    def test_n_correct(self):
        from sgdjscc_lab.utils.metrics_io import summarize_metrics
        rows = [{"psnr": i} for i in range(5)]
        result = summarize_metrics(rows)
        assert result["n"] == 5


class TestFlattenMetricDict:
    def test_flat_dict_unchanged(self):
        from sgdjscc_lab.utils.metrics_io import flatten_metric_dict
        d = {"psnr": 30.0, "ssim": 0.9}
        result = flatten_metric_dict(d)
        assert result == d

    def test_nested_dict_flattened(self):
        from sgdjscc_lab.utils.metrics_io import flatten_metric_dict
        d = {"srs": {"clip": 0.8, "pres": 0.9}}
        result = flatten_metric_dict(d)
        assert result["srs_clip"] == 0.8
        assert result["srs_pres"] == 0.9

    def test_deeply_nested(self):
        from sgdjscc_lab.utils.metrics_io import flatten_metric_dict
        d = {"a": {"b": {"c": 42}}}
        result = flatten_metric_dict(d)
        assert result["a_b_c"] == 42


# ─────────────────────────────────────────────────────────────────────────────
# utils/csv_logger
# ─────────────────────────────────────────────────────────────────────────────

class TestCSVLogger:
    def test_creates_file_with_header(self, tmp_path):
        from sgdjscc_lab.utils.csv_logger import CSVLogger
        csv_path = tmp_path / "out.csv"
        with CSVLogger(csv_path, fieldnames=["a", "b"]) as log:
            log.write_row({"a": 1, "b": 2})
        lines = csv_path.read_text().splitlines()
        assert lines[0] == "a,b"
        assert lines[1] == "1,2"

    def test_write_multiple_rows(self, tmp_path):
        from sgdjscc_lab.utils.csv_logger import CSVLogger
        csv_path = tmp_path / "out.csv"
        rows = [{"x": i} for i in range(5)]
        with CSVLogger(csv_path, fieldnames=["x"]) as log:
            log.write_rows(rows)
        lines = csv_path.read_text().splitlines()
        assert len(lines) == 6  # header + 5 data rows

    def test_append_mode(self, tmp_path):
        from sgdjscc_lab.utils.csv_logger import CSVLogger
        csv_path = tmp_path / "out.csv"
        with CSVLogger(csv_path, fieldnames=["x"]) as log:
            log.write_row({"x": 1})
        with CSVLogger(csv_path, fieldnames=["x"], append=True) as log:
            log.write_row({"x": 2})
        lines = csv_path.read_text().splitlines()
        # Header appears only once (in first write)
        header_count = sum(1 for l in lines if l == "x")
        assert header_count == 1
        assert len(lines) == 3   # header + 2 data rows

    def test_parent_dirs_created(self, tmp_path):
        from sgdjscc_lab.utils.csv_logger import CSVLogger
        csv_path = tmp_path / "nested" / "dir" / "out.csv"
        with CSVLogger(csv_path, fieldnames=["x"]) as log:
            log.write_row({"x": 1})
        assert csv_path.exists()

    def test_extra_keys_ignored(self, tmp_path):
        from sgdjscc_lab.utils.csv_logger import CSVLogger
        csv_path = tmp_path / "out.csv"
        with CSVLogger(csv_path, fieldnames=["a"]) as log:
            log.write_row({"a": 1, "z": 99})  # 'z' not in fieldnames
        lines = csv_path.read_text().splitlines()
        assert lines[1] == "1"


# ─────────────────────────────────────────────────────────────────────────────
# EvalContext (unit-level)
# ─────────────────────────────────────────────────────────────────────────────

class TestEvalContext:
    def test_default_metrics_set_non_empty(self):
        from sgdjscc_lab.pipelines.eval_pipeline import EvalContext
        ctx = EvalContext()
        assert len(ctx.enabled_metrics) > 0

    def test_custom_metrics_set(self):
        from sgdjscc_lab.pipelines.eval_pipeline import EvalContext
        ctx = EvalContext(enabled_metrics={"psnr", "ssim"})
        assert ctx.enabled_metrics == {"psnr", "ssim"}


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_dataset with mock inference
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateDataset:
    """Integration tests using mock inference and mock evaluators."""

    def _make_cfg(self, tmp_path):
        from omegaconf import OmegaConf
        from PIL import Image as PILImage
        # Create two 256×256 PNG images
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for name in ("img_a.png", "img_b.png"):
            img = PILImage.new("RGB", (256, 256), color=(100, 150, 200))
            img.save(img_dir / name)
        cfg = OmegaConf.create({
            "input_path": str(img_dir),
            "output_dir": str(tmp_path / "out"),
            "snr_db": 10,
            "device": "cpu",
        })
        return cfg

    def _make_mock_reconstruct_fn(self):
        """Return a no-op reconstruct function: returns identical original."""
        def fn(fpath, models, cfg):
            from sgdjscc_lab.io import load_image_as_tensor
            orig = load_image_as_tensor(fpath)
            return orig, orig.clone()
        return fn

    def _make_quality_only_ctx(self):
        from sgdjscc_lab.pipelines.eval_pipeline import EvalContext
        return EvalContext(enabled_metrics={"psnr", "ssim"})

    def test_returns_one_row_per_image(self, tmp_path):
        from sgdjscc_lab.pipelines.eval_pipeline import evaluate_dataset
        cfg = self._make_cfg(tmp_path)
        ctx = self._make_quality_only_ctx()
        rows = evaluate_dataset(
            cfg, models=None, eval_ctx=ctx, snr_db=10.0,
            reconstruct_fn=self._make_mock_reconstruct_fn(),
        )
        assert len(rows) == 2

    def test_rows_contain_filename_and_snr(self, tmp_path):
        from sgdjscc_lab.pipelines.eval_pipeline import evaluate_dataset
        cfg = self._make_cfg(tmp_path)
        ctx = self._make_quality_only_ctx()
        rows = evaluate_dataset(
            cfg, models=None, eval_ctx=ctx, snr_db=5.0,
            reconstruct_fn=self._make_mock_reconstruct_fn(),
        )
        for row in rows:
            assert "filename" in row
            assert row["snr_db"] == 5.0

    def test_csv_written_incrementally(self, tmp_path):
        from sgdjscc_lab.pipelines.eval_pipeline import evaluate_dataset
        from sgdjscc_lab.utils.csv_logger import CSVLogger, RESULT_COLUMNS
        cfg = self._make_cfg(tmp_path)
        ctx = self._make_quality_only_ctx()
        csv_path = tmp_path / "results.csv"
        with CSVLogger(csv_path, fieldnames=RESULT_COLUMNS) as log:
            rows = evaluate_dataset(
                cfg, models=None, eval_ctx=ctx, snr_db=10.0,
                csv_logger=log,
                reconstruct_fn=self._make_mock_reconstruct_fn(),
            )
        assert csv_path.exists()
        lines = csv_path.read_text().splitlines()
        assert len(lines) == 3   # header + 2 images

    def test_snr_sweep_produces_correct_row_count(self, tmp_path):
        from sgdjscc_lab.pipelines.eval_pipeline import evaluate_snr_sweep
        cfg = self._make_cfg(tmp_path)
        ctx = self._make_quality_only_ctx()
        snr_list = [5.0, 10.0, 15.0]
        sweep = evaluate_snr_sweep(
            cfg, models=None, eval_ctx=ctx, snr_list=snr_list,
            reconstruct_fn=self._make_mock_reconstruct_fn(),
        )
        assert set(sweep.keys()) == {5.0, 10.0, 15.0}
        for snr, result in sweep.items():
            assert len(result["rows"]) == 2   # 2 images per SNR

    def test_identical_reconstruction_psnr_inf(self, tmp_path):
        from sgdjscc_lab.pipelines.eval_pipeline import evaluate_dataset
        cfg = self._make_cfg(tmp_path)
        ctx = self._make_quality_only_ctx()
        rows = evaluate_dataset(
            cfg, models=None, eval_ctx=ctx, snr_db=10.0,
            reconstruct_fn=self._make_mock_reconstruct_fn(),
        )
        for row in rows:
            assert row["psnr"] == float("inf")

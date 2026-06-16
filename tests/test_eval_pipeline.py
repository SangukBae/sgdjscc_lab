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

class TestMetricProfiles:
    def test_paper_profile_is_paper_set(self):
        from sgdjscc_lab.utils.metric_profiles import resolve_profile, PAPER_METRICS
        assert resolve_profile("paper") == PAPER_METRICS
        assert "fid" in resolve_profile("paper")
        assert "ssim" not in resolve_profile("paper")   # SSIM not in paper set

    def test_extended_keeps_ssim_excludes_fid(self):
        from sgdjscc_lab.utils.metric_profiles import resolve_profile
        ext = resolve_profile("extended")
        assert "ssim" in ext and "semantic_reliability_score" in ext
        assert "fid" not in ext
        assert "fid" in resolve_profile("full")

    def test_profile_columns_order_and_membership(self):
        from sgdjscc_lab.utils.metric_profiles import profile_columns
        cols = profile_columns("paper")
        assert cols[:2] == ["filename", "snr_db"]
        # fid pulls in its provenance column so results stay unambiguous
        assert cols == ["filename", "snr_db", "psnr", "lpips",
                        "clip_image_image", "clip_text_image", "fid", "fid_backend"]

    def test_columns_for_metrics_tracks_narrowed_set(self):
        # simulate --profile paper + --no-clip: CLIP columns must drop from header
        from sgdjscc_lab.utils.metric_profiles import resolve_profile, columns_for_metrics
        enabled = resolve_profile("paper") - {
            "clip_image_image", "clip_text_image",
        }
        cols = columns_for_metrics(enabled)
        assert "clip_image_image" not in cols and "clip_text_image" not in cols
        assert cols == ["filename", "snr_db", "psnr", "lpips", "fid", "fid_backend"]

    def test_ssim_flagged_non_paper(self):
        from sgdjscc_lab.utils.metric_profiles import NON_PAPER_METRICS, is_paper_metric
        assert "ssim" in NON_PAPER_METRICS and not is_paper_metric("ssim")
        assert is_paper_metric("psnr") and is_paper_metric("fid")

    def test_unknown_profile_raises(self):
        import pytest
        from sgdjscc_lab.utils.metric_profiles import resolve_profile
        with pytest.raises(ValueError, match="Unknown metric profile"):
            resolve_profile("bogus")


class TestFIDEvaluator:
    """Unit tests for the dataset-level FID evaluator (injected feature_fn)."""

    def _feat(self):
        return lambda x: x.flatten(1)[:, :16]

    def test_identical_sets_zero_different_positive(self):
        import torch
        from sgdjscc_lab.evaluators.fid import FIDEvaluator
        torch.manual_seed(0)
        a = torch.rand(3, 3, 32, 32)
        b = torch.rand(3, 3, 32, 32)
        # identical sets → FID ≈ 0
        f0 = FIDEvaluator(feature_fn=self._feat())
        f0.add(a, a.clone())
        assert f0.is_proxy is True
        assert abs(f0.compute()) < 1e-4
        # different sets → FID > 0
        f1 = FIDEvaluator(feature_fn=self._feat())
        f1.add(a, b)
        assert f1.compute() > 0.0

    def test_graceful_none_when_no_samples(self):
        from sgdjscc_lab.evaluators.fid import FIDEvaluator
        assert FIDEvaluator(feature_fn=self._feat()).compute() is None

    def test_ensure_backend_rejects_proxy(self):
        from sgdjscc_lab.evaluators.fid import FIDEvaluator
        # injected proxy feature_fn is NOT a real Inception backend
        ev = FIDEvaluator(feature_fn=self._feat())
        assert ev.ensure_backend() is False
        assert ev.backend_name == "proxy"
        # no feature_fn → ensure_backend returns a bool; backend is inception or
        # unavailable depending on the environment (no assertion on which).
        res = FIDEvaluator().ensure_backend()
        assert isinstance(res, bool)


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

    def _noisy_reconstruct_fn(self):
        """Reconstruct = original + small deterministic perturbation (FID > 0)."""
        def fn(fpath, models, cfg):
            import torch
            from sgdjscc_lab.io import load_image_as_tensor
            orig = load_image_as_tensor(fpath)
            torch.manual_seed(int(abs(hash(fpath.name)) % 1000))
            return orig, (orig + 0.1 * torch.rand_like(orig)).clamp(0, 1)
        return fn

    def test_fid_fills_column_and_defers_csv(self, tmp_path):
        from sgdjscc_lab.pipelines.eval_pipeline import EvalContext, evaluate_dataset
        from sgdjscc_lab.utils.csv_logger import CSVLogger, RESULT_COLUMNS
        cfg = self._make_cfg(tmp_path)
        # Cheap injected feature extractor (no Inception/network) → proxy FID.
        feat = lambda x: x.flatten(1)[:, :16]
        ctx = EvalContext(enabled_metrics={"psnr", "fid"}, fid_feature_fn=feat)
        csv_path = tmp_path / "fid.csv"
        with CSVLogger(csv_path, fieldnames=RESULT_COLUMNS) as log:
            rows = evaluate_dataset(
                cfg, models=None, eval_ctx=ctx, snr_db=10.0, csv_logger=log,
                reconstruct_fn=self._noisy_reconstruct_fn(),
            )
        # every row carries the SAME (dataset-level) FID value, and it's a float
        assert all(r.get("fid") is not None for r in rows)
        assert len({round(float(r["fid"]), 6) for r in rows}) == 1
        assert float(rows[0]["fid"]) > 0.0
        # provenance recorded per row so a proxy FID is not mistaken for Inception
        assert all(r.get("fid_backend") == "proxy" for r in rows)
        # CSV written via the deferred path: header + 2 rows, fid columns populated
        lines = csv_path.read_text().splitlines()
        assert len(lines) == 3
        header = lines[0].split(",")
        assert "fid" in header and "fid_backend" in header

    def test_regeneration_fid_uses_final_reconstruction(self, tmp_path, monkeypatch):
        """Regression: when use_regeneration_loop replaces the reconstruction, FID
        must accumulate the FINAL (regenerated) recon, not the stale initial one."""
        import torch
        from sgdjscc_lab.pipelines.eval_pipeline import EvalContext, evaluate_dataset
        import sgdjscc_lab.pipelines.regeneration_loop as rl

        cfg = self._make_cfg(tmp_path)
        cfg.use_regeneration_loop = True
        cfg.regeneration_threshold = 1.0      # always below → regenerate
        cfg.regeneration_max_retries = 1

        class _StubSRS:                        # avoids loading CLIP
            def evaluate(self, original, reconstructed, **kw):
                return {"semantic_reliability_score": 0.0,
                        "clip_image_image": 0.0, "clip_text_image": 0.0,
                        "object_preservation_rate": 0.0, "missing_object_rate": 0.0,
                        "additional_object_rate": 0.0, "hallucination_score": 0.0}

        # Regeneration returns a clearly-marked recon (all 0.5).
        monkeypatch.setattr(
            rl, "regenerate_if_needed",
            lambda original, reconstructed, metrics, **kw: (torch.full_like(reconstructed, 0.5), {}))
        monkeypatch.setattr(rl, "build_regeneration_pipeline", lambda cfg: None)

        seen = []
        def feat(x):
            seen.append(x.clone())
            return x.flatten(1)[:, :16]

        ctx = EvalContext(
            enabled_metrics={"psnr", "fid", "semantic_reliability_score",
                             "clip_image_image"},
            fid_feature_fn=feat,
        )
        ctx.srs_evaluator = _StubSRS()
        evaluate_dataset(
            cfg, models=object(), eval_ctx=ctx, snr_db=10.0,
            reconstruct_fn=self._make_mock_reconstruct_fn(),
        )
        # The FID 'fake' inputs (the reconstructions) must be the regenerated 0.5
        # tensors — two images ⇒ two such tensors. With the bug they would be the
        # initial recon (a copy of the original PNG) and this count would be 0.
        marked = [t for t in seen if torch.allclose(t, torch.full_like(t, 0.5))]
        assert len(marked) == 2

    def test_regeneration_repacketizes_final_reconstruction(self, tmp_path, monkeypatch):
        """Regression: use_packet_eval + use_regeneration_loop (no packet_regen) must
        re-extract the recon packet from the REGENERATED reconstruction, so packet
        metrics/artifacts are not left stale on the initial one."""
        import torch
        from sgdjscc_lab.pipelines.eval_pipeline import EvalContext, evaluate_dataset
        import sgdjscc_lab.pipelines.regeneration_loop as rl

        cfg = self._make_cfg(tmp_path)
        cfg.use_phase4 = True
        cfg.use_packet_eval = True
        cfg.use_packet_regeneration = False
        cfg.use_regeneration_loop = True
        cfg.regeneration_threshold = 1.0      # always below → regenerate

        extracted = []                        # (frame_id, mean) of each extraction
        class _StubExtractor:
            def extract(self, image, frame_id=None):
                extracted.append((frame_id, float(image.mean())))
                return {"frame_id": frame_id, "objects": []}

        class _StubSRS:
            def evaluate(self, original, reconstructed, orig_packet=None,
                         recon_packet=None, **kw):
                return {"semantic_reliability_score": 0.0,
                        "clip_image_image": 0.0, "clip_text_image": 0.0,
                        "object_preservation_rate": 0.0, "missing_object_rate": 0.0,
                        "additional_object_rate": 0.0, "hallucination_score": 0.0,
                        "srs_base": 0.0, "srs_packet": 0.0}

        monkeypatch.setattr(
            rl, "regenerate_if_needed",
            lambda original, reconstructed, metrics, **kw: (torch.full_like(reconstructed, 0.5), {}))
        monkeypatch.setattr(rl, "build_regeneration_pipeline", lambda cfg: None)

        ctx = EvalContext(enabled_metrics={"psnr", "semantic_reliability_score",
                                           "clip_image_image"})
        ctx.packet_extractor = _StubExtractor()
        ctx.srs_evaluator = _StubSRS()
        evaluate_dataset(
            cfg, models=object(), eval_ctx=ctx, snr_db=10.0,
            reconstruct_fn=self._make_mock_reconstruct_fn(),
        )
        # A recon-frame extraction with mean≈0.5 (the regenerated recon) must exist:
        # the packet was re-extracted from the FINAL reconstruction. With the bug,
        # recon packets would only be the initial recon (mean ≠ 0.5).
        recon_means = [m for (fid, m) in extracted
                       if fid is not None and not str(fid).endswith("_orig")]
        assert any(abs(m - 0.5) < 1e-3 for m in recon_means)

    def test_fid_disabled_streams_normally_no_fid(self, tmp_path):
        from sgdjscc_lab.pipelines.eval_pipeline import EvalContext, evaluate_dataset
        cfg = self._make_cfg(tmp_path)
        ctx = EvalContext(enabled_metrics={"psnr"})   # no fid
        rows = evaluate_dataset(
            cfg, models=None, eval_ctx=ctx, snr_db=10.0,
            reconstruct_fn=self._make_mock_reconstruct_fn(),
        )
        assert all(r.get("fid") is None for r in rows)

    def test_packet_caption_objects_config_wired(self):
        """Regression: packet_caption_objects must flow cfg → SemanticPacketExtractor
        (previously the param existed but was never read from config → always True)."""
        from omegaconf import OmegaConf
        from sgdjscc_lab.pipelines.eval_pipeline import EvalContext

        # Default (key absent) → True.
        ctx = EvalContext(enabled_metrics={"semantic_reliability_score"})
        ext = ctx._get_packet_extractor(models=None, cfg=OmegaConf.create({}))
        assert ext.caption_objects is True

        # Explicit false → propagated to the extractor.
        ctx2 = EvalContext(enabled_metrics={"semantic_reliability_score"})
        ext2 = ctx2._get_packet_extractor(
            models=None, cfg=OmegaConf.create({"packet_caption_objects": False}))
        assert ext2.caption_objects is False

    def test_acceleration_step_budget_reaches_reconstruction(self, tmp_path):
        """Phase 5-B: the configured sampler step budget reaches reconstruct cfg."""
        from sgdjscc_lab.pipelines.eval_pipeline import evaluate_dataset
        from omegaconf import OmegaConf
        from PIL import Image as PILImage

        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        PILImage.new("RGB", (256, 256), color=(80, 80, 80)).save(img_dir / "a.png")
        cfg = OmegaConf.create({
            "input_path": str(img_dir), "snr_db": 10, "device": "cpu",
            "diffusion_step": 50,
            "use_phase5": True,
            "acceleration": {"sampler": "ddim", "sampler_steps": 7},
        })
        ctx = self._make_quality_only_ctx()

        seen = {}

        def recon_fn(fpath, models, run_cfg):
            from sgdjscc_lab.io import load_image_as_tensor
            seen["diffusion_step"] = int(run_cfg.diffusion_step)
            orig = load_image_as_tensor(fpath)
            return orig, orig.clone()

        evaluate_dataset(cfg, models=None, eval_ctx=ctx, snr_db=10.0, reconstruct_fn=recon_fn)
        assert seen["diffusion_step"] == 7

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


# ─────────────────────────────────────────────────────────────────────────────
# Packet-aware evaluation: SNR-namespaced output (regression, no CLIP needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestPacketSNRNamespacing:
    """Packet JSON / error reports must not overwrite across an SNR sweep."""

    def _make_cfg(self, tmp_path):
        from omegaconf import OmegaConf
        from PIL import Image as PILImage
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        PILImage.new("RGB", (256, 256), color=(120, 120, 120)).save(img_dir / "img_a.png")
        return OmegaConf.create({
            "input_path": str(img_dir),
            "snr_db": 10,
            "device": "cpu",
            "use_phase4": True,
            "use_packet_eval": True,
            "packet_dir": str(tmp_path / "packets"),
        })

    def _ctx_with_mocks(self):
        from sgdjscc_lab.pipelines.eval_pipeline import EvalContext
        from sgdjscc_lab.guidance.semantic_packet_extractor import build_packet

        class _MockPacketExtractor:
            def extract(self, image, frame_id=None, caption=None):
                return build_packet(objects=["car"], scene="s", frame_id=frame_id)

        class _MockSRS:
            def evaluate(self, original, reconstructed, text_list=None,
                         orig_packet=None, recon_packet=None):
                return {
                    "semantic_reliability_score": 0.8, "srs_base": 0.8,
                    "srs_packet": 0.8, "clip_image_image": 0.9,
                    "object_match_rate": 1.0, "relation_consistency": 1.0,
                    "attribute_consistency": 1.0, "segmentation_consistency": None,
                    "scene_match": True, "missing_object_count": 0,
                    "additional_object_count": 0, "relation_error_count": 0,
                    "attribute_error_count": 0,
                    "error_report": {"missing_object_count": 0},
                }

        ctx = EvalContext(enabled_metrics={"semantic_reliability_score"})
        ctx.packet_extractor = _MockPacketExtractor()
        ctx.srs_evaluator = _MockSRS()
        return ctx

    def _reconstruct_fn(self):
        def fn(fpath, models, cfg):
            from sgdjscc_lab.io import load_image_as_tensor
            orig = load_image_as_tensor(fpath)
            return orig, orig.clone()
        return fn

    def test_srs_v2_column_when_enabled(self, tmp_path):
        """Phase 5-C: use_srs_v2 adds an srs_v2 value computed from base layers."""
        from sgdjscc_lab.pipelines.eval_pipeline import EvalContext, _compute_metrics
        import torch

        class _MockSRS:
            def evaluate(self, original, reconstructed, text_list=None,
                         orig_packet=None, recon_packet=None):
                return {"semantic_reliability_score": 0.8, "srs_base": 0.8,
                        "hallucination_score": 0.2, "clip_image_image": 0.9}

        ctx = EvalContext(enabled_metrics={"semantic_reliability_score"}, use_srs_v2=True)
        ctx.srs_evaluator = _MockSRS()
        row = _compute_metrics(
            torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8), ctx,
            filename="x.png", snr_db=10.0,
        )
        assert row.get("srs_v2") is not None
        assert 0.0 <= row["srs_v2"] <= 1.0

    def test_sweep_writes_per_snr_dirs(self, tmp_path):
        from sgdjscc_lab.pipelines.eval_pipeline import evaluate_dataset
        cfg = self._make_cfg(tmp_path)
        ctx = self._ctx_with_mocks()
        for snr in (0.0, 5.0):
            evaluate_dataset(cfg, models=None, eval_ctx=ctx, snr_db=snr,
                             reconstruct_fn=self._reconstruct_fn())
        base = tmp_path / "packets"
        assert (base / "snr_0").exists()
        assert (base / "snr_5").exists()
        # Each SNR keeps its own packet + error report (no overwrite).
        assert (base / "snr_0" / "img_a.packet.json").exists()
        assert (base / "snr_0" / "img_a.error_report.json").exists()
        assert (base / "snr_5" / "img_a.packet.json").exists()

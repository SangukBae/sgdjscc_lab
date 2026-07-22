"""tests/test_temporal_srs_calibration.py – Temporal SRS Calibration scaffold tests (ETRI 5차, step 10).

All target scores here are synthetic/mock — no real VLM judge or human label
is a test dependency (matching the module's own scope note).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _synthetic_temporal_samples(n=12, seed=0):
    from sgdjscc_lab.evaluators.temporal_srs_calibration import CalibrationSample
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        ptc, sfr, sdi = rng.random(), rng.random() * 0.3, rng.random() * 0.3
        target = 0.5 * ptc - 0.3 * sfr - 0.2 * sdi
        samples.append(CalibrationSample(features={"ptc": ptc, "sfr": sfr, "sdi": sdi}, target_score=target))
    return samples


def _synthetic_srs_samples(n=12, seed=1):
    from sgdjscc_lab.evaluators.temporal_srs_calibration import CalibrationSample
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        feats = {
            "clip_image_image": rng.random(), "clip_text_image": rng.random(),
            "object_preservation_rate": rng.random(), "missing_object_rate": rng.random() * 0.3,
            "additional_object_rate": rng.random() * 0.3,
        }
        target = (0.3 * feats["clip_image_image"] + 0.25 * feats["clip_text_image"]
                  + 0.25 * feats["object_preservation_rate"] - 0.1 * feats["missing_object_rate"]
                  - 0.1 * feats["additional_object_rate"])
        samples.append(CalibrationSample(features=feats, target_score=target))
    return samples


class TestWeightsIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        from sgdjscc_lab.evaluators.temporal_srs_calibration import save_weights, load_weights
        weights = {"w_ptc": 0.5, "w_sfr": -0.3, "w_sdi": -0.2}
        path = save_weights(weights, tmp_path / "weights.json")
        assert path.exists()
        assert load_weights(path) == weights


class TestFitWeightsLeastSquares:
    def test_recovers_known_linear_combination(self):
        from sgdjscc_lab.evaluators.temporal_srs_calibration import (
            fit_weights_least_squares, TEMPORAL_FEATURE_NAMES,
        )
        samples = _synthetic_temporal_samples()
        fit = fit_weights_least_squares(samples, TEMPORAL_FEATURE_NAMES)
        assert fit["weights"]["ptc"] == pytest.approx(0.5, abs=1e-6)
        assert fit["weights"]["sfr"] == pytest.approx(-0.3, abs=1e-6)
        assert fit["weights"]["sdi"] == pytest.approx(-0.2, abs=1e-6)
        assert fit["residual"] < 1e-6
        assert fit["n_samples"] == len(samples)

    def test_too_few_samples_raises(self):
        from sgdjscc_lab.evaluators.temporal_srs_calibration import (
            fit_weights_least_squares, TEMPORAL_FEATURE_NAMES,
        )
        with pytest.raises(ValueError):
            fit_weights_least_squares(_synthetic_temporal_samples(n=2), TEMPORAL_FEATURE_NAMES)

    def test_fit_intercept(self):
        from sgdjscc_lab.evaluators.temporal_srs_calibration import (
            CalibrationSample, fit_weights_least_squares,
        )
        # target = 2*x + 3 (needs an intercept to fit exactly)
        samples = [CalibrationSample(features={"x": float(i)}, target_score=2.0 * i + 3.0) for i in range(5)]
        fit = fit_weights_least_squares(samples, ("x",), fit_intercept=True)
        assert fit["weights"]["x"] == pytest.approx(2.0, abs=1e-6)
        assert fit["intercept"] == pytest.approx(3.0, abs=1e-6)


class TestTemporalSRSCalibration:
    def test_default_weights_present(self):
        from sgdjscc_lab.evaluators.temporal_srs_calibration import (
            TemporalSRSCalibration, DEFAULT_SRS_WEIGHTS, DEFAULT_TEMPORAL_WEIGHTS,
        )
        calib = TemporalSRSCalibration()
        assert calib.srs_weights == DEFAULT_SRS_WEIGHTS
        assert calib.temporal_weights == DEFAULT_TEMPORAL_WEIGHTS

    def test_fit_temporal_weights_updates_in_place_no_duplicate_keys(self):
        from sgdjscc_lab.evaluators.temporal_srs_calibration import TemporalSRSCalibration
        calib = TemporalSRSCalibration()
        calib.fit_temporal_weights(_synthetic_temporal_samples())
        assert set(calib.temporal_weights.keys()) == {"w_ptc", "w_sfr", "w_sdi"}
        assert calib.temporal_weights["w_ptc"] == pytest.approx(0.5, abs=1e-6)

    def test_fit_srs_weights_updates_in_place_no_duplicate_keys(self):
        from sgdjscc_lab.evaluators.temporal_srs_calibration import TemporalSRSCalibration
        calib = TemporalSRSCalibration()
        calib.fit_srs_weights(_synthetic_srs_samples())
        assert set(calib.srs_weights.keys()) == {"w_img", "w_txt", "w_pres", "w_miss", "w_add"}
        assert calib.srs_weights["w_img"] == pytest.approx(0.3, abs=1e-6)

    def test_save_load_roundtrip(self, tmp_path):
        from sgdjscc_lab.evaluators.temporal_srs_calibration import TemporalSRSCalibration
        calib = TemporalSRSCalibration()
        calib.fit_temporal_weights(_synthetic_temporal_samples())
        path = calib.save(tmp_path / "calib.json")
        loaded = TemporalSRSCalibration.load(path)
        assert loaded.temporal_weights == calib.temporal_weights
        assert loaded.srs_weights == calib.srs_weights

    def test_from_config_reads_overrides(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.evaluators.temporal_srs_calibration import TemporalSRSCalibration
        cfg = OmegaConf.create({
            "temporal_srs_calibration": {"temporal_weights": {"w_ptc": 0.9}},
        })
        calib = TemporalSRSCalibration.from_config(cfg)
        assert calib.temporal_weights["w_ptc"] == pytest.approx(0.9)
        assert calib.temporal_weights["w_sfr"] == pytest.approx(-0.3)   # untouched default

    def test_from_config_defaults_when_absent(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.evaluators.temporal_srs_calibration import (
            TemporalSRSCalibration, DEFAULT_TEMPORAL_WEIGHTS,
        )
        calib = TemporalSRSCalibration.from_config(OmegaConf.create({}))
        assert calib.temporal_weights == DEFAULT_TEMPORAL_WEIGHTS

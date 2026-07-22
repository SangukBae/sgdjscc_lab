"""evaluators/temporal_srs_calibration.py – Temporal SRS Calibration scaffold (ETRI 5차, step 10).

Scope note (read before touching this file)
--------------------------------------------
**No real VLM judge or human-label collection happens here.** This module
only provides:

1. A place to load/save SRS and Temporal-SRS weight configs in one JSON
   format (:func:`load_weights` / :func:`save_weights`,
   :class:`TemporalSRSCalibration`).
2. A weight-*fitting* function (:func:`fit_weights_least_squares`) that works
   today against whatever ``(feature_vector, target_score)`` pairs the caller
   supplies — in tests these are mock/synthetic scores, never a real VLM
   call. A genuine Temporal SRS Calibration would supply GT-annotated or
   VLM-judged ``target_score`` values; wiring that data source is out of
   scope for 5차 (see docs/etri_strategy.md 5차 구현 결과).

The two weight dicts intentionally mirror existing, already-used schemes:
``srs_weights`` mirrors ``evaluators/semantic_reliability.py``'s
``_DEFAULT_WEIGHTS`` (``w_img``/``w_txt``/``w_pres``/``w_miss``/``w_add``),
and ``temporal_weights`` gives ``PTC``/``SFR``/``SDI`` (from
``evaluators/temporal_consistency.py``) a placeholder linear combination —
this is where a fitted weight replaces the initial guess once real
calibration data exists.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

logger = logging.getLogger(__name__)

# Mirrors semantic_reliability.py::_DEFAULT_WEIGHTS — kept as a separate copy
# here (rather than importing it) so this module has no import-time
# dependency on the evaluator, and its own default can drift independently
# once real calibration data starts adjusting it.
DEFAULT_SRS_WEIGHTS: Dict[str, float] = {
    "w_img": 0.30, "w_txt": 0.25, "w_pres": 0.25, "w_miss": 0.10, "w_add": 0.10,
}
# Placeholder combination for a "temporal SRS" composite over PTC/SFR/SDI —
# PTC is a quality term (higher = better), SFR/SDI are error terms (higher =
# worse), hence the negative signs. Tune (or fit, see fit_weights_least_squares)
# once real GT/VLM-judged sequences exist.
DEFAULT_TEMPORAL_WEIGHTS: Dict[str, float] = {
    "w_ptc": 0.5, "w_sfr": -0.3, "w_sdi": -0.2,
}

SRS_FEATURE_NAMES = (
    "clip_image_image", "clip_text_image", "object_preservation_rate",
    "missing_object_rate", "additional_object_rate",
)
# Maps each SRS_FEATURE_NAMES entry to its DEFAULT_SRS_WEIGHTS key, in order —
# feature names and weight-dict keys use different naming schemes (the weight
# keys are short aliases), so fit_srs_weights() renames through this before
# updating self.srs_weights (otherwise the fitted weights would sit under a
# second, inconsistent set of keys instead of overwriting the defaults).
SRS_WEIGHT_KEYS = ("w_img", "w_txt", "w_pres", "w_miss", "w_add")

TEMPORAL_FEATURE_NAMES = ("ptc", "sfr", "sdi")
TEMPORAL_WEIGHT_KEYS = ("w_ptc", "w_sfr", "w_sdi")


@dataclass
class CalibrationSample:
    """One ``(feature vector, target score)`` pair for weight fitting.

    ``target_score`` is a GT/VLM judge's score for the SAME frame/sequence the
    feature values were computed from. **This 5차 scaffold never produces
    ``target_score`` itself** — the caller supplies it (mock/synthetic in
    tests; a real GT/VLM pipeline would be wired in later, out of scope here).
    """

    features: Dict[str, float]
    target_score: float
    item_id: Optional[object] = None


def load_weights(path) -> Dict:
    with open(Path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_weights(weights: Dict, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(weights, fh, indent=2)
    return p


def fit_weights_least_squares(
    samples: Sequence[CalibrationSample],
    feature_names: Sequence[str],
    fit_intercept: bool = False,
) -> Dict:
    """Fit linear weights ``target ≈ Σ w_i · feature_i (+ b)`` via ordinary
    least squares (``numpy.linalg.lstsq``).

    This is the "weight fitting" STUB referenced in the module docstring — it
    is a generic linear-regression fit, not a Temporal-SRS-specific model; it
    becomes a real calibration only once *real* GT/VLM ``target_score``s are
    supplied by the caller.

    Returns
    -------
    dict with ``weights`` (``{feature_name: coefficient}``), ``intercept``
    (``0.0`` when ``fit_intercept`` is False), ``residual`` (sum of squared
    residuals), ``n_samples``, ``rank``.
    """
    import numpy as np

    min_samples = len(feature_names) + (1 if fit_intercept else 0)
    if len(samples) < min_samples:
        raise ValueError(
            f"Need at least {min_samples} samples to fit {len(feature_names)} "
            f"weight(s){' + intercept' if fit_intercept else ''}; got {len(samples)}."
        )

    X = np.array([[float(s.features.get(f, 0.0)) for f in feature_names] for s in samples])
    y = np.array([float(s.target_score) for s in samples])
    if fit_intercept:
        X = np.hstack([X, np.ones((X.shape[0], 1))])

    coeffs, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if fit_intercept:
        weights = {f: float(c) for f, c in zip(feature_names, coeffs[:-1])}
        intercept = float(coeffs[-1])
    else:
        weights = {f: float(c) for f, c in zip(feature_names, coeffs)}
        intercept = 0.0

    residual = float(residuals[0]) if len(residuals) else float(np.sum((X @ coeffs - y) ** 2))
    return {
        "weights": weights, "intercept": intercept,
        "residual": residual, "n_samples": len(samples), "rank": int(rank),
    }


class TemporalSRSCalibration:
    """SRS / Temporal-SRS weight-config holder + fitting entry point.

    Parameters
    ----------
    srs_weights / temporal_weights:
        Current weight dicts; missing keys fall back to
        :data:`DEFAULT_SRS_WEIGHTS` / :data:`DEFAULT_TEMPORAL_WEIGHTS`.
    """

    def __init__(
        self,
        srs_weights: Optional[Dict[str, float]] = None,
        temporal_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.srs_weights = dict(DEFAULT_SRS_WEIGHTS)
        if srs_weights:
            self.srs_weights.update(srs_weights)
        self.temporal_weights = dict(DEFAULT_TEMPORAL_WEIGHTS)
        if temporal_weights:
            self.temporal_weights.update(temporal_weights)

    @classmethod
    def from_config(cls, cfg) -> "TemporalSRSCalibration":
        from omegaconf import OmegaConf
        srs_w = OmegaConf.select(cfg, "temporal_srs_calibration.srs_weights", default=None)
        temp_w = OmegaConf.select(cfg, "temporal_srs_calibration.temporal_weights", default=None)
        return cls(
            srs_weights=dict(srs_w) if srs_w else None,
            temporal_weights=dict(temp_w) if temp_w else None,
        )

    def fit_srs_weights(self, samples: Sequence[CalibrationSample], fit_intercept: bool = False) -> Dict:
        """Fit ``srs_weights`` in place from *samples*; returns the fit dict
        with ``weights`` renamed to the canonical ``w_img``/``w_txt``/...
        keys (overwriting, not duplicating, the current defaults)."""
        fit = fit_weights_least_squares(samples, SRS_FEATURE_NAMES, fit_intercept=fit_intercept)
        renamed = {wk: fit["weights"][fn] for fn, wk in zip(SRS_FEATURE_NAMES, SRS_WEIGHT_KEYS)}
        self.srs_weights.update(renamed)
        fit["weights"] = renamed
        return fit

    def fit_temporal_weights(self, samples: Sequence[CalibrationSample], fit_intercept: bool = False) -> Dict:
        """Fit ``temporal_weights`` in place from *samples*; returns the fit
        dict with ``weights`` renamed to the canonical ``w_ptc``/``w_sfr``/
        ``w_sdi`` keys (overwriting, not duplicating, the current defaults)."""
        fit = fit_weights_least_squares(samples, TEMPORAL_FEATURE_NAMES, fit_intercept=fit_intercept)
        renamed = {wk: fit["weights"][fn] for fn, wk in zip(TEMPORAL_FEATURE_NAMES, TEMPORAL_WEIGHT_KEYS)}
        self.temporal_weights.update(renamed)
        fit["weights"] = renamed
        return fit

    def to_dict(self) -> Dict:
        return {"srs_weights": dict(self.srs_weights), "temporal_weights": dict(self.temporal_weights)}

    def save(self, path) -> Path:
        return save_weights(self.to_dict(), path)

    @classmethod
    def load(cls, path) -> "TemporalSRSCalibration":
        data = load_weights(path)
        return cls(srs_weights=data.get("srs_weights"), temporal_weights=data.get("temporal_weights"))

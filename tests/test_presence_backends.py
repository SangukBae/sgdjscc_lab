"""tests/test_presence_backends.py – presence backend + calibrator tests (ETRI 5차, step 8).

Only :class:`MockPresenceBackend` (and small in-file stub backends) are
exercised here — no OWLv2/VQA weights or `transformers` extras are a test
dependency. Real backends (:class:`Owlv2PresenceBackend`,
:class:`VqaPresenceBackend`) are checked only for their "unavailable" error
path, never against real weights.
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
# PresenceResult / MockPresenceBackend
# ─────────────────────────────────────────────────────────────────────────────

class TestMockPresenceBackend:
    def test_present_object_high_confidence(self):
        from sgdjscc_lab.evaluators.presence_backends import MockPresenceBackend
        backend = MockPresenceBackend()
        r = backend.check("car", packet={"objects": ["car", "dog"]})
        assert r.object_name == "car"
        assert r.present is True
        assert r.backend == "mock"
        assert 0.0 <= r.confidence <= 1.0
        assert r.confidence > 0.5

    def test_absent_object_low_confidence(self):
        from sgdjscc_lab.evaluators.presence_backends import MockPresenceBackend
        backend = MockPresenceBackend()
        r = backend.check("cat", packet={"objects": ["car", "dog"]})
        assert r.present is False
        assert r.confidence < 0.5

    def test_case_insensitive(self):
        from sgdjscc_lab.evaluators.presence_backends import MockPresenceBackend
        r = MockPresenceBackend().check("Car", packet={"objects": ["car"]})
        assert r.present is True

    def test_result_to_dict_json_serialisable(self):
        import json
        from sgdjscc_lab.evaluators.presence_backends import MockPresenceBackend
        r = MockPresenceBackend().check("car", packet={"objects": ["car"]})
        json.dumps(r.to_dict())
        assert set(r.to_dict().keys()) == {"object_name", "present", "confidence", "backend", "evidence"}

    def test_no_image_needed(self):
        from sgdjscc_lab.evaluators.presence_backends import MockPresenceBackend
        # Must not raise even though image=None (mock never needs pixels).
        r = MockPresenceBackend().check("car", image=None, packet={"objects": ["car"]})
        assert r.present is True


# ─────────────────────────────────────────────────────────────────────────────
# Image-based backends require an image (Rx-legal / interface contract check)
# ─────────────────────────────────────────────────────────────────────────────

class TestImageRequiredBackends:
    def test_clip_backend_requires_image(self):
        from sgdjscc_lab.evaluators.presence_backends import ClipPresenceBackend, PresenceBackendUnavailableError
        with pytest.raises(PresenceBackendUnavailableError):
            ClipPresenceBackend().check("car", image=None)

    def test_vqa_backend_without_vqa_fn_unavailable(self):
        from sgdjscc_lab.evaluators.presence_backends import VqaPresenceBackend, PresenceBackendUnavailableError
        with pytest.raises(PresenceBackendUnavailableError):
            VqaPresenceBackend(vqa_fn=None).check("car", image=torch.rand(1, 3, 4, 4))

    def test_vqa_backend_with_mock_vqa_fn(self):
        from sgdjscc_lab.evaluators.presence_backends import VqaPresenceBackend

        def fake_vqa(image, question):
            return "yes" if "car" in question else "no"

        backend = VqaPresenceBackend(vqa_fn=fake_vqa)
        r = backend.check("car", image=torch.rand(1, 3, 4, 4))
        assert r.present is True
        assert r.backend == "vqa"
        r2 = backend.check("cat", image=torch.rand(1, 3, 4, 4))
        assert r2.present is False

    def test_owlv2_backend_unavailable_when_transformers_lacks_owlv2(self):
        # Not asserting transformers is missing — just that a bad/unreachable
        # model_id degrades to PresenceBackendUnavailableError, never a crash
        # or a silent wrong answer.
        from sgdjscc_lab.evaluators.presence_backends import Owlv2PresenceBackend, PresenceBackendUnavailableError
        backend = Owlv2PresenceBackend(model_id="this-model-does-not-exist/owlv2-fake")
        with pytest.raises(PresenceBackendUnavailableError):
            backend.check("car", image=torch.rand(1, 3, 8, 8))

    def test_gt_backend_missing_annotation_unavailable(self):
        from sgdjscc_lab.evaluators.presence_backends import GtPresenceBackend, PresenceBackendUnavailableError
        backend = GtPresenceBackend(gt_metadata={"car": True})
        with pytest.raises(PresenceBackendUnavailableError):
            backend.check("dog")

    def test_gt_backend_with_annotation(self):
        from sgdjscc_lab.evaluators.presence_backends import GtPresenceBackend
        backend = GtPresenceBackend(gt_metadata={"car": True, "dog": {"present": False}})
        assert backend.check("car").present is True
        assert backend.check("dog").present is False
        assert backend.check("dog").confidence == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# build_presence_backend registry
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPresenceBackend:
    def test_builds_mock(self):
        from sgdjscc_lab.evaluators.presence_backends import build_presence_backend, MockPresenceBackend
        assert isinstance(build_presence_backend("mock"), MockPresenceBackend)

    def test_builds_clip(self):
        from sgdjscc_lab.evaluators.presence_backends import build_presence_backend, ClipPresenceBackend
        backend = build_presence_backend("clip", threshold=0.4)
        assert isinstance(backend, ClipPresenceBackend)
        assert backend.threshold == pytest.approx(0.4)

    def test_unknown_backend_raises(self):
        from sgdjscc_lab.evaluators.presence_backends import build_presence_backend
        with pytest.raises(NotImplementedError):
            build_presence_backend("not_a_real_backend")


# ─────────────────────────────────────────────────────────────────────────────
# PresenceCalibrator
# ─────────────────────────────────────────────────────────────────────────────

class _FixedBackend:
    """Test-only stub: always returns the same present/confidence regardless
    of packet contents — used to prove the calibrator can actually disagree
    with what a packet says (MockPresenceBackend, by design, never can)."""

    def __init__(self, name, present, confidence):
        self.backend_name = name
        self._present = present
        self._confidence = confidence

    def check(self, object_name, image=None, packet=None, gt_metadata=None):
        from sgdjscc_lab.evaluators.presence_backends import PresenceResult
        return PresenceResult(object_name=object_name, present=self._present,
                               confidence=self._confidence, backend=self.backend_name)


class TestPresenceCalibrator:
    def test_clip_only_mode_uses_clip_backend(self):
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator
        stub = _FixedBackend("clip", True, 0.8)
        cal = PresenceCalibrator({"clip": stub}, mode="clip_only")
        out = cal.calibrate("car")
        assert out.final_present is True
        assert out.final_confidence == pytest.approx(0.8)
        assert out.contributing_backends == ["clip"]

    def test_only_mode_missing_backend_raises(self):
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator
        from sgdjscc_lab.evaluators.presence_backends import PresenceBackendUnavailableError
        cal = PresenceCalibrator({}, mode="clip_only")
        with pytest.raises(PresenceBackendUnavailableError):
            cal.calibrate("car")

    def test_ensemble_majority_agrees(self):
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator
        backends = {
            "a": _FixedBackend("a", True, 0.9),
            "b": _FixedBackend("b", True, 0.6),
            "c": _FixedBackend("c", False, 0.2),
        }
        cal = PresenceCalibrator(backends, mode="ensemble_majority")
        out = cal.calibrate("car")
        assert out.final_present is True   # 2/3 vote present
        assert set(out.contributing_backends) == {"a", "b", "c"}

    def test_ensemble_majority_tie_is_false(self):
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator
        backends = {"a": _FixedBackend("a", True, 0.9), "b": _FixedBackend("b", False, 0.1)}
        cal = PresenceCalibrator(backends, mode="ensemble_majority")
        assert cal.calibrate("car").final_present is False   # 1/2 is not a strict majority

    def test_ensemble_weighted_respects_weights(self):
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator
        backends = {"strong": _FixedBackend("strong", True, 0.9), "weak": _FixedBackend("weak", False, 0.0)}
        cal = PresenceCalibrator(
            backends, mode="ensemble_weighted", weights={"strong": 5.0, "weak": 1.0}, threshold=0.5,
        )
        out = cal.calibrate("car")
        # weighted mean = (5*0.9 + 1*0.0) / 6 = 0.75 >= threshold 0.5
        assert out.final_confidence == pytest.approx(0.75)
        assert out.final_present is True

    def test_ensemble_weighted_below_threshold_is_absent(self):
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator
        backends = {"a": _FixedBackend("a", True, 0.2)}
        cal = PresenceCalibrator(backends, mode="ensemble_weighted", threshold=0.5)
        out = cal.calibrate("car")
        assert out.final_confidence == pytest.approx(0.2)
        assert out.final_present is False

    def test_unavailable_backend_skipped_in_ensemble(self):
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator
        from sgdjscc_lab.evaluators.presence_backends import GtPresenceBackend, MockPresenceBackend
        backends = {
            "gt": GtPresenceBackend(gt_metadata={}),   # will raise Unavailable (no annotation)
            "mock": MockPresenceBackend(),
        }
        cal = PresenceCalibrator(backends, mode="ensemble_majority")
        out = cal.calibrate("car", packet={"objects": ["car"]})
        assert out.contributing_backends == ["mock"]   # gt silently skipped, not crashed

    def test_invalid_mode_rejected(self):
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator
        with pytest.raises(ValueError):
            PresenceCalibrator({}, mode="not_a_real_mode")

    def test_calibrated_presence_to_dict_json_serialisable(self):
        import json
        from sgdjscc_lab.evaluators.presence_calibration import PresenceCalibrator
        cal = PresenceCalibrator({"clip": _FixedBackend("clip", True, 0.7)}, mode="clip_only")
        json.dumps(cal.calibrate("car").to_dict())


# ─────────────────────────────────────────────────────────────────────────────
# build_presence_calibrator (config-driven)
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPresenceCalibrator:
    def test_disabled_by_default(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.evaluators.presence_calibration import build_presence_calibrator
        assert build_presence_calibrator(OmegaConf.create({})) is None

    def test_enabled_builds_configured_backends(self):
        from omegaconf import OmegaConf
        from sgdjscc_lab.evaluators.presence_calibration import build_presence_calibrator
        cfg = OmegaConf.create({
            "verifier": {
                "use_presence_calibration": True,
                "presence_mode": "ensemble_majority",
                "presence_backends": ["mock"],
            }
        })
        cal = build_presence_calibrator(cfg)
        assert cal is not None
        assert cal.mode == "ensemble_majority"
        assert "mock" in cal.backends

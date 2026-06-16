"""tests/test_vqa_backend.py – Phase 5-C local VQA backend tests (offline)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ─────────────────────────────────────────────────────────────────────────────
# build_vqa_backend
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildVQABackend:
    def test_mock_backend_rules(self):
        from sgdjscc_lab.evaluators.vqa_backend import build_vqa_backend
        fn = build_vqa_backend({"type": "mock", "rules": {"dog": "yes"}, "mock_answer": "no"})
        assert fn is not None
        img = torch.rand(1, 3, 8, 8)
        assert fn(img, "Is there a dog in the image?") == "yes"
        assert fn(img, "Is there a cat in the image?") == "no"

    def test_type_none_returns_none(self):
        from sgdjscc_lab.evaluators.vqa_backend import build_vqa_backend
        assert build_vqa_backend({"type": "none"}) is None
        assert build_vqa_backend(None) is None

    def test_unknown_type_returns_none(self):
        from sgdjscc_lab.evaluators.vqa_backend import build_vqa_backend
        assert build_vqa_backend({"type": "does_not_exist"}) is None

    def test_unavailable_backend_degrades(self, monkeypatch):
        # Simulate transformers being unavailable → builder returns None (fallback).
        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *a, **k):
            if name == "transformers" or name.startswith("transformers."):
                raise ImportError("no transformers")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        from sgdjscc_lab.evaluators.vqa_backend import build_vqa_backend
        assert build_vqa_backend({"type": "blip2"}) is None


# ─────────────────────────────────────────────────────────────────────────────
# VQAHallucinationEvaluator.from_config
# ─────────────────────────────────────────────────────────────────────────────

class TestFromConfig:
    def test_mock_backend_wired(self):
        from sgdjscc_lab.evaluators.hallucination_vqa import VQAHallucinationEvaluator
        ev = VQAHallucinationEvaluator.from_config(
            vqa_backend_cfg={"type": "mock", "rules": {"dog": "yes"}})
        # 'dog' confirmed only in recon (mock says yes for any 'dog' question on both
        # images → in_orig True → NOT hallucinated). Use a recon-only rule instead:
        assert ev.vqa_fn is not None

    def test_no_backend_falls_back_to_clip(self):
        from sgdjscc_lab.evaluators.hallucination_vqa import VQAHallucinationEvaluator

        class _MockFallback:
            def evaluate(self, o, r):
                return {"hallucination_score": 0.0, "extra_objects": []}

        ev = VQAHallucinationEvaluator.from_config(vqa_backend_cfg={"type": "none"})
        assert ev.vqa_fn is None
        ev._fallback = _MockFallback()
        out = ev.evaluate(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8))
        assert out["method"] == "clip_fallback"

    def test_runtime_error_degrades_to_clip(self):
        from sgdjscc_lab.evaluators.hallucination_vqa import VQAHallucinationEvaluator

        def _boom(image, q):
            raise RuntimeError("backend exploded")

        class _MockFallback:
            def evaluate(self, o, r):
                return {"hallucination_score": 0.0, "extra_objects": []}

        ev = VQAHallucinationEvaluator(vqa_fn=_boom)
        ev._fallback = _MockFallback()
        out = ev.evaluate(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8), objects=["dog"])
        assert out["method"] == "vqa_error_fallback"

    def test_failure_disables_vqa_no_retry(self):
        """Regression: after the first backend failure the VQA path is disabled for
        the run — no repeated heavy retries / warnings on every object."""
        from sgdjscc_lab.evaluators.hallucination_vqa import VQAHallucinationEvaluator

        calls = {"n": 0}

        def _boom(image, q):
            calls["n"] += 1
            raise RuntimeError("ModelWrapper boom")

        class _MockFallback:
            def evaluate(self, o, r):
                return {"hallucination_score": 0.0, "extra_objects": []}

        ev = VQAHallucinationEvaluator(vqa_fn=_boom)
        ev._fallback = _MockFallback()
        o, r = torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)
        first = ev.evaluate(o, r, objects=["dog"])
        assert first["method"] == "vqa_error_fallback"
        assert ev.vqa_fn is None                      # disabled after first failure
        # Subsequent calls go straight to CLIP fallback without touching the backend.
        second = ev.evaluate(o, r, objects=["cat"])
        third = ev.evaluate(o, r, objects=["car"])
        assert second["method"] == "clip_fallback"
        assert third["method"] == "clip_fallback"
        assert calls["n"] == 1                         # backend attempted exactly once

    def test_blip2_default_model_is_coco(self):
        from sgdjscc_lab.evaluators.vqa_backend import Blip2VQABackend
        assert Blip2VQABackend().model_id == "Salesforce/blip2-opt-2.7b-coco"

    def test_blip2_load_failure_is_cached(self):
        """A failed load is cached so answer() does not re-attempt it every call."""
        import sys
        import types
        from sgdjscc_lab.evaluators.vqa_backend import Blip2VQABackend

        attempts = {"n": 0}
        fake = types.ModuleType("transformers")

        class _AP:
            @staticmethod
            def from_pretrained(*a, **k):
                attempts["n"] += 1
                raise RuntimeError("cannot parse checkpoint")

        fake.AutoProcessor = _AP
        fake.Blip2ForConditionalGeneration = _AP
        old = sys.modules.get("transformers")
        sys.modules["transformers"] = fake
        try:
            be = Blip2VQABackend(device="cpu")
            for _ in range(3):
                with pytest.raises(Exception):
                    be.answer(torch.rand(1, 3, 8, 8), "Is there a dog?")
            assert be._failed is True
            assert attempts["n"] == 1                  # heavy load attempted once
        finally:
            if old is not None:
                sys.modules["transformers"] = old
            else:
                sys.modules.pop("transformers", None)


# ─────────────────────────────────────────────────────────────────────────────
# CLIP model cache: shared (model_name, device) → one loaded instance / one log
# ─────────────────────────────────────────────────────────────────────────────

class TestClipModelCache:
    def test_two_evaluators_share_cached_model(self):
        import sys
        import types
        from sgdjscc_lab.evaluators import clip_score

        loads = {"n": 0}
        fake = types.ModuleType("clip")

        class _FakeModel:
            def eval(self_inner):
                return self_inner

        def _load(name, device=None):
            loads["n"] += 1
            return _FakeModel(), (lambda img: img)

        fake.load = _load
        key = ("TEST/dummy-vit", "cpu")
        clip_score._CLIP_MODEL_CACHE.pop(key, None)
        old = sys.modules.get("clip")
        sys.modules["clip"] = fake
        try:
            a = clip_score.CLIPScoreEvaluator(model_name="TEST/dummy-vit", device=None)
            b = clip_score.CLIPScoreEvaluator(model_name="TEST/dummy-vit", device=None)
            a._load()
            b._load()
            # clip.load called once; both evaluators share the same underlying model.
            assert loads["n"] == 1
            assert a._model is b._model
        finally:
            clip_score._CLIP_MODEL_CACHE.pop(key, None)
            if old is not None:
                sys.modules["clip"] = old
            else:
                sys.modules.pop("clip", None)


# ─────────────────────────────────────────────────────────────────────────────
# EvalContext wiring: SRS-v2 actually uses the VQA layer
# ─────────────────────────────────────────────────────────────────────────────

class TestEvalContextVQAWiring:
    def test_srs_v2_uses_vqa_backend(self):
        from sgdjscc_lab.pipelines.eval_pipeline import EvalContext

        # Image-dependent mock VQA: object present only in the high-mean image.
        def vqa_fn(image, question):
            return "yes" if float(image.mean()) > 0.5 else "no"

        ctx = EvalContext(
            enabled_metrics={"semantic_reliability_score"},
            use_srs_v2=True, use_vqa_hallucination=True, vqa_fn=vqa_fn,
        )
        srs_v2 = ctx._get_srs_v2()
        # The SRS-v2 evaluator must carry a VQA evaluator that uses our backend.
        assert srs_v2._vqa is not None
        assert srs_v2._vqa.vqa_fn is vqa_fn

        from sgdjscc_lab.guidance.semantic_packet_extractor import build_packet
        original = torch.zeros(1, 3, 8, 8)        # mean 0 → object absent
        recon = torch.ones(1, 3, 8, 8)            # mean 1 → object present (hallucination)
        out = srs_v2.evaluate(
            original, recon,
            base_result={"srs_base": 0.9},
            recon_packet=build_packet(objects=["dog"], scene="s"),
        )
        # VQA detected a hallucinated 'dog' → hallucination layer > 0.
        assert out.get("hallucination_score", 0.0) > 0.0

    def test_vqa_backend_cfg_built_lazily(self):
        from sgdjscc_lab.pipelines.eval_pipeline import EvalContext
        ctx = EvalContext(
            enabled_metrics={"semantic_reliability_score"},
            use_srs_v2=True, use_vqa_hallucination=True,
            vqa_backend_cfg={"type": "mock", "rules": {"dog": "yes"}},
        )
        srs_v2 = ctx._get_srs_v2()
        assert srs_v2._vqa is not None
        assert srs_v2._vqa.vqa_fn is not None      # mock backend wired from cfg

"""tests/test_regeneration_search.py – Phase 5-C regeneration-search tests (offline)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ─────────────────────────────────────────────────────────────────────────────
# AdaptiveSearchPolicy ordering
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveSearchPolicy:
    def test_missing_object_prioritises_strong_text(self):
        from sgdjscc_lab.controllers.adaptive_search_policy import AdaptiveSearchPolicy
        order = AdaptiveSearchPolicy().order(error_report={"missing_object_count": 2})
        assert order[0] == "strong_text_weak_edge"

    def test_hallucination_prioritises_weak_text(self):
        from sgdjscc_lab.controllers.adaptive_search_policy import AdaptiveSearchPolicy
        order = AdaptiveSearchPolicy().order(
            error_report={"additional_object_count": 1}, hallucination_score=0.3)
        assert "weak_text_strong_edge" in order
        assert "unconditional" in order

    def test_blind_channel_first(self):
        from sgdjscc_lab.controllers.adaptive_search_policy import AdaptiveSearchPolicy
        order = AdaptiveSearchPolicy().order(channel_state={"csi": "none"})
        assert order[0] == "channel_conditioned_retry"

    def test_default_when_no_signal(self):
        from sgdjscc_lab.controllers.adaptive_search_policy import AdaptiveSearchPolicy
        order = AdaptiveSearchPolicy().order()
        assert order == ["strong_text_weak_edge", "weak_text_strong_edge", "unconditional"]

    def test_deduplicated_and_capped(self):
        from sgdjscc_lab.controllers.adaptive_search_policy import AdaptiveSearchPolicy
        order = AdaptiveSearchPolicy().order(
            error_report={"additional_object_count": 1, "relation_error_count": 1},
            hallucination_score=0.5, max_strategies=2)
        assert len(order) == 2
        assert len(set(order)) == len(order)     # no duplicates


# ─────────────────────────────────────────────────────────────────────────────
# apply_search_strategy
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyStrategy:
    def _cfg(self):
        return OmegaConf.create({"guidance_scale": 4.0, "controlnet_scale": 0.3, "use_text": True})

    def test_strong_text_weak_edge(self):
        from sgdjscc_lab.evaluators.regeneration_search import apply_search_strategy, SEARCH_STRATEGIES
        out = apply_search_strategy(self._cfg(), SEARCH_STRATEGIES["strong_text_weak_edge"])
        assert out.guidance_scale > 4.0
        assert out.controlnet_scale < 0.3
        assert out.use_text is True

    def test_unconditional_disables_text(self):
        from sgdjscc_lab.evaluators.regeneration_search import apply_search_strategy, SEARCH_STRATEGIES
        out = apply_search_strategy(self._cfg(), SEARCH_STRATEGIES["unconditional"])
        assert out.use_text is False

    def test_channel_retry_sets_flag(self):
        from sgdjscc_lab.evaluators.regeneration_search import apply_search_strategy, SEARCH_STRATEGIES
        out = apply_search_strategy(self._cfg(), SEARCH_STRATEGIES["channel_conditioned_retry"])
        assert out.get("use_channel_conditioning") is True

    def test_does_not_mutate_input(self):
        from sgdjscc_lab.evaluators.regeneration_search import apply_search_strategy, SEARCH_STRATEGIES
        cfg = self._cfg()
        apply_search_strategy(cfg, SEARCH_STRATEGIES["strong_text_weak_edge"])
        assert cfg.guidance_scale == 4.0


# ─────────────────────────────────────────────────────────────────────────────
# RegenerationSearch selects best verified output
# ─────────────────────────────────────────────────────────────────────────────

class TestRegenerationSearch:
    def test_picks_highest_verified_score(self):
        from sgdjscc_lab.evaluators.regeneration_search import RegenerationSearch

        # reconstruct_fn returns the cfg's guidance_scale; verify_fn rewards it.
        rs = RegenerationSearch(
            reconstruct_fn=lambda cfg: float(cfg.guidance_scale),
            verify_fn=lambda recon: recon,    # higher guidance → higher score
        )
        base = OmegaConf.create({"guidance_scale": 4.0, "controlnet_scale": 0.3, "use_text": True})
        out = rs.search(base, error_report={"missing_object_count": 1})
        # strong_text_weak_edge multiplies guidance by 1.5 → 6.0, the max.
        assert out["best_strategy"] == "strong_text_weak_edge"
        assert out["best_score"] == pytest.approx(6.0)

    def test_initial_candidate_can_win(self):
        from sgdjscc_lab.evaluators.regeneration_search import RegenerationSearch
        rs = RegenerationSearch(
            reconstruct_fn=lambda cfg: 0.0,    # all retries score 0
            verify_fn=lambda recon: recon,
        )
        base = OmegaConf.create({"guidance_scale": 4.0, "controlnet_scale": 0.3, "use_text": True})
        out = rs.search(base, order=["unconditional"], initial_recon="orig", initial_score=0.9)
        assert out["best_strategy"] == "none"
        assert out["best_recon"] == "orig"

    def test_history_records_all_attempts(self):
        from sgdjscc_lab.evaluators.regeneration_search import RegenerationSearch
        rs = RegenerationSearch(reconstruct_fn=lambda cfg: 1.0, verify_fn=lambda r: r)
        base = OmegaConf.create({"guidance_scale": 4.0, "controlnet_scale": 0.3, "use_text": True})
        out = rs.search(base, order=["strong_text_weak_edge", "unconditional"])
        assert [h["strategy"] for h in out["history"]] == ["strong_text_weak_edge", "unconditional"]


# ─────────────────────────────────────────────────────────────────────────────
# _run_regeneration_search honours the configured verify_metric (srs vs srs_v2)
# ─────────────────────────────────────────────────────────────────────────────

class TestEvalSearchVerifyMetric:
    def _ctx(self):
        import torch  # noqa: F401
        from sgdjscc_lab.pipelines.eval_pipeline import EvalContext

        class _MockBase:
            # base SRS ranks OPPOSITE to guidance (so a wrong objective is obvious)
            def evaluate(self, original, reconstructed, **kw):
                m = float(reconstructed.mean())
                return {"semantic_reliability_score": -m, "srs_base": -m,
                        "hallucination_score": 0.0, "clip_image_image": 0.5}

        class _MockV2:
            # srs_v2 ranks WITH guidance (higher guidance → higher srs_v2)
            def evaluate(self, original, reconstructed, base_result=None, **kw):
                return {"srs_v2": float(reconstructed.mean())}

        ctx = EvalContext(enabled_metrics={"semantic_reliability_score"})
        ctx.srs_evaluator = _MockBase()
        ctx.srs_v2_evaluator = _MockV2()
        return ctx

    def test_srs_v2_objective_selects_high_guidance(self, monkeypatch):
        import torch
        from omegaconf import OmegaConf
        import sgdjscc_lab.pipelines.eval_pipeline as ep

        # Reconstruction encodes the guidance scale as the tensor mean.
        def _fake_recon(original, models, cfg):
            return torch.full((1, 3, 8, 8), float(cfg.guidance_scale))

        monkeypatch.setattr(ep, "_reconstruct_with_cfg", _fake_recon)

        cfg = OmegaConf.create({
            "guidance_scale": 4.0, "controlnet_scale": 0.3, "use_text": True,
            "csi": "perfect",
            "regeneration_search": {"verify_metric": "srs_v2", "max_strategies": 3},
        })
        ctx = self._ctx()
        original = torch.zeros(1, 3, 8, 8)
        reconstructed = torch.full((1, 3, 8, 8), 4.0)
        row = {"semantic_reliability_score": -4.0,
               "_error_report": {"missing_object_count": 1, "additional_object_count": 1},
               "hallucination_score": 0.5}

        best, _pkt, new_row = ep._run_regeneration_search(
            original, reconstructed, None, None, row,
            eval_cfg=cfg, models=object(), eval_ctx=ctx, packet_extractor=None,
            filename="x.png", snr_db=10.0, text_list=None, cfg=cfg,
        )
        # strong_text_weak_edge has the highest guidance (1.5×) → highest srs_v2.
        # Under the (opposite) base metric it would be the worst, so this proves
        # the srs_v2 objective was actually used.
        assert new_row["regeneration_strategy"] == "strong_text_weak_edge"
        assert float(best.mean()) == pytest.approx(6.0)

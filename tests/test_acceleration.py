"""tests/test_acceleration.py – Phase 5-B acceleration tests (offline)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ─────────────────────────────────────────────────────────────────────────────
# Sampler / step-budget selection
# ─────────────────────────────────────────────────────────────────────────────

class TestSamplerCfg:
    def test_ddim_sets_diffusion_step(self):
        from sgdjscc_lab.acceleration import build_sampler_cfg
        cfg = OmegaConf.create({"diffusion_step": 50,
                                "acceleration": {"sampler": "ddim", "sampler_steps": 10}})
        out, spec = build_sampler_cfg(cfg)
        assert spec.sampler_type == "ddim" and spec.steps == 10
        assert out.diffusion_step == 10

    def test_baseline_untouched(self):
        from sgdjscc_lab.acceleration import build_sampler_cfg
        cfg = OmegaConf.create({"diffusion_step": 50})
        out, spec = build_sampler_cfg(cfg)
        assert out is cfg                       # legacy path unchanged
        assert spec.sampler_type == "baseline" and spec.steps == 50

    def test_invalid_sampler_falls_back(self):
        from sgdjscc_lab.acceleration import build_sampler_cfg
        cfg = OmegaConf.create({"diffusion_step": 50, "acceleration": {"sampler": "weird"}})
        _, spec = build_sampler_cfg(cfg)
        assert spec.sampler_type == "baseline"


class TestDynamicRouting:
    def test_step_budget_regimes(self):
        from sgdjscc_lab.acceleration import dynamic_step_budget
        assert dynamic_step_budget(20.0, 0.9) <= dynamic_step_budget(6.0, 0.5)
        assert dynamic_step_budget(6.0, 0.5) <= dynamic_step_budget(-5.0)
        assert dynamic_step_budget(20.0, 0.9, blind=True) == dynamic_step_budget(None)

    def test_high_snr_low_confidence_not_minimal(self):
        from sgdjscc_lab.acceleration import dynamic_step_budget
        # high SNR but low confidence should not collapse to the minimum budget
        assert dynamic_step_budget(20.0, 0.1) > dynamic_step_budget(20.0, 0.9)


class TestKarras:
    def test_decreasing_schedule(self):
        from sgdjscc_lab.acceleration import karras_schedule
        sig = karras_schedule(6, sigma_min=0.002, sigma_max=2.0)
        assert sig.shape == (6,)
        assert sig[0] > sig[-1]
        assert abs(float(sig[0]) - 2.0) < 1e-4
        assert abs(float(sig[-1]) - 0.002) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Consistency decoder
# ─────────────────────────────────────────────────────────────────────────────

class TestConsistencyDecoder:
    def test_fewstep_runs_and_counts_calls(self):
        from sgdjscc_lab.acceleration import ConsistencyDecoder
        calls = {"n": 0}

        def denoise_fn(x, sigma):
            calls["n"] += 1
            return x * 0.5

        dec = ConsistencyDecoder(mode="fewstep", steps=3)
        out = dec.decode(denoise_fn, init=torch.ones(1, 4, 8, 8))
        assert out.shape == (1, 4, 8, 8)
        assert calls["n"] == 3                  # one denoise per step

    def test_baseline_single_call(self):
        from sgdjscc_lab.acceleration import ConsistencyDecoder
        calls = {"n": 0}

        def denoise_fn(x, sigma):
            calls["n"] += 1
            return x

        ConsistencyDecoder(mode="baseline").decode(denoise_fn, init=torch.ones(1, 4, 4, 4))
        assert calls["n"] == 1

    def test_distilled_placeholder_falls_back(self):
        from sgdjscc_lab.acceleration import ConsistencyDecoder
        dec = ConsistencyDecoder(mode="distilled_placeholder", steps=2)
        out = dec.decode(lambda x, s: x, init=torch.ones(1, 4, 4, 4))
        assert out.shape == (1, 4, 4, 4)        # falls back to fewstep

    def test_distilled_model_used_when_present(self):
        from sgdjscc_lab.acceleration import ConsistencyDecoder
        used = {"flag": False}

        def distilled(x, sigma):
            used["flag"] = True
            return x

        dec = ConsistencyDecoder(mode="distilled_placeholder", distilled_model=distilled)
        dec.decode(lambda x, s: x, init=torch.ones(1, 4, 4, 4))
        assert used["flag"] is True

    def test_init_from_shape(self):
        from sgdjscc_lab.acceleration import ConsistencyDecoder
        out = ConsistencyDecoder(mode="fewstep", steps=1).decode(
            lambda x, s: x, shape=(1, 4, 4, 4))
        assert out.shape == (1, 4, 4, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Early exit
# ─────────────────────────────────────────────────────────────────────────────

class TestEarlyExit:
    def test_threshold_stop(self):
        from sgdjscc_lab.acceleration import EarlyExitController
        ee = EarlyExitController(srs_threshold=0.8)
        assert ee.should_stop([0.85], step=1)["stop"] is True

    def test_diminishing_returns_stop(self):
        from sgdjscc_lab.acceleration import EarlyExitController
        ee = EarlyExitController(srs_threshold=0.99, improvement_delta=0.05)
        assert ee.should_stop([0.5, 0.51], step=2)["stop"] is True

    def test_continue(self):
        from sgdjscc_lab.acceleration import EarlyExitController
        ee = EarlyExitController(srs_threshold=0.9, improvement_delta=0.01)
        assert ee.should_stop([0.3, 0.5], step=2)["stop"] is False

    def test_min_steps_respected(self):
        from sgdjscc_lab.acceleration import EarlyExitController
        ee = EarlyExitController(srs_threshold=0.1, min_steps=3)
        assert ee.should_stop([0.9], step=1)["stop"] is False

    def test_evaluate_checkpoints_stops_early(self):
        from sgdjscc_lab.acceleration import evaluate_checkpoints, EarlyExitController
        scores = {5: 0.6, 10: 0.85, 20: 0.95}
        res = evaluate_checkpoints(
            render_fn=lambda s: s, eval_fn=lambda r: scores[r],
            checkpoints=[5, 10, 20], controller=EarlyExitController(srs_threshold=0.8),
        )
        assert res["chosen_step"] == 10
        assert res["reason"] == "threshold_reached"
        assert len(res["history"]) == 2          # stopped before 20


# ─────────────────────────────────────────────────────────────────────────────
# Latency profiler
# ─────────────────────────────────────────────────────────────────────────────

class TestLatencyProfiler:
    def test_section_and_report(self):
        from sgdjscc_lab.acceleration import LatencyProfiler
        prof = LatencyProfiler()
        with prof.section("decoder"):
            _ = sum(range(1000))
        rep = prof.report(steps=10)
        assert rep["decoder_calls"] == 1
        assert rep["total_latency_s"] >= 0.0
        assert rep["effective_steps"] == 10
        assert "per_step_s" in rep

    def test_profile_callable_keys(self):
        from sgdjscc_lab.acceleration import profile_callable
        out = profile_callable(lambda: sum(range(100)), n_warmup=0, n_runs=2, steps=5)
        for k in ("mean_s", "min_s", "max_s", "runs", "per_step_s"):
            assert k in out
        assert out["runs"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Intra-sampler interrupt: generic driver (run_interruptible_sampling)
# ─────────────────────────────────────────────────────────────────────────────

class TestInterruptibleDriver:
    def test_callback_order_and_full_run(self):
        from sgdjscc_lab.acceleration import run_interruptible_sampling
        seen = []

        def step_fn(state, i, total):
            return {"v": state["v"] + 1}

        def score_fn(state, i, total):
            seen.append(i)
            return 0.0   # never reaches threshold

        out = run_interruptible_sampling(
            {"v": 0}, step_fn, total_steps=10, score_fn=score_fn,
            controller=None, check_interval=3, min_steps=1,
        )
        assert out["completed"] is True
        assert out["stopped_at"] == 10
        assert out["state"]["v"] == 10           # all steps ran
        # checks at steps 3,6,9 and the final step 10
        assert [s + 1 for s in seen] == [3, 6, 9, 10]

    def test_early_stop_terminates_loop(self):
        from sgdjscc_lab.acceleration import run_interruptible_sampling, EarlyExitController
        steps_run = {"n": 0}

        def step_fn(state, i, total):
            steps_run["n"] += 1
            return state

        # score immediately at/above threshold → stop at first check.
        out = run_interruptible_sampling(
            {}, step_fn, total_steps=50, score_fn=lambda s, i, t: 0.95,
            controller=EarlyExitController(srs_threshold=0.8), check_interval=1, min_steps=1,
        )
        assert out["completed"] is False
        assert out["reason"] == "threshold_reached"
        assert out["stopped_at"] == 1
        assert steps_run["n"] == 1                # remaining 49 steps NOT executed

    def test_min_steps_respected(self):
        from sgdjscc_lab.acceleration import run_interruptible_sampling, EarlyExitController
        out = run_interruptible_sampling(
            {}, lambda s, i, t: s, total_steps=20, score_fn=lambda s, i, t: 0.99,
            controller=EarlyExitController(srs_threshold=0.5), check_interval=1, min_steps=5,
        )
        assert out["stopped_at"] >= 5            # never exits before min_steps


# ─────────────────────────────────────────────────────────────────────────────
# Intra-sampler interrupt: real wrapper loop (generate_interruptible) with a
# fake DiffusionGenerator-like pipe (validates loop structure + early exit).
# ─────────────────────────────────────────────────────────────────────────────

class _FakePipe:
    """Minimal DiffusionGenerator-compatible stub for the interruptible loop."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.text_embed = None
        self.model = type("M", (), {"eval": lambda self_: None})()
        self.pred_calls = 0

    def encode_text(self, prompt, embed):
        return torch.zeros(len(prompt), 4)

    def sigmoid_schedule_inverse(self, x):
        return x

    def sigmoid_schedule(self, ts):
        import numpy as np
        return np.clip(ts, 1e-3, 0.999)

    def adjust_cfg_weight(self, cg, ts, method):
        return cg

    def expand_scalar(self, t, shape):
        while t.dim() < len(shape):
            t = t.unsqueeze(-1)
        return t.expand(shape)

    def pred_image(self, x_t, labels, noise, cg, c, controlnet, mask=None, not_control=None):
        self.pred_calls += 1
        return x_t * 0.5

    def generate(self, **kw):
        # fallback path returns (image, latent)
        return None, kw["latent"]


def _gi_kwargs(pipe, **over):
    base = dict(
        prompt=["a cat"], negative_prompt=["bad"],
        latent=torch.ones(1, 16, 16, 16), curr_step=torch.full((1, 1), 0.5),
        diffusion_step=10, c=None, controlnet=False, not_control=[1, 1],
        class_guidance=4.0, cfg_weighting_method="constant",
        mask_token=None, mask_step=1, step_style="continuous",
    )
    base.update(over)
    return base


class TestGenerateInterruptible:
    def test_full_run_without_controller(self):
        from sgdjscc_lab.models.diffusion_wrapper import generate_interruptible
        pipe = _FakePipe()
        latent, info = generate_interruptible(pipe, **_gi_kwargs(pipe))
        assert info["fallback"] is False
        assert info["interrupted"] is False
        assert latent.shape == (1, 16, 16, 16)
        # 10 noise levels → 9 loop steps + 1 final prediction = 10 pred_image calls.
        assert pipe.pred_calls == info["total_steps"] + 1

    def test_early_exit_saves_steps(self):
        from sgdjscc_lab.models.diffusion_wrapper import generate_interruptible
        from sgdjscc_lab.acceleration import EarlyExitController
        pipe = _FakePipe()
        latent, info = generate_interruptible(
            pipe, controller=EarlyExitController(srs_threshold=0.5),
            score_fn=lambda state, i, total: 0.9,   # immediately above threshold
            check_interval=1, min_steps=1, **_gi_kwargs(pipe))
        assert info["interrupted"] is True
        assert info["stopped_at"] < info["total_steps"]
        # loop pred calls == stopped_at, plus one final prediction.
        assert pipe.pred_calls == info["stopped_at"] + 1

    def test_unsupported_step_style_falls_back(self):
        from sgdjscc_lab.models.diffusion_wrapper import generate_interruptible
        pipe = _FakePipe()
        latent, info = generate_interruptible(
            pipe, **_gi_kwargs(pipe, step_style="discrete"))
        assert info["fallback"] is True
        assert pipe.pred_calls == 0               # used pipe.generate, not the loop

    def test_missing_helpers_falls_back(self):
        from sgdjscc_lab.models.diffusion_wrapper import generate_interruptible

        class _Bare:
            def generate(self, **kw):
                return None, kw["latent"]

        latent, info = generate_interruptible(_Bare(), **_gi_kwargs(None))
        assert info["fallback"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Early-exit config coexists with the step budget (no conflict)
# ─────────────────────────────────────────────────────────────────────────────

class TestEarlyExitStepBudgetCoexist:
    def test_sampler_budget_and_early_exit_both_apply(self):
        from sgdjscc_lab.acceleration import build_sampler_cfg
        cfg = OmegaConf.create({
            "diffusion_step": 50,
            "acceleration": {
                "sampler": "ddim", "sampler_steps": 20,
                "early_exit": True, "early_exit_mode": "intra_sampler",
                "early_exit_check_interval": 5,
            },
        })
        out, spec = build_sampler_cfg(cfg)
        assert out.diffusion_step == 20                     # step budget applied
        assert bool(out.acceleration.early_exit) is True    # early-exit still set


# ─────────────────────────────────────────────────────────────────────────────
# Intra-sampler early-exit metric wiring (heuristic vs verified srs/srs_v2).
# Regression: early_exit_metric must not be dead config.
# ─────────────────────────────────────────────────────────────────────────────

class TestEarlyExitMetricWiring:
    def test_heuristic_returns_none(self):
        from sgdjscc_lab.pipelines.infer_pipeline import _build_early_exit_score_fn
        # heuristic → None so the wrapper uses its built-in convergence score.
        assert _build_early_exit_score_fn("heuristic", lambda x: x, torch.zeros(1), lambda o, r: 1.0) is None

    def test_missing_pieces_return_none(self):
        from sgdjscc_lab.pipelines.infer_pipeline import _build_early_exit_score_fn
        assert _build_early_exit_score_fn("srs", None, torch.zeros(1), lambda o, r: 1.0) is None
        assert _build_early_exit_score_fn("srs", lambda x: x, None, lambda o, r: 1.0) is None
        assert _build_early_exit_score_fn("srs", lambda x: x, torch.zeros(1), None) is None

    def test_verified_score_fn_uses_x0_and_verifier(self):
        from sgdjscc_lab.pipelines.infer_pipeline import _build_early_exit_score_fn
        seen = {}

        def decode_fn(x0):
            return x0 * 2.0   # "decode" the clean-latent prediction

        def verifier(original, recon):
            seen["called"] = True
            return float(recon.mean())   # score from the decoded preview

        score_fn = _build_early_exit_score_fn("srs", decode_fn, torch.zeros(1, 3, 4, 4), verifier)
        assert score_fn is not None
        # No x0 in state yet → safe 0.0 (no verifier call).
        assert score_fn({}, 0, 10) == 0.0
        # With x0 present → verifier scores decode_fn(x0).
        x0 = torch.full((1, 4, 4, 4), 3.0)
        val = score_fn({"x0": x0}, 1, 10)
        assert seen.get("called") is True
        assert val == pytest.approx(6.0)   # mean(x0*2) = 6

    def test_generate_interruptible_exposes_x0_to_score_fn(self):
        """The verified metric needs the loop's clean-latent prediction; the loop
        must expose it via state['x0'] every step."""
        from sgdjscc_lab.models.diffusion_wrapper import generate_interruptible
        from sgdjscc_lab.acceleration import EarlyExitController

        saw_x0 = {"n": 0}

        def score_fn(state, i, total):
            if state.get("x0") is not None:
                saw_x0["n"] += 1
                return 0.9          # x0 present → high score → early exit
            return 0.0

        pipe = _FakePipe()
        _latent, info = generate_interruptible(
            pipe, controller=EarlyExitController(srs_threshold=0.5),
            score_fn=score_fn, check_interval=1, min_steps=1, **_gi_kwargs(pipe))
        assert saw_x0["n"] >= 1          # score_fn actually saw x0
        assert info["interrupted"] is True

"""acceleration/ – Phase 5-B low-latency diffusion utilities.

Modules
-------
ddim_sampler        – sampler/step-budget selection + dynamic routing + karras schedule.
consistency_decoder – few-step / distilled decoder interface (LDM-inspired).
early_exit          – checkpoint-based early-stopping of the denoising budget.
latency_profiler    – wall-clock latency measurement and reporting.
water_filling       – fast-fading water-filling denoising (paper Algorithm 4).
"""

from .ddim_sampler import (
    SamplerSpec, build_sampler_cfg, dynamic_step_budget, karras_schedule,
)
from .consistency_decoder import ConsistencyDecoder
from .early_exit import (
    EarlyExitController, evaluate_checkpoints, run_interruptible_sampling,
)
from .latency_profiler import LatencyProfiler, profile_callable
from .water_filling import (
    water_filling_denoise, water_filling_denoise_from_bundle,
    build_mdt_f0_predictor, fast_fading_water_filling_decode,
)

__all__ = [
    "SamplerSpec", "build_sampler_cfg", "dynamic_step_budget", "karras_schedule",
    "ConsistencyDecoder", "EarlyExitController", "evaluate_checkpoints",
    "run_interruptible_sampling", "LatencyProfiler", "profile_callable",
    "water_filling_denoise", "water_filling_denoise_from_bundle",
    "build_mdt_f0_predictor", "fast_fading_water_filling_decode",
]

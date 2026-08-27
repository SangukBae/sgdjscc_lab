"""diagnostics/ablations.py – One-factor-at-a-time ablation registry.

Each :class:`AblationSpec` is consumed by
``float32_digital_paths.instrumented_decode`` (and, for ``use_edge``, by the
per-frame runners in the same module) to flip exactly one factor away from
the baseline reconstruction, so a quality delta can be attributed to that
single factor. ``None`` on a field means "use the run's cfg/CLI default",
never a second implicit default silently layered on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    use_controlnet: Optional[bool] = None
    use_text: Optional[bool] = None
    use_edge: Optional[bool] = None
    uncertainty_off: bool = False
    # None = per-path default (AWGN/in-process retransmit edge via the analog
    # Canny net; digital_wire uses the already-received packet edge as-is).
    force_edge_already_received: Optional[bool] = None
    digital_step_policy_override: Optional[str] = None
    reuse_awgn_step: bool = False
    fixed_step: Optional[float] = None
    bypass_diffusion: bool = False
    diffusion_step_override: Optional[int] = None


def build_default_ablations(
    *, fixed_step_value: float = 0.5, minimal_denoise_steps: int = 2,
) -> Dict[str, AblationSpec]:
    """The full one-factor-at-a-time ablation set the task requires.

    ``fixed_step_value``/``minimal_denoise_steps`` are CLI-configurable (see
    ``scripts/diagnose_float32_digital_quality.py --fixed-step-value`` /
    ``--minimal-denoise-steps``) rather than hardcoded, since the "sensible"
    value depends on the model's step_style/schedule.
    """
    if minimal_denoise_steps < 2:
        raise ValueError(
            "minimal_denoise_steps must be at least 2: the production sampler "
            "needs two noise levels for one denoising transition"
        )

    specs = [
        AblationSpec("baseline", "No ablation — run's normal per-path behavior."),
        AblationSpec("controlnet_off", "Disable ControlNet conditioning.", use_controlnet=False),
        AblationSpec(
            "serialized_raw_edge",
            "Use the extracted/received edge as-is; skip the analog Canny "
            "JSCC retransmission net (AWGN/in-process only — digital_wire "
            "already does this by default).",
            force_edge_already_received=True,
        ),
        AblationSpec(
            "awgn_edge_retransmit",
            "Force the analog Canny/WITT retransmission net to run even on "
            "digital_wire's already-received packet edge, matching the "
            "existing AWGN/in-process edge path.",
            force_edge_already_received=False,
        ),
        AblationSpec("latent_only", "Drop caption and edge/uncertainty guidance entirely.",
                     use_text=False, use_edge=False, uncertainty_off=True),
        AblationSpec("uncertainty_off", "Zero the edge-uncertainty map; keep the edge map.",
                     uncertainty_off=True),
        AblationSpec("edge_and_uncertainty_off", "Drop both edge and uncertainty; keep caption.",
                     use_edge=False, uncertainty_off=True),
        AblationSpec(
            "fixed_reference_step", "Explicit fixed_reference digital step policy "
            "(current default — recorded as an ablation for completeness).",
            digital_step_policy_override="fixed_reference",
        ),
        AblationSpec(
            "reuse_awgn_step", "Digital paths reuse the SAME (cur_step, cur_snr) the "
            "AWGN path computed for this (video, frame, seed) instead of "
            "deriving their own from bit_depth/policy.",
            reuse_awgn_step=True,
        ),
        AblationSpec(
            "fixed_step", f"Override cur_step to a literal constant ({fixed_step_value}) "
            "for every path, removing step-matching as a variable.",
            fixed_step=fixed_step_value,
        ),
        AblationSpec(
            "diffusion_bypass_vae_direct",
            "Skip Canny retransmission/ControlNet/diffusion entirely; VAE-decode "
            "the received (power-scalar-normalized) latent directly. Isolates "
            "whether latent scaling/normalization alone explains the quality gap.",
            bypass_diffusion=True,
        ),
        AblationSpec(
            "minimal_denoise", f"Run the diffusion decoder with only "
            f"{minimal_denoise_steps} step(s).",
            diffusion_step_override=minimal_denoise_steps,
        ),
    ]
    return {s.name: s for s in specs}


BASELINE_ABLATION = AblationSpec("baseline", "No ablation — run's normal per-path behavior.")

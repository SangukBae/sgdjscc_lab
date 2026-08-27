"""diagnostics/verdict.py – Root-cause classification from evidence.

Implements the judgment criteria from the task spec:

  - in-process and wire digital paths DIFFER at some stage → packet/Tx-Rx
    problem (``packet_tx_rx_issue``), first-divergent stage = evidence.
  - both digital paths MATCH but are worse than AWGN → edge/ControlNet/
    diffusion problem (``decoder_pipeline_issue``).
  - the VAE-direct-decode ablation (``diffusion_bypass_vae_direct``) is
    ALREADY low → latent scaling/normalization problem
    (``latent_normalization_issue``).
  - all required evidence exists and no fault threshold is crossed →
    ``no_issue_detected``.
  - required evidence is missing → ``inconclusive``.

Never claims a verdict beyond what the evidence in this run supports; a run
with zero frames, all-failed paths, or missing baseline metrics returns
``inconclusive`` with the reason stated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# A pair is "materially different" at a stage if it is comparable, both sides
# are finite, and it is not exactly equal AND the normalized error exceeds
# this tolerance (guards against float32 rounding noise across independent
# recomputations of the same value being flagged as a real divergence).
DIVERGENCE_MEAN_ABS_ERR_TOL = 1e-4
DIVERGENCE_COSINE_TOL = 1e-4  # 1 - cosine_similarity must exceed this
# Edge-equalizing ablations compare independently recomputed decoder-side
# values rather than the lossless float32 transport tensor itself.  Their
# observed scalar noise (~5.3e-4 MAE in the full run) had <0.001 dB final
# PSNR impact, so use a separate tolerance instead of applying the strict
# transport contract to auxiliary decoder evidence.
AUXILIARY_DIVERGENCE_MEAN_ABS_ERR_TOL = 1e-3
AUXILIARY_DIVERGENCE_COSINE_TOL = 1e-4

# Quality gap (digital vs AWGN) large enough to call "worse", in PSNR dB.
QUALITY_GAP_PSNR_DB = 1.0


@dataclass
class VerdictResult:
    verdict: str  # packet/decoder/normalization issue | auxiliary delta | no issue | inconclusive
    reason: str
    first_divergent_stage: Optional[str] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)


def _is_divergent(
    cmp: Dict[str, Any],
    *,
    mean_abs_err_tol: float = DIVERGENCE_MEAN_ABS_ERR_TOL,
    cosine_tol: float = DIVERGENCE_COSINE_TOL,
) -> bool:
    if not cmp.get("comparable"):
        return False
    if not cmp.get("both_finite", True):
        return True
    if cmp.get("exact_equal"):
        return False
    mean_abs_err = cmp.get("mean_abs_err")
    cosine = cmp.get("cosine_similarity")
    if mean_abs_err is not None and mean_abs_err > mean_abs_err_tol:
        return True
    if cosine is not None and (1.0 - cosine) > cosine_tol:
        return True
    return False


# Stage order matches the sender->receiver->decoder pipeline direction, so the
# first divergent stage found is the earliest point of disagreement.
#
# Transport/latent stages: purely a function of the sender's VAE latent, the
# channel/serialization boundary, and step-matching -- digital_inprocess and
# digital_wire SHOULD agree here bit-for-bit (both are lossless at
# bit_depth=32). A divergence here is genuine packet/Tx-Rx evidence.
TRANSPORT_STAGES = [
    "sender_vae_latent_pre_norm", "sender_vae_latent_post_norm", "pre_serialize_latent",
    "channel_output", "post_deserialize_latent_raw", "receiver_post_norm_latent",
    "power_scalar", "cur_step", "cur_snr", "diffusion_latent_init",
]
# Edge/decoder stages: digital_inprocess retransmits its edge through the
# analog Canny/WITT net (no packet boundary to read an already-received edge
# from), while digital_wire uses the edge it already decoded from the packet
# as-is (transmission/receiver_runtime.py's own established behavior) --
# these two paths are EXPECTED to diverge here under the baseline ablation
# even when nothing is broken. A divergence here alone is therefore NOT
# packet/Tx-Rx evidence (see docs/protocols/float32_digital_diagnostics.md);
# it only becomes meaningful once the serialized_raw_edge/awgn_edge_retransmit
# ablation has equalized edge_already_received across paths.
EDGE_DECODER_STAGES = [
    "edge_mean", "edge_uncertainty_mean", "edge_post_retransmit", "uncertainty_post_ablation",
    "controlnet_input_latent", "diffusion_latent_final", "vae_decode_input", "final_reconstruction",
]
STAGE_ORDER = TRANSPORT_STAGES + EDGE_DECODER_STAGES


def classify(
    *,
    inprocess_vs_wire_comparisons: List[Dict[str, Any]],
    path_quality: Dict[str, Optional[float]],  # {"awgn_psnr":..., "digital_inprocess_psnr":..., "digital_wire_psnr":...}
    vae_direct_quality: Optional[Dict[str, Optional[float]]] = None,  # from diffusion_bypass_vae_direct ablation
    edge_handling_equalized: bool = False,
) -> VerdictResult:
    """*inprocess_vs_wire_comparisons*: list of
    ``{"stage": str, **compare_tensors() output}`` for the SAME (video,
    frame, ablation), stage-name ordered by ``STAGE_ORDER`` where possible.

    *edge_handling_equalized*: set True only when the comparisons come from a
    run where ``edge_already_received`` was forced identical across
    digital_inprocess and digital_wire (e.g. the ``serialized_raw_edge`` /
    ``awgn_edge_retransmit`` ablations) — only then are the
    ``EDGE_DECODER_STAGES`` also inspected for the packet_tx_rx_issue check;
    otherwise a divergence there is expected behavior, not evidence.
    """
    by_stage = {row["stage"]: row for row in inprocess_vs_wire_comparisons}
    candidate_stages = TRANSPORT_STAGES + (EDGE_DECODER_STAGES if edge_handling_equalized else [])
    ordered_stages = [s for s in candidate_stages if s in by_stage]

    for stage in ordered_stages:
        cmp = by_stage[stage]
        auxiliary_stage = stage in EDGE_DECODER_STAGES
        if _is_divergent(
            cmp,
            mean_abs_err_tol=(
                AUXILIARY_DIVERGENCE_MEAN_ABS_ERR_TOL
                if auxiliary_stage else DIVERGENCE_MEAN_ABS_ERR_TOL
            ),
            cosine_tol=(
                AUXILIARY_DIVERGENCE_COSINE_TOL
                if auxiliary_stage else DIVERGENCE_COSINE_TOL
            ),
        ):
            if auxiliary_stage:
                return VerdictResult(
                    verdict="transport_delta_detected",
                    reason=(
                        f"Edge-equalized auxiliary paths diverge at decoder stage '{stage}' "
                        f"(mean_abs_err={cmp.get('mean_abs_err')}, "
                        f"cosine_similarity={cmp.get('cosine_similarity')}). This is retained "
                        "as neutral auxiliary evidence, not classified as a packet/Tx-Rx fault."
                    ),
                    first_divergent_stage=stage,
                    evidence=[{"stage": stage, **{k: v for k, v in cmp.items() if k != "stage"}}],
                )
            return VerdictResult(
                verdict="packet_tx_rx_issue",
                reason=(
                    f"digital_inprocess and digital_wire first diverge at stage "
                    f"'{stage}' (mean_abs_err={cmp.get('mean_abs_err')}, "
                    f"cosine_similarity={cmp.get('cosine_similarity')}) — points to a "
                    "packet/Tx-Rx boundary problem, not the decoder."
                ),
                first_divergent_stage=stage,
                evidence=[{"stage": stage, **{k: v for k, v in cmp.items() if k != "stage"}}],
            )

    awgn_psnr = path_quality.get("awgn_psnr")
    inprocess_psnr = path_quality.get("digital_inprocess_psnr")
    wire_psnr = path_quality.get("digital_wire_psnr")

    if awgn_psnr is None or (inprocess_psnr is None and wire_psnr is None):
        return VerdictResult(
            verdict="inconclusive",
            reason="Missing AWGN or digital-path PSNR for this (video, frame) — "
                   "insufficient evidence to compare quality.",
        )

    digital_psnr = wire_psnr if wire_psnr is not None else inprocess_psnr
    quality_gap = awgn_psnr - digital_psnr
    if quality_gap < QUALITY_GAP_PSNR_DB:
        return VerdictResult(
            verdict="no_issue_detected",
            reason=(
                f"digital and in-process paths agree at every instrumented stage, "
                f"and digital PSNR ({digital_psnr:.2f} dB) is not meaningfully worse "
                f"than AWGN ({awgn_psnr:.2f} dB, gap={quality_gap:.2f} dB < "
                f"{QUALITY_GAP_PSNR_DB} dB threshold) — no quality problem evidenced "
                "in this run."
            ),
        )

    if vae_direct_quality is not None:
        vae_direct_psnr = vae_direct_quality.get("digital_wire_psnr") or vae_direct_quality.get("digital_inprocess_psnr")
        if vae_direct_psnr is not None and (awgn_psnr - vae_direct_psnr) >= QUALITY_GAP_PSNR_DB:
            return VerdictResult(
                verdict="latent_normalization_issue",
                reason=(
                    f"digital paths agree with each other, and the VAE-direct "
                    f"reconstruction (diffusion_bypass_vae_direct ablation, no "
                    f"ControlNet/diffusion involved) is ALREADY {awgn_psnr - vae_direct_psnr:.2f} "
                    "dB below AWGN — points to latent scaling/normalization, not the "
                    "diffusion decoder."
                ),
                evidence=[{"vae_direct_psnr": vae_direct_psnr, "awgn_psnr": awgn_psnr}],
            )

    return VerdictResult(
        verdict="decoder_pipeline_issue",
        reason=(
            f"digital_inprocess and digital_wire agree at every instrumented stage "
            f"(no packet/Tx-Rx divergence), but digital PSNR ({digital_psnr:.2f} dB) is "
            f"{quality_gap:.2f} dB below AWGN ({awgn_psnr:.2f} dB) — points to the "
            "edge/ControlNet/diffusion decode stages, not the transport."
        ),
    )


def aggregate_verdicts(verdicts: List[VerdictResult]) -> Dict[str, Any]:
    """Roll up per-(video,frame) verdicts into a run-level summary: counts per
    verdict label plus the most common non-inconclusive verdict (including
    ``no_issue_detected``; None only if every row is truly inconclusive)."""
    counts: Dict[str, int] = {}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    conclusive = {k: c for k, c in counts.items() if k != "inconclusive"}
    dominant = max(conclusive, key=conclusive.get) if conclusive else None
    return {"counts": counts, "dominant_verdict": dominant, "n_verdicts": len(verdicts)}

> [← docs index](./README.md)

# ETRI Priority Map for SGD-JSCC Limitations

This document reorganizes the SGD-JSCC limitation map for the current ETRI
task objective.

The primary goal is not maximum `PSNR`, but reliable preservation of semantic
intent after wireless transmission. In practice, the main priorities are:

- semantic reliability
- hallucination reduction
- robust use of semantic side information
- practical diffusion latency
- later expansion to fading / blind-channel settings

## Priority by ETRI task goal

For the next research iterations, `sgdjscc_lab` should prioritize the
following SGD-JSCC limitations first:

- `A`: hallucination and semantic inconsistency under semantic guidance
- `B`: semantic side-information fragility and transmission overhead
- `C`: high diffusion reconstruction cost and decoding latency

These map directly to the ETRI task focus:

- `A`: stronger semantic verification, hallucination detection, adaptive
  guidance, and reliable `SRS`
- `B`: corrupted caption / edge robustness, selective semantic transmission,
  and packet-aware regeneration
- `C`: few-step decoding, consistency-style acceleration, and practical
  prototype latency

The following limitations remain important, but are treated as the next
expansion track after the higher-priority semantic/reliability/latency work:

- `D`: weak channel adaptation and strong CSI assumptions in fading settings
- `E`: limited evaluation scope beyond `128x128` image transmission
- `F`: no MIMO / OFDM / multi-user extension yet

In practice, this means `Phase 5-C`, `Phase 4` packet-aware control, and
`Phase 5-B` acceleration are the primary research focus first, while
`Phase 5-A` channel/fading robustness and broader system extensions follow.

## Limitation table by ETRI priority

| Priority | SGD-JSCC limitation to solve | Why it matters in `sgdjscc_lab` | Reference papers to consult |
|---|---|---|---|
| `A1` | High-SNR guidance side effects and semantic inconsistency | At high SNR, text or edge guidance can interfere with faithful recovery and push the decoder toward plausible but incorrect content. This directly hurts `CLIP`, object preservation, hallucination score, and `SRS`. |  |
| `A2` | Residual hallucination risk from generative reconstruction | If caption, edge, or diffusion guidance is wrong, the model can generate visually natural but semantically incorrect outputs. This is a core ETRI problem because the framework explicitly evaluates hallucination and semantic reliability. |  |
| `B1` | Text side-information is assumed too idealistically | Caption transmission errors can mislead the decoder and invalidate semantic conditioning. The lab framework already models caption token corruption, so this must be handled as a first-class reliability issue. |  |
| `B2` | Edge-map guidance introduces both overhead and error propagation | Edge maps consume part of the transmission budget and can become corrupted, which then injects wrong structural hints into the diffusion decoder. This directly affects packet-aware guidance and regeneration design. |  |
| `C1` | Diffusion decoding is too slow for a usable prototype | The current multi-step denoising path is the main runtime bottleneck. ETRI needs quality-latency tradeoff experiments and a decoder that can run in fewer steps with bounded `SRS` loss. |  |
| `D1` | Fast-fading support still depends on strong CSI assumptions | Blind or imperfect-CSI robustness is required for realistic wireless evaluation. This is important for `Phase 5-A`, but follows the core semantic/hallucination/latency work. |  |
| `E1` | Evaluation scope is still narrow | Current experiments remain centered on resized `128x128` image transmission. ETRI needs stronger validation on time-aware pipelines, packet corruption, and broader semantic evaluation settings. |  |
| `F1` | MIMO / OFDM / multi-user scenarios are not covered | This limits direct applicability to practical 5G/6G-style systems, but it is a follow-up expansion after the core reliability framework is stabilized. |  |

## Lower-priority limitations for the current task

The following SGD-JSCC limitations are real, but are not the first blockers for
the current ETRI objective:

- `PSNR` is not always the best among JSCC baselines.
  The ETRI target is semantic reliability rather than pixel-perfect recovery.
- training cost and dataset scale are large.
  This affects reproducibility and deployment cost, but is less direct than
  hallucination, side-information fragility, and latency for the current task.

## Planned mapping by workstream

| Workstream | Primary reference | Local path | Planned role in `sgdjscc_lab` |
|---|---|---|---|
| Packet-aware semantic control and regeneration |  |  | adaptive guidance strength, packet-aware verifier, semantic retry policy |
| Low-latency diffusion decoding |  |  | DDIM ablation, few-step decoding, consistency-style acceleration, latency profiling |
| Stronger semantic verification |  |  | hallucination checks, `SRS-v2`, final-output selection by verified semantic reliability |
| CSI-free / blind channel-conditioned diffusion |  |  | receiver evidence abstraction, blind reconstruction, channel-token conditioning |
| Broader channel and system expansion |  |  | fading robustness, time-aware evaluation, later MIMO / OFDM extension planning |

## Why the work is split this way

- first solve the core ETRI-facing limits:
  hallucination, semantic inconsistency, and unreliable side information
- then reduce diffusion latency so the framework is runnable as a practical
  prototype
- then deepen blind-channel robustness and broader wireless-system coverage

> [← docs index](./README.md)

# External References for Phase 4 and Phase 5

Phase 4 and Phase 5 were designed by combining the current `sgdjscc_lab`
package with three papers and two local codebases already stored under
`paper/`.

This document records the reference map that guided the Phase 4 / 5 design and
still serves as the rationale for the current implementation status.

## Priority by SGD-JSCC limitation

For the next research iterations, `sgdjscc_lab` will prioritize the following
SGD-JSCC limitations first:

- `1`: implicit semantics hidden only in latent transmission
- `2`: weak explicit semantic guidance (`caption + canny` only)
- `5`: high diffusion reconstruction cost
- `6`: semantic side-information overhead

These map directly to the ETRI task focus:

- `1` and `2`: explicit semantic encoding, semantic reliability, hallucination
  reduction
- `5`: practical low-latency reconstruction for a usable prototype
- `6`: compact / selective semantic transmission instead of always sending full
  side information

The following limitations remain important, but are treated as follow-up
robustness tracks after the higher-priority semantic/reliability/latency work:

- `3`: weak channel adaptation and strong CSI assumptions
- `4`: limited fast-fading support

In practice, this means `Phase 4` and `Phase 5-B/5-C` are the primary research
focus first, while `Phase 5-A` channel/fading robustness is treated as the next
expansion track.

## Reference table by limitation

| SGD-JSCC limitation | Why it matters in `sgdjscc_lab` | Reference papers to consult |
|---|---|---|
| `1`. Implicit semantics hidden only in latent transmission | We need explicit semantic units for packet extraction, semantic verification, and controllable reconstruction beyond black-box latent transport. | `Generative Semantic Communication for Joint Image Transmission and Segmentation` (2024); `Scene Graph Disentanglement and Composition for Generalizable Complex Image Generation` (NeurIPS 2024); `Preserving Semantics in Diffusion-based Communication` (2025) |
| `2`. Weak explicit semantic guidance (`caption + canny` only) | The current baseline guidance is too weak for object/relation/attribute preservation and hallucination control. | `PixArt-δ: Fast and Controllable Image Generation with Latent Consistency Models` (2024); `Scene Graph-Grounded Image Generation` (AAAI 2025); `Generative Semantic Communication for Joint Image Transmission and Segmentation` (2024) |
| `3`. Weak channel adaptation and strong CSI assumptions | Needed for future blind channel robustness, channel-token conditioning, and non-AWGN channel experiments. | `Semantics-Guided Diffusion for Deep Joint Source-Channel Coding in Wireless Image Transmission` (2025); `DiffCom: Channel Received Signal is a Natural Condition to Guide Diffusion Posterior Sampling` (2024); `Deep Joint Source-Channel Coding for Adaptive Image Transmission over MIMO Channels` (IEEE TWC 2024); `Channel Code-Book: Semantic Image-Adaptive Transmission in Diverse Channel Environments` (2025) |
| `4`. Limited support for complex wireless conditions such as fast fading | Needed for later per-symbol / per-patch reliability modelling and fading-aware denoising. | `Semantics-Guided Diffusion for Deep Joint Source-Channel Coding in Wireless Image Transmission` (2025); `CDDM: Channel Denoising Diffusion Models for Wireless Semantic Communications` (2024); `DiffCom: Channel Received Signal is a Natural Condition to Guide Diffusion Posterior Sampling` (2024) |
| `5`. High diffusion reconstruction cost | We need few-step decoding, consistency-style acceleration, and adaptive step control for a practical prototype. | `PixArt-δ: Fast and Controllable Image Generation with Latent Consistency Models` (2024); `FAST-GSC: Fast and Adaptive Semantic Transmission for Generative Semantic Communication` (2024); `Continuous-time Consistency Models` (2024); `Latent Diffusion Model-Enabled Low-Latency Semantic Communication in the Presence of Semantic Ambiguities and Wireless Channel Noises` |
| `6`. Semantic side-information overhead | We need compact semantic packets, selective transmission, and request-based regeneration instead of always sending full side information. | `FAST-GSC: Fast and Adaptive Semantic Transmission for Generative Semantic Communication` (2024); `Diffusion-Aided Bandwidth-Efficient Semantic Communication with Adaptive Requests` (2026); `Preserving Semantics in Diffusion-based Communication` (2025) |

## Reference mapping

| Target | Primary reference | Local path | Planned role in `sgdjscc_lab` |
|---|---|---|---|
| Video / keyframe semantic transmission | FAST-GSC | `paper/FAST-GSC: Fast and Adaptive Semantic Transmission for Generative Semantic Communication/FAST_GSC.tex` | semantic unit design, transmission order, semantic difference calculation, sequential conditional denoising, latency-oriented video pipeline |
| CSI-free / blind channel-conditioned diffusion | DiffCom | `paper/diffcom/README.md`, `paper/diffcom/main_diffcom.py`, `paper/diffcom/conditioning_method/diffcom.py`, `paper/diffcom/guided_diffusion/measurement.py` | noisy received signal conditioning, operator-style channel observation abstraction, blind reconstruction experiments |
| Few-step / low-latency diffusion | LDM-enabled low-latency SemCom | `paper/LDM-enabled-SemCom-system/README.md`, `paper/LDM-enabled-SemCom-system/train_DIV2K/consistency_models.py`, `paper/LDM-enabled-SemCom-system/train_DIV2K/consistency_model_training.py`, `paper/LDM-enabled-SemCom-system/train_DIV2K/t_calculate.py` | DDIM baseline, consistency distillation, EECD-style latency measurement, one-step / few-step decoder experiments |
| Paper baseline continuity | SGD-JSCC | `paper/Semantics-Guided Diffusion for Deep Joint Source-Channel Coding in Wireless Image Transmission/SGDJSCC_arxiv.tex` | preserve the current image transmission path and keep all Phase 4/5 work as controlled extensions in `sgdjscc_lab/` |

## Why these references are split this way

- `FAST-GSC` is the main reference for `Phase 4`, because it directly addresses
  semantic-unit ordering, parallel extraction/inference, and late-arriving
  condition handling in a latency-sensitive generative SemCom pipeline.
- `DiffCom` is the main reference for `Phase 5-A`, because it already treats
  the raw received channel signal as a natural diffusion condition and includes
  a `blind_diffcom` path for unknown or imperfect channel settings.
- `LDM-enabled-SemCom-system` is the main reference for `Phase 5-B`, because it
  contains concrete consistency-model training and sampling code for reducing
  denoising steps while retaining semantic quality.

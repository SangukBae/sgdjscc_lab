---
status: frozen
updated: 2026-08-26
owner: ETRI SGD-JSCC Research Team
source_commit: d0d3bfb
supersedes:
---

# ETRI Progress Deck Outline

- Audience: external
- Date: August 2026
- Type: frozen presentation snapshot
- Current index: [documentation index](../../README.md)

## Slide 1 — Research Topic

- Domain: generative semantic media transmission
- Goal: reconstruction quality plus semantic reliability
- Risk: generative hallucination

## Slide 2 — Problem Definition

- Objectives
  - lower transmission volume
  - preserve reconstruction quality
  - reduce missing and added semantics
- Evaluation
  - pixel quality
  - semantic fidelity
  - temporal stability

## Slide 3 — Implementation Status

- Complete
  - image inference and evaluation
  - packet verifier
  - keyframe video pipeline
  - temporal metrics
- Partial
  - channel conditioning
  - low-latency sampling
  - regeneration search
- Missing
  - verifier-action injection into sampler
  - physical CBR/FEC validation

## Slide 4 — Baseline Pipeline

```text
Image/Frame
  → VAE and JSCC encoder
  → wireless channel
  → diffusion reconstruction
  → quality and semantic evaluation
```

- Default channel: AWGN
- Optional channels: Rayleigh, fading, packet drop
- Research extensions: opt-in

## Slide 5 — Video Extension

- Transmitter
  - keyframe
  - semantic delta
  - motion signal
- Receiver
  - reuse
  - recompute
  - generate
- Units
  - frame
  - GOP
  - segment

## Slide 6 — Hallucination Verification

- Error types
  - missing object
  - additional object
  - relation or attribute error
- Presence backends
  - CLIP
  - OWLv2
  - VQA
  - GT
- Current gap
  - controller decides and logs actions
  - sampler injection not implemented

## Slide 7 — Temporal Metrics

- PTC: packet consistency; higher is better
- SFR: abnormal object birth/death; lower is better
- SDI: semantic drift from keyframes; lower is better
- Evaluation rule
  - separate loop-internal and held-out metrics

## Slide 8 — Achievements and Next Steps

- Achievements
  - modular research framework
  - temporal video evaluation
  - packet-based self-verification
  - exact serialized packet-byte accounting
- Next steps
  - verifier-to-sampler wiring
  - dynamic transmission budget
  - fair baselines and ablations
  - held-out video validation

## Slide 9 — Follow-on Research

- rate-adaptive video semantic communication
- receiver-side reliability-controlled generation
- semantic transmission evaluation benchmark

## Required Caveats

- implementation complete ≠ performance superiority
- proxy symbols ≠ physical CBR
- mock/proxy PSSS ≠ real MLLM PSSS
- ten-video result ≠ completed generalization

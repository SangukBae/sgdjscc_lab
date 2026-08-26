---
status: frozen
updated: 2026-08-26
owner: ETRI SGD-JSCC Research Team
source_commit: d0d3bfb
supersedes:
---

# ETRI Appendix Slide Outline

- Scope: evaluation inputs, metrics, and calculations
- Date: August 2026
- Type: frozen snapshot

## Appendix A — Evaluation Inputs

- Images
  - Kodak
  - SNR sweep
- Videos
  - ten ETRI clips
  - 512×256
  - 10fps
  - 100 frames per clip
- Semantic GT
  - sampled-frame object presence
  - separate closed-world and open-world use

## Appendix B-1 — Image Metrics

- PSNR: pixel error; higher is better
- SSIM: structural similarity; higher is better
- LPIPS: perceptual distance; lower is better
- CLIP
  - image-image similarity
  - text-image similarity
- SRS
  - CLIP similarity
  - object preservation
  - missing and additional penalties

## Appendix B-2 — Packet and Video Metrics

- Packet verifier
  - object
  - relation
  - attribute
  - scene
- PTC: mean packet consistency over time
- SFR: abnormal object birth/death rate
- SDI: mismatch slope over keyframe distance
- Temporal hallucination: additional-object rate over time

## Evaluation Rules

- loop-internal metrics
  - regeneration and control only
- held-out metrics
  - final reporting only
- rate
  - separate exact packet bytes and proxy symbols
- statistics
  - paired per-video results
  - mean, standard deviation, confidence interval

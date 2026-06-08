> [← docs index](./README.md)

# Phase 1–3 Summary

## Phase 1 Summary

Phase 1 established the minimum runnable package:

- AWGN single-image / folder inference
- config-driven CLI
- output image save path
- original inference path preserved

Completion criterion:

```bash
python scripts/infer_images.py --config configs/default.yaml
```

---

## Phase 2 Summary

Phase 2 transformed the monolithic script structure into a modular package.

### Main structural changes

| Before | After |
|---|---|
| inline AWGN channel inside model | `channels/awgn.py` |
| flat runtime loader | `models/jscc_model.py` + `models/diffusion_wrapper.py` + `runtime.py` |
| flat pipeline | `pipelines/infer_pipeline.py` |
| top-level preprocessing | `utils/preprocessing.py` |
| seed and memory helpers inside scripts/pipeline | `utils/seed.py`, `utils/memory.py` |
| no fragment config system | `_defaults_` composition in `config.py` |

### Phase 2 completion points

- modular package structure
- editable install support
- config composition
- unit tests for config / I/O / AWGN channel

---

## Phase 3 Summary

Phase 3 established the actual research-evaluation foundation.

### Evaluators

- `quality.py` — PSNR / SSIM / LPIPS
- `clip_score.py` — CLIP image-image and text-image similarity
- `object_preservation.py` — object preservation rate
- `hallucination.py` — hallucination score
- `semantic_reliability.py` — Semantic Reliability Score (SRS)

### Guidance extensions

- `depth_extractor.py` — DPT monocular depth
- `segmentation_extractor.py` — SegFormer semantic segmentation

### Evaluation pipelines

- `eval_pipeline.py` — single-SNR and SNR-sweep evaluation
- `regeneration_loop.py` — SRS-triggered retry path
- `evaluate.py` — evaluation CLI

### Dataset configs

- Kodak
- COCO val2017
- ADE20K validation

### Semantic Reliability Score

```text
SRS = 0.30 × clip_image_image
    + 0.25 × clip_text_image
    + 0.25 × object_preservation_rate
    - 0.10 × missing_object_rate
    - 0.10 × additional_object_rate
```

### Phase 3 heuristic limitations at that point

- Object preservation and hallucination are still heuristic CLIP-based metrics
- POPE-style VQA is not yet integrated in the Phase 3 stack
- Depth / segmentation models require external downloads on first use
- Regeneration loop is a lightweight prototype

Later phases add packet-aware verification, temporal metrics, local VQA-backed
hallucination checks, SRS-v2, channel-conditioned evaluation, and latency /
early-exit experiments. This section is intentionally a **Phase 3 snapshot**,
not the final state of the current package.

# sgdjscc_lab Development Plan

## Purpose

`sgdjscc_lab` is the research and development fork for extending `SGDJSCC`
without modifying the original `SGDJSCC/` package. The original repository is
kept as a read-only reference and paper baseline, while `sgdjscc_lab` serves as
the clean package for modularization, evaluation, and future research.

---

## Phase Status

| Phase | Status | Completion Criterion |
|-------|--------|---------------------|
| 1 | ✅ Complete | `python scripts/infer_images.py --config configs/default.yaml` runs AWGN inference |
| 2 | ✅ Complete | channels / guidance / models / pipelines 분리, `_defaults_` composition |
| 3 | ✅ Complete | Full evaluator suite, SNR-sweep CSV, depth/seg guidance, regeneration loop |
| 4 | 🔲 Planned | Video keyframe consistency and temporal metrics |
| 5 | 🔲 Planned | Rayleigh fading, DiT/DiTJSCC, stronger semantic evaluation |

---

## Repository Strategy

### `SGDJSCC/`
- original code preservation
- reproduction reference
- paper baseline
- never modified by research iterations in `sgdjscc_lab`

### `sgdjscc_lab/`
- clean research fork
- config-driven CLI
- structural reorganization
- evaluator and experiment framework
- future guidance / channel / video extensions

---

## Current Directory Layout

```text
sgdjscc_lab/
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── default.yaml
│   ├── composed.yaml
│   ├── channel/awgn.yaml
│   ├── model/sgdjscc.yaml
│   ├── infer/awgn.yaml
│   ├── eval/default.yaml
│   └── dataset/
│       ├── kodak.yaml
│       ├── coco.yaml
│       └── ade20k.yaml
├── scripts/
│   ├── infer_images.py
│   └── evaluate.py
├── src/sgdjscc_lab/
│   ├── config.py
│   ├── io.py
│   ├── runtime.py
│   ├── channels/
│   │   └── awgn.py
│   ├── guidance/
│   │   ├── text_extractor.py
│   │   ├── edge_extractor.py
│   │   ├── depth_extractor.py
│   │   └── segmentation_extractor.py
│   ├── models/
│   │   ├── jscc_model.py
│   │   ├── diffusion_wrapper.py
│   │   └── model_bundle.py
│   ├── pipelines/
│   │   ├── infer_pipeline.py
│   │   ├── eval_pipeline.py
│   │   └── regeneration_loop.py
│   ├── evaluators/
│   │   ├── quality.py
│   │   ├── clip_score.py
│   │   ├── object_preservation.py
│   │   ├── hallucination.py
│   │   └── semantic_reliability.py
│   └── utils/
│       ├── preprocessing.py
│       ├── memory.py
│       ├── seed.py
│       ├── csv_logger.py
│       └── metrics_io.py
└── tests/
    ├── test_config.py
    ├── test_io.py
    ├── test_channels.py
    ├── test_evaluators.py
    └── test_eval_pipeline.py
```

---

## Development Principles

### Principle 1: Preserve the original algorithm path

All core forward-pass computations remain aligned with the original
`SGDJSCC/inference_one.py`:

- VAE encode/decode with scaling factor `15.45`
- AWGN noise injection
- blind SNR prediction
- step matching
- canny retransmission
- canny latent VAE encoding
- diffusion generate path
- final normalized decode

### Principle 2: Separate interfaces before adding research ideas

The package is designed so that each concern can be replaced independently:

- `channels/` for channel models
- `guidance/` for semantic and structural extractors
- `models/` for JSCC and diffusion wrappers
- `pipelines/` for inference and evaluation orchestration
- `evaluators/` for research metrics

### Principle 3: Keep the original repository read-only

Any new idea should be implemented in `sgdjscc_lab/`, not in `SGDJSCC/`.

---

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

### Current heuristic limitations

- Object preservation and hallucination are still heuristic CLIP-based metrics
- POPE-style VQA is not yet integrated
- Depth / segmentation models require external downloads on first use
- Regeneration loop is a lightweight prototype

---

## Phase 4 Plan

Phase 4 extends the image package into a video/keyframe research framework.

Planned tasks:

1. keyframe extraction and grouping
2. temporal consistency metrics
3. video-oriented pipelines
4. keyframe-conditioned reconstruction flow

The intended approach is to reuse the Phase 3 inference/evaluation API rather
than build a separate codebase.

---

## Phase 5 Plan

Phase 5 is for deeper channel/model research:

1. `channels/rayleigh.py`
2. `configs/channel/rayleigh.yaml`
3. DiT / DiTJSCC style backbone experiments
4. stronger semantic evaluation and VQA-based hallucination analysis
5. multi-strategy regeneration and search

---

## Recommended Research Workflow

1. Use `SGDJSCC/` only as a paper-reference baseline.
2. Run inference and evaluation from `sgdjscc_lab/`.
3. Add new guidance, channel, or evaluator modules inside the modular package.
4. Compare ideas through Phase 3 metrics before extending to video or new channels.

---

## Related Documents

- [../README.md](../README.md) — user-facing package usage
- [framework_comparison.md](./framework_comparison.md) — original `SGDJSCC` vs `sgdjscc_lab` structure comparison

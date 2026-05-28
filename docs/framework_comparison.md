# Framework Comparison

## Purpose

This document compares:

- the original `SGDJSCC/` end-to-end inference framework
- the `sgdjscc_lab/` **Phase 2** modularised framework

The goal is to show how the same AWGN semantic image transmission pipeline was
kept algorithmically similar while being structurally reorganised for research
and extension.

---

## Block Diagram: Original SGDJSCC

```mermaid
flowchart TB
    O1["Entry Script\ninference_one.py"]
    O2["Config / Arguments\ninference_config.py\nhardcoded paths + runtime flags"]
    O3["Inline Model Construction\nJSCC model / BLIP2 / MuGE\nDiffusion backbone / ControlNet\nCLIP / shared VAE"]
    O4["Input Image / Dataset\nImageFolder or direct image"]
    O5["Patch Preparation\nsplit_image_v2()\npreprocess inline"]
    O6["Semantic Guidance Extraction\nBLIP2 caption\nMuGE soft edge"]
    O7["JSCC Encode\nVAE encode / scaling_factor=15.45 / L2 normalize"]
    O8["Wireless Channel\ninline AWGN injection"]
    O9["Step Matching\nblind SNR prediction / power scalar / mask token"]
    O10["Canny Re-Transmission\ncanny TX net / canny latent encoding"]
    O11["Diffusion Denoising\nMDTv2 / ControlNet\nDiffusionGenerator.generate()"]
    O12["Final Decode\nVAE decode / save / log / evaluate inline"]

    O1 --> O2 --> O3 --> O4 --> O5 --> O6 --> O7 --> O8 --> O9 --> O10 --> O11 --> O12
```

---

## Block Diagram: sgdjscc_lab Phase 2

```mermaid
flowchart TB
    L1["CLI Entry\nscripts/infer_images.py"]
    L2["Config Layer\nconfig.py / default.yaml / CLI override merge"]
    L3["Runtime Assembly\nruntime.py shim → ModelBundle"]
    L4["Model Builders\nmodels/jscc_model.py\nmodels/diffusion_wrapper.py\nmodels/model_bundle.py"]
    L5["Input / I/O Layer\nio.py — single image or folder"]
    L6["Preprocessing Layer\nutils/preprocessing.py\nprepare_patches() / split / merge"]
    L7["Guidance Layer\nguidance/text_extractor.py\nguidance/edge_extractor.py"]
    L8["JSCC Core\nmodels/jscc_model.py\nVAE encode / normalize"]
    L9["Channel Layer\nchannels/awgn.py\nAWGNChannel.transmit()"]
    L10["Inference Pipeline\npipelines/infer_pipeline.py\nstep matching / mask token / canny retransmission"]
    L11["Diffusion Layer\nDiffusionGenerator\nMDTv2 / ControlNet"]
    L12["Output / Extension Layer\nsave image / evaluators/quality.py scaffold\ntests/ + docs"]

    L1 --> L2 --> L3 --> L4 --> L5 --> L6 --> L7 --> L8 --> L9 --> L10 --> L11 --> L12
```

---

## Structural Difference Summary

| Topic | Original `SGDJSCC/` | `sgdjscc_lab` Phase 2 |
|---|---|---|
| Entry point | `inference_one.py` 중심 | `scripts/infer_images.py` |
| Config handling | script 내부 결합 + 일부 하드코딩 | `config.py` + YAML + CLI override |
| Model loading | 한 파일 내부에서 inline 구성 | `models/` + `runtime.py` assembly |
| Channel logic | `_JSCCModel.channel()` 내부 | `channels/awgn.py` |
| Guidance logic | script 내부 함수 | `guidance/` 하위 모듈 |
| Inference flow | script 중심 monolithic | `pipelines/infer_pipeline.py` |
| Preprocessing | script와 util 혼합 | `utils/preprocessing.py` |
| Evaluation | script 끝단에 섞임 | `evaluators/` scaffold 분리 |
| Extensibility | 구조상 확장 어려움 | channel / guidance / evaluator 확장 용이 |
| Original code modification | 해당 없음 | `SGDJSCC/`는 read-only reference 유지 |

---

## Interpretation

### 1. What stayed the same

The following algorithmic blocks are intentionally preserved:

- VAE encode / decode
- scaling factor `15.45`
- AWGN channel corruption
- blind SNR prediction
- step matching
- mask token generation
- canny retransmission
- canny latent conditioning
- diffusion denoising with MDTv2 / ControlNet

In other words, `sgdjscc_lab` Phase 2 is **not a new transmission algorithm**.
It is a **modular re-packaging** of the original `SGDJSCC` inference path.

### 2. What changed structurally

The major Phase 2 change is separation of responsibilities:

- `channels/` isolates wireless corruption logic
- `guidance/` isolates semantic extraction logic
- `models/` isolates construction of core model components
- `pipelines/` isolates the orchestration flow
- `utils/` collects preprocessing, seed, and memory helpers
- `evaluators/` provides a clear insertion point for Phase 3 metrics

### 3. Why this matters

This separation makes later work practical:

- AWGN → Rayleigh channel replacement
- edge guidance → depth / segmentation guidance expansion
- metric loop insertion without touching inference core
- easier testing and clearer failure isolation

---

## Phase 2 Position

Phase 2 should be understood as:

- **algorithm-preserving**
- **structure-improving**
- **research-extension ready**

It is the bridge between:

- **Phase 1**: "make the original AWGN inference reproducible"
- **Phase 3**: "add evaluation, richer guidance, and research features"

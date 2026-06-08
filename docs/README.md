# sgdjscc_lab Development Plan

## Purpose

`sgdjscc_lab` is the research and development fork for extending `SGDJSCC`
without modifying the original `SGDJSCC/` package. The original repository is
kept as a read-only reference and paper baseline, while `sgdjscc_lab` serves as
the clean package for modularization, evaluation, and future research.

This file is the **index**. The detailed content has been split into focused
documents (see "Document Map" below).

---

## Document Map

| Document | Contents |
|---|---|
| [phases_1to3.md](./phases_1to3.md) | Phase 1 / 2 / 3 summaries (inference CLI, modular package, evaluation framework + SRS) |
| [limitation_reference_map.md](./limitation_reference_map.md) | External references for Phase 4/5: SGD-JSCC limitation priority, reference tables, reference mapping (FAST-GSC / DiffCom / LDM-SemCom) |
| [phase4.md](./phase4.md) | Phase 4 plan + implementation status: 4-A packet-aware verifier + adaptive guidance, 4-B keyframe / temporal; delivered modules, config/CLI usage, limitations |
| [phase5.md](./phase5.md) | Phase 5 plan + implementation status: 5-A channel conditioning, 5-B low-latency/consistency, 5-C verifier/search; per-module `implemented / wired / approximated / fallback / not-yet` tags, integration status, resolved + remaining limitations |
| [framework_comparison.md](./framework_comparison.md) | original `SGDJSCC/` vs `sgdjscc_lab/` structure comparison |
| [framework_file_roles.md](./framework_file_roles.md) | file-by-file framework role map in execution order |

---

## Phase Status

| Phase | Status | Completion Criterion |
|-------|--------|---------------------|
| 1 | ✅ Complete | `python scripts/infer_images.py --config configs/default.yaml` runs AWGN inference |
| 2 | ✅ Complete | channels / guidance / models / pipelines 분리, `_defaults_` composition |
| 3 | ✅ Complete | Full evaluator suite, SNR-sweep CSV, depth/seg guidance, regeneration loop |
| 4 | ✅ Complete | Phase 4-A packet-aware verifier + adaptive guidance; Phase 4-B keyframe / temporal pipeline (see [phase4_status.md](./phase4.md)) |
| 5 | ✅ Scaffolded | Phase 5-A channel-conditioned diffusion (Rayleigh/fast-fading/packet-drop + measurement bundle), 5-B low-latency sampling/consistency/early-exit, 5-C SRS-v2 + regeneration search (see [phase5_status.md](./phase5.md)) |

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

> The layout above shows the Phase 1–3 core. Phase 4/5 add `controllers/`,
> `acceleration/`, `video/`, more `channels/` `guidance/` `evaluators/` modules,
> and extra config presets — see [phase4_status.md](./phase4.md) and
> [phase5_status.md](./phase5.md) for the full module lists.

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

## Recommended Research Workflow

1. Use `SGDJSCC/` only as a paper-reference baseline.
2. Run inference and evaluation from `sgdjscc_lab/`.
3. Add new guidance, channel, or evaluator modules inside the modular package.
4. Compare ideas through Phase 3 metrics before extending to video or new channels.

---

## Related Documents

- [../README.md](../README.md) — user-facing package usage
- [framework_comparison.md](./framework_comparison.md) — original `SGDJSCC` vs `sgdjscc_lab` structure comparison
- [framework_file_roles.md](./framework_file_roles.md) — file-by-file framework role map

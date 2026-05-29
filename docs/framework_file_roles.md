# sgdjscc_lab Framework File Roles

## Purpose

This document maps `sgdjscc_lab` files to their roles in the overall framework,
ordered by the actual execution flow.

The primary ordering is:

1. inference framework
2. evaluation framework
3. optional extension and compatibility modules

The core point is that `sgdjscc_lab` preserves the original `SGDJSCC`
algorithmic path, but reorganizes the code into explicit modules.

For the added paper mapping column, the Figure 1(b) block names are:

- `DeepJSCC Encoder`
- `Semantic Extractor`
- `Semantic side information encoder`
- `Wireless Channel`
- `Semantic side information decoder`
- `Diffusion Denoiser`
- `DeepJSCC Decoder`

If a file is only infrastructure for running experiments, it is marked as
`Outside Figure 1(b)`. If one file coordinates several blocks together, it is
marked as `Cross-block orchestration`.

---

## 1. Inference Framework Order

| Order | Framework stage | Main file(s) | Module role in the framework | Figure 1(b) block |
|---|---|---|---|---|
| 1 | CLI entry | `scripts/infer_images.py` | Starts inference, parses CLI args, loads config, resolves device, builds models, and launches batch inference. | Outside Figure 1(b) |
| 2 | Config loading and override | `src/sgdjscc_lab/config.py` | Loads YAML config, merges `_defaults_` fragments, resolves relative paths, and applies CLI overrides. | Outside Figure 1(b) |
| 3 | Config fragments | `configs/default.yaml` | Single-file default config for basic AWGN inference. | Outside Figure 1(b) |
| 4 | Config fragments | `configs/composed.yaml` | Composed config entry that assembles channel/model/infer/eval fragments. | Outside Figure 1(b) |
| 5 | Config fragments | `configs/channel/awgn.yaml` | Defines channel-level AWGN settings such as `snr_db`. | Wireless Channel |
| 6 | Config fragments | `configs/model/sgdjscc.yaml` | Defines model/guidance/diffusion-related inference options. | Cross-block orchestration |
| 7 | Config fragments | `configs/infer/awgn.yaml` | Defines input path, output path, and runtime device defaults. | Outside Figure 1(b) |
| 8 | Reproducibility setup | `src/sgdjscc_lab/utils/seed.py` | Fixes random seeds for Python, NumPy, PyTorch, and cuDNN. | Outside Figure 1(b) |
| 9 | Device and model assembly | `src/sgdjscc_lab/runtime.py` | Converts device string to `torch.device` and assembles the full model bundle for inference. | Cross-block orchestration |
| 10 | External code bridge | `src/sgdjscc_lab/_sgdjscc.py` | Injects `SGDJSCC/` into `sys.path` so the lab package can reuse original baseline modules. | Outside Figure 1(b) |
| 11 | JSCC core model | `src/sgdjscc_lab/models/jscc_model.py` | Builds the VAE encoder/decoder, blind SNR predictor, and canny transmission network used by the original SGDJSCC forward path. | DeepJSCC Encoder / Semantic side information encoder / Semantic side information decoder / DeepJSCC Decoder |
| 12 | Channel module | `src/sgdjscc_lab/channels/awgn.py` | Implements the AWGN latent transmission step as a replaceable channel module. | Wireless Channel |
| 13 | Diffusion semantic pipeline | `src/sgdjscc_lab/models/diffusion_wrapper.py` | Loads the diffusion backbone, optional ControlNet, CLIP, and shared VAE for semantic reconstruction. | Diffusion Denoiser |
| 14 | Model container | `src/sgdjscc_lab/models/model_bundle.py` | Packs JSCC model, diffusion pipeline, guidance extractors, and device/offload settings into one runtime bundle. | Outside Figure 1(b) |
| 15 | Input file discovery and image I/O | `src/sgdjscc_lab/io.py` | Lists image files, loads them as tensors, and saves reconstructed results. | Outside Figure 1(b) |
| 16 | Patch preprocessing | `src/sgdjscc_lab/utils/preprocessing.py` | Crops/resizes inputs if needed, splits images into `128x128` patches, and merges reconstructed patches back. | Pre-stage support before DeepJSCC Encoder |
| 17 | Batch orchestration | `src/sgdjscc_lab/pipelines/infer_pipeline.py` | Main inference pipeline: loops over images, prepares patches, runs per-patch inference, and writes outputs. | Cross-block orchestration |
| 18 | Text guidance extraction | `src/sgdjscc_lab/guidance/text_extractor.py` | Generates BLIP2 captions used as semantic text guidance when `use_text=true`. | Semantic Extractor |
| 19 | Edge guidance extraction | `src/sgdjscc_lab/guidance/edge_extractor.py` | Generates MuGE soft edge maps and uncertainty maps used as structural guidance. | Semantic Extractor |
| 20 | Core forward pass | `src/sgdjscc_lab/pipelines/infer_pipeline.py` | Runs the main ordered blocks: soft edge preprocessing -> VAE encode/normalize -> AWGN -> mask/power scalar -> step matching -> canny retransmission -> canny latent -> diffusion denoising -> final decode. | DeepJSCC Encoder -> Semantic side information encoder -> Wireless Channel -> Semantic side information decoder -> Diffusion Denoiser -> DeepJSCC Decoder |
| 21 | Output save | `src/sgdjscc_lab/io.py` | Saves the reconstructed image tensor to the output directory as an image file. | Outside Figure 1(b) |

### Inference Summary

The inference backbone is effectively:

`infer_images.py -> config.py -> runtime.py -> model builders -> io/preprocessing -> infer_pipeline.py -> guidance -> JSCC + AWGN + diffusion -> save`

---

## 2. Evaluation Framework Order

| Order | Framework stage | Main file(s) | Module role in the framework | Figure 1(b) block |
|---|---|---|---|---|
| 1 | Evaluation CLI entry | `scripts/evaluate.py` | Starts evaluation, parses SNR options, loads config, builds eval context, builds models, and launches single-SNR or SNR-sweep evaluation. | Outside Figure 1(b) |
| 2 | Evaluation config fragment | `configs/eval/default.yaml` | Defines enabled metrics, SNR sweep list, CSV path, SRS weights, and regeneration-loop options. | Outside Figure 1(b) |
| 3 | Dataset config fragments | `configs/dataset/kodak.yaml`, `configs/dataset/coco.yaml`, `configs/dataset/ade20k.yaml` | Provide dataset-specific input/reference/annotation settings for evaluation experiments. | Outside Figure 1(b) |
| 4 | Evaluation pipeline | `src/sgdjscc_lab/pipelines/eval_pipeline.py` | Wraps the inference pipeline with dataset iteration, metric computation, optional CSV logging, and SNR sweep control. | Cross-block orchestration |
| 5 | Reuse of inference core | `src/sgdjscc_lab/pipelines/infer_pipeline.py` | Performs the actual reconstruction inside evaluation. Evaluation does not replace the inference algorithm; it wraps it. | DeepJSCC Encoder -> Semantic side information encoder -> Wireless Channel -> Semantic side information decoder -> Diffusion Denoiser -> DeepJSCC Decoder |
| 6 | Quality metrics | `src/sgdjscc_lab/evaluators/quality.py` | Computes PSNR, SSIM, and LPIPS. | Outside Figure 1(b) |
| 7 | CLIP semantic metrics | `src/sgdjscc_lab/evaluators/clip_score.py` | Computes CLIP image-image and text-image similarity. | Outside Figure 1(b) |
| 8 | Object preservation metric | `src/sgdjscc_lab/evaluators/object_preservation.py` | Estimates how many objects present in the original survive in the reconstruction. | Outside Figure 1(b) |
| 9 | Hallucination metric | `src/sgdjscc_lab/evaluators/hallucination.py` | Estimates which extra objects appear in the reconstruction but not in the original. | Outside Figure 1(b) |
| 10 | Semantic Reliability Score | `src/sgdjscc_lab/evaluators/semantic_reliability.py` | Combines semantic metrics into the headline SRS score. | Outside Figure 1(b) |
| 11 | CSV result logging | `src/sgdjscc_lab/utils/csv_logger.py` | Streams per-image metric rows into CSV files during long evaluations. | Outside Figure 1(b) |
| 12 | Summary formatting | `src/sgdjscc_lab/utils/metrics_io.py` | Aggregates metric rows and formats console summary tables. | Outside Figure 1(b) |
| 13 | Optional retry path | `src/sgdjscc_lab/pipelines/regeneration_loop.py` | Re-runs reconstruction when semantic reliability is below threshold and keeps the best retry result. | Cross-block orchestration around Diffusion Denoiser / DeepJSCC Decoder |

### Evaluation Summary

The evaluation backbone is effectively:

`evaluate.py -> config.py/eval config -> eval_pipeline.py -> infer_pipeline.py -> evaluators -> csv_logger.py -> metrics_io.py`

---

## 3. Optional Extension Modules

| File | Role | Figure 1(b) block |
|---|---|---|
| `src/sgdjscc_lab/guidance/depth_extractor.py` | Optional depth-guidance extractor for future structural conditioning experiments. | Semantic Extractor |
| `src/sgdjscc_lab/guidance/segmentation_extractor.py` | Optional semantic-segmentation extractor for region-aware guidance and analysis. | Semantic Extractor |

These files are not part of the default AWGN inference path today, but they
exist as extension points for future research phases.

---

## 4. Compatibility Shim Modules

| File | Role | Figure 1(b) block |
|---|---|---|
| `src/sgdjscc_lab/pipeline.py` | Backward-compatible re-export of the old top-level pipeline API. | Outside Figure 1(b) |
| `src/sgdjscc_lab/preprocessing.py` | Backward-compatible re-export of preprocessing helpers moved into `utils/preprocessing.py`. | Outside Figure 1(b) |
| `src/sgdjscc_lab/__init__.py` | Package root marker and import surface for the Python package. | Outside Figure 1(b) |

These files are not the main execution path themselves; they keep older import
paths from breaking after the Phase 2 modular reorganization.

---

## 5. One-Line Interpretation

`sgdjscc_lab` is organized as:

- `scripts/`: user entry points
- `configs/`: experiment settings
- `runtime/models/channels/guidance/`: model assembly and core components
- `pipelines/`: execution order and orchestration
- `evaluators/`: research metrics
- `utils/`: I/O, preprocessing, reproducibility, logging
- compatibility shims: old import path support

So the package is not just "a model file"; it is a full experiment framework
that separates entry, configuration, inference, evaluation, and extension
points into different modules.

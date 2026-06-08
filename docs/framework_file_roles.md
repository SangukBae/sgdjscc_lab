# sgdjscc_lab Framework File Roles

## Purpose

This document maps `sgdjscc_lab` files to their roles in the overall framework,
ordered by the actual execution flow.

The primary ordering is:

1. inference framework
2. evaluation framework
3. Phase 4 / 5 extension modules
4. compatibility modules

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

## 3. Phase 4 / 5 Extension Modules

### Phase 3 structural guidance extensions

| File | Role | Figure 1(b) block |
|---|---|---|
| `src/sgdjscc_lab/guidance/depth_extractor.py` | Optional depth-guidance extractor for structural conditioning and evaluation. | Semantic Extractor |
| `src/sgdjscc_lab/guidance/segmentation_extractor.py` | Optional semantic-segmentation extractor for region-aware guidance and analysis. | Semantic Extractor |

### Phase 4 semantic / temporal extensions

| File(s) | Role | Figure 1(b) block |
|---|---|---|
| `src/sgdjscc_lab/controllers/adaptive_guidance_controller.py`, `src/sgdjscc_lab/controllers/snr_guidance_policy.py` | SNR-aware guidance/step control layered on top of the unchanged image forward path. | Cross-block orchestration |
| `src/sgdjscc_lab/guidance/semantic_packet_extractor.py`, `object_extractor.py`, `relation_extractor.py`, `importance_estimator.py` | Build semantic packets from caption/object/relation/attribute/segmentation/depth cues. | Semantic Extractor |
| `src/sgdjscc_lab/evaluators/semantic_packet_matcher.py`, `relation_consistency.py`, `attribute_consistency.py` | Compare original vs reconstructed packets and compute packet-aware semantic consistency. | Outside Figure 1(b) |
| `src/sgdjscc_lab/controllers/regeneration_policy.py` | Chooses failure-mode-aware retry strategies from packet/verifier outputs. | Cross-block orchestration |
| `src/sgdjscc_lab/video/scene_change_detector.py`, `keyframe_extractor.py`, `semantic_delta.py`, `motion_residual.py`, `temporal_pipeline.py` | Keyframe/GOP split, scene-change detection, packet delta logic, temporal reuse, and staged prompt scheduling. | Cross-block orchestration |
| `src/sgdjscc_lab/evaluators/temporal_consistency.py` | Temporal SRS, identity consistency, temporal segmentation IoU, temporal hallucination. | Outside Figure 1(b) |
| `scripts/evaluate_video.py` | Video/keyframe evaluation CLI. | Outside Figure 1(b) |

### Phase 5 channel / acceleration / verifier extensions

| File(s) | Role | Figure 1(b) block |
|---|---|---|
| `src/sgdjscc_lab/channels/rayleigh.py`, `fast_fading.py`, `packet_drop.py`, `measurement.py` | Additional channel models plus receiver-evidence / measurement abstractions for channel-conditioned evaluation. | Wireless Channel |
| `src/sgdjscc_lab/models/channel_condition_encoder.py`, `reliability_head.py`, `diffusion_wrapper_channel.py` | Encodes received-channel evidence into condition features and applies adapter-level channel-conditioned decoding policies. | Diffusion Denoiser / Cross-block orchestration |
| `src/sgdjscc_lab/controllers/channel_condition_policy.py`, `src/sgdjscc_lab/pipelines/channel_conditioned_infer.py` | One-pass channel-conditioned inference path with latent / joint / blind modes. | Cross-block orchestration |
| `src/sgdjscc_lab/acceleration/ddim_sampler.py`, `consistency_decoder.py`, `early_exit.py`, `latency_profiler.py` | Step-budget control, few-step decoding interfaces, intra-sampler early exit, and latency measurement. | Diffusion Denoiser / Outside Figure 1(b) |
| `src/sgdjscc_lab/evaluators/hallucination_vqa.py`, `vqa_backend.py`, `semantic_reliability_v2.py`, `regeneration_search.py` | Stronger semantic verification with local VQA backends, SRS-v2, and multi-strategy regeneration search. | Outside Figure 1(b) |
| `src/sgdjscc_lab/controllers/adaptive_search_policy.py` | Orders regeneration strategies by failure mode and channel state. | Cross-block orchestration |
| `scripts/benchmark_latency.py`, `scripts/benchmark_sampling.py` | Benchmark CLIs for latency / sampling tradeoff experiments. | Outside Figure 1(b) |

These files are opt-in research extensions. They are not part of the default
Phase 1–3 AWGN inference path unless explicitly enabled through config.

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

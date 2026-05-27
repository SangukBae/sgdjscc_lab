# sgdjscc_lab Development Plan

## Purpose

`sgdjscc_lab` is the research and development fork for extending `SGDJSCC`
without modifying the original `SGDJSCC/` package. The original repository is
kept as a read-only reference and paper baseline.

---

## Phase Status

| Phase | Status | Completion Criterion |
|-------|--------|---------------------|
| 1 | ✅ Complete | `python scripts/infer_images.py --config configs/default.yaml` runs AWGN inference |
| 2 | ✅ Complete (구조 분리) | channels / guidance / models / pipelines 분리, 24개 테스트 통과 |
| 2 | ⚠ Scaffold only | config composition: fragment YAML 생성됨, 로더 연결은 Phase 3 |
| 3 | 🔲 Planned | Dataset, SNR, model selection changeable via config only + full metric loop |
| 4 | 🔲 Planned | Video keyframe consistency |
| 5 | 🔲 Planned | Rayleigh fading, DiT/DiTJSCC, Adapter-based condition injection |

---

## Repository Strategy

### `SGDJSCC/`
- original code preservation
- reproduction reference
- paper baseline
- **never modified**

### `sgdjscc_lab/`
- clean research fork
- structural reorganisation (Phase 2 ✅)
- new module integration (Phase 3+)
- experiment and evaluation framework (Phase 3+)

---

## Implemented Directory Structure (Phase 2)

```text
sgdjscc_lab/
  pyproject.toml
  requirements.txt

  configs/
    default.yaml           ← single config used by infer_images.py
    channel/awgn.yaml      ← channel fragment (Phase 5: add rayleigh.yaml)
    model/sgdjscc.yaml     ← model fragment
    infer/awgn.yaml        ← I/O fragment
    eval/default.yaml      ← evaluation fragment (Phase 3+)

  scripts/
    infer_images.py        ← CLI entry (Phase 1/2 ✅)
    evaluate.py            ← stub (Phase 3+)

  src/
    sgdjscc_lab/
      __init__.py          ← version 0.2.0

      config.py            ← load_config, merge_cli_overrides
      io.py                ← list_image_files, load/save image tensor
      runtime.py           ← build_models() assembly point

      channels/
        awgn.py            ← AWGNChannel.transmit() ✅
                           ← Phase 5: add rayleigh.py

      guidance/
        text_extractor.py  ← TextExtractor (BLIP2) ✅
        edge_extractor.py  ← EdgeExtractor (MuGE) ✅
                           ← Phase 3: add depth_extractor.py
                           ← Phase 3: add segmentation_extractor.py

      models/
        jscc_model.py      ← JSCCModel, build_jscc_model ✅
        diffusion_wrapper.py ← build_diffusion_pipeline ✅
        model_bundle.py    ← ModelBundle dataclass ✅

      pipelines/
        infer_pipeline.py  ← run_batch, run_single_image ✅
                           ← Phase 3: add eval_pipeline.py
                           ← Phase 3: add regeneration_loop.py

      evaluators/
        quality.py         ← PSNR, SSIM scaffold ✅ (Phase 3: full impl)
                           ← Phase 3: clip_score.py
                           ← Phase 3: object_preservation.py
                           ← Phase 3: hallucination.py
                           ← Phase 3: semantic_reliability.py

      utils/
        preprocessing.py   ← prepare_patches, merge_patches ✅
        memory.py          ← release_cuda_memory ✅
        seed.py            ← set_global_seed ✅

  tests/
    test_config.py         ← 12 tests (no GPU) ✅
    test_io.py             ← 12 tests (no GPU) ✅
```

---

## Phase 2 Structural Changes

### What changed from Phase 1

| Before (Phase 1) | After (Phase 2) |
|---|---|
| `runtime.py` (monolithic model loader) | `models/jscc_model.py` + `models/diffusion_wrapper.py` + `runtime.py` (assembly shim) |
| `_JSCCModel.channel()` inline AWGN | `channels/awgn.py` `AWGNChannel.transmit()` |
| `_build_caption_model()` in runtime | `guidance/text_extractor.py` `TextExtractor` |
| `_build_canny_net()` in runtime | `guidance/edge_extractor.py` `EdgeExtractor` |
| `pipeline.py` (flat) | `pipelines/infer_pipeline.py` with block helpers |
| `preprocessing.py` (top-level) | `utils/preprocessing.py` |
| `_release_cuda_memory()` in pipeline | `utils/memory.py` `release_cuda_memory()` |
| `_set_seed()` in infer_images.py | `utils/seed.py` `set_global_seed()` |
| No evaluators | `evaluators/quality.py` (PSNR/SSIM scaffold) |
| No pyproject.toml | `pyproject.toml` (editable install) |
| No tests | `tests/` (24 unit tests, no GPU) |

### Compatibility shims

`pipeline.py` and `preprocessing.py` are kept as thin re-export shims so
any code that imports from the old locations continues to work.

### Algorithm preservation

All forward-pass computations are identical to `SGDJSCC/inference_one.py`:
- VAE encode/decode (scaling factor = 15.45)
- AWGN noise injection formula
- Blind SNR estimation via snr_prediction_net
- Continuous/discrete step matching
- Mask token generation
- Canny JSCC retransmission
- Canny latent VAE encoding
- DiffusionGenerator.generate() arguments
- Final `(decode(normalize(denoised)) + 1) / 2` decode

---

## Development Principles

### Principle 1: Interfaces defined in Phase 2

The following interfaces are now usable:

```python
class AWGNChannel:
    def transmit(self, latent: Tensor, snr_db: float) -> Tensor: ...

class TextExtractor:
    def extract(self, image, device, offload_device, offload_after) -> list: ...

class EdgeExtractor:
    def extract(self, image, device, offload_device, offload_after) -> tuple: ...

class JSCCModel(nn.Module):
    def normalize(self, x) -> Tensor: ...
    def channel(self, x) -> Tensor: ...  # delegates to AWGNChannel

# Phase 3+ stubs (follow same pattern):
class DepthExtractor:
    def extract(self, image) -> dict: ...

class SegmentationExtractor:
    def extract(self, image) -> dict: ...
```

### Principle 2: config/interface separation

- Channel type change: swap `awgn.yaml` for `rayleigh.yaml` + replace `AWGNChannel`
- Guide change: swap `edge_extractor` for `depth_extractor` or `segmentation_extractor`
- Model change: replace `jscc_model.py` builder
- Evaluator addition: add a new file under `evaluators/`

---

## Phase 3 Task List

### Config composition 완성 (Phase 2 미완 항목)
- `config.py:load_config()` 에 `defaults` 리스트 지원 추가
  - 예: `default.yaml`에 `defaults: [channel/awgn, model/sgdjscc, infer/awgn]` 기록
  - OmegaConf.merge()로 fragment들을 순서대로 합성
- 이 기능이 완성되면 채널/모델/I/O를 독립적으로 교체 가능해짐

### New evaluators
- `evaluators/clip_score.py` – CLIP image-image + text-image similarity
- `evaluators/object_preservation.py` – object presence rate
- `evaluators/hallucination.py` – POPE-style added-object detection
- `evaluators/semantic_reliability.py` – SRS composite metric

### New guidance
- `guidance/depth_extractor.py` – MiDaS/DPT depth estimation
- `guidance/segmentation_extractor.py` – Segment Anything / SEEM

### New pipelines
- `pipelines/eval_pipeline.py` – SNR sweep + CSV result writing
- `pipelines/regeneration_loop.py` – semantic mismatch detection + re-generation

### Script
- `scripts/evaluate.py` – full evaluation CLI

### Tests
- `tests/test_awgn_channel.py` – noise power verification
- `tests/test_evaluators.py` – PSNR/SSIM output range

---

## Structural Guide Corruption Rules (Phase 3)

Do NOT apply AWGN directly to guide data:

| Data type | Recommended corruption |
|-----------|----------------------|
| Semantic feature tensor / channel symbol | AWGN / Rayleigh fading |
| Canny edge map | dropout, blur, random erasing, salt-pepper |
| Segmentation map | class dropout, region removal, erosion/dilation |
| Caption tokens | token dropout, word replacement |

---

## Phase 5 Rayleigh Extension

To add Rayleigh fading:
1. Add `channels/rayleigh.py` implementing `RayleighChannel.transmit()`
2. Add `configs/channel/rayleigh.yaml`
3. Update `models/jscc_model.py` to accept a channel type argument or
   inject via config

---

## Phase 4: Integrate with `garam`

Do this **only after** the standalone package is stable.

1. Develop independently in `sgdjscc_lab/`
2. Stabilise the experiment API
3. Add a `garam` adapter at the final stage

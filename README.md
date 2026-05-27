# sgdjscc_lab

**Phase 1 complete · Phase 2 complete · Phase 3 planned**

`sgdjscc_lab` is a clean research fork of [SGDJSCC](https://github.com/MauroZMJ/SGDJSCC).
It wraps the original inference pipeline in a modular, config-driven package without
modifying the original code.

> **Important:** This package runs inside the **SGDJSCC original environment** (`ptest` conda env).
> Model code is imported from `../SGDJSCC/` via `sys.path` injection — that directory is never modified.

---

## Current scope

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | AWGN single-image / folder inference, config-driven CLI |
| 2 | ✅ Complete | Modular structure: channels / guidance / models / pipelines separated |
| 3 | 🔲 Planned | Full metric evaluation + config composition (OmegaConf fragment merge) |
| 4 | 🔲 Planned | Video keyframe support |
| 5 | 🔲 Planned | Rayleigh fading, DiT/DiTJSCC |

> **Phase 2 설계 한계 (Phase 3 과제):**
> `configs/channel/`, `configs/model/`, `configs/infer/` 아래 fragment YAML은
> **참고 전용**이며 현재 `config.py` 로더에 연결되어 있지 않습니다.
> Phase 3에서 `load_config()`의 OmegaConf composition 지원과 함께 연결됩니다.

---

## Directory layout (Phase 2)

```
sgdjscc_lab/
├── pyproject.toml              ← editable install (pip install -e .)
├── requirements.txt
├── configs/
│   ├── default.yaml            ← single source of truth for all settings
│   ├── channel/awgn.yaml       ← channel fragment (reference)
│   ├── model/sgdjscc.yaml      ← model fragment (reference)
│   ├── infer/awgn.yaml         ← inference I/O fragment (reference)
│   └── eval/default.yaml       ← evaluation settings (Phase 3+)
├── scripts/
│   ├── infer_images.py         ← CLI entry point
│   └── evaluate.py             ← evaluation stub (Phase 3+)
├── src/
│   └── sgdjscc_lab/
│       ├── config.py           ← YAML loader + CLI override merge
│       ├── io.py               ← file discovery, load, save
│       ├── runtime.py          ← model loading assembly point
│       ├── channels/
│       │   └── awgn.py         ← AWGNChannel (extracted from JSCCModel)
│       ├── guidance/
│       │   ├── text_extractor.py  ← BLIP2 caption extraction
│       │   └── edge_extractor.py  ← MuGE soft edge extraction
│       ├── models/
│       │   ├── jscc_model.py      ← JSCCModel + build_jscc_model
│       │   ├── diffusion_wrapper.py ← MDTv2 + ControlNet loader
│       │   └── model_bundle.py    ← ModelBundle dataclass
│       ├── pipelines/
│       │   └── infer_pipeline.py  ← run_batch / run_single_image
│       ├── evaluators/
│       │   └── quality.py         ← PSNR / SSIM wrappers (Phase 3 scaffold)
│       └── utils/
│           ├── preprocessing.py   ← patch split/merge (CropLongSide + split_image_v2)
│           ├── memory.py          ← release_cuda_memory()
│           └── seed.py            ← set_global_seed()
└── tests/
    ├── test_config.py          ← 12 config tests (no GPU required)
    └── test_io.py              ← 12 I/O tests (no GPU required)
```

---

## Environment setup

```bash
conda create -n ptest python=3.9
conda activate ptest
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 \
    -c pytorch -c nvidia
pip install -r sgdjscc_lab/requirements.txt
```

**Editable install** (optional – needed for `import sgdjscc_lab` without the
`sys.path` prefix):

```bash
pip install -e sgdjscc_lab/
```

---

## Checkpoint placement

Download from [HuggingFace murjun/SGDJSCC](https://huggingface.co/murjun/SGDJSCC/tree/main):

```
sgdjscc_lab/checkpoints/
├── JSCC_model.pth
├── diffusion_backbone.pth
├── diffusion_controlnet.pth      ← needed when use_controlnet: true
└── muge-epoch-19-checkpoint.pth  ← needed when use_semantic: true
```

---

## Running inference

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest

# Infer with default config (SNR=10 dB, ControlNet+text guidance)
python scripts/infer_images.py --config configs/default.yaml

# Override at runtime
python scripts/infer_images.py --config configs/default.yaml \
    --input /path/to/images/ --output /path/to/out/ --snr 5 --device cuda:0
```

Reconstructed images are saved as `<original_stem>.png` in `output_dir`.

---

## Running tests

```bash
conda activate ptest
pip install pytest lark   # one-time setup
cd sgdjscc_lab
python -m pytest tests/test_config.py tests/test_io.py -v
```

All 24 tests pass; no GPU or checkpoints required.

---

## Key config fields (`configs/default.yaml`)

> **경로 해석 주의:** `input_path`, `output_dir`, `model_root`는 `load_config()` 내부에서
> **config 파일의 위치(configs/) 기준**으로 resolve됩니다. 아래 표의 값은 `default.yaml`에
> 기록된 원문(relative)이며, 실제 절대 경로로 변환되어 사용됩니다.

| Field | Value in default.yaml | Resolved from configs/ |
|---|---|---|
| `input_path` | `../inputs/` | `sgdjscc_lab/inputs/` |
| `output_dir` | `../outputs/` | `sgdjscc_lab/outputs/` |
| `model_root` | `../checkpoints/` | `sgdjscc_lab/checkpoints/` |
| `snr_db` | `10` | AWGN channel SNR (dB) |
| `device` | `cuda:0` | |
| `use_semantic` | `true` | Enable diffusion-based denoising |
| `use_text` | `true` | Use BLIP2 text caption guidance |
| `use_controlnet` | `true` | Use ControlNet + edge map guidance |
| `canny_cr` | `"0.2"` | Bandwidth ratio for edge JSCC TX |
| `diffusion_step` | `50` | Denoising steps |
| `guidance_scale` | `4.0` | Classifier-free guidance weight |
| `controlnet_scale` | `0.3` | ControlNet conditioning scale |

---

## Memory considerations

Running the full pipeline (BLIP2 + MuGE + JSCC + MDTv2 + ControlNet) requires ~18 GB
VRAM on a single GPU.  For constrained environments:

- Set `use_text: false` to skip BLIP2 (saves ~6 GB)
- Set `use_controlnet: false` to use backbone only (saves ~4 GB)
- Set `use_semantic: false` for raw JSCC VAE decode only (VRAM minimal)

---

## Input image size

The model processes images as 128×128 patches via `split_image_v2()`.
Patch count is **not** simply `⌈H/128⌉ × ⌈W/128⌉`.  The actual stride formula is:

```
stride = max(128 - (H % 128), 1)   # same for W
```

When H is a multiple of 128 the stride equals 128 (non-overlapping).
When H is **not** a multiple of 128 the stride shrinks and patches overlap heavily:

| Input size | H % 128 | stride | patches (H axis) |
|---|---|---|---|
| 256 × 256 | 0 | 128 | 2 — safe |
| 512 × 512 | 0 | 128 | 4 — safe |
| 192 × 192 | 64 | 64 | 3 (overlap 64 px) |
| 255 × 255 | 127 | 1 | 129 per axis = 16 641 total ⚠ |

**Rule:** resize inputs so that both H and W are multiples of 128 (128, 256, 384, 512 …)
before running inference.  A 255×255 image produces 16 641 patches and will exhaust VRAM.

---

## Phase 3 roadmap

- `evaluators/quality.py` – full PSNR/SSIM/LPIPS integration
- `evaluators/clip_score.py` – image–image + text–image CLIP similarity
- `evaluators/object_preservation.py` – object presence rate
- `evaluators/hallucination.py` – POPE-style added-object detection
- `evaluators/semantic_reliability.py` – SRS composite metric
- `pipelines/eval_pipeline.py` – SNR-sweep evaluation loop
- `scripts/evaluate.py` – full evaluation CLI

---

## Relationship to SGDJSCC

This package imports model code from `../SGDJSCC/` via `sys.path`.
**The original `SGDJSCC/` directory is never modified.**
All algorithms are identical to the original paper implementation.

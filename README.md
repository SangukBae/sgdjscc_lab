# sgdjscc_lab
## Description
This repository contains the modular research fork for `SGDJSCC`.
It preserves the original AWDN image-transmission inference path while adding
config-driven execution, evaluation, and an extensible package structure.

`sgdjscc_lab` runs inside the original `SGDJSCC` environment and imports model
code from `../SGDJSCC/` without modifying the original package.

## Installation
Use the same environment family as the original project.
We use Python 3.9, PyTorch 2.1.0, and CUDA 11.8.

```bash
conda create -n ptest python=3.9
conda activate ptest
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 \
    -c pytorch -c nvidia
pip install -r sgdjscc_lab/requirements.txt
```

Optional editable install:

```bash
pip install -e sgdjscc_lab/
```

## Inference
Before running inference, download the pretrained checkpoints from
[HuggingFace murjun/SGDJSCC](https://huggingface.co/murjun/SGDJSCC/tree/main)
and place them in `sgdjscc_lab/checkpoints/`.

We currently provide AWGN inference for single images or folders.

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest

python scripts/infer_images.py --config configs/default.yaml
```

Composed config example:

```bash
python scripts/infer_images.py --config configs/composed.yaml
```

Runtime override example:

```bash
python scripts/infer_images.py --config configs/composed.yaml \
    --input /path/to/images/ \
    --output /path/to/out/ \
    --snr 5 \
    --device cuda:0
```

## Evaluation
Phase 3 evaluation supports PSNR, SSIM, LPIPS, CLIP-based metrics, SRS, and
SNR-sweep CSV logging.

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest

python scripts/evaluate.py --config configs/composed.yaml --snr 10
python scripts/evaluate.py --config configs/composed.yaml --snr-list -5,0,5,10,15,20,25
python scripts/evaluate.py --config configs/composed.yaml --snr 10 --no-clip
```

Dataset config example:

```bash
python scripts/evaluate.py --config configs/dataset/kodak.yaml
```

## Training

`sgdjscc_lab` provides a **stage-aware** training CLI via `scripts/train.py`.
Each stage has a real, differentiable forward pass + loss; the core baseline is
the paper's three stages (`jscc` → `text_dm` → `controlnet`), plus a supporting
`edge_codec` step and an optional `end_to_end_ft` extension. All existing
inference/evaluation paths are unaffected. Full design:
[docs/training_scaffold.md](./docs/training_scaffold.md). To verify training
actually runs with real models in 1–2 steps: [docs/smoke_training.md](./docs/smoke_training.md).

### Core baseline stages

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest

# Stage 1 — JSCC (image-only, fixed AWGN SNR=10dB)
python scripts/train.py --config configs/composed_train_jscc.yaml \
    --train-list /data/imagenet/train/ --device cuda:0 --epochs 20

# Stage 2 — text-guided DM (caption sidecars)
python scripts/train.py --config configs/composed_train_text_dm.yaml \
    --train-list /data/pairs/train/ --device cuda:0

# Supporting — train the Stage-3 edge codec (BCE+Dice; no heavy checkpoints)
python scripts/train.py --config configs/composed_train_edge_codec.yaml \
    --train-list /data/edges/train/ --device cuda:0 --epochs 50

# Stage 3 — edge ControlNet, BASELINE = dedicated edge_jscc transport
#   (point train.controlnet.edge_jscc.checkpoint at the edge_codec result)
python scripts/train.py --config configs/composed_train_controlnet.yaml \
    --train-list /data/pairs/train/ --device cuda:0
```

`--stage {jscc|text_dm|controlnet|edge_codec|end_to_end_ft}` overrides the
config stage; `--max-steps N` switches to step-based training.

### Dry-run (no checkpoints, no GPU required)

```bash
python scripts/train.py --config configs/composed_train_jscc.yaml \
    --train-list /path/to/images/ --no-models --epochs 1
```

### Resume from checkpoint

```bash
python scripts/train.py --config configs/composed_train_controlnet.yaml \
    --train-list /data/train/ --resume outputs/checkpoints/controlnet/latest.pth
```

### Key config options (`configs/train/default.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `train.epochs` | 10 | Number of training epochs |
| `train.batch_size` | 4 | Batch size |
| `train.lr` | 1e-4 | Learning rate (AdamW) |
| `train.save_every` | 5 | Save `epoch_N.pth` every N epochs |
| `trainable_modules.freeze_*` | `true` | Freeze each module (all frozen by default) |
| `loss.reconstruction_type` | `"l1"` | `"l1"` / `"mse"` / `"huber"` |
| `checkpoint_dir` | `outputs/checkpoints` | Where checkpoints are saved |
| `train_log_path` | `outputs/train_log.jsonl` | JSONL training log |

See [docs/README.md](./docs/README.md) for the full training scaffold design.

## Tests
```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest
python -m pytest tests/ -v
```

## Documentation
For the development roadmap, phase plan, repository strategy, and research
extension guide, see:

- [docs/README.md](./docs/README.md)
- [docs/etri_overview.md](./docs/etri_overview.md)
- [docs/phase4.md](./docs/phase4.md)
- [docs/phase5.md](./docs/phase5.md)
- [docs/framework_comparison.md](./docs/framework_comparison.md)

## TODO List
- [x] Phase 1: AWGN single-image / folder inference.
- [x] Phase 2: Modular package structure and config composition.
- [x] Phase 3: Evaluation framework and research metrics.
- [x] Phase 4: Packet-aware verifier + adaptive guidance (4-A) and keyframe / temporal pipeline (4-B).
- [x] Phase 5 (scaffold): channel-conditioned diffusion (Rayleigh/fast-fading/packet-drop, 5-A), low-latency sampling/consistency/early-exit (5-B), SRS-v2 + regeneration search (5-C).
- [x] Stage-aware training CLI: `scripts/train.py` with the paper's 3 core stages (`jscc`/`text_dm`/`controlnet`), a supporting `edge_codec` stage (BCE+Dice edge codec → Stage-3 `edge_jscc` baseline transport), and an optional `end_to_end_ft` extension; step/epoch modes, grad-accum, AMP, resume, JSONL logging; real-model smoke path (`docs/smoke_training.md`).

## Acknowledgements
The development of `sgdjscc_lab` is based on the original `SGDJSCC` project and
its upstream dependencies:

- [SGDJSCC](https://github.com/MauroZMJ/SGDJSCC)
- [transformer_latent_diffusion](https://github.com/apapiu/transformer_latent_diffusion)
- [MDT](https://github.com/sail-sg/MDT)
- [SwinJSCC](https://github.com/semcomm/SwinJSCC)
- [latent-diffusion](https://github.com/CompVis/latent-diffusion)

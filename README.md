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

`sgdjscc_lab` provides a training CLI scaffold via `scripts/train.py`.
The loop is wired end-to-end (data → forward → loss → checkpoint → log),
but the **forward pass is a placeholder** until a differentiable training
target is selected.  All existing inference/evaluation paths are unaffected.

### Dry-run (no checkpoints, no GPU required)

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest

python scripts/train.py \
    --config configs/composed_train.yaml \
    --train-list /path/to/images/ \
    --no-models --epochs 1
```

### Full run with GPU

```bash
python scripts/train.py \
    --config configs/composed_train.yaml \
    --train-list /data/kodak/train/ \
    --val-list   /data/kodak/val/ \
    --device cuda:0 --epochs 20
```

### Resume from checkpoint

```bash
python scripts/train.py \
    --config configs/composed_train.yaml \
    --train-list /data/train/ \
    --resume outputs/checkpoints/latest.pth
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
- [x] Training CLI scaffold: `scripts/train.py`, config-driven training loop, loss scaffold, checkpoint save/load, JSONL logging, dry-run mode.

## Acknowledgements
The development of `sgdjscc_lab` is based on the original `SGDJSCC` project and
its upstream dependencies:

- [SGDJSCC](https://github.com/MauroZMJ/SGDJSCC)
- [transformer_latent_diffusion](https://github.com/apapiu/transformer_latent_diffusion)
- [MDT](https://github.com/sail-sg/MDT)
- [SwinJSCC](https://github.com/semcomm/SwinJSCC)
- [latent-diffusion](https://github.com/CompVis/latent-diffusion)

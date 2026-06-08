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

## Phase 4: packet-aware & temporal evaluation

Phase 4 adds **opt-in** semantic-packet evaluation, SNR-adaptive guidance, and a
keyframe/temporal video pipeline. All flags default to *off*, so the commands
above keep their Phase-3 behaviour unless you enable them.

### Phase 4-A — packet-aware image evaluation

Enable in `configs/eval/default.yaml` (or your composed config):

```yaml
use_packet_eval: true          # build semantic packets; emit srs_base / srs_packet
use_adaptive_guidance: true    # scale guidance by SNR regime (strong/moderate/weak)
use_packet_regeneration: true  # error-type-aware retries keyed on the packet report
```

```bash
python scripts/evaluate.py --config configs/composed.yaml --snr 0
```

Per image this writes `*.orig_packet.json`, `*.packet.json` and
`*.error_report.json` under `packet_dir`, and adds packet columns
(`srs_base`, `srs_packet`, relation/attribute/segmentation consistency, error
counts, `guidance_regime`) to the results CSV.

### Phase 4-B — keyframe / temporal video evaluation

Process a folder of *ordered* frames (an extracted video sequence):

```bash
# Full run (loads SGD-JSCC + CLIP/BLIP2)
python scripts/evaluate_video.py --config configs/composed_video.yaml \
    --input /path/to/ordered_frames/ --snr 5 --device cuda:0

# Dry run of the keyframe / delta / temporal logic (no checkpoints). Add
# --captions to make packets (and thus semantic delta/metrics) meaningful:
python scripts/evaluate_video.py --config configs/composed_video.yaml \
    --input /path/to/ordered_frames/ --no-models --captions /path/captions.txt
```

Outputs: `keyframes.json` (GOP structure), `temporal_frames.csv` (per-frame log)
and `temporal_metrics.csv` (temporal SRS, object identity consistency, temporal
segmentation IoU, temporal hallucination rate, and `overhead_reduction` — the
semantic-unit saving vs naive per-frame full transmission). Note: the staged
denoising prompt is wired into reconstruction via `cfg.prompt_override` (text
condition), not injected per sampler step — see docs for the precise scope.

See [docs/README.md](./docs/README.md) → "Phase 4 Implementation Status" for the
full module map and reference mapping (FAST-GSC / SGD-JSCC).

## Phase 5: channel conditioning, low-latency, stronger verification

Phase 5 is an **opt-in scaffold** (everything off by default). Highlights:

### 5-A — channel-conditioned diffusion (DiffCom-inspired)

Rayleigh / fast-fading / packet-drop channels with a `MeasurementBundle` of
receiver evidence, and a config-gated channel-conditioned inference path:

```bash
# Rayleigh + channel conditioning + (optional) blind CSI
python scripts/evaluate.py --config configs/composed_phase5.yaml --snr 0
```

Channel/CSI/condition-mode are config-driven (`configs/channel/*.yaml`,
`configs/model/channel_conditioned.yaml`): `channel`, `csi`
(`perfect`/`imperfect`/`none`), `condition_mode`
(`latent_conditioned`/`joint_conditioned`/`blind_conditioned`).

### 5-B — low-latency sampling

DDIM step-budget ablation, few-step consistency decoder interface, checkpoint
early-exit, dynamic SNR→budget routing, and latency benchmarks:

```bash
python scripts/benchmark_latency.py  --config configs/composed.yaml -i ../inputs/test_1.png --steps 50
python scripts/benchmark_sampling.py --config configs/composed.yaml -i ../inputs/test_1.png --steps 50,20,10,5
```

**Intra-sampler early-exit** stops the denoising loop mid-run once a score
plateaus / passes a threshold (no checkpoint re-rendering):

```yaml
acceleration:
  early_exit: true
  early_exit_mode: "intra_sampler"   # or "checkpoint_legacy"
  early_exit_check_interval: 5
  early_exit_metric: "heuristic"     # heuristic (default) | srs | srs_v2
```

Configure via `configs/acceleration/default.yaml` (`acceleration.sampler`,
`sampler_steps`, `early_exit*`, `dynamic_routing`).

### 5-C — SRS-v2 + regeneration search + local VQA

A stronger verifier (`use_srs_v2`) combining base SRS + packet + temporal +
(optionally **VQA**) hallucination, and a multi-strategy regeneration search that
keeps the best reconstruction by the configured `verify_metric`. Enable a real
local VQA backend (lazy-loaded; CLIP fallback if unavailable):

```yaml
use_vqa_hallucination: true
vqa_backend: { type: "blip2" }       # mock | blip2 | llava | mplug | none
```

See `configs/eval/phase5.yaml`.

These flags (`use_channel_conditioning`, `acceleration.*`, `use_srs_v2`,
`use_regeneration_search`) are wired into the main `scripts/evaluate.py` loop.
Channel conditioning runs in a **single pass** (no extra measurement forward): the
channel is sampled once per patch and that exact received latent feeds both the
condition and the decoder.

> Scope note: Phase 5 delivers runnable structure + tests, not trained SOTA.
> **Resolved this iteration**: a real local VQA backend (BLIP-2/LLaVA/mPLUG, CLIP
> fallback); one-pass channel conditioning (no throwaway measurement forward); and
> an intra-sampler early-exit that truly terminates the continuous denoising loop.
> **Still pending**: channel condition *tokens* are not consumed by the frozen
> denoiser; no trained consistency/distilled student; GPU numeric parity of the
> re-driven sampler loop; intra-sampler exit covers the continuous sampler only.
> See [docs/README.md](./docs/README.md) → "Resolved limitations (this iteration)"
> for the precise implemented / wired / approximated / fallback / not-yet breakdown.

## Training
`sgdjscc_lab` currently does **not** provide a standalone training CLI such as
`scripts/train.py`.

Current status:

- inference CLI: available
- evaluation CLI: available
- training CLI: not yet implemented in `sgdjscc_lab`

This means there is **no supported training command** in the package yet.
If training support is added later, it will be documented here as an explicit
CLI command.

For now, the practical workflow is:

```bash
# Inference
python scripts/infer_images.py --config configs/default.yaml

# Evaluation
python scripts/evaluate.py --config configs/composed.yaml --snr 10
```

Notes:

- The original [SGDJSCC README](../SGDJSCC/README.md) also exposes inference only.
- Fine-tuning / training workflow for `sgdjscc_lab` is a future extension item,
  not a completed feature of the current package.

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
- [docs/framework_comparison.md](./docs/framework_comparison.md)

## TODO List
- [x] Phase 1: AWGN single-image / folder inference.
- [x] Phase 2: Modular package structure and config composition.
- [x] Phase 3: Evaluation framework and research metrics.
- [x] Phase 4: Packet-aware verifier + adaptive guidance (4-A) and keyframe / temporal pipeline (4-B).
- [x] Phase 5 (scaffold): channel-conditioned diffusion (Rayleigh/fast-fading/packet-drop, 5-A), low-latency sampling/consistency/early-exit (5-B), SRS-v2 + regeneration search (5-C).

## Acknowledgements
The development of `sgdjscc_lab` is based on the original `SGDJSCC` project and
its upstream dependencies:

- [SGDJSCC](https://github.com/MauroZMJ/SGDJSCC)
- [transformer_latent_diffusion](https://github.com/apapiu/transformer_latent_diffusion)
- [MDT](https://github.com/sail-sg/MDT)
- [SwinJSCC](https://github.com/semcomm/SwinJSCC)
- [latent-diffusion](https://github.com/CompVis/latent-diffusion)

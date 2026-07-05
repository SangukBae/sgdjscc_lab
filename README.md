# sgdjscc_lab
## 개요
**semantic media transmission reliability**를 위한 `SGDJSCC`의 모듈형 연구 fork.
원본 AWGN 이미지 전송 inference 경로를 그대로 보존하고(forward pass 수치 동일),
그 위에 config 기반 실행, semantic 평가 suite(SRS), stage-aware training, opt-in
Phase 4/5 확장(packet-aware 평가, video/temporal, non-AWGN 채널, channel-conditioned
및 low-latency decoding)을 추가한다.

`sgdjscc_lab`은 원본 `SGDJSCC` 환경에서 실행되며 `../SGDJSCC/`의 모델 코드를 원본
패키지 수정 없이 import한다. 모든 확장은 **기본 off**라 `use_phase4`/`use_phase5`를
설정하지 않으면 동작이 원본 inference 경로와 동일하다.

## 설치
원본 프로젝트와 같은 환경 계열을 사용한다. Python 3.9, PyTorch 2.1.0, CUDA 11.8.

```bash
conda create -n ptest python=3.9
conda activate ptest
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 \
    -c pytorch -c nvidia
pip install -r sgdjscc_lab/requirements.txt
```

editable install(선택):

```bash
pip install -e sgdjscc_lab/
```

## Inference
inference 실행 전, pretrained checkpoint를
[HuggingFace murjun/SGDJSCC](https://huggingface.co/murjun/SGDJSCC/tree/main)에서
받아 `sgdjscc_lab/checkpoints/`에 둔다.

단일 이미지/폴더 AWGN inference(non-AWGN 채널은 Phase 5 opt-in):

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest

python scripts/infer_images.py --config configs/default.yaml       # flat config
python scripts/infer_images.py --config configs/composed.yaml       # _defaults_ composition

# 런타임 override
python scripts/infer_images.py --config configs/composed.yaml \
    --input /path/to/images/ --output /path/to/out/ --snr 5 --device cuda:0
```

## Evaluation
Phase 3 평가는 PSNR, SSIM, LPIPS, CLIP 기반 지표, SRS, SNR-sweep CSV 로깅을 지원한다.

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest

python scripts/evaluate.py --config configs/composed.yaml --snr 10
python scripts/evaluate.py --config configs/composed.yaml --snr-list -5,0,5,10,15,20,25
python scripts/evaluate.py --config configs/composed.yaml --snr 10 --no-clip
```

데이터셋 config 예시:

```bash
python scripts/evaluate.py --config configs/dataset/kodak.yaml
```

## Training

`sgdjscc_lab`은 `scripts/train.py`로 **stage-aware** training CLI를 제공한다.
각 stage는 실제 미분 가능한 forward pass + loss를 가진다. core baseline은 논문의
3개 stage(`jscc` → `text_dm` → `controlnet`)에 보조 `edge_codec` step과 선택적
`end_to_end_ft` 확장을 더한 것이다. 기존 inference/evaluation 경로는 영향받지 않는다.
전체 설계: [docs/training_scaffold.md](./docs/training_scaffold.md). 실제 모델로
1–2 step training이 도는지 확인: [docs/smoke_training.md](./docs/smoke_training.md).

### Core baseline stage

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest

# Stage 1 — JSCC (image-only, 고정 AWGN SNR=10dB)
python scripts/train.py --config configs/composed_train_jscc.yaml \
    --train-list /data/imagenet/train/ --device cuda:0 --epochs 20

# Stage 2 — text-guided DM (caption sidecar)
python scripts/train.py --config configs/composed_train_text_dm.yaml \
    --train-list /data/pairs/train/ --device cuda:0

# 보조 — Stage-3 edge codec 학습 (BCE+Dice; heavy checkpoint 불필요)
python scripts/train.py --config configs/composed_train_edge_codec.yaml \
    --train-list /data/edges/train/ --device cuda:0 --epochs 50

# Stage 3 — edge ControlNet, BASELINE = 전용 edge_jscc transport
#   (train.controlnet.edge_jscc.checkpoint를 edge_codec 결과로 지정)
python scripts/train.py --config configs/composed_train_controlnet.yaml \
    --train-list /data/pairs/train/ --device cuda:0
```

`--stage {jscc|text_dm|controlnet|edge_codec|csi_estimation|end_to_end_ft}`로
config stage를 override하고, `--max-steps N`으로 step 기반 training으로 전환한다.
Multi-GPU: `torchrun --standalone --nproc_per_node=N scripts/train.py …`.

### Dry-run (checkpoint·GPU 불필요)

```bash
python scripts/train.py --config configs/composed_train_jscc.yaml \
    --train-list /path/to/images/ --no-models --epochs 1
```

### Checkpoint에서 resume

```bash
python scripts/train.py --config configs/composed_train_controlnet.yaml \
    --train-list /data/train/ --resume outputs/checkpoints/controlnet/latest.pth
```

### 주요 config 옵션 (`configs/train/default.yaml`)

| Key | Default | 설명 |
|-----|---------|-------------|
| `train.epochs` | 10 | training epoch 수 |
| `train.batch_size` | 4 | batch size |
| `train.lr` | 1e-4 | learning rate (AdamW) |
| `train.save_every` | 5 | N epoch마다 `epoch_N.pth` 저장 |
| `trainable_modules.freeze_*` | `true` | 각 모듈 freeze (기본 전부 freeze) |
| `loss.reconstruction_type` | `"l1"` | `"l1"` / `"mse"` / `"huber"` |
| `checkpoint_dir` | `outputs/checkpoints` | checkpoint 저장 위치 |
| `train_log_path` | `outputs/train_log.jsonl` | JSONL training log |

전체 training scaffold 설계는 [docs/README.md](./docs/README.md) 참조.

## Tests
```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest
python -m pytest tests/ -v
```

## 문서
전체 문서 맵은 [docs/README.md](./docs/README.md) 참조. 주요 항목:

- [docs/etri_overview.md](./docs/etri_overview.md) — 프로젝트 목표, pipeline, SRS, 실험 설정
- [docs/phase4.md](./docs/phase4.md) / [docs/phase5.md](./docs/phase5.md) — 확장 설계 & 상태
- [docs/training_scaffold.md](./docs/training_scaffold.md) — stage-aware training
- [docs/framework_comparison.md](./docs/framework_comparison.md) — 원본 vs lab 구조 + 논문 충실도
- [docs/paper_gap_closure.md](./docs/paper_gap_closure.md) — `paper_mode` guardrail & DDP

## TODO List
- [x] Phase 1: AWGN 단일 이미지 / 폴더 inference.
- [x] Phase 2: 모듈형 패키지 구조와 config composition.
- [x] Phase 3: 평가 프레임워크와 연구 지표.
- [x] Phase 4: packet-aware verifier + adaptive guidance (4-A), keyframe / temporal pipeline (4-B).
- [x] Phase 5 (scaffold): channel-conditioned diffusion (Rayleigh/fast-fading/packet-drop, 5-A), low-latency sampling/consistency/early-exit (5-B), SRS-v2 + regeneration search (5-C).
- [x] Stage-aware training CLI: `scripts/train.py`, 논문 3개 core stage(`jscc`/`text_dm`/`controlnet`), 보조 `edge_codec` stage(BCE+Dice edge codec → Stage-3 `edge_jscc` baseline transport), 선택적 `end_to_end_ft` 확장; step/epoch 모드, grad-accum, AMP, resume, JSONL 로깅; 실제 모델 smoke 경로(`docs/smoke_training.md`).

## Acknowledgements
`sgdjscc_lab`의 개발은 원본 `SGDJSCC` 프로젝트와 그 상위 의존성에 기반한다:

- [SGDJSCC](https://github.com/MauroZMJ/SGDJSCC)
- [transformer_latent_diffusion](https://github.com/apapiu/transformer_latent_diffusion)
- [MDT](https://github.com/sail-sg/MDT)
- [SwinJSCC](https://github.com/semcomm/SwinJSCC)
- [latent-diffusion](https://github.com/CompVis/latent-diffusion)

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

`scripts/train.py`는 논문 3-stage(`jscc` → `text_dm` → `controlnet`)에 보조
`edge_codec`/`csi_estimation`과 선택적 `end_to_end_ft`를 더한 **stage-aware** CLI다.
inference/evaluation 경로는 영향받지 않는다.

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab && conda activate ptest

python scripts/train.py --config configs/composed_train_jscc.yaml \
    --train-list /data/imagenet/train/ --device cuda:0 --epochs 20
```

`--stage`로 stage override, `--max-steps N`으로 step 모드, `--no-models`로 GPU 없는
dry-run, `--resume latest`로 재개, Multi-GPU는 `torchrun`. 전체 stage·config·freeze·
export·DDP는 [docs/training_scaffold.md](./docs/training_scaffold.md), 실제 모델로
1–2 step 배선 검증은 [docs/smoke_training.md](./docs/smoke_training.md) 참조.

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

Phase 1~4 완료, Phase 5 스캐폴드. 상세 현황은 [docs/README.md](./docs/README.md#phase-현황) 참조.

## Acknowledgements
`sgdjscc_lab`의 개발은 원본 `SGDJSCC` 프로젝트와 그 상위 의존성에 기반한다:

- [SGDJSCC](https://github.com/MauroZMJ/SGDJSCC)
- [transformer_latent_diffusion](https://github.com/apapiu/transformer_latent_diffusion)
- [MDT](https://github.com/sail-sg/MDT)
- [SwinJSCC](https://github.com/semcomm/SwinJSCC)
- [latent-diffusion](https://github.com/CompVis/latent-diffusion)

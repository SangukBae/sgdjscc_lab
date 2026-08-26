# SGD-JSCC Lab

- 생성형 AI 기반 이미지·영상 시맨틱 통신 연구 프레임워크다. SGD-JSCC 복원 경로에
  전송 packet, 의미 신뢰도 평가, 할루시네이션 검증, 영상 keyframe 처리를 결합한다.
  기존 추론 경로는 유지하며 연구 기능은 config로 선택한다.

## 주요 기능

- 이미지·영상 JSCC 추론과 AWGN/Rayleigh/fading/drop 채널 실험
- PSNR, SSIM, LPIPS, CLIP, SRS와 영상 시간축 지표 평가
- semantic packet 검증과 재생성 후보 탐색
- keyframe 기반 영상 복원과 외부 Wan/SVD worker 연동
- 4/6/8/16-bit packet 직렬화와 정확한 bundle byte 집계
- stage별 학습, checkpoint export, 단일·다중 GPU 실행

> Bundle byte는 실제 직렬화 크기다. 변조·FEC·물리 채널 symbol 수는 현재 proxy이며
> 실제 전송량과 구분해 해석해야 한다.

## 실행 환경

| 항목 | 권장 환경 |
|---|---|
| OS | Linux |
| Python | 3.9 |
| PyTorch | 2.1.0 |
| CUDA | 11.8 |
| GPU | CUDA GPU 권장 |

- 주요 패키지는 `diffusers==0.26.3`, `transformers==4.44.2`,
  `numpy==1.23.2`, `openai-clip`, `lpips`다. 전체 버전은
  [requirements.txt](./requirements.txt)를 따른다. 모델 구현을 불러오기 위해 원본
  `SGDJSCC/`가 기본적으로 이 저장소의 형제 경로 `../SGDJSCC/`에 있어야 한다.

## 설치

```bash
cd /path/to/sgdjscc_lab

conda create -n ptest python=3.9
conda activate ptest
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt
pip install -e .
```

- editable install 후 `sgdjscc-infer`, `sgdjscc-evaluate`, `sgdjscc-evaluate-video`,
  `sgdjscc-train` 명령도 사용할 수 있다.

## 모델과 데이터 준비

- [murjun/SGDJSCC](https://huggingface.co/murjun/SGDJSCC/tree/main)에서 다음 파일을
  받아 `checkpoints/`에 둔다.

```text
checkpoints/
├── JSCC_model.pth
├── diffusion_backbone.pth
├── diffusion_controlnet.pth
└── muge-epoch-19-checkpoint.pth
```

- 이미지와 영상 입력은 config 또는 `--input`으로 지정한다. 데이터 역할은
  [데이터셋 지침](./docs/protocols/datasets.md), baseline/custom checkpoint 구분은
  [재현성 지침](./docs/protocols/reproducibility.md)을 참고한다.

- 저장소 밖의 대용량 파일은 아래 환경변수로 경로를 바꿀 수 있다.

```bash
export SGDJSCC_ROOT=/path/to/SGDJSCC
export SGDJSCC_DATA_ROOT=/path/to/data
export SGDJSCC_MODEL_ROOT=/path/to/models
export SGDJSCC_RUN_ROOT=/path/to/outputs
export SGDJSCC_CACHE_ROOT=/path/to/cache
```

## 사용법

- 모든 명령은 `sgdjscc_lab/`에서 실행한다.

### 이미지 추론

```bash
python scripts/infer_images.py \
    --config configs/recipes/inference/composed.yaml \
    --input /path/to/images \
    --output outputs/inference \
    --snr 5 --device cuda:0
```

### 이미지 평가

```bash
# 단일 SNR
python scripts/evaluate.py \
    --config configs/recipes/inference/composed.yaml --snr 10

# SNR sweep
python scripts/evaluate.py \
    --config configs/recipes/inference/composed.yaml \
    --snr-list=-5,0,5,10,15,20,25
```

### 영상 평가

```bash
python scripts/evaluate_video.py \
    --config configs/recipes/video/composed_video.yaml \
    --input data/etri_video_eval/processed/01_person_walk.mp4 \
    --snr 5 --device cuda:0 --save-video
```

- ETRI 10개 영상 일괄 평가는 다음 명령으로 실행한다.

```bash
python scripts/run_etri_video_eval.py --stages all --snr 5 --device cuda:0
python scripts/summarize_etri_video_eval.py --output-root outputs/etri_video_eval
python scripts/generate_etri_final_report.py --output-root outputs/etri_video_eval
```

### 전송 packet 비교

```bash
python scripts/run_transmission_reduction_eval.py \
    --configs fixed_awgn,fixed_int16,fixed_int8,fixed_int6,fixed_int4,skem_int16,skem_int8,skem_int6,skem_int4 \
    --device cuda:0 \
    --output-root outputs/transmission_reduction
```

- 현재 `int4`는 analog AWGN 임시 기준의 픽셀 품질 gate를 통과한 잠정 후보다.
  reliable-digital 기준과 SRS·할루시네이션 평가 전에는 기본 operating point로
  확정하지 않는다.

### 학습

```bash
# Stage 1: JSCC
python scripts/train.py \
    --config configs/recipes/training/composed_train_jscc.yaml \
    --train-list /path/to/train --val-list /path/to/val --device cuda:0

# Stage 3: ControlNet
python scripts/train.py \
    --config configs/recipes/training/composed_train_controlnet.yaml \
    --train-list /path/to/train --resume latest --device cuda:0

# 모델을 로드하지 않는 설정 검증
python scripts/train.py \
    --config configs/recipes/training/composed_train_jscc.yaml \
    --train-list /path/to/train --no-models
```

- 다중 GPU는 `torchrun --standalone --nproc_per_node=N scripts/train.py ...` 형식으로
  실행한다. stage별 입력과 export 방식은 [학습 지침](./docs/protocols/training.md)에 있다.

### 테스트

```bash
python -m pytest tests/ -q
```

## 주요 출력

- 이미지: 복원 이미지와 이미지·SNR별 품질/의미 지표 CSV
- 영상: 복원 프레임·MP4, `temporal_metrics.csv`, `segments.json`, 요약 리포트
- 전송: 직렬화 `.sgbundle`, packet 구성 byte, Pareto 결과
- 학습: `outputs/checkpoints/<stage>/`의 checkpoint와 `train_log.jsonl`

## 프로젝트 구조

```text
configs/       기본·실험 설정
data/          데이터 설명과 소규모 평가셋
docs/          현재 상태·설계·절차·실험 기록
scripts/       추론·평가·학습 진입점
src/           sgdjscc_lab 패키지
tests/         CPU 중심 자동 테스트
transmission/  양자화·packet 직렬화
```

- 현재 구현 범위와 남은 한계는 [현재 상태](./docs/current/status.md),
  [향후 계획](./docs/current/roadmap.md), [알려진 문제](./docs/current/open_issues.md)를
  기준으로 판단한다.

## 문서 안내

| 문서 | 설명 |
|---|---|
| [docs/README.md](./docs/README.md) | 전체 문서 색인 |
| [docs/architecture/metrics.md](./docs/architecture/metrics.md) | 지표 정의 |
| [docs/architecture/system.md](./docs/architecture/system.md) | 시스템 구조 |
| [docs/architecture/tx_rx_contract.md](./docs/architecture/tx_rx_contract.md) | Tx/Rx 계약 |
| [docs/archive/etri_implementation_log.md](./docs/archive/etri_implementation_log.md) | 구현 이력 |
| [docs/archive/framework_comparison.md](./docs/archive/framework_comparison.md) | 과거 프레임워크 비교 |
| [docs/archive/limitation_reference_map.md](./docs/archive/limitation_reference_map.md) | 과거 한계 근거 |
| [docs/archive/paper_gap_closure.md](./docs/archive/paper_gap_closure.md) | 과거 논문 보완안 |
| [docs/archive/paper_training_alignment.md](./docs/archive/paper_training_alignment.md) | 과거 학습 정합성 |
| [docs/archive/phase4_2026-07.md](./docs/archive/phase4_2026-07.md) | Phase 4 기록 |
| [docs/archive/phase5_2026-07.md](./docs/archive/phase5_2026-07.md) | Phase 5 기록 |
| [docs/archive/phases_1to3.md](./docs/archive/phases_1to3.md) | Phase 1–3 기록 |
| [docs/archive/video_extension_lgvsc_2026-07.md](./docs/archive/video_extension_lgvsc_2026-07.md) | 과거 LGVSC 설계 |
| [docs/current/open_issues.md](./docs/current/open_issues.md) | 알려진 문제 |
| [docs/current/roadmap.md](./docs/current/roadmap.md) | 향후 계획 |
| [docs/current/status.md](./docs/current/status.md) | 현재 상태 |
| [docs/experiments/2026-07-17_stage1_video_pipeline.md](./docs/experiments/2026-07-17_stage1_video_pipeline.md) | 영상 1차 검증 |
| [docs/experiments/2026-07-24_video_speed_optimization.md](./docs/experiments/2026-07-24_video_speed_optimization.md) | 영상 속도 실험 |
| [docs/experiments/2026-07-28_owlv2_vqa_calibration.md](./docs/experiments/2026-07-28_owlv2_vqa_calibration.md) | Presence 보정 실험 |
| [docs/experiments/2026-07_lgvsc_1b_worker_validation.md](./docs/experiments/2026-07_lgvsc_1b_worker_validation.md) | LGVSC worker 검증 |
| [docs/experiments/2026-07_lgvsc_1c_reproduction.md](./docs/experiments/2026-07_lgvsc_1c_reproduction.md) | LGVSC 재현 준비 |
| [docs/experiments/2026-07_lgvsc_psss_skem.md](./docs/experiments/2026-07_lgvsc_psss_skem.md) | PSSS/SKEM 검증 |
| [docs/experiments/2026-08-16_remote_hq_validation.md](./docs/experiments/2026-08-16_remote_hq_validation.md) | 원격 HQ 검증 |
| [docs/experiments/2026-08-18_transmission_reduction.md](./docs/experiments/2026-08-18_transmission_reduction.md) | 전송량 실험 |
| [docs/protocols/datasets.md](./docs/protocols/datasets.md) | 데이터 지침 |
| [docs/protocols/evaluation.md](./docs/protocols/evaluation.md) | 평가 지침 |
| [docs/protocols/reproducibility.md](./docs/protocols/reproducibility.md) | 재현성 지침 |
| [docs/protocols/training.md](./docs/protocols/training.md) | 학습 지침 |
| [docs/protocols/video_rate_benchmark.md](./docs/protocols/video_rate_benchmark.md) | 전송률 비교 절차 |
| [docs/reference/framework_file_roles.md](./docs/reference/framework_file_roles.md) | 파일 역할 지도 |
| [docs/reference/paper_alignment.md](./docs/reference/paper_alignment.md) | 논문 정합성 |
| [docs/reference/paper_writing_notes.md](./docs/reference/paper_writing_notes.md) | 논문 작성 메모 |
| [docs/reports/2026-08-16_etri/Appendix_Slide_Explanations_EN.md](./docs/reports/2026-08-16_etri/Appendix_Slide_Explanations_EN.md) | 영문 부록 설명 |
| [docs/reports/2026-08-16_etri/Slide_Detailed_Notes_External_EN.md](./docs/reports/2026-08-16_etri/Slide_Detailed_Notes_External_EN.md) | 영문 외부 발표 노트 |
| [docs/reports/2026-08-16_etri/Slide_Detailed_Notes_Internal_EN.md](./docs/reports/2026-08-16_etri/Slide_Detailed_Notes_Internal_EN.md) | 영문 내부 발표 노트 |
| [docs/reports/2026-08-16_etri/부록슬라이드_상세설명_KO.md](./docs/reports/2026-08-16_etri/부록슬라이드_상세설명_KO.md) | 국문 부록 설명 |
| [docs/reports/2026-08-16_etri/슬라이드상세설명_내부용_KO.md](./docs/reports/2026-08-16_etri/슬라이드상세설명_내부용_KO.md) | 국문 내부 발표 노트 |
| [docs/reports/2026-08-16_etri/슬라이드상세설명_외부공유용_KO.md](./docs/reports/2026-08-16_etri/슬라이드상세설명_외부공유용_KO.md) | 국문 외부 발표 노트 |
| [docs/reports/README.md](./docs/reports/README.md) | 보고 자료 안내 |
| [docs/reports/etri_qna_reply.md](./docs/reports/etri_qna_reply.md) | 과제 Q&A |

# sgdjscc_lab

무선 채널 잡음을 통과한 뒤에도 이미지·영상의 **의미(semantic intent)**가 얼마나
보존되는지 측정하는, `SGDJSCC` 기반 End-to-End semantic 전송 시뮬레이션 프레임워크다.
목표는 픽셀 화질이 아니라 **전송량을 줄이면서 복원 성능과 의미 신뢰도를 유지하고,
할루시네이션(없던 객체 생성/누락)을 억제하는 것**이다.

## 핵심 기능

- **원본 SGDJSCC 경로 보존** — VAE latent 인코딩 + AWGN 채널 + diffusion 복원의
  forward-pass 수치는 원본과 동일. 모든 확장은 opt-in.
- **Semantic Reliability Score(SRS)** — PSNR/SSIM과 별개로 CLIP 유사도, 객체
  보존/누락/추가율을 합쳐 의미 보존을 점수화.
- **패킷 인식 검증·재생성** — 전송 semantic packet과 복원 결과를 비교해 오류
  유형(추가/누락/구조 왜곡)별로 재생성을 조정하는 Packet Verifier.
- **영상 확장** — 키프레임 + reuse/recompute/generate 3-way 분기, 시간축 지표
  (`PTC`/`SFR`/`SDI`), LGVSC 논문을 참고한 segment 단위 생성 복원.
- **실제 전송량 절감** — 4/6/8/16-bit 실제 binary packet 양자화(`transmission/`)로
  채널 심볼/비트 단위 전송량을 직접 줄이고 측정.
- **비-AWGN 채널·가속** — Rayleigh/fast-fading/packet-drop, SNR 적응형 가이던스,
  저지연 샘플링(모두 opt-in, 기본 off).

## 동작 구조

```
원본 이미지/키프레임
  → Tx: JSCC 인코더 (VAE latent + Canny 구조 가이드)
  → 무선 채널 (AWGN 기본; Rayleigh/fast-fading/packet-drop opt-in)
  → Rx: diffusion 복원 (MDTv2 + 선택적 ControlNet)
  → 의미 일치·할루시네이션 평가 (SRS, packet verifier, 시간축 지표)
  → CSV/리포트
```

## 지원 환경 및 주요 의존성

- Python 3.9, PyTorch 2.1.0 + torchvision 0.16.0, CUDA 11.8(conda)
- `diffusers` 0.26.3, `transformers` 4.44.2, `numpy` 1.23.2, `openai-clip`, `lpips`,
  `clean-fid`/`torch-fidelity` — 전체 버전은 [requirements.txt](./requirements.txt) 참고
- 원본 `SGDJSCC/` 저장소가 `../SGDJSCC/`(또는 `SGDJSCC_ROOT` 환경변수)에 있어야
  모델 코드를 import할 수 있다

## 설치

```bash
conda create -n ptest python=3.9
conda activate ptest
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 \
    -c pytorch -c nvidia
pip install -r sgdjscc_lab/requirements.txt
```

editable install(선택 — 되면 `sgdjscc-infer`/`sgdjscc-train`/`sgdjscc-evaluate`/
`sgdjscc-evaluate-video` 커맨드도 같은 스크립트를 그대로 실행한다):

```bash
pip install -e sgdjscc_lab/
```

## Checkpoint와 데이터 준비

[HuggingFace `murjun/SGDJSCC`](https://huggingface.co/murjun/SGDJSCC/tree/main)에서
`JSCC_model.pth`, `diffusion_backbone.pth`, `diffusion_controlnet.pth`,
`muge-epoch-19-checkpoint.pth`를 받아 `sgdjscc_lab/checkpoints/`에 둔다.

데이터셋 준비/역할 매핑은 [docs/dataset_status.md](./docs/dataset_status.md),
ETRI 10-영상 평가셋은 [data/etri_video_eval/README.md](./data/etri_video_eval/README.md)
참고. 이미지는 128×128 배수 크기로 리사이즈해 타일링한다
(예시: [configs/base/dataset/kodak.yaml](./configs/base/dataset/kodak.yaml)).

대용량 데이터/checkpoint/실행결과를 저장소 밖에 둘 수도 있다 — 환경변수
(`SGDJSCC_DATA_ROOT`/`SGDJSCC_MODEL_ROOT`/`SGDJSCC_RUN_ROOT`/`SGDJSCC_CACHE_ROOT`/
`SGDJSCC_ROOT`)를 설정하지 않으면 기존 저장소 내부 경로 그대로 동작한다. 자세한
규칙은 `src/sgdjscc_lab/paths.py` 참고.

## 이미지 추론

```bash
cd sgdjscc_lab && conda activate ptest

python scripts/infer_images.py --config configs/base/default.yaml
python scripts/infer_images.py --config configs/recipes/inference/composed.yaml \
    --input /path/to/images/ --output /path/to/out/ --snr 5 --device cuda:0
```

## 이미지 평가

```bash
python scripts/evaluate.py --config configs/recipes/inference/composed.yaml --snr 10
python scripts/evaluate.py --config configs/recipes/inference/composed.yaml \
    --snr-list -5,0,5,10,15,20,25
python scripts/evaluate.py --config configs/recipes/inference/composed.yaml \
    --snr 10 --no-clip
```

`--profile {paper,extended,full}`로 지표 집합을, `--require-real-fid`로 FID 실측을
강제할 수 있다.

## 비디오 평가

```bash
python scripts/evaluate_video.py --config configs/recipes/video/composed_video.yaml \
    --snr 10 --device cuda:0
```

Phase 4-B 키프레임/시간적 파이프라인은 기본 off(`use_phase4`)다. `--no-clip`,
`--force-interframe-reuse`, `--recon-caption-mode {own,skip}`, `--packet-cache-dir`
등으로 세부 동작을 제어한다. ETRI 10-영상 배치 실행은
[docs/README.md](./docs/README.md#7-개별-실험-결과)의 개별 실험 결과 문서들과
`scripts/run_etri_video_eval.py` 참고.

## 학습

`scripts/train.py`는 논문 3-stage(`jscc` → `text_dm` → `controlnet`)에 보조
`edge_codec`/`csi_estimation`과 선택적 `end_to_end_ft`를 더한 stage-aware CLI다.
inference/evaluation 경로는 영향받지 않는다.

```bash
python scripts/train.py --config configs/recipes/training/composed_train_jscc.yaml \
    --train-list /path/to/train/ --val-list /path/to/val/ --device cuda:0
python scripts/train.py --config configs/recipes/training/composed_train_controlnet.yaml \
    --resume latest
python scripts/train.py --config configs/recipes/training/composed_train_jscc.yaml \
    --no-models   # GPU 없는 dry-run
```

`--stage`로 override, `--max-steps`/`--epochs`로 종료 조건, Multi-GPU는
`torchrun --standalone --nproc_per_node=N scripts/train.py ...`. 전체 stage·config·
freeze·export·DDP는 [docs/training_scaffold.md](./docs/training_scaffold.md), 실제
모델 1~2 step 배선 검증은 [docs/dev/smoke_training.md](./docs/dev/smoke_training.md).

## 테스트

```bash
python -m pytest tests/ -v
```

## 주요 출력 결과

- 이미지 평가: 이미지 × SNR별 한 행으로 `psnr, ssim, lpips, clip_image_image,
  clip_text_image, object_preservation_rate, missing_object_rate,
  additional_object_rate, hallucination_score, semantic_reliability_score`
  (`src/sgdjscc_lab/utils/csv_logger.py::RESULT_COLUMNS`)를 CSV로 기록.
- 영상 평가: `temporal_metrics.csv`(`PTC`/`SFR`/`SDI`), `segments.json`,
  `summary_metrics.csv/md`, `final_report.md` — 예시는 `outputs/etri_video_eval/`.
- 전송량 절감 PoC: `accounting_summary.json`(bit/symbol 절감률),
  `rate_reliability_curve.csv`(전송량-신뢰도 trade-off).

## 현재 구현 범위와 중요한 한계

Phase 1~4(AWGN 추론, 모듈 분리, 평가기 세트, 패킷 검증+영상 파이프라인)는 완료,
Phase 5(채널 조건화, 저지연 샘플링, SRS-v2)는 구조 완성·일부 실모델 검증 진행 중이다.
전송량 절감은 실제 4~16bit binary packet 양자화(`transmission/`)까지 구현됐고, 10개
영상 실험에서 4-bit 양자화가 평균 ΔPSNR −0.13dB로 화질 저하 없이 선택됐다(영상별
분산·SRS 변화는 추가 검증 필요). 할루시네이션 검증은 Packet Verifier가 오류 유형을
판정·기록하지만, 그 판정을 실제 diffusion 샘플러(negative prompt 등)에 반영하는
배선은 아직 없다 — 판정과 로그까지만 완료다. LGVSC 참고 영상 생성 확장은 재현
baseline 실행 준비까지 완료했고 실제 10-영상 실행 결과는 아직 없다. 정확한
완료/스캐폴드/미구현 구분은 [docs/etri_strategy.md](./docs/etri_strategy.md)(현재
상태)와 [docs/roadmap.md](./docs/roadmap.md)(향후 계획) 참고.

## Acknowledgements

`sgdjscc_lab`의 개발은 원본 `SGDJSCC` 프로젝트와 그 상위 의존성에 기반한다:

- [SGDJSCC](https://github.com/MauroZMJ/SGDJSCC)
- [transformer_latent_diffusion](https://github.com/apapiu/transformer_latent_diffusion)
- [MDT](https://github.com/sail-sg/MDT)
- [SwinJSCC](https://github.com/semcomm/SwinJSCC)
- [latent-diffusion](https://github.com/CompVis/latent-diffusion)

## 문서

아래는 활성 문서 목록이다. 과거 완료 기록(`docs/archive/`)과 발표·보고 자료
(`docs/reports/`)를 포함한 전체 색인은 [docs/README.md](./docs/README.md) 참고.

| 문서 | 설명 |
|---|---|
| [docs/README.md](./docs/README.md) | 전체 문서 색인 (archive/reports/notes 포함) |
| [docs/etri_overview.md](./docs/etri_overview.md) | 프로젝트 목표, 파이프라인, SRS, 실험 설정 |
| [docs/etri_strategy.md](./docs/etri_strategy.md) | 핵심 한계 3가지와 현재 구현 상태 |
| [docs/roadmap.md](./docs/roadmap.md) | 향후 연구개발 계획 (연구 목표 기준), 일정, ETRI 협의 필요사항 |
| [docs/phase4.md](./docs/phase4.md) | Phase 4: 패킷 검증기 + 적응형 가이드, 키프레임/시간적 파이프라인 |
| [docs/phase5.md](./docs/phase5.md) | Phase 5: 채널 조건화, 저지연 샘플링, SRS-v2 |
| [docs/paper_alignment.md](./docs/paper_alignment.md) | 논문 정합성, `paper_mode`, 하이퍼파라미터 출처 |
| [docs/framework_file_roles.md](./docs/framework_file_roles.md) | 파일별 실행 흐름과 역할 지도 |
| [docs/video_extension_lgvsc.md](./docs/video_extension_lgvsc.md) | LGVSC 논문을 참고한 비디오 확장 시스템 구조 |
| [docs/training_scaffold.md](./docs/training_scaffold.md) | stage-aware 학습 CLI 상세 |
| [docs/checkpoint_usage.md](./docs/checkpoint_usage.md) | checkpoint 경로, export, 로컬/원격 가중치 |
| [docs/dataset_status.md](./docs/dataset_status.md) | 데이터셋 역할·stage 매핑·변환 워크플로 |
| [docs/dev/smoke_training.md](./docs/dev/smoke_training.md) | 실제 모델 1~2 step 학습 배선 검증 |
| [docs/etri_stage1_validation.md](./docs/etri_stage1_validation.md) | 1차 구현 검증 리포트 |
| [docs/etri_video_speed_optimization.md](./docs/etri_video_speed_optimization.md) | 영상 파이프라인 속도 병목·가속화 옵션 |
| [docs/remote_hq_validation.md](./docs/remote_hq_validation.md) | 원격 GPU 고품질 검증 |
| [docs/etri_video_rate_benchmark.md](./docs/etri_video_rate_benchmark.md) | 의미 payload vs H.264/H.265/AV1 크기·품질 비교 |
| [docs/etri_owlv2_vqa_readiness.md](./docs/etri_owlv2_vqa_readiness.md) | OWLv2/VQA presence calibration 검증 결과 |
| [docs/lgvsc_1b_worker_readiness.md](./docs/lgvsc_1b_worker_readiness.md) | LGVSC 1B 외부 생성 worker 실제 GPU 검증 |
| [docs/lgvsc_1c_reproduction_readiness.md](./docs/lgvsc_1c_reproduction_readiness.md) | LGVSC 1C 재현 baseline 준비 |
| [docs/lgvsc_psss_skem_readiness.md](./docs/lgvsc_psss_skem_readiness.md) | PSSS/SKEM keyframe selector 검증 |

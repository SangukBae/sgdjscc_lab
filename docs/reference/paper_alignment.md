---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes:
---

> [← 문서 색인](../README.md)

# 논문 정합 정리

- 역할
  - 논문 정합성 단일 기준
  - 원본과 동일한 부분 구분
  - 근사·확장·미구현 구분

- 대상 논문
  - *Semantics-Guided Diffusion for Deep Joint Source-Channel Coding in Wireless Image Transmission*

## 큰 그림

- **추론 경로**는 원본 `SGDJSCC/` forward를 재사용하는 `paper-faithful` 경로다.
- 비동일 범위
  - 학습 scaffold
  - 채널 확장
  - 평가 체계
  - 분류: paper-like·scaffold·ETRI 확장 혼합
- `paper_mode: true`는 논문 재현 경로만 허용하도록 non-faithful 대체물을 차단한다.

## 원본 코드 vs `sgdjscc_lab`

| 항목 | 원본 `SGDJSCC/` | `sgdjscc_lab` |
|---|---|---|
| 진입점 | `inference_one.py` 중심 | `scripts/infer_images.py` |
| config | script 내부 결합 | YAML + CLI override |
| 채널/가이드/평가 | 한 파일에 섞임 | `channels/`, `guidance/`, `evaluators/`로 분리 |
| 추론 알고리즘 | monolithic | 동일 수치를 유지한 모듈식 래핑 |
| 원본 수정 | 직접 수정 필요 | `SGDJSCC/`는 읽기 전용, 확장은 `sgdjscc_lab/`에 구현 |

- 보존 블록
  - VAE encode·decode
  - scaling `15.45`
  - AWGN
  - blind SNR 예측
  - step matching
  - canny·edge 조건화
  - MDTv2·ControlNet 복원

## 충실도 분류

- **paper-faithful**: 공개 코드/논문과 수치 또는 구조가 사실상 일치
- **paper-like**: 의도와 구조는 같지만 일부 세부값·구현이 근사
- **scaffold**: 배선과 인터페이스는 있으나 학습된 수치나 완성형 동작은 미보장
- **ETRI 확장**: 논문에 없고 과제 목적을 위해 추가한 기능

## 핵심 정합 표

| 항목 | 상태 | 충실도 |
|---|---|---|
| 추론 forward-pass 전반 | 원본 재사용 | paper-faithful |
| step matching / blind SNR / VAE scaling | 구현 완료 | paper-faithful |
| MDTv2 masked+unmasked DM 구조 | stage runner에 반영 | paper-like |
| ControlNet freeze 정책 | 구조 반영 | paper-like |
| MuGE 기반 edge 경로 | 구조 반영 | paper-like → faithful structure |
| edge codec 전용 링크 | 학습 stage 포함 | paper-like |
| MMSE equalization / fast-fading 배선 | 연결 | paper-faithful(실수 gain 기준) / scaffold |
| complex phase / joint CSI | 일부 필드·연산만 | 부분/미구현 |
| FID / SRS / temporal / packet 기반 평가 | 논문 밖 확장 포함 | ETRI 확장 |

## `paper_mode`

- `paper_mode: true`는 논문 재현 경로를 강제한다.

- auto-caption, `filename` caption source 차단
- Canny stand-in 차단, MuGE sidecar 요구
- `shared_vae` edge transport 차단, `edge_jscc` 경로 요구
- Stage 3 `edge_jscc`는 학습된 edge codec checkpoint를 요구
- Stage 1 JSCC는 MSE-only가 아니라 MSE + patch-GAN 구조를 요구
- zero-vector CFG null 차단, learned null token 요구
- 논문 미공개값은 `paper_assumed_hparams`에 명시하고 실제 `train.*` 값과 일치해야 함
- 확장 기능(Phase 4/5, packet, regeneration 등) 비활성 요구
- eval metric set을 논문 보고 set에 맞춤

- 즉, 확장 기능을 지우는 것이 아니라 **논문 실험과 섞이지 않게 guardrail을 거는 것**이다.

## 하이퍼파라미터 출처

- Ground truth 우선순위
  1. 공개 `SGDJSCC/` 코드
  2. 논문 표
  3. 미공개 값에 대한 명시적 assumption

| 항목 | repo 선택 | 근거 |
|---|---|---|
| `diffusion_step=50`, `guidance_scale=4.0`, `controlnet_scale=0.3` | 유지 | 공개 코드 confirmed |
| backbone hidden size 512 / timestep embedding 256 | 유지 | 공개 코드 confirmed |
| JSCC stage 기본 SNR 10 dB | 유지 | 논문 + 공개 코드 관례 |
| `cfg_dropout_prob=0.1`, `lr=1e-4`, `weight_decay=1e-5` | `paper_assumed_hparams`에 분리 | 논문 미공개 |
| Stage-1 `gan.weight=0.5`, hinge PatchGAN 설정 | `paper_assumed_hparams`에 분리 | 논문 미공개 |
| edge codec ViT / multi-SNR range 0~20 dB | recent faithful approximation | 공개 exact WITT 재사용 불가 |

## 학습 경로의 주요 비등가

- Stage 1
  - 기본: MSE + patch-GAN
  - 가정값: GAN weight·discriminator 세부값
  - MSE-only: ablation
- Stage 3
  - baseline: `edge_jscc`
  - ablation: `shared_vae`
  - `paper_mode`: 학습된 `edge_codec` checkpoint 필수
- `end_to_end_ft`는 논문 baseline이 아니라 추가 실험이다.
- 논문의 대규모(~14M pair, 250k step) 데이터/스케줄은 repo에 번들되지 않았다.
- complex transport와 joint CSI는 layer 수준 scaffold이지 e2e faithful 재학습 경로는 아니다.

## 검증 상태

- 단위/통합 테스트 기준으로 paper-mode 경로와 주요 배선은 검증됨
- real-model smoke 학습과 일부 multi-GPU 경로는 별도 검증 문서가 있음
- 실제 FID 수치, large-scale DM 재현, water-filling 실수치 등은 체크포인트·데이터·GPU에 의존

## 관련 문서

- [../protocols/training.md](../protocols/training.md) — 학습 CLI + real-model smoke 검증
- [framework_file_roles.md](./framework_file_roles.md)
- [../architecture/tx_rx_contract.md](../architecture/tx_rx_contract.md) — 채널 조건화 설계(구 phase5)

---
status: active
updated: 2026-08-27
owner: ETRI SGD-JSCC 연구팀
source_commit: 607f727
supersedes: docs/etri_strategy.md, docs/phase4.md, docs/phase5.md
---

> [← 문서 색인](../README.md)

# 현재 구현 상태

- 문서 범위
  - 완료·PoC·스캐폴드 상태
  - 연구 목표 기준 현황
- 연결 문서
  - 설계: [architecture/](../architecture/)
  - 향후 작업: [roadmap.md](./roadmap.md)
  - 한계·기술 부채: [open_issues.md](./open_issues.md)
  - 실험 근거: `docs/experiments/`
  - 과거 구현 순서: [etri_implementation_log.md](../archive/etri_implementation_log.md)

## 핵심 연구 문제별 대응 현황

- [architecture/system.md](../architecture/system.md)의 세 핵심 연구 문제에 대한 현재 대응.

| 연구 문제 | 현재 대응 |
|---|---|
| 1. 시간축·영상 신뢰성 | keyframe pipeline, scene change, temporal evaluator, semantic delta + motion 이중 게이트, `PTC`/`SFR`/`SDI`, LGVSC 참고 3-way 생성 분기 — 완료(아래 "영상 확장" 참고) |
| 2. 할루시네이션 | semantic packet verifier, 오류 유형별 regeneration controller, OWLv2/VQA 보강 — 판정·로그까지 완료, 실제 sampler 개입은 미구현(아래 "할루시네이션 완화" 참고) |
| 3. 평가 체계 신뢰도 | loop-internal/held-out 지표 분리, `PTC`/`SFR`/`SDI`, Presence Calibration — 구조·실측 완료, GT/VLM 기반 Temporal SRS Calibration은 스캐폴드만(아래 "평가 체계" 참고) |

## 기능별 구현 상태

### 이미지 추론·평가 코어

- 상태: 완료
- 구성
  - 원본 SGD-JSCC forward pass 수치 보존
  - 모듈 분리: `channels`, `guidance`, `models`, `pipelines`
  - 지표: PSNR·SSIM·LPIPS·CLIP·SRS·FID
  - SNR sweep CSV
  - regeneration loop
- 호환성
  - `use_phase4=false`, `use_phase5=false`: 원본과 byte 단위 동일

### 시맨틱 패킷 평가 (`use_packet_eval`)

- 상태: 완료
- 패킷 구성
  - 캡션
  - 객체·관계
  - 가이드 요약
- 평가
  - 원본·복원 패킷 비교
  - 누락·추가 객체와 관계·속성 오류 집계
  - `srs_base`, `srs_packet`
- 제어
  - SNR 적응형 가이드: `use_adaptive_guidance`
  - 실패 유형별 재생성: `use_packet_regeneration`
- 게이트
  - `use_phase4` + 개별 플래그
  - 기본값: off
- 실행: [평가 프로토콜](../protocols/evaluation.md)

### 채널 조건화 (`use_channel_conditioning`)

- 상태
  - adapter 레벨 구현 완료
  - 실수치 검증 일부 완료
- 구성
  - Rayleigh·fast-fading·packet-drop
  - `MeasurementBundle`
  - 채널 조건 인코더
  - reliability 기반 guidance·step 조절
- 한계
  - frozen denoiser가 조건 token을 직접 사용하지 않음
  - water-filling은 배선·CPU stub만 검증
  - 실제 수치는 MDTv2 checkpoint 의존
- 설계: [Tx/Rx 계약 §4](../architecture/tx_rx_contract.md)

### 저지연 샘플링 (`acceleration.*`)

- 상태: 구현 완료
- 기능
  - Step-budget·DDIM
  - sampler early-exit: `heuristic`, `srs`, `srs_v2`
  - 지연 profiler
  - CLI: `benchmark_latency.py`, `benchmark_sampling.py`
- 한계
  - 학습된 consistency/distilled student는 placeholder

### 강화 검증기 (VQA/SRS-v2, `use_srs_v2`)

- 상태: 구현·연결 완료
- VQA backend
  - `mock`
  - `blip2`
  - `llava`
  - `mplug`
- 결합 지표: SRS-v2(base + packet + temporal + VQA)
- 탐색 기준: `srs`, `srs_v2`

### 영상 확장

- **핵심 파이프라인 완료, LGVSC 재현선 준비 완료(실행은 사용자), 학습형 개선선은 미착수.**

| 구성 요소 | 상태 |
|---|---|
| mp4 IO, 복원 frame/mp4 저장 | 완료 (`utils/video_io.py`) |
| 시맨틱 델타 + motion 이중 게이트 | 완료, 기본 off. 실데이터 threshold 튜닝은 미완([open_issues.md](./open_issues.md)) |
| GOP/segment 추상화 | 완료 (`video/segment.py::SegmentRecord`) |
| `PTC`/`SFR`/`SDI` 시간축 지표 | 완료 — CLIP/packet 기반으로 시작, OWLv2/VQA 보강 후 10개 영상 held-out 재측정까지 완료 ([experiments/2026-07-28_owlv2_vqa_calibration.md](../experiments/2026-07-28_owlv2_vqa_calibration.md)) |
| 3-way 분기(`reuse`/`recompute`/`generate`) | 완료 — start-only + bidirectional(mock 보간) 구조 검증 완료 |
| Rx-legal segment 생성 계약 | 완료 (`SegmentGenerationRequest`/`SegmentGenerationResult`/`generate_segment()`) — 설계는 [architecture/tx_rx_contract.md](../architecture/tx_rx_contract.md) §5.2 |
| 외부 worker 실제 생성 모델(Wan/SVD) | **실제 GPU 검증 완료** — Wan start-only + bidirectional(체크포인트 자동 선택), SVD start-only. 상세: [experiments/2026-07_lgvsc_1b_worker_validation.md](../experiments/2026-07_lgvsc_1b_worker_validation.md) |
| LGVSC 재현선 4-모드(`SKIM+SFA`/`SKEM+DSA` 대응) | 재현 준비 완료(config+batch driver), keyframe 선택은 4모드 공통(SKIM/SKEM 구분 아직 없음). 상세: [experiments/2026-07_lgvsc_1c_reproduction.md](../experiments/2026-07_lgvsc_1c_reproduction.md) |
| PSSS/SKEM variable-length keyframe selector | 코드/CPU 스모크 완료, 실제 MLLM 가중치 실행은 사용자 몫. 상세: [experiments/2026-07_lgvsc_psss_skem.md](../experiments/2026-07_lgvsc_psss_skem.md) |
| 학습형 bidirectional adapter / selector / critic (ETRI 개선선) | **미착수** — [roadmap.md](./roadmap.md) §1 |

### 할루시네이션 완화

- **판정·로그 완료, 실제 sampler 개입 미구현.**

- Packet Verifier: 완료
  - 구현: `evaluators/packet_verifier.py`
  - controller: `controllers/verifier_controller.py`
  - action: accept·suppress extra·strengthen missing·strengthen structure·fallback
- Presence calibration: 완료
  - backend: `clip`, `owlv2`, `vqa`, `gt`, `mock`
  - ensemble calibrator·held-out 재측정
  - 검증: [10개 영상 OWLv2/VQA 실험](../experiments/2026-07-28_owlv2_vqa_calibration.md)
- 미구현
  - candidate action의 실제 diffusion sampler 주입
  - 현재 controller는 결정·로그만 수행
  - 상세: [open_issues.md](./open_issues.md)

### 평가 체계

- 지표 역할 분리: 완료
  - loop-internal: `srs_packet`, VQA
  - held-out: 재생성에 관여하지 않은 지표
  - 태그: `metric_role`
- Temporal SRS Calibration: scaffold
  - 구현: synthetic target 기반 least-squares fitting
  - 미구현: 실제 GT·VLM 연결
- DISTS/downstream task 지표 — **미구현**.

### 전송량 절감

- 해석 단위
  - semantic-unit 절감
  - channel-symbol 절감: PoC
  - 직렬화 packet byte 절감: 실측

| 단계 | 상태 |
|---|---|
| Semantic-unit 절감 (키프레임+델타 재사용) | 완료 |
| Channel-symbol/bit accounting PoC (`accounting/bit_accounting.py`) | 완료 — proxy 상수 기반, 실제 CBR/표준 bitstream 검증 아님 |
| 실제 binary packet 전송 (`transmission/`, 4/6/8/16/32-bit 양자화) | **정상화·3-GPU 실측 완료** — 10영상×11설정 110/110 pair, 실패·NaN/Inf 0건. `fixed_int6`은 보수적 후보, `fixed_int4`는 최대 절감 후보. digital 절대 품질 저하와 SKEM rate matching 불완전은 미해결 — [2026-08-26 결과](../experiments/2026-08-26_transmission_normalization.md) |
| Importance-aware / 채널 신호 연동 bit allocation | 미착수 — [roadmap.md](./roadmap.md) §3 |

### 학습 CLI

- 상태: 완료
- stage
  - 논문 경로: `jscc`, `text_dm`, `controlnet`
  - 보조 경로: `edge_codec`, `csi_estimation`
  - 확장 실험: `end_to_end_ft`
- 기능
  - DDP
  - step·epoch 실행
  - auto-resume
  - 메모리 toggle
- 상세: [학습 프로토콜](../protocols/training.md)

## 관련 문서
- [roadmap.md](./roadmap.md) — 향후 연구개발 계획
- [open_issues.md](./open_issues.md) — 알려진 한계·기술 부채
- [architecture/](../architecture/) — 장기 시스템 설계
- `docs/experiments/` — 완료된 실험과 결과
- [etri_implementation_log.md](../archive/etri_implementation_log.md) — 과거 구현 요약

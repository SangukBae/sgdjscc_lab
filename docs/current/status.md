---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes: docs/etri_strategy.md, docs/phase4.md, docs/phase5.md
---

> [← 문서 색인](../README.md)

# 현재 구현 상태

- 이 문서는 **지금 기준으로 무엇이 완료/PoC/스캐폴드인지**만 다룬다. 설계 자체는
  [architecture/](../architecture/), 앞으로 할 일은 [roadmap.md](./roadmap.md), 알려진
  한계·기술 부채는 [open_issues.md](./open_issues.md), 실험 근거는
  `docs/experiments/`를 따른다. 과거 Phase 1~5 단위·1차~6차 구현 순서의 상세
  로그는 `docs/archive/`에 있다 — 이 문서는 그 순서를 반복하지 않고 **연구 목표
  기준**으로 정리한다.

## 핵심 연구 문제별 대응 현황

- [architecture/system.md](../architecture/system.md)의 세 핵심 연구 문제에 대한 현재 대응.

| 연구 문제 | 현재 대응 |
|---|---|
| 1. 시간축·영상 신뢰성 | keyframe pipeline, scene change, temporal evaluator, semantic delta + motion 이중 게이트, `PTC`/`SFR`/`SDI`, LGVSC 참고 3-way 생성 분기 — 완료(아래 "영상 확장" 참고) |
| 2. 할루시네이션 | semantic packet verifier, 오류 유형별 regeneration controller, OWLv2/VQA 보강 — 판정·로그까지 완료, 실제 sampler 개입은 미구현(아래 "할루시네이션 완화" 참고) |
| 3. 평가 체계 신뢰도 | loop-internal/held-out 지표 분리, `PTC`/`SFR`/`SDI`, Presence Calibration — 구조·실측 완료, GT/VLM 기반 Temporal SRS Calibration은 스캐폴드만(아래 "평가 체계" 참고) |

## 기능별 구현 상태

### 이미지 추론·평가 코어

- **완료.** 원본 SGD-JSCC forward pass 수치 보존 + 모듈 분리(`channels/guidance/models/pipelines`) +
  평가기 세트(PSNR/SSIM/LPIPS/CLIP/SRS/FID) + SNR-sweep CSV + regeneration loop.
  `use_phase4`/`use_phase5`가 모두 false면 원본과 byte 단위로 동일하게 동작한다.

### 시맨틱 패킷 평가 (`use_packet_eval`)

- **완료.** 시맨틱 패킷(캡션+객체/관계+가이드 요약) 구성, 원본 vs 복원 패킷 비교로
  누락/추가 객체·관계/속성 오류를 개수로 집계, `srs_base`/`srs_packet` 확장, SNR
  적응형 가이드(`use_adaptive_guidance`), 실패 양상별 regeneration(`use_packet_regeneration`).
  게이트: `use_phase4` + 개별 플래그(기본 off). 실행 절차는 [protocols/evaluation.md](../protocols/evaluation.md).

### 채널 조건화 (`use_channel_conditioning`)

- **구현 완료(adapter 레벨), 실 수치 검증은 부분적.** Rayleigh/fast-fading/packet-drop
  채널 + `MeasurementBundle` + 채널 조건 인코더 + reliability-스케일 guidance/steps.
  frozen SGD-JSCC denoiser가 조건 토큰을 직접 소비하지는 않는다(근사 지점 — 설계는
  [architecture/tx_rx_contract.md](../architecture/tx_rx_contract.md) §4). Fast-fading
  water-filling(논문 Algorithm 4)은 배선·CPU stub 검증 완료, 실제 수치는 MDTv2
  체크포인트 의존.

### 저지연 샘플링 (`acceleration.*`)

- **구현 완료.** Step-budget/DDIM, 샘플러 내부 early-exit(`heuristic`/`srs`/`srs_v2`
  기준 조기 종료), 지연 프로파일러, 벤치마크 CLI(`benchmark_latency.py`/`benchmark_sampling.py`).
  학습된 consistency/distilled student는 placeholder(인터페이스만 완성).

### 강화 검증기 (VQA/SRS-v2, `use_srs_v2`)

- **구현·연결 완료.** VQA 할루시네이션 검출(`mock`/`blip2`/`llava`/`mplug` backend),
  SRS-v2(base+packet+temporal+VQA), regeneration search(여러 전략 중 `srs`/`srs_v2`
  기준 최적 선택).

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

- Packet Verifier(`evaluators/packet_verifier.py`) + 오류 유형별 controller
  (`controllers/verifier_controller.py`: accept/suppress_extra/strengthen_missing/
  strengthen_structure_guidance/fallback_recompute/keyframe_fallback) — 완료.
- Presence backend 인터페이스(`clip`/`owlv2`/`vqa`/`gt`/`mock`) + ensemble
  calibrator + held-out 재측정 — 완료, 실제 OWLv2/VQA weight로 10개 영상 재검증까지
  완료([experiments/2026-07-28_owlv2_vqa_calibration.md](../experiments/2026-07-28_owlv2_vqa_calibration.md)).
- **미구현**: candidate action(negative prompt 강화 등)을 실제 diffusion 샘플러에
  주입하는 배선 — controller는 여전히 결정·로그만 남긴다([open_issues.md](./open_issues.md)).

### 평가 체계

- loop-internal(`srs_packet`/VQA)과 held-out(재생성에 관여하지 않은 지표) 분리
  원칙과 `metric_role` 태깅 — 완료.
- Temporal SRS Calibration(GT/VLM 기준 가중치 보정) — **스캐폴드만**, synthetic
  target 기준 least-squares fitting 구조는 있으나 실제 GT/VLM 연결은 미완.
- DISTS/downstream task 지표 — **미구현**.

### 전송량 절감

- `semantic-unit 절감`, `channel-symbol 절감(PoC)`, `직렬화 packet byte 절감(실측)`을
  구분해서 읽는다.

| 단계 | 상태 |
|---|---|
| Semantic-unit 절감 (키프레임+델타 재사용) | 완료 |
| Channel-symbol/bit accounting PoC (`accounting/bit_accounting.py`) | 완료 — proxy 상수 기반, 실제 CBR/표준 bitstream 검증 아님 |
| 실제 binary packet 전송 (`transmission/`, 4/6/8/16-bit 양자화) | **구현 완료, operating point 검증 중** — 10개 영상에서 `int4`가 analog AWGN 임시 기준의 픽셀 품질 gate를 통과했다. reliable-digital 기준과 SRS·할루시네이션 평가는 남아 있다. 상세: [experiments/2026-08-18_transmission_reduction.md](../experiments/2026-08-18_transmission_reduction.md) |
| Importance-aware / 채널 신호 연동 bit allocation | 미착수 — [roadmap.md](./roadmap.md) §3 |

### 학습 CLI

- **완료.** 논문 3-stage(`jscc`/`text_dm`/`controlnet`) + 보조 stage(`edge_codec`/
  `csi_estimation`) + 확장 실험(`end_to_end_ft`). DDP 지원, step/epoch 겸용,
  auto-resume, 메모리 토글. 상세: [protocols/training.md](../protocols/training.md).

## 관련 문서
- [roadmap.md](./roadmap.md) — 향후 연구개발 계획
- [open_issues.md](./open_issues.md) — 알려진 한계·기술 부채
- [architecture/](../architecture/) — 장기 시스템 설계
- `docs/experiments/` — 완료된 실험과 결과
- `docs/archive/` — Phase 1~5, 1차~6차 구현 순서의 상세 이력(과거 스냅샷)

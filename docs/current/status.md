---
status: active
updated: 2026-09-04
owner: ETRI SGD-JSCC 연구팀
source_commit: aae9e26
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

## 활성 논문 연구선과 G0

- 기준 계획: [negative_semantic_paper_plan.md](./negative_semantic_paper_plan.md)
- 2026-09-03 v1.2 완료: OVIS validation frame·official with-GT와 기존
  YouTube-VOS/DAVIS archive SHA-256·영상별 tree hash·사용 승인 기록 동결
- 데이터 manifest: OVIS Pilot 40 / Train 3,471 / Development 10 / Validation 507 /
  source-disjoint Held-out 30, 총 4,058개 영상
- Pilot: 서로 다른 OVIS 영상 40개에서 official-GT-derived ENTER/EXIT/OCCLUDE/
  REAPPEAR 각 10개; annotated timestep 단위
- 자동 감사: 13개 조건 중 13개 통과
- 현재 gate: **G1 v1.1 정식 실행 완료, `runner PASSED` + `g1_summary.json.gate_status: PASSED`(declared-seed 정의)**.
  40영상 × `{few10,full50}` × 3 declared seed, child run 6개 × 40영상 =
  240/240 완료, 실패 0, held-out 미접근. 독립 감사는
  `scripts/audit_negative_semantics_g1.py --require-pass`(exit 0)가 수행한다.
- **v1.2 amendment(2026-09-06)로 한계 3가지를 정량화했다**: (1) 이
  reconstruction 경로는 seed를 소비하지 않아 `effective_seed_count=1`(선언
  3-seed 대비)이며, 같은 gate threshold를 effective-seed 지표에 재적용하면
  `affected_seed_count`/`not_concentrated_in_one_seed`가 FAIL로 뒤집혀
  few10 기준 gate가 **`NOT_PASSED`**가 된다. (2) source-paired
  additional-object로 다시 계산하면 h_add가 약 47~51% 낮아지고(few10
  0.019793→0.010493, full50 0.018536→0.009081) full50은 `h_add_min=0.01`
  미달이다. (3) ghost survival uncensored 표본은 effective-seed 기준
  policy당 4개뿐이다. GPU 재실행 없이 기존 `detection_rows.jsonl`만으로
  재계산했다(`scripts/derive_negative_semantics_g1_v1_2.py`).
- G1은 GPU 학습이 아니라 frozen checkpoint 추론·자동 평가다. smoke output은 논문 근거가
  아니며 정식 Pilot 결과만 G1 gate에 사용한다.
- 2026-09-04 ETRI 신규 개발·시연의 기본 양자화 운용점은 `fixed_int6`로 변경했다.
  G1 v1.1은 비교 가능성을 위해 `fixed_int4` stress configuration을 유지한다.
  `fixed_int4 vs fixed_int6` paired bridge(동일 OVIS Pilot·both-omit·
  few10/full50·effective seed 1개)는 코드·설정·테스트를 완료해 **실행 준비
  상태**이며, GPU 실행은 아직 하지 않았다 — 정확한 명령은
  [bridge 준비 기록](../experiments/2026-09-06_negative_semantics_int6_bridge_preparation.md) 참고.
- 첫 formal v1.0 시도는 비정상 overlapping-patch 증가에 따른 RTX 4080 OOM으로 전체
  무효화했다. v1.1은 resize 후 128-grid padding, 평가 전 padding crop을 동결했고 실제
  실패 영상 smoke를 통과한 뒤 정식 240/240을 완료했다. v1.0에서는 검증된 source-only
  caption만 명시적 provenance 검사 후 재사용했다.
- 구현 경계: negative packet codec, RSM, joint allocator, sampler의 object-level negative
  control은 아직 미구현이다. G1은 additional-object 현상 확인이고 Oracle ABSENT 제어는
  통과 후 G2에서 수행한다. G2 진입 전 위 v1.2 결과(effective-seed NOT_PASSED,
  full50 source-paired h_add 미달)를 논문 claim 경계에 반영해야 한다.
- 근거: [G0 v1.2 official-GT amendment](../experiments/2026-09-03_negative_semantics_g0_official_gt_amendment_v1_2.md),
  [G1 v1.1 결과·감사](../experiments/2026-09-06_negative_semantics_g1_v1_1_pilot_results.md),
  [G1 v1.2 amendment](../experiments/2026-09-06_negative_semantics_g1_v1_2_amendment.md)
- 실행: [G1 준비·동결 절차](../experiments/2026-09-03_negative_semantics_g1_preparation.md)
- 보존: [results/negative_semantics_g1_pilot_rtx4080_v1_1](../../results/negative_semantics_g1_pilot_rtx4080_v1_1/),
  [results/negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2](../../results/negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2/)

통합 개발 요약기는 이후 실행부터 `run_integrity_passed`, point-mean candidate,
CI-screening candidate를 별도 필드로 기록하고 reference baseline 자체를 후보 선택에서
제외한다. Development role에서는 `validation_passed=null`이며, manifest가 Validation을
명시한 경우에만 모든 paired CI 상한 통과 여부를 기록한다. 2026-08-29 보존 결과는
historical artifact이므로 재작성하지 않았다.

## 핵심 연구 문제별 대응 현황

- [architecture/system.md](../architecture/system.md)의 세 핵심 연구 문제에 대한 현재 대응.

| 연구 문제 | 현재 대응 |
|---|---|
| 1. 시간축·영상 신뢰성 | keyframe pipeline, scene change, temporal evaluator, semantic delta + motion 이중 게이트, `PTC`/`SFR`/`SDI`, LGVSC 참고 3-way 생성 분기 — **기본 파이프라인 완료**. real MLLM PSSS·10영상×4모드 재현·학습형 개선선은 미완(아래 "영상 확장" 참고) |
| 2. 할루시네이션 | semantic packet verifier, 오류 유형별 regeneration controller, OWLv2/VQA 보강 — 판정·로그까지 완료, 실제 sampler 개입은 미구현(아래 "할루시네이션 완화" 참고) |
| 3. 평가 체계 신뢰도 | loop-internal/held-out 지표 분리, `PTC`/`SFR`/`SDI`, Presence Calibration — 구조·기존 실측 완료. GT/VLM 기반 Temporal SRS Calibration·DISTS/downstream·최종 paired held-out 검증은 미완(아래 "평가 체계" 참고) |

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
| 실제 binary packet 전송 (`transmission/`, 4/6/8/16/32-bit 양자화) | **전송 안정성 검증 완료·ETRI 기본 운용점 `fixed_int6`** — 10영상×6설정 60/60 pair, 실패·NaN/Inf 0건. `fixed_float32` 대비 int16/int8/int6/int4가 모두 품질 허용 기준을 통과했다. 당시 rate-first 규칙은 최소 bit-depth `fixed_int4`를 선택했지만, 2026-09-04 ETRI 운용 목적에서는 679.936 bytes/frame의 추가 전송으로 baseline에 더 가까운 PSNR·SSIM을 얻는 `fixed_int6`를 신규 기본값으로 결정했다. AWGN은 참고행이며 digital Pareto baseline에서 제외 — [10dB 재평가](../experiments/2026-08-28_quantization_reevaluation_10db.md), [int6 결정](../experiments/2026-09-04_int6_etri_operating_point_decision.md), [보존 결과](../../results/quantization_reevaluation_10db_20260828/README.md) |
| float32 digital 품질 저하 진단 harness (`diagnostics/`, `scripts/diagnose_float32_digital_quality.py`) | **10dB step 정상화·full 검증·결과 고정 완료** — 3 core condition×100프레임에서 digital wire가 AWGN 대비 PSNR `+0.721dB`, SSIM `+0.00552`, LPIPS `-0.00223`; `-1dB` 이하 저하 0/300. in-process/wire 최대 PSNR 차이 `0.000752dB`, wire round-trip 300/300 bit-exact, 실패·NaN/Inf·conflict 0건. instrumented 20/20과 metric-only stage 6 300/300 모두 `no_issue_detected`; 핵심 원본 50개·보정 report·checksum을 registry에 고정 — [full 실측](../experiments/2026-08-28_float32_digital_step_normalization_full.md), [보존 결과](../../results/float32_digital_normalization_full_20260827/README.md) |
| 10dB 양자화 재평가 실행 환경 | **3-GPU full 실행·원격 회수·결과 registry 고정 완료** — fixed selector에서 AWGN 참고 + float32/int16/int8/int6/int4 비교, seed 2025. 원격·로컬 13,144개 파일/1,210,877,488 bytes와 전 파일 SHA-256 일치. worker `cuda:0/1/2` provenance 및 10dB plan/signature/resolved-config 계약 확인 |
| fixed–SKEM exact matched-rate 재평가 | **3-GPU full 실행·결과 registry 고정 완료** — 100/100 pair, 50/50 rate row, 실패·NaN/Inf 0; actual transmitting count exact, raw byte 최대 차이 0.004953%, padding 후 effective byte exact. 단, proxy SKEM은 10/10 영상에서 fixed와 keyframe/transmitting index가 같아 품질 차이도 정확히 0이었다. SKEM 우위가 아니라 fixed schedule로 수렴한 null 결과다. 당시 int4 실험은 historical evidence로 유지하고 신규 ETRI bit-depth는 별도 결정에 따라 int6를 사용한다 — [실험](../experiments/2026-08-28_fixed_skem_matched_rate_10db.md), [int6 결정](../experiments/2026-09-04_int6_etri_operating_point_decision.md), [보존 결과](../../results/fixed_skem_matched_rate_10db_20260828/README.md) |
| edge·uncertainty 전송 절감 ablation | **16-profile·3-GPU full 실행·결과 registry 고정 완료** — 160/160 pair, 실패·NaN/Inf 0. 시험한 후보 중 `combined_ds4`가 356,824.7 bytes/video로 `fixed_int4` baseline 대비 85.11% 절감했고 pixel quality gate를 통과했다. 단, uncertainty-only 5종은 모든 영상에서 지표가 baseline과 정확히 같았으며 reliable-digital decoder가 uncertainty를 소비하지 않는 경로가 확인됐다. edge 영향도 수치적으로 매우 작아 혼합 omit/downsample과 semantic·시간축 검증 전까지 조건부 후보로만 유지 — [실험](../experiments/2026-08-28_edge_uncertainty_ablation_10db.md), [보존 결과](../../results/edge_uncertainty_ablation_10db_20260828/README.md) |
| 통합 semantic·hallucination·temporal 평가 | **3-GPU full 완료·결과 registry 고정, held-out은 데이터 준비 전까지 연기** — 4 guide × 3 decoder × 10영상의 120/120 pair, 총 12,000 frame, 실패·non-finite 0. CLIP·OWLv2·VQA가 각각 47,744건 기여했다. 평균 gate의 개발셋 잠정 후보는 `few10 + both-omit`: 219,459.7 bytes/video(-90.843%), 39.454s/video(-63.486%), PSNR -0.2435dB, SSIM -0.00188, LPIPS +0.01558. hallucination/additional-object CI 상한 0.0525/0.0570이 margin 0.05를 넘어 최종 운영점은 확정하지 않는다. 새 독립 데이터가 준비될 때까지 guide 계약·verifier 폐루프·동적 controller를 먼저 진행 — [결과](../experiments/2026-08-29_integrated_semantic_validation_10db.md), [보존 결과](../../results/integrated_semantic_validation_10db_20260829/README.md), [연기 결정과 작업 순서](./roadmap.md#held-out-연기-결정-2026-08-29) |
| VAE-direct 후보 | **통합 개발평가에서 strict SSIM gate 실패** — both-omit에서 23.4885s/video로 full50보다 4.60배 빠르고 PSNR·LPIPS는 개선됐지만 SSIM 하락 0.01129가 margin 0.01을 넘었다. primary held-out 후보에서는 제외하고 탐색적 비교로 유지 — [통합 결과](../experiments/2026-08-29_integrated_semantic_validation_10db.md) |
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

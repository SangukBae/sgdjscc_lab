# sgdjscc_lab 문서 색인

- 프로젝트 관계
  - 원본 `SGDJSCC/`: 읽기 전용 논문 baseline
  - `sgdjscc_lab/`: 모듈화·평가·연구 확장
  - 기본 추론: 원본 forward-pass 연산 계약 보존이 목표. 구성요소 회귀는 검증됐지만
    원본 script·실제 checkpoint와의 종단간 byte-parity는 아직 별도 입증되지 않음
- 문서 원칙
  - 역할별 분리
  - 현재 계획·상태·실험 이력 분리

## 연구개발할 때 먼저 볼 문서

| 목적 | 기준 문서 |
|---|---|
| 다음 채팅에서 전체 상황 인계 | **[current/next_chat_handoff.md](./current/next_chat_handoff.md)** — 검증·잠정 결론·다음 작업 요약 |
| 다음 구현 작업과 우선순위 | **[current/roadmap.md](./current/roadmap.md)** — 메인 작업 문서 |
| SAVER-JSCC 모델 단일 기준 | **[current/saver_jscc_model_plan.md](./current/saver_jscc_model_plan.md)** — 구조·학습·SV0~SV5 gate와 claim 경계 |
| negative-semantics 선행 검증 기준 | [current/negative_semantic_paper_plan.md](./current/negative_semantic_paper_plan.md) — SAVER SV0/SV1의 G0~G11 데이터·Oracle·packet/RSM 근거 |
| 실제 완료·PoC·미구현 판단 | [current/status.md](./current/status.md) |
| 알려진 제약 확인 | [current/open_issues.md](./current/open_issues.md) |
| 지표·평가 설계 | [architecture/metrics.md](./architecture/metrics.md) |
| 실험 실행·비교 규약 | [protocols/evaluation.md](./protocols/evaluation.md) |

- 연구개발 흐름
  1. [roadmap.md](./current/roadmap.md)에서 작업 선택
  2. 완료 후 [status.md](./current/status.md) 갱신
  3. 결과를 `experiments/YYYY-MM-DD_<name>.md`에 고정
  4. `archive/`는 과거 근거 확인에만 사용

| 폴더 | 역할 |
|---|---|
| [`current/`](./current/) | 지금 기준 상태 — 구현 현황, 향후 계획, 알려진 한계 |
| [`architecture/`](./architecture/) | 장기 시스템 설계 — 바뀌지 않는 구조·지표 정의 |
| [`protocols/`](./protocols/) | 평가·재현·학습 실행 절차 |
| [`experiments/`](./experiments/) | 완료된 실험과 그 결과(날짜 기준, 스냅샷) |
| [`reference/`](./reference/) | 논문 정합성, 파일별 역할 지도 등 참조 문서 |
| [`reports/`](./reports/) | 대외 발표·보고 자료 |
| [`archive/`](./archive/) | 더 이상 활성이 아닌 과거 문서 |

## 1. 현재 상태 (`current/`)

| 문서 | 내용 |
|---|---|
| [current/next_chat_handoff.md](./current/next_chat_handoff.md) | 다음 채팅용 전체 검증·잠정 모델·후속 작업 인계 요약 |
| [current/status.md](./current/status.md) | 기능별 현재 구현 상태 — 완료/PoC/스캐폴드 구분 |
| [current/roadmap.md](./current/roadmap.md) | 연구 목표 기준 향후 계획, 일정, ETRI 협의 필요사항 |
| [current/open_issues.md](./current/open_issues.md) | 알려진 한계·기술 부채 |
| [current/saver_jscc_model_plan.md](./current/saver_jscc_model_plan.md) | 활성 제안 모델의 단일 기준; 핵심 prototype `IMPLEMENTED_UNTRAINED`, G2 Pilot 실행 중, formal evidence 없음 |
| [current/negative_semantic_paper_plan.md](./current/negative_semantic_paper_plan.md) | SAVER 선행 연구선; G0 `PASSED`, G1 effective-seed `NOT_PASSED`, fixed_int4 primary |

## 2. 장기 시스템 설계 (`architecture/`)

| 문서 | 내용 |
|---|---|
| [architecture/system.md](./architecture/system.md) | baseline과 SAVER target을 분리한 전체 파이프라인·모듈 구조 |
| [architecture/tx_rx_contract.md](./architecture/tx_rx_contract.md) | 기존 Tx/Rx와 SAVER signed action·versioned memory·SM-DiT 계약 |
| [architecture/metrics.md](./architecture/metrics.md) | SRS·시간축·SAVER ghost/revocation 지표 정의, loop-internal/held-out 분리 원칙 |

## 3. 평가·재현 절차 (`protocols/`)

| 문서 | 내용 |
|---|---|
| [protocols/evaluation.md](./protocols/evaluation.md) | 이미지/영상/SAVER gate 평가 절차, 실험 설정 규약, presence 보정 재측정 |
| [protocols/video_rate_benchmark.md](./protocols/video_rate_benchmark.md) | 의미통신 payload vs H.264/H.265/AV1 코덱 비교 방법 |
| [protocols/datasets.md](./protocols/datasets.md) | 기존/SAVER 데이터 역할·stage 매핑·held-out 개봉 규칙 |
| [protocols/reproducibility.md](./protocols/reproducibility.md) | baseline/SAVER checkpoint 분리, `paper_mode`와 fingerprint 규칙 |
| [protocols/results_registry.md](./protocols/results_registry.md) | 추적 `results/` 구조, run manifest 스키마·생성 절차 |
| [protocols/training.md](./protocols/training.md) | 기존 stage-aware 학습과 SAVER 단계·gradient·checkpoint 계약 |
| [protocols/transmission_normalization.md](./protocols/transmission_normalization.md) | 전송 정상화와 SAVER source/wireless rate 단위 분리 |
| [protocols/float32_digital_diagnostics.md](./protocols/float32_digital_diagnostics.md) | float32 digital 복원 품질 진단 harness(경로 비교·stage 계측·ablation) — 10dB full 300프레임 검증 완료 |

- 데이터 문서
  - 전체 데이터: [data/README.md](../data/README.md)
  - ETRI 10영상: [etri_video_eval/README.md](../data/etri_video_eval/README.md)

## 4. 완료된 실험 (`experiments/`)

| 문서 | 내용 |
|---|---|
| [experiments/2026-07-17_stage1_video_pipeline.md](./experiments/2026-07-17_stage1_video_pipeline.md) | 영상 파이프라인 1차 구현(mp4 IO, 시간축 지표, motion 게이트) 검증 리포트 |
| [experiments/2026-07-24_video_speed_optimization.md](./experiments/2026-07-24_video_speed_optimization.md) | 영상 실모델 검증 속도 병목 분석·가속화·원격 GPU 실측 |
| [experiments/2026-07-28_owlv2_vqa_calibration.md](./experiments/2026-07-28_owlv2_vqa_calibration.md) | OWLv2/VQA presence calibration 10-영상 실측 결과 |
| [experiments/2026-07_lgvsc_1b_worker_validation.md](./experiments/2026-07_lgvsc_1b_worker_validation.md) | LGVSC 1B 외부 생성 worker(Wan/SVD) 실제 GPU 검증 |
| [experiments/2026-07_lgvsc_1c_reproduction.md](./experiments/2026-07_lgvsc_1c_reproduction.md) | LGVSC 1C 재현 baseline 4모드 준비·실행 절차 |
| [experiments/2026-07_lgvsc_psss_skem.md](./experiments/2026-07_lgvsc_psss_skem.md) | PSSS/SKEM variable-length keyframe selector 검증 |
| [experiments/2026-08-16_remote_hq_validation.md](./experiments/2026-08-16_remote_hq_validation.md) | 원격 3×RTX 4090 고품질 최종 검증 |
| [experiments/2026-08-18_transmission_reduction.md](./experiments/2026-08-18_transmission_reduction.md) | 직렬화 packet byte Pareto sweep(4/6/8/16-bit 양자화 vs 화질) |
| [experiments/2026-08-26_transmission_normalization.md](./experiments/2026-08-26_transmission_normalization.md) | 전송 정상화·3-GPU 실측 결과와 후속 과제 |
| [experiments/2026-08-28_float32_digital_step_normalization.md](./experiments/2026-08-28_float32_digital_step_normalization.md) | 60dB→10dB decoder-step 정상화 3-GPU short 실측 |
| [experiments/2026-08-28_float32_digital_step_normalization_full.md](./experiments/2026-08-28_float32_digital_step_normalization_full.md) | 10dB decoder-step 정상화 3-GPU full 300프레임 최종 실측 |
| [experiments/2026-08-28_quantization_reevaluation_10db.md](./experiments/2026-08-28_quantization_reevaluation_10db.md) | 10dB fixed-selector float32/int16/int8/int6/int4 양자화 재평가와 당시 rate-first 4-bit 운영점 확정 |
| [experiments/2026-08-28_fixed_skem_matched_rate_10db.md](./experiments/2026-08-28_fixed_skem_matched_rate_10db.md) | fixed–SKEM actual-transmission/effective-byte exact matching과 proxy SKEM의 fixed schedule 수렴 null 결과 |
| [experiments/2026-08-28_edge_uncertainty_ablation_10db.md](./experiments/2026-08-28_edge_uncertainty_ablation_10db.md) | fixed_int4 edge·uncertainty 16-profile 3-GPU full ablation; combined_ds4 조건부 최소-byte 후보와 digital uncertainty bypass 확인 |
| [experiments/2026-08-28_edge_uncertainty_ablation_preparation.md](./experiments/2026-08-28_edge_uncertainty_ablation_preparation.md) | 위 full ablation의 실행 전 profile·wire-accounting 계약 |
| [experiments/2026-08-28_integrated_semantic_validation_preparation.md](./experiments/2026-08-28_integrated_semantic_validation_preparation.md) | 4 guide × 3 decoder × 10영상 통합 semantic·hallucination·temporal 3-GPU 검증 프로토콜 |
| [experiments/2026-08-29_integrated_semantic_validation_10db.md](./experiments/2026-08-29_integrated_semantic_validation_10db.md) | 120-pair 통합 개발평가; both-omit guide, few10 잠정 후보, hallucination CI 경고 |
| [experiments/2026-09-02_negative_semantics_g0_protocol.md](./experiments/2026-09-02_negative_semantics_g0_protocol.md) | negative-semantics G0 v1.0 역사 기록; 7/10 통과 시점 |
| [experiments/2026-09-02_negative_semantics_g0_acquisition_amendment_v1_1.md](./experiments/2026-09-02_negative_semantics_g0_acquisition_amendment_v1_1.md) | 공식 데이터·split·hash·승인·held-out 봉인 완료; 12/13, 사람 검수 대기 |
| [experiments/2026-09-03_negative_semantics_g0_official_gt_amendment_v1_2.md](./experiments/2026-09-03_negative_semantics_g0_official_gt_amendment_v1_2.md) | OVIS official-GT Pilot 40개·자동 audit 13/13; G0 통과 |
| [experiments/2026-09-03_negative_semantics_g1_preparation.md](./experiments/2026-09-03_negative_semantics_g1_preparation.md) | G1 동결 matrix, 독립 OWLv2 calibration, 단일 RTX 4080 재개형 실행 절차 |
| [experiments/2026-09-03_negative_semantics_g1_memory_amendment_v1_1.md](./experiments/2026-09-03_negative_semantics_g1_memory_amendment_v1_1.md) | v1.0 OOM 원인, 128-grid padding/crop, 기존 run 무효화와 v1.1 재실행 경계 |
| [experiments/2026-09-04_int6_etri_operating_point_decision.md](./experiments/2026-09-04_int6_etri_operating_point_decision.md) | 당시 fixed_int6 임시 결정 기록; 2026-09-07 fixed_int4 결정으로 superseded된 역사적 artifact |
| [experiments/2026-09-06_negative_semantics_g1_v1_1_pilot_results.md](./experiments/2026-09-06_negative_semantics_g1_v1_1_pilot_results.md) | G1 v1.1 정식 240/240 Pilot 실행 감사(`audit_negative_semantics_g1.py`)와 runner PASSED vs 논문 주장 범위 구분 |
| [experiments/2026-09-06_negative_semantics_g1_v1_2_amendment.md](./experiments/2026-09-06_negative_semantics_g1_v1_2_amendment.md) | effective-seed 집계(seed 중복 발견, effective_seed_count=1)와 source-paired additional-object 분리; gate 재적용 시 NOT_PASSED로 뒤집힘 |
| [experiments/2026-09-06_negative_semantics_int6_bridge_preparation.md](./experiments/2026-09-06_negative_semantics_int6_bridge_preparation.md) | fixed_int4 vs fixed_int6 paired bridge 실행 준비(코드/설정/테스트 완료, GPU 실행은 미실시) — tmux 명령과 판독 절차 |
| [experiments/2026-09-07_negative_semantics_int6_bridge_results.md](./experiments/2026-09-07_negative_semantics_int6_bridge_results.md) | OVIS Pilot 40영상 formal bridge 결과; int6 품질 gate 통과, bytes +44.217%, H_add 유의차 없음 |
| [experiments/2026-09-07_negative_semantics_g2_implementation.md](./experiments/2026-09-07_negative_semantics_g2_implementation.md) | SV0/G2 Oracle ABSENT receiver path·four-arm runner·read-only audit 구현; CPU 회귀만 완료, GPU 결과 없음 |
| [experiments/2026-09-07_saver_jscc_structural_implementation.md](./experiments/2026-09-07_saver_jscc_structural_implementation.md) | SAVER 전체 구조 prototype·packet·학습 기반과 G3 harness 구현; 전체 1708 passed, 미학습·formal evidence 없음 |
| [experiments/2026-09-07_fixed_int4_primary_operating_point_decision.md](./experiments/2026-09-07_fixed_int4_primary_operating_point_decision.md) | bridge 결과와 통신 효율 목표를 반영해 fixed_int4를 primary development bit-depth로 재결정 |

## 5. 참조 문서 (`reference/`)

| 문서 | 내용 |
|---|---|
| [reference/paper_alignment.md](./reference/paper_alignment.md) | SGD-JSCC/LGVSC/SAVER 정합성, `paper_mode`, 충실도·claim 분류 |
| [reference/framework_file_roles.md](./reference/framework_file_roles.md) | 기존 파일 실행 흐름과 SAVER 목표 파일 역할 지도 |
| [reference/paper_writing_notes.md](./reference/paper_writing_notes.md) | 이전 reliability-layer 연구선 메모(superseded; 역사 참고용) |

## 6. 발표·보고 자료 (`reports/`)

- [reports/README.md](./reports/README.md)
  - 외부공유용 국·영문
  - 부록 국·영문
  - Q&A 및 artifact 관리

## 7. 과거 문서 (`archive/`)

- 용도
  - 과거 구현 순서 확인
  - 현재 판단 근거로 사용하지 않음
  - 상세 원문은 Git 이력에서 확인

| 문서 | 내용 |
|---|---|
| [archive/etri_implementation_log.md](./archive/etri_implementation_log.md) | 통합 구현 이력 요약 |

## 개발 원칙

1. **알고리즘 경로 보존** — `SGDJSCC/inference_one.py`의 forward 수치를 그대로 유지한다.
2. **관심사 분리**
   - 채널: `channels/`
   - 가이드: `guidance/`
   - 모델: `models/`
   - 오케스트레이션: `pipelines/`
   - 지표: `evaluators/`
3. **원본 읽기 전용** — 새 아이디어는 `SGDJSCC/`가 아니라 `sgdjscc_lab/`에 구현한다.
4. **SAVER opt-in** — SAVER 전용 config에서만 `use_saver_jscc=true`를 사용하며 기존
   production default·SGD-JSCC/LGVSC-inspired 경로를 변경하지 않는다.
   SV0/G2의 `--negative-condition-manifest`도 명시적 opt-in이며 Oracle text는 packet에
   직렬화하거나 rate로 계산하지 않는다.

- 추가 기준
  - 전체 디렉터리·Phase gate: [system.md](./architecture/system.md)

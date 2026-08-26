# sgdjscc_lab 문서 색인

`sgdjscc_lab`은 원본 `SGDJSCC/` 패키지를 **수정하지 않고** 확장하는 연구용 fork다.
원본은 논문 베이스라인이자 읽기 전용 참조로 두고, 이 패키지는 모듈화·평가·연구
확장을 위한 깨끗한 계층을 얹는다. 추론 forward-pass는 원본과 수치적으로 동일하다.

문서는 **역할별**로 나뉜다 — 한 문서가 여러 역할을 겸하지 않는다.

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
| [current/status.md](./current/status.md) | 기능별 현재 구현 상태 — 완료/PoC/스캐폴드 구분 |
| [current/roadmap.md](./current/roadmap.md) | 연구 목표 기준 향후 계획, 일정, ETRI 협의 필요사항 |
| [current/open_issues.md](./current/open_issues.md) | 알려진 한계·기술 부채 |

## 2. 장기 시스템 설계 (`architecture/`)

| 문서 | 내용 |
|---|---|
| [architecture/system.md](./architecture/system.md) | 과제 목표, 핵심 연구 문제, 전체 파이프라인, 모듈 구조 |
| [architecture/tx_rx_contract.md](./architecture/tx_rx_contract.md) | Tx/Rx 모듈 설계, 패킷 검증·채널 조건화 계약, LGVSC 참고 영상 확장 설계 |
| [architecture/metrics.md](./architecture/metrics.md) | SRS·시간축 지표(`PTC`/`SFR`/`SDI`) 공식 정의, loop-internal/held-out 분리 원칙 |

## 3. 평가·재현 절차 (`protocols/`)

| 문서 | 내용 |
|---|---|
| [protocols/evaluation.md](./protocols/evaluation.md) | 이미지/영상 평가 실행 절차, 실험 설정 규약, presence 보정 재측정 |
| [protocols/video_rate_benchmark.md](./protocols/video_rate_benchmark.md) | 의미통신 payload vs H.264/H.265/AV1 코덱 비교 방법 |
| [protocols/datasets.md](./protocols/datasets.md) | 데이터셋 역할·stage 매핑·변환 워크플로 |
| [protocols/reproducibility.md](./protocols/reproducibility.md) | checkpoint 선택 기준, `paper_mode` 사용법 |
| [protocols/training.md](./protocols/training.md) | stage-aware 학습 CLI, export, real-model smoke 검증 |

데이터 디렉터리 자체 문서는 [../data/README.md](../data/README.md),
ETRI 10-영상 평가셋은 [../data/etri_video_eval/README.md](../data/etri_video_eval/README.md).

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

## 5. 참조 문서 (`reference/`)

| 문서 | 내용 |
|---|---|
| [reference/paper_alignment.md](./reference/paper_alignment.md) | 논문 정합성, `paper_mode`, 하이퍼파라미터 출처, 충실도 분류 |
| [reference/framework_file_roles.md](./reference/framework_file_roles.md) | 파일별 실행 흐름과 역할 지도 |
| [reference/paper_writing_notes.md](./reference/paper_writing_notes.md) | 논문 작성용 내부 메모(draft) |

## 6. 발표·보고 자료 (`reports/`)

[reports/README.md](./reports/README.md) 참고 — 2026-08-16 슬라이드 상세설명(국/영문,
내부/외부공유용)과 Q&A 문서, PPTX/ZIP 원본 관리 방침.

## 7. 과거 문서 (`archive/`)

더 이상 활성 상태가 아닌 과거 계획·Phase 단위 구현 기록. 현재 상태와 충돌하지
않도록 분리했다 — 최신 판단 기준은 항상 `current/`다.

| 문서 | 내용 |
|---|---|
| [archive/etri_implementation_log.md](./archive/etri_implementation_log.md) | 1차~6차, LGVSC 1A/1B/1C 상세 구현 로그 |
| [archive/phase4_2026-07.md](./archive/phase4_2026-07.md) | Phase 4(패킷 인식 평가 + 영상 확장) 구현 당시 스냅샷 |
| [archive/phase5_2026-07.md](./archive/phase5_2026-07.md) | Phase 5(채널 조건화·저지연·강화 검증) 구현 당시 스냅샷 |
| [archive/video_extension_lgvsc_2026-07.md](./archive/video_extension_lgvsc_2026-07.md) | LGVSC 매핑 설계 + 로드맵 원안(현재는 `architecture/tx_rx_contract.md` + `current/roadmap.md`로 분리됨) |
| [archive/phases_1to3.md](./archive/phases_1to3.md) | 초기 Phase 1~3 스냅샷 |
| [archive/etri_development_plan_v2.md](./archive/etri_development_plan_v2.md) | 통합 전 개발계획서 보관본 |
| [archive/etri_development_roadmap.md](./archive/etri_development_roadmap.md) | 통합 전 로드맵 보관본 |
| [archive/limitation_reference_map.md](./archive/limitation_reference_map.md) | 통합 전 한계점 지도 보관본 |
| [archive/framework_comparison.md](./archive/framework_comparison.md) | 통합 전 프레임워크 비교 보관본 |
| [archive/paper_gap_closure.md](./archive/paper_gap_closure.md) | 통합 전 paper-mode 정책 문서 보관본 |
| [archive/paper_training_alignment.md](./archive/paper_training_alignment.md) | 통합 전 학습 정합 문서 보관본 |

## 개발 원칙

1. **알고리즘 경로 보존** — `SGDJSCC/inference_one.py`의 forward 수치를 그대로 유지한다.
2. **관심사 분리** — 채널은 `channels/`, 가이드는 `guidance/`, 모델은 `models/`,
   오케스트레이션은 `pipelines/`, 지표는 `evaluators/`로 독립 교체 가능하게 둔다.
3. **원본 읽기 전용** — 새 아이디어는 `SGDJSCC/`가 아니라 `sgdjscc_lab/`에 구현한다.

전체 디렉터리 구성과 Phase 게이트 규칙은 [architecture/system.md](./architecture/system.md)
참고.

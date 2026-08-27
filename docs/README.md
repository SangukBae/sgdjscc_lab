# sgdjscc_lab 문서 색인

- 프로젝트 관계
  - 원본 `SGDJSCC/`: 읽기 전용 논문 baseline
  - `sgdjscc_lab/`: 모듈화·평가·연구 확장
  - 기본 추론: 원본 forward-pass와 수치 동일
- 문서 원칙
  - 역할별 분리
  - 현재 계획·상태·실험 이력 분리

## 연구개발할 때 먼저 볼 문서

| 목적 | 기준 문서 |
|---|---|
| 다음 구현 작업과 우선순위 | **[current/roadmap.md](./current/roadmap.md)** — 메인 작업 문서 |
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
| [protocols/results_registry.md](./protocols/results_registry.md) | 추적 `results/` 구조, run manifest 스키마·생성 절차 |
| [protocols/training.md](./protocols/training.md) | stage-aware 학습 CLI, export, real-model smoke 검증 |
| [protocols/transmission_normalization.md](./protocols/transmission_normalization.md) | 전송 정상화와 단일/3-GPU 안전 실행 절차 |

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

## 5. 참조 문서 (`reference/`)

| 문서 | 내용 |
|---|---|
| [reference/paper_alignment.md](./reference/paper_alignment.md) | 논문 정합성, `paper_mode`, 하이퍼파라미터 출처, 충실도 분류 |
| [reference/framework_file_roles.md](./reference/framework_file_roles.md) | 파일별 실행 흐름과 역할 지도 |
| [reference/paper_writing_notes.md](./reference/paper_writing_notes.md) | 논문 작성용 내부 메모(draft) |

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

- 추가 기준
  - 전체 디렉터리·Phase gate: [system.md](./architecture/system.md)

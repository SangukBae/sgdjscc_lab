---
status: archived
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: ec367bb
supersedes: docs/etri_strategy.md
---

> [← 문서 색인](../README.md)

# ETRI 구현 이력 요약

- 용도
  - 과거 구현 순서 확인
  - 현재 계획·상태 판단에는 사용하지 않음
- 현재 기준
  - 계획: [current/roadmap.md](../current/roadmap.md)
  - 상태: [current/status.md](../current/status.md)
  - 상세 실험: `docs/experiments/`
- 원문
  - Git 이력의 옛 `docs/etri_strategy.md`

## 1차~6차

### 1차 — 영상 파이프라인

- 구현
  - MP4·프레임 입출력
  - keyframe 추출
  - temporal evaluator
  - motion·semantic 이중 gate
- 결과
  - 구조 검증 완료
  - 실데이터 threshold 튜닝 미완료

### 2차 — Packet Verifier

- 구현
  - 누락·추가·관계·속성 오류 분류
  - severity 계산
  - controller action 결정
- 제한
  - action을 실제 sampler에 주입하지 않음

### 3차 — Segment Generate

- 구현
  - `reuse/recompute/generate` 3-way 분기
  - segment 단위 생성 계약
  - mock backend 검증

### 4차 — Bidirectional Generate

- 구현
  - start/end keyframe 계약
  - fallback 정책
  - Rx-legal 입력 경계
- 제한
  - 초기 보간은 mock

### 5차 — Presence Calibration

- 구현
  - CLIP·OWLv2·VQA·GT backend
  - closed/open-world filter
  - held-out 재측정
- 결과
  - ETRI 10영상 실측 완료

### 6차 — Transmission Accounting

- 구현
  - semantic-unit accounting
  - 4/6/8/16-bit packet 직렬화
  - exact bundle byte 측정
- 제한
  - 물리 채널 symbol·FEC는 proxy

## LGVSC 후속

### 1A — Segment 계약

- 완료
  - `SegmentGenerationRequest`
  - `SegmentGenerationResult`
  - `generate_segment()`

### 1B — 외부 Worker

- 완료
  - subprocess IPC
  - SVD start-only GPU 검증
  - Wan start-only GPU 검증
  - Wan FLF2V bidirectional GPU 검증
- 상세
  - [1B worker 검증](../experiments/2026-07_lgvsc_1b_worker_validation.md)

### 1C — 재현선

- 준비 완료
  - 4개 모드 config
  - batch driver
  - summary 생성
- 미완료
  - 10영상×4모드 실제 GPU 실행
- 상세
  - [1C 재현 준비](../experiments/2026-07_lgvsc_1c_reproduction.md)

### PSSS/SKEM

- 완료
  - PSSS 수식 구현
  - fixed/variable selector 분리
  - mock·proxy CPU 검증
- 미완료
  - 실제 MLLM PSSS 실행
  - 공정 CBR matched 비교
- 상세
  - [PSSS/SKEM 검증](../experiments/2026-07_lgvsc_psss_skem.md)

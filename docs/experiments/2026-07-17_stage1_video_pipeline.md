---
status: frozen
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
experiment_commit: f000b3c
documentation_commit: ec367bb
supersedes:
---

> [← 문서 색인](../README.md)

# Stage 1 영상 파이프라인 검증

## 판정

- 상태: 완료
- 범위: 개발 순서 0~4
- 주의
  - PTC/SFR/SDI: 당시 CLIP·packet 기반 잠정치
  - 최종 판단: OWLv2/VQA 재측정 결과 사용

## 구현

- 입력·출력
  - MP4↔frame 변환
  - 복원 frame·MP4 저장
  - stale output 정리
- 시간축
  - `PTC`, `SFR`, `SDI`
  - semantic+motion 이중 gate
  - `SegmentRecord`, `segments.json`
- 호환성
  - motion gate 기본 OFF
  - 기존 이미지 경로 불변

## 검증

- 로컬
  - 집중 테스트: 86 passed
  - 전체 테스트: 518 passed
- 원격 컨테이너
  - 집중 테스트: 82 passed
  - 8프레임→3프레임 재실행
    - stale frame 제거 확인
    - 최종 frame 정확히 3개 확인
- 실제 모델 smoke
  - 입력: 6프레임, 256×256
  - 설정: SNR 5dB, diffusion 10 step
  - 결과
    - keyframe: 2
    - semantic recompute: 4
    - `recon.mp4`, `segments.json`, temporal CSV 생성

## 주요 파일

- `utils/video_io.py`
- `video/segment.py`
- `video/temporal_pipeline.py`
- `evaluators/temporal_consistency.py`
- `scripts/evaluate_video.py`

## 후속

- motion threshold 실데이터 튜닝
- OWLv2/VQA presence 보정
- generate branch 연결
- packet verifier·전송 accounting 연동

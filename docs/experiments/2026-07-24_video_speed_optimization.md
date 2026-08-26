---
status: frozen
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
experiment_commit: unknown
documentation_commit: ec367bb
supersedes:
---

> [← 문서 색인](../README.md)

# 영상 평가 속도 최적화

## 병목

- 입력: 512×256 영상
- patch: 프레임당 128×128 패치 8개
- 주요 비용
  - 패치별 diffusion 순차 실행
  - 기본 50 step
  - 반복 BLIP2·CLIP 호출
  - 진행률 부재

## 개선

- profiling
  - frame별 diffusion·BLIP2·CLIP 호출 수
  - `progress.json`
  - `profiling_summary.json`
- 캐시
  - CLIP text embedding
  - 원본 frame packet
- 실행 옵션
  - diffusion step 조절
  - keyframe-only
  - caption·CLIP 생략
  - multi-GPU batch
- 버그 수정
  - 생성 config의 `model_root` 절대경로 처리
  - `.cuda()` 하드코딩 경로의 GPU remapping

## 실측

| 모드 | 100프레임 시간 | diffusion 호출 | PTC | SFR |
|---|---:|---:|---:|---:|
| keyframe-only, step10 | 253.9초 | 72 | 0.504 | 0.037 |
| all-frames, step10 | 311.6초 | 208 | 0.523 | 0.111 |

- 환경
  - RTX 4090×3
  - 영상: `01_person_walk`
- 해석
  - reuse frame: 약 0.03초
  - recompute frame: 약 8.2초
  - 표본 1개이므로 품질 우위 주장 금지

## 운영 권장

1. step10 keyframe-only로 전체 배선 확인
2. 모션이 큰 영상에 all-frames 적용
3. 필요 영상만 step50 실행
4. `--no-models` 결과는 품질 결과로 사용 금지

## 검증

- 전체 테스트: 753 passed
- multi-GPU
  - GPU 0·1 동시 사용 확인
  - 최대 사용률 98%·88%
- 상세 실행 옵션
  - [평가 절차](../protocols/evaluation.md)

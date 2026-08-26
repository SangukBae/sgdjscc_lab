---
status: frozen
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes:
---

# ETRI 부록 슬라이드 개요

- 대상: 평가 입력·지표·계산 방식
- 기준: 2026년 8월
- 성격: 동결 스냅샷

## 부록 A — 평가 입력

- 이미지
  - Kodak
  - SNR sweep
- 영상
  - ETRI 10개
  - 512×256
  - 10fps
  - 영상당 100프레임
- 의미 GT
  - sampled-frame object presence
  - closed/open-world 해석 분리

## 부록 B-1 — 이미지 지표

- PSNR
  - 픽셀 오차
  - 높을수록 우수
- SSIM
  - 구조 유사도
  - 높을수록 우수
- LPIPS
  - 지각 거리
  - 낮을수록 우수
- CLIP
  - image-image
  - text-image
- SRS
  - CLIP 유사도
  - object preservation
  - missing/additional penalty

## 부록 B-2 — Packet·영상 지표

- Packet Verifier
  - object
  - relation
  - attribute
  - scene
- PTC
  - frame별 packet 일치도 평균
- SFR
  - 비정상 object birth/death 비율
- SDI
  - keyframe 거리 대비 불일치 기울기
- temporal hallucination
  - 시간축 additional object 비율

## 평가 원칙

- loop-internal
  - 재생성·제어용
- held-out
  - 최종 성능 보고용
- rate
  - exact packet byte와 proxy symbol 분리
- 통계
  - 영상별 paired 결과
  - 평균·표준편차·신뢰구간

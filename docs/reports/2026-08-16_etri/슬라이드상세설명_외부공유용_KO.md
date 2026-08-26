---
status: frozen
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes:
---

# ETRI 진행상황 슬라이드 개요

- 대상: 외부 공유
- 기준: 2026년 8월
- 성격: 발표 시점 동결본
- 현재 문서: [문서 색인](../../README.md)

## 발표 흐름

1. 과제 목표
2. 구현 상태
3. 기준 파이프라인
4. 영상 확장
5. 할루시네이션 검증
6. 시간축 평가
7. 후속 계획

## 슬라이드 1 — 연구 주제

- 대상: 생성형 AI 기반 시맨틱 미디어 전송
- 핵심: 복원 화질+의미 신뢰성
- 위험: 생성 모델 할루시네이션

## 슬라이드 2 — 문제 정의

- 목표
  - 전송량 감소
  - 복원 성능 유지
  - 의미 누락·추가 억제
- 평가
  - 픽셀 품질
  - 의미 보존
  - 시간축 안정성

## 슬라이드 3 — 구현 상태

- 완료
  - 이미지 추론·평가
  - packet verifier
  - keyframe 영상 파이프라인
  - temporal metric
- 부분 완료
  - channel conditioning
  - low-latency sampling
  - regeneration search
- 미완료
  - verifier action의 sampler 주입
  - 실제 CBR/FEC 검증

## 슬라이드 4 — 기준 파이프라인

```text
Image/Frame
  → VAE·JSCC encoder
  → Wireless channel
  → Diffusion reconstruction
  → Quality·semantic evaluation
```

- 기본 채널: AWGN
- 선택 채널: Rayleigh, fading, packet drop
- 기본 확장: 모두 OFF

## 슬라이드 5 — 영상 확장

- 송신
  - keyframe
  - semantic delta
  - motion signal
- 수신
  - reuse
  - recompute
  - generate
- 단위
  - frame
  - GOP
  - segment

## 슬라이드 6 — 할루시네이션 검증

- 검출
  - missing object
  - additional object
  - relation·attribute error
- 보정
  - CLIP
  - OWLv2
  - VQA
  - GT
- 현재 한계
  - controller는 action 결정·기록
  - sampler 반영 미구현

## 슬라이드 7 — 시간축 지표

- PTC
  - packet 일치도
  - 높을수록 우수
- SFR
  - 비정상 object birth/death
  - 낮을수록 우수
- SDI
  - keyframe 거리별 의미 drift
  - 낮을수록 우수
- 원칙
  - loop-internal·held-out 분리

## 슬라이드 8 — 성과와 다음 단계

- 성과
  - 모듈형 연구 프레임워크
  - 영상 시간축 평가
  - packet 기반 자기검증
  - 실제 packet byte accounting
- 다음 단계
  - verifier→sampler 연결
  - 동적 전송 예산
  - 공정 baseline·ablation
  - held-out 영상 검증

## 슬라이드 9 — 후속 연구

- 전송률 적응형 영상 시맨틱 통신
- 수신단 신뢰성 제어형 생성 복원
- 시맨틱 전송 평가 벤치마크

## 발표 시 제한 표현

- 구현 완료 ≠ 성능 우위
- proxy symbol ≠ 실제 CBR
- mock/proxy PSSS ≠ real MLLM PSSS
- 10영상 결과 ≠ 일반화 완료

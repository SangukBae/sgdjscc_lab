---
status: frozen
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes:
---

# ETRI 질의응답 요약

## 시스템

- 송신
  - 이미지 latent
  - edge·caption 등 semantic side-info
- 채널
  - 기본: AWGN
  - 확장: Rayleigh, fading, packet drop
- 수신
  - SNR 추정
  - diffusion denoising
  - image decode
- 목표
  - 저 SNR 의미 보존
  - 할루시네이션 검증

## 질문별 답변

### 1. 오류정정 방식

- 전통 parity 기반 오류정정: 아님
- 적용 방식
  - 잡음에 강한 JSCC 표현 학습
  - diffusion 기반 잡음 제거
  - SNR→diffusion step matching

### 2. 채널 포함 여부

- 포함
- 표준 실험
  - SNR: -5~25dB
  - channel sweep

### 3. Keyframe

- 의미: 대표 정지 frame 1장
- 처리
  - keyframe: 전체 전송
  - interframe: reuse·delta·generate
- 음성: 미포함

### 4. 전송 항목

- 주 정보
  - VAE image latent
- 보조 정보
  - edge
  - caption token
  - mask·structure
- 관계
  - side-info는 복원 조건
  - image latent를 대체하지 않음

### 5. Encoder 의미

- 문서상 의미: 송신단 전체
- 내부 블록
  - image encoder
  - semantic extractor
  - side-info encoder

### 6. Diffusion Denoiser

- 위치: 수신단
- 채널과 관계: 별도 모듈
- 역할
  - channel noise 제거
  - semantic condition 반영

### 7. 전통 통신 대비 장점

- 핵심
  - cliff effect 완화
  - graceful degradation
  - 저 SNR 의미 보존
- 전송량
  - 단일 이미지 절감: 제한적
  - 영상 중복 제거: 주요 절감 지점

### 8. Guidance 최적화

- 단일 이미지 byte 절감: 작음
- 주 목적
  - 채널별 guidance 강도 조절
  - 과도한 prior 억제
  - 복원 신뢰성 유지

### 9. 영상 데이터

- 현재
  - ETRI 10영상 평가셋
  - 대규모 학습셋 없음
- 필요
  - 실제 모션
  - 장면 전환
  - 객체 등장·소멸

### 10. 영상 지표

- 신규 과제 지표
  - temporal SRS
  - SRS flicker
  - PTC
  - SFR
  - SDI
  - temporal hallucination
- 전송 효율
  - exact packet byte
  - overhead reduction
  - proxy symbol 분리

## 핵심 제한

- 구현 완료 ≠ 성능 우위
- 단일 이미지 side-info 절감 ≠ 큰 rate 절감
- 영상 학습 데이터 부족
- 물리 CBR/FEC 검증 미완료

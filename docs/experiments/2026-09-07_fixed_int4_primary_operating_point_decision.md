---
status: active-decision
date: 2026-09-07
owner: ETRI SGD-JSCC 연구팀
source_commit: 8fbe6d98b1bc541387f120e4b7b98944aea8efbb
supersedes: docs/experiments/2026-09-04_int6_etri_operating_point_decision.md
---

> [← 문서 색인](../README.md) ·
> [int6 bridge 결과](./2026-09-07_negative_semantics_int6_bridge_results.md)

# 논문 primary 운용점 `fixed_int4` 재결정

## 결정

2026-09-07부터 negative-semantics 논문 연구선과 신규 primary 비교의 기본
bit-depth를 `fixed_int4`로 사용한다. `fixed_int6`는 primary/default가 아니라
품질 민감도·bit-depth robustness ablation 및 명시적 시연 opt-in 조건으로 유지한다.

이 결정은 2026-09-04의 `fixed_int6` 임시 결정을 대체한다. 과거 int4/int6
실험과 당시 결정 문서는 역사적 artifact로 수정하지 않는다.

## 근거

- 기존 10dB 양자화 재평가에서 int4는 float32 대비 품질 budget을 통과한 최소
  bit-depth였다.
- OVIS Pilot 40영상 formal bridge에서 int6는 int4보다 PSNR +0.100~0.146 dB,
  SSIM +0.00168~0.00228, LPIPS -0.00123~-0.00171로 소폭 개선됐다.
- 같은 bridge에서 int6 bundle bytes는 두 정책 모두 44.217% 증가했다.
- raw/source-paired H_add의 int6−int4 paired CI는 모두 0을 포함해 int6의
  hallucination 완화 이득이 확인되지 않았다.
- ghost의 유효 uncensored 표본은 정책당 4개뿐이며 int6 선택 근거로 사용할 수 없다.

따라서 int6는 작은 품질 이득을 제공하지만 통신 효율 중심의 primary 모델을
지배하지 않는다.

## 이후 실험 계약

- G2 이후 primary condition과 baseline/method 비교는 모두 `fixed_int4`로 맞춘다.
- 동일 rate·compute 비교에서는 method만 바꾸고 bit-depth를 섞지 않는다.
- 핵심 결론의 bit-depth 민감도를 확인할 때만 동일 설정을 `fixed_int6`로 반복한다.
- 시연에서 int6를 쓰면 `demo-quality opt-in`으로 manifest에 명시하고 primary 논문
  결과와 분리한다.
- G1 v1.1 int4 결과는 그대로 보존한다. 이번 결정은 G1 scientific gate를 통과로
  바꾸지 않는다.
- held-out 전에는 fixed_int4를 최종 일반화 operating point라고 부르지 않고
  `primary development operating point`로 표현한다.

## 바로 적용할 범위

- G2 Oracle ABSENT Pilot
- G3 이후 RSM/REVOKE/packet/allocator 개발 실험
- matched-rate·matched-compute ablation의 기준 condition
- 향후 held-out 후보 구성

`fixed_int6` bridge 원본은
`outputs/negative_semantics_int6_bridge_rtx4080/`에 유지하며, 재실행 없이 결과를
비교 근거로 사용한다.

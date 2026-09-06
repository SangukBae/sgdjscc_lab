---
status: active-decision
date: 2026-09-04
owner: ETRI SGD-JSCC 연구팀
source_commit: aae9e26
supersedes_operating_decision: fixed_int4
---

> [← 문서 색인](../README.md) · [현재 상태](../current/status.md) ·
> [논문 계획](../current/negative_semantic_paper_plan.md)

# ETRI 기본 양자화 운용점 `fixed_int6` 결정

## 결정

- 2026-09-04 이후 **ETRI 과제의 신규 개발·시연·최종 운용점 후보는
  `fixed_int6`**를 기본으로 한다.
- `fixed_int4`는 최대 압축 비교점과 G1 v1.1의 동결된 hallucination stress
  configuration으로 유지한다.
- 완료된 `fixed_int4` 실험 문서·CSV·manifest·registry는 역사적 실측 근거이므로
  다시 쓰거나 `int6` 결과로 대체하지 않는다.
- 이 결정은 개발 운용점 변경이며 held-out 통과나 최종 일반화 성능 확정을 뜻하지 않는다.

## 근거

10영상×100프레임 10 dB fixed-selector 재평가에서 두 설정 모두 사전 품질 기준을
통과했다.

| 설정 | bytes/frame | float32 대비 절감 | PSNR 변화 | SSIM 변화 | LPIPS 변화 |
|---|---:|---:|---:|---:|---:|
| `fixed_int6` | 24,626.437 | 26.42% | -0.002218 dB | -0.000108 | -0.0000541 |
| `fixed_int4` | 23,946.501 | 28.45% | -0.052631 dB | -0.002006 | -0.000487 |

`int4`가 `int6`보다 추가로 줄이는 양은 679.936 bytes/frame이며 `int6` 전송량
기준 2.76%다. 반면 `int6`의 PSNR·SSIM은 float32 baseline에 더 가깝다. ETRI
과제에서는 이 작은 추가 절감보다 안정적인 복원 품질과 대외 설명 가능성을 우선한다.
LPIPS 차이는 매우 작고 `int4`가 수치상 조금 낮으므로 `int6`가 모든 품질 지표에서
우월하다고 주장하지 않는다.

기존 `fixed_int4` 선택은 품질 기준을 통과한 설정 중 byte가 가장 작은
rate-first 규칙의 결과였다. 이번 결정은 같은 Pareto frontier에서 ETRI 운용 목적에
맞게 보수적인 품질점을 선택한 것이다.

## 적용 범위와 증거 경계

| 범위 | bit-depth | 상태 |
|---|---|---|
| 2026-08-28 양자화·matched-rate·guide 실험 | `fixed_int4` | 완료된 historical evidence, 변경 금지 |
| 2026-08-29 통합 개발평가 | `fixed_int4` | 완료된 historical evidence, 변경 금지 |
| Negative Semantics G1 v1.1 | `fixed_int4` | 동결 stress configuration; 기존 protocol 유지 |
| ETRI 신규 개발·시연·최종 후보 | `fixed_int6` | 2026-09-04 이후 기본값 |
| Negative Semantics G2 이후 primary method | `fixed_int6` 후보 | bridge validation 통과 후 사용 |

현재 `int6 + both-omit` exact bundle byte와 hallucination 지표는 아직 실측하지
않았다. 따라서 기존 `int4 + both-omit`의 90.843% 절감률을 `int6` 결과로 인용하지
않는다.

## 필수 후속 검증

1. G1 v1.1은 등록된 `fixed_int4 + both-omit` 조건으로 완료한다.
2. 같은 공개 데이터·diffusion step·seed에서 `fixed_int4 + both-omit`과
   `fixed_int6 + both-omit`을 paired bridge ablation으로 비교한다.
3. exact serialized bytes, PSNR·SSIM·LPIPS, additional-object, ghost-track, latency를
   함께 기록한다.
4. `int6`의 exact-byte 증가가 사전 합의 범위 안이고 품질 또는 semantic reliability가
   개선되면 G2 이후와 ETRI 최종 후보를 `fixed_int6`로 확정한다.
5. 개선이 재현되지 않으면 `int6`를 자동 확정하지 않고 Pareto 결정을 다시 기록한다.

근거 수치는
[10 dB 양자화 재평가](./2026-08-28_quantization_reevaluation_10db.md)를 따른다.

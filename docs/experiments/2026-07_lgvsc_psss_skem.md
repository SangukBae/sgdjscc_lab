---
status: frozen
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
experiment_commit: unknown
documentation_commit: ec367bb
supersedes:
---

> [← 문서 색인](../README.md)

# PSSS/SKEM Selector 검증

## 판정

- 수식·selector·metadata: 완료
- mock/proxy smoke: 완료
- 실제 MLLM PSSS: 미완료
- faithful LGVSC reproduction: 아님

## 구성

- PSSS backend
  - `mock`: 단어 Jaccard
  - `proxy`: CLIP text similarity
  - `real`: MLLM next-token P(Yes)/P(No)
- selector
  - `fixed`: 기존 scene-change 기반
  - `fixed_interval`: SKIM 근사
  - `psss`: variable-length SKEM
- provenance
  - backend kind
  - model ID
  - raw logits
  - threshold·selection reason

## 수식

```text
S_abs = P(Yes | Info A, Info B, Semantic Focus)
S_rel = P(No  | ...) - P(Yes | ...)
```

- real backend
  - yes/no 표면형 확률 합산
  - multi-token continuation 지원
  - special token 제외
  - unavailable 시 명시적 실패

## 주요 수정

- GPU input/model device mismatch
- continuation BOS/EOS 오염
- fixed interval selector 부재
- PSSS provenance 누락
- trigger-only score 통계 오류
- exact keyframe-count matching

## CPU smoke 결과

| mode | video | segment 수 | 길이 min/max/mean | PSSS 평균 |
|---|---|---:|---|---:|
| fixed | person | 2 | 2/12/7.0 | N/A |
| mock PSSS | person | 7 | 1/3/2.0 | 0.562 |
| mock PSSS | car | 3 | 4/6/4.67 | 0.320 |
| proxy PSSS | person | 2 | 2/12/7.0 | -0.901 |
| proxy PSSS | car | 2 | 2/12/7.0 | -0.907 |

## 해석

- variable-length 동작: 확인
- proxy score
  - placeholder caption에서 threshold 미도달
  - real PSSS 성능 근거로 사용 금지
- fixed-count match
  - keyframe 수만 맞춤
  - 실제 symbol CBR 동일성은 별도 확인

## 검증

- CUDA device 회귀: 통과
- 실제 Wan+fixed selector smoke: 통과
- 전체 테스트 기록
  - 1090 passed
  - 후속 수정 후 1117 passed

## 남은 작업

- 실제 MLLM model ID 확정
- real backend GPU 실행
- 10영상 전체 비교
- exact CBR matched 결과

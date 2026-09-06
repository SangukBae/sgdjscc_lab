# negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2 — G1 v1.2 amendment (derived, read-only)

> 원본 소스: [negative_semantics_g1_pilot_rtx4080_v1_1](../negative_semantics_g1_pilot_rtx4080_v1_1/)
> (수정하지 않음) · 해석:
> [2026-09-06 G1 v1.2 amendment 문서](../../docs/experiments/2026-09-06_negative_semantics_g1_v1_2_amendment.md)
> 생성: `scripts/derive_negative_semantics_g1_v1_2.py` (GPU/추론 없음, 순수
> 후처리)

## 무엇을 다시 계산했나

동결된 v1.1 run의 `evaluator/detection_rows.jsonl`(108,990행)과
`evaluator/source/*.json`(source-only OWLv2 캐시)만 읽어, GPU 재실행 없이
세 가지를 다시 계산한다.

1. **effective seed count** — 정책별로 seed 간 OWLv2 점수가 완전히 동일한지
   확인해 canonical(유효) seed 집합을 구한다.
2. **source-paired additional-object** — GT-absent 이면서 source 프레임도
   threshold 미만(진짜 음성)이었던 opportunity만으로 "새로 추가된" 오탐율을
   따로 집계한다 (raw H_add는 source가 이미 양성이었던 경우도 포함한다).
3. **gate 재적용** — 동일 `g1_protocol.yaml`의 gate threshold를
   effective-seed 지표에 그대로 대입해, declared-seed 판정과 다른 결과가
   나오는 항목을 표시한다.

## 핵심 결과

| 항목 | declared-seed (v1.1) | effective-seed (v1.2) |
|---|---:|---:|
| effective_seed_count (few10/full50) | 3 (명목) | **1 / 1** |
| additional_event_count (few10) | 495 | **165** |
| additional_detections (few10) | 897 | **299** |
| h_add (few10, 비율이라 불변) | 0.019793 | 0.019793 |
| affected_seed_count (few10) | 3 | **1** |
| max_single_seed_event_share (few10) | 0.3333 | **1.0** |

**gate 재적용 (primary policy = few10, 원 protocol의 h_add_min/additional_events_min/
affected_videos_min/affected_seeds_min/concentration threshold 그대로 사용)**:

- declared-seed 기준: 11개 중 seed 관련 6개 checkpoint 전부 PASS → `PASSED`
- effective-seed 기준: `affected_seed_count`(1 < 2)와
  `not_concentrated_in_one_seed`(1.0 > 0.75) **FAIL로 뒤집힘** → 같은
  threshold를 정직하게 적용하면 **`NOT_PASSED`**

**source-paired additional-object (effective-seed 기준)**:

| policy | raw_h_add | source_paired_h_add | source_paired_opportunities | unpairable |
|---|---:|---:|---:|---:|
| few10 | 0.019793 | **0.010493** | 14,867 | 0 |
| full50 | 0.018536 | **0.009081** | 14,867 | 0 |

full50의 source-paired h_add(0.009081)는 원 protocol의 `h_add_min=0.01`보다
낮다 — reconstruction이 실제로 "새로 추가"한 오탐만 세면, full50 정책은
prevalence gate를 만족하지 못한다.

**ghost survival (effective-seed 기준)**: unique EXIT 이벤트 10개 중
uncensored 4개(정책당), right-censored 12개. declared-seed 표는 이 값을
그대로 3배(12/36)로 보고했었다.

## 무엇이 바뀌지 않았나

- v1.1의 원본 파일(`g1_summary.json`, `detection_rows.jsonl` 등)은 전혀
  수정하지 않았다. 이 디렉터리는 별도 산출물이다.
- 재구성 프레임을 다시 만들거나 GPU 추론을 다시 실행하지 않았다 (결정론적
  디코더이므로 실제로 다시 돌려도 동일 결과가 나올 것으로 예상되지만,
  검증되지 않은 가정을 새로 추가하지 않기 위해 실행하지 않았다).
- OWLv2 threshold는 v1.1에서 동결된 값(0.2)을 그대로 재사용했다 —
  bit-depth/정책별로 다시 calibration하지 않았다(그렇게 하면 evaluator가
  더 이상 method-blind가 아니게 된다).

## fail-closed 규칙

`require_effective_seed_declaration()`은 `declared_deterministic_decoder=True`를
명시하지 않으면 seed 붕괴를 감지한 즉시 `ValueError`를 낸다. 이 run은 위
사실(결정론적 디코더, PNG-hash로 확인됨)을 알고 명시적으로 선언했기 때문에
통과했다 — 향후 실제로 확률적인 디코더 경로에서 이 함수가 조용히 통과한다면
그것은 버그(예: seed 배선 누락)를 의미하도록 설계했다.

# negative_semantics_g1_pilot_rtx4080_v1_1 — G1 정식 Pilot 실행 (frozen)

> 원본: `outputs/negative_semantics_g1_pilot_rtx4080_v1_1/` (git 비추적, 대용량)
> 해석: [2026-09-06 G1 v1.1 결과 문서](../../docs/experiments/2026-09-06_negative_semantics_g1_v1_1_pilot_results.md)
> 후속 개정: [negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2](../negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2/)

## 이 run이 검증하는 것

`negative_semantics_g1_v1_1` 동결 protocol
(`configs/experiments/negative_semantics/g1_protocol.yaml`)의 정식 실행. OVIS
official-GT Pilot 40영상 × `{few10, full50}` × `{seed 2025, 2026, 2027}` = child
run 6개 × 40영상 = **240/240**, 실패 0, held-out 미접근으로 **runner 완료**했다.
독립적인 재검증은 `scripts/audit_negative_semantics_g1.py --require-pass`가
수행하며, 그 결과가 `audit_report.json`이다 (`runner_gate_status: "PASSED"`,
`exit 0`).

## runner PASSED ≠ 논문 주장 범위

이 문서에서 "runner PASSED"와 "논문에서 주장 가능한 범위"를 명확히 구분한다.

- **runner PASSED (사실)**: 240/240 조합이 실패 없이 완료됐고, 정해진
  `evaluator/evaluator_freeze.json` threshold(0.2, recall 0.8349/FPR 0.0158)로
  reconstruction을 채점했으며, held-out 데이터를 열지 않았다.
- **`g1_summary.json`의 `gate_status: PASSED`도 사실이다** — 단, 이 판정은
  `seeds: [2025, 2026, 2027]`을 **독립 표본**으로 가정한 11개 gate 중
  `affected_seed_count >= 2`와 `not_concentrated_in_one_seed`를 포함한다.
- **이 reconstruction 경로는 seed를 소비하지 않는다.** `few10`/`full50` 세
  seed의 reconstruction PNG는 픽셀까지 완전히 동일하고(`sha256` 일치 확인),
  `detection_rows.jsonl`의 OWLv2 점수도 seed 간 완전히 동일하다. 즉
  **effective seed count = 1**이며 "3 seed에 걸쳐 나타나는 현상"이라는
  claim은 이 run만으로는 성립하지 않는다.
  [v1.2 amendment](../negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2/)가
  이 gate를 effective-seed 기준으로 재적용하면 **`affected_seed_count`와
  `not_concentrated_in_one_seed` 두 항목이 FAIL로 뒤집혀 primary policy
  `few10` 기준 gate는 `NOT_PASSED`가 된다.**
- 따라서 이 run이 논문에서 뒷받침하는 것은 "OVIS Pilot 40영상, `fixed_int4`
  stress 조건, **effective seed 1개**에서 additional-object/ghost-track
  현상이 관측됐다"이다. "여러 무작위 seed에 걸쳐 재현된다"는 주장은 별도의
  실제 확률적 seed(예: diffusion sampler noise를 실제로 소비하는 경로)로
  재검증하기 전까지 하지 않는다.

## 핵심 수치 (v1.1 원 정의, declared-seed 기준)

| policy | h_add | absent_opportunities | additional_detections | additional_event_count | affected_video_count | affected_seed_count |
|---|---:|---:|---:|---:|---:|---:|
| few10 | 0.019793 | 45,318 | 897 | 495 | 24 | 3 (declared) |
| full50 | 0.018536 | 45,318 | 840 | 441 | 23 | 3 (declared) |

`h_add_video_clustered_bootstrap_95ci`(few10) = [0.00954, 0.03331]. 이
video-clustered bootstrap 자체는 seed 중복의 영향을 받지 않는다 (분자·분모가
seed마다 동일 배수로 커져 영상당 비율은 변하지 않음).

## 알려진 한계 (v1.2에서 정량화)

1. **seed 중복**: `effective_seed_count = 1` (few10, full50 모두). declared
   3-seed 대비 event/추가-검출 raw count가 정확히 3배 부풀려져 있다.
2. **source-paired additional-object 미분리**: v1.1의 `h_add`는 GT-absent
   opportunity 전체를 분모로 쓴다. 그중 source 프레임이 이미 OWLv2 threshold를
   넘는(즉 detector가 원본에서도 오탐하는) 경우까지 "추가 생성"으로 집계한다.
   v1.2에서 source가 실제로 음성이었던 opportunity만으로 다시 계산하면
   few10 h_add가 0.019793 → **0.010493**(약 47% 감소), full50은 0.018536 →
   **0.009081**로 `h_add_min=0.01` 미달로 낮아진다.
3. **ghost 표본 크기**: EXIT 이벤트는 Pilot 전체에서 unique 10개뿐이고, 16
   annotated timestep 동안 GT가 계속 부재해 uncensored로 남는 것은
   effective-seed 기준 policy당 4개뿐이다 (right-censored 12개). ghost
   survival AUC는 이 4-표본 기준의 기술 통계이며 신뢰구간을 보고하지 않는다.

상세 수치·재현 명령·해석은 위 실험 문서를 참조.

## 이 디렉터리에 없는 것

`results/`는 git 추적 대상이라 큰 바이너리/대용량 로그를 복사하지 않는다.
원본은 `outputs/negative_semantics_g1_pilot_rtx4080_v1_1/`에 있으며,
`manifest.json`의 `extra.large_artifacts_not_copied`에 해당 파일의
sha256·크기를 기록했다 (`evaluator/detection_rows.jsonl` 17MB,
`operator.log` 30MB). 240개 reconstruction PNG/packet 트리(3.4GB)도 원본에만
있다.

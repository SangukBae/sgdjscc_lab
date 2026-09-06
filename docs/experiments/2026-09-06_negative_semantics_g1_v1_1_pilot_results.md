---
status: frozen
updated: 2026-09-06
gate: G1
protocol_id: negative_semantics_g1_v1_1
run_id: negative_semantics_g1_pilot_rtx4080_v1_1
heldout_opened: false
supersedes: (none — first formal-result write-up for this run)
amended_by: 2026-09-06_negative_semantics_g1_v1_2_amendment.md
---

> [← 문서 색인](../README.md)

# G1 v1.1 정식 Pilot 결과와 감사

이 문서는 `outputs/negative_semantics_g1_pilot_rtx4080_v1_1`에서 완료된 정식
G1 실행을 **원본은 수정하지 않고** 감사·보존한 기록이다. 보존 사본과
checksum은 [results/negative_semantics_g1_pilot_rtx4080_v1_1/](../../results/negative_semantics_g1_pilot_rtx4080_v1_1/)에
있다. 이 run의 metric 정의(declared-seed, raw H_add)를 다시 계산한
[v1.2 amendment](./2026-09-06_negative_semantics_g1_v1_2_amendment.md)는 이
문서의 결과를 대체하지 않고 병기한다.

## 실행 개요

- protocol: `negative_semantics_g1_v1_1`
  (`configs/experiments/negative_semantics/g1_protocol.yaml`,
  sha256 `9edda50e...d68c2da`, [G1 준비 기록](./2026-09-03_negative_semantics_g1_preparation.md))
- git commit: `aae9e268dccce2cf3db9727720fa8f64883c1ad0` (dirty: false, run_spec.json 기준)
- 데이터: OVIS official-GT Pilot 40영상 (held-out 미포함, `data/negative_semantics/g1/pilot`)
- 조건: `fixed_int4`, `candidate_both_omit` guide, fixed selector(max_gop 16,
  reuse 0.2), 10dB fixed-reference, `{few10=10, full50=50}` diffusion step,
  seed `{2025, 2026, 2027}`
- evaluator: `google/owlv2-base-patch16-ensemble`, source-only calibration으로
  동결한 threshold `0.2` (recall 0.8349, FPR 0.0158) — reconstruction
  scoring에는 method/seed/step label을 주지 않았다

## 1. runner 완료 검증 (`scripts/audit_negative_semantics_g1.py --require-pass`)

새로 작성한 read-only 감사 스크립트로 다음을 확인했다 (`exit 0`,
`runner_gate_status: PASSED`, 10/10 required check):

| 확인 항목 | 결과 |
|---|---|
| smoke 아님 (`run_spec.smoke=false`) | PASS |
| held-out 미접근 (`run_spec.heldout_accessed=false`, video_id 40개 전부 `ovis_valid_*`) | PASS |
| child run 6개(=2 policy × 3 seed) × 40영상 = **240/240**, `n_failed_pairs=0`, 전부 `run_status=completed` | PASS |
| `detection_rows.jsonl` 108,990행, 7개 concept 전부 존재 | PASS |
| `g1_summary.json.evidence_scope == "OVIS_PILOT_G1"`, `heldout_accessed=false` | PASS |
| `g1_summary.json.gate_checks` 11개 전부 `true` (`gate_status: PASSED`) | PASS |
| held-out 소스(DAVIS) video_id 없음 | PASS |

held-out 관련 로그(`operator.log`)의 유일한 `davis`/`heldout` 언급은
`_require_g0_pass()`가 호출하는 G0 감사의 **manifest-level seal 확인**
(video 수·해시 검증)이며, 실제 held-out 미디어 파일을 읽지 않는다. 이 감사
자체도 G1 protocol의 고정 요구사항(“G0가 먼저 통과해야 G1 실행 허용”)이다.

보존 사본: `results/negative_semantics_g1_pilot_rtx4080_v1_1/audit_report.json`.

## 2. runner PASSED와 논문 주장 가능 범위의 구분

**두 문장을 섞지 않는다.**

- **사실 A (runner)**: 240/240 조합이 실패 없이 완료됐고 held-out을 열지
  않았다. → 이 사실 자체는 v1.2 amendment로도 바뀌지 않는다.
- **사실 B (`g1_summary.json`의 `gate_status: PASSED`)**: `few10`(primary
  policy) 기준 11개 gate check가 모두 참이라 protocol이 정의한 대로
  `PASSED`다. 이 사실도 v1.1 정의로는 그대로 유지된다(수정하지 않음).
- **한계 (여기서 새로 확인한 것)**: 사실 B의 `affected_seed_count`,
  `not_concentrated_in_one_seed` 두 gate는 `seeds=[2025,2026,2027]`이
  **독립 표본**이라는 암묵적 가정에 기대고 있다. 그런데 이 reconstruction
  경로(고정 selector + 고정 diffusion schedule)는 `seed` 인자를 실제 샘플링
  잡음에 사용하지 않는다 — 세 seed의 reconstruction PNG가 픽셀 단위로
  완전히 동일함을 `sha256`으로 직접 확인했고, `detection_rows.jsonl`의
  OWLv2 점수도 seed 간 완전히 동일하다. 즉 **effective seed count = 1**
  (few10/full50 모두)이며, "3개 독립 seed에 걸쳐 나타난다"는 claim은 이
  run만으로 성립하지 않는다.

이 한계를 정량화한 재계산이
[v1.2 amendment](./2026-09-06_negative_semantics_g1_v1_2_amendment.md)이며,
같은 gate threshold를 effective-seed 지표에 그대로 대입하면
`affected_seed_count`와 `not_concentrated_in_one_seed` 두 항목이 FAIL로
뒤집혀 few10 기준 gate가 `NOT_PASSED`가 된다.

**따라서 이 run이 논문에서 뒷받침하는 것은**: "OVIS Pilot 40영상,
`fixed_int4` stress 조건, effective seed 1개에서 additional-object/
ghost-track 현상이 관측됐고 (raw h_add few10 0.0198, full50 0.0185, video
전체에 분산되어 있음: `affected_video_count` 24/23, `max_single_video_event_share`
0.18/0.20)". **뒷받침하지 않는 것**: "여러 무작위 seed에 걸쳐 재현된다"는
주장, "reconstruction이 새로 만들어낸 오탐"이라는 정량 주장(→ source-paired
분리는 v1.2에서), "held-out에 일반화된다"는 주장(G1은 Pilot 전용, held-out은
G1 통과 후 별도 단계).

## 3. 알려진 한계 요약

1. **seed 중복** — `effective_seed_count=1`. §2 참고, 정량화는 v1.2.
2. **source-paired additional-object 미분리** — v1.1의 `h_add`는 GT-absent
   opportunity 전체를 분모로 쓰며, source 프레임이 이미 detector 양성이었던
   경우(=reconstruction이 만든 게 아니라 detector가 원본에서도 틀리는 경우)를
   구분하지 않는다. v1.2에서 분리하면 few10 h_add가 0.019793 → 0.010493(약
   47% 감소), full50은 0.018536 → 0.009081로 `h_add_min=0.01` 미달이 된다.
3. **ghost 표본 크기** — Pilot 전체 unique EXIT 이벤트는 10개뿐이고, 16
   timestep 동안 GT-absent가 유지돼 uncensored로 남는 것은 effective-seed
   기준 policy당 4개뿐이다(right-censored 12개). declared-seed 표는 이를
   그대로 3배(12/36)로 부풀려 보고했다. ghost survival AUC는 이 4-표본
   기술 통계이며 신뢰구간이 없다 — 별도 confidence interval을 보고하지
   않은 것은 amendment가 아니라 애초 v1.1 protocol에서도 정의되지 않았다.

## 4. 다음 단계

- G1 v1.2 amendment: [문서](./2026-09-06_negative_semantics_g1_v1_2_amendment.md),
  [보존 결과](../../results/negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2/README.md)
- G1 통과 후, G2 전 `fixed_int6 + both-omit` bridge validation:
  [준비 기록](./2026-09-06_negative_semantics_int6_bridge_preparation.md)
  (GPU 실행은 사용자가 별도로 수행)
- ghost 표본을 늘리려면 Pilot보다 큰 독립 데이터셋에서 EXIT 이벤트를 추가
  확보해야 한다 (현재 40영상 Pilot의 event 구성은 동결되어 변경하지 않음).

## 재현

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest
python scripts/audit_negative_semantics_g1.py \
  --run-root outputs/negative_semantics_g1_pilot_rtx4080_v1_1 --require-pass
```

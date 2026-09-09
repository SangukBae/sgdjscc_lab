---
status: completed_negative_result
date: 2026-09-09
owner: ETRI SGD-JSCC 연구팀
gate: SV0_G2
protocol_id: negative_semantics_g2_oracle_absent_v1_0
run_id: negative_semantics_g2_pilot_20260907_v1
source_commit: 6f9593d8ee49d86a98351ae6652930fc1c85e8d6
evidence_scope: EXPLORATORY_MECHANISM_FEASIBILITY_G1_NOT_PASSED
runner_gate_status: PASSED
provisional_gate_status: NOT_PASSED
scientific_gate_status: NOT_CONFIRMATORY_DEPENDENCY_G1_NOT_PASSED
heldout_opened: false
track_decision: STOP_UNDER_FROZEN_PLAN
---

> [← 문서 색인](../README.md) ·
> [G2 구현 기록](./2026-09-07_negative_semantics_g2_implementation.md) ·
> [Negative-semantics 계획](../current/negative_semantic_paper_plan.md) ·
> [SAVER-JSCC 계획](../current/saver_jscc_model_plan.md)

# SV0/G2 Oracle ABSENT Pilot 결과

## 결론

OVIS Pilot 40영상에서 `fixed_int4 + candidate_both_omit`의
`no_negative`/`random_negative`/`frequency_negative`/`oracle_negative`를
`few10`/`full50`으로 비교했다. 실행·산출물 무결성과 matched-compute 계약은
통과했지만, primary `few10`에서 Oracle ABSENT가 source-paired H_add를 줄이지 못했다.

- 실행 결과: **정상 완료**, 320/320 video run, 실패 0
- provisional mechanism gate: **`NOT_PASSED`**
- G1 dependency: **`NOT_PASSED`**
- 최종 scientific gate: **`NOT_CONFIRMATORY_DEPENDENCY_G1_NOT_PASSED`**
- 동결 계획상 결정: **Stop Track**. 동일 구조의 G3 실행과 formal SAVER 학습은 진행하지
  않는다. 계속하려면 receiver control mechanism을 바꾼 versioned 재설계가 필요하다.

`complete`는 runner가 끝났다는 뜻이지 G2 가설이 통과했다는 뜻이 아니다. 최종 출력은
G1 dependency를 앞에 표시하지만 `g2_summary.json.provisional_gate_status`도 별도로
`NOT_PASSED`다.

## 실행 조건과 무결성

| 항목 | 결과 |
|---|---|
| Git | `6f9593d8ee49d86a98351ae6652930fc1c85e8d6`, 시작 시 tracked-clean |
| 데이터 | OVIS official-GT Pilot 40영상, Held-out 미접근 |
| 조건 | 4 arm × `{few10=10, full50=50}` × seed 2025 = 8 child run |
| 채널·가이드 | `fixed_int4`, 10 dB fixed-reference, `candidate_both_omit` |
| 완료 | 320/320 video run, `n_failed_pairs=0` |
| evaluator | frozen OWLv2, threshold 0.2, source-paired H_add |
| detection grid | 145,320/145,320 unique row |
| accounting grid | 320/320 row |
| matched compute/rate | 240 comparison, mismatch 0, `PASSED` |
| 기록된 핵심 checksum | 10/10 일치 |
| Oracle rate 경계 | `ORACLE_EVAL_ONLY`, packet 미직렬화, rate 미계상 |
| wall time | 약 35시간 51분 |

Read-only 감사 결과:

- `--require-runner-complete`: exit 0
- `--require-provisional-gate`: exit 3
- protocol hash, condition manifest, child completion, detection/accounting grid,
  matched compute/rate, 기록 checksum, Held-out seal 검사를 모두 통과했다.

## primary few10 결과

Primary estimand는 원본 프레임에서도 evaluator가 양성이었던 경우를 제외한
source-paired H_add이며 낮을수록 좋다.

| arm | source-paired H_add | 추가 검출/기회 | no-negative 대비 상대 변화 |
|---|---:|---:|---:|
| `no_negative` | 1.0493% | 156/14,867 | 기준 |
| `random_negative` | 1.0359% | 154/14,867 | 1.28% 감소 |
| `frequency_negative` | 1.0493% | 156/14,867 | 변화 없음 |
| `oracle_negative` | **1.0493%** | **156/14,867** | **0% 감소** |

동결 gate는 Oracle의 상대 감소율 25% 이상과 candidate-minus-reference delta의
one-sided 95% CI 상한이 0보다 작을 것을 요구했다. 실제 결과는 다음과 같다.

| provisional check | 요구 | 결과 | 판정 |
|---|---:|---:|---|
| Oracle H_add 상대 감소 | ≥25% | 0% | FAIL |
| H_add delta one-sided 95% CI 상한 | <0%p | +0.0406%p | FAIL |
| false suppression one-sided 95% 상한 | ≤2% | 0.4147% | PASS |
| PSNR non-inferiority | drop ≤0.5 dB | delta +0.0421 dB | PASS |
| SSIM non-inferiority | drop ≤0.01 | delta +0.000489 | PASS |
| LPIPS non-inferiority | rise ≤0.02 | delta -0.000942 | PASS |

Oracle false suppression point estimate는 5/2,413 = 0.2072%였다. 품질과 실제
GT-present 객체 보존 조건은 통과했지만, 목표였던 additional-object 억제 효과가 없었다.

## full50 보조 결과

| arm | source-paired H_add | 추가 검출/기회 | no-negative 대비 상대 변화 |
|---|---:|---:|---:|
| `no_negative` | 0.9081% | 135/14,867 | 기준 |
| `random_negative` | 0.9417% | 140/14,867 | 3.70% 악화 |
| `frequency_negative` | 0.9350% | 139/14,867 | 2.96% 악화 |
| `oracle_negative` | **0.9282%** | **138/14,867** | **2.22% 악화** |

Oracle quality delta는 PSNR +0.0432 dB, SSIM +0.000608, LPIPS -0.000725였고,
false suppression은 4/2,428 = 0.1647%(one-sided 95% 상한 0.3772%)였다. 악화
point estimate 자체를 유의한 악화로 주장하지 않지만, 개선 근거는 아니다.

## raw H_add와 source-paired H_add의 구분

few10 raw H_add만 보면 Oracle이 1.9793%에서 1.9661%로 소폭 감소한다. 그러나
source detector가 원본에서도 양성이었던 opportunity를 제외하면 156건에서 156건으로
완전히 같아진다. 따라서 raw 변화만으로 Oracle 억제 효과를 주장하지 않는다.

Random few10은 2건 감소했지만 상대 감소 1.28%로 25% gate에 크게 못 미치며 bootstrap
CI도 개선을 확정하지 못했다. Oracle이 random/frequency보다 우수하지 않으므로 absence
정보의 semantic specificity가 복원 효과로 전달됐다는 근거도 없다.

## 연구 해석과 후속 결정

1. condition manifest, 실행 grid와 output metric은 정상이며 실행 실패로 인한
   `NOT_PASSED`가 아니다.
2. 현재 frozen diffusion에 confirmed-absent concept을 negative text로 append하는
   prompt-only 제어는 SV0의 필요 효과를 만들지 못했다.
3. 이 Oracle 입력은 packet 밖의 rate-free upper-bound이므로, 통과했더라도 SAVER의
   rate-bearing 성능 근거는 아니었다.
4. G1도 effective seed 1개와 full50 source-paired prevalence 미달 때문에 scientific
   gate가 `NOT_PASSED`다. G2 결과를 G1 통과 또는 confirmatory evidence로 소급하지 않는다.
5. Held-out은 열지 않았으며, 이번 Pilot은 본 뒤 threshold·prompt를 튜닝해 같은 데이터로
   confirmatory 재사용하지 않는다.
6. 현재 동결 계획에서는 Stop Track이다. 후속 연구를 계속하려면 prompt-only 입력을
   반복 조정하는 대신, Development split에서 SM-DiT 등 구조적 receiver-state injection을
   진단하고 새 protocol/version/run root로 재설계해야 한다.

구현돼 있는 G3 harness와 SAVER prototype은 삭제하지 않지만, 이 결과만으로 G3 GPU
matrix, SV1/SV2 formal 학습 또는 최종 contribution 채택을 시작하지 않는다.

## 핵심 산출물과 재감사

```text
outputs/negative_semantics_g2_pilot_20260907_v1/
├── run_spec.json
├── preflight.json
├── conditions/{no,random,frequency,oracle}_negative.json
├── reconstruction/<arm>/<policy>/seed_2025/
├── evaluator/detection_rows.jsonl
├── accounting_rows.csv
├── matched_compute_audit.json
├── g2_summary.json
└── artifact_checksums.json
```

```bash
python scripts/audit_negative_semantics_g2.py \
  --run-root outputs/negative_semantics_g2_pilot_20260907_v1 \
  --require-runner-complete

# 결과 gate까지 요구하므로 현재 결과에서는 exit 3이 정상이다.
python scripts/audit_negative_semantics_g2.py \
  --run-root outputs/negative_semantics_g2_pilot_20260907_v1 \
  --require-provisional-gate
```

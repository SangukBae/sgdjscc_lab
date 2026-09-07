---
status: implementation_complete_validation_pending
date: 2026-09-07
gate: SV0_G2
evidence_scope: IMPLEMENTATION_ONLY
heldout_accessed: false
---

# SV0/G2 Oracle ABSENT 구현 기록

## 판정

`fixed_int4 + candidate_both_omit`에서 no/random/frequency/oracle receiver-negative를
비교하는 G2 경로를 구현했다. CPU 단위·통합 회귀만 수행했으며 GPU smoke, 40-video
Pilot와 scientific 판정은 아직 수행하지 않았다. 따라서 이 문서는 성능 결과가 아니다.

G1 v1.2 effective-seed gate가 `NOT_PASSED`인 동안 향후 G2 Pilot가 provisional gate를
통과해도 `EXPLORATORY_MECHANISM_FEASIBILITY_G1_NOT_PASSED`로만 기록한다.

## 구현 범위

- `negative_conditioning.py`: 기존 SGD-JSCC quality-negative를 기본값으로 보존하고,
  명시적 semantic suffix를 일반/early-exit/water-filling diffusion 경로에 공통 주입
- G2 condition: `no_negative`, `random_negative`, `frequency_negative`,
  `oracle_negative`; random/frequency는 Oracle과 frame별 concept 수를 맞춤
- Oracle 정의: official GT의 closed-vocabulary complement
- `run_negative_semantics_g2.py`: 조건 생성, fixed_int4 child run, frozen OWLv2 scoring,
  source-paired H_add, false suppression, 품질 CI와 resume metadata
- `audit_negative_semantics_g2.py`: condition/hash/grid/child completion/matched compute를
  read-only로 검사
- Oracle text는 `ORACLE_EVAL_ONLY`, `serialized_in_packet=false`,
  `rate_accounted=false`다. 네 arm의 bundle bytes와 visual schedule은 동일해야 한다.

## 검증 명령

먼저 구현 commit을 만든다. formal child runner는 tracked-dirty checkout을 거부한다.

```bash
python scripts/run_negative_semantics_g2.py \
  --run-root outputs/negative_semantics_g2_smoke_<run_id> \
  --device cuda:0 --smoke
```

smoke가 완료된 뒤 새 output root에서 Pilot를 실행한다.

```bash
python scripts/run_negative_semantics_g2.py \
  --run-root outputs/negative_semantics_g2_pilot_<run_id> \
  --device cuda:0
```

실행 완결성과 provisional mechanism gate는 분리해 감사한다.

```bash
python scripts/audit_negative_semantics_g2.py \
  --run-root outputs/negative_semantics_g2_pilot_<run_id> \
  --require-runner-complete

python scripts/audit_negative_semantics_g2.py \
  --run-root outputs/negative_semantics_g2_pilot_<run_id> \
  --require-provisional-gate
```

## 예상 산출물

```text
<run_root>/
  conditions/{no,random,frequency,oracle}_negative.json
  reconstruction/<arm>/<policy>/seed_2025/
  evaluator/detection_rows.jsonl
  accounting_rows.csv
  matched_compute_audit.json
  g2_summary.json
  artifact_checksums.json
```

`g2_summary.json.provisional_gate_status`는 Oracle mechanism 기준의 판정이고,
`gate_status`는 G1 dependency까지 반영한 최종 증거 범위다. 둘을 같은 의미로 보고하지
않는다.

## 로컬 CPU 회귀

다음 핵심 범위를 GPU 없이 실행해 `78 passed`를 확인했다. diffusion route와 receiver
회귀를 추가한 확대 실행은 총 `189 passed`였다.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_negative_semantics_g2.py \
  tests/test_transmission_reduction_eval.py \
  tests/test_transmission_reduction_temporal_integration.py
```

이 결과는 코드 배선·fail-closed 계약 확인일 뿐 모델 품질 또는 G2 통과 근거가 아니다.

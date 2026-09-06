# negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2b — hash-verified re-derivation (read-only)

> 원본 소스: [negative_semantics_g1_pilot_rtx4080_v1_1](../negative_semantics_g1_pilot_rtx4080_v1_1/)
> (수정하지 않음) · 이전 버전:
> [negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2](../negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2/)
> (역사적 보존, 코드 리뷰로 대체되지 않음) · 해석:
> [2026-09-06 G1 v1.2 amendment 문서](../../docs/experiments/2026-09-06_negative_semantics_g1_v1_2_amendment.md)

## 이 버전이 `_derived_v1_2`와 다른 점

동일한 v1.1 원본에서 **같은 결론**을 다시 계산하지만, seed 붕괴 근거를
detector score 일치만으로 주장하지 않는다.

- **`_derived_v1_2` (이전)**: `declared_deterministic_decoder` 기본값이
  `True`였고, seed 붕괴를 오직 `detection_rows.jsonl`의 점수 일치로만
  판단했다.
- **`_v1_2b` (이 버전)**: `declared_deterministic_decoder` 기본값을 `False`로
  바꾸고, 240개 reconstruction 폴더의 실제 PNG 프레임 15,570개를 전부
  다시 읽어 `sha256` 트리 해시를 계산한 뒤(`reconstruction_frame_hash_manifest`),
  detector-score 기반 판단과 **교차 검증**했다(`cross_validate_seed_evidence`).
  둘이 일치하지 않으면 즉시 실패하도록 설계했다 — 이번에는 일치했다
  (`seed_evidence_method: "score_and_hash_dual_verified"`).
- 이 버전은 또한 실행 전에 `g1_protocol.yaml`·`preparation_manifest.json`·
  `ground_truth_index.json`의 현재 해시를 `run_spec.json`/자체 기록과
  대조해, 동결 protocol이나 데이터셋이 그 사이 바뀌지 않았음을 검증한다
  (`--output-root`가 비어있지 않을 때 signature 불일치 시 덮어쓰기도 거부).

## 결론은 바뀌지 않았다

| 항목 | 값 |
|---|---:|
| `seed_evidence_method` | `score_and_hash_dual_verified` |
| effective_seed_count (few10/full50) | 1 / 1 |
| `gate_reapplication.effective_seed_status` (few10) | **`NOT_PASSED`** |
| `checks_that_flip_to_fail_under_effective_seeds` | `affected_seed_count`, `not_concentrated_in_one_seed` |
| few10 raw_h_add → source_paired_h_add | 0.019793 → 0.010493 |
| full50 raw_h_add → source_paired_h_add | 0.018536 → 0.009081 (`h_add_min=0.01` 미달) |

수치는 `_derived_v1_2`와 완전히 동일하다 — 이 재실행은 **결론을 바꾸지
않고 증거의 종류만 강화**했다(score-only → score+독립 hash 이중 검증).

## `_derived_v1_2`를 왜 그대로 남겨두는가

- `_derived_v1_2`는 `results/registry.csv`에 `frozen`으로 등록된 결과다.
  코드가 더 엄격해졌다고 해서 이미 등록된 frozen 결과를 조용히 덮어쓰지
  않는다.
- `_derived_v1_2`의 `checksums.sha256`은 생성 직후 `README.md`의 오탈자
  (evaluator threshold "0.125" → 실제 값 "0.2")를 고치면서 깨졌던 것을
  이번 검토에서 발견해 다시 생성했다(내용 오탈자 수정 후 checksum을 마지막
  단계로 재계산 — 수치·결론은 전혀 바뀌지 않았다). 자세한 경위는 해당
  디렉터리의 `manifest.json`의 `extra.checksum_incident`에 기록했다.

## 재현

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest
python scripts/derive_negative_semantics_g1_v1_2.py \
  --run-root outputs/negative_semantics_g1_pilot_rtx4080_v1_1 \
  --output-root outputs/negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2b
```

`--declared-deterministic-decoder`를 주지 않으면(기본값) 위 명령이 매번
15,570개 프레임을 다시 해시한다(로컬 SSD 기준 약 40초, GPU 불필요). 같은
`--output-root`로 다시 실행하면 `derivation_signature`가 일치해 그대로
재확인되며, 다른 조건으로 덮어쓰려면 `--allow-overwrite`가 필요하다.

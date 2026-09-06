---
status: frozen
updated: 2026-09-06
gate: G1
protocol_id: negative_semantics_g1_v1_2
run_id: negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2
parent_protocol: negative_semantics_g1_v1_1
supersedes_protocol_version: 1.1.0
amendment_record: this document
heldout_opened: false
---

> [← 문서 색인](../README.md)

# G1 v1.2 amendment — effective-seed 집계와 source-paired additional-object

이 amendment는 [G1 v1.1 정식 Pilot 결과](./2026-09-06_negative_semantics_g1_v1_1_pilot_results.md)가
드러낸 두 가지 한계(seed 중복, source-paired 미분리)를 **GPU 재실행 없이**
정량화한다. `scripts/derive_negative_semantics_g1_v1_2.py`가 동결된 v1.1의
`evaluator/detection_rows.jsonl`과 `evaluator/source/*.json`만 읽어 새 요약을
쓰고, v1.1의 어떤 파일도 수정하지 않는다. 결과는
[results/negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2/](../../results/negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2/)에
보존했다.

## 발생 사실

G1 v1.1은 `fixed_int4` selector(고정 keyframe, 고정 diffusion schedule)를
사용한다. `seed`는 `run_negative_semantics_g1.py`가
`run_transmission_reduction_eval.py --seed`로 전달하지만, 이 reconstruction
경로에는 그 seed가 실제로 소비할 확률적 샘플링 지점이 없다. 그 결과:

- `seed_2025/2026/2027`의 reconstruction PNG가 픽셀 단위로 완전히 동일함을
  `sha256`으로 확인했다 (`frame_00000.png` 등 임의 프레임 표본 검사).
- `evaluator/detection_rows.jsonl`의 OWLv2 점수도 `(video_id, frame_index,
  concept_id)`마다 세 seed에서 완전히 동일하다.

즉 **effective seed count = 1**이며, `additional_event_count`,
`additional_detections`, `affected_seed_count` 등 seed를 단위로 세는 모든
raw 지표가 정확히 3배로 부풀려져 있다(비율 지표인 `h_add`는 분자·분모가
같은 배수로 늘어나 영향을 받지 않는다).

또한 v1.1의 `h_add` 정의(`GT-absent일 때 detector 양성 / GT-absent
opportunity`)는 reconstruction이 "새로 만들어낸" 오탐과, source 프레임이
애초에 이미 detector를 속이던 경우(=reconstruction과 무관하게 발생)를
구분하지 않는다.

## v1.2 변경

`negative_semantics.py`에 순수 함수 4개를 **추가**했다(기존
`summarize_g1`/`collapse_additional_events`는 변경하지 않아 v1.1 재현성을
보존한다):

- `effective_seed_groups(rows)` — 정책별로 seed 간 점수가 완전히 동일한지
  비교해 canonical(유효) seed 집합을 구한다.
- `require_effective_seed_declaration(groups, declared_deterministic_decoder)` —
  seed 붕괴가 감지됐는데 `declared_deterministic_decoder=True`가 명시되지
  않으면 `ValueError`로 **fail-closed**한다. 향후 실제로 확률적이어야 할
  경로에서 seed 배선이 끊기는 회귀를 조용히 통과시키지 않기 위함이다.
- `raw_and_paired_h_add(recon_rows, source_rows, threshold, effective_seeds=None)` —
  `raw_h_add`(v1.1과 동일 정의)와 `source_paired_h_add`(GT-absent **이면서**
  source도 threshold 미만이었던 opportunity만 분모로 사용)를 분리한다.
- `reapply_gate_checks(policy_metrics, gate_cfg)` — 동결된 protocol의 gate
  threshold를 임의의 policy 지표 dict에 재적용한다(원본 로직인
  `run_negative_semantics_g1.py::_write_summary`의 checks 부분과 동일 규칙).
- `summarize_g1_v1_2(...)` — 위 넷을 조합해 raw view, effective-seed view,
  source-paired 분리, gate 재적용 결과를 한 번에 반환한다.

CLI: `scripts/derive_negative_semantics_g1_v1_2.py --run-root <v1.1 run> \
--output-root <새 디렉터리>`. `--output-root`가 `--run-root`와 같거나 그
하위 경로이면 즉시 중단한다(v1.1 원본 보호). 기본 output-root는
`outputs/negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2`.

## 결과

### effective seed count

| policy | declared | effective | canonical seed |
|---|---:|---:|---|
| few10 | 3 | **1** | 2025 |
| full50 | 3 | **1** | 2025 |

### declared-seed(raw) vs effective-seed 지표

| 지표 (few10) | declared-seed (v1.1) | effective-seed (v1.2) | 비율 |
|---|---:|---:|---:|
| additional_detections | 897 | 299 | 정확히 ×3 |
| additional_event_count | 495 | 165 | 정확히 ×3 |
| h_add | 0.019793 | 0.019793 | 불변(비율) |
| affected_seed_count | 3 | 1 | — |
| max_single_seed_event_share | 0.3333 | **1.0** | — |

full50도 동일 패턴(840→280 detections, 441→147 events).

### gate 재적용 (primary policy = few10, threshold는 동결 protocol 그대로: `h_add_min=0.01`, `additional_events_min=20`, `affected_videos_min=5`, `affected_seeds_min=2`, `max_single_video_event_share<=0.50`, `max_single_seed_event_share<=0.75`)

| gate check | declared-seed | effective-seed |
|---|---|---|
| h_add_prevalence | PASS | PASS |
| additional_event_count | PASS | PASS |
| affected_video_count | PASS | PASS |
| **affected_seed_count** | PASS | **FAIL** |
| not_concentrated_in_one_video | PASS | PASS |
| **not_concentrated_in_one_seed** | PASS | **FAIL** |
| **종합** | **PASSED** | **NOT_PASSED** |

같은 threshold를 정직하게(seed 중복 없이) 적용하면 few10 기준 gate는
`NOT_PASSED`로 뒤집힌다. 이는 v1.1이 잘못 계산됐다는 뜻이 아니라 — v1.1은
정의된 대로 정확히 계산했다 — 그 정의가 "3개의 독립 seed"를 가정했는데
실제로는 1개였다는 뜻이다.

### source-paired additional-object (effective-seed 기준)

| policy | absent_opportunities | source_paired_opportunities | unpairable | raw_h_add | source_paired_h_add |
|---|---:|---:|---:|---:|---:|
| few10 | 15,106 | 14,867 | 0 | 0.019793 | **0.010493** |
| full50 | 15,106 | 14,867 | 0 | 0.018536 | **0.009081** |

`unpairable_opportunities=0`은 source-only calibration과 reconstruction
scoring이 정확히 같은 `(video, frame, concept)` 키 공간을 사용함을 확인한
sanity check다. source-paired h_add는 raw 대비 약 47~51% 낮으며, **full50은
`h_add_min=0.01` 미달**이다 — reconstruction이 실제로 "새로 만든" 오탐만
세면 full50 정책은 v1.1이 정의한 prevalence gate를 통과하지 못한다.

### ghost survival (effective-seed 기준)

- unique EXIT 이벤트: 10개 (Pilot 전체, 정책·seed 무관)
- effective-seed uncensored: policy당 4개, right-censored: 12개
- declared-seed 표는 이 값을 그대로 3배(12/36 uncensored/censored)로
  보고했다. 4-표본 기준 ghost survival AUC는 신뢰구간을 계산하지 않는다
  (원 protocol도 정의하지 않음) — 표본이 매우 작다는 점을 명시한다.

## 유지한 것 (변경하지 않음)

- v1.1의 원본 파일 전부 — `g1_summary.json`, `g1_policy_summary.csv`,
  `detection_rows.jsonl`, reconstruction PNG/packet 트리.
- OWLv2 threshold(0.2) — v1.1에서 동결한 값을 그대로 재사용했다.
  policy/bit-depth별로 다시 calibration하면 evaluator가 더 이상
  method-blind가 아니게 된다.
- GPU reconstruction — 결정론적 경로라 다시 돌려도 동일 결과가 나올 것으로
  예상되지만, 검증되지 않은 가정을 추가하지 않기 위해 실행하지 않았다.
  effective seed 1개는 **문서화된 사실**이지 인위적으로 만든 stochasticity가
  아니다.

## 구현·검증 범위

- 신규 순수 함수 4개 + `summarize_g1_v1_2`: `tests/test_negative_semantics_g1_v1_2.py`
  15개 테스트로 단위 검증(seed 붕괴/비붕괴, fail-closed 예외, source-paired
  분리, gate flip 재현, hash 기반 교차검증). 실제 Pilot 데이터(108,990행)로
  CLI를 실행해 위 수치를 재현했다(GPU 불필요).
- 회귀: `tests/test_negative_semantics_g0.py`, `tests/test_negative_semantics_g1.py`
  기존 9개 테스트 변경 없이 통과 — v1.1 `summarize_g1` 동작은 그대로다.

### 2026-09-06 코드 리뷰 보강 (score-only 증거 → score+hash 이중 검증)

최초 버전은 seed 붕괴를 `detection_rows.jsonl`의 점수 일치만으로 판단했다.
detector score 일치가 픽셀 동일성의 증명은 아니라는 지적에 따라 다음을
추가했다(원본 v1.1 파일은 여전히 수정하지 않는다):

- `effective_seed_groups_from_hashes`, `cross_validate_seed_evidence`
  (`negative_semantics.py`) — 실제 reconstruction PNG를
  `sgdjscc_lab.utils.frame_hash.frame_tree_sha256`로 직접 해시해 독립
  증거를 만들고, score 기반 판정과 불일치하면 즉시 실패한다.
- `derive_negative_semantics_g1_v1_2.py`의 `--declared-deterministic-decoder`
  기본값을 `True`에서 **`False`로 변경**했다. 기본 경로는 이제 v1.1의
  6개 child run 전체(정책×seed) 아래 240개 영상의 reconstruction PNG
  **15,570개를 전부 다시 읽어 해시**하고 score 기반 근거와 교차검증한 뒤에만
  진행한다(로컬 SSD 기준 GPU 없이 약 40초). 명시적 `--declared-deterministic-decoder`
  플래그로 이 검증을 건너뛸 수 있으나 그 경우 `evidence_method: "operator_declared"`로
  구분해 기록한다.
- `run_spec.json`의 `protocol_sha256`/`preparation_manifest_sha256`과
  현재 파일 해시를 실행 전에 대조하고, `evaluator_freeze.json`의 자체
  `protocol_sha256`과도 교차검증한다 — 동결 protocol이나 데이터셋이 그
  사이 바뀌면 즉시 중단한다.
- 출력 디렉터리가 비어 있지 않으면 `derivation_signature`(입력·근거 방법
  전부를 포함한 해시)가 일치할 때만 재실행을 허용하고, 다르면
  `--allow-overwrite` 없이는 거부한다. CSV도 atomic write + LF 강제로
  바꿨다(`csv.writer`의 기본 `\r\n`이 git 추적 CSV를 CRLF로 오염시키는
  문제가 실제로 발생해 `results/registry.csv`도 함께 고쳤다).
- 결론은 **바뀌지 않았다**: `seed_evidence_method: "score_and_hash_dual_verified"`로
  기존 수치(few10/full50 `effective_seed_count=1`, gate
  `NOT_PASSED`)를 그대로 재확인했다. 이 하드닝을 반영한 재실행은
  frozen `results/negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2`를
  덮어쓰지 않고 별도 버전
  [`results/…_derived_v1_2b`](../../results/negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2b/README.md)로
  보존했다.
- 회귀: `tests/test_derive_negative_semantics_g1_v1_2.py` 13개(합성
  fixture로 protocol/dataset/ground-truth hash 불일치, evidence 불일치,
  output-root resume 충돌, CSV LF, 실제 v1.1 데이터 이중검증까지 포함).
- 전체 non-GPU suite: `python -m pytest tests/` — **1613 passed**, 0 failed
  (G1 v1.1/v1.2/int6 bridge 작업 전체 기준 1541에서 신규 72개 증가;
  hash 이중검증 보강분은 `test_derive_negative_semantics_g1_v1_2.py` 13개).

## v1_2b의 provenance 한계 (2026-09-06 code review에서 확인)

[`results/negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2b`](../../results/negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2b/)는
checksum이 고정된 frozen 결과이지만, **그 결과를 만든 derivation 코드
자체가 당시 git commit에 포함되지 않은 code-dirty 상태에서 생성됐다.**
`manifest.json`은 `git.dirty: true`만 기록했을 뿐, 정확히 무엇이
dirty였는지는 명시하지 않는다 — 실제로는 `scripts/derive_negative_semantics_g1_v1_2.py`,
`src/sgdjscc_lab/evaluators/int6_bridge.py`,
`src/sgdjscc_lab/utils/frame_hash.py` 자체가 **당시 전부 untracked**였고
`src/sgdjscc_lab/evaluators/negative_semantics.py`는 tracked이지만
uncommitted 상태였다. 즉 `manifest.json`에 적힌 `git_commit`만으로는 이
결과를 만든 코드를 실제로 복원할 수 없다 — 이는 이번 int6 bridge
provenance 강화 작업(critical-file tracked-at-HEAD 검증, 이 문서 상단의
LPIPS/formal provenance 절 참고)이 정확히 막으려는 상황이 v1_2b 자체에
이미 발생해 있었다는 뜻이다.

이 사실은 v1_2b의 **결론을 무효화하지 않는다** — 같은 코드가 나중에
정식으로 커밋된 뒤 처음부터 다시 파생해도 동일한 입력(`source_run_root`,
`source_detection_rows_sha256` 등은 이미 checksum으로 고정됨)에서 같은
계산을 재현할 것으로 예상된다. 다만 "이 결과를 만든 정확한 코드가 git
이력에 존재한다"는 재현성 보장은 v1_2b에 대해서는 성립하지 않는다.

**지금 이 문서를 작성하는 시점에는 새 결과를 생성하지 않는다.** 관련
코드가 clean commit으로 확정된 뒤, 별도 경로
(`outputs/negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2c`)에
`--allow-overwrite` 없이 새로 파생하고 `results/`에 별도 버전으로 보존하는
것을 다음 작업으로 남긴다. `v1_2`/`v1_2b`는 그대로 frozen 상태로 둔다.

## 다음 단계

- `fixed_int6 + both-omit` bridge validation: [준비 기록](./2026-09-06_negative_semantics_int6_bridge_preparation.md).
  이 bridge도 동일한 effective-seed 원칙(단일 seed로 설계, 3배 부풀리지
  않음)을 처음부터 적용하고, formal 실행 전 critical file이 모두
  git-tracked·HEAD와 일치하는지 검증한다(v1_2b가 놓쳤던 바로 그 검증).
- 코드가 commit된 뒤 `v1_2c` 재파생을 수행해 v1_2b의 provenance 한계를
  해소한다.
- G2(Oracle ABSENT 제어) 진입 전, 이 문서의 `NOT_PASSED`(effective-seed)와
  `source_paired_h_add < h_add_min`(full50) 결과를 반영해 논문 claim
  경계를 재검토한다.

## 재현

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest
python scripts/derive_negative_semantics_g1_v1_2.py \
  --run-root outputs/negative_semantics_g1_pilot_rtx4080_v1_1 \
  --output-root outputs/negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2
python -m pytest tests/test_negative_semantics_g1_v1_2.py \
  tests/test_negative_semantics_g0.py tests/test_negative_semantics_g1.py -v
```

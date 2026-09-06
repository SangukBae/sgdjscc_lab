---
status: ready_for_formal_run
updated: 2026-09-06
gate: (none — comparison protocol, not a phenomenon gate; see G1 for the gate)
protocol_id: negative_semantics_int6_bridge_v1_0
heldout_opened: false
---

> [← 문서 색인](../README.md)

# fixed_int4 vs fixed_int6 bridge — 실행 준비 (GPU 실행은 아직 하지 않음)

2026-09-04부터 ETRI 신규 기본 양자화 운용점은 `fixed_int6`다
([결정 기록](./2026-09-04_int6_etri_operating_point_decision.md)). G1은
`fixed_int4` stress 조건으로 동결·완료했으므로([v1.1 결과](./2026-09-06_negative_semantics_g1_v1_1_pilot_results.md),
[v1.2 amendment](./2026-09-06_negative_semantics_g1_v1_2_amendment.md)),
`int6 + both-omit`이 같은 조건에서 exact byte·품질·additional-object·ghost·
latency 관점에서 `int4`와 어떻게 다른지 아직 실측되지 않았다. 이 문서는 그
실측을 위한 코드·설정·절차를 준비한 기록이다. **이 문서를 작성한 세션은
GPU 실험도 GPU smoke도 실행하지 않았다** — 아래 명령은 사용자가 직접
실행한다.

## 무엇을 새로 만들었나

| 파일 | 역할 |
|---|---|
| `configs/experiments/negative_semantics/g1_int6_bridge_protocol.yaml` | 동결 protocol. bit_depth만 `{fixed_int4, fixed_int6}`로 바뀌고 guide/selector/시공간 변환/evaluator는 G1과 완전히 동일 |
| `src/sgdjscc_lab/evaluators/int6_bridge.py` | 순수 함수: 조건별 exact-byte/PSNR/SSIM/LPIPS/latency 집계, video-clustered bootstrap CI, reference 대비 delta, 품질 gate 판정 |
| `scripts/run_negative_semantics_int6_bridge.py` | 실행기: int4는 재사용, int6만 새로 reconstruction+채점. `run()`을 호출하기 전까지 torch/CUDA를 import하지 않는다 |
| `tests/test_int6_bridge.py` | 순수 집계 로직 단위 테스트 (6개) |
| `tests/test_negative_semantics_int6_bridge.py` | 실행기 로직 테스트: 실제 동결된 v1.1 run에 대해 재사용 검증·행 필터링을 GPU 없이 검증 (5개) |

## 핵심 설계: 무엇을 재사용하고 무엇을 새로 만드는가

| 구성요소 | fixed_int4 | fixed_int6 |
|---|---|---|
| reconstruction 프레임 | **재사용** — `outputs/negative_semantics_g1_pilot_rtx4080_v1_1`에서 읽음, 재실행 안 함 | 새로 생성 — `run_transmission_reduction_eval.py --configs fixed_int6` |
| accounting (byte/PSNR/SSIM/LPIPS/latency) | **재사용** — v1.1의 `per_video_metrics.csv` | 새로 생성 |
| detector 점수(OWLv2) | **재사용** — v1.1의 `detection_rows.jsonl`을 seed 2025로 필터링, 재채점 안 함 | 새로 채점 — 단, threshold는 v1.1과 동일하게 재사용(재calibration 안 함) |
| source 프레임 OWLv2 점수 | **재사용** — v1.1의 `evaluator/source/*.json` (bit-depth 무관) | 동일 |
| caption | **재사용** — `data/negative_semantics/g1/pilot/captions` (bit-depth 무관, 공유 위치) | 동일 |
| seed | **1개(2025)만** — G1 v1.2 amendment가 확인한 effective_seed_count=1을 반영해 처음부터 3배로 부풀리지 않음 | 동일 |

`scripts/run_negative_semantics_int6_bridge.py::verify_int4_reuse()`가 재사용
전에 다음을 강제한다: 재사용 소스의 `protocol_sha256`이 동결된 값과 일치,
smoke가 아님, 요청한 seed·video_id가 소스에 실제로 존재, 6개 child run
전부 `run_status=completed`이고 `n_failed_pairs=0`, evaluator calibration이
`PASSED`. 하나라도 어긋나면 `SystemExit`로 즉시 중단한다 — 잘못된 재사용을
조용히 넘어가지 않는다.

## preflight / smoke / formal 구분

G1과 동일한 3단계를 따른다.

1. **`--preflight-only`**: checkpoint 존재·해시, CUDA 가용성, git 상태,
   OWLv2 evaluator snapshot 해석까지만 확인하고 종료한다. reconstruction은
   전혀 실행하지 않으며 `--run-root`를 생성하거나 수정하지도 않는다. 결과는
   표준 출력으로만 확인한다.
2. **`--smoke`**: `g1_int6_bridge_protocol.yaml`의 `gate.smoke_video_id`
   (G1과 동일한 `ovis_valid_1b664206`, v1.1 OOM 회귀 대상 영상) 1개, 2 frame만
   `fixed_int6`로 새로 reconstruct+채점한다. **논문 근거가 아니다** — 구조적
   경로(재사용 검증 → int6 reconstruction → 채점 → 요약)가 끝까지
   동작하는지만 확인한다.
3. **정식(formal)**: `--smoke` 없이, 40영상 전체 × `{few10, full50}` ×
   seed 2025만 `fixed_int6`로 reconstruct+채점하고, int4는 전부 재사용한다.
   git tracked checkout이 clean해야 한다(`_preflight`가 dirty면 중단).

## 실행 명령 (사용자가 직접 실행)

### 1. preflight만

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest
python scripts/run_negative_semantics_int6_bridge.py \
  --run-root outputs/negative_semantics_int6_bridge_rtx4080 \
  --device cuda:0 --preflight-only
```

### 2. GPU smoke (구조 검증, 논문 근거 아님)

```bash
tmux new -s int6_bridge_smoke
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest
TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_negative_semantics_int6_bridge.py \
  --run-root outputs/negative_semantics_int6_bridge_smoke_rtx4080 \
  --device cuda:0 --smoke \
  2>&1 | tee -a outputs/negative_semantics_int6_bridge_smoke_rtx4080/operator.log
# Ctrl-b d로 detach, `tmux attach -t int6_bridge_smoke`로 재접속
```

smoke 성공 판독: `int6_bridge_summary.json`이 생성되고
`evidence_scope == "NOT_EVIDENCE_SMOKE"`, `reconstruction/fixed_int6/few10/seed_2025/summary.json`의
`run_status == "completed"`이고 `n_failed_pairs == 0`이면 구조적으로
통과한 것이다. 비대칭 accounting의 absolute/delta/quality gate는 모두
`OMITTED_ASYMMETRIC_SMOKE`, 2-frame으로 계산할 수 없는 16-frame ghost는
`OMITTED_INSUFFICIENT_HORIZON_SMOKE`여야 한다.

### 3. 정식 실행 (40영상 × 2 policy × 1 seed, fixed_int6만 새로 계산)

```bash
tmux new -s int6_bridge_formal
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest
git status --short   # tracked 변경 없이 clean한지 먼저 확인
set -o pipefail
TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_negative_semantics_int6_bridge.py \
  --run-root outputs/negative_semantics_int6_bridge_rtx4080 \
  --device cuda:0 \
  2>&1 | tee -a outputs/negative_semantics_int6_bridge_rtx4080/operator.log
```

다른 GPU 작업과 동시에 실행하지 않는다. `few10 + fixed_int6` 40영상은
few10 diffusion step이 `fixed_int4`와 같으므로 대략 G1 few10 1-seed
소요시간(약 168초, 136 frame 기준 실측 비례 추정)과 비슷한 자릿수를
예상하되, `full50`은 그보다 5배 가까운 step이라 훨씬 오래 걸린다 — 정확한
값은 이 문서를 처음 작성하는 시점에 실측하지 않았으므로 실행 중
`operator.log`와 `nvidia-smi`로 직접 관찰한다.

```bash
tail -f outputs/negative_semantics_int6_bridge_rtx4080/operator.log
nvidia-smi
```

### 재개(resume)

`run_transmission_reduction_eval.py`가 이미 완료된 (video, config) pair를
signature 기반으로 건너뛰므로, 같은 `--run-root`로 동일 명령을 다시
실행하면 완료된 부분은 재사용하고 남은 부분만 이어서 계산한다. `--run-root`
자체를 바꾸지 않는 한 재실행은 항상 안전하다(별도 `--resume` 플래그 불필요).

## 판독 절차

1. `outputs/negative_semantics_int6_bridge_rtx4080/run_spec.json` — mode,
   protocol/dataset hash, video_ids, int4 재사용 소스의 detection_rows·
   per_video_metrics hash, checkpoint hash, evaluator revision, git commit,
   device, seed. 같은 `--run-root`로 재실행했을 때 이 파일과 하나라도
   다르면 즉시 중단된다(`check_run_spec_resume`).
2. `outputs/negative_semantics_int6_bridge_rtx4080/int4_reuse_verification.json` —
   재사용 검증이 통과했는지, 어떤 seed·policy를 재사용했는지 확인.
3. `outputs/negative_semantics_int6_bridge_rtx4080/reconstruction/fixed_int6/{few10,full50}/seed_2025/summary.json` —
   `run_status == "completed"`, `n_failed_pairs == 0`, `n_videos == 40`.
4. `outputs/negative_semantics_int6_bridge_rtx4080/int6_bridge_summary.json`:
   - `fixed_int6_vs_reference_accounting_delta.<policy>.<field>` —
     video ID로 inner-join한 paired delta(`mean_delta`, `per_video_delta`,
     `bootstrap_95ci`)를 `total_bundle_bytes`/`total_bundle_bytes_per_frame`/
     `mean_psnr`/`mean_ssim`/`mean_lpips`(양쪽 조건 모두 사후 계산)/
     `total_elapsed_s`마다 제공한다.
   - `fixed_int6_quality_gate.<policy>.passed` — 저장소 표준 품질 gate
     (`psnr_drop_db_max=0.5`, `ssim_drop_max=0.01`, `lpips_rise_max=0.02`,
     `g1_int6_bridge_protocol.yaml`의 `metrics.quality_gate`)를 위 paired
     delta에 적용한 결과.
   - `fixed_int6_vs_reference_h_add_delta.<policy>.{raw_h_add,source_paired_h_add}` —
     raw/source-paired H_add도 video 단위로 짝지은 paired delta.
   - `ghost_survival_delta.<policy>` — 두 조건 모두 uncensored인 EXIT
     이벤트만 짝지은 paired AUC delta(`n_jointly_uncensored_events`,
     `excluded_event_keys`로 한쪽만 censored인 이벤트를 명시).
5. 이 요약을 `docs/experiments/`에 날짜 기반 결과 문서로 추가하고
   `results/`에 보존한다(이 반복에서 사용한 것과 동일한 절차 —
   [v1.1 보존 예시](./2026-09-06_negative_semantics_g1_v1_1_pilot_results.md) 참고).
   effective seed 1개이므로 v1.2와 같은 별도 dedup 단계는 필요 없다(처음부터
   1개로 설계했기 때문).

## 구현·검증 범위 (이 세션에서 실제로 한 것)

- `src/sgdjscc_lab/evaluators/int6_bridge.py`의 순수 함수 4개:
  `tests/test_int6_bridge.py` 6개 테스트로 검증(합성 fixture, GPU 불필요).
- `scripts/run_negative_semantics_int6_bridge.py`의 `build_summary()`(순수
  집계): 합성 조건으로 delta·품질 gate 계산 검증.
- `verify_int4_reuse()`, `load_int4_rows()`, `load_int4_accounting()`: **실제
  동결된 `outputs/negative_semantics_g1_pilot_rtx4080_v1_1`에 대해 GPU 없이
  실행**해 검증했다. 이 과정에서 실제 버그 2개를 발견해 수정했다:
  1. evaluator threshold를 문서 초안에 0.125로 잘못 기재했으나 실제 동결값은
     `0.2`(recall 0.8349, FPR 0.0158)였다 — 코드는 항상 `evaluator_freeze.json`에서
     동적으로 읽으므로 정량 결과에는 영향 없음(문서 오타만 수정).
  2. `per_video_metrics.csv`의 `config` 컬럼이 `fixed_int4`가 아니라
     `fixed_int4__candidate_both_omit`(base_config와 guide_profile 결합)로
     기록된다는 사실을 실제 CSV로 확인하고 `load_int4_accounting`/
     `load_int6_accounting`의 필터 조건을 수정했다 — 수정 전에는 int4
     accounting 재사용이 항상 빈 리스트를 반환하는 조용한 버그였다.
- **실행하지 않은 것**: `--preflight-only`, `--smoke`, 정식 실행 — 전부
  CUDA를 요구하므로 이 세션에서는 실행하지 않았다. 위 명령을 사용자가
  직접 실행한다.

### 2026-09-06 코드 리뷰 보강

초안은 `--no-lpips`와 v1.1 재사용 데이터의 상호작용, video-set 검증,
paired 통계, run 서명을 충분히 다루지 않았다. 다음을 수정했다.

1. **LPIPS**: v1.1과 int6 reconstruction 둘 다 `--no-lpips`로 실행되어
   `per_video_metrics.csv`의 `mean_lpips`가 빈 문자열이었다(실제 v1.1 CSV로
   확인). `compute_condition_mean_lpips()`가 두 조건에 **동일한
   함수·evaluator·device**로 LPIPS를 사후 계산해 채운다
   (`fill_condition_lpips()`). `evaluator_factory`/`load_tensor`를
   주입 가능하게 해 GPU/torch 없이 오케스트레이션(짝짓기, 평균, frame-count
   불일치 실패)을 테스트했다.
2. **smoke 범위**: `load_int4_rows`/`load_int4_accounting`/`load_source_rows`가
   이제 `video_ids`(및 detection row는 `max_frames`)를 받는다. smoke는
   fixed_int4·fixed_int6 모두 같은 1개 영상(및 detection row는 같은 2
   frame)만 비교한다. accounting(`per_video_metrics.csv`)은 frame 단위가
   아니라 영상 전체 집계라 frame 단위 절단은 불가능하다는 한계를 코드
   주석과 함수 docstring에 명시했다.
3. **paired 통계**: `int6_bridge.py`에 `paired_delta_from_video_values`/
   `paired_condition_metrics`를 추가해 모든 accounting 지표를 video ID
   inner-join 후 paired delta + paired bootstrap 95% CI로 계산하도록
   바꿨다(기존 `delta_vs_reference`는 평균끼리의 차이였고 paired CI가
   없어 제거했다). `negative_semantics.py`에 `h_add_by_video`를 추가해
   raw/source-paired H_add도 동일하게 video 단위로 짝지었다. ghost-track은
   `build_ghost_delta()`가 두 조건 모두 uncensored인 EXIT 이벤트만
   짝짓고(`n_jointly_uncensored_events`), 한쪽만 censored인 이벤트는
   `excluded_event_keys`에 명시한다("가능한 ghost delta").
4. **fail-closed 검증**: `build_summary()`가 조건/영상 수 불일치, 중복
   영상, blank/NaN/Inf 값, detector concept_id 집합 불일치를 요약 작성
   전에 실패시킨다(`coerce_finite_float`, `validate_exact_video_count`).
   quality gate가 선언한 정책을 모두 포함하지 않으면 완료 처리하지 않는다.
5. **run 서명과 evaluator cache**: `run()`이 이제 mode(smoke/formal),
   protocol/dataset hash, video IDs, int4 source의 detection_rows·
   per_video_metrics hash, checkpoint hash, evaluator revision, git
   commit, device, seed를 `run_spec.json`에 기록하고
   (`check_run_spec_resume`) 재실행 시 하나라도 다르면 중단한다. OWLv2
   cache metadata에는 `reconstruction_frame_tree_sha256`
   (`sgdjscc_lab.utils.frame_hash.frame_tree_sha256`)을 포함해, 프레임
   개수는 같지만 내용이 달라진 재실행에서 캐시가 조용히 재사용되지
   않도록 했다.
6. **감사 도구 분리**: `audit_negative_semantics_g1.py`가
   `--require-runner-complete`(구조적 완료)와
   `--require-scientific-gate`(v1.2 effective-seed 재적용 결과, 파생
   안 됐으면 `UNKNOWN_V1_2_NOT_DERIVED`로 fail-closed)를 분리했다.
   기존 `--require-pass`는 `--require-runner-complete`의 **deprecated
   alias**로 유지하고 stderr에 경고를 낸다.

회귀: `tests/test_int6_bridge.py`(24), `tests/test_negative_semantics_int6_bridge.py`(19),
`tests/test_audit_negative_semantics_g1.py`(5), `tests/test_frame_hash.py`(5) — blank
LPIPS/실제 v1.1 CSV, smoke 1:1 범위, video 수 불일치/중복/누락,
paired delta·CI, NaN/Inf·조건 누락 거부, frame hash 변경 시 cache 무효화,
run_spec 재개 충돌 거부를 모두 포함한다.

전체 non-GPU test suite: `python -m pytest tests/` — **1613 passed**, 0 failed.

### 2026-09-06 (2회차) 코드 리뷰 보강

1차 보강(같은 날짜, 위 절들) 이후 추가 리뷰에서 발견된 문제를 수정했다.

1. **LPIPS 원본 비교 오류** — 이전 구현은 raw source(예: 실제 smoke 영상
   1280×630)를 reconstruction(512×256, resize+center-pad 후)과 직접
   비교했다. shape가 달라 비교 자체가 성립하지 않는 잠재적 버그였다.
   `run_transmission_reduction_eval.py::_load_frames`(resize
   `aspect_preserving_bilinear_antialias_no_upscale` + pad
   `symmetric_constant_zero_extra_pixel_bottom_right`)를 공용 함수로
   재사용해 source에도 동일 canonical 변환을 적용하도록
   `build_canonical_source_frames()`를 추가했다. shape·frame 수가 정확히
   같지 않으면 `compute_condition_mean_lpips()`가 즉시 실패한다. 실제
   1280×630→512×256 사례를 재현하는 GPU-free 회귀 테스트를 추가했다
   (`test_canonical_transform_reproduces_the_real_1280x630_to_512x256_smoke_case`).
2. **LPIPS evaluator 공유·strict 검증** — 조건마다 새 `QualityEvaluator`를
   만들던 것을 실행당 1개로 바꿔 int4/int6/두 policy 전체가 같은 backbone
   인스턴스를 쓰도록 했다. LPIPS 값이 `None`/NaN/Inf이면 예외를 삼키지
   않고 즉시 실패한다. `net`, package version(`importlib.metadata`),
   preprocessing 설명, deterministic state-dict sha256을
   `lpips_evaluator_provenance()`로 계산해 preflight에서 protocol의
   `metrics.lpips.net`과 대조하고 `run_spec.json`/완료 manifest에 기록한다
   (다르면 재개 시 `check_run_spec_resume`이 그대로 감지한다). int4/int6/
   source 각각의 LPIPS 입력 frame-tree hash도 `lpips_input_frame_hashes()`로
   완료 manifest에 남긴다.
3. **formal provenance/resume** — `git diff --quiet` 방식의 전역 dirty
   검사는 **untracked 파일을 놓친다**(새 스크립트가 `git add`되지 않아도
   "clean"으로 보고됨). `verify_critical_files_tracked_at_head()`가 이
   스크립트·의존 모듈·frozen config 목록을 하나씩 git-tracked·무변경·
   HEAD와 byte-identical인지 검사하고, 무관한 untracked 파일은 검사 대상이
   아니므로 차단하지 않는다. `run_spec.json` 호환성 확인 전에는 run-root에
   아무것도 쓰지 않도록 `gather_provenance()`(run_root 인자 자체가 없음)로
   분리했고, `--preflight-only`는 결과를 표준 출력으로만 보여 주며
   `--run-root`를 생성하거나 수정하지 않는다.
   비호환 재개 시도가 기존 `run_spec.json`/`preflight.json`/
   `int4_reuse_verification.json`을 건드리지 않는 것과, preflight-only가
   기존 formal `preflight.json`을 보존하는 것을 실제 v1.1 데이터로
   검증하는 통합 테스트를 추가했다.
4. **smoke·formal 검증 범위** — smoke에서 fixed_int4(영상 전체)와
   fixed_int6(2 frame)의 accounting/quality gate를 비교하던 것을 제거하고
   absolute/delta/quality gate 전부를 `OMITTED_ASYMMETRIC_SMOKE`로
   명시했다(detection 기반 H_add 지표는 두 조건
   모두 같은 frame budget이라 계속 실제로 계산한다). formal에서는 조건별
   예상 밖 condition 거부, 요청 video ID와 GT video ID 집합의 정확한 일치,
   source/int4/int6의 전체 (video,frame,concept) GT-grid 동일성, official
   GT에서 재계산한 `gt_present` 일치를 `build_summary()`가 검증한다.
   `h_add_by_video`는 source key 중복·누락 pair를 fail-closed
   하도록 바꿨다(`negative_semantics.py`). pooled absolute H_add(분자/분모
   포함)와 video-paired delta를 `h_add_estimands.pooled_absolute`/
   `paired_video_delta`로 분리해 서로 다른 estimand임을 명시했다.
5. **ghost/report** — ghost paired delta의 bootstrap을 event 단위가 아니라
   **video 단위**로 클러스터링하도록
   (`paired_delta_video_clustered`/`video_clustered_bootstrap_mean`)
   바꿨다. 한 영상에 EXIT event가 2개인 회귀 테스트를 추가했다. jointly
   uncensored/reference-only censored/candidate-only censored/jointly
   censored 4가지를 event·video 표본 수와 함께 기록한다
   (`censoring_breakdown`). 2-frame smoke의 ghost는 horizon 16을 충족하지
   못하므로 계산하지 않고 `OMITTED_INSUFFICIENT_HORIZON_SMOKE`로 기록한다.
   완료 시 `status.json`(RUNNING→PASSED/FAILED)과, logs/status/operator.log를
   제외한 reconstruction PNG/MP4·child CSV/summary/manifest·evaluator cache를
   재귀적으로 포함하는 `artifact_checksums.json`을 atomic하게 쓴다.

회귀 테스트: `tests/test_negative_semantics_int6_bridge.py`(45),
`tests/test_int6_bridge.py`(35, video-clustered bootstrap·validate_* 함수
포함), `tests/test_negative_semantics_g1_v1_2.py`(17, `h_add_by_video`
fail-closed 포함). 관련 97개와 전체 suite를 다시 실행해
`1652 passed, 0 failed`를 확인했다(기존 Torch nested-tensor warning 3개).

## 남은 한계

- `run_transmission_reduction_eval.py`가 `fixed_int6`에 대해 실제로 어느
  정도의 처리 시간이 걸리는지 이 문서 작성 시점에는 실측하지 않았다(추정만
  가능).
- bridge는 `candidate_both_omit` guide 1개만 비교한다 — G1이 이미 그 guide로
  동결했기 때문이며, guide 자체의 int4/int6 상호작용은 범위 밖이다.
- effective seed 1개로 설계했으므로, 만약 향후 디코더가 실제로 확률적이
  된다면(예: 실제 stochastic sampler 도입) 이 protocol은 재동결해야 한다.
- smoke의 accounting/quality gate는 근본적으로 비대칭(fixed_int4 영상
  전체 vs fixed_int6 2 frame)이라 `OMITTED_ASYMMETRIC_SMOKE`로 명시하고
  계산하지 않는다 — smoke는 구조 검증 전용이며 `evidence_scope:
  NOT_EVIDENCE_SMOKE`로 항상 표시된다.
- LPIPS 사후 계산·preflight의 LPIPS provenance 계산은 실제 `lpips` VGG
  가중치 로드가 필요해 이 세션에서 GPU/네트워크로 실행하지 않았다 —
  정식 실행 시 사용자가 처음 실행할 때 한 번 받는다.
- 이 시점(2026-09-06)에 `run_negative_semantics_int6_bridge.py`를 포함한
  모든 관련 코드는 여전히 **untracked/uncommitted 상태**다. 즉
  `verify_critical_files_tracked_at_head()`를 지금 formal 모드로 실행하면
  스스로도 실패한다 — 이는 버그가 아니라 "커밋 전에는 formal을 허용하지
  않는다"는 설계가 정확히 의도대로 작동하는 것이다. 사용자가 커밋한 뒤에만
  formal 실행이 preflight를 통과한다.

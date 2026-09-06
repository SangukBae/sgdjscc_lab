---
status: active
updated: 2026-09-06
owner: ETRI SGD-JSCC 연구팀
source_commit: aae9e26
---

> [← 문서 색인](../README.md)

# 다음 채팅용 연구개발 인계 요약

## 2026-09-06 G1 v1.1 완료·감사, v1.2 amendment, int6 bridge 실행 준비

- **G1 v1.1 정식 실행이 완료됐다**: OVIS Pilot 40영상 × `{few10,full50}` ×
  3 declared seed, child run 6개 × 40영상 = **240/240**, 실패 0, held-out
  미접근. 신규 `scripts/audit_negative_semantics_g1.py --require-pass`(exit 0)로
  독립 재검증했다. `g1_summary.json.gate_status: PASSED`(declared-seed 정의).
  보존: [results/negative_semantics_g1_pilot_rtx4080_v1_1](../../results/negative_semantics_g1_pilot_rtx4080_v1_1/),
  해석: [실험 문서](../experiments/2026-09-06_negative_semantics_g1_v1_1_pilot_results.md).
- **v1.2 amendment로 세 가지 한계를 GPU 재실행 없이 정량화했다**
  (`scripts/derive_negative_semantics_g1_v1_2.py`, 기존
  `detection_rows.jsonl`만 읽음):
  1. 이 reconstruction 경로(고정 selector + 고정 diffusion schedule)는
     `seed`를 실제로 소비하지 않는다 — 세 seed의 reconstruction PNG/OWLv2
     점수가 완전히 동일하다. **`effective_seed_count=1`**(few10/full50
     모두). 같은 gate threshold를 effective-seed 지표에 그대로 재적용하면
     `affected_seed_count`와 `not_concentrated_in_one_seed`가 FAIL로
     뒤집혀 few10 기준 gate가 **`NOT_PASSED`**가 된다.
  2. source-paired additional-object(원본 프레임이 이미 오탐이던 경우 제외)로
     다시 계산하면 h_add가 raw 대비 약 47~51% 낮아진다(few10 0.019793→
     0.010493, full50 0.018536→0.009081). **full50은 `h_add_min=0.01`
     미달**이다.
  3. ghost survival uncensored 표본은 effective-seed 기준 policy당 4개뿐
     (unique EXIT 이벤트 10개 중 12개 right-censored)이다.
  v1.1의 원본 파일은 전혀 수정하지 않았다. 보존:
  [results/…_derived_v1_2](../../results/negative_semantics_g1_pilot_rtx4080_v1_1_derived_v1_2/),
  해석: [v1.2 amendment 문서](../experiments/2026-09-06_negative_semantics_g1_v1_2_amendment.md).
- **`fixed_int4 vs fixed_int6` paired bridge를 실행 준비까지 완료했다** —
  코드(`scripts/run_negative_semantics_int6_bridge.py`,
  `src/sgdjscc_lab/evaluators/int6_bridge.py`), 동결 protocol
  (`configs/experiments/negative_semantics/g1_int6_bridge_protocol.yaml`),
  테스트 11개(합성 fixture + 실제 v1.1 run에 대한 GPU-free 검증)를 완료했다.
  **GPU 실험·GPU smoke는 이 세션에서 실행하지 않았다** — 정확한 tmux
  명령과 preflight/smoke/formal 구분, 판독 절차는
  [bridge 준비 기록](../experiments/2026-09-06_negative_semantics_int6_bridge_preparation.md)에
  있다. 사용자가 직접 실행해야 한다.
- 회귀 없음: `python -m pytest tests/` — **1552 passed** (기존 1541 + 신규
  11).
- 다음 채팅에서 int6 bridge formal 결과가 나오면: (1) `int6_bridge_summary.json`을
  판독하고 (2) `docs/experiments/`에 날짜 문서·`results/`에 보존 사본을
  추가하고 (3) 이 문서·`status.md`·`roadmap.md`를 갱신하고 (4) G2 Oracle
  ABSENT 진입 여부를 v1.2/int6 bridge 결과를 반영해 재검토한다.

## 2026-09-04 ETRI 양자화 운용점 변경

- 신규 ETRI 개발·시연·최종 후보의 기본 bit-depth는 `fixed_int6`다.
- 기존 `fixed_int4` 선택은 최소-byte rate-first 결정이었으며 완료 실험과 G1 v1.1
  stress configuration은 역사적 비교 가능성을 위해 그대로 보존한다.
- `int6 + both-omit`의 exact byte와 hallucination 결과는 아직 없으므로 G1 뒤 G2 전에
  동일 공개 데이터·seed·diffusion step의 paired bridge validation을 수행한다.
- 근거와 적용 범위:
  [int6 ETRI 운용점 결정](../experiments/2026-09-04_int6_etri_operating_point_decision.md)

## 2026-09-03 최우선 변경

- 활성 논문 연구선은
  [negative semantic paper plan](./negative_semantic_paper_plan.md)이다. 이전
  `paper_writing_notes.md`는 superseded historical artifact다.
- G0 v1.2에서 OVIS official-GT Pilot 40개와 기존 YouTube-VOS/DAVIS 5-way split,
  archive·영상별 hash, 승인, DAVIS Held-out 재봉인을 완료했다.
- G0 자동 감사는 13/13, `--require-pass` exit 0으로 **`PASSED`**다.
- 사람 A/B/C 검수는 G0 조건에서 제거됐고, claim은 official-GT-anchored automatic
  additional-object/ghost-track으로 제한한다.
- 다음 행동은 G1에서 selector와 독립인 evaluator threshold를 Pilot에 고정하고 Oracle
  ABSENT GPU smoke 후 정식 paired run을 실행하는 것이다.
- 통합 요약기는 이후 평균/CI candidate를 분리하고 baseline을 후보에서 제외한다.
  Development role의 `validation_passed`는 `null`이며, Validation manifest가 명시된
  실행에서만 CI 기반 bool이다. 2026-08-29 결과 파일은 역사적 무결성을 위해 변경하지 않았다.
- 근거: [G0 v1.2 기록](../experiments/2026-09-03_negative_semantics_g0_official_gt_amendment_v1_2.md)

## 프로젝트 기준

- 저장소: `/home/sangukbae/ETRI/Semantic/sgdjscc_lab`
- 브랜치: `main`, 원격: `origin/main`
- 원본 `SGDJSCC/`는 읽기 전용 baseline이며 확장은 `sgdjscc_lab/`에서 수행한다.
- GPU 장시간 실험은 원격 Docker에서 3 GPU 병렬로 실행하되, 컨테이너 ID는 매번
  `docker ps`로 다시 확인한다.
- 로컬의 PPTX·ZIP·`scripts/make_english_pptx.py` 등 무관한 untracked 사용자 파일은
  수정·삭제·커밋하지 않는다.
- tracked 파일 변경 시 관련 검증 후 커밋하고 `origin/main`에 푸시한다.

## 문서와 결과 파일의 역할

| 파일·디렉터리 | 역할 |
|---|---|
| [`docs/README.md`](../README.md) | 전체 문서 색인 |
| [`status.md`](./status.md) | 지금 실제로 완료·PoC·미구현인 기능과 실험 상태 |
| [`roadmap.md`](./roadmap.md) | 다음 구현 순서와 held-out 연기 결정의 기준 문서 |
| [`open_issues.md`](./open_issues.md) | 알려진 한계, 일반화 금지 조건, 기술 부채 |
| `docs/experiments/` | 날짜별 실행 조건·결과·해석을 고정한 실험 기록 |
| `docs/protocols/` | 평가·재현·전송 정상화·결과 보존 절차 |
| `docs/architecture/` | Tx/Rx 계약, 지표 정의 등 장기 설계 |
| [`results/README.md`](../../results/README.md) | Git 추적 결과 보존 규칙 |
| [`results/registry.csv`](../../results/registry.csv) | 보존된 모든 run의 경로·commit·핵심 결론 색인 |
| `results/<run>/README.md` | 해당 run의 핵심 수치와 과학적 해석 |
| `results/<run>/manifest.json` | commit·조건·dataset·provenance |
| `results/<run>/checksums.sha256` | 보존 artifact의 SHA-256 무결성 |
| `outputs/` | Git 비추적 원본 프레임·packet·로그·대용량 산출물 |

다음 채팅에서는 먼저 이 문서, `status.md`, `roadmap.md`, `open_issues.md`,
`results/registry.csv`, 최신 Git 상태를 직접 확인한다. 과거 실험의 정확한 수치는
해당 `results/<run>/`의 CSV·JSON을 기준으로 한다.

## 완료된 핵심 검증과 결론

### 1. float32 reliable-digital 전송 진단

- 10dB, 3 core condition × 100 frame, 실패·non-finite·stage conflict 0.
- float32 wire round-trip 300/300 bit-exact.
- digital wire는 AWGN보다 평균 PSNR +0.721dB였고 in-process/wire 최대 PSNR 차이는
  0.000752dB였다.
- 결론: **float32 wire 직렬화·전송은 품질 저하 원인이 아니다.**
- 근거: [실험 문서](../experiments/2026-08-28_float32_digital_step_normalization_full.md),
  [보존 결과](../../results/float32_digital_normalization_full_20260827/README.md)

### 2. 10dB 양자화 재평가

- 10영상 × fixed AWGN/float32/int16/int8/int6/int4 = 60/60 pair, 실패·non-finite 0.
- int16/int8/int6/int4가 모두 float32 대비 품질 기준을 통과했다.
- 최소 bit-depth `fixed_int4`: byte -28.45%, PSNR -0.0526dB,
  SSIM -0.00201, LPIPS 변화 -0.000487.
- AWGN은 참고 기준이며 digital Pareto baseline으로 사용하지 않는다.
- 당시 결론: **rate-first operating point는 `fixed_int4`.** 2026-09-04부터 신규 ETRI
  기본 운용점은 baseline에 더 가까운 PSNR·SSIM을 보인 `fixed_int6`로 변경했다.
- 근거: [실험 문서](../experiments/2026-08-28_quantization_reevaluation_10db.md),
  [보존 결과](../../results/quantization_reevaluation_10db_20260828/README.md)

### 3. fixed–SKEM exact matched-rate

- 100/100 pair와 50/50 rate row 완료, actual transmitting count exact.
- raw byte 차이는 최대 0.004953%, padding 후 effective byte는 exact match.
- proxy SKEM이 10/10 영상에서 fixed와 같은 keyframe/transmission schedule로 수렴해
  품질 차이도 정확히 0이었다.
- 결론: **현재 proxy SKEM은 fixed 대비 이점이 없다.** 이 실험의 int4 조건은
  historical evidence로 유지하며 신규 ETRI bit-depth 결정은 int6 기록을 따른다.
- 근거: [실험 문서](../experiments/2026-08-28_fixed_skem_matched_rate_10db.md),
  [보존 결과](../../results/fixed_skem_matched_rate_10db_20260828/README.md)

### 4. edge·uncertainty 전송 절감

- 10영상 × 16 guide profile = 160/160 pair, 실패·non-finite 0.
- baseline `fixed_int4` packet의 edge와 uncertainty가 각각 약 45.41%, 합계 90.83%.
- reliable-digital 경로에서 uncertainty는 실제 decoder에 소비되지 않았다.
- 1차 최소 후보 `combined_ds4`는 356,824.7 bytes/video(-85.11%)였고 pixel 변화는
  수치 오차 수준이었다.
- 이후 통합 검증에서 빈 조합을 닫았으며 `candidate_both_omit`은
  219,459.7 bytes/video로 baseline 대비 **90.843% 절감**했다.
- full50에서는 both-omit의 PSNR/SSIM/LPIPS 변화가 사실상 0이고 closed/open semantic
  지표가 baseline과 정확히 같았다.
- 결론: **현재 reliable-digital 개발 조건의 guide 후보는 both-omit.** 기본 경로를
  즉시 교체하지 말고 opt-in Tx/Rx 정책으로 정리해야 한다.
- 근거: [1차 실험](../experiments/2026-08-28_edge_uncertainty_ablation_10db.md),
  [보존 결과](../../results/edge_uncertainty_ablation_10db_20260828/README.md)

### 5. 통합 semantic·hallucination·temporal 평가

- 조건: fixed selector, `fixed_int4`, 10dB, seed 2025.
- 10영상 × 3 decoder(full50/few10/VAE-direct) × 4 guide = 120/120 pair,
  총 12,000 frame, 실패·non-finite 0.
- CLIP·OWLv2·VQA가 각각 47,744건 기여했고 3-GPU provenance가 정상이다.
- 공식 base: `fixed_int4 + baseline guides + full50`.
- 개발셋 잠정 후보: `fixed_int4 + candidate_both_omit + few10`.

| 지표 | base | 잠정 후보 | 변화 |
|---|---:|---:|---:|
| bundle bytes/video | 2,396,632.7 | 219,459.7 | -90.843% |
| reconstruction elapsed/video | 108.0504 s | 39.4540 s | -63.486%, 2.74× |
| PSNR | 23.47628 | 23.23277 | -0.24351dB |
| SSIM | 0.733301 | 0.731419 | -0.001882 |
| LPIPS | 0.253619 | 0.269202 | +0.015583 |
| closed PTC | 0.7716 | 0.7703 | -0.0013 |
| open hallucination rate | 0.0160 | 0.0355 | +0.0195 |
| additional objects/100 frames | 2.7 | 4.8 | +2.1 |

- PSNR·SSIM·LPIPS와 closed semantic/temporal 평균 gate는 통과했다.
- open hallucination 증가 CI 상한 0.0525와 additional-object 증가 CI 상한 0.0570이
  margin 0.05를 조금 넘었다. 증가는 `01_person_walk`, `02_car_pass`에 집중됐다.
- VAE-direct는 23.4885s/video로 가장 빠르고 PSNR·LPIPS도 개선됐지만 SSIM 하락
  0.01129가 margin 0.01을 넘어 탈락했다.
- 결론: **`fixed_int4 + both-omit + few10`은 최종 모델이 아니라 개발셋 잠정
  후보**다. 보수적 비교점은 `fixed_int4 + both-omit + full50`이다.
- 근거: [실험 문서](../experiments/2026-08-29_integrated_semantic_validation_10db.md),
  [보존 결과](../../results/integrated_semantic_validation_10db_20260829/README.md)

### 6. negative-semantics G1 additional-object/ghost-track phenomenon

- 조건: `fixed_int4`, `candidate_both_omit`, `{few10,full50}`, OVIS Pilot
  40영상, 3 declared seed, OWLv2 threshold 0.2(source-only calibration).
- 240/240 완료, 실패 0, held-out 미접근. declared-seed 정의로 `gate_status:
  PASSED`(few10 h_add 0.019793, full50 0.018536, 24/23개 영상에 분산).
- v1.2 amendment: 이 경로는 `seed`를 소비하지 않아 `effective_seed_count=1`이다.
  같은 gate threshold를 정직하게 재적용하면 few10 기준 **`NOT_PASSED`**로
  뒤집힌다. source-paired h_add(원본이 이미 오탐이던 경우 제외)는 raw 대비
  약 47~51% 낮고, full50은 prevalence 조건 미달이다.
- 결론: **P1 현상 자체는 존재하지만(effective seed 1개, video 전반에 분산),
  "여러 독립 seed에 걸쳐 재현"과 "reconstruction이 만든 오탐이 두 정책 모두
  prevalence 기준을 만족"이라는 더 강한 주장은 아직 성립하지 않는다.**
- 근거: [v1.1 결과·감사](../experiments/2026-09-06_negative_semantics_g1_v1_1_pilot_results.md),
  [v1.2 amendment](../experiments/2026-09-06_negative_semantics_g1_v1_2_amendment.md),
  [보존 결과](../../results/negative_semantics_g1_pilot_rtx4080_v1_1/README.md)

## 현재 임시 결정

- ETRI 신규 기본 bit-depth: `fixed_int6`
- G1 v1.1 stress bit-depth: `fixed_int4` 유지
- selector: fixed. proxy SKEM은 현재 이점 없음.
- guide: `candidate_both_omit`을 opt-in 개발 후보로 사용.
- decoder: `few10`을 잠정 후보, `full50`을 보수적 기준으로 유지.
- 실측된 과거 잠정 후보: **`fixed_int4 + candidate_both_omit + few10`**.
- 신규 ETRI 목표 조합: **`fixed_int6 + candidate_both_omit + few10`**이며 bridge
  validation 전에는 실측 후보나 final/best generalized operating point라고 부르지 않는다.

## held-out 검증 상태

- 별도 독립 데이터셋을 당장 만들 수 없어 2026-08-29에 **데이터 준비 시점까지
  명시적으로 연기**했다. 취소하거나 통과한 것으로 간주하지 않는다.
- 기존 개발 10영상으로 threshold를 더 튜닝하지 않는다.
- 새 데이터가 준비되면 권장 20영상 이상, 영상당 100 frame으로 다음 3조건을 비교한다.
  1. `fixed_int6 + baseline + full50`
  2. `fixed_int6 + both-omit + full50`
  3. `fixed_int6 + both-omit + few10`
- `fixed_int4 + both-omit`은 G1 stress reference와 int6 bridge 비교점으로 별도 보존한다.
- 최종 판정에서는 평균뿐 아니라 hallucination/additional-object paired CI 상한까지
  margin 안에 들어와야 한다.

## 바로 이어서 할 작업

1. **int6 bridge 정식 GPU 실행 (코드·설정·테스트는 완료, 실행만 남음)**
   - `docs/experiments/2026-09-06_negative_semantics_int6_bridge_preparation.md`의
     명령대로 preflight → smoke → 정식 실행 순서를 tmux에서 수행한다.
   - `int6_bridge_summary.json`을 판독해 exact byte/PSNR·SSIM·LPIPS/latency
     delta와 quality gate, raw/source-paired additional-object, ghost를
     확인하고 날짜 문서·`results/`에 보존한다.
   - G1 v1.1(240/240 완료, declared-seed `PASSED`)과 v1.2 amendment
     (effective-seed 기준 `NOT_PASSED`, full50 source-paired h_add 미달)는
     이미 끝났다 — 재실행하지 않는다.
2. **guide Tx/Rx 계약 정리**
   - both-omit을 명시적 opt-in 정책으로 고정한다.
   - baseline 동작을 유지하고 manifest, packet accounting, resume signature에 정책을
     기록한다.
3. **verifier를 실제 sampler에 연결**
   - 현재 결정·로그만 하는 action을 retry, stop, negative prompt, prompt emphasis,
     fallback에 실제 반영한다.
   - 최대 retry, 실패 fallback, 추가 지연·시도 횟수·추가 전송 byte를 기록한다.
4. **verifier 폐루프 ablation**
   - OFF / 로그 전용 / retry·stop / prompt 제어 포함 조건을 비교한다.
   - 품질, hallucination, temporal, 추가 연산량과 지연을 함께 본다.
5. **동적 전송 예산 controller**
   - 채널 상태, uncertainty, verifier 위험도로 전송량과 복원 연산량을 결정한다.
   - feedback, retransmission byte, RTT를 accounting에 포함한다.
6. **데이터 준비 후 held-out과 최종 문서 마감**
   - 최종 operating point, paired CI, Pareto 표·그래프·재현 명령·checksum을 확정한다.

## 작업 시 주의할 과학적 경계

- both-omit의 무손실 결론은 현재 reliable-digital checkpoint/config 개발 조건에 한정한다.
- few10은 학습된 distilled/consistency model이 아니라 production sampler의 10-step
  근사다.
- 물리 channel symbol·FEC 환산은 여전히 proxy이며 bundle byte만 exact하다.
- verifier action은 아직 sampler에 실제 개입하지 않는다.
- held-out 전에는 최종 일반화·최종 운영점 주장을 하지 않는다.

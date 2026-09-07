---
status: active
updated: 2026-09-07
owner: ETRI SGD-JSCC 연구팀
source_commit: 4544094
---

> [← 문서 색인](../README.md)

# 다음 채팅용 연구개발 인계 요약

## 2026-09-07 SAVER-JSCC 연구선 등록

- 활성 제안 모델을 **SAVER-JSCC: Signed Assertions and Versioned Entity Memory for
  Revocation-Aware Generative Video JSCC**로 정했다.
- 단일 기준: [saver_jscc_model_plan.md](./saver_jscc_model_plan.md).
- SAVER 핵심 모델은 `PROTOTYPE_IMPLEMENTED_UNTRAINED`다. SAT/JASR/action-conditioned
  codec/VREM/SM-DiT, generic backbone bridge, signed packet/fault channel,
  source-only tensor dataset, seven-loss stage runner와 versioned checkpoint가 별도 branch에
  구현됐다. 실제 SAVER 학습 checkpoint와 formal 성능 근거는 없다.
- 기존 [negative-semantics 계획](./negative_semantic_paper_plan.md)은 삭제하지 않고
  SAVER SV0/SV1의 데이터·Oracle·packet/RSM 선행 gate로 유지한다.
- G3 Oracle REVOKE는 `no_rsm`/`append_only_rsm`/`revocable_rsm` condition manifest,
  frame-aligned video injection, receiver-state 전환 GOP split, event-cluster ghost/identity
  evaluator와 read-only audit까지 구현했다. protocol은 G2 통과 전 draft이고 GPU 결과는 없다.
- `SV0 = G2 Oracle ABSENT` 1-video smoke는 완료됐고 `NOT_EVIDENCE`다. 40-video formal
  Pilot는 기존 clean commit `6f9593d`에서 실행 중이다. 구조 코드는 선행 구현했지만,
  Pilot 판정 전에 대규모 SAVER 학습을 시작하지 않는다.
- 호환성은 production default와 기존 package export를 바꾸지 않고 SAVER 전용 config와
  import로만 opt-in하도록 구현했다.
- 문서 역할: 설계·gate는 SAVER plan, 실제 상태는 `status.md`, 다음 작업은
  `roadmap.md`, 실행 결과는 새 날짜 기반 `docs/experiments/` 문서에 기록한다.

## 2026-09-07 int6 bridge 완료와 fixed_int4 primary 재결정

- **G1 v1.1 정식 실행이 완료됐다**: OVIS Pilot 40영상 × `{few10,full50}` ×
  3 declared seed, child run 6개 × 40영상 = **240/240**, 실패 0, held-out
  미접근. `scripts/audit_negative_semantics_g1.py --require-runner-complete`는
  exit 0, `--require-scientific-gate`는 exit 5다.
  `g1_summary.json.gate_status: PASSED`는 declared-seed 정의에만 해당한다.
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
- code-clean 상태에서 `outputs/…_derived_v1_2c`를 재파생했고
  `score_and_hash_dual_verified`, effective-seed `NOT_PASSED` 결론이 유지됐다.
  v1_2c의 최소 artifact를 `results/`에 보존하는 작업은 남아 있다.
- bridge hardening까지 포함한 회귀 검증은 `python -m pytest tests/` 기준
  **1652 passed, 0 failed**였다.
- **`fixed_int4 vs fixed_int6` paired bridge formal을 완료했다** — OVIS Pilot
  40영상 × `{few10,full50}`, seed 2025, int6 child run 정책당 40/40, 실패 0,
  held-out 미접근. 실행 상태와 두 품질 gate는 `PASSED`다.
- int6는 int4 대비 PSNR +0.100~0.146dB, SSIM +0.00168~0.00228,
  LPIPS -0.00123~-0.00171로 소폭 개선됐지만 bundle bytes가 두 정책 모두
  **44.217% 증가**했다. raw/source-paired H_add delta의 95% CI는 모두 0을
  포함한다. ghost는 정책당 uncensored EXIT event가 4개뿐이다.
- 10,589개 immutable artifact, 1,217,972,224 bytes를 독립 재해시해 mismatch
  0을 확인했다. formal wall time은 약 8시간 49분이다.
- 위 결과로 2026-09-04의 int6 임시 결정을 대체하고 **논문 primary development
  bit-depth를 `fixed_int4`로 확정**했다. int6는 robustness/품질 민감도
  ablation과 시연 opt-in으로만 유지한다.
- 근거: [bridge 결과](../experiments/2026-09-07_negative_semantics_int6_bridge_results.md),
  [fixed_int4 결정](../experiments/2026-09-07_fixed_int4_primary_operating_point_decision.md).
- 다음 작업은 구현된 fixed_int4 기반 G2 Oracle ABSENT의 GPU smoke/Pilot 검증이다.
  G1 effective-seed gate가 `NOT_PASSED`인 동안 G2 결과는 mechanism feasibility를 보는
  탐색 근거로만 기록한다.

## 2026-09-04 int6 임시 결정 — superseded

- 당시 baseline-guide 10영상 결과만으로 int6를 신규 기본값으로 정했으나,
  2026-09-07 OVIS both-omit bridge와 통신 효율 우선순위에 따라 superseded됐다.
- 과거 결정과 산출물은 역사적 artifact로 보존한다. 현재 적용 기준은
  [fixed_int4 결정](../experiments/2026-09-07_fixed_int4_primary_operating_point_decision.md)이다.

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
- 당시 결론은 **rate-first operating point `fixed_int4`**였다. 2026-09-04의
  fixed_int6 임시 결정은 후속 OVIS bridge 결과로 superseded됐고 현재 primary도
  `fixed_int4`다.
- 근거: [실험 문서](../experiments/2026-08-28_quantization_reevaluation_10db.md),
  [보존 결과](../../results/quantization_reevaluation_10db_20260828/README.md)

### 3. fixed–SKEM exact matched-rate

- 100/100 pair와 50/50 rate row 완료, actual transmitting count exact.
- raw byte 차이는 최대 0.004953%, padding 후 effective byte는 exact match.
- proxy SKEM이 10/10 영상에서 fixed와 같은 keyframe/transmission schedule로 수렴해
  품질 차이도 정확히 0이었다.
- 결론: **현재 proxy SKEM은 fixed 대비 이점이 없다.** 이 실험의 int4 조건은
  historical evidence이자 현재 primary bit-depth와 일치한다.
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

### 7. fixed_int4–fixed_int6 formal bridge

- OVIS Pilot 40영상 × `{few10,full50}`, seed 2025, int6 reconstruction 실패 0.
- int6 quality gate는 모두 통과했지만 exact bundle bytes가 int4 대비 44.217%
  증가했다.
- raw/source-paired H_add의 video-paired 95% CI는 모두 0을 포함했다.
- 결론: int6는 작은 화질 이득이 있지만 통신 효율과 negative-semantics 측면에서
  primary를 바꿀 근거가 없다.
- 근거: [bridge 결과](../experiments/2026-09-07_negative_semantics_int6_bridge_results.md)

## 현재 결정

- 논문 primary development bit-depth: `fixed_int4`
- robustness/품질 민감도 및 시연 opt-in: `fixed_int6`
- selector: fixed. proxy SKEM은 현재 이점 없음.
- guide: `candidate_both_omit`을 opt-in 개발 후보로 사용.
- decoder: `few10`을 잠정 후보, `full50`을 보수적 기준으로 유지.
- primary 개발 조합: **`fixed_int4 + candidate_both_omit + few10`**.
- held-out 전에는 이를 final/best generalized operating point라고 부르지 않는다.

## held-out 검증 상태

- 별도 독립 데이터셋을 당장 만들 수 없어 2026-08-29에 **데이터 준비 시점까지
  명시적으로 연기**했다. 취소하거나 통과한 것으로 간주하지 않는다.
- 기존 개발 10영상으로 threshold를 더 튜닝하지 않는다.
- 새 데이터가 준비되면 권장 20영상 이상, 영상당 100 frame으로 다음 3조건을 비교한다.
  1. `fixed_int4 + baseline + full50`
  2. `fixed_int4 + both-omit + full50`
  3. `fixed_int4 + both-omit + few10`
- 핵심 결론의 bit-depth sensitivity는 위 조건의 선택된 subset만 `fixed_int6`로
  반복하고 primary 판정과 분리한다.
- 최종 판정에서는 평균뿐 아니라 hallucination/additional-object paired CI 상한까지
  margin 안에 들어와야 한다.

## 바로 이어서 할 작업

1. **SAVER SV0 / G2 Oracle ABSENT GPU 검증 마감**
   - receiver injection, four-arm runner, matched-compute check와 read-only audit 구현 완료
   - 1-video smoke는 완료(`NOT_EVIDENCE`); 실행 중 40-video Pilot 종료 후 감사를 수행
   - `fixed_int4 + candidate_both_omit`에서 no negative / random / frequency /
     oracle negative를 동일 generation·step budget으로 비교한다.
   - H_add 상대 감소, paired one-sided CI, false suppression, PSNR/SSIM/LPIPS와
     추가 compute를 함께 기록한다.
   - G1 effective-seed gate가 `NOT_PASSED`인 동안 탐색적 mechanism feasibility로만
     판정하고 held-out은 열지 않는다.
2. **int6 bridge 결과 보존 마감**
   - 날짜 결과 문서는 작성됐다. 필요한 최소 artifact를 `results/`에 복사하고
     manifest/checksum/registry를 추가한다. 1.2GB 원본 전체를 Git에 넣지 않는다.
3. **SV0 통과 시 실제 SAVER 학습 데이터와 SV1/SV2 실행**
   - prompt-only receiver RSM, signed packet/ledger, SAT/VREM/SM-DiT code는 구현됐다.
   - OVIS/YouTube-VOS source-only GOP tensor manifest를 만들고 SV1 memory ablation부터
     실행한다. 이후 real frozen backbone에 SM-DiT bridge를 연결한다.
4. **SV2 통과 시 JASR/codec 및 channel robustness formal**
   - 구현된 router/codec/fault simulator를 실제 학습해 네 budget Pareto와
     loss/reorder/corruption/fading을 평가한다.
5. **데이터 준비 후 held-out과 최종 문서 마감**
   - 최종 operating point, paired CI, Pareto 표·그래프·재현 명령·checksum을 확정한다.

## 작업 시 주의할 과학적 경계

- SAVER는 구현됐지만 미학습 prototype이다. 기존 G0/G1/int4-int6 결과와 CPU test를
  SAVER 성능으로 인용하지 않는다.
- both-omit의 무손실 결론은 현재 reliable-digital checkpoint/config 개발 조건에 한정한다.
- few10은 학습된 distilled/consistency model이 아니라 production sampler의 10-step
  근사다.
- 물리 channel symbol·FEC 환산은 여전히 proxy이며 bundle byte만 exact하다.
- verifier action은 아직 sampler에 실제 개입하지 않는다.
- held-out 전에는 최종 일반화·최종 운영점 주장을 하지 않는다.

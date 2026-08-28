---
status: completed
date: 2026-08-28
experiment_commit: 2218824f301ad238129a72ade42b0210be2f21a5
run_id: fixed_skem_matched_rate_10db_20260828_011620
---

> [← 문서 색인](../README.md)

# fixed–SKEM exact matched-rate 10dB 재평가

## 목적과 조건

기존 selector 비교는 keyframe 수만 맞췄고 `recompute_semantic`을 포함한 실제 visual
전송 프레임과 bundle byte가 달라 50개 비교 중 35개만 10% rate tolerance를
통과했다. 이번 실험은 다음 조건으로 재평가했다.

- ETRI 10영상 × 100 frame
- fixed/SKEM × float32/int16/int8/int6/int4 = 100 pair
- fixed-reference decoder step 10dB, seed 2025
- 3×RTX 4090 병렬 실행
- fixed max-GOP 유지, 영상별 proxy SKEM threshold/max-segment 보정
- 실제 visual-transmitting frame count exact match
- raw bundle byte 차이 ≤1%, 작은 쪽 padding 후 effective byte exact match
- AWGN 제외: paired digital 비교에만 집중

실행 명령은 `scripts/run_fixed_skem_matched_rate_10db.sh`에 고정했고, 자동 검증기는
완료 pair/rate row, finite, 10dB, worker provenance, plan-vs-actual schedule과 byte gate를
fail-closed로 확인했다.

## 완료·재현성

- 100/100 pair, 50/50 rate row, plan 10/10
- 실패·NaN/Inf 0, quality frame 10,000/10,000
- worker_00/01/02 return code 0, physical cuda:0/1/2 provenance 일치
- 자동 check 14개 전부 PASS
- 실행 시간: 2026-08-28 10:16:36~12:24:47 KST(약 2시간 8분)
- 원격→로컬: 21,194개 파일, 1,926,849,641 bytes, 전 파일 SHA-256 일치
- 보존 결과: [results/fixed_skem_matched_rate_10db_20260828](../../results/fixed_skem_matched_rate_10db_20260828/README.md)

## exact-rate 결과

50개 video-channel 비교 모두 실제 전송 frame count가 같고 raw byte 차이는 1%보다
훨씬 작았다.

| Channel | 평균 raw 차이 | 최대 raw 차이 | padding 후 |
|---|---:|---:|---|
| float32 | 0.003239% | 0.003544% | exact |
| int16 | 0.003866% | 0.004232% | exact |
| int8 | 0.004280% | 0.004687% | exact |
| int6 | 0.004398% | 0.004817% | exact |
| int4 | 0.004522% | 0.004953% | exact |

fixed raw bundle은 영상×channel마다 SKEM보다 100 byte 컸다. packet component를 독립
합산한 결과 이 차이는 100개 frame의 `manifest_bytes`에서만 frame당 1 byte였고,
실질 payload 구성요소는 모두 같았다. SKEM 쪽에 100 byte/video를 계상하면 effective
rate는 정확히 같다. 따라서 raw Pareto가 `skem_int4`를 100 byte 작다고 선택한 것은
config label 길이의 metadata 효과일 뿐 selector 전송률 이득이 아니다.

## 핵심 해석: proxy SKEM의 fixed schedule 수렴

보정 결과 9영상은 threshold 0.35/max-segment 16, `09_scene_cut_chair_car`는
threshold -0.65/max-segment 16을 선택했다. 하지만 최종적으로:

- fixed/SKEM keyframe index 동일: 10/10
- fixed/SKEM actual transmitting index 동일: 10/10
- 모든 영상·bit-depth의 paired PSNR/SSIM/LPIPS 차이: 정확히 0

즉, 품질 동률은 서로 다른 semantic schedule을 공정하게 비교한 결과가 아니라 proxy
SKEM이 exact-rate 보정에서 fixed schedule로 수렴한 결과다. 이번 실험은 이전의 rate
mismatch를 해소했지만 **proxy SKEM의 우위를 입증하지 못했고, 현 operating point에
SKEM을 추가할 근거가 없다는 null 결과**로 판정한다.

다른 schedule의 semantic 효용을 계속 평가하려면 exact-count 후보 중 fixed와 다른
index를 강제하거나 실제 MLLM PSSS backend로 재검증해야 한다. 이 항목은 proxy 기반
전송 operating point 결정과 분리된 후속 연구다.

## bit-depth 재확인과 다음 작업

같은 fixed schedule 기준 float32 대비 int4 결과는 다음과 같다.

- bytes/video: 3,346,860.5 → 2,394,650.1(-28.45%)
- PSNR: -0.052631dB
- SSIM: -0.00200559
- LPIPS rise: -0.000487415(개선 방향)

따라서 `fixed_int4` operating point를 유지한다. 해당 packet의 edge+uncertainty가
90.90%, visual latent가 5.89%이므로 다음 우선 작업은 edge·uncertainty 선택 전송,
양자화, 해상도 축소 ablation이다.

## 주요 산출물

- `matched_rate_plan.csv`: 영상별 후보 탐색과 선택 schedule
- `rate_matching.csv`: raw/padding/effective byte 검증
- `matched_rate_validation.json`, `MATCHED_RATE_REPORT.md`: 자동 verdict
- `matched_rate_quality_effect.csv`, `selector_effect.csv`: selector paired 결과
- `validation_report.json`: 독립 교차검증과 전송 무결성
- `checksums.sha256`: 보존 artifact checksum

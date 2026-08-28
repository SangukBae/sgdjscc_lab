---
status: completed
updated: 2026-08-28
owner: ETRI SGD-JSCC 연구팀
run_id: edge_uncertainty_ablation_10db_20260828_051915
source_commit: 1d33fa312d0f6c540a319e791102f98b5f0118b7
---

> [← 문서 색인](../README.md) · [보존 결과](../../results/edge_uncertainty_ablation_10db_20260828/README.md)

# fixed_int4 edge·uncertainty 전송 절감 ablation

## 질문

10dB `fixed_int4` packet의 대부분을 차지하는 edge·uncertainty를 양자화, 공간 축소,
시간 재사용 또는 생략했을 때 실제 직렬화 byte를 얼마나 줄일 수 있으며 기존 pixel
quality gate를 만족하는가?

## 실행 조건

- 10개 영상 × 16 profile = 160 video-profile pair
- 각 영상 100 frame, 총 16,000 quality frame
- fixed selector, int4 visual latent, fixed-reference 10dB
- seed 2025, 3×RTX 4090 (`cuda:0,1,2`)
- commit `1d33fa312d0f6c540a319e791102f98b5f0118b7`, tracked dirty false
- 품질 gate: PSNR 하락 ≤0.5dB, SSIM 하락 ≤0.01, LPIPS 증가 ≤0.02

독립 profile은 edge와 uncertainty 각각 q4, ds2, ds4, reuse2, omit이며 결합 profile은
q4, ds2, ds4, reuse2, q4+ds2+reuse2다. 준비와 wire-accounting 계약은
[준비 문서](./2026-08-28_edge_uncertainty_ablation_preparation.md)에 기록돼 있다.

## 완전성

- 160/160 pair 완료, 실패 0, NaN/Inf 0
- 세 worker return code 0, 계획 GPU와 worker manifest GPU 일치
- 전 row fixed-reference SNR 10dB
- 16,000개 packet component row의 합이 실제 bundle byte와 일치
- 자동 fail-closed check 11개와 사후 독립 검산 모두 통과
- 원격/로컬 33,256개 파일, 2,813,319,377 bytes 및 전 파일 SHA-256 일치

## 결과

baseline은 평균 2,396,632.7 bytes/video이고 edge와 uncertainty가 각각 45.41%,
합계 90.83%를 차지했다. 시험한 16개 profile은 모두 pixel quality gate를 통과했다.

| 후보 | bytes/video | baseline 대비 절감 | PSNR 하락 | SSIM 하락 | LPIPS 증가 |
|---|---:|---:|---:|---:|---:|
| baseline | 2,396,632.7 | 0% | 0 | 0 | 0 |
| combined_ds4 | **356,824.7** | **85.11%** | 0.0000098dB | 0.000000115 | -0.000000564 |
| combined_q4_ds2_reuse2 | 370,956.1 | 84.52% | 0.0000047dB | 0.000000074 | -0.000000233 |
| combined_ds2 | 764,786.3 | 68.09% | 0.0000076dB | 0.000000084 | -0.000000415 |
| uncertainty_omit | 1,307,996.4 | 45.42% | 0 | 0 | 0 |
| edge_omit | 1,308,096.0 | 45.42% | 0.0000074dB | 0.000000081 | 0.000000089 |

자동 Pareto는 `combined_ds4`를 gate 내 최소-byte profile로 선택했다. 이는
`combined_q4_ds2_reuse2`보다도 14,131.4 bytes/video 작다. ds4는 양 축을 1/4로
줄여 pixel 수를 1/16로 만드는 반면, q4+ds2+reuse2는 payload 감소가 비슷해도
양자화/재사용 item overhead가 남기 때문이다.

## 중요한 경로 해석

모든 uncertainty-only profile은 **10개 영상 각각에서** baseline과 PSNR, SSIM,
LPIPS가 정확히 같았다. 이는 단순히 “uncertainty 압축이 우수하다”는 결과가 아니다.

`receiver_runtime.py`의 reliable-digital 경로는 직렬화된 edge를 이미 복원했으므로
`_decode_diffusion(..., edge_already_received=True)`를 호출한다. 이때
`infer_pipeline.py`는 analog Canny 재전송 `_retransmit_canny()`를 건너뛴다.
uncertainty는 그 함수에서 edge와 concatenate될 때만 소비되므로, 평가한 digital
복원 경로에서는 uncertainty 값이 결과에 영향을 줄 수 없다. 현재 uncertainty
payload는 wire byte에는 포함되지만 decoder에는 기여하지 않는 중복 정보다.

edge는 VAE를 거쳐 ControlNet latent로 실제 연결된다. 하지만 `edge_omit`의 영상별
최대 절대 변화도 PSNR 0.0000747dB, SSIM 0.000000444, LPIPS 0.00000252뿐이었다.
이는 이 checkpoint와 `controlnet_scale=0.3` 조건에서 edge 영향이 수치적으로 매우
약하다는 신호다. pixel metric만으로 구조·semantic guide 효용까지 부정할 수는 없다.

## 판정

- 기술적 실행·전송 회계·품질 gate: **PASS**
- 시험한 profile 중 최소 byte: **`combined_ds4`**
- 최종 운영점: **보류**

`combined_ds4`를 최종 운영점으로 확정하지 않는 이유는 다음과 같다.

1. uncertainty가 현재 digital decoder에서 소비되지 않아 ablation이 구조적으로 null이다.
2. `edge_ds4 + uncertainty_omit`, `edge_omit + uncertainty_omit` 혼합 후보를 시험하지 않았다.
3. SRS, hallucination, temporal consistency와 전체 latency를 측정하지 않았다.
4. 개발 10영상 결과이며 별도 held-out 일반화 검증이 아니다.

## 후속 작업

1. uncertainty를 reliable-digital 기본 packet에서 제거할지, decoder 조건으로 실제
   연결할지 계약을 결정한다.
2. 혼합 후보 `edge_ds4 + uncertainty_omit`과 `edge_omit + uncertainty_omit`을
   추가하고 ControlNet off/scale sensitivity를 함께 점검한다.
3. 통과 후보를 VAE-direct/few-step/full diffusion 통합 평가에 넣어 rate, pixel
   quality, SRS, hallucination, temporal consistency, end-to-end latency를 비교한다.

핵심 원시 결과와 독립 검산은 [보존 결과](../../results/edge_uncertainty_ablation_10db_20260828/README.md)의
`guide_ablation_effect.csv`, `guide_component_bytes.csv`,
`guide_ablation_validation.json`, `validation_report.json`을 기준으로 한다.

---
status: frozen
updated: 2026-08-27
owner: ETRI SGD-JSCC 연구팀
experiment_commit: 607f72798da24dc3e0c065574efcf6fce90683f3
documentation_commit: unknown
supersedes:
---

> [← 문서 색인](../README.md)

# 전송 실험 정상화 결과

> **후속 상태(2026-08-28):** 이 문서의 60 dB 품질 수치와 fixed operating point는
> legacy 근거다. 10 dB fixed-only 재평가에서 `fixed_int4`가 품질 허용 기준을
> 통과한 최소 bit-depth로 재확정됐다. 최신 판정은
> [10 dB 양자화 재평가](./2026-08-28_quantization_reevaluation_10db.md)를 사용한다.
> SKEM 결과는 여전히 actual-byte matched-rate 재평가 전까지 잠정이다.

## 범위

- 데이터: ETRI 10영상 × 100프레임
- 설정: fixed/SKEM proxy × float32/16/8/6/4-bit + AWGN 참고 행
- 실행: RTX 4090 3장 병렬
- 결과: 110/110 pair 완료, 실패·NaN/Inf 0건
- 산출물(보존): [`results/transmission_normalization_20260826/`](../../results/transmission_normalization_20260826/README.md)
  (원본 대용량 artifact: `outputs/transmission_normalization_parallel_20260826_093313/`,
  Git 미포함)

## 핵심 결과

| 설정 | bytes/frame | float32 대비 절감 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| fixed float32 | 35,836.7 | 기준 | 11.318 | 0.0810 | 0.7394 |
| fixed int6 | 26,355.5 | 26.46% | 11.309 | 0.0808 | 0.7400 |
| fixed int4 | 25,626.4 | 28.49% | 11.207 | 0.0765 | 0.7459 |
| SKEM int4 | 24,505.6 | 31.62% | 11.195 | 0.0760 | 0.7462 |

## 판정

- 전송 정상화
  - reliable-digital baseline: `fixed_float32`
  - 전 bit depth finite, int16은 float32와 사실상 동일
- 운영점
  - 보수적 후보: `fixed_int6`
  - 최대 절감 후보: `fixed_int4`
  - `skem_int4`: 잠정 후보 — selector 비교 보완 필요
- 병목
  - int4 packet의 edge + uncertainty: 약 91%
  - visual latent: 약 5.9%

## 해석 제한

- digital 절대 품질
  - AWGN 참고: PSNR 23.34, SSIM 0.731, LPIPS 0.254
  - float32 digital: PSNR 11.32, SSIM 0.081, LPIPS 0.739
  - Tx/Rx edge·ControlNet·diffusion step 계약 점검 전에는 복원 성능 유지 주장 금지
- SKEM
  - rate matching 통과: 35/50 영상×bit-depth 비교
  - 불일치 영상: `02_car_pass`, `07_person_enter`, `09_scene_cut_chair_car`
  - 실제 transmitting frame/byte를 맞춘 재평가 필요
- 미평가
  - real MLLM PSSS
  - SRS, missing/additional object, temporal hallucination

## 다음 작업

1. ~~핵심 CSV·JSON·manifest를 `results/` registry에 고정~~ — 완료:
   [`results/transmission_normalization_20260826/`](../../results/transmission_normalization_20260826/README.md),
   `results/registry.csv`에 등록
2. float32 digital 절대 품질 저하 원인 분리 ablation
3. 실제 transmitting frame/byte 기준 fixed–SKEM matched-rate 재실행
4. edge·uncertainty 압축 후 SRS·할루시네이션 통합 평가

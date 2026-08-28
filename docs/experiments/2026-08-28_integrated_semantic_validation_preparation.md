# 통합 semantic · hallucination · temporal 평가 준비

## 목적

`fixed_int4`의 guide 전송 후보와 receiver 복원 정책을 같은 10개 개발 영상,
10 dB fixed-reference 조건에서 함께 비교한다. Pixel 지표만으로 `combined_ds4`나
VAE-direct를 확정하지 않고, 실제 CLIP·OWLv2·VQA 앙상블의 semantic 보존,
open-world hallucination, PTC/SFR/SDI와 end-to-end reconstruction 시간을 같이 본다.

## 고정 평가 격자

- 영상: ETRI 개발 10개, 영상당 100 frame
- 전송: fixed selector, `fixed_int4`, fixed-reference SNR 10 dB, seed 2025
- guide profile 4개:
  - `baseline`
  - `combined_ds4`
  - `candidate_edge_ds4_uncertainty_omit`
  - `candidate_both_omit`
- decoder policy 3개:
  - `full50`: production diffusion, 50 steps
  - `few10`: production diffusion, 10 steps
  - `vae_direct`: received JSCC latent direct VAE decode, effective diffusion step 0
- 총 120 video-policy-profile pair
- 3개 GPU는 영상 집합을 독립 worker 디렉터리로 나눠 처리한다.

`few10`은 학습된 consistency/distilled 모델이 아니라 현재 production sampler의
10-step 근사임을 결과에 명시한다. `vae_direct`는 기본 경로를 변경하지 않는 opt-in
ablation이며, received latent의 power normalization을 되돌린 뒤 VAE로 직접 decode한다.

## held-out metric 역할과 provenance

복원 PNG를 재사용하므로 semantic 평가는 diffusion을 다시 실행하지 않는다.
원본/복원 packet은 동일 CLIP extractor로 다시 추출한다.

- semantic preservation: GT vocabulary를 사용하는 closed-world filter
- hallucination: GT vocabulary로 닫지 않는 open-world object-noise filter
- presence calibration: ensemble-weighted CLIP + OWLv2 + BLIP2 VQA
- temporal: temporal SRS, object identity consistency, temporal hallucination,
  PTC, SFR, SDI
- 각 calibration object에 세 backend가 모두 기여하지 않으면 실패한다.
- mismatch가 있으나 calibration evidence가 없거나 전체 backend evidence가 0이면
  최종 merge를 거부한다.

## 통계와 개발 screening

기준점은 `full50 + baseline`이다. 모든 차이는 영상별 paired difference로 계산하고
5,000회 deterministic bootstrap 95% CI를 함께 기록한다. 현재 margin은 최종 논문
판정이 아니라 다음 held-out 후보를 줄이기 위한 사전 선언 개발 screening이다.

| 지표 | 허용 margin |
|---|---:|
| PSNR 하락 | 0.5 dB |
| SSIM 하락 | 0.01 |
| LPIPS 증가 | 0.02 |
| closed PTC 하락 | 0.05 |
| closed severity 증가 | 0.05 |
| open temporal hallucination 증가 | 0.05 |
| open additional object/frame 증가 | 0.05 |
| closed SFR 증가 | 0.05 |
| closed SDI 절대 변화 | 0.01 |

평균 gate를 통과한 후보 중 exact bundle byte를 먼저 최소화하고, 동률이면 measured
reconstruction elapsed time을 최소화한다. CI는 별도로 보고하며 별도 held-out 최종
검증 전에는 최종 operating point라고 부르지 않는다.

## 실행과 산출물

```bash
RUN_DIR="integrated_semantic_validation_10db_$(date +%Y%m%d_%H%M%S)"
bash scripts/run_integrated_semantic_validation_10db.sh \
  --output-root "outputs/$RUN_DIR"
```

중단 후에는 같은 디렉터리를 `--resume`으로 지정한다. 재구성 pair와 semantic pair는
각각 독립적으로 재개된다. 핵심 산출물은 `integrated_per_video.csv`,
`integrated_effect.csv`, `integrated_screening_frontier.csv`,
`integrated_validation.json`, `INTEGRATED_EVALUATION_REPORT.md`,
`artifact_sha256.json`, `run_manifest.json`이다.

# float32 digital 복원 품질 저하 진단 결과

> 관련 문서: [docs/protocols/float32_digital_diagnostics.md](../../../docs/protocols/float32_digital_diagnostics.md), [docs/current/open_issues.md](../../../docs/current/open_issues.md)

## 실행 정보

- run_kind: `float32_digital_diagnostics`
- dry_run: `False`
- videos: 1, frames/video: 20, ablations: 2
- failed_cases: 0

## 판정 (baseline, 최종 확정분만 집계)

`verdict_summary`는 **`ablation == "baseline"` AND `status == "final"`** 행만 집계한다 — `serialized_raw_edge`/`awgn_edge_retransmit` 보조 증거나 아직 `diffusion_bypass_vae_direct` 결과를 기다리는 provisional 판정을 더해 과대 집계하지 않는다.

- 종합 판정(최다, baseline·final 기준): `inconclusive`
  - `inconclusive`: 20건

## (video, frame)별 판정

`ablation` 열이 `baseline`인 행만 위 종합 판정에 집계된다. serialized_raw_edge/awgn_edge_retransmit는 별도 보조 증거이며, `status`가 `provisional`인 행은 아직 확정되지 않았다. `evidence_level`은 baseline 판정이 VAE-direct 증거까지 반영했는지 명시한다.

| video | frame | ablation | status | evidence_level | verdict | first_divergent_stage | reason |
|---|---:|---|---|---|---|---|---|
| 01_person_walk | 0 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (35.31 dB) is not meaningfully worse than AWGN (34.59 dB, gap=-0.72 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 1 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (35.10 dB) is not meaningfully worse than AWGN (34.18 dB, gap=-0.92 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 2 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (35.21 dB) is not meaningfully worse than AWGN (33.99 dB, gap=-1.22 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 3 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (35.32 dB) is not meaningfully worse than AWGN (34.95 dB, gap=-0.37 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 4 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (35.32 dB) is not meaningfully worse than AWGN (34.81 dB, gap=-0.51 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 5 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (35.25 dB) is not meaningfully worse than AWGN (34.42 dB, gap=-0.83 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 6 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (35.22 dB) is not meaningfully worse than AWGN (34.46 dB, gap=-0.76 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 7 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (35.29 dB) is not meaningfully worse than AWGN (34.24 dB, gap=-1.04 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 8 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (35.25 dB) is not meaningfully worse than AWGN (34.48 dB, gap=-0.77 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 9 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (35.18 dB) is not meaningfully worse than AWGN (34.34 dB, gap=-0.84 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 10 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (35.30 dB) is not meaningfully worse than AWGN (34.67 dB, gap=-0.63 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 11 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (35.17 dB) is not meaningfully worse than AWGN (34.63 dB, gap=-0.54 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 12 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (34.98 dB) is not meaningfully worse than AWGN (32.84 dB, gap=-2.14 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 13 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (35.04 dB) is not meaningfully worse than AWGN (34.21 dB, gap=-0.83 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 14 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (34.98 dB) is not meaningfully worse than AWGN (34.39 dB, gap=-0.59 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 15 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (35.15 dB) is not meaningfully worse than AWGN (34.43 dB, gap=-0.72 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 16 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (34.96 dB) is not meaningfully worse than AWGN (34.40 dB, gap=-0.56 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 17 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (34.89 dB) is not meaningfully worse than AWGN (33.72 dB, gap=-1.17 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 18 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (34.97 dB) is not meaningfully worse than AWGN (33.99 dB, gap=-0.98 dB < 1.0 dB threshold) — no quality problem evidenc |
| 01_person_walk | 19 | baseline | final | baseline_with_vae_direct | inconclusive |  | digital and in-process paths agree at every instrumented stage, and digital PSNR (35.03 dB) is not meaningfully worse than AWGN (34.30 dB, gap=-0.73 dB < 1.0 dB threshold) — no quality problem evidenc |

## 산출물

- `path_comparison.csv`: `path_comparison.csv`
- `tensor_stage_stats.jsonl`: `tensor_stage_stats.jsonl`
- `tensor_pair_comparison.csv`: `tensor_pair_comparison.csv`
- `failed_cases.csv`: `failed_cases.csv`
- `verdicts.jsonl`: `verdicts.jsonl`
- `summary.json`: `summary.json`

## 판정 기준

- in-process와 wire가 다름 → packet/Tx-Rx 문제
- 두 digital 경로는 같지만 AWGN보다 낮음 → edge·ControlNet·diffusion 문제
- `diffusion_bypass_vae_direct` ablation부터 이미 낮음 → latent scaling/normalization 문제
- 증거가 부족하면 `inconclusive`


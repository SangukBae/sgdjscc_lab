# float32 digital 복원 품질 저하 진단 결과

> 관련 문서: [docs/protocols/float32_digital_diagnostics.md](../../../../docs/protocols/float32_digital_diagnostics.md), [docs/current/open_issues.md](../../../../docs/current/open_issues.md)

## 실행 정보

- run_kind: `float32_digital_diagnostics`
- dry_run: `False`
- videos: 1, frames/video: 100, ablations: 1
- failed_cases: 0

## 판정 (baseline, 최종 확정분만 집계)

`verdict_summary`는 **`ablation == "baseline"` AND `status == "final"`** 행만 집계한다 — `serialized_raw_edge`/`awgn_edge_retransmit` 보조 증거나 아직 `diffusion_bypass_vae_direct` 결과를 기다리는 provisional 판정을 더해 과대 집계하지 않는다.

증거 부족 — `inconclusive`.

## (video, frame)별 판정

`ablation` 열이 `baseline`인 행만 위 종합 판정에 집계된다. serialized_raw_edge/awgn_edge_retransmit는 별도 보조 증거이며, `status`가 `provisional`인 행은 아직 확정되지 않았다. `evidence_level`은 baseline 판정이 VAE-direct 증거까지 반영했는지 명시한다.

| video | frame | ablation | status | evidence_level | verdict | first_divergent_stage | reason |
|---|---:|---|---|---|---|---|---|

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


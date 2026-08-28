# Edge·uncertainty ablation validation — edge_uncertainty_ablation_10db_20260828_051915

- 검증 상태: `PASS`
- 영상: 10개
- profile: 16개
- 완료 pair: 160/160
- 실패 pair: 0개
- fixed-reference SNR: [10.0]
- 품질 gate: PSNR 하락 ≤ 0.5 dB, SSIM 하락 ≤ 0.01, LPIPS 증가 ≤ 0.02
- gate 내 최소 byte profile: `combined_ds4`
- failed checks: 없음

`guide_ablation_effect.csv`는 `fixed_int4__baseline` 대비 품질·byte 변화를,
`guide_component_bytes.csv`는 실제 직렬화 bundle 구성과 transmit/reuse/zero 횟수를,
`guide_pareto_frontier.csv`는 품질 gate 내 다목적 Pareto 후보를 기록한다.

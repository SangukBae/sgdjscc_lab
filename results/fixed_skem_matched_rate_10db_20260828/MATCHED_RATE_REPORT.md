# Fixed–SKEM exact matched-rate validation

- Verdict: **PASS**
- Video-config pairs: 100 / 100
- Rate rows: 50 / 50
- Failed pairs: 0
- GPUs: cuda:0, cuda:1, cuda:2

## Checks

- [x] `parallel_plan_present`
- [x] `actual_transmission_mode_locked`
- [x] `three_unique_gpu_workers`
- [x] `worker_status_complete`
- [x] `worker_manifest_gpu_provenance`
- [x] `all_video_config_pairs_present_once`
- [x] `failed_pairs_zero`
- [x] `nonfinite_zero`
- [x] `snr_10db_all_pairs`
- [x] `one_exact_plan_per_video`
- [x] `all_rate_rows_present_once`
- [x] `actual_transmission_counts_exact`
- [x] `raw_byte_difference_within_1pct`
- [x] `effective_bytes_exact_after_padding`

## Per-channel raw byte matching

- `float32`: n=10, ΔPSNR=0.0, ΔSSIM=0.0, ΔLPIPS=0.0, mean diff=3.238893288961056e-05, max diff=3.5436765218673194e-05, fixed padding=0, SKEM padding=1000 bytes
- `int16`: n=10, ΔPSNR=0.0, ΔSSIM=0.0, ΔLPIPS=0.0, mean diff=3.8657867480453436e-05, max diff=4.231951572931761e-05, fixed padding=0, SKEM padding=1000 bytes
- `int4`: n=10, ΔPSNR=0.0, ΔSSIM=0.0, ΔLPIPS=0.0, mean diff=4.522156354614576e-05, max diff=4.9534082420750425e-05, fixed padding=0, SKEM padding=1000 bytes
- `int6`: n=10, ΔPSNR=0.0, ΔSSIM=0.0, ΔLPIPS=0.0, mean diff=4.3977418676513756e-05, max diff=4.816593743437391e-05, fixed padding=0, SKEM padding=1000 bytes
- `int8`: n=10, ΔPSNR=0.0, ΔSSIM=0.0, ΔLPIPS=0.0, mean diff=4.2799901838549214e-05, max diff=4.687133817670495e-05, fixed padding=0, SKEM padding=1000 bytes

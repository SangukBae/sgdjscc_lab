---
status: frozen
updated: 2026-08-28
owner: ETRI SGD-JSCC 연구팀
experiment_commit: c5721cb6f3501af53555137fb0aaf131b5c2d71b
documentation_commit: unknown
supersedes: docs/experiments/2026-08-28_float32_digital_step_normalization.md
---

> [← 문서 색인](../README.md)

# float32 digital decoder-step 정상화 full 검증

## 목적

[short 검증](./2026-08-28_float32_digital_step_normalization.md)에서 확인한
60dB→10dB `fixed_reference` 수정의 품질 회복이 일반 움직임·semantic change·
scene cut의 300프레임에서도 유지되는지, float32 wire transport가 전 프레임에서
bit-exact인지 최종 확인했다.

## 실행·보관 계약

- 원격 원본: `outputs/f32dig_20260827_164017/`
- 로컬 복사본: `outputs/f32dig_20260827_164017/` (Git 미포함)
- 파일: 660개, 76MB
- 원격–로컬 정렬 SHA-256 tree hash:
  `1acebe14e825a96a8d11f446fd34625ea736ecf8a32fbaa0283a59047c0519e7`
- execution plan SHA-256:
  `075fcbf589317fe199276bbdaca82fe191f3f7d707e3de89455e4fcae13fe86a`
- 실행 commit: `c5721cb6f3501af53555137fb0aaf131b5c2d71b`
- 환경: Python 3.9.25, PyTorch 2.1.0+cu118, CUDA 11.8, RTX 4090 3장
- seed: `2025`
- 정책: float32, `fixed_reference`, `fixed_reference_snr_db=10.0`, `cur_step=1/11`
- 실행 명령:
  `bash scripts/run_float32_digital_diagnostics.sh --profile full --parallel-devices 0,1,2 --fixed-reference-snr-db 10`
- profile: `full`
  - stage 3: 1영상×1프레임, AWGN/in-process/wire tensor 계약
  - stage 4: 동일 프레임의 12개 ablation
  - stage 5: `01_person_walk` 20프레임, baseline + VAE-direct
  - stage 6: 3 core condition 영상×100프레임, baseline

## 실행 상태

| 구간 | 소요 시간 | 결과 |
|---|---:|---|
| 테스트 + dry-run | 47초 | exit 0 |
| stage 3 | 174초 | exit 0 |
| stage 4 | 558초 | exit 0 |
| stage 5 | 1,059초 | exit 0 |
| stage 6 normal motion | 5,263초 | exit 0 |
| stage 6 scene cut | 5,301초 | exit 0 |
| stage 6 semantic change | 5,323초 | exit 0 |
| 전체 | 약 1시간 47분 | `stage_failures=0` |

- 중단·실패 path·NaN/Inf·stage conflict: 모두 0
- stage 3/4/5 계측 tensor의 non-finite row: 0
- stage 6 CSV: worker별 300행(AWGN/in-process/wire × 100프레임), 총 900행
- float32 `digital_wire` round-trip: 300/300 bit-exact
- 6개 stage run manifest: commit·checkpoint set·resolved config·10dB 정책 일치

## 300프레임 품질

| 조건 | 경로 | PSNR | SSIM | LPIPS ↓ |
|---|---|---:|---:|---:|
| normal motion | AWGN | 34.006 | 0.9274 | 0.1271 |
| normal motion | digital wire | 34.852 | 0.9327 | 0.1222 |
| semantic change | AWGN | 34.345 | 0.9268 | 0.1187 |
| semantic change | digital wire | 35.012 | 0.9325 | 0.1177 |
| scene cut | AWGN | 33.661 | 0.9345 | 0.1335 |
| scene cut | digital wire | 34.310 | 0.9401 | 0.1326 |
| **전체 300프레임** | **AWGN** | **34.004** | **0.9296** | **0.1264** |
| **전체 300프레임** | **digital wire** | **34.725** | **0.9351** | **0.1242** |

Digital wire − AWGN paired 차이:

- PSNR: 평균 `+0.721dB`, median `+0.716dB`, 범위 `-0.316∼+2.141dB`
  - digital이 낮은 프레임: 6/300, 전부 scene cut
  - `-1dB` 이하의 의미 있는 저하: 0/300
- SSIM: 평균 `+0.00552`, digital이 높은 프레임 300/300
- LPIPS: 평균 `-0.00223`; digital이 낮은(개선) 프레임 212/300,
  높은 프레임 88/300

평균뿐 아니라 최악 scene-cut 프레임에서도 PSNR 차이는 `-0.316dB`로,
현재 verdict의 품질 저하 기준 `-1dB`보다 충분히 작았다.

## Tx/Rx 계약

- in-process–wire 최대 절대 차이
  - PSNR: `0.0007515dB`
  - SSIM: `2.33e-6`
  - LPIPS: `1.65e-5`
- digital in-process/wire 300프레임 평균은 모든 품질 지표에서 소수점 반올림
  범위로 일치했다.
- 판정: float32 latent 직렬화·역직렬화·수신 normalize는 품질 저하
  원인이 아니며, 60dB decoder-step 계약이 원인이었음이 full에서도 확인됐다.

## VAE-direct 신호(stage 5, 20프레임)

| digital wire | PSNR | SSIM | LPIPS ↓ | harness 계측 latency |
|---|---:|---:|---:|---:|
| 50-step diffusion baseline | 35.146 | 0.9366 | 0.1202 | 6,170ms |
| VAE-direct | 36.736 | 0.9474 | 0.0735 | 48.6ms |

VAE-direct는 pixel fidelity와 receiver decode 구간 latency에서 모두 우수했다. 다만 이 latency는
harness가 계측하는 path 구간이며 sender·bundle 전체 지연이 아니다. 또한 이 실험은
pixel 지표만 비교했으므로 semantic fidelity·할루시네이션·시간축 지표를 포함한
후속 ablation 전에 float32 최종 운영 정책으로 확정하지 않는다.

## 리포트 해석 제한

- `INTEGRATED_REPORT.md`의 overall verdict는 `inconclusive` 20건이다. 이 라벨은 이 실험에서
  품질 문제 발견 실패가 아니라 **현재 fault 분류에서 문제가 감지되지 않음**을 뜻한다.
- stage 6은 `--no-instrument-tensors`로 실행되어 300프레임의 metric row는 보존하지만
  verdict row를 만들지 않는다. 따라서 overall 20건은 stage 5만 집계한 것이며,
  위 300프레임 결론은 stage 6 CSV를 별도 paired 집계한 결과이다.
- stage 4 auxiliary edge 판정 2건은 미세 `edge_mean` 차이를 `packet_tx_rx_issue`로
  표시한다. 최종 품질·bit-exact latent 계약에 영향을 주지 않지만 후속으로
  auxiliary tolerance·중립 label을 보완해야 한다.

## 결론

- **float32 digital 품질 저하 원인 수정·baseline full 정상화: 완료.**
- **float32 raw output 로컬 복사·원격 대조: 완료.**
- **재현성 고정: 미완료.** 핵심 CSV·JSON·manifest·checksum을 `results/` registry에
  보존하고 문서 commit을 연결해야 한다.
- full 범위는 3개 core condition 영상이며 ETRI 10영상 전체·별도 held-out은 아니다.
- 60dB 정책으로 얻은 기존 int16/int8/int6/int4 운영 후보는 10dB에서 재평가해야 한다.

## 다음 작업

1. full 핵심 결과를 `results/` + `results/registry.csv`에 고정.
2. `inconclusive`과 `no_issue_detected`를 분리하고 stage 6 metric-only 판정을 통합 리포트에 포함.
3. auxiliary edge 미세 차이의 tolerance·label 보완.
4. 10dB baseline에서 int16/int8/int6/int4 양자화 품질·Pareto 재평가.
5. VAE-direct/few-step/full diffusion을 semantic·할루시네이션·시간축 지표로 비교.

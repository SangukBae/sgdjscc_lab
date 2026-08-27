---
status: frozen
updated: 2026-08-28
owner: ETRI SGD-JSCC 연구팀
experiment_commit: c5721cb6f3501af53555137fb0aaf131b5c2d71b
documentation_commit: unknown
supersedes:
---

> [← 문서 색인](../README.md)

# float32 digital decoder-step 정상화 short 검증

## 목적

[2026-08-26 전송 정상화 실험](./2026-08-26_transmission_normalization.md)의
float32 digital 절대 품질 저하가 packet/Tx·Rx 직렬화 문제인지, decoder step 정책
문제인지 분리하고, `fixed_reference` 정책을 60dB에서 AWGN 기준 10dB로
변경한 후 품질이 회복되는지 검증했다.

## 실행 계약

- 원격 산출물: `outputs/f32dig_20260827_155549/` (Git 미포함)
- 실행 commit: `c5721cb6f3501af53555137fb0aaf131b5c2d71b`
- 장비: NVIDIA GeForce RTX 4090 3장, GPU별 독립 worker
- 환경: Python 3.9.25, PyTorch 2.1.0+cu118, CUDA 11.8
- seed: `2025`
- 정책: float32, `fixed_reference`, `fixed_reference_snr_db=10.0`, `cur_step=1/11`
- profile: `short`
  - stage 3: `01_person_walk` 1프레임, 3경로 tensor 계약
  - stage 4: 동일 프레임, 12개 ablation
  - stage 5: `01_person_walk` 20프레임, baseline + VAE-direct
  - stage 6: 일반 움직임·semantic change·scene cut 3영상 × 10프레임
- dataset content SHA-256(stage 5):
  `c1995a7b6e03a2a11299f25ce46d317a1cc37fc10cb1c19b3cc73e71eb3c34c4`
- execution plan SHA-256:
  `89384fcbc1eb1fed3bd89ab0fa0fe1753718958f5dad7ba78a57bedd6f42b3f7`

## 실행 상태

| 구간 | 소요 시간 | 결과 |
|---|---:|---|
| 테스트 + dry-run | 35초 | exit 0 |
| stage 3 | 170초 | exit 0 |
| stage 4 | 554초 | exit 0 |
| stage 5 | 1,046초 | exit 0 |
| stage 6, 3 worker 병렬 | 641–643초 | 모두 exit 0 |
| 전체 | 약 29분 | `stage_failures=0` |

- 실패 path: 0
- NaN/Inf: 0
- stage conflict: 0
- float32 `digital_wire` round-trip: 전부 bit-exact

## 핵심 결과

### 수정 전·후 20프레임 비교

| 경로 | PSNR | SSIM | LPIPS ↓ |
|---|---:|---:|---:|
| AWGN | 34.302 | 0.9317 | 0.1229 |
| 수정 전 digital wire, 60dB | 11.543 | 0.0876 | 0.7050 |
| 수정 후 digital in-process, 10dB | 35.146 | 0.9366 | 0.1202 |
| 수정 후 digital wire, 10dB | 35.146 | 0.9366 | 0.1202 |

수정 후 digital wire는 수정 전 대비 PSNR `+23.604dB`, SSIM `+0.8490`,
LPIPS `-0.5848`을 기록했다. 같은 20프레임에서 AWGN 대비 digital wire의 paired
평균은 PSNR `+0.844dB`, SSIM `+0.00486`, LPIPS `-0.00271`로, 기존의 큰 절대 품질
격차가 사라졌다.

### 3개 core condition, 30프레임

| 경로 | PSNR | SSIM | LPIPS ↓ |
|---|---:|---:|---:|
| AWGN | 35.686 | 0.9511 | 0.1267 |
| digital in-process | 36.162 | 0.9544 | 0.1266 |
| digital wire | 36.162 | 0.9544 | 0.1266 |

## 경로 계약 확인

- 실측 step
  - AWGN 프레임 0: `cur_step=0.0956804`, `cur_snr=9.75499dB`
  - digital in-process/wire: `cur_step=0.0909091`, `cur_snr=10.0dB`
- in-process–wire 최대 절대 차이(stage 5 + stage 6)
  - PSNR: `0.0007515dB` 이하
  - SSIM: `1.73e-6` 이하
  - LPIPS: `1.04e-5` 이하
- 해석: float32 latent 직렬화·역직렬화는 품질 저하 원인이 아니었고,
  60dB `fixed_reference` 가정이 diffusion 시작 step을 약 `1e-6`로 만든 것이 핵심 원인이었다.

## 판정 해석

- 통합 리포트의 baseline dominant verdict는 `inconclusive`이다.
- 이 경우의 `inconclusive`은 실패가 아니라, 두 digital 경로가 일치하고 digital PSNR이
  AWGN보다 1dB 이상 낮지 않아 현재 폴트 분류가 어느 문제도 선택하지 않았다는 뜻이다.
- `serialized_raw_edge`/`awgn_edge_retransmit` 보조 판정 2건은 `edge_mean`의
  MAE `0.0005328`, cosine similarity `0.9999906`을 `packet_tx_rx_issue`로 분류했다.
  최종 품질 차이는 최대 `0.001dB` 미만이므로 float32 정상화를 막는 문제는 아니며,
  후속으로 auxiliary 임계값·라벨 해석을 정리할 필요가 있다.

## 결론과 제한

- **short 범위 결론**: float32 digital 절대 품질 저하는 decoder-step 정책 수정으로 해소됐다.
- **아직 미확정**: full profile(3 core condition × 100프레임), ETRI 10영상 전체,
  int16/int8/int6/int4 양자화 operating point.
- 이 short 원본 산출물은 원격 `outputs/`에만 있고 `results/` registry에는 아직
  고정되지 않았다. full 결과 확정 후 핵심 CSV·JSON·manifest와 checksum을 보존한다.
- run manifest의 git dirty 값은 container에 Git CLI가 없어 `unknown`으로 기록되었지만,
  host checkout과 execution plan의 commit은 `c5721cb`로 일치함을 별도 확인했다.

## 다음 작업

1. 같은 10dB 정책으로 3-GPU full profile 실행.
2. full 성공 후 핵심 산출물을 `results/` + registry에 고정.
3. 수정된 step 정책으로 int16/int8/int6/int4 양자화 품질·Pareto 재평가.
4. auxiliary edge 미세 차이의 임계값·판정 라벨 보완.

```bash
bash scripts/run_float32_digital_diagnostics.sh \
  --profile full --parallel-devices 0,1,2 --fixed-reference-snr-db 10
```

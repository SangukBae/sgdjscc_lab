---
status: frozen
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
experiment_commit: unknown
documentation_commit: uncommitted
supersedes:
---

> [← 문서 색인](../README.md)

# 직렬화 packet 전송량 Pareto sweep (2026-08-18)

- `transmission/` 패키지(실제 4/6/8/16-bit 양자화 + binary packet bundling)를 써서,
  SKEM/fixed keyframe 선택 × 채널 bit-depth 조합의 **전송 bundle byte 수 대 전체
  영상 화질(PSNR/SSIM/LPIPS)** trade-off를 10개 영상 전체에서 실측한 실험이다.
  `docs/current/status.md`의 "전송량 절감 → 실제 binary packet 전송" 항목의 근거.

## Config·명령

```bash
python scripts/run_transmission_reduction_eval.py \
    --configs fixed_awgn,fixed_int16,fixed_int8,fixed_int6,fixed_int4,skem_int16,skem_int8,skem_int6,skem_int4 \
    --device cuda:0 --output-root outputs/transmission_reduction_full_20260818_043425
```

- `--configs`는 `{selector}_{channel}` 조합이다: `selector ∈ {fixed, skem}`
  (keyframe 선택 방식), `channel ∈ {awgn, int16, int8, int6, int4}`
  (`awgn`=analog 기존 경로, `intN`=`channels/digital_packet.py`의 실제 N-bit
  양자화 + binary packet).
- keyframe 선택은 scene-change 병용(`--use-scene-detector`), SKEM은 이 실행에서
  `--psss-backend proxy`(CLIP 텍스트 유사도 근사 — **real PSSS 아님**, PSSS 정의는
  [architecture/tx_rx_contract.md](../architecture/tx_rx_contract.md) §5.1 참고).
- quality gate: PSNR drop ≤ 0.5dB, SSIM drop ≤ 0.01, LPIPS rise ≤ 0.02. 원칙상
  reliable-digital 기준을 사용하지만, 이 실행에서는 아래 제한 때문에 `fixed_awgn`이
  임시 기준으로 사용됐다.

## 결과 위치

```
outputs/transmission_reduction_full_20260818_043425/
  aggregate.csv            9개 config × 10영상 평균 (PSNR/SSIM/LPIPS + bundle bytes)
  pareto_frontier.csv       quality gate를 만족하는 가장 작은 config
  per_video_metrics.csv     영상별 전체 프레임(키프레임만이 아님) 화질 + 정확한 전송 bundle byte
  packet_components.csv     프레임별 정확한 bundle byte 분해(caption/edge/visual/manifest)
  packets/<video>/<config>/frame_*.sgbundle   실제 직렬화된 전송 bundle
  recon_videos/<video>/<config>/recon.mp4     전체 복원 영상
  summary.json              실행 설정 + 선택된 config + baseline
  README.md                 이 실행이 생성한 자기 설명 리포트(컬럼 정의, 알려진 한계)
```

## 핵심 결과 (10개 영상, 영상당 100프레임)

- `aggregate.csv`의 `mean_total_bundle_bytes`는 **프레임당 값이 아니라 영상당 총
  bundle byte의 10개 영상 평균**이다. 아래 프레임당 값은 이를 100으로 나눈 단순
  환산값이다.

| config | bit depth | PSNR | SSIM | LPIPS | exact bundle bytes/video | bytes/frame | valid ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_awgn`(임시 품질 기준) | analog | 23.34 | 0.7292 | 0.2562 | N/A | N/A | 1.000 |
| `fixed_int16` | 16 | 23.83 | 0.7348 | 0.2361 | 2,802,712 | 28,027 | 0.948 |
| `fixed_int8` | 8 | 23.83 | 0.7352 | 0.2364 | 2,530,637 | 25,306 | 0.930 |
| `fixed_int6` | 6 | 23.71 | 0.7363 | 0.2402 | 2,462,644 | 24,626 | 0.964 |
| `fixed_int4` | 4 | 23.47 | 0.7335 | 0.2519 | 2,394,650 | 23,947 | 1.000 |
| `skem_int16` | 16 | 23.83 | 0.7348 | 0.2361 | 2,802,612 | 28,026 | 0.948 |
| `skem_int8` | 8 | 23.83 | 0.7352 | 0.2364 | 2,530,537 | 25,305 | 0.930 |
| `skem_int6` | 6 | 23.71 | 0.7363 | 0.2402 | 2,462,544 | 24,625 | 0.964 |
| **`skem_int4`**(잠정 후보) | 4 | 23.47 | 0.7335 | 0.2519 | 2,394,550 | 23,946 | 1.000 |

- `fixed_awgn`의 `mean_total_bundle_bytes=2,251,907`은 analog visual waveform을
  포함하지 않은 digital side-information만의 값이므로 디지털 bundle과 직접 비교하지
  않는다. `skem_int4`는 이 analog AWGN 임시 기준 대비 ΔPSNR −0.13dB / ΔSSIM
  −0.004 / ΔLPIPS −0.004로 픽셀 품질 gate를 통과한 가장 작은 **잠정 후보**다.
  `fixed_int4`와 거의 같은 결과이므로 이 sweep만으로 SKEM의 이점도 확인되지 않았다.

## 읽을 때 주의할 점 (README.md와 동일한 경고)

- **baseline이 analog(`fixed_awgn`)다.** `fixed_int16`은 실행됐지만 52개 NaN/Inf
  프레임 때문에 reliable-digital 기준 자격을 잃었다. float32 digital config는
  실행하지 않아 `fixed_awgn`이 임시 품질 기준으로 fallback됐다. 따라서 양자화
  손실과 AWGN 잡음이 섞인 비교이며, reliable-digital 기준 재실행이 필요하다.
- **`int8`/`int16`은 valid_frame_ratio가 1.0 미만이다** — GPU 검증 중 발견된
  기존 버그(이 기능 자체의 버그 아님): blind SNR 예측기(`jscc.snr_prediction_net`,
  `pipelines/infer_pipeline.py`, AWGN 왜곡으로만 학습됨)가 거친 양자화 입력에서
  signal scale ≥ 1을 예측해 `log10`이 비정상 입력을 받아 NaN이 전파되는 프레임이
  일부 있다(`int16`: 52/1000, `int8`: 70/1000, `int6`: 36/1000). `int4`/`awgn`은
  0건이다. 이런 config는 quality gate를 자동으로 통과하지 못하고 Pareto 후보에서
  제외된다 — `n_nan_or_inf_frames`/`valid_frame_ratio`로 명시된다.
- **`psss_backend_kind: proxy`** — 이 실행의 SKEM 선택은 CLIP 텍스트 유사도
  근사이지 실제 MLLM PSSS가 아니다. `real` backend로 재실행하면 SKEM/fixed
  차이가 달라질 수 있다(미검증).
- **`estimated_digital_channel_symbols`/`estimated_wire_bytes`는 라벨링된 proxy**다
  (`--bits-per-symbol` 미지정) — 실제 변조기/FEC coder는 이 저장소에 없다.
  반면 `total_bundle_bytes`(패킷 byte 수)는 실제 직렬화 byte 수로 **정확한 값**이다.
- **의미 신뢰도는 아직 비교하지 않았다.** 이 실행은 PSNR/SSIM/LPIPS만 사용했다.
  SRS, 누락/추가 객체율, hallucination score를 포함한 재검증 전에는 `int4`를 기본
  operating point로 확정하지 않는다.

## 관련 문서
- [current/status.md](../current/status.md) — 전송량 절감 현재 상태 요약
- [current/roadmap.md](../current/roadmap.md) — importance-aware allocation 등 후속 계획
- [architecture/tx_rx_contract.md](../architecture/tx_rx_contract.md) — 채널/전송 설계

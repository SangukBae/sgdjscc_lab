---
status: frozen
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
experiment_commit: unknown
documentation_commit: ec367bb
supersedes:
---

> [← 문서 색인](../README.md)

# Packet 전송량 Pareto Sweep

## 설정

- 영상: 10개
- frame: 영상당 100개
- selector
  - fixed
  - SKEM proxy
- bit depth
  - 16, 8, 6, 4
- quality gate
  - PSNR 감소 ≤ 0.5dB
  - SSIM 감소 ≤ 0.01
  - LPIPS 증가 ≤ 0.02

## 실행

```bash
python scripts/run_transmission_reduction_eval.py \
    --configs fixed_awgn,fixed_int16,fixed_int8,fixed_int6,fixed_int4,skem_int16,skem_int8,skem_int6,skem_int4 \
    --device cuda:0 \
    --output-root outputs/transmission_reduction_full_20260818_043425
```

## 결과

| config | PSNR | SSIM | LPIPS | bytes/video | bytes/frame | valid ratio |
|---|---:|---:|---:|---:|---:|---:|
| fixed int16 | 23.83 | 0.7348 | 0.2361 | 2,802,712 | 28,027 | 0.948 |
| fixed int8 | 23.83 | 0.7352 | 0.2364 | 2,530,637 | 25,306 | 0.930 |
| fixed int6 | 23.71 | 0.7363 | 0.2402 | 2,462,644 | 24,626 | 0.964 |
| fixed int4 | 23.47 | 0.7335 | 0.2519 | 2,394,650 | 23,947 | 1.000 |
| SKEM int4 | 23.47 | 0.7335 | 0.2519 | 2,394,550 | 23,946 | 1.000 |

## 잠정 판정

- 후보: `skem_int4`
- AWGN 임시 기준 대비
  - ΔPSNR: -0.13dB
  - ΔSSIM: -0.004
  - ΔLPIPS: -0.004
- SKEM 효과
  - fixed int4와 차이 미미
  - 이 sweep에서 이점 미확인

## 해석 제한

- baseline
  - int16: NaN/Inf 52프레임
  - float32: 미실행
  - AWGN: 임시 품질 기준
- AWGN byte
  - visual waveform 미포함
  - digital side-information만 포함
  - digital bundle과 직접 비교 금지
- SKEM
  - CLIP proxy
  - real MLLM 아님
- 미평가
  - SRS
  - missing/additional object
  - hallucination score
- 결론
  - int4 기본 operating point 확정 금지
  - reliable-digital·의미 평가 후 재판정

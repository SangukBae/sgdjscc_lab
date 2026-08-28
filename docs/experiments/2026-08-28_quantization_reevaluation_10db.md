---
status: frozen
updated: 2026-08-28
owner: ETRI SGD-JSCC 연구팀
experiment_commit: 6d6c4ed60c05f91794bc5a2b9b71e18c022b1521
documentation_commit: unknown
supersedes:
---

> [← 문서 색인](../README.md)

# 10 dB fixed-selector 양자화 재평가

## 범위와 실행

- 원본: `outputs/transmission_normalization_parallel_20260827_225722/`
- 보존: [`results/quantization_reevaluation_10db_20260828/`](../../results/quantization_reevaluation_10db_20260828/README.md)
- 데이터: ETRI 10영상 × 100프레임
- 설정: `fixed_awgn`, `fixed_float32`, `fixed_int16`, `fixed_int8`,
  `fixed_int6`, `fixed_int4`
- 고정값: seed 2025, `fixed_reference`, 10 dB
- GPU: RTX 4090 3장, worker별 `cuda:0`/`cuda:1`/`cuda:2`
- 실행 시간: 약 1시간 20분
- 범위 제외: SKEM, fixed–SKEM rate matching

## 완전성·안전성

- video-config pair: 60/60, 누락·중복·예상 외 pair 0
- worker return code: 모두 0
- 실패 pair와 NaN/Inf: 모두 0
- 모든 설정: 10영상, 1,000 quality frame, `valid_frame_ratio=1.0`
- 원격→로컬 전체 원본: 13,144파일, 1,210,877,488 bytes, 전 파일
  SHA-256 일치
- 실험 commit: `6d6c4ed`; 컨테이너에 git 바이너리가 없어 manifest dirty는
  `unknown`이지만, 원격 호스트의 같은 bind-mounted checkout은 tracked clean으로
  확인했다.

## 결과

| 설정 | bytes/frame | byte 절감 | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | 판정 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fixed float32 | 33,468.605 | 기준 | 23.5289 | — | 0.735307 | — | 0.254106 | — | digital baseline |
| fixed int16 | 28,027.117 | 16.26% | 23.5289 | +0.000002 | 0.735307 | +0.000000002 | 0.254106 | -0.000000380 | 통과 |
| fixed int8 | 25,306.373 | 24.39% | 23.5282 | -0.000716 | 0.735302 | -0.00000490 | 0.254105 | -0.00000134 | 통과 |
| fixed int6 | 24,626.437 | 26.42% | 23.5267 | -0.002218 | 0.735198 | -0.000108 | 0.254052 | -0.0000541 | 통과 |
| fixed int4 | 23,946.501 | 28.45% | 23.4763 | -0.052631 | 0.733301 | -0.002006 | 0.253619 | -0.000487 | **통과·선정** |
| fixed AWGN (참고) | 22,519.067 | 비교 제외 | 23.3521 | — | 0.729407 | — | 0.256220 | — | Pareto 제외 |

품질 허용 기준은 PSNR 하락 ≤ 0.5 dB, SSIM 하락 ≤ 0.01, LPIPS 증가
≤ 0.02다. 4개 정수 bit-depth가 모두 통과하므로 **최소 통과 bit-depth는
4-bit**다.

## effect·Pareto 검증

- `quantization_effect.csv`의 delta와 byte ratio를 `aggregate.csv`에서 독립
  계산했으며 전 항목이 절대 오차 1e-12 이내로 일치했다.
- `pareto_frontier.csv`: int4→int6→int8→int16 4행, 모든 baseline이
  `fixed_float32`, AWGN 행 없음.
- `summary.json`: `baseline_is_analog=false`, `baseline_valid=true`, 선택 행은
  `fixed_int4`.
- `fixed_int4` bundle 중 edge+uncertainty 90.90%, visual 5.89%다. 이후 추가
  절감은 visual bit-depth보다 edge/uncertainty 표현을 우선해야 한다.

## 10 dB 기록 해석

- plan, worker signature, resolved config와 digital CSV 행은 모두 10 dB를 기록한다.
- `fixed_awgn` CSV의 `fixed_reference_snr_db`는 공란이다. 이 필드는 digital
  decoder-step reference용이며 AWGN 실제 채널 SNR은 resolved config의
  `snr_db=10.0`으로 확인된다.
- 원본 CSV를 사후 수정하지 않았고 이 스키마 차이를 보존 README와
  `validation_report.json`에 명시했다.

## 판정과 다음 작업

1. fixed selector 양자화 operating point는 `fixed_int4`로 확정한다.
2. `fixed_awgn`은 참고 기준으로만 유지하고 digital Pareto 비교에는 사용하지 않는다.
3. 다음 작업은 실제 transmitting frame과 byte를 맞춘 fixed–SKEM exact
   matched-rate 재평가다.
4. 그 다음은 packet의 90.90%를 차지하는 edge·uncertainty 선택 전송·양자화·
   해상도 축소 ablation이다.

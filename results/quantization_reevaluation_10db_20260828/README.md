# 10 dB fixed-selector 양자화 재평가 (2026-08-28)

> **보존 사본.** 원본은 Git 비추적 경로
> `outputs/transmission_normalization_parallel_20260827_225722/`에 있다. 원격 GPU
> 서버의 전체 원본 13,144개 파일을 로컬로 회수한 뒤 파일 수, 총 byte, 파일별
> SHA-256을 비교했으며 모두 일치했다. 이 디렉터리에는 재현·판정에 필요한 핵심
> CSV·JSON·manifest만 보존한다.

## 결론

- 10영상 × 6설정의 **60/60 video-config pair가 모두 완료**됐고 실패와
  NaN/Inf는 0건이다.
- reliable-digital 기준은 `fixed_float32`다. `fixed_awgn`은 참고 기준일 뿐
  quantization/Pareto baseline으로 사용하지 않았다.
- int16/int8/int6/int4가 모두 PSNR 하락 ≤ 0.5 dB, SSIM 하락 ≤ 0.01,
  LPIPS 증가 ≤ 0.02를 만족했다.
- 따라서 허용 기준을 만족하는 **최소 bit-depth는 4-bit**이며, 이번 범위의
  operating point는 `fixed_int4`다.
- `fixed_int4`는 float32보다 전송 byte를 28.45% 줄이면서 PSNR 0.0526 dB,
  SSIM 0.00201만 낮아졌고 LPIPS는 0.000487 낮아졌다(음의 rise는 개선 방향).

## 집계 결과

| 설정 | bytes/video | bytes/frame | PSNR | SSIM | LPIPS | float32 대비 byte 절감 | 허용 기준 |
|---|---:|---:|---:|---:|---:|---:|---|
| fixed AWGN (참고) | 2,251,906.7 | 22,519.067 | 23.3521 | 0.729407 | 0.256220 | 비교 제외 | Pareto 제외 |
| fixed float32 | 3,346,860.5 | 33,468.605 | 23.5289 | 0.735307 | 0.254106 | 기준 | baseline |
| fixed int16 | 2,802,711.7 | 28,027.117 | 23.5289 | 0.735307 | 0.254106 | 16.26% | 통과 |
| fixed int8 | 2,530,637.3 | 25,306.373 | 23.5282 | 0.735302 | 0.254105 | 24.39% | 통과 |
| fixed int6 | 2,462,643.7 | 24,626.437 | 23.5267 | 0.735198 | 0.254052 | 26.42% | 통과 |
| fixed int4 | 2,394,650.1 | 23,946.501 | 23.4763 | 0.733301 | 0.253619 | 28.45% | **통과·선정** |

`quantization_effect.csv`의 float32 대비 delta와 byte ratio를 `aggregate.csv`에서
독립 재계산해 전 항목이 절대 오차 1e-12 이내로 일치함을 확인했다.
`pareto_frontier.csv`는 int4, int6, int8, int16 순으로 4행이며 모두
`baseline_config=fixed_float32`다. AWGN 행은 없고 int4만
`selected_as_smallest_in_budget=True`다.

## 완전성·finite·GPU provenance

- `per_video_metrics.csv`: 60행, unique pair 60개, 누락·중복·예상 외 pair 0개
- `failed_pairs.csv`: 0 byte, 실패 pair 0개
- quality coverage: 모든 설정 `valid_frame_ratio=1.0`, 총 6,000 quality frame
- per-video/aggregate NaN·Inf: 0개, 모든 PSNR/SSIM/LPIPS/byte 값 finite
- worker 종료: worker_00/01/02 모두 return code 0
- 물리 GPU 매핑:
  - worker_00 → `cuda:0` → 4영상
  - worker_01 → `cuda:1` → 3영상
  - worker_02 → `cuda:2` → 3영상
- 각 worker manifest: NVIDIA GeForce RTX 4090, Python 3.9.25,
  torch 2.1.0+cu118, CUDA 11.8
- 실험 commit: `6d6c4ed60c05f91794bc5a2b9b71e18c022b1521`, seed 2025
- 실행 시간: 약 1시간 20분(계획 파일 2026-08-27 22:57:30 UTC부터 최상위
  manifest 생성 2026-08-28 00:17:35 UTC까지)

컨테이너 내부에는 `git` 실행 파일이 없어 run manifest의 `git.dirty`는
`unknown`이다. 같은 bind-mounted 작업 경로를 원격 호스트의 Git으로 확인했을
때 HEAD와 origin/main은 위 commit으로 같았고 tracked 변경은 없었다. 이 확인은
원본 manifest를 사후 수정하지 않고 `validation_report.json`에 별도로 기록했다.

## 10 dB 계약 검증

- `parallel_plan.json`: `fixed_reference_snr_db=10.0`
- worker 3개 `run_signature.json`: 모두 `fixed_reference_snr_db=10.0`
- worker 3개 resolved config: `snr_db=10.0` 및
  `digital_fixed_reference_snr_db=10.0`
- digital 5설정의 per-video/aggregate/quantization/Pareto CSV:
  `digital_step_policy=fixed_reference`, `fixed_reference_snr_db=10.0`
- `fixed_awgn`의 CSV `fixed_reference_snr_db`는 공란이다. 이 열은 digital
  decoder-step reference 전용이라 AWGN에는 적용하지 않았기 때문이다. 실제 AWGN
  채널 SNR은 각 worker resolved config의 `snr_db=10.0`으로 확인했다.

따라서 실제 여섯 설정 모두 10 dB 계약으로 실행됐지만, “모든 CSV 행의 같은 열에
10.0이 있어야 한다”는 형식 기준으로는 AWGN 행이 해당하지 않는다. 이 공란을
10.0으로 사후 채우지 않아 원본 측정 파일의 무결성을 유지했다.

## 전송 구성과 다음 작업

`fixed_int4`의 총 bundle byte 중 edge + uncertainty가 90.90%, visual latent가
5.89%다(`packet_components.csv` 총합 독립 계산). 4-bit까지 품질 허용 기준을
통과했지만 visual bit-depth를 더 낮춰 얻을 수 있는 추가 절감은 작다. 다음 우선순위는
계획대로 실제 transmitting frame/byte 기준 fixed–SKEM exact matched-rate 재평가이며,
그 다음은 edge·uncertainty의 선택 전송·양자화·해상도 축소 ablation이다.

## 보존 파일

- 집계·판정: `aggregate.csv`, `quantization_effect.csv`,
  `pareto_frontier.csv`, `summary.json`, `validation_report.json`
- 원자료 표: `per_video_metrics.csv`, `packet_components.csv`,
  `quantization_diagnostics.csv`, `keyframe_selection.csv`,
  `source_size_report.csv`, `failed_pairs.csv`
- 범위 제외 증거: 빈 `rate_matching.csv`, `selector_effect.csv`
- provenance: `parallel_plan.json`, `parallel_worker_status.json`,
  `manifest.json`, worker별 `run_manifest.json`/`run_signature.json`,
  `config_source/composed.yaml`
- 원본 자동 README: `README.original.md`
- 해시: `checksums.sha256`

`manifest.json`은 원본 `run_manifest.json`을, `README.original.md`는 원본
`README.md`를 이름만 바꿔 byte-identical하게 보존했다. `README.md`와
`validation_report.json`은 이번 검증에서 새로 작성한 파일이다.

## 원본 회수·체크섬

- 원격 파일: 13,144개 / 1,210,877,488 bytes
- 로컬 파일: 13,144개 / 1,210,877,488 bytes
- 원격·로컬 전체 파일 SHA-256: 전부 일치
- 이 보존 디렉터리의 `checksums.sha256`: 자체 파일을 제외한 모든 보존 파일의
  SHA-256을 기록

## 관련 문서

- [실험 해석](../../docs/experiments/2026-08-28_quantization_reevaluation_10db.md)
- [현재 상태](../../docs/current/status.md)
- [로드맵](../../docs/current/roadmap.md)
- [열린 이슈](../../docs/current/open_issues.md)

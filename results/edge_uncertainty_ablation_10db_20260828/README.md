# edge·uncertainty 전송 절감 ablation (2026-08-28)

> **보존 사본.** 원본은 Git 비추적 경로
> `outputs/edge_uncertainty_ablation_10db_20260828_051915/`에 있다. 원격 GPU
> 서버의 전체 원본을 로컬로 회수한 뒤 파일 수, 총 byte, 파일별 SHA-256을
> 비교했으며 모두 일치했다. 이 디렉터리에는 재현·판정에 필요한 핵심
> CSV·JSON·manifest만 보존한다.

## 결론

- 10영상 × 16 profile의 **160/160 video-profile pair가 모두 완료**됐고 실패와
  NaN/Inf는 0건이다. 세 worker도 모두 return code 0으로 종료했다.
- baseline `fixed_int4` packet은 평균 2,396,632.7 bytes/video이며 edge와
  uncertainty가 각각 45.41%, 합계 **90.83%**를 차지했다.
- 시험한 profile은 모두 품질 gate를 통과했다. 그중 `combined_ds4`가 평균
  **356,824.7 bytes/video**, baseline 대비 **85.11% 절감**으로 최소였다.
  품질 변화는 PSNR -0.0000098dB, SSIM -0.000000115, LPIPS -0.000000564다.
- 그러나 `combined_ds4`는 **시험한 16개 profile 안의 조건부 후보**다. 모든
  uncertainty-only profile은 10영상 각각에서 baseline과 세 품질 지표가 정확히
  같았다. 코드 대조 결과 reliable-digital 수신기는 edge가 이미 수신됐으므로
  analog Canny 재전송을 건너뛰며, uncertainty는 그 재전송 함수에서만 소비된다.
  즉 이 경로에서 uncertainty payload는 복원에 영향을 주지 않는다.
- edge는 ControlNet latent로 연결되지만 `edge_omit`조차 영상별 최대 차이가
  PSNR 0.0000747dB, SSIM 0.000000444, LPIPS 0.00000252에 불과했다. 따라서 현재
  checkpoint/config에서 구조 가이드 영향이 매우 약하다는 경고 신호로 해석한다.
- 이번 profile 집합에는 `edge_ds4 + uncertainty_omit`과
  `edge_omit + uncertainty_omit`이 없다. 또한 SRS·할루시네이션·시간축 지표를
  측정하지 않았으므로 최종 운영점 확정은 보류한다.

## 완전성·provenance

- `per_video_metrics.csv`: 160행, unique video-profile pair 160개
- `packet_components.csv`: 16,000행, 모든 component 합이 bundle byte와 일치
- quality coverage: 총 16,000 frame, 모든 profile `valid_frame_ratio=1.0`
- fixed-reference SNR: 전 row 10dB
- worker provenance:
  - worker_00 → physical `cuda:0` → 4영상
  - worker_01 → physical `cuda:1` → 3영상
  - worker_02 → physical `cuda:2` → 3영상
- GPU/환경: NVIDIA GeForce RTX 4090, Python 3.9.25, torch 2.1.0+cu118,
  CUDA 11.8
- 실험 commit: `1d33fa312d0f6c540a319e791102f98b5f0118b7`, tracked dirty false,
  seed 2025
- 실행 시간: 2026-08-28 14:19:31~17:43:58 KST, 약 3시간 24분

## profile별 결과

| profile | bytes/video | 절감률 | PSNR 하락 | SSIM 하락 | LPIPS 증가 |
|---|---:|---:|---:|---:|---:|
| baseline | 2,396,632.7 | 기준 | 0 | 0 | 0 |
| combined_ds4 | **356,824.7** | **85.11%** | 0.0000098 | 0.000000115 | -0.000000564 |
| combined_q4_ds2_reuse2 | 370,956.1 | 84.52% | 0.0000047 | 0.000000074 | -0.000000233 |
| combined_ds2 | 764,786.3 | 68.09% | 0.0000076 | 0.000000084 | -0.000000415 |
| uncertainty_omit | 1,307,996.4 | 45.42% | 0 | 0 | 0 |
| edge_omit | 1,308,096.0 | 45.42% | 0.0000074 | 0.000000081 | 0.000000089 |
| combined_q4 | 1,308,735.1 | 45.39% | 0.0000011 | 0.000000121 | -0.000000319 |
| edge_ds4 / uncertainty_ds4 | 1,376,728.7 | 42.56% | edge만 미세 변화 | edge만 미세 변화 | edge만 미세 변화 |
| combined_reuse2 | 1,426,085.7 | 40.50% | 0.0000059 | -0.000000079 | -0.000000374 |
| edge_ds2 / uncertainty_ds2 | 1,580,709.5 | 34.04% | edge만 미세 변화 | edge만 미세 변화 | edge만 미세 변화 |
| edge_q4 / uncertainty_q4 | 1,852,683.9 | 22.70% | edge만 미세 변화 | edge만 미세 변화 | edge만 미세 변화 |
| uncertainty_reuse2 | 1,911,337.0 | 20.25% | 0 | 0 | 0 |
| edge_reuse2 | 1,911,381.4 | 20.25% | 0.0000059 | -0.000000079 | -0.000000374 |

정확한 원시 수치는 `guide_ablation_effect.csv`를 기준으로 한다. 표에서 같은 byte를
공유하는 edge/uncertainty profile의 품질 변화는 edge 조작에만 나타났고 uncertainty
조작은 baseline과 정확히 같다.

## packet 구성

baseline 10영상 합계 23,966,327 bytes의 구성은 다음과 같다.

| 성분 | 합계 byte | 비율 |
|---|---:|---:|
| edge | 10,884,039 | 45.41% |
| uncertainty | 10,884,039 | 45.41% |
| visual latent | 1,410,336 | 5.88% |
| semantic packet | 507,439 | 2.12% |
| manifest | 160,503 | 0.67% |
| caption | 32,067 | 0.13% |
| bundle overhead | 87,904 | 0.37% |

`combined_ds4`는 edge와 uncertainty를 각각 684,999 bytes로 낮췄고 나머지
payload는 유지했다. `combined_q4_ds2_reuse2`는 각 guide 756,470 bytes이며,
성분별 83회 중 46회 전송·37회 receiver cache 재사용이다.

## 해석과 다음 검증

1. reliable-digital packet에서는 사용되지 않는 uncertainty를 기본 payload에서
   제거할지, 또는 uncertainty를 실제 decoder 조건화에 연결할지 설계를 먼저 정한다.
2. 최소한 `edge_ds4 + uncertainty_omit`과 `edge_omit + uncertainty_omit`을 추가해
   현재 16-profile 탐색의 빈 조합을 닫는다.
3. edge 영향이 실제로 약한지 ControlNet off/scale 비교와 SRS·할루시네이션·시간축
   지표로 확인한다.
4. 이 검증을 통과한 후보만 VAE-direct/few-step/full diffusion 통합 평가의
   전송 operating point로 사용한다.

## 원본 회수·체크섬

- 원격 파일: 33,256개 / 2,813,319,377 bytes
- 로컬 파일: 33,256개 / 2,813,319,377 bytes
- 원격·로컬 전체 파일 SHA-256: 전부 일치
- 최초 rsync에서 root 소유 `0600` 파일 56개가 누락됐다. 해당 run 디렉터리에
  읽기 권한만 추가한 후 증분 rsync했고, 최종 전 파일 checksum으로 일치를 확인했다.
- 보존한 원본 artifact 23개도 원본과 개별 SHA-256이 모두 일치한다.
- 이 보존 디렉터리의 `checksums.sha256`은 자체 파일을 제외한 모든 보존 파일의
  SHA-256을 기록한다.

## 보존 파일

- 자동 검증·판정: `guide_ablation_validation.json`,
  `GUIDE_ABLATION_REPORT.md`, `validation_report.json`
- 품질·Pareto: `aggregate.csv`, `per_video_metrics.csv`,
  `guide_ablation_effect.csv`, `guide_pareto_frontier.csv`
- packet 구성: `guide_component_bytes.csv`, `packet_components.csv`
- provenance: `parallel_plan.json`, `parallel_worker_status.json`, `manifest.json`,
  worker별 `run_manifest.json`/`run_signature.json`, `config_source/composed.yaml`
- 원본 자동 README: `README.original.md`

## 관련 문서

- [실험 해석](../../docs/experiments/2026-08-28_edge_uncertainty_ablation_10db.md)
- [현재 상태](../../docs/current/status.md)
- [로드맵](../../docs/current/roadmap.md)
- [열린 이슈](../../docs/current/open_issues.md)

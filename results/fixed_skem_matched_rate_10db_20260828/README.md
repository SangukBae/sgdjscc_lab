# fixed–SKEM exact matched-rate 재평가 (2026-08-28)

> **보존 사본.** 원본은 Git 비추적 경로
> `outputs/fixed_skem_matched_rate_10db_20260828_011620/`에 있다. 원격 GPU
> 서버의 전체 원본을 로컬로 회수한 뒤 파일 수, 총 byte, 파일별 SHA-256을
> 비교했으며 모두 일치했다. 이 디렉터리에는 재현·판정에 필요한 핵심
> CSV·JSON·manifest만 보존한다.

## 결론

- 10영상 × 10설정의 **100/100 video-config pair가 모두 완료**됐고 실패와
  NaN/Inf는 0건이다. 3개 worker도 모두 return code 0으로 종료했다.
- 10영상 × 5 digital channel의 **50/50 fixed–SKEM 비교가 실제 visual 전송
  프레임 수 exact match**, raw bundle byte 차이 1% 이내, padding 후 effective
  byte exact match를 모두 만족했다.
- 그러나 보정된 proxy SKEM은 **10/10 영상에서 fixed와 keyframe index 및 실제
  transmitting index가 완전히 동일**했다. 모든 bit-depth에서 PSNR·SSIM·LPIPS
  차이가 정확히 0인 이유도 동일한 스케줄·seed·decoder 조건 때문이다.
- 따라서 이번 결과는 “SKEM이 동일 rate에서 우수하다”는 근거가 아니다.
  **현재 proxy PSSS/SKEM이 exact-rate 제약 아래 fixed selector로 수렴했다는
  퇴화된 null 결과**다.
- bit-depth 결과는 이전 fixed-only 재평가와 일치한다. float32 대비 int4는 byte
  28.45% 절감, PSNR -0.0526dB, SSIM -0.00201, LPIPS -0.000487로 허용 기준을
  통과한다. selector 이득이 없으므로 운영점은 계속 `fixed_int4`다.

## 완전성·검증

- `per_video_metrics.csv`: 100행, unique pair 100개
- `matched_rate_plan.csv`: 10행, 영상별 exact plan 10개
- `rate_matching.csv`: 50행, `rate_matched=True` 50/50
- `failed_pairs.csv`: 0 byte, 실패 pair 0개
- quality coverage: 총 10,000 frame, 모든 config `valid_frame_ratio=1.0`
- 자동 검증: `matched_rate_validation.json`의 14개 check 전부 통과
- worker provenance:
  - worker_00 → physical `cuda:0` → 4영상
  - worker_01 → physical `cuda:1` → 3영상
  - worker_02 → physical `cuda:2` → 3영상
- GPU/환경: NVIDIA GeForce RTX 4090, Python 3.9.25, torch 2.1.0+cu118,
  CUDA 11.8
- 실험 commit: `2218824f301ad238129a72ade42b0210be2f21a5`, tracked dirty false,
  seed 2025, fixed-reference 10dB
- 실행 시간: 2026-08-28 10:16:36~12:24:47 KST, 약 2시간 8분

## exact matched-rate 상세

모든 channel에서 fixed와 SKEM의 실제 transmitting frame 수는 영상별로 같다.
전체 영상 합계는 selector당 channel별 83 frame이다. 계획 index와 full pipeline의
실제 `FrameRecord.decision` index도 완전히 일치했다.

| Channel | 평균 raw byte 차이 | 최대 raw byte 차이 | SKEM padding 합계 | effective byte |
|---|---:|---:|---:|---|
| float32 | 0.003239% | 0.003544% | 1,000 | exact |
| int16 | 0.003866% | 0.004232% | 1,000 | exact |
| int8 | 0.004280% | 0.004687% | 1,000 | exact |
| int6 | 0.004398% | 0.004817% | 1,000 | exact |
| int4 | 0.004522% | 0.004953% | 1,000 | exact |

raw fixed bundle은 모든 영상×channel에서 SKEM보다 정확히 100 byte 크다. 구성요소를
독립 합산하면 차이는 100 frame의 `manifest_bytes`에서만 frame당 1 byte 발생한다.
caption, edge, uncertainty, semantic packet, visual payload, bundle overhead 차이는 모두
0이다. 이는 `fixed_*`와 `skem_*` config label 길이 차이에 해당하는 실험 metadata
효과이며 selector의 실질 전송률 이득으로 해석하면 안 된다. SKEM 쪽에 영상당 100
byte를 padding해 effective rate를 같게 만들면 전 channel에서 완전 동률이다.

## selector 계획과 퇴화 판정

- 9영상: selected proxy PSSS threshold 0.35, max segment 16
- `09_scene_cut_chair_car`: threshold -0.65, max segment 16
- 영상당 440개 후보 평가, actual-transmission count exact 후보 7~40개
- 최종 선택 결과:
  - fixed/SKEM keyframe index 동일: 10/10
  - fixed/SKEM actual transmitting index 동일: 10/10
  - 최대 paired PSNR/SSIM/LPIPS 차이: 각각 0

`pareto_frontier.csv`는 raw metadata byte가 100 byte 작은 `skem_int4`를 최소 후보로
표시하지만, 이는 위 config-label 길이 차이뿐이다. padding 후 `fixed_int4`와
`skem_int4`는 전송량과 품질이 모두 같으므로 자동 Pareto의 SKEM 선택을 실제 selector
우위로 사용하지 않는다.

공정 비교 자체는 완료됐지만 semantic selector의 효용을 추가로 연구하려면 다음 중
하나가 필요하다.

1. exact-count 후보 중 fixed와 다른 schedule만 허용하고 동일 byte budget에서 비교
2. 실제 MLLM PSSS(`real` backend)로 selector 품질 재검증

현재 proxy 결과만 기준으로는 SKEM을 운영점에 채택할 근거가 없다.

## bit-depth 결과

fixed와 SKEM의 schedule/품질이 같으므로 아래는 fixed 행을 대표로 기록한다.

| 설정 | bytes/video | PSNR | SSIM | LPIPS | float32 대비 byte 절감 |
|---|---:|---:|---:|---:|---:|
| float32 | 3,346,860.5 | 23.5289 | 0.735307 | 0.254106 | 기준 |
| int16 | 2,802,711.7 | 23.5289 | 0.735307 | 0.254106 | 16.26% |
| int8 | 2,530,637.3 | 23.5282 | 0.735302 | 0.254105 | 24.39% |
| int6 | 2,462,643.7 | 23.5267 | 0.735198 | 0.254052 | 26.42% |
| int4 | 2,394,650.1 | 23.4763 | 0.733301 | 0.253619 | **28.45%** |

`fixed_int4` packet에서 edge + uncertainty는 90.90%, visual latent는 5.89%다.
따라서 다음 전송량 절감 작업은 계획대로 edge·uncertainty의 선택 전송, 양자화,
해상도 축소 ablation이 우선이다.

## 원본 회수·체크섬

- 원격 파일: 21,194개 / 1,926,849,641 bytes
- 로컬 파일: 21,194개 / 1,926,849,641 bytes
- 원격·로컬 전체 파일 SHA-256: 전부 일치
- 최초 rsync에서 root 소유 `0600` 표 47개가 host 계정에 거부됐다. 해당 run에
  읽기 권한만 추가한 후 증분 rsync했고, 최종 전 파일 checksum으로 내용 일치를
  확인했다.
- 이 보존 디렉터리의 `checksums.sha256`은 자체 파일을 제외한 모든 보존 파일의
  SHA-256을 기록한다.

## 보존 파일

- 자동 검증·판정: `matched_rate_validation.json`, `MATCHED_RATE_REPORT.md`,
  `matched_rate_quality_effect.csv`, `validation_report.json`
- selector/rate: `matched_rate_plan.csv`, `rate_matching.csv`,
  `selector_effect.csv`, `keyframe_selection.csv`
- 품질·양자화: `aggregate.csv`, `per_video_metrics.csv`,
  `quantization_effect.csv`, `pareto_frontier.csv`, `quantization_diagnostics.csv`
- packet 구성: `packet_components.csv`
- provenance: `parallel_plan.json`, `parallel_worker_status.json`, `manifest.json`,
  worker별 `run_manifest.json`/`run_signature.json`, `config_source/composed.yaml`
- 원본 자동 README: `README.original.md`

## 관련 문서

- [실험 해석](../../docs/experiments/2026-08-28_fixed_skem_matched_rate_10db.md)
- [현재 상태](../../docs/current/status.md)
- [로드맵](../../docs/current/roadmap.md)
- [열린 이슈](../../docs/current/open_issues.md)

# 전송 정상화 실험 결과 (2026-08-26)

> **보존 사본.** 이 디렉터리는
> `outputs/transmission_normalization_parallel_20260826_093313/`(원본, Git 미포함)의
> 재현·검증에 필요한 핵심 파일만 복사한 것이다. 원본 파일은 바이트 단위로 그대로
> 복사했으며 내용을 수정하지 않았다 — 각 파일의 SHA-256이 원본과 일치함을
> `checksums.sha256`에 기록했다(§ 검증 참고). 원본에 있던 README는
> `README.md.orig`로 이름만 바꿔 원문 그대로 보존했고, 이 파일이 과제 요구사항에
> 맞춘 신규 작성 문서다.

## 목적과 상태

- 목적: `transmission/` 실 binary-packet 경로(4/6/8/16/32-bit 양자화 + fixed/SKEM
  키프레임 선택)의 digital NaN 수정 이후, 3-GPU 병렬 실행으로 10영상 전체를
  정상화 스윕하여 reliable-digital 운영점 후보를 실측 데이터로 판정.
- 상태: **완료(frozen)**. 상세 해석과 남은 과제는
  [docs/experiments/2026-08-26_transmission_normalization.md](../../docs/experiments/2026-08-26_transmission_normalization.md)
  참고.
- **잠정 결과**: 아래 두 가지 미해결 문제 때문에, 이 실행에서 도출한 운영점 후보는
  최종 확정이 아니라 잠정 후보다.
  1. **float32 digital 절대 복원 품질 문제** — digital 경로는 quantization
     bit-depth와 무관하게 AWGN 참고 경로보다 크게 낮은 품질을 보인다(PSNR
     11.32 vs 23.34dB). Tx/Rx edge·ControlNet·`fixed_reference` step 계약을
     분리 점검하기 전에는 "품질 유지"를 주장할 수 없다.
  2. **fixed–SKEM rate matching 불완전** — 키프레임 *개수*는 fixed/SKEM 간에
     맞았지만(`keyframe_count_matched=True`), `recompute_semantic` 경로 때문에
     실제 transmitting frame 수/byte가 달라져 50개 (영상×bit-depth) 비교 중
     35개만 허용 오차(10%) 안에서 일치했다(`rate_matching.csv`). 불일치 영상:
     `02_car_pass`, `07_person_enter`, `09_scene_cut_chair_car`.

## 실행 규모

- 데이터셋: ETRI 10영상 × 100프레임(영상당) = 총 1,000 quality 프레임
- 설정: 11개 = `{fixed, skem} × {float32, int16, int8, int6, int4}` + `fixed_awgn`
  (analog 참고행)
- (영상, 설정) pair: **110/110 완료**, 실패 0건
- 프레임 단위 NaN/Inf: **0건** (`nan_or_failure_counts.total_nan_or_inf_frames = 0`,
  `manifest.json`)
- GPU: RTX 4090 × 3 (`cuda:0`/`cuda:1`/`cuda:2`), worker당 3~4개 영상 분배
  (`parallel_plan.json`)
- 실행 시간: 약 **2시간 22분** — `parallel_plan.json`은 자체 타임스탬프 필드가
  없어 원본 파일 mtime으로 추정: 시작 `2026-08-26 18:33`(`parallel_plan.json`
  mtime) ~ 종료 `2026-08-26 20:55`(`manifest.json` `generated_at` UTC
  11:55:32 = KST 20:55:32, 원본 worker 로그 `worker_NN.log`의 mtime과도 일치).
- 실험 Git commit: `607f72798da24dc3e0c065574efcf6fce90683f3`
  (`manifest.json`의 `git.commit`, `git.dirty: false`로 실행 시점 clean 상태 확인됨)

## 핵심 결과 요약

`quantization_effect.csv` / `pareto_frontier.csv` 기준, `fixed_float32`를
reliable-digital baseline으로 사용:

| 설정 | bytes/frame | float32 대비 절감 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| fixed float32 (baseline) | 35,836.7 | 기준 | 11.318 | 0.0810 | 0.7394 |
| fixed int6 | 26,355.5 | 26.46% | 11.309 | 0.0808 | 0.7400 |
| fixed int4 | 25,626.4 | 28.49% | 11.207 | 0.0765 | 0.7459 |
| skem int4 (`pareto_frontier.csv` rank 0) | 24,505.6 | 31.62% | 11.195 | 0.0760 | 0.7462 |
| *(참고) fixed AWGN — analog, digital 아님* | 24,095.8 | — | 23.342 | 0.7312 | 0.2544 |

- 운영점 판정: `fixed_int6`은 보수적 후보(품질 손실 거의 없음), `fixed_int4`는
  최대 절감 후보. `skem_int4`는 byte 최소 기준 1위이나 위 rate-matching 문제 때문에
  잠정 후보.
- int4 packet 구성 중 edge+uncertainty가 약 91%, visual latent는 약 5.9%
  (`packet_components.csv`) — 추가 절감은 latent bit-depth가 아니라
  edge/uncertainty 압축이 다음 목표.

## 재현성 정보가 있는 위치

| 항목 | 위치 |
|---|---|
| 데이터셋 경로·해시 | `manifest.json` → `dataset.ref` / `dataset.hash`; per-worker `workers/worker_NN/run_manifest.json` → `extra.run_signature.dataset_artifact_sha256`(영상별 processed/captions/gt 해시) |
| 체크포인트 해시 | `workers/worker_NN/run_manifest.json` → `checkpoints.items` (JSCC_model.pth, diffusion_backbone.pth, diffusion_controlnet.pth, muge-epoch-19-checkpoint.pth 각 SHA-256) |
| seed | `manifest.json` → `seed: 2025`; 각 worker manifest에도 동일하게 기록 |
| 실행 환경 | `workers/worker_NN/run_manifest.json` → `environment` (Python 3.9.25, torch 2.1.0+cu118, CUDA 11.8, RTX 4090) |
| 실행 명령 | `workers/worker_NN/run_manifest.json` → `command.text`/`command.argv` (`source: "captured"`, 재구성이 아니라 실제 캡처된 커맨드라인) |
| resolved config (전체 실행 파라미터) | `workers/worker_NN/run_manifest.json` → `resolved_config.resolved.inline` |
| config source (원본 composed.yaml) | `config_source/composed.yaml` (worker_00 기준 — 3 worker 모두 동일 `resolved_config_sha256` 사용, 하단 검증 절 참고) |
| worker→영상 배분 | `parallel_plan.json` |
| 최상위 실행 메타 | `manifest.json` (schema_version 2, `run_id`, `git`, `command`, `accounting` 필드 정의) |

## 원본 대용량 artifact

- 경로: `outputs/transmission_normalization_parallel_20260826_093313/` (이 저장소
  로컬에만 존재, **Git에 포함하지 않음**)
- 이 디렉터리에만 있고 여기에는 복사하지 않은 것: worker별 `packets/`(직렬화
  `.sgbundle`), `recon_frames/`·복원 영상, `per_video_metrics.csv`(프레임 단위 전체
  품질 로그), `quantization_diagnostics.csv`, `keyframe_selection.csv`/
  `keyframe_sweep.csv`, `source_size_report.csv`, `worker_NN.log`(실행 로그) — 모두
  용량이 크거나 이 디렉터리의 집계본(`aggregate.csv`, `packet_components.csv` 등)
  으로 이미 요약되어 있어 재현성 확인에는 불필요하다고 판단해 제외.

## 이 디렉터리의 파일

| 파일 | 내용 | 원본과의 관계 |
|---|---|---|
| `README.md` | 이 문서 | 신규 작성 (원본에 README는 있었으나 과제 요구사항에 맞춰 재작성) |
| `README.md.orig` | 원본 `outputs/.../README.md` | 원본 그대로, 이름만 변경 |
| `summary.json` | 실행 요약(완료 pair, worker 상태, pareto 선정 결과) | 원본과 byte-identical |
| `normalization_effect_summary.json` | quantization/selector effect 파일 존재성 요약 | 원본과 byte-identical |
| `aggregate.csv` | (선택자×채널×bit-depth) 단위 집계 (video 수, PSNR/SSIM/LPIPS, byte) | 원본과 byte-identical |
| `pareto_frontier.csv` | 품질 게이트 통과 config의 byte 최소 랭킹 | 원본과 byte-identical |
| `quantization_effect.csv` | bit-depth 단독 효과 (float32 기준 대비 drop/rise) | 원본과 byte-identical |
| `selector_effect.csv` | fixed vs skem 선택자 효과 | 원본과 byte-identical |
| `rate_matching.csv` | fixed–SKEM byte/keyframe rate-matching 판정 (50행) | 원본과 byte-identical |
| `failed_pairs.csv` | 실패한 (영상, 설정) pair 목록 — 0행 | 원본과 byte-identical (빈 파일) |
| `packet_components.csv` | (영상×설정×프레임) 단위 packet byte 구성 상세(caption/edge/uncertainty/visual/manifest) | 원본과 byte-identical, 전체 사본 (요약본 아님 — 크기가 재현성 확인에 감당 가능한 수준(≈624KB)이라 축약하지 않음) |
| `parallel_plan.json` | 3-GPU worker 배분 계획(영상 목록, 예상 프레임 수, dataset/commit 서명) | 원본과 byte-identical |
| `parallel_worker_status.json` | worker별 종료 코드 | 원본과 byte-identical |
| `manifest.json` | 최상위 실행 manifest (원본 `run_manifest.json`을 이름만 변경해 복사) | 원본 `run_manifest.json`과 byte-identical |
| `config_source/composed.yaml` | 실행에 쓰인 resolved 이전 원본 config (worker_00 사본) | 원본 `workers/worker_00/configs/composed.yaml`과 byte-identical |
| `workers/worker_00,01,02/run_manifest.json` | worker별 상세 manifest(체크포인트 해시, 데이터셋 해시, 환경, 커맨드, resolved config 전체) | 각 원본과 byte-identical |
| `workers/worker_00,01,02/run_signature.json` | worker별 재현성 서명(재개 시 불일치 감지용) | 각 원본과 byte-identical |
| `checksums.sha256` | 위 파일들의 SHA-256 목록(원본 경로와의 매핑 포함) | 이 디렉터리 작업 중 신규 계산 |

## 검증

### 파일 무결성 (SHA-256)

`checksums.sha256`에 기록된 모든 항목을 원본(`outputs/transmission_normalization_parallel_20260826_093313/`)과
재대조 완료 — **12개 최상위 파일 + worker 3개 × 2개 파일(run_manifest.json,
run_signature.json) + config_source/composed.yaml + README.md.orig, 총 20개
전부 일치**. 검증 명령:

```bash
sha256sum outputs/transmission_normalization_parallel_20260826_093313/summary.json \
          results/transmission_normalization_20260826/summary.json
# ... (동일 방식으로 checksums.sha256의 전 항목 반복)
```

`run_manifest.json`(원본) ↔ `manifest.json`(사본)은 파일명만 다르고 내용은
`diff`로 완전 동일함을 확인.

### 형식 검사

- 모든 `.csv` 파일: 표준 `csv` 모듈로 파싱 성공, 헤더/행 수 확인
  (`rate_matching.csv` 50행, `aggregate.csv`/`quantization_effect.csv` 등 정상 파싱).
- 모든 `.json` 파일: `json.load`로 파싱 성공.

### 재현성 필드 확인 (manifest 기반, 실측)

| 항목 | 값 | 출처 |
|---|---|---|
| dataset ref | `/home/wilco/.../sgdjscc_lab/data/etri_video_eval` (원격 GPU 서버 경로) | `manifest.json.dataset.ref` |
| dataset hash | `8e5e192304cfd7582f42cf0087a1ac28ad063003fb20cadbfb751b3dc56f930c` | `manifest.json.dataset.hash` |
| seed | `2025` | `manifest.json.seed`, 각 worker manifest 동일 |
| 실험 commit | `607f72798da24dc3e0c065574efcf6fce90683f3`, dirty=false | `manifest.json.git` |
| checkpoint 해시 | `JSCC_model.pth`/`diffusion_backbone.pth`/`diffusion_controlnet.pth`/`muge-epoch-19-checkpoint.pth` 각 SHA-256 기록 | `workers/worker_00/run_manifest.json.checkpoints.items` |
| resolved config SHA-256 | `24e670885b7a014a91dfe0aa285891a2a61056e268b8164149f39450da5b8442` (3 worker 동일) | `workers/worker_NN/run_manifest.json.extra.run_signature.resolved_config_sha256` |
| 최상위 `manifest.json`의 `resolved_config`/`checkpoints` | **확인 불가** — 이 필드들은 `status: "unknown"`으로 최상위 manifest에는 채워지지 않음 (병렬 드라이버가 worker 결과를 병합할 때 개별 worker manifest로만 상세 정보를 보존하는 방식) — worker manifest에서 위와 같이 확인 가능하므로 실질적 정보 손실은 아님 |
| environment (Python/torch/CUDA/GPU) | Python 3.9.25 / torch 2.1.0+cu118 / CUDA 11.8 / RTX 4090 × 3 | worker_00/01/02 `run_manifest.json.environment` 3개 모두 실측 확인 — 동일 값 (physical device만 각각 `cuda:0`/`cuda:1`/`cuda:2`로 상이, `extra.physical_cuda_device`) |

### Markdown 링크

이 파일 내 상대 링크(`../../docs/experiments/2026-08-26_transmission_normalization.md`)
경로 존재 확인 완료.

### Git 포함 여부

- `packets/`, `recon_frames/`, `*.mp4`, `worker_NN.log`, 원본 `outputs/` 트리 전체
  중 어떤 것도 이 커밋에 포함되지 않음 — staged 파일은 위 표에 나열된 소용량
  텍스트/CSV/JSON 파일뿐.

## 관련 문서

- [docs/experiments/2026-08-26_transmission_normalization.md](../../docs/experiments/2026-08-26_transmission_normalization.md)
  — 이 실행의 해석·판정·다음 작업
- [docs/protocols/transmission_normalization.md](../../docs/protocols/transmission_normalization.md)
  — 실행 절차, 3-GPU 병렬 안전성 설계, 알려진 한계
- [docs/current/open_issues.md](../../docs/current/open_issues.md) — float32 digital
  품질 저하, fixed–SKEM rate matching 불완전 이슈 등록
- [docs/current/roadmap.md](../../docs/current/roadmap.md) — 후속 작업 순서
- 과거 실행(비교용, 이 수정 이전 상태): [results/transmission_20260818/](../transmission_20260818/README.md)

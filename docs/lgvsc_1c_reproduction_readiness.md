> [← 문서 색인](./README.md)

# 1C — LGVSC 재현선 검증 준비 (reproduction-baseline readiness)

이 문서는 `docs/etri_strategy.md` "후속 딥러닝 4단계"의 **1C (LGVSC 재현선
검증)** 준비 상태를 정리한다. **실제 검증 실행은 사용자가 직접 한다** — 이
작업의 목적은 검증을 "명령어 한 줄이면 되는" 상태로 만드는 것이지, 검증
결과 자체를 만드는 것이 아니다. 1B가 완성한 것(mock/SVD/Wan start-only/Wan
bidirectional의 real-GPU 검증)을 그대로 재사용해 4개의 재현 가능한 baseline
config + batch driver + summary 생성기를 만들었다.

> **업데이트(PSSS/SKEM 단계, 후속 작업):** 이 문서가 정의한 4개 모드
> (`mock_baseline`/`svd_start_only`/`wan_skim_sfa`/`wan_skem_dsa`)는 **여전히
> 전부 동일한 keyframe 선택**을 쓴다 — 이 문서의 "keyframe 선택은 네 모드
> 전부 동일하다"는 서술은 이 4개 모드에 한해 그대로 유효하다. 이후
> `skim_sfa_fixed`/`skem_dsa_psss`/`skem_dsa_mock_psss`/`skem_dsa_proxy_psss`
> 4개 모드가 추가되어 **처음으로 SKIM(fixed)과 SKEM(PSSS 기반) selector가
> 실제로 다르게 동작**한다 — 자세한 내용과 mock/proxy/real PSSS 구분은
> [lgvsc_psss_skem_readiness.md](./lgvsc_psss_skem_readiness.md) 참조.

**이 문서 전체에서 반복하는 핵심 경고: 아래 어떤 config/모드도 LGVSC 논문의
faithful reproduction이 아니다.** LGVSC의 `SKIM`/`SKEM` keyframe 선택,
`SFA`/`DSA` decoder adapter, PSSS 점수화, side-info 인코딩은 논문에 세부
구현이 공개돼 있지 않다. 아래 네 모드는 "LGVSC-style reproducible
baseline"이며, 정확히 어느 부분이 실제 대응이고 어느 부분이 근사인지 이
문서와 각 config 파일 헤더에 명시한다.

## 1C 모드 정의

| 모드 | LGVSC 대응 | 실제로 무엇을 하는가 | 근사/한계 |
|---|---|---|---|
| `mock_baseline` | 없음 (이 저장소 자체 mock) | `BidirectionalInterpolationGenerator` — 시작/끝 keyframe 복원의 선형 블렌드. 학습된 모델 아님 | LGVSC 구성요소가 아니라 "실제 생성 모델이 전혀 없을 때의 바닥선" 비교용 |
| `svd_start_only` | 없음 (real diffusion, 최소 조건화) | `StableVideoDiffusionPipeline` — 시작 keyframe 이미지 1장만 조건화 | caption도 end-keyframe도 입력받지 않는 파이프라인이라 SFA도 DSA도 대응시킬 수 없음 — "실제 diffusion 모델, 최소 조건화" 참고점일 뿐 |
| `wan_skim_sfa` | `SKIM+SFA`의 **SFA(Start-frame Adapter)** 측 | `WanImageToVideoPipeline`, 시작 keyframe + caption 조건화 (real, 1B에서 실제 GPU 검증됨) | **SKIM(keyframe 선택)은 대응되지 않음** — 아래 "keyframe 선택은 4개 모드 전부 동일" 참조. `side_infos`는 accepted이지만 조건화에 안 씀(PSSS/side-info adapter 없음) |
| `wan_skem_dsa` | `SKEM+DSA`의 **DSA(Dual-side Adapter)** 측 | `WanImageToVideoPipeline`, 시작+끝 keyframe + caption 조건화 (real, 1B에서 실제 GPU 검증됨 — segment별 체크포인트 자동 선택) | **SKEM(의미 기반 keyframe 선택)은 대응되지 않음** — 아래 참조. `side_infos` 미사용은 wan_skim_sfa와 동일 |

**keyframe 선택(SKIM vs SKEM)은 네 모드 전부 동일하다.** 이 저장소는
`configs/video/default.yaml`의 `keyframe.max_gop`(고정 간격) + scene-change
detector(히스토그램 기반) 하나만 갖고 있고, 이는 SKIM에 가까운 방식이지
LGVSC의 SKEM이 뜻하는 의미/PSSS 기반 keyframe 선택과 동일하지 않다. 네
config 어느 것도 "SKEM 방식으로 keyframe을 골랐다"는 뜻이 아니다 — `wan_skim_sfa`와
`wan_skem_dsa`의 실제 차이는 오직 **decoder 측 조건화**(단일 keyframe vs
두 keyframe)뿐이다. 이는 과장이 아니라 정확한 서술이다 — 결과표를 인용할
때 반드시 이 구분을 함께 적을 것.

## Config ↔ 소스 대응

| 1C config | 기반 (1B 실제 GPU 검증 완료) |
|---|---|
| `configs/etri_lgvsc_1c_mock_baseline.yaml` | 없음 — 이 저장소 mock backend (`video_generator.backend: auto` + `conditioning_mode: bidirectional`) |
| `configs/etri_lgvsc_1c_svd_start_only.yaml` | `configs/etri_video_eval_lgvsc_worker_svd.yaml` (worker 블록 그대로 복사) |
| `configs/etri_lgvsc_1c_wan_skim_sfa.yaml` | `configs/etri_video_eval_lgvsc_worker_wan_start_only.yaml` (worker 블록 그대로 복사 — `tests/test_batch_lgvsc_1c_reproduce.py`가 두 파일의 `video_generator.worker`가 일치하는지 검증) |
| `configs/etri_lgvsc_1c_wan_skem_dsa.yaml` | `configs/etri_video_eval_lgvsc_worker_wan_bidirectional_fixed.yaml` (worker 블록 그대로 복사, segment별 `Wan2.1-I2V-14B-480P`/`Wan2.1-FLF2V-14B-720P` 자동 선택 로직 포함 — 동일하게 테스트로 검증) |

각 config는 (1) `configs/`에 직접 두고 단일 영상 수동 실행에도 쓸 수 있고,
(2) `scripts/batch_lgvsc_1c_reproduce.py`가 이 파일을 **base template**으로
읽어 영상별로 output-path만 다시 쓴 생성 config를
`outputs/etri_video_eval/lgvsc_1c_reproduce/_generated_configs/<mode>/<video_id>.yaml`에
써서 그걸 실제로 `evaluate_video.py --config`에 넘긴다 — 배치 실행에서
실제로 쓰이는 파일은 후자다.

## 실행 명령어

### 준비 확인 (dry-run — 아무 것도 실행하지 않음)

```bash
python scripts/batch_lgvsc_1c_reproduce.py \
    --modes wan_skim_sfa,wan_skem_dsa \
    --videos 01_person_walk,05_camera_pan_person \
    --max-frames 14 --device cuda:0 --dry-run
```

실행될 정확한 `evaluate_video.py` 명령어와 output 경로만 출력한다 —
subprocess를 전혀 실행하지 않는다(`tests/test_batch_lgvsc_1c_reproduce.py::TestDryRun`로
보장).

### Smoke (GPU 없이, 빠르게 — mock_baseline만 진짜 실행 비용이 없음)

```bash
python scripts/batch_lgvsc_1c_reproduce.py \
    --modes mock_baseline --videos 01_person_walk \
    --max-frames 14 --no-models
```

### 실제 GPU 검증 (사용자가 직접 실행)

```bash
# 모드 하나, 영상 하나, smoke 크기 (권장 첫 실행)
python scripts/batch_lgvsc_1c_reproduce.py \
    --modes wan_skim_sfa --videos 01_person_walk \
    --max-frames 14 --device cuda:0

# wan_skem_dsa는 end keyframe이 있는 segment가 필요 — keyframe.max_gop
# 기본값(12)보다 큰 --max-frames를 써야 bidirectional 경로가 실제로 실행된다.
python scripts/batch_lgvsc_1c_reproduce.py \
    --modes wan_skem_dsa --videos 01_person_walk \
    --max-frames 14 --device cuda:0

# 4개 모드 × 10개 영상 전체 (오래 걸림 — Wan 14B 모델, VRAM/시간 여유 있을 때)
python scripts/batch_lgvsc_1c_reproduce.py --modes all --device cuda:0

# 중간에 끊겼으면 이미 끝난 job은 건너뛰고 이어서
python scripts/batch_lgvsc_1c_reproduce.py --modes all --device cuda:0 --skip-existing

# 일부 영상이 실패해도 나머지 계속 진행
python scripts/batch_lgvsc_1c_reproduce.py --modes all --device cuda:0 --continue-on-error
```

### `--no-models`의 의미 (중요)

`--no-models`는 `evaluate_video.py`의 SGD-JSCC 본체(keyframe 복원 모델)만
끈다 — identity 복원으로 대체될 뿐이다. `svd_start_only`/`wan_skim_sfa`/
`wan_skem_dsa`의 generate-branch worker는 완전히 별도 프로세스(별도 conda
env, 기본 `semantic-diffusers`)이므로 `--no-models`를 줘도 **실제 GPU
생성 모델은 그대로 실행된다.** 즉 이 세 모드에 `--no-models`를 줘도 여전히
실제 GPU 비용이 든다 — SGD-JSCC 쪽 복원만 건너뛰어 조금 더 빨라질 뿐이다.
GPU 비용 없이 확인하고 싶다면 `--dry-run`을 쓰거나 `mock_baseline`만
돌린다.

### summary만 재생성 (실행 없이, 디스크에 있는 결과로부터)

```bash
python scripts/batch_lgvsc_1c_reproduce.py --summary-only
```

## 결과 산출물

```
outputs/etri_video_eval/lgvsc_1c_reproduce/
  mock_baseline/<video_id>/        recon.mp4, temporal_metrics.csv, segments.json,
                                   generated_frames/, keyframes.json, run.log
  svd_start_only/<video_id>/       (동일 파일 구성)
  wan_skim_sfa/<video_id>/         (동일 파일 구성)
  wan_skem_dsa/<video_id>/         (동일 파일 구성)
  _generated_configs/<mode>/<video_id>.yaml   실제 evaluate_video.py --config에 쓰인 파일
  batch_status.json                 (mode, video)별 실행 상태/명령/returncode 기록
  summary_metrics.csv / .md / .json 비교 결과표 (아래 컬럼)
```

`summary_metrics.*`의 컬럼: `mode`, `video_id`, `status`, `n_frames`,
`n_keyframes`, `n_interframes`, `n_generate`, `n_reused`,
`n_recompute_semantic`, `n_recompute_motion`, `temporal_srs`, `srs_flicker`,
`ptc`, `sfr`, `sdi`, `temporal_hallucination_rate`, `transmitted_units`,
`naive_units`, `overhead_reduction`, `generated_frame_count`,
`conditioning_modes_observed`(해당 영상에서 실제 관측된 conditioning_mode
집합), `backends_observed`(실제 관측된 backend 문자열 집합 — 모델 ID까지
포함), `has_end_keyframe`(해당 영상의 어느 segment든 end keyframe이 있었는지),
`error_log_path`(실패 시 `run.log` 경로).

`status`는 `ok`(성공) / `failed`(subprocess가 non-zero로 종료) /
`skipped`(`--skip-existing`으로 건너뜀) / `dry_run`(`--dry-run`으로 실행
안 함) / `missing`(이 (mode, video) 조합이 아직 한 번도 시도되지 않음) 중
하나다.

## 결과 해석 시 주의사항

1. **네 모드를 그대로 비교하려면 `temporal.reuse_threshold`를 통일하라.**
   각 `configs/etri_lgvsc_1c_*.yaml`은 `reuse_threshold: 0.0`으로 기본
   설정돼 있다 — smoke 테스트가 짧은 `--max-frames`에서도 generate 분기를
   반드시 거치게 하려는 것이다. 품질 비교표를 만들 때는 네 config 모두
   같은 (0이 아닌) 값으로 맞춰야 "같은 프레임이 reuse/recompute/generate로
   갈렸다"는 전제가 성립한다.
2. **`wan_skim_sfa`/`wan_skem_dsa`를 "SKIM"/"SKEM 재현"으로 인용하지 마라.**
   위에서 설명했듯 keyframe 선택은 네 모드가 전부 동일한 이 저장소의
   추출기를 쓴다 — 이름의 SKIM/SKEM은 LGVSC 논문 구성 요소와의 "가장
   가까운 대응(nearest reproducible)"을 표시한 것이지, 그 알고리즘을 그대로
   재현했다는 뜻이 아니다.
3. **`side_infos`는 어느 Wan 모드도 조건화에 쓰지 않는다.** LGVSC의
   side-info 인코더/PSSS에 대응하는 실제 구현이 없다는 뜻이며,
   `scripts/lgvsc_generate_worker.py::run_wan_backend`의 docstring에 이미
   명시돼 있다.
4. **`wan_skem_dsa`의 bidirectional 경로를 실제로 보려면 `--max-frames`가
   `keyframe.max_gop`(기본 12)보다 커야 한다.** 그보다 작으면 영상 전체가
   keyframe 하나짜리 GOP 하나가 되어 end keyframe이 없는 start-only
   경로만 실행된다 — `conditioning_modes_observed`가 `start_only`만
   나오는 게 그 신호다.
5. **`svd_start_only`는 caption/end-keyframe을 아예 못 받는다** — SFA/DSA
   어느 쪽과도 비교 축이 다르다. "real diffusion, 최소 조건화"라는 별도
   참고선으로만 해석하라.
6. **1B에서 검증된 것은 "1개 segment/짧은 smoke가 성공한다"는 것이지,
   "출력 품질이 좋다"는 것이 아니다.** 이 문서/config들은 재현 파이프라인이
   실제로 동작함을 보장할 뿐, PTC/SFR/SDI/SRS 수치의 절대적 우열을 보장하지
   않는다 — 그 판단은 사용자가 실제 실행 결과로 직접 내려야 한다.
7. **`Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers`는 처음 실행 시 별도로 ~84GB를
   다운로드한다**(I2V-480P와 별개 체크포인트) — `wan_skem_dsa`의 첫 실행은
   두 체크포인트 합쳐 ~180GB가 필요할 수 있다. 자세한 사항은
   `docs/lgvsc_1b_worker_readiness.md` 참조.

## 테스트

```bash
python -m pytest tests/test_batch_lgvsc_1c_reproduce.py tests/test_video_generator.py tests/test_video.py -q
python -m pytest tests/ -q
```

`tests/test_batch_lgvsc_1c_reproduce.py`(29개 — PSSS/SKEM 단계에서 9개 추가)는
모드→config 선택, output 경로 격리, dry-run이 subprocess를 호출하지 않음,
`--max-frames`/`--device`/`--no-models`가 명령어에 반영됨, `--summary-only`가
디스크의 기존 결과에서 summary를 재생성함, `--continue-on-error`로 실패 job
이후에도 다음 job이 실행됨, `wan_skim_sfa`/`wan_skem_dsa`/`svd_start_only`
config가 각각의 1B 검증 완료 config를 기반으로 함, 그리고(PSSS/SKEM 단계 추가분)
새 4모드의 selector/psss config provenance, 확장된 summary 필드, aggregate
비교표를 검증한다. 실제 GPU 실행/전체 batch run은 이 테스트 범위 밖이다 —
사용자가 직접 실행해서 확인한다.

## 관련 문서

- [etri_strategy.md](./etri_strategy.md) — "후속 딥러닝 4단계", 1A/1B/1C 구현 결과
- [video_extension_lgvsc.md](./video_extension_lgvsc.md) — LGVSC 재현선과 ETRI
  개선선 구분, 6.0-a/6.2/6.3
- [lgvsc_1b_worker_readiness.md](./lgvsc_1b_worker_readiness.md) — 1B 실제
  GPU 검증 상세(SVD/Wan start-only/Wan bidirectional), 환경 구성, 체크포인트
  용량/시간
- [lgvsc_psss_skem_readiness.md](./lgvsc_psss_skem_readiness.md) — 이 문서의
  후속 단계: PSSS 기반 SKEM selector, variable-length segment, SKIM/SFA vs
  SKEM/DSA 비교 config 4종 + batch summary 확장

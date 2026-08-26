---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: 029f97a
supersedes:
---

> [← 문서 색인](../README.md)

# 전송 실험 정상화 (digital blind-SNR NaN 수정 + 재현성 + 양자화/selector 효과 분리)

- 연결 문서
  - 전송 스윕 드라이버: [scripts/run_transmission_reduction_eval.py](../../scripts/run_transmission_reduction_eval.py)
  - 과거 실험 기록(불변): [experiments/2026-08-18_transmission_reduction.md](../experiments/2026-08-18_transmission_reduction.md)
  - run manifest 절차(정식 의존성): [results_registry.md](./results_registry.md)
  - Tx/Rx 계약: [../architecture/tx_rx_contract.md](../architecture/tx_rx_contract.md)

## 배경 — 무엇이 고장났었나, 어떻게 고쳤나

- 증상
  - `digital_packet` 채널(4/6/8/16/32-bit 양자화)로 복원한 프레임 일부가 NaN/Inf
  - 과거 실측(2026-08-18): `fixed_int16` 52프레임 NaN — reliable-digital 기준(int16)
    자체가 깨져 있어 Pareto/baseline으로 쓸 수 없었음
- 원인 (`pipelines/infer_pipeline.py::_compute_step`)
  - `step_style="continuous"` + blind(`use_gt_csi=False`) 분기가 채널 종류와 무관하게
    `jscc.snr_prediction_net`(AWGN 잡음 통계로만 학습된 신경망)에 수신 latent를 통과시킴
  - 양자화 latent는 AWGN과 통계 구조가 전혀 달라 `predicted_signal_scale >= 1`을
    예측하는 경우가 생기고 → `cur_step = 1 - predicted_signal_scale <= 0`
  - `cur_snr = 10*log10(1/cur_step - 1)`에서 `log10(비양수)` → NaN/Inf
- 1차 수정 (2026-08-26 오전)
  - `jscc.channel_model`이 `DigitalPacketChannel`일 때만 blind 분기를 우회,
    `_digital_quant_snr_db(bit_depth) = min(20*log10(2**bit_depth-1), 60dB)`로 계산
  - `_apply_channel`/`_compute_step`/`_decode_diffusion` 출력에 `assert_finite` 추가
- 2차 정상화 (2026-08-26 오후, 이 문서가 서술하는 범위) — 아래 각 절 참고
  - digital step 정책을 `fixed_reference`/`bitdepth_proxy`/`quant_nmse` 3종으로 분리하고
    양자화 비교의 기본값을 `fixed_reference`로 변경(decoder step 변화가 섞이지 않게)
  - receiver가 **패킷 자체의 metadata**(bit_depth, 실측 quantization SNR)를 사용하도록
    변경 — 더 이상 전역 `jscc.channel_model` 객체에 의존하지 않음
  - non-finite 발생 시 해당 (video, config) 쌍을 **즉시 중단**(NaN placeholder로 계속
    처리하지 않음)
  - `run_manifest.py`를 소프트 의존성에서 **정식(하드) 의존성**으로 전환, 초기/최종
    manifest + run signature 기반 resume 안전성 도입
  - `FixedCountKeyframeSelector`로 fixed selector의 keyframe 수를 SKEM과 **정확히** 일치

## digital step 정책 3종 (`--digital-step-policy`)

`pipelines/infer_pipeline.py::DIGITAL_STEP_POLICIES`:

| 정책 | 의미 | SNR 출처 |
|---|---|---|
| `fixed_reference` (기본, 양자화 비교용) | 모든 bit_depth를 float32 기준과 동일한 step으로 디코딩 — decoder step 변화가 섞이지 않은 순수 양자화 효과 | 항상 상한(60dB), bit_depth 무관 |
| `bitdepth_proxy` | bit_depth만으로 결정되는 결정론적 휴리스틱 — **실측 SNR 아님**, 데이터 의존성 없음 | `20*log10(2**bit_depth-1)`, [-20,60]dB clamp |
| `quant_nmse` | 송신단이 **실측한** quantization NMSE/SNR(패킷 metadata로 전송) | receiver가 패킷 자체에서 읽음, 없으면 즉시 `ValueError` |

- `bit_depth=32`(float32, 무손실)는 정책과 무관하게 항상 상한 — lossless transport는
  정책 선택 대상이 아니라 구조적 사실
- `fixed_reference`가 아닌 정책으로 실행하려면 `--ablation-label`이 **필수**(양자화
  비교(`quantization_effect.csv`)에 decoder-step ablation이 섞여 들어가지 않도록 강제)
- `quant_nmse`의 실측값 계산: 송신단이 `quantize_tensor()` 직후 같은 텐서를
  `dequantize_tensor()`로 복원해 `10*log10(signal_power/mse)`를 계산
  - `quant_nmse`: packet JSON metadata에 실어 전송하고 byte에 포함
  - `fixed_reference`/`bitdepth_proxy`: packet에서는 제외하고
    `quantization_diagnostics.csv`에만 기록
  - 구현:
  `transmission/packet_bundle.py::measure_quantization_error`

## receiver가 packet metadata를 쓴다 (전역 channel 객체 아님)

- `transmission/packet_bundle.py::decode_frame_bundle`가 이제 `visual_metadata`(패치별
  `bit_depth`/`quant_snr_db`/`quant_mse`)를 `visual_latents`와 함께 반환
- `transmission/receiver_runtime.py::reconstruct_frame_from_bundle_bytes`가 이 metadata를
  `_compute_step(..., digital_bit_depth=meta["bit_depth"], digital_quant_snr_db=meta["quant_snr_db"])`로
  직접 전달 — `jscc.channel_model`을 절대 참조하지 않음(receiver 경계 유지)
- (참고) `_encode_and_transmit`의 단순 in-process 경로(`DigitalPacketChannel.transmit()`이
  텐서를 직접 반환하는, packet 경계가 없는 경로)는 여전히 `jscc.channel_model.bit_depth`를
  읽는다 — 이건 "receiver가 packet만 봐야 한다"는 계약과 무관(애초에 packet 자체가 없는
  경로이므로)

## finite 검사 + 실패 처리 (abort-on-first-failure)

- `utils/finite_checks.py::assert_finite`가 다음 stage마다 추가됨: encode latent, channel
  output, power scalar, step/SNR, canny output, canny latent, diffusion latent(init +
  water-filling + early-exit + 표준 경로 각각), VAE 최종 출력
- non-finite 발생 시
  - **NaN placeholder로 대체해 이후 프레임을 계속 처리하지 않는다** — 해당 (video,
    config) 쌍을 그 자리에서 즉시 중단
  - 실패 stage·frame index·NaN/Inf 수를 `failed_pairs.csv`에 기록하고 다음 pair로 이동
  - 그 pair는 `per_video_metrics.csv`에 아예 행이 생기지 않음(baseline/Pareto/effect
    비교에서 자동으로 배제됨)
- 예상하지 못한 일반 예외(`NonFiniteError`가 아닌 모든 것)는 잡지 않고 그대로 전파 —
  전체 실행이 실패함(pair 하나만 건너뛰지 않음)
- 성공한 pair에서 사후적으로 non-finite recon이 다시 발견되면(있어서는 안 되는 상황)
  이는 `assert_finite` stage 커버리지의 실제 공백이므로 조용히 넘기지 않고
  `RuntimeError`로 즉시 실패

## 재현성 — run manifest 정식 의존성 + seed + resume 안전성

- `run_manifest.py`는 이제 **하드 의존성** — `scripts/run_transmission_reduction_eval.py`
  모듈 최상단에서 직접 import, 모듈이 없으면 `--help`조차 즉시 실패(soft-fallback 없음)
- `--seed`(기본 2025): Python/NumPy/PyTorch/CUDA 전역 seed +
  `utils/seed.py::derive_frame_seed(base_seed, video_key, frame_index)`로 **영상·프레임별
  결정적 seed** — 같은 (video, frame)을 다른 config가 복원할 때 동일 RNG 상태를 최대한
  공유(비교가 채널/양자화/selector 차이만 반영하도록)
- run signature (`run_signature.json`): commit·dataset/config/checkpoint hash·seed·영상
  목록과 프레임 수·granularity·PSSS 설정·평가 옵션을 담음
  - 최초 실행 시 생성
  - `--resume` 시 현재 조건과 다르면 **즉시 거부**하고 차이를 출력(다른 run이 같은
    디렉터리에 섞여 들어가는 것을 방지)
  - commit이 `unknown`이면 실험 시작을 거부
  - 컨테이너에 `git`이 없어도 `.git/HEAD`와 refs에서 commit을 직접 읽음
  - `.git`도 없으면 검증한 host 값을 `SGDJSCC_GIT_COMMIT`으로 주입
- `run_manifest_initial.json`: 영상 처리 시작 **전**에 기록(의도한 run spec)
- `run_manifest_final.json` (= `run_manifest.json`): effect summarizer까지 끝나 모든
  산출물(CSV/JSON/README)이 기록된 **후** 생성, `extra.output_artifact_sha256`에
  `quantization_effect*`/`selector_effect*`/`normalization_effect_summary.json`을 포함한
  핵심 artifact SHA-256 저장
  (`_hash_output_artifacts` — 실제 존재하는 파일만, 없는 파일은 절대 조작하지 않음)

## fixed/SKEM keyframe 수 정확히 일치 (`--match-fixed-keyframes`)

- 기존: `max_gop` 근사값 계산(부정확)
- 현재: `video/keyframe_extractor.py::FixedCountKeyframeSelector(count=N)`로 fixed
  selector를 SKEM이 고른 keyframe 수와 **정확히** 동일하게 강제
  (`keyframe.selector: fixed_count`)
- 불가능한 경우(SKEM 개수가 0이거나 프레임 수를 초과) `--fixed-max-gop`로 폴백하고
  `keyframe_count_matched=False`를 명시적으로 기록(숨기지 않음)
- `rate_matching.csv`: 영상×channel별 fixed vs SKEM의 실제 keyframe 수·bytes/video·
  bytes/frame·byte 차이 비율(`byte_diff_ratio`)을 기록. **byte 차이가
  `RATE_MATCH_BYTE_TOLERANCE`(10%) 이내일 때만** `rate_matched=true` — keyframe 수만
  맞다고 "rate-matched"라 부르지 않음
- 효과 표
  - `fixed_reference`: `quantization_effect.csv` / `selector_effect.csv`
  - decoder-step ablation: `quantization_effect_ablation.csv` /
    `selector_effect_ablation.csv`
  - `ablation_label`은 per-video·aggregate·effect 표·run signature에 보존

## bytes/video vs bytes/frame — 단위 혼동 금지

- `per_video_metrics.csv`/`aggregate.csv`: `total_bundle_bytes`(**bytes/video**)와
  `total_bundle_bytes_per_frame`(**bytes/frame**, 전체 프레임 기준)을 분리된 컬럼으로 기록
- `aggregate.csv`의 `mean_total_bundle_bytes_per_video`/`mean_total_bundle_bytes_per_frame`도
  동일 원칙(과거 `mean_total_bundle_bytes`라는 모호한 이름은 제거)

## 유효성 조건 강화 (baseline / Pareto / effect 비교 공통)

`run_transmission_reduction_eval.py::_pareto_frontier`의 `_row_is_valid`, 다음 전부 만족해야 유효:

1. non-finite frame 0건 (`total_nan_or_inf_frames == 0`)
2. PSNR·SSIM·LPIPS 전부 finite (`all_finite_metrics`)
3. `valid_frame_ratio == 1`(reuse/generate를 포함한 전체 복원 프레임 기준)
4. 기대 영상이 모두 완료됨 (`all_expected_videos_present`)
5. (baseline 대비 후보만) 후보의 영상 집합이 baseline의 영상 집합과 **동일**
   (`video_set_mismatch_vs_baseline`)

- AWGN
  - visual waveform의 wire byte가 없으므로 Pareto 후보에서 항상 제외
  - 별도 품질 참고 행으로만 유지
- 요약기
  - 필수 컬럼 누락·파싱 실패·NaN 품질값을 모두 invalid로 처리(fail-closed)

미달이면 baseline·Pareto·effect delta에서 제외하되 실패 이유(`quality_gate_failure_reason`)를
남기고, 가장 가까운 후보라도 숨기지 않고 나열한다.

## PSSS/SKEM 표기 — proxy를 real로 표기하지 않는다

- one-command wrapper(`run_transmission_normalization.sh`)가
  `--psss-backend`/`--psss-model-id`/`--psss-device`/`--psss-dtype`/`--psss-threshold`/
  `--psss-max-segment-length`/`--use-scene-detector`를 그대로 전달
- 모든 SKEM 행에 `psss_backend_kind`(`mock`|`proxy`|`real`) 컬럼이 항상 붙음 —
  `real`이 아니면 절대 "real SKEM"으로 표기하지 않음
- `selector_effect.csv`의 `skem_psss_backend_kind`, run README의 "이번 run의 SKEM 행에
  실제 관측된 backend" 문구로 항상 명시

## 실행 스크립트 — `scripts/run_transmission_normalization.sh`

```bash
bash scripts/run_transmission_normalization.sh
bash scripts/run_transmission_normalization.sh --preflight-only
bash scripts/run_transmission_normalization.sh --dry-run
bash scripts/run_transmission_normalization.sh --resume outputs/transmission_normalization_<timestamp>
bash scripts/run_transmission_normalization.sh --resume outputs/transmission_normalization_<timestamp> --retry-failed
bash scripts/run_transmission_normalization.sh --digital-step-policy bitdepth_proxy --ablation-label bp_ablation_v1
```

- preflight (항상 먼저 실행, 실패 시 즉시 종료): 정확한 Git commit, 데이터 manifest,
  checkpoint 4종, 디스크 여유, 선택한 CUDA ordinal, `nvidia-smi`,
  `torch.cuda.is_available()`(NVML 오류는 별도로 명확히 보고)
- 기본 config grid (11개): `fixed_awgn, {fixed,skem}_{float32,int16,int8,int6,int4}`
- resume: python 드라이버가 `run_signature.json`으로 조건 일치를 확인하고, 완료된
  (video, config)는 건너뛰며, 매 pair마다 CSV를 원자적(`os.replace`)으로 즉시 갱신 —
  중단돼도 마지막 완료 pair부터 이어서 실행됨
  - 실패 pair도 기본적으로 skip해 중복 행을 만들지 않음
  - `--retry-failed`를 지정할 때만 기존 실패 행을 제거하고 재시도
  - 실패가 남으면 `completed_with_failures`와 비정상 종료 코드 3을 기록·반환

## 3-GPU 안전 병렬 실행

```bash
bash scripts/run_transmission_normalization_parallel.sh --preflight-only
bash scripts/run_transmission_normalization_parallel.sh --dry-run
bash scripts/run_transmission_normalization_parallel.sh --devices cuda:0,cuda:1,cuda:2
bash scripts/run_transmission_normalization_parallel.sh \
  --resume outputs/transmission_normalization_parallel_<timestamp> \
  --devices cuda:0,cuda:1,cuda:2 --retry-failed
```

- 작업 분배
  - `manifest.csv`의 프레임 수를 기준으로 영상을 longest-first 방식으로 균등 배분
  - GPU별 단일 프로세스·단일 `--device` 사용
  - 기본 10영상은 3개 worker에 `4/3/3`개로 분할
- 파일 안전성
  - `workers/worker_00`·`worker_01`·`worker_02`가 독립 CSV·packet·복원 영상을 기록
  - worker 사이에 공유하는 가변 CSV·manifest 없음
  - 모든 worker가 종료된 뒤 상위 디렉터리에 aggregate·Pareto·effect 표를 재계산
  - 대용량 packet·복원 영상은 worker 디렉터리에 유지하고 중복 복사하지 않음
- 재개
  - `parallel_plan.json`에 commit·GPU·영상 배분·실험 설정을 고정
  - 계획이 바뀐 동일 출력 디렉터리 재사용은 거부
  - worker별 기존 signature/resume 검증도 그대로 적용
  - 실패 pair는 `--retry-failed`에서만 해당 worker가 재시도
- 결과
  - 상위 `run_manifest.json`이 worker manifest hash와 병합 산출물 hash를 기록
  - worker 실패가 남으면 병합 결과도 `completed_with_failures`, 종료 코드 3

## 알려진 한계

- **16GB급 단일 GPU에서 digital_packet 설정이 OOM 날 수 있음** — ModelBundle만으로
  이미 ~13GB 점유. AWGN 경로는 동일 GPU에서 안정적으로 통과 — 이 코드 수정 자체의
  결함이 아니라 리소스 문제로 확인됨. 실측은 VRAM 여유가 큰 원격 다중-GPU 서버에서
  수행할 것
- 과거 `results/transmission_20260818` run의 `int16: 52 NaN` 수치는 이 수정 **이전**
  상태의 기록이다 — 재현 시도 시 새 run으로 남기고 과거 수치를 덮어쓰지 않음

## 테스트

- `tests/test_digital_step_matching.py` — `_digital_quant_snr_db`/`_digital_effective_snr_db`/
  `_digital_signal_scale`의 3개 정책·경계값 finite 검증, AWGN 경로 수식 불변 회귀,
  `_compute_step`의 `digital_bit_depth`/`digital_policy` 명시적 override(전역 channel
  객체 없이도 동작) 검증
- `tests/test_packet_bundle.py` — `measure_quantization_error` 실측 NMSE/SNR,
  `decode_frame_bundle`의 `visual_metadata` round-trip
- `tests/test_receiver_runtime.py` — receiver가 패킷 metadata에서 bit_depth/policy를
  도출하고 `jscc.channel_model`을 참조하지 않음을 명시적으로 검증
- `tests/test_transmission_reduction_eval.py` — signature/resume 안전성, 유효성 조건
  5가지, rate_matching, artifact hashing, run manifest hard-dependency
- `tests/test_summarize_transmission_normalization.py` — bytes/video vs bytes/frame,
  유효성 조건 재검증, skem backend 라벨링, ablation 정책 경고
- `tests/test_run_manifest.py` (35개) — commit/dirty·seed·checkpoint hash·resolved
  config 판정
- GPU 불필요(전부 CPU) — 실제 모델 smoke는 위 "알려진 한계" 절 참고

---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: 076a26d
supersedes:
---

> [← 문서 색인](../README.md)

# 전송 실험 정상화 (digital blind-SNR NaN 수정 + 양자화/selector 효과 분리)

- 연결 문서
  - 전송 스윕 드라이버: [scripts/run_transmission_reduction_eval.py](../../scripts/run_transmission_reduction_eval.py)
  - 과거 실험 기록(불변): [experiments/2026-08-18_transmission_reduction.md](../experiments/2026-08-18_transmission_reduction.md)
  - run manifest 절차: [results_registry.md](./results_registry.md)
  - Tx/Rx 계약: [../architecture/tx_rx_contract.md](../architecture/tx_rx_contract.md)

## 배경 — 무엇이 고장났었나

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
- 수정
  - `jscc.channel_model`이 `DigitalPacketChannel`일 때만 blind 분기를 우회
  - `_digital_quant_snr_db(bit_depth) = min(20*log10(2**bit_depth - 1), 60dB)`로
    **양자화 metadata(bit_depth)만으로** 결정되는 SNR을 계산 — 데이터 의존성 없음,
    항상 유한
  - `bit_depth=32`(무손실 float32)는 상한(60dB)으로 고정
  - `signal_scale = snr_scale/(snr_scale+1)`(기존 `use_gt_csi` 경로와 동일한, 항상
    `(0,1)` 안에 머무는 공식)로 `cur_step`을 재사용 — 0/1 경계에 절대 닿지 않음
  - AWGN 경로(`channel_model is None`)는 분기 자체를 타지 않음 — 수식·타입 100% 동일
  - `discrete` step_style의 digital 분기도 동일한 analytic SNR로 교체
  - `_apply_channel`/`_compute_step`/`_decode_diffusion` 출력에
    `utils/finite_checks.py::assert_finite`를 추가 — stage 이름을 붙여 NaN/Inf가
    생기는 즉시(전체 diffusion 루프를 다 돌리기 전에) 예외로 드러남

## 실행 스크립트 — `scripts/run_transmission_normalization.sh`

- 단일 진입점
  ```bash
  bash scripts/run_transmission_normalization.sh
  bash scripts/run_transmission_normalization.sh --preflight-only
  bash scripts/run_transmission_normalization.sh --dry-run
  bash scripts/run_transmission_normalization.sh --resume outputs/transmission_normalization_<timestamp>
  ```
- preflight (항상 먼저 실행, 실패 시 즉시 종료)
  - 데이터: `<dataset-root>/manifest.csv` 존재 확인
  - checkpoint: `JSCC_model.pth`/`diffusion_backbone.pth`/`diffusion_controlnet.pth`/
    `muge-epoch-19-checkpoint.pth` 4종
  - 디스크: 출력 경로 여유 공간 (기본 20GiB 미만이면 실패)
  - GPU/CUDA/NVML: `nvidia-smi` 실행 확인 → `torch.cuda.is_available()` 확인 →
    stderr에 `nvml` 문자열이 있으면 "드라이버/컨테이너 GPU passthrough 문제"로 별도 보고
    (컨테이너 재생성 유도하지 않음, 상태만 보고하고 중단)
- 기본 config grid (11개): `fixed_awgn, {fixed,skem}_{float32,int16,int8,int6,int4}`
- `--match-fixed-keyframes`(기본 ON): 동일 영상에서 SKEM이 고른 keyframe 수에 맞춰
  `fixed` selector의 `max_gop`을 자동 계산 — fixed vs SKEM을 **거의 동일 keyframe
  수**로 비교 가능
- resume: `run_transmission_reduction_eval.py`가 `--output-root`에 이미 있는
  `per_video_metrics.csv`를 읽어 완료된 (video, config) 쌍을 건너뛰고, 매 쌍마다
  CSV를 즉시 append/재기록 — 중단돼도 마지막으로 완료된 쌍부터 이어서 실행됨
  (같은 `--resume DIR`로 재실행하면 됨; 별도 python 플래그 불필요)

## 산출물 분리 — 양자화 효과 vs selector 효과

`scripts/summarize_transmission_normalization.py --run-root <output_root>`가
`aggregate.csv`에서 두 표를 분리 생성한다 (같은 스크립트가 정상화 스크립트
마지막 단계로 자동 호출됨):

- `quantization_effect.csv` — **selector 고정**, bit_depth만 변화
  - selector별로 자기 자신의 float32(없으면 int16) 행을 기준으로 삼아
    psnr_drop/ssim_drop/lpips_rise/byte_ratio 계산
  - fixed의 양자화 효과와 skem의 양자화 효과가 서로 섞이지 않음(각자 자기 기준과만 비교)
- `selector_effect.csv` — **bit_depth 고정**, selector만 변화(fixed vs skem)
  - 같은 bit_depth의 fixed/skem 쌍만 비교, keyframe 수 delta도 함께 기록
- 두 표 모두 `total_nan_or_inf_frames > 0`인 쪽이 끼면 `valid=false` + 사유를 남기고
  delta 계산은 비움 — 숨기지 않고 나열만 함

## baseline / Pareto 자격 규칙

- `BASELINE_PREFERENCE = [fixed_float32, fixed_int16, skem_float32, skem_int16]`
  — **AWGN은 baseline 후보에서 완전히 제외**(analog 잡음과 양자화 손실을 섞으면
  "reliable digital 기준"이 될 수 없음), fallback도 없음
- 후보가 `n_nan_or_inf_frames == 0`이 아니면 baseline도, Pareto 후보도 될 수 없음
  (기존 "존재하면 AWGN으로 폴백" 동작 제거)
- 아무 baseline도 유효하지 않으면 `pareto_frontier.csv`는 비고, `summary.json`에
  이유가 명시됨(숨기지 않음)

## run manifest

- `run_transmission_reduction_eval.py`가 매 실행 종료 시
  `<output_root>/run_manifest.json`을 `sgdjscc_lab.utils.run_manifest`로 생성
  (soft dependency — 모듈이 없으면 `status: "unavailable"`만 남기고 스윕 자체는
  실패시키지 않음)
- 기록 항목: git commit/dirty, 실행 argv(`captured`), resolved config, seed
  (`not_set` — 이 스크립트는 `--seed` 인자가 없음), dataset manifest sha256,
  존재하는 checkpoint 4종의 sha256, Python/CUDA/GPU 환경, exact/proxy 필드 목록,
  `total_nan_or_inf_frames`

## 알려진 한계

- **16GB급 단일 GPU에서 digital_packet 설정이 OOM 날 수 있음** — ModelBundle
  (VAE + MDTv2 backbone + ControlNet + CLIP ViT-L/14 + MuGE Canny + 캡션 모델 +
  LPIPS)만으로 이미 ~13GB를 점유, canny 재전송 net(WITT decoder) forward에서
  추가 할당이 필요한 순간 여유가 없으면 `CUDA out of memory`가 남. AWGN 경로는
  동일 GPU에서 안정적으로 통과했지만 digital 경로(양쪽 다 fresh process, 동일
  프레임)는 재현적으로 실패해 — 이 코드 수정 자체의 결함이 아니라(NaN 재현 경로는
  실제로 `_compute_step`을 지나 diffusion decode까지 정상 진입했다) 리소스 문제로
  확인됨. 실측은 VRAM 여유가 큰 원격 다중-GPU 서버에서 수행할 것.
- 확인 방법: `bash scripts/run_transmission_normalization.sh --video-ids
  01_person_walk --max-frames 2 --configs fixed_int4 --device cuda:0` 형태의
  최소 smoke만으로도 재현 가능 — 전체 스윕 전에 소형 GPU에서는 이 smoke조차
  실패할 수 있다는 점을 인지할 것.
- 과거 `results/transmission_20260818` run의 `int16: 52 NaN` 수치는 이 수정
  **이전** 상태의 기록이다 — 재현 시도 시 새 run으로 남기고 과거 수치를 덮어쓰지 않음.

## 테스트

- `tests/test_digital_step_matching.py` (32개) — `_digital_quant_snr_db`/
  `_digital_signal_scale`의 전 bit_depth·경계값 finite 검증, AWGN 경로 수식 불변
  회귀, `assert_finite`/`NonFiniteError` 유닛 테스트, canny 재전송이 기대하는
  텐서 타입 회귀
- `tests/test_transmission_reduction_eval.py` — baseline AWGN 배제, resume
  CSV round-trip, run manifest 연동
- `tests/test_summarize_transmission_normalization.py` — 양자화/selector 효과
  표 분리 로직
- GPU 불필요(전부 CPU) — 실제 모델 smoke는 위 "알려진 한계" 절 참고

---
status: active
updated: 2026-08-27
owner: ETRI SGD-JSCC 연구팀
source_commit: a3935b4
supersedes:
---

> [← 문서 색인](../README.md)

# float32 digital 복원 품질 저하 진단 환경

- 배경
  - [2026-08-26 전송 정상화 결과](../experiments/2026-08-26_transmission_normalization.md): float32(reliable-digital baseline) 복원이
    AWGN 참고 경로보다 크게 낮음 (PSNR 11.32 vs 23.34, SSIM 0.081 vs 0.731, LPIPS 0.739 vs 0.254).
  - [roadmap.md](../current/roadmap.md) §2 / [open_issues.md](../current/open_issues.md): 원인이 packet/Tx-Rx 계약
    문제인지 edge·ControlNet·diffusion 문제인지 latent scaling/normalization 문제인지 미분리.
  - **이 문서가 서술하는 범위는 진단 환경(harness)의 구현이다. 서버 실측 결과는 아직 없다 —
    아래 어떤 판정도 이 harness가 실제 GPU에서 실행되기 전까지는 결론이 아니다.**

## 무엇을 비교하는가

동일 (영상, 프레임, seed)에서 세 가지 실제 production Tx/Rx 경로를 비교한다
(구현: [`src/sgdjscc_lab/diagnostics/float32_digital_paths.py`](../../src/sgdjscc_lab/diagnostics/float32_digital_paths.py)):

| 경로 | 의미 | 사용하는 실제 production 코드 |
|---|---|---|
| `awgn` | 기존 production AWGN 경로 (기준) | `pipelines/infer_pipeline.py::_encode_and_transmit`/`_decode_diffusion`의 구성 함수 |
| `digital_inprocess` | `jscc.channel_model`에 `DigitalPacketChannel(bit_depth=32)`를 꽂음 — AWGN과 동일한 `_apply_channel()` 호출 지점을 그대로 사용, 프레임 단위 packet-bundle byte 경계를 거치지 않음 | 위와 동일 + `channels/digital_packet.py::DigitalPacketChannel` |
| `digital_wire` | 실제 `transmission.receiver_runtime` byte 경계 — `encode_frame_to_bundle_bytes`가 만든 실제 직렬화 bytes를 `decode_frame_bundle`이 독립적으로 파싱 | `transmission/packet_bundle.py`, `transmission/wire_packet.py`, `transmission/quantization.py` |

세 경로 모두 `mask_method=none`으로 고정한다 (`transmission/receiver_runtime.py`가 이미 요구하는 제약과 동일 —
sender-only mask 통계는 digital_wire 수신 경계를 정당하게 넘을 수 없음).

`diagnostics/float32_digital_paths.py::instrumented_decode`는 `pipelines/infer_pipeline.py::_decode_diffusion`의
제어 흐름을 재현하되(stage tensor 캡처 + ablation 개입 지점 추가), 모든 실제 텐서 연산은 그대로 real production
함수(`_compute_power_scalar`, `_compute_step`, `_retransmit_canny`, `_encode_canny_latent`, `_run_diffusion`,
`jscc.vae.decode`/`jscc.normalize`)를 호출한다. `pipelines/infer_pipeline.py` 자체는 전혀 수정하지 않으므로
기존 기본 전송 동작은 이 harness와 무관하게 보존된다 — 이 중복 orchestration이 실제 `_decode_diffusion`과
어긋나지 않는지는 `tests/test_float32_digital_diagnostics.py::TestDecodeParity`가 baseline ablation에서
bit-exact 동등성으로 검증한다.

## 계측 stage

`sender_vae_latent_pre_norm`/`sender_vae_latent_post_norm`(sender VAE latent, scaling/normalization 전후,
3경로 공통) · `pre_serialize_latent`(digital_wire 전용, 직렬화 직전 — `sender_vae_latent_post_norm`과 동일
tensor의 명시적 별칭) · `post_deserialize_latent_raw`(digital_wire, 역직렬화 직후, normalize 전) ·
`channel_output`/`receiver_post_norm_latent`(3경로 공통, normalize 직후 — awgn/digital_inprocess는
`_apply_channel()` 안에서 channel 적용과 normalize가 한 호출로 합쳐지므로 같은 tensor에 두 이름을 모두 붙이고,
digital_wire는 역직렬화 후 별도 `jscc.normalize()` 호출이 있으므로 이 이름이 그 지점을 가리킨다) · `power_scalar` ·
`cur_step`/`cur_snr`(실제 decoder 정책, 3경로 모두 기록) · `edge_mean`/`edge_uncertainty_mean` ·
`edge_post_retransmit`/`uncertainty_post_ablation` · `controlnet_input_latent` · `diffusion_latent_init`/
`diffusion_latent_final` · `vae_decode_input` · `final_reconstruction`. 각 tensor는 shape/dtype/finite/
NaN·Inf 수/min/max/mean/std/norm/fingerprint(SHA-256)를 기록하고(`tensor_stage_stats.jsonl`), 경로 쌍마다
exact equality/max·mean absolute error/MSE/cosine similarity/norm ratio를 계산한다(`tensor_pair_comparison.csv`).
float32(`bit_depth=32`)는 무손실이므로 `digital_wire`의 직렬화 왕복이 canonical(contiguous, CPU, float32) 기준
bitwise identical인지 별도로 검사한다(`roundtrip_bitexact` 컬럼).

## 판정 기준

`src/sgdjscc_lab/diagnostics/verdict.py::classify`:

1. **transport/latent stage**(`TRANSPORT_STAGES` — sender latent·channel/역직렬화·normalize·power scalar·
   step/SNR·`diffusion_latent_init`)에서 `digital_inprocess`와 `digital_wire`가 다름(그 중 최초 stage) →
   **`packet_tx_rx_issue`**.
   - `edge_mean`/`edge_post_retransmit`/`controlnet_input_latent`/`diffusion_latent_final`/
     `vae_decode_input`/`final_reconstruction` 같은 **edge/decoder stage**(`EDGE_DECODER_STAGES`)는
     baseline ablation에서 기본적으로 이 판정에 포함하지 않는다 — `digital_inprocess`는 edge를 analog
     Canny/WITT 재전송망으로 다시 보내고, `digital_wire`는 packet에서 이미 받은 edge를 그대로 쓰는 것이
     설계상 정상 동작(`transmission/receiver_runtime.py`)이라 이 두 경로가 여기서 다른 것은 그 자체로
     packet 오류의 증거가 아니다. `serialized_raw_edge`/`awgn_edge_retransmit` ablation으로
     `edge_already_received`를 양쪽에서 동일하게 강제한 뒤(`classify(..., edge_handling_equalized=True)`)에만
     이 stage들의 불일치도 증거로 쓴다.
2. 두 digital 경로는 (transport stage 기준) 일치하지만 AWGN보다 낮음(PSNR 기준 ≥ 1.0 dB 차이) →
   **`decoder_pipeline_issue`**
3. `diffusion_bypass_vae_direct` ablation(Canny 재전송·ControlNet edge latent encode·diffusion을 **전부**
   생략하고 받은 latent를 VAE로 바로 복원 — diffusion 호출만 건너뛰는 것이 아니라 edge 처리 자체가 아예
   실행되지 않음)부터 이미 AWGN보다 낮음 → **`latent_normalization_issue`**
4. 위 어느 것도 근거가 부족하면 → **`inconclusive`**

## Ablation (one-factor-at-a-time)

`src/sgdjscc_lab/diagnostics/ablations.py::build_default_ablations`:

| 이름 | 효과 |
|---|---|
| `baseline` | ablation 없음 |
| `controlnet_off` | ControlNet 비활성화 |
| `serialized_raw_edge` | 수신/추출된 edge를 그대로 사용, analog Canny 재전송망 생략 |
| `awgn_edge_retransmit` | digital_wire도 analog Canny 재전송망을 강제로 통과 |
| `latent_only` | 캡션·edge·uncertainty 모두 비활성화 |
| `uncertainty_off` | uncertainty만 0으로 |
| `edge_and_uncertainty_off` | edge+uncertainty 비활성화, 캡션은 유지 |
| `fixed_reference_step` | digital step 정책을 `fixed_reference`로 명시(현재 기본값) |
| `reuse_awgn_step` | digital 경로가 같은 프레임의 AWGN `(cur_step, cur_snr)`를 재사용 |
| `fixed_step` | 모든 경로의 `cur_step`을 리터럴 상수로 고정(`--fixed-step-value`) |
| `diffusion_bypass_vae_direct` | Canny 재전송/ControlNet/diffusion 전체 생략, 수신 latent를 VAE로 직접 복원 |
| `minimal_denoise` | diffusion step 수를 최소화(`--minimal-denoise-steps`) |

## CLI

```bash
# dry-run: 계획만 출력, 아무것도 건드리지 않음
python scripts/diagnose_float32_digital_quality.py \
    --output-root outputs/f32dig_smoke --video-ids 01_person_walk --frames 0 --dry-run

# CPU/mock 구조 검증 (checkpoint/GPU 불필요 — 실제 품질 측정 아님)
python scripts/diagnose_float32_digital_quality.py \
    --output-root outputs/f32dig_smoke --video-ids 01_person_walk --frames 0 \
    --no-models --device cpu

# 서버 실측: 1영상 x 1프레임, 전체 ablation
python scripts/diagnose_float32_digital_quality.py \
    --output-root outputs/f32dig_run --video-ids 01_person_walk --frames 0 \
    --ablations all --device cuda:0

# resume (동일 output-root, run_signature.json 불일치 시 즉시 거부)
python scripts/diagnose_float32_digital_quality.py --output-root outputs/f32dig_run ... --resume
```

주요 옵션: `--video-ids`(comma), `--frames`(`"0"`/`"0,5,9"`/`"0-19"`/`"0-4,10,20-24"`), `--seed`,
`--paths`(subset of `awgn,digital_inprocess,digital_wire`), `--ablations`(`baseline`|`all`|comma list),
`--bit-depth`(기본 32), `--granularity`, `--digital-step-policy`, `--fixed-step-value`,
`--minimal-denoise-steps`, `--no-instrument-tensors`(대규모 다중 프레임 실행에서 tensor 계측 생략),
`--save-tensors`(선택적 `.pt` 저장), `--no-models`(CPU/mock), `--resume`.

### Resume 안전성

이미 완료된 (video, frame, ablation) group을 건너뛰는 것은 항상(플래그와 무관하게) 켜져 있다
(`run_transmission_reduction_eval.py`의 기존 관행과 동일). `--resume`이 실제로 게이트하는 것은 **비어있지
않은 `--output-root`를 재사용해도 되는가**이다 — `--resume` 없이 이미 결과가 있는 `--output-root`를 다시
가리키면 CSV를 중복 기록하는 대신 즉시 거부한다(재현된 "3행→6행" 버그의 수정). `run_signature.json`은 git
commit·dataset manifest hash·config hash·checkpoint hash·seed·video/frame·ablation·bit-depth·granularity·
digital-step-policy·**tensor 계측 여부(`--no-instrument-tensors`)**·`--record-patch-index`를 모두 포함하며,
하나라도 다르면 `--resume`이어도 즉시 거부한다. 판정(`verdicts.jsonl`)은 `path_comparison.csv`와 별도로
`(video, frame)`당 한 줄만 누적 기록되고, 매 실행 시작 시 전부 다시 읽어 `verdict_summary`/`REPORT.md`를
구성하므로 — 이번 실행에서 해당 baseline group이 이미 완료되어 건너뛰었어도 `verdict_summary`가 `None`으로
덮어써지지 않는다.

산출물(`--output-root` 하위): `run_manifest_initial.json`/`run_manifest.json`(commit·argv·config·checkpoint
hash·**dataset manifest hash**·환경 — `utils/run_manifest.py` 재사용), `run_signature.json`(resume 안전성),
`path_comparison.csv`(경로별 PSNR/SSIM/LPIPS/latency/diffusion step/실패 + AWGN 대비 delta — PSNR/SSIM/LPIPS
전부), `tensor_stage_stats.jsonl`, `tensor_pair_comparison.csv`, `verdicts.jsonl`((video, frame)당 판정 1줄,
resume 시 누적 재사용), `failed_cases.csv`(non-finite로 중단된 **path별** 행 — `summary.json`/종료 코드에
쓰이는 실패 건수는 group 수가 아니라 이 CSV의 실제 행 수), `summary.json`, `REPORT.md`(판정·근거·최초 불일치
stage — `--no-models`/dry-run에서는 "진단 환경 구현 완료, 서버 실측 대기"만 기록하고 원인 결론을 절대
주장하지 않음).

## 서버 단일 실행

```bash
bash scripts/run_float32_digital_diagnostics.sh                # profile=full
bash scripts/run_float32_digital_diagnostics.sh --dry-run
bash scripts/run_float32_digital_diagnostics.sh --profile smoke
bash scripts/run_float32_digital_diagnostics.sh --profile short
bash scripts/run_float32_digital_diagnostics.sh --profile full
bash scripts/run_float32_digital_diagnostics.sh --resume outputs/f32dig_<timestamp>
bash scripts/run_float32_digital_diagnostics.sh --cuda-visible-devices 0
```

stage 순서(항목별 `--output-root` 하위 디렉터리로 분리):

1. preflight — git commit provenance, dataset manifest, checkpoint 4종, 디스크 여유, GPU/CUDA/NVML
   (`--device cpu`면 GPU 검사 생략)
2. 관련 테스트(`tests/test_float32_digital_diagnostics.py` 등) + dry-run
3. `stage3_single_frame_paths/` — 1영상(`01_person_walk`) x 1프레임, 3경로 + tensor 계약 검사
4. `stage4_single_frame_ablations/` — 동일 (영상, 프레임), 전체 ablation
5. `stage5_paired_frames/` — 1영상 x N프레임(`smoke`=2, `short`=5, `full`=20) paired 진단
   (`baseline` + `diffusion_bypass_vae_direct`)
6. `stage6_core_conditions/` — 3영상(`01_person_walk` 일반 움직임, `07_person_enter` semantic 변화,
   `09_scene_cut_chair_car` scene cut) x N프레임(`smoke`=3, `short`=10, `full`=100), `baseline`만,
   tensor 계측 생략(`--no-instrument-tensors`, 규모상 metrics만)
7. 결과 검증 + 산출물 SHA-256 해시 + `INTEGRATED_REPORT.md`(각 stage의 `summary.json`/`verdicts.jsonl`을
   실제로 읽어 stage별 dominant verdict·판정 개수·실패 건수 표와 전체 통합 판정을 만든다 — 파일 해시 목록만이
   아니다)

불필요한 ablation Cartesian product는 만들지 않는다 — ablation 전체 스윕은 stage 4(1영상 x 1프레임)로만
한정하고, 다중 프레임/다중 영상 stage는 `baseline`(+ stage 5의 `diffusion_bypass_vae_direct`)만 실행한다.

- python 인터프리터 탐색: `PYTHON_BIN` 환경변수 명시 > conda `ptest` env 활성화(conda가 PATH에 있을 때) >
  `~/anaconda3`/`~/miniconda3`/`~/miniforge3`/`/opt/conda`/`/usr/local/anaconda3`의 `envs/ptest/bin/python`
  순으로 시도 — 매 후보는 실제로 `import torch`가 성공하는지 검증한 뒤에만 채택한다(비대화형 셸에서 conda가
  PATH에 없어 시스템 python으로 조용히 넘어가는 문제의 수정). 전부 실패하면 `PYTHON_BIN`을 직접 지정하라는
  메시지와 함께 즉시 종료한다.
- 매 stage(2~6)의 stdout/stderr·시작/종료 시각·소요 시간(초)·종료 코드가
  `$OUTPUT_ROOT/stage_logs/<stage_name>.log`에 보존된다(터미널에도 동시에 출력 — `tee`).
- preflight 실패 → 즉시 종료(아무 stage도 실행되지 않음)
- 독립 stage(3~6) 실패 → 기록 후 계속 진행, 최종 exit code는 non-zero(3)
- SIGINT/SIGTERM → 현재 실행 중인 Python 단계가 자체적으로 manifest/summary/report를 저장한 뒤 종료 코드
  130으로 끝나고, 셸 드라이버는 즉시 나머지 stage를 건너뛰며 exit 130 — 동일 `--output-root`로 `--resume` 재실행
- OOM 무한 재시도 없음 — 매 stage는 정확히 한 번만 실행되고, 재시도는 항상 사용자가 명시적으로 재실행(선택적
  `--resume`)해야 함

## 상태

**진단 환경 구현 완료, 서버 실측 대기.** 이 문서와 harness 자체는 CPU/mock 테스트와 dry-run으로만 검증되었다
(`tests/test_float32_digital_diagnostics.py`, 40개 테스트 통과 — routing·float32 round-trip·tensor 비교·ablation
효과(VAE-direct bypass가 Canny/ControlNet을 실제로 호출하지 않는지 포함)·NaN 전파·decode parity·verdict 분류
(edge 비대칭 오탐 방지 포함)·resume 안전성(중복 방지·판정 보존)·CLI end-to-end). 실제 GPU 추론 기반 판정
(`packet_tx_rx_issue`/`decoder_pipeline_issue`/`latent_normalization_issue`)은 `scripts/
run_float32_digital_diagnostics.sh`를 서버에서 실행한 뒤에만 유효하다 — 이 문서는 원인 해결이나 품질 정상화를
주장하지 않는다.

## 관련 문서

- [docs/experiments/2026-08-26_transmission_normalization.md](../experiments/2026-08-26_transmission_normalization.md) — float32 digital 품질 저하가 처음 관측된 실험
- [docs/protocols/transmission_normalization.md](./transmission_normalization.md) — 전송 정상화 실행 절차(이 harness가 재사용하는 run manifest/seed/resume 패턴의 출처)
- [docs/architecture/tx_rx_contract.md](../architecture/tx_rx_contract.md) — Tx/Rx 계약
- [docs/current/roadmap.md](../current/roadmap.md) §2, [docs/current/open_issues.md](../current/open_issues.md) — 이 진단이 해결하려는 미해결 항목

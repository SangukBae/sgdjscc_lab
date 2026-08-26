---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes: docs/etri_overview.md
---

> [← 문서 색인](../README.md)

# 시스템 개요

## 과제 목표

- 생성 AI 기반 **시맨틱 미디어 전송 신뢰성(Semantic Media Transmission Reliability)**을
  정량 평가하는 End-to-End 시뮬레이션 프레임워크 개발.

> 핵심: 선명한 복원(PSNR 최대화)이 아니라, **노이즈 채널을 통과한 뒤에도 원본의
> 의미(semantic intent)가 얼마나 신뢰성 있게 전달되는가**를 측정한다.

## 핵심 연구 문제

1. **시간축·영상 신뢰성** — 정지 이미지 평가만으로는 시간 흐름·장면 전환·프레임 간
   의미 일관성을 볼 수 없다. 영상 확장이 필요하다.
2. **수신단 생성 복원의 할루시네이션** — 확산 기반 복원은 그럴듯해 보여도 없던
   객체를 만들거나 있어야 할 정보를 누락·왜곡할 수 있다. 이를 검출·억제하는
   장치가 필요하다.
3. **평가 체계 신뢰도** — PSNR/SSIM/CLIP만으로는 시간축 의미 일관성, 객체 깜빡임,
   의미 drift를 충분히 설명하지 못한다. 재생성 판단에 쓰는 지표와 최종 평가
   지표가 같으면 순환 평가가 생긴다.

- 세 문제에 대해 지금 무엇이 구현/검증됐는지는 [current/status.md](../current/status.md),
  남은 과제는 [current/roadmap.md](../current/roadmap.md)를 따른다. 세 문제의 원인·해법
  설계는 [tx_rx_contract.md](./tx_rx_contract.md)에 있다.

## 시스템 파이프라인

```
Original Image / Keyframe
  → [Tx]  JSCC 시맨틱 인코더 (VAE latent, scaling 15.45 + MuGE 구조 가이드 + L2-norm)
  → [Ch]  무선 채널 (AWGN 기본 / Rayleigh·fast-fading·packet-drop opt-in)
  → [Rx]  확산 복원 (MDTv2 + 선택적 ControlNet, blind SNR → step matching → 디노이징 → decode)
  → [Eval] 시맨틱 일관성 & 할루시네이션 평가
  → outputs/results.csv
```

- 공식 지표 정의(SRS, CSV 컬럼, `PTC`/`SFR`/`SDI`)는 [metrics.md](./metrics.md) 참고.

## 저장소 & 모듈 구조

```
Semantic/
├── sgdjscc_lab/        ← PRIMARY 개발 패키지
├── SGDJSCC/            ← 원본 READ-ONLY (논문 베이스라인, 런타임 재사용)
├── CLIP/ Deep-JSCC-PyTorch/ DiffJSCC/ WITT/ POPE/ diffusers/  ← 외부 베이스라인/참고
└── reference/           ← 매뉴스크립트 + 참고 코드
```

- `src/sgdjscc_lab/` 디렉터리 구성:

```text
src/sgdjscc_lab/
├── config.py, paths.py, runtime.py, io.py, phase_gates.py, paper_mode.py
├── core/           stage definitions · noise schedule · semantic vocabulary
├── channels/       awgn · rayleigh · fast_fading · packet_drop · digital_packet · measurement · complex_ops
├── guidance/       text · edge · depth · segmentation · semantic_packet · object · relation
├── models/         jscc_model · diffusion_wrapper(_channel) · edge_jscc · csi_estimation
├── pipelines/      infer · eval · train · regeneration_loop · channel_conditioned_infer
├── evaluators/     quality · clip_score · object_preservation · hallucination(_vqa)
│                   · semantic_reliability(_v2) · packet_matcher · temporal_consistency · fid
├── controllers/    adaptive_guidance · snr_guidance · regeneration · channel_condition · search
├── acceleration/   ddim_sampler · consistency_decoder · early_exit · latency_profiler · water_filling
├── transmission/   real digital packet path — quantization(4/6/8/16-bit) · wire_packet(checksum)
│                   · packet_bundle · byte_accounting · receiver_runtime
├── video/          keyframe · temporal_pipeline · generation contracts/backends/worker/factory
├── training/       stage validation · stage_runners · losses · freeze · interrupt · perf
├── data/           datasets · image_dataset · transforms
└── utils/          preprocessing · csv_logger · metrics_io · metric_profiles · packet_io · seed
```

- 파일별 실행 흐름과 역할 지도는 [reference/framework_file_roles.md](../reference/framework_file_roles.md)에
  있다.

## 개발 원칙

1. **알고리즘 경로 보존** — `SGDJSCC/inference_one.py`의 순전파 수치 변경 금지
   (VAE scaling `15.45`, AWGN 공식, blind SNR, step matching 등).
2. **관심사 분리** — 채널은 `channels/`, 가이드는 `guidance/`, 모델은 `models/`,
   오케스트레이션은 `pipelines/`, 지표는 `evaluators/`로 독립 교체 가능하게 둔다.
3. **원본 읽기 전용** — 새 아이디어는 `SGDJSCC/`가 아니라 `sgdjscc_lab/`에 구현한다.

- 모든 opt-in 확장(패킷 검증, 채널 조건화, 영상 확장, 저지연 샘플링 등)은 상위
  게이트(`use_phase4`/`use_phase5`, `phase_gates.py`)가 꺼져 있으면 원본 SGD-JSCC
  추론과 수치적으로 동일하게 동작한다. `paper_mode`(`paper_mode.py`)는 별도로
  논문 재현 경로만 허용하는 guardrail이다 — [protocols/reproducibility.md](../protocols/reproducibility.md) 참고.

## 관련 문서
- [metrics.md](./metrics.md) — SRS·CSV 컬럼·시간축 지표 정의
- [tx_rx_contract.md](./tx_rx_contract.md) — Tx/Rx 모듈 설계, 영상 확장 시스템 구조
- [reference/framework_file_roles.md](../reference/framework_file_roles.md) — 파일별 실행 흐름
- [current/status.md](../current/status.md) — 현재 구현 상태

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

- 개발 대상
  - 생성 AI 기반 End-to-End 시뮬레이션 프레임워크
- 핵심 평가 대상
  - 노이즈 채널 이후의 의미 전달 신뢰성
- 비핵심 목표
  - PSNR만 최대화하는 선명도 경쟁

## 핵심 연구 문제

1. **시간축·영상 신뢰성**
   - 문제: 정지 이미지 지표로 시간 흐름·장면 전환·프레임 일관성을 평가하기 어려움
   - 방향: 영상 파이프라인과 시간축 지표 확장
2. **수신단 생성 복원의 할루시네이션**
   - 문제: 객체 추가·누락·왜곡 가능성
   - 방향: 검출·제어·제한적 재생성
3. **평가 체계 신뢰도**
   - 문제: PSNR/SSIM/CLIP만으로 의미 drift와 객체 깜빡임을 설명하기 어려움
   - 위험: 제어 지표와 보고 지표를 재사용하면 순환 평가 발생
- 연결 문서
  - 현재 상태: [current/status.md](../current/status.md)
  - 향후 계획: [current/roadmap.md](../current/roadmap.md)
  - 원인·해법 설계: [tx_rx_contract.md](./tx_rx_contract.md)

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

- 파일별 실행 흐름: [framework_file_roles.md](../reference/framework_file_roles.md)

## 개발 원칙

1. **알고리즘 경로 보존**
   - 대상: `SGDJSCC/inference_one.py`
   - 불변값: VAE scaling `15.45`, AWGN, blind SNR, step matching
2. **관심사 분리**
   - 채널: `channels/`
   - 가이드: `guidance/`
   - 모델: `models/`
   - 오케스트레이션: `pipelines/`
   - 지표: `evaluators/`
3. **원본 읽기 전용** — 새 아이디어는 `SGDJSCC/`가 아니라 `sgdjscc_lab/`에 구현한다.

- 호환성 규칙
  - opt-in 확장: 패킷 검증·채널 조건화·영상 확장·저지연 샘플링
  - 상위 게이트: `use_phase4`, `use_phase5`, `phase_gates.py`
  - 게이트 off: 원본 SGD-JSCC 추론과 수치 동일
  - `paper_mode`: 논문 재현 경로만 허용
  - 세부 규칙: [재현성 프로토콜](../protocols/reproducibility.md)

## 관련 문서
- [metrics.md](./metrics.md) — SRS·CSV 컬럼·시간축 지표 정의
- [tx_rx_contract.md](./tx_rx_contract.md) — Tx/Rx 모듈 설계, 영상 확장 시스템 구조
- [reference/framework_file_roles.md](../reference/framework_file_roles.md) — 파일별 실행 흐름
- [current/status.md](../current/status.md) — 현재 구현 상태

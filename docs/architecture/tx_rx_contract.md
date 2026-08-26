---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes: docs/video_extension_lgvsc.md, docs/phase4.md, docs/phase5.md
---

> [← 문서 색인](../README.md)

# Tx/Rx 시스템 설계

- 범위
  - 장기 Tx/Rx 설계
  - 구현 상태: [current/status.md](../current/status.md)
  - 검증 결과: `docs/experiments/`

## 1. 이미지 경로 (Tx → Ch → Rx)

```text
[프레임 단위]
  JSCC 인코딩 (VAE latent + Canny)   models/jscc_model.py
    → 채널 (AWGN 기본, opt-in Rayleigh/fast-fading/packet-drop/digital_packet)
                                      channels/
    → 이미지 diffusion 복원          models/diffusion_wrapper.py (MDTv2+ControlNet)
```

- 설계 원칙
  - 원본 이미지 경로 보존
  - 신규 기능: `use_phase4`, `use_phase5` 뒤에 배치
  - 기본값: off
  - gate off: 원본 SGD-JSCC와 수치 동일

## 2. 의미 단위(packet) 검증 계약 — Rx-legal self-verification

- 비교 입력
  - 전송 semantic packet
  - 복원 결과에서 재추출한 packet
- 검출 대상
  - 객체 추가·누락
  - 관계·구조 왜곡

- Rx-legal 제약
  - 판정 기준: 수신단이 보유한 전송 packet
  - 원본 직접 대조: eval-only
  - 수신단 원본 참조: 금지

```text
packet_verifier: 전송 packet vs 복원 packet 비교
  → severity score + 오류 유형(추가/누락/구조 왜곡)
  → verifier_controller: 오류 유형별 accept/suppress_extra/
     strengthen_missing/strengthen_structure_guidance/fallback_recompute 결정
```

- 객체 판정 backend
  - `clip`, `owlv2`, `vqa`, `gt`
- 불변 계약
  - 전송 packet 기준 판정
  - 정의: [metrics.md](./metrics.md)

## 3. 적응형 가이드 계약

- 입력: 추정 SNR
- 정책
  - 저 SNR: 강한 guidance + 최대 step
  - 고 SNR: 약한 guidance 또는 skip
- 불변 조건
  - 이미지 복원 경로 유지
  - 실행 config만 선택

## 4. 채널 조건화 계약 (DiffCom 영감)

- 수신된 채널 신호 자체를 복원 조건으로 쓴다.

```text
channel.observe() → MeasurementBundle (received/equalized/gain/noise_var/mask/SNR/reliability)
  → 조건 인코더 (models/channel_condition_encoder.py)
  → reliability head → guidance/steps 스케일, 조건 토큰 부착
  → diffusion_wrapper_channel.py (adapter 레벨 조건화)
```

- 현재 근사
  - 조건 token: config에 부착
  - frozen denoiser: 조건 token 미사용
  - 실제 조건화: received-latent initialization + reliability 기반 guidance·step 조절
- 향후 필요
  - 조건 인식 denoiser 재학습
  - FiLM·cross-attention·posterior-gradient guidance 검토

## 5. 영상 확장 — LGVSC 참고 설계

- 성격
  - LGVSC-inspired extension
  - faithful reproduction 아님
- 대응 범위
  - PSSS/SKEM
  - 키프레임·텍스트·side-info 분리 전송
  - world model + DSA 세그먼트 생성
- 해석 원칙
  - 역할·인터페이스만 대응
  - 개별 모듈은 논문 구현과 다를 수 있음

### 5.1 LGVSC 파이프라인과 sgdjscc_lab 대응

| LGVSC 구성요소 | 역할 | sgdjscc_lab 대응 | 성격 |
|---|---|---|---|
| **PSSS** (S_rel = P(No)−P(Yes)) | MLLM 확률 기반 프레임쌍 의미 유사도 | `video/psss.py::MllmTokenProbPsssBackend` | 실제 구현 (yes/no 토큰 logits→softmax) |
| **SKEM** | PSSS 유도 자기회귀 키프레임 선택 | `video/skem_selector.py::PsssKeyframeSelector` | 실제 구현 (variable-length) |
| **SKIM** | 고정 간격 분할 | `video/keyframe_extractor.py::FixedIntervalKeyframeSelector` | 실제 구현 (논문 문자 그대로) |
| **I_text** 캡션 | 세그먼트 텍스트 시맨틱 | `guidance/text_extractor.py` (BLIP2/Qwen) | 재사용 |
| **I_side** 사이드 (논문: optical flow) | 모션 힌트 | `video/motion_residual.py` (block residual) | 근사 — 광류 아님 |
| **NTSCC** 키프레임 전송 | 키프레임 딥 JSCC | `models/jscc_model.py` + `channels/` | 역할 대응 — 동일 모델 아님 |
| 캡션·사이드 비트 전송 | 디지털 경로(LDPC+변조) | 가이드 손상 규칙(token dropout) | 근사 — AWGN 대신 dropout 계열 |
| **World model**(Open-Sora) | 조건부 세그먼트 생성 | `video/video_generator.py` (`ExternalSegmentWorkerGenerator`, mock/svd/wan/callable backend) | 역할 대응 |
| **SFA/DSA** | 가변 길이 세그먼트 적응 | segment 계약의 길이 파라미터 | DSA-inspired 인터페이스 — VAE latent dimension 동적 조정 자체는 아님 |
| 세그먼트 연결 | 임의 길이 영상 조립 | `video/temporal_pipeline.py` (3-way 분기 + mp4 IO) | 확장 |

> **손상 모델 주의** — 논문은 텍스트·사이드도 AWGN + LDPC + 변조로 보낸다.
> 이 프로젝트는 가이드에 AWGN을 직접 걸지 않고 token dropout 계열 손상을
> 쓴다(가이드 손상 규칙, 아래 [평가 프로토콜](../protocols/evaluation.md) 참고).
> 논문 faithful한 LDPC+변조 경로는 별도 비교 baseline으로만 남겨둔다.

### 5.2 세그먼트 생성 계약 (Rx-legal)

- `video_generator`는 단일 frame 함수가 아니라 **GOP/segment 단위 생성 계약**이다:

```python
SegmentGenerationRequest(
    start_keyframe_recon, start_keyframe_index,
    end_keyframe_recon=None, end_keyframe_index=None,   # bidirectional일 때만
    captions, packets, side_infos, fps, segment_length, target_indices,
) → VideoGenerator.generate_segment() → SegmentGenerationResult
```

- Rx-legal 경계
  - 요청 구조체에 원본 target frame 필드 없음
  - 허용 입력: 복원 keyframe·caption·packet·side-info
  - 원본 대조: eval-only

- 결과 검증: `validate_segment_result()`
  - frame 수
  - 순서
  - shape
  - metadata
  - 외부 process backend에는 필수 적용

### 5.3 3-way 프레임/세그먼트 정책

```text
분기 게이트: semantic delta(캡션 변화) + motion_residual 이중 게이트
  → 정지 구간            → reuse      (키프레임 복원 재사용)
  → 의미 변화 큼/키프레임 → recompute  (이미지 파이프라인 재계산, pipelines/infer_pipeline.py)
  → 모션 있으나 재사용 불가 → generate  (video_generator 세그먼트 생성)
```

- `reuse`/`recompute`는 생성기를 거치지 않는다. `generate`만 위 5.2 계약을 탄다.

### 5.4 최종 시스템 블록 다이어그램

```text
                ┌─ 송신단 ─────────────────────┐   ┌─ 채널 ──────────┐   ┌─ 수신단 ──────────────────────────────────────┐
                │ 키프레임 선택                  │   │                 │   │ 이미지 diffusion 복원 ─┐                       │
                │  scene|interval|PSSS          │   │                 │   │  (키프레임만)          ├─▶ 3-way 프레임/GOP 정책 │
                │  ├─▶ JSCC 인코더 ─────────────┼──▶│ 채널(AWGN 등) ──┼──▶│                        │    reuse|recompute|    │
                │  │   VAE latent+Canny(≙NTSCC) │   │                 │   │ 복호 캡션·사이드 ───────┘    generate            │
                │  └─▶ 캡션·모션 사이드 ─────────┼──▶│ 가이드 손상 ────┼──▶│  (+ 세그먼트 길이)          + 이중 게이트         │
                │      I_text + I_side          │   │ (token dropout) │   │                              │                   │
                └───────────────────────────────┘   └─────────────────┘   │  비디오 세그먼트 생성 ▼(≙world model+DSA)       │
                                                                            │  세그먼트 연결 → mp4 인코딩 + 지표 CSV          │
                                                                            └──────────────────────────────────────────────┘
```

- 평가 계층
  - 비교: 원본 vs 복원
  - 구현: `evaluators/temporal_consistency.py`
  - 지표: `temporal_srs`, `srs_flicker`, `PTC`, `SFR`, `SDI`
  - 정의: [metrics.md](./metrics.md)

## 6. 전제와 리스크 (설계 차원에서 항상 유효)

| 구분 | 내용 |
|---|---|
| 실시간이 아니다 | 논문 실측 기준 SKEM 전처리·world model 생성 모두 오프라인·엣지클라우드급 지연. 저지연 축(early-exit/DDIM)과는 별도 트랙. |
| 생성 비중↑ = 할루시네이션 위험↑ | 안 보낸 프레임을 통째로 생성하므로 원본에 없던 움직임·객체를 지어낼 수 있다 — packet verifier·시간축 지표가 이 위험을 채점하는 안전망. |
| 논문 재현이 아닌 근사 | NTSCC≈기존 JSCC, DSA≈길이 파라미터 인터페이스, I_side≈block residual proxy 모두 논문과 동일 구현이 아닌 역할 대응이다. LGVSC의 CBR·성능 수치는 이 설계로 직접 재현되지 않으며 별도 실측 대상이다. |
| value-per-bit 선택기의 반사실 추정 문제 | "이 키프레임을 안 보내면 생성기가 얼마나 못 하나?"는 Tx에서 알 수 없다(Tx엔 Rx 생성기 출력이 없음) — Rx-legal verifier가 사후적으로 같은 목적을 반사실 추정 없이 달성한다. |
| 지표 순환 평가 위험 | 재생성 구동에 쓴 지표로 우위를 보고하면 결과가 부풀려진다 — loop-internal/held-out 분리 원칙([metrics.md](./metrics.md)) 필수. |

## 관련 문서
- [system.md](./system.md) — 전체 파이프라인 개요
- [metrics.md](./metrics.md) — 이 설계가 검증하는 지표 정의
- [current/status.md](../current/status.md) — 이 설계 중 무엇이 구현·검증됐는지
- [current/roadmap.md](../current/roadmap.md) — 이 설계 위에 남은 연구 과제(학습형 adapter/selector/critic)
- 논문 원문: `reference/paper/LGVSC…/main.tex`

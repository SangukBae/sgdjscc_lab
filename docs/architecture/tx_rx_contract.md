---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes: docs/video_extension_lgvsc.md, docs/phase4.md, docs/phase5.md
---

> [← 문서 색인](../README.md)

# Tx/Rx 시스템 설계

이 문서는 **장기적으로 유효한 설계**만 다룬다. 무엇이 실제로 구현·검증됐는지는
[current/status.md](../current/status.md), 실험 결과는 `docs/experiments/`,
Phase 4/5 시절의 상세 구현 기록은 `docs/archive/phase4_2026-07.md` /
`docs/archive/phase5_2026-07.md`(과거 스냅샷)를 참고한다.

## 1. 이미지 경로 (Tx → Ch → Rx)

```text
[프레임 단위]
  JSCC 인코딩 (VAE latent + Canny)   models/jscc_model.py
    → 채널 (AWGN 기본, opt-in Rayleigh/fast-fading/packet-drop/digital_packet)
                                      channels/
    → 이미지 diffusion 복원          models/diffusion_wrapper.py (MDTv2+ControlNet)
```

설계 원칙 — **이미지 경로는 한 줄도 바꾸지 않는다.** 새 기능은 전부 게이트
(`use_phase4`/`use_phase5`) 뒤에 두고 기본 off — 게이트가 꺼지면 원본 SGD-JSCC와
수치 동일.

## 2. 의미 단위(packet) 검증 계약 — Rx-legal self-verification

전송 semantic packet(캡션·객체·장면·관계·속성·엣지/세그/depth 요약)과 복원
결과에서 재추출한 packet을 비교해 추가/누락 객체, 관계·구조 왜곡을 검출한다.

핵심 제약: **판정 기준은 원본 이미지가 아니라 수신단이 실제로 보유한 전송
packet이다.** 원본 프레임과의 직접 대조는 eval-only이며, 수신단 로직으로
배치할 수 없다(수신단은 원본을 가질 수 없으므로). 이 경계를 어기면 "Rx가
원본을 몰래 참조한다"는 실험 무결성 문제가 된다.

```text
packet_verifier: 전송 packet vs 복원 packet 비교
  → severity score + 오류 유형(추가/누락/구조 왜곡)
  → verifier_controller: 오류 유형별 accept/suppress_extra/
     strengthen_missing/strengthen_structure_guidance/fallback_recompute 결정
```

객체 존재 판정은 여러 backend(`clip`/`owlv2`/`vqa`/`gt`)로 교체 가능하며,
어떤 backend를 쓰든 이 계약(전송 packet 기준 판정) 자체는 바뀌지 않는다 —
정의는 [metrics.md](./metrics.md).

## 3. 적응형 가이드 계약

추정 SNR에 따라 가이드 강도·확산 스텝을 조절한다: 저 SNR에서는 강한 가이드 +
최대 스텝, 고 SNR에서는 약한 가이드 또는 skip 경로. 이 결정은 이미지
복원 경로 자체를 바꾸지 않고, 그 경로가 **어떤 config로 실행될지**만 선택한다.

## 4. 채널 조건화 계약 (DiffCom 영감)

수신된 채널 신호 자체를 복원 조건으로 쓴다.

```text
channel.observe() → MeasurementBundle (received/equalized/gain/noise_var/mask/SNR/reliability)
  → 조건 인코더 (models/channel_condition_encoder.py)
  → reliability head → guidance/steps 스케일, 조건 토큰 부착
  → diffusion_wrapper_channel.py (adapter 레벨 조건화)
```

근사인 지점: 인코딩된 조건 토큰은 cfg에 부착되지만 **frozen SGD-JSCC denoiser가
소비하지 않는다** — FiLM/cross-attention/posterior-gradient guidance는 재학습된
조건 인식 denoiser가 있어야 가능하다. 현재는 received-latent init + reliability
스케일 guidance/steps로만 작동한다.

## 5. 영상 확장 — LGVSC 참고 설계

LGVSC 논문(Ma et al., "LGVSC: A Large-Model-Driven Generative Video Semantic
Communication Framework")의 구조를 참고해, 임의 길이 비디오의 전송·복원
설계에 대응시킨 것이다. **LGVSC의 faithful 재현이 아니라 LGVSC-inspired
extension**이다 — 논문의 핵심 3축(PSSS/SKEM, 키프레임·텍스트·사이드 분리 전송,
world model+DSA 세그먼트 생성)의 *역할과 인터페이스*를 sgdjscc_lab 구조에
옮겨오되, 개별 모듈은 논문과 동일 구현이 아닌 대응물이다.

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

`video_generator`는 단일 frame 함수가 아니라 **GOP/segment 단위 생성 계약**이다:

```python
SegmentGenerationRequest(
    start_keyframe_recon, start_keyframe_index,
    end_keyframe_recon=None, end_keyframe_index=None,   # bidirectional일 때만
    captions, packets, side_infos, fps, segment_length, target_indices,
) → VideoGenerator.generate_segment() → SegmentGenerationResult
```

Rx-legal 경계: 요청 구조체에는 **원본(미전송) target 프레임 필드가 아예 없다**.
수신단은 자신이 복원한 keyframe과 전송받은 캡션/packet/side-info만으로
중간 프레임을 생성해야 한다 — 원본 대조는 eval-only.

`validate_segment_result()`가 반환된 결과의 frame 수·순서·shape·metadata가
요청과 실제로 일치하는지 계약 수준에서 검사한다(외부 프로세스 backend는
신뢰 경계이므로 이 검증이 필수).

### 5.3 3-way 프레임/세그먼트 정책

```text
분기 게이트: semantic delta(캡션 변화) + motion_residual 이중 게이트
  → 정지 구간            → reuse      (키프레임 복원 재사용)
  → 의미 변화 큼/키프레임 → recompute  (이미지 파이프라인 재계산, pipelines/infer_pipeline.py)
  → 모션 있으나 재사용 불가 → generate  (video_generator 세그먼트 생성)
```

`reuse`/`recompute`는 생성기를 거치지 않는다. `generate`만 위 5.2 계약을 탄다.

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

평가 계층은 원본과 복원을 비교한다 — `evaluators/temporal_consistency.py`의
`temporal_srs`/`srs_flicker`/`PTC`/`SFR`/`SDI` (정의는 [metrics.md](./metrics.md)).

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

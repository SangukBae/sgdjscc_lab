> [← 문서 색인](./README.md)

# LGVSC를 참고한 비디오 전송·복원 확장 설계

LGVSC 논문(Ma et al., "LGVSC: A Large-Model-Driven Generative Video Semantic
Communication Framework", IEEE)의 구조를 현재 `sgdjscc_lab` 코드에 대응시켜,
어떤 모듈을 재사용·확장·신설하면 임의 길이 비디오의 전송과 복원이 가능해지는지
정리한 설계 문서다. [etri_strategy.md](./etri_strategy.md)의 핵심 한계 3
(정지 이미지 중심 한계)의 장기 해결 축(video diffusion)에 해당한다.

- 논문 원문: `reference/paper/LGVSC…/main.tex`
- 시각화 원본(Artifact): <https://claude.ai/code/artifact/deee634a-9077-4d44-b3ae-2f2a76a8d1b0>

> **문서 성격 (중요)** — 이 문서는 LGVSC 논문의 **faithful 재현(reproduction)이
> 아니라 LGVSC-inspired extension**이다. 즉 논문의 핵심 3축(PSSS/SKEM,
> 키프레임·텍스트·사이드 분리 전송, world model+DSA 세그먼트 생성)의 **역할과
> 인터페이스**를 sgdjscc_lab 구조에 옮겨오되, 개별 모듈은 논문과 동일 구현이
> 아닌 대응물이다. 특히 아래 세 지점은 "논문과 같다"가 아니라 "논문에서 착안한
> 근사"임을 문서 전반에서 보수적으로 읽어야 한다:
> - **NTSCC ≈ 기존 JSCC** — 동일 모델이 아니라 "키프레임 딥 JSCC 전송 경로"라는 역할 대응
> - **DSA ≈ 길이 파라미터 인터페이스** — VAE latent dimension 동적 조정이 아니라 DSA-inspired 계약
> - **I_side ≈ motion_residual** — optical flow가 아니라 block residual proxy (paper-aligned 버전은 RAFT/flow)
>
> LGVSC의 성능·CBR(10⁻⁴~10⁻³) 재현 가능성도 이 문서 범위 밖이며, 별도 실측으로만
> 확인할 수 있다.

**작업 구분 범례** — 문서 전체에서 다음 표기를 쓴다.

| 표기 | 의미 |
|---|---|
| `[재사용]` | 현재 코드 그대로 |
| `[확장]` | 기존 모듈에 모드 추가 |
| `[신규]` | 새 모듈 작성 |

## 1. LGVSC 논문의 파이프라인

핵심 아이디어 세 가지 — ① MLLM 확률 기반 의미 유사도(PSSS)로 키프레임을
고르고(SKEM), ② 키프레임·캡션·사이드 정보를 **서로 다른 전송 경로**로 보내고,
③ 수신단의 world model(Open-Sora)이 가변 길이 어댑터(DSA)로 세그먼트를
**생성**해 이어붙인다. CBR 10⁻⁴~10⁻³ 수준의 초저대역 전송을 달성한다.

PSSS: `S_rel = P("No") − P("Yes")` — MLLM의 yes/no 토큰 확률차를 연속 유사도
점수로 사용. `η_th = 0.35` 초과 시 새 키프레임.

```text
[송신단]
  입력 영상 𝒳 (임의 길이 F프레임)
    → 키프레임 선택: SKIM(고정 간격) 또는 SKEM(PSSS 유도, InternVL 캡션 비교)
    → 세그먼트 인코딩: 세그먼트별 {캡션 I_text, 키프레임 I_frame, 사이드 I_side(광류)}

[채널]
  키프레임        → NTSCC(딥 JSCC)로 아날로그 심벌 전송 ─┐
  캡션·사이드     → 채널코딩(LDPC)+변조, 디지털 전송      ├→ AWGN 채널 (SNR 6~10 dB 실험)
                                                        ─┘
[수신단]
  NTSCC 복호 키프레임
    → World model(Open-Sora)이 {복원 키프레임 + 캡션 + 사이드}를 조건으로 세그먼트 생성
    → SFA/DSA 어댑터 (DSA: 세그먼트 길이에 맞춰 VAE latent 차원을 동적 조정 → 임의 길이 지원)
    → 세그먼트 연결: 𝒳̂ = {Ŝ₁; Ŝ₂; …; Ŝ_N}
```

## 2. 현재 sgdjscc_lab 구조

모델은 **이미지 단위**로만 동작한다. 비디오 계층(`video/`)은 그 위의
오케스트레이션으로, inter-frame을 "키프레임 복원 **복사**" 아니면 이미지
파이프라인 **재계산** 둘 중 하나로 처리한다 — 생성으로 메꾸는 경로가 없어
정지-점프와 깜빡임이 생긴다.

> **2026-07 ETRI 1차 구현 반영** — 아래 §6.3 로드맵의 1~3단계에 해당하는
> 기반은 구현됐다: mp4 입출력(`utils/video_io.py`), 복원 frame/mp4 저장,
> `PTC`/`SFR`/`SDI` 시간축 지표(**초기/잠정** — CLIP/packet 기반),
> keyframe-anchored **의미 델타 + 모션 이중 게이트**(기본 off), GOP
> `SegmentRecord` 추상화(`video/segment.py`). generate 분기(4~5단계)와
> verifier 고도화(6단계)는 여전히 후속이다. 진행 상태의 단일 기준은
> [etri_strategy.md](./etri_strategy.md)의 "1차 구현 결과" 절이다.

```text
[프레임 단위 — 전부 재사용 대상]
  JSCC 인코딩 (VAE latent + Canny)   models/jscc_model.py
    → AWGN 채널                      channels/awgn.py
    → 이미지 diffusion 복원          models/diffusion_wrapper.py (MDTv2+ControlNet, 프레임별 독립)

[비디오 계층 — Phase 4-B + 1차 확장]
  (mp4 입력 시) mp4 → frames         utils/video_io.py
  장면 전환 감지                     video/scene_change_detector.py
    → 2-way 정책 + 이중 게이트       video/temporal_pipeline.py
        의미 델타 < θ AND 모션 < θ_m → 키프레임 복원 복사 (reuse)
        아니면                        → 프레임 재계산 (recompute_semantic|_motion)
        (모션 게이트는 temporal.motion_threshold 설정 시에만 활성, 기본 off)
    → 지표·구조 출력                 scripts/evaluate_video.py
        keyframes.json · segments.json · temporal CSV (PTC/SFR/SDI 포함)
        · 복원 frame folder · (옵션) 복원 mp4
```

## 3. 제안: LGVSC 매핑 목표 구조 (실행 흐름 기준)

설계 원칙 — **이미지 경로는 한 줄도 바꾸지 않는다.** 키프레임 전송은 LGVSC의
NTSCC 자리에 기존 SGD-JSCC 경로를 그대로 쓰고(교체 불필요), 캡션·사이드는 기존
가이드 손상 규칙(AWGN 금지, dropout 계열)을 따른다. 새 기능은 전부 게이트
(`use_video_gen`) 뒤에 두고 기본 off — 게이트가 꺼지면 현재 Phase 4-B와 수치
동일.

> **손상 모델 주의** — LGVSC 논문은 텍스트·사이드도 **AWGN 채널 위의 디지털
> 전송(LDPC + 변조)** 으로 보낸다. 반면 이 프로젝트의 현재 정책은 가이드에
> AWGN을 직접 걸지 않고 **token dropout 계열 손상**을 쓴다(가이드 손상 규칙,
> [etri_strategy.md](./etri_strategy.md)). 즉 현재 설계는 dropout 손상이 기본이고,
> **논문 faithful한 LDPC+변조 경로는 추후 비교 baseline**으로 둔다.

실제 호출 순서를 따른다: `scripts` 진입 → `config`/게이트 →
`runtime.build_models()` → `temporal_pipeline.run()` 프레임 루프 → 출력·평가.

```text
① 진입·설정
  scripts/evaluate_video.py [구현됨]  --input clip.mp4 입력 → 프레임 추출 IO (1차 완료)
    → config.py · configs/composed_video.yaml [재사용]  _defaults_ 프래그먼트 병합
    → phase_gates.py [확장]  use_video_gen 신설 — 활성 조건: use_phase4 && use_video_gen
                             (상위 게이트 use_phase4가 꺼지면 use_video_gen도 무시. 기본 off = 현행과 수치 동일)

② 모델 로딩
  runtime.build_models() → ModelBundle [재사용]
    ├ models/jscc_model.py [재사용]        VAE+blind SNR+Canny TX — LGVSC의 NTSCC 역할, 교체 불필요
    ├ models/diffusion_wrapper.py [재사용]  MDTv2+ControlNet+CLIP — 키프레임 복원 담당
    ├ guidance/text_extractor.py [재사용]   BLIP2/Qwen 캡션 — LGVSC의 I_text
    └ models/video_generator.py [신규]      world model 대응(DSA-inspired). Open-Sora/SVD 별도
                                            conda env 워커, 파일 IPC — lazy 연결이라 게이트 off면 로드 안 됨

③ 프레임 루프 (video/temporal_pipeline.run)
  키프레임 선택 [확장]  backend 플래그: scene_change(현행) | interval(SKIM) | psss(SKEM)
    ← video/scene_change_detector.py [재사용]
    ← guidance/psss_scorer.py [신규]  Qwen2.5-VL 캡션 + P(No)−P(Yes) 상대 확률
        ※ 캡션 env 재활용만으로는 부족 — generation 결과가 아니라 yes/no 토큰의
          logits/probabilities를 반환하는 backend 경로가 별도로 필요 (§5 주석 참조)
        ※ MVP 임계경로 아님 — baseline은 scene_change+interval+semantic_delta로 가고
          PSSS는 Tx 선택 고도화(로드맵 7단계)로 둔다
  → packet_fn [확장]   guidance/semantic_packet_extractor.py — 캡션+객체/관계+모션 사이드(I_side) 동봉
  → 분기 게이트 [확장]  semantic_delta(재사용) + motion_residual keyframe-anchor 이중 게이트
                        (etri_strategy.md 한계 3 해결 1과 공유)
  → 3-way 분기:
      keyframe / 의미 변화 큼 → reconstruct_fn [재사용]
          pipelines/infer_pipeline.py: prepare_patches → VAE encode → channels/awgn.py
          → diffusion denoise → merge_patches
      정지 구간 → reuse [재사용]  키프레임 복원 재사용 (현행)
      모션 구간 → generate [신규]  video_generator 호출:
          (복원 키프레임, 캡션, 사이드, 세그먼트 길이) → 프레임들
          — 길이 파라미터는 DSA-inspired 인터페이스 (논문의 VAE latent dimension
            동적 조정 DSA 자체는 아님)

④ 출력·평가
  src/sgdjscc_lab/utils/video_io.py [구현됨]  mp4→frames, frames→mp4 (원본 fps)
      ※ 1차에서 구현 완료 — evaluate_video.py가 --input clip.mp4 입력과
        복원 frame folder / --save-video mp4 저장을 지원한다
  → 프레임 재구성 함수(_reconstruct_with_cfg) + evaluators/temporal_consistency.evaluate_sequence [재사용]
      ※ eval_pipeline 전체를 무수정 재사용하는 게 아니라, 프레임 단위 재구성 함수와
        기존 evaluator 일부(evaluate_sequence)를 재사용한다
  → outputs/: keyframes.json · temporal_frames.csv · temporal_metrics.csv · [신규] 복원.mp4
```

> **주의 (segment 계약)** — 현재 `TemporalPipeline`은 frame-wise
> keyframe/reuse/recompute 구조이므로, `video_generator`는 단일 frame 함수가
> 아니라 **GOP/segment 단위 생성 계약**으로 도입해야 한다:
> `(start_keyframe, end_keyframe, segment packets, side_info, length) → frames`.
> 따라서 generate 분기 구현 전에 `SegmentRecord`/`GOPSegment` 자료구조를 먼저
> 정의한다(로드맵 3단계).

## 4. 최종 sgdjscc_lab 시스템 블록 다이어그램

비디오 확장을 반영한 최종 시스템 구조(왼쪽 → 오른쪽 신호 흐름). 위 줄은
**키프레임 스트림**(픽셀 → JSCC latent), 아래 줄은 **시맨틱 스트림**(캡션·모션
사이드)이며, 두 스트림이 수신단에서 합류해 프레임 정책과 생성기를 거친다.

```text
                ┌─ 송신단 Transmitter ─────────────────┐   ┌─ 무선 채널 ─────────┐   ┌─ 수신단 Receiver ────────────────────────────────────────────┐
                │                                      │   │                     │   │                                                              │
 입력 영상 𝒳 ──▶│ 키프레임 선택 [확장]                   │   │                     │   │ 이미지 diffusion 복원 [재사용] ─┐                             │
 (F×H×W×C)     │  scene|interval|PSSS (≙ SKIM/SKEM)    │   │                     │   │  MDTv2+ControlNet (키프레임만)  │                             │
                │  video/keyframe_extractor.py          │   │                     │   │  pipelines/infer_pipeline.py   ├─▶ 3-way 프레임/GOP 정책      │
                │  guidance/psss_scorer.py [신규]        │   │                     │   │                                │    reuse|recompute|generate │
                │   │                                   │   │                     │   │ 복호 캡션·사이드 ───────────────┘    의미델타+모션 게이트      │
                │   ├─▶ JSCC 인코더 [재사용] ────────────┼──▶│ AWGN [재사용] ──────┼──▶│  (+ 세그먼트 길이)                  video/temporal_pipeline  │
                │   │   VAE latent+Canny (≙ NTSCC)      │   │  channels/awgn.py   │   │                                          │                   │
                │   │   models/jscc_model.py            │   │  SNR −5~25 dB       │   │                                          ▼                   │
                │   │                                   │   │                     │   │  비디오 세그먼트 생성 [신규]                                   │
                │   └─▶ 캡션·모션 사이드 [확장] ─────────┼──▶│ 가이드 손상 [확장] ──┼──▶│   Open-Sora/SVD 별도 env                                      │
                │                                      │   │                     │   │   (≙ world model + DSA-inspired 길이 계약)                     │
                │       I_text + I_side 패킹            │   │  token dropout 계열  │   │   models/video_generator.py                                  │
                │       guidance/text_extractor.py      │   │  (가이드 AWGN 금지)  │   │      │                                                       │
                │       video/motion_residual.py        │   │                     │   │      ▼                                                       │
                └──────────────────────────────────────┘   └─────────────────────┘   │  세그먼트 연결 [신규]                                          │
                                                                                     │   프레임 저장 + mp4 인코딩                                      │
                                                                                     │   utils/video_io.py ──────────────────▶ 복원 영상 𝒳̂ + 지표 CSV │
                                                                                     └──────────────────────────────────────────────────────────────┘
```

- **평가 계층** — 원본 𝒳 와 복원 𝒳̂ 를 비교: `temporal_consistency.evaluate_sequence`
  (temporal_srs · srs_flicker · object_identity · temporal_hallucination) + 프레임
  재구성 함수 재사용. eval_pipeline 전체를 무수정 재사용하는 게 아니다.
- **게이트** — generate 분기는 **`use_phase4 && use_video_gen`일 때만 활성**
  (`phase_gates.py`). 상위 게이트 `use_phase4`가 꺼지면 `use_video_gen`을 켜도
  무시된다(기존 phase 게이트 정책과 동일). 둘 중 하나라도 off면 generate 분기
  비활성 → 현행 Phase 4-B와 수치 동일.
- reuse/recompute 분기는 생성기를 거치지 않고 키프레임 복원(재사용) 또는 이미지
  파이프라인(재계산)으로 처리된다.
- **작업 규모 (정확히)** — 완전히 새로 만드는 파일은
  `src/sgdjscc_lab/guidance/psss_scorer.py` ·
  `src/sgdjscc_lab/models/video_generator.py` ·
  `src/sgdjscc_lab/utils/video_io.py` **3개**지만, 이것만으로 끝나지 않는다.
  기존 파일 다수에 확장·수정이 필요하다: `phase_gates.py`(게이트 신설),
  `configs/`(비디오·생성 설정), `scripts/evaluate_video.py`(mp4 IO),
  `video/keyframe_extractor.py`(backend 플래그), `video/temporal_pipeline.py`
  (3-way 분기), `video/motion_residual.py`(사이드 직렬화),
  `guidance/semantic_packet_extractor.py`(사이드 동봉), 그리고 각 변경에 대한
  **테스트**. "신규 3개"는 새 파일 수일 뿐, 구현 범위 전체가 아니다.

## 5. 모듈 매핑표

LGVSC 구성요소별로 현재 코드의 대응물과 필요한 작업.

| LGVSC 구성요소 | 역할 | sgdjscc_lab 대응 | 작업 |
|---|---|---|---|
| **PSSS** (S_rel = P(No)−P(Yes)) | MLLM 확률 기반 프레임쌍 의미 유사도 | 없음 — `semantic_delta`는 packet 집합 비교 | `[신규]` `guidance/psss_scorer.py` — **yes/no 토큰 logits/probabilities 반환 backend 필요** (아래 주석) |
| **SKEM** | PSSS 유도 자기회귀 키프레임 선택 | `video/keyframe_extractor.py` (장면 전환 기반) | `[확장]` backend 플래그 {scene_change \| interval \| psss} |
| **SKIM** | 고정 간격 분할 (저지연용) | 간격 모드 없음 | `[확장]` interval 모드 추가 (소규모) |
| **I_text** 캡션 | 세그먼트 텍스트 시맨틱 | `guidance/text_extractor.py` (BLIP2/Qwen) | `[재사용]` |
| **I_side** 사이드 (논문: optical flow) | 모션 힌트 | `video/motion_residual.py` (**block residual**, 광류 아님) | `[확장]` 1차 경량 proxy로 직렬화. **paper-aligned 버전은 RAFT/flow backend** (line 18 확장점 예고) |
| **NTSCC** 키프레임 전송 | 키프레임 딥 JSCC | `models/jscc_model.py` + `channels/` (VAE latent + AWGN) | `[재사용]` — 동일 모델 아님, "키프레임 딥 JSCC 전송 경로" 역할 대응 |
| 캡션·사이드 비트 전송 | 디지털 경로 (LDPC+변조) | 가이드 손상 규칙 (token dropout — 가이드에 AWGN 금지) | `[확장]` 현재 정책은 dropout 손상. **논문 faithful LDPC+변조는 추후 baseline** |
| **World model** (Open-Sora) | 조건부 세그먼트 생성 | `models/diffusion_wrapper.py` (MDTv2, 이미지 단위) | `[신규]` `models/video_generator.py` — 별도 env 워커 + 파일 IPC |
| **SFA / DSA** | 가변 길이 세그먼트 적응 | 없음 | `[신규]` **DSA-inspired**: 워커 계약에 길이 파라미터. 논문의 VAE latent dimension 동적 조정 DSA 자체는 아님 |
| 세그먼트 연결 | 임의 길이 영상 조립 | `video/temporal_pipeline.py` (reuse/recompute 2-way, [line 313]) | `[확장]` generate 분기 추가 → 3-way + mp4 IO |
| 평가 (CLIP·DISTS·LPIPS·PSNR/SSIM·downstream) | 의미 충실도 채점 | `evaluators/` — `temporal_consistency`에 temporal_srs·srs_flicker·object_identity·temporal_hallucination에 더해 **1차에서 `PTC`(packet consistency)·`SFR`(object birth/death flicker)·`SDI`(keyframe-distance drift)가 잠정 구현됨** (CLIP/packet 기반 — OWLv2/VQA 보강 후 재측정 필요). **track drift·flow-warp·DISTS·downstream은 부재** | `[확장]` 신규 지표 + LGVSC 비교용 metric profile |

> **PSSS backend 요구** — PSSS는 모델이 생성한 "yes"/"no" 문자열이 아니라 두 토큰의
> **확률(logits→softmax)** 이 필요하다(`S_rel = P("No") − P("Yes")`). 현재 캡션
> 파이프라인(`generate_captions.py`, Qwen2.5-VL)은 **텍스트 생성만** 하므로, 그
> 환경을 재활용하되 **로짓/확률을 노출하는 별도 호출 경로**를 `psss_scorer.py`에
> 구현해야 한다 — 환경 재활용 ≠ 구현 완료.

> **LGVSC 비교용 metric profile** — 논문은 CLIP·DISTS·LPIPS·PSNR·SSIM + downstream
> task(video captioning / action classification / depth estimation)로 평가한다.
> 현재 `evaluators/`는 SRS·CLIP·quality(PSNR/SSIM/LPIPS)·FID는 있으나 **DISTS와
> downstream task는 없다.** LGVSC와 직접 비교하려면 이들을 추가한 별도 metric
> profile(`utils/metric_profiles.py` 확장)이 필요하다. SRS·temporal·flicker는
> 이 프로젝트 고유 지표로 병행 보고한다.

## 6. 연구 목표와 도입 로드맵

### 6.0 목표와 비목표

**목표** — LGVSC의 keyframe 기반 비디오 시맨틱 통신 구조를 참고하되, sgdjscc_lab의
기존 자산(semantic packet, SRS, hallucination verification, regeneration loop)을
활용해 **의미 보존과 시간 일관성이 더 좋은 비디오 복원 파이프라인**을 만든다.
직접 노리는 것:

- keyframe 기반 비디오 전송 파이프라인 + generate 분기를 포함한 복원 경로
- temporal SRS · flicker · object preservation 기반 평가
- open-loop 생성이 아니라 **closed-loop semantic verification** 기반 복원 제어

**비목표 (명시)** — 아래는 1차 목표가 아니다. 착수하려면 별도 스코핑을 통과해야 한다.

- LGVSC의 CBR 수치 직접 재현
- JSCC backbone 재학습 / Open-Sora·SVD adapter 대규모 학습
- full value-per-bit 최적 키프레임 정책 (uncertainty-aware optimal)

### 6.1 검증 설계 (전략이 아니라 "무엇을·무엇과·언제까지")

로드맵을 실행·반증 가능하게 만드는 3대 장치. 모든 단계에 공통 적용한다.

**(a) Baseline 집합** — "차별점"은 LGVSC 정면비교가 아니라(재현 불가) **아래 baseline
대비 내부 ablation**으로 정의한다. 이 점을 결과 보고에서 정직하게 명시한다.

| baseline | 내용 |
|---|---|
| copy-reuse | 현행 Phase 4-B (키프레임 복원 복사) — 하한 |
| per-frame recompute | 프레임별 독립 재계산 — flicker 상한 참조 |
| SKIM류 | 고정 간격 keyframe + 생성 |
| (가능 시) LGVSC류 셋업 | 동일 데이터·SNR에서 재현 가능한 범위만 |
| **제안** | bidirectional + verifier 파이프라인 |

**(b) 평가 데이터·인프라** — temporal SRS·flicker·drift는 **실제 모션이 있는 비디오**가
있어야 측정된다. 프레임워크는 이미지(Kodak) 중심이었으므로 **WebVid/Kinetics 등 비디오
데이터셋 확보 + 프레임 추출**을 1~2단계에 포함한다. 평가기 현황(정정):

- **이미 있음** — `evaluators/temporal_consistency.py`에 `temporal_srs`, `srs_flicker`
  (연속 프레임 SRS 변동), `object_identity_consistency`, `temporal_hallucination_rate`.
- **아직 없음(신규)** — **object birth/death flicker**(객체가 나타났다 사라짐),
  **object-track 기반 drift**, **flow-warp temporal consistency**. 기존 `srs_flicker`와
  신규 birth/death flicker는 **구분해서** 보고한다.

**(c) 지표 순환 분리** — regeneration을 **구동하는 지표**와 우위를 **보고하는 지표**를
분리한다. 최적화·선별에 쓴 지표로 승리를 주장하면 순환이다.

- loop-internal (재생성 구동): `srs_packet` / VQA hallucination
- held-out (우위 보고): 루프에 쓰지 않은 지표 (별도 DISTS·downstream, 또는 재생성에
  관여 안 한 temporal 지표)

### 6.2 구현 계약 주의 (착수 전 정의할 것)

> **주의 1 (segment 계약)** — 현재 `TemporalPipeline`은 frame-wise
> keyframe/reuse/recompute 구조다(`temporal_pipeline.py:269`). `video_generator`는
> 단일 frame 함수가 아니라 **GOP/segment 단위 생성 계약**
> `(start_keyframe, end_keyframe, segment packets, side_info, length) → frames`으로
> 도입해야 한다. 따라서 generate 분기 구현 **전에** `SegmentRecord`/`GOPSegment`
> 자료구조를 먼저 정의한다(3단계).

> **주의 2 (bit accounting ≠ semantic-unit count)** — 현재 파이프라인의
> `transmitted_units`(`temporal_pipeline.py:158 count_units`)는 **semantic unit 수
> proxy이지 실제 CBR/bit가 아니다.** LGVSC식 CBR 또는 "동일 전송량 비교"를 주장하려면
> 별도 bitrate accounting을 먼저 정의한다(7단계):
> - **전송 bit로 합산** — keyframe JSCC 심볼 수 + caption/token payload + side info payload
> - **정규화 분모** — segment length / frame count / 해상도 (bits per frame·pixel 산출용)
>
> 즉 generated segment length는 전송 payload가 아니라 **CBR 계산의 분모(구간 길이)**다.
> 이 accounting 전까지는 "동일 CBR" 대신 "동일 semantic-unit overhead"처럼 무엇을 고정한
> 비교인지 명시한다.

### 6.3 단계별 계획

구현 정직성을 위해 generate를 segment 추상화 → start-only → bidirectional로 쪼개고,
난도 높고 생성 분기와 무의존인 **PSSS는 MVP 임계경로에서 내려 7단계로** 둔다.

#### 0단계 — 환경·데이터·백엔드 능력 spike (선행 게이트)

**여기서 실현가능성을 먼저 확인하지 않으면 5단계가 조용히 학습축으로 붕괴한다.**

- 별도 conda env 구성 (SVD/Open-Sora + GPU; ptest는 py3.9/torch2.1이라 미호환)
- 평가 비디오 데이터셋 확보 + 프레임 추출
- **백엔드 능력 spike** — 학습 없이 (a) start-only, (b) **start+end(양방향)**,
  (c) +구조 조건(flow/depth) 어디까지 되나

**판정 기준** — 양방향이 학습 없이 되면 5단계 진행. **안 되면 5단계를 "학습/adapter
과제"로 분리**하고 6단계(verifier)를 우선한다.

#### 1단계 — video_io + mp4 왕복 ✅ (1차 구현 완료, 2026-07)

- `utils/video_io.py` [신규]: mp4 → frames, frames → mp4(원본 fps) — **구현됨**
  (cv2 → ffmpeg CLI 백엔드 자동 선택)
- 기존 **2-way 파이프라인 결과를 영상 파일로 저장** (generate 없이) — **구현됨**
  (`evaluate_video.py --save-video`, `video_io.recon_frames_dir`)

**판정 기준** — mp4 왕복에서 **프레임 수·fps·해상도·정렬 순서가 보존**되고 재조립 영상이
정상 재생되며, 현행 2-way 복원이 영상으로 나오면 진행. (MP4는 lossy 코덱이므로 픽셀
무손실이 아니라 구조·순서 보존을 기준으로 한다.) → `tests/test_video_io.py`의 왕복
테스트로 확인됨.

#### 2단계 — temporal metric 정리 + flicker 확장 🟡 (지표 구현됨 — 잠정, 실측 검증 남음)

- 기존 `temporal_srs`/`srs_flicker`/`object_identity`/`temporal_hallucination` 정리
- **신규**: object **birth/death flicker** (기존 `srs_flicker`와 구분해 보고) —
  **`SFR`로 구현됨**, `PTC`/`SDI`와 함께 `temporal_metrics.csv`에 기록.
  세 지표 모두 **CLIP/packet 기반 초기/잠정 지표**이며 OWLv2/VQA 보강(로드맵
  6단계 이후, etri_strategy 5차) 후 재측정해야 한다.
- baseline 집합(§6.1-a)을 실제 모션 비디오에서 산출 — **미완** (실제 모션 비디오
  데이터 확보 후)

**판정 기준** — 지표가 baseline 간(copy-reuse vs per-frame) **분별력**을 보이면 진행.

#### 3단계 — segment abstraction ✅ (1차 구현 완료, 2026-07)

- **generate 전에** GOP 단위 `SegmentRecord`/`GOPSegment` 자료구조 정의
  (start · end · inter_frames · packets · side_info · length) — 주의 1 —
  **구현됨** (`video/segment.py::SegmentRecord`; generate 결과 부착점은
  `generation` 필드로 예약, side_info는 delta/motion 요약으로 시작)
- `TemporalPipeline`이 frame-wise 결과와 **동치인 segment 재구성**을 내도록 배선 —
  **구현됨** (frame-wise 로그는 그대로 유지, `segments.json` 병행 출력; segment의
  frame 합집합 = frame-wise 인덱스임을 테스트로 보증)

**판정 기준** — segment 재구성 결과가 기존 frame-wise 출력과 일치하면 진행. →
`tests/test_video.py::TestSegmentRecords`로 확인됨.

#### 4단계 — start-only generate 연결

- `video_generator` (segment 계약) + `use_phase4 && use_video_gen` 뒤 3-way 분기
- 입력: start keyframe + caption + side info(초기 `motion_residual` proxy) + length
- **목표는 성능이 아니라 파이프라인 연결 + mp4 산출**

**판정 기준** — generate가 reuse 대비 최소 동등하고 안정적으로 mp4 산출.

#### 5단계 — start+end bidirectional generation (첫 차별점)

- **선행: 0단계 spike 통과 필수.** 중간 프레임을 **start+end keyframe 함께 조건**으로
  → drift를 양 끝에서 억제
- 0단계 spike가 실패하면 **이 단계는 학습/adapter 과제로 분리**

**판정 기준** — unidirectional 대비 drift·flicker가 **유의미하게 감소**하면 채택.

#### 6단계 — Rx-legal semantic verifier + regeneration/selection (핵심 고유 기여)

LGVSC(open-loop)와 갈라지는 신규성. **0단계 양방향이 막혀도 독립 착수 가능.**

- 검증: `semantic_packet_matcher` · SRS · 필요 시 VQA
- **Rx-legal 경계** — **생성 세그먼트 packet vs 전송된 packet** 대조만 수신단에서
  가능(원본 프레임 대조는 **eval-only**, 배치 불가). 이 구분을 코드·문서에 못박는다.
- **종료 조건** — 재생성 예산(최대 재시도 N) + 실패 시 폴백(recompute로 되돌림)
- 추가 keyframe 요청은 **피드백 채널 가정 옵션** — CBR 리포트에서 분리 기록

**판정 기준** — verifier on이 off 대비 **held-out 지표(§6.1-c)** 에서 이득이면 채택.

#### 7단계 — 고도화 (side info · PSSS · adaptive policy · bit accounting)

MVP 임계경로가 아닌 고도화 묶음.

- **side info** — `motion_residual` 대신/병행 optical flow(RAFT)·depth·seg. 무거운
  백엔드는 옵션, **bit overhead·연산량 함께 기록**. "비용 대비 temporal SRS 이득"으로 평가
- **PSSS** — Tx 키프레임 선택 고도화. yes/no token logit 경로 필요(난도 있음), 생성
  분기와 무의존이라 여기로 내린다
- **adaptive keyframe policy** — 초기 proxy(semantic delta + motion + bit cost) →
  후속 generator difficulty proxy(❗반사실 추정·Tx/Rx 경계는 §7 참조)
- **bit accounting 모델** — 주의 2대로 실제 전송량 산정. 이게 있어야 "동일 CBR" 비교 가능

**판정 기준** — 각 항목이 **비용 대비 held-out 지표 이득**을 보이면 채택.

### 6.4 최종 정리 문안

> 이 계획은 LGVSC를 그대로 재현하거나 정면으로 능가하는 것을 1차 목표로 두지 않는다.
> 대신 sgdjscc_lab의 기존 JSCC·semantic packet·SRS·hallucination verification 자산을
> 활용해, LGVSC의 keyframe-based video semantic communication 구조를 참고한 비디오
> 전송·복원 기능을 구현한다. 우선순위는 video_io/mp4 → temporal metric 정리 → segment
> abstraction → start-only generate → bidirectional generation → Rx-legal verifier +
> regeneration → side info·PSSS·adaptive policy·bit accounting 고도화 순이다. 새
> backbone 학습보다 **의미 보존과 시간 일관성을 실제로 개선하는 시스템 수준의 비디오
> semantic communication 확장**을 만드는 것이 목표다.

## 7. 전제와 리스크

| 구분 | 내용 |
|---|---|
| ⚠️ 실시간이 아니다 | 논문 실측: SKEM 전처리 ~2,444s/영상 1분(RTX 4090), 복원 ~24.5s/생성 1초(RTX 4080S). LGVSC 스스로 오프라인·엣지클라우드용으로 규정한다. 저지연 축(Phase 5-B)과는 별도 트랙으로 관리. |
| ⚠️ 생성 비중↑ = 할루시네이션 위험↑ | 안 보낸 프레임을 통째로 생성하므로 원본에 없던 움직임·객체를 지어낼 수 있다. 한계 1 평가 체계(SRS·VQA·flicker)가 이 위험을 채점하는 안전망 — 두 연구 축이 상호 보강된다. |
| ✅ PSSS ≈ 기존 제안 VLM-judge | PSSS의 상대 확률 판정(P(No)−P(Yes))은 한계 2 해결책으로 제안한 VLM-judge와 같은 계열. 한 번 구현하면 Tx 키프레임 선택과 평가(SRS 가중치 적합)에 동시 활용된다. |
| ✅ 알고리즘 보존 원칙 유지 | 이미지 forward 수치는 불변. 신규 기능은 전부 `use_phase4 && use_video_gen` 게이트 뒤, 기본 off에서 Phase 4-B와 동일 동작을 회귀 테스트로 보증한다. |
| ⚠️ 논문 재현이 아닌 근사 | NTSCC≈기존 JSCC, DSA≈길이 파라미터 인터페이스, I_side≈block residual proxy 모두 논문과 **동일 구현이 아닌 역할 대응**이다. LGVSC의 CBR·성능 수치는 이 설계로 직접 재현되지 않으며 별도 실측 대상이다. |
| ⚠️ value-per-bit 선택기의 숨은 난제 (7단계) | "이 키프레임을 안 보내면 생성기가 얼마나 못 하나?"(복원 난이도)는 **반사실 추정** 문제 — Tx에서 생성기를 돌려야 알 수 있어 자기모순. 또 **Tx엔 수신단 생성기 출력이 없다**(Tx/Rx 정보 경계). Tx용 경량 proxy를 별도로 만들어야 하므로 그 자체가 별도 연구다. 초기엔 6단계 verifier가 사후적으로 같은 목적("복원 나쁨 → keyframe 추가")을 반사실 추정 없이 달성한다. |
| ⚠️ 지표 순환 평가 위험 | SRS·flicker 등 휴리스틱 지표를 재생성 구동에 쓰면서 같은 지표로 우위를 보고하면 결과가 부풀려진다. §6.1-c대로 **loop-internal(구동) 지표와 held-out(보고) 지표를 반드시 분리**한다. |

## 관련 문서

- [etri_strategy.md](./etri_strategy.md) — 핵심 한계 3가지와 해결 방안 (이 문서는 한계 3의 장기 확장 축)
- [phase4.md](./phase4.md) — 키프레임/시간적 파이프라인 (4-B)
- [framework_file_roles.md](./framework_file_roles.md) — 파일별 실행 흐름 지도
- 출처 — 논문: `reference/paper/LGVSC…/main.tex` · 코드: `src/sgdjscc_lab/` ·
  시각화 원본: <https://claude.ai/code/artifact/deee634a-9077-4d44-b3ae-2f2a76a8d1b0>

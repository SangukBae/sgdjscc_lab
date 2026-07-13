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
정지-점프와 깜빡임이 생긴다. 복원 프레임을 영상 파일로 내보내는 단계도 없다.

```text
[프레임 단위 — 전부 재사용 대상]
  JSCC 인코딩 (VAE latent + Canny)   models/jscc_model.py
    → AWGN 채널                      channels/awgn.py
    → 이미지 diffusion 복원          models/diffusion_wrapper.py (MDTv2+ControlNet, 프레임별 독립)

[비디오 계층 — Phase 4-B]
  장면 전환 감지                     video/scene_change_detector.py
    → 2-way 정책                     video/temporal_pipeline.py
        의미 델타 < θ → 키프레임 복원 복사
        델타 ≥ θ     → 프레임 재계산
    → 지표만 출력                    scripts/evaluate_video.py
        keyframes.json · temporal CSV (복원 영상 파일 없음)
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
  scripts/evaluate_video.py [확장]  --video in.mp4 입력 → 프레임 추출 IO 추가
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
  src/sgdjscc_lab/utils/video_io.py [신규]  FrameRecord.recon → PNG 저장 → 원본 fps로 mp4 재조립
  → pipelines/eval_pipeline.py · evaluators/ [재사용]
      SRS·temporal·flicker — 복원기가 이미지든 비디오 생성이든 원본↔복원 프레임만
      보므로 무수정 채점
  → outputs/: keyframes.json · temporal_frames.csv · temporal_metrics.csv · 복원.mp4
```

## 4. 최종 sgdjscc_lab 시스템 블록 다이어그램

비디오 확장을 반영한 최종 시스템 구조(왼쪽 → 오른쪽 신호 흐름). 위 줄은
**키프레임 스트림**(픽셀 → JSCC latent), 아래 줄은 **시맨틱 스트림**(캡션·모션
사이드)이며, 두 스트림이 수신단에서 합류해 프레임 정책과 생성기를 거친다.

```text
                ┌─ 송신단 Transmitter ─────────────────┐   ┌─ 무선 채널 ─────────┐   ┌─ 수신단 Receiver ────────────────────────────────────────────┐
                │                                      │   │                     │   │                                                              │
 입력 영상 𝒳 ──▶│ 키프레임 선택 [확장]                   │   │                     │   │ 이미지 diffusion 복원 [재사용] ─┐                             │
 (F×H×W×C)     │  scene|interval|PSSS (≙ SKIM/SKEM)    │   │                     │   │  MDTv2+ControlNet (키프레임만)  │                             │
                │  video/keyframe_extractor.py          │   │                     │   │  pipelines/infer_pipeline.py   ├─▶ 3-way 프레임 정책 [확장]   │
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

- **평가 계층** — 원본 𝒳 와 복원 𝒳̂ 를 비교: SRS · srs_packet · temporal SRS ·
  flicker (`evaluators/`, 복원기 종류와 무관하게 무수정 채점).
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
| 평가 (CLIP·DISTS·LPIPS·PSNR/SSIM·downstream) | 의미 충실도 채점 | `evaluators/` — SRS · temporal · flicker(예정). **DISTS·downstream 부재** | `[확장]` LGVSC 비교용 metric profile 별도 필요 (아래 주석) |

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

## 6. 도입 로드맵

파이프라인을 건드리기 전에 가치를 검증하고, Rx부터 통합한 뒤 Tx를 고도화한다.

### 1단계 — 오프라인 가치 검증

파이프라인 밖에서 생성 품질부터 확인한다.

- 복원 키프레임 2장 + 캡션을 SVD/Open-Sora에 넣어 세그먼트 보간 생성
- 현행 reuse(복사) 방식과 나란히 비교: temporal SRS · flicker · 육안
- 별도 conda env 구성 (ptest는 py3.9/torch2.1이라 미호환)

**판정 기준** — 생성이 복사 대비 flicker·자연스러움에서 이기면 2단계 진행.

### 2단계 — Rx 통합 (generate 분기)

수신단에 생성 경로를 게이트 뒤로 추가한다.

- `models/video_generator.py` 워커 + 파일 IPC 계약
- `temporal_pipeline`에 3-way 정책 (모션 게이트 연동)
- mp4 입출력 IO — 복원 영상 파일 산출
- `use_video_gen` 게이트(활성 조건 `use_phase4 && use_video_gen`), 기본 off → 기존 수치 불변

**산출물** — 원본 vs 복원 영상 나란히 재생하는 데모 + 지표 CSV.

### 3단계 — Tx 고도화 (SKEM/PSSS)

송신단 키프레임 선택을 의미 기반으로 승격한다.

- `psss_scorer` (Qwen2.5-VL, **logits/확률 반환 경로** — generation만으로는 불가)
- keyframe_extractor backend 플래그 3종
- 캡션·사이드 손상 모델 → 전송 신뢰성 실험
- CBR–SRS 곡선: 키프레임 수(η_th) sweep
- LGVSC 직접 비교 시 DISTS·downstream task를 metric profile에 추가

**핵심 결과물** — rate–semantics tradeoff 곡선 (전송량 vs 의미 보존). LGVSC의
CBR(10⁻⁴~10⁻³) 재현은 목표가 아니라 **실측으로만 확인** — 이 프레임워크는 SGD-JSCC
키프레임 경로를 쓰므로 논문과 CBR·성능이 다를 수 있다.

## 7. 전제와 리스크

| 구분 | 내용 |
|---|---|
| ⚠️ 실시간이 아니다 | 논문 실측: SKEM 전처리 ~2,444s/영상 1분(RTX 4090), 복원 ~24.5s/생성 1초(RTX 4080S). LGVSC 스스로 오프라인·엣지클라우드용으로 규정한다. 저지연 축(Phase 5-B)과는 별도 트랙으로 관리. |
| ⚠️ 생성 비중↑ = 할루시네이션 위험↑ | 안 보낸 프레임을 통째로 생성하므로 원본에 없던 움직임·객체를 지어낼 수 있다. 한계 1 평가 체계(SRS·VQA·flicker)가 이 위험을 채점하는 안전망 — 두 연구 축이 상호 보강된다. |
| ✅ PSSS ≈ 기존 제안 VLM-judge | PSSS의 상대 확률 판정(P(No)−P(Yes))은 한계 2 해결책으로 제안한 VLM-judge와 같은 계열. 한 번 구현하면 Tx 키프레임 선택과 평가(SRS 가중치 적합)에 동시 활용된다. |
| ✅ 알고리즘 보존 원칙 유지 | 이미지 forward 수치는 불변. 신규 기능은 전부 `use_phase4 && use_video_gen` 게이트 뒤, 기본 off에서 Phase 4-B와 동일 동작을 회귀 테스트로 보증한다. |
| ⚠️ 논문 재현이 아닌 근사 | NTSCC≈기존 JSCC, DSA≈길이 파라미터 인터페이스, I_side≈block residual proxy 모두 논문과 **동일 구현이 아닌 역할 대응**이다. LGVSC의 CBR·성능 수치는 이 설계로 직접 재현되지 않으며 별도 실측 대상이다. |

## 관련 문서

- [etri_strategy.md](./etri_strategy.md) — 핵심 한계 3가지와 해결 방안 (이 문서는 한계 3의 장기 확장 축)
- [phase4.md](./phase4.md) — 키프레임/시간적 파이프라인 (4-B)
- [framework_file_roles.md](./framework_file_roles.md) — 파일별 실행 흐름 지도
- 출처 — 논문: `reference/paper/LGVSC…/main.tex` · 코드: `src/sgdjscc_lab/` ·
  시각화 원본: <https://claude.ai/code/artifact/deee634a-9077-4d44-b3ae-2f2a76a8d1b0>

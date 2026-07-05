> [← 문서 색인](./README.md)

# 프레임워크 비교

두 가지를 비교한다: (1) 원본 `SGDJSCC/` monolithic 추론과 `sgdjscc_lab/`의 모듈화
구조, (2) **SGD-JSCC 논문**과 `sgdjscc_lab` 구현의 정합 수준.

## 1. 구조 비교 — 원본 vs sgdjscc_lab

동일한 AWGN 시맨틱 이미지 전송 파이프라인을 **알고리즘은 보존**하면서 연구·확장을
위해 구조만 재구성했다.

| 항목 | 원본 `SGDJSCC/` | `sgdjscc_lab` |
|---|---|---|
| 진입점 | `inference_one.py` 중심 | `scripts/infer_images.py` |
| Config | script 결합 + 하드코딩 | `config.py` + YAML + CLI override |
| 모델 로딩 | 한 파일 inline | `models/` + `runtime.py` |
| 채널 | `_JSCCModel.channel()` 내부 | `channels/awgn.py` |
| 가이드 | script 내부 함수 | `guidance/` 하위 모듈 |
| 추론 흐름 | script monolithic | `pipelines/infer_pipeline.py` |
| 평가 | script 끝단 혼재 | `evaluators/` 분리 |
| 원본 수정 | — | `SGDJSCC/`는 read-only |

**보존되는 알고리즘 블록**: VAE encode/decode + scaling `15.45`, AWGN 손상,
blind SNR 예측, step matching, mask token, canny 재전송/latent 조건화,
MDTv2/ControlNet 확산 디노이징. 즉 새 전송 알고리즘이 아니라 원본 추론 경로의
**모듈식 재포장**이다.

책임 분리 덕분에 이후 작업이 실용적이다: AWGN→Rayleigh 채널 교체, 엣지→depth/seg
가이드 확장, 추론 코어를 건드리지 않는 지표 삽입, 쉬운 테스트·실패 격리.

## 2. 논문 정합 비교 (Paper ↔ sgdjscc_lab)

**추론 경로 = paper-faithful**(원본 SGDJSCC 코드를 런타임 재사용). 그 위의 (A) 학습
scaffold, (B) 채널/CSI, (C) 평가는 **paper-faithful / paper-like / scaffold / 미구현**으로
구분한다.

| # | 논문 구성요소 | 코드 상태 | 충실도 | 비고 |
|---|---|---|---|---|
| 1 | 추론 forward-pass 전반 | 원본 재사용 | **paper-faithful** | scaling 15.45, step matching, canny 재전송, blind SNR |
| 2 | 연속 timestep DiT, sigmoid schedule, f0 예측, 결정적 역과정 | `SigmoidNoiseScheduler` + 추론 | **paper-faithful(구조)** | — |
| 3 | MDTv2 masked+unmasked DM 손실 | `TextDMStageRunner` | **paper-like** | 구조 일치 |
| 4 | CFG 학습(null-conditioning) | `apply_cfg_label_dropout` + `LearnedNullToken` | **paper-like** | `cfg_null_mode=learned` 지원, dropout 확률은 논문 미공개 |
| 5 | Step matching `m=S⁻¹(σ²/(σ²+\|h\|²))` | 추론 + `inverse_beta_bar` | **paper-faithful** | — |
| 6 | Text guidance(BLIP2) | `guidance/text_extractor` | **paper-faithful** | — |
| 7 | Edge codec (foreground 확률, BCE+Dice) | `EdgeJSCC` `conv`\|`vit` + `edge_codec` stage | **paper-like** | ViT 선택 가능(WITT-exact 아님), multi-SNR + SNR-adaLN 연결 |
| 8 | ControlNet(base DM frozen) | freeze 정책/구조 일치 | **paper-like** | MuGE repr(기본 `edge_uncertainty`)를 latent `c`로 조건화 |
| 9 | JSCC enc/dec MSE+λGAN(+LPIPS) | `JSCCStageLoss` | **scaffold→구조정렬** | LPIPS 결합 추가(기본 off), patch-GAN 수치 미보장 |
| 10 | Blind SNR 추정망 | 원본 `Prediction_Model` | **paper-faithful** | — |
| 11 | MMSE 등화 `y/√(g²+σ²)` | `channels/measurement.mmse_equalize` | **paper-faithful(실수 gain)** | 복소 위상 `e^{-jφ}` 미재현 |
| 12 | Fast-fading water-filling(Alg.4) | 루프+어댑터+CSI정책 + decode-swap + patch별 evidence | **알고리즘 faithful / 배선 connected / 수치=stub 검증** | 실제 수치만 MDTv2 체크포인트 의존 |
| 13 | per-element 잡음레벨 `dᵢ`(eq.12) | `MeasurementBundle.noise_level` | **paper-faithful** | — |
| 14 | Phase 추정 + joint CSI(Alg.3) | `phase_est` 필드만 | **미구현/부분** | 전용 phase망·반복 루프 없음(AWGN/실수 fading엔 무관) |
| 15 | 평가지표 PSNR/LPIPS/CLIP/**FID** | `evaluators/fid.py` + `--require-real-fid` | **연결** | 실제 Inception은 torchvision/가중치 필요(세션 외) |
| 16 | 지표 세트(논문 vs 확장) | `utils/metric_profiles.py` | **정합/리포팅** | SSIM은 비논문 플래그 |

**코드에 있으나 논문에 없음 (ETRI 확장)**: SRS·hallucination·object-preservation,
Phase 4/5(packet-drop, video/temporal, regeneration search, channel-conditioning,
consistency/early-exit), controllers, `end_to_end_ft`. 반대로 논문의 FID는 연결됨(#15).

**남는 갭**: water-filling 실제 수치(#12), phase/joint CSI 학습(#14), edge codec ViT
baseline급 학습(#7), patch-GAN/LPIPS 수치·~14M-pair 250k-step DM 재현(#9), 실제
Inception-FID 수치(#15) — 모두 데이터/컴퓨트/네트워크 의존.

### paper_mode — 논문 재현 경로 분리

확장 기능과 논문 충실 경로를 섞지 않도록 **`paper_mode`** guardrail을 둔다
(`paper_mode.py`, top-level `paper_mode: true`, 기본 false). 켜면 비충실 stand-in
(auto-caption·Canny edge·`shared_vae` transport·zero-vector CFG null·단일 고정 SNR
edge codec)을 체크포인트 로딩 전에 `PaperModeError`로 차단한다. 논문 경로는
`configs/paper_train_{jscc,text_dm,edge_codec,controlnet}.yaml` +
`paper_eval_awgn.yaml`. 항목별 상세는 [paper_gap_closure.md](./paper_gap_closure.md).

### 검증 상태

`ptest` 환경 단위/통합·synthetic 테스트 전부 통과. real-model stage smoke / 실제 DM
water-filling 수치 / Inception-FID 수치는 체크포인트·GPU·네트워크 의존이라 코드 경로
기준으로만 검증됨.

### 관련 문서
- [paper_gap_closure.md](./paper_gap_closure.md) · [training_scaffold.md](./training_scaffold.md) · [phase5.md](./phase5.md)
</content>

# ETRI 과제 개요: 생성 AI 기반 시맨틱 미디어 전송 신뢰성 평가 프레임워크

## 과제 목표

생성 AI 기반의 **시맨틱 미디어 전송 신뢰성(Semantic Media Transmission Reliability)**을 정량적으로 평가하는 End-to-End 시뮬레이션 프레임워크 개발.

> 핵심: 시각적으로 선명한 복원(PSNR 최대화)이 목표가 아니라, **노이즈 채널을 통과한 후에도 원본 이미지의 의미(semantic intent)가 얼마나 신뢰성 있게 전달되는가**를 측정하는 것이 목표다.

---

## 시스템 파이프라인

```
Original Image / Keyframe
        │
        ▼
[Tx] JSCC 시맨틱 인코더
  - VAE latent 인코딩 (스케일링 팩터 15.45)
  - Canny 구조 가이드 추출 (MuGE edge map)
  - L2-normalization
        │
        ▼
[Channel] 무선 채널
  - Phase 1~4: AWGN (Additive White Gaussian Noise)
  - Phase 5:   Rayleigh fading / fast-fading / packet-drop
        │
        ▼
[Rx] 확산 모델(Diffusion) 기반 복원
  - MDTv2 backbone
  - ControlNet (선택적 구조 가이드 조건화)
  - Blind SNR 예측 → 확산 스텝 매칭
  - Canny 재전송 → 확산 디노이징 → 최종 디코딩
        │
        ▼
[Eval] 시맨틱 일관성 & 할루시네이션 평가
  - PSNR / SSIM / LPIPS (픽셀/구조 품질)
  - CLIP 유사도 (이미지-이미지, 텍스트-이미지)
  - 객체 보존률 / 누락률 / 추가률
  - 할루시네이션 점수
  - Semantic Reliability Score (SRS, 최종 헤드라인 지표)
        │
        ▼
outputs/results.csv
```

---

## 핵심 평가 지표: SRS (Semantic Reliability Score)

```python
SRS = (  0.30 * clip_image_image
       + 0.25 * clip_text_image
       + 0.25 * object_preservation_rate
       - 0.10 * missing_object_rate
       - 0.10 * additional_object_rate )
```

`results.csv`에 이미지 × SNR 조합별로 아래 컬럼이 기록된다:

| 컬럼 | 설명 |
|------|------|
| `psnr` | Peak Signal-to-Noise Ratio |
| `ssim` | Structural Similarity |
| `lpips` | Learned Perceptual Image Patch Similarity |
| `clip_image_image` | 원본↔복원 CLIP 임베딩 코사인 유사도 |
| `clip_text_image` | BLIP2 캡션↔복원 CLIP 유사도 |
| `object_preservation_rate` | 원본 객체 중 복원에서 유지된 비율 |
| `missing_object_rate` | 원본 객체 중 누락된 비율 |
| `additional_object_rate` | 원본에 없던 객체가 복원에서 생성된 비율 |
| `hallucination_score` | 할루시네이션 정도 (낮을수록 좋음) |
| `semantic_reliability_score` | SRS (종합 지표) |

---

## 실험 설정

### SNR 범위
```
[-5, 0, 5, 10, 15, 20, 25] dB
```

### 비교 그룹

| 그룹 | 설명 |
|------|------|
| WITT baseline | 생성 복원 없는 트랜스포머 시맨틱 전송 |
| DiffJSCC / SGDJSCC baseline | 확산 모델 기반 JSCC (가이드 없음) |
| 제안 방법 | SGDJSCC + 구조 가이드 + 시맨틱/할루시네이션 평가 |

### 가이드 손상 규칙
- JSCC latent / 채널 심볼에만 AWGN / Rayleigh 적용
- Canny / segmentation 가이드에는 AWGN 직접 적용 **금지**
  - Edge map: dropout / blur / erasing으로 손상
  - Seg map: 클래스 dropout / 영역 제거
  - 캡션 토큰: token dropout

### 입력 크기
- 128×128 패치 타일링
- H, W가 128의 배수가 되도록 리사이즈

---

## 소프트웨어 구조

### 저장소 레이아웃

```
Semantic/
├── sgdjscc_lab/        ← PRIMARY 개발 패키지 (모든 신규 코드)
├── SGDJSCC/            ← 원본 코드 READ-ONLY (논문 베이스라인)
├── CLIP/               ← 외부 베이스라인 (OpenAI CLIP)
├── Deep-JSCC-PyTorch/  ← 외부 베이스라인 (Deep JSCC)
├── DiffJSCC/           ← 외부 베이스라인 (Diffusion-aided JSCC)
├── WITT/               ← 외부 베이스라인 (Transformer 시맨틱 전송)
├── POPE/               ← 객체 할루시네이션 평가 참고
├── diffusers/          ← Stable Diffusion / ControlNet 참고
└── paper/              ← 논문 원고 / 그림
```

### `sgdjscc_lab/` 모듈 구조

```
src/sgdjscc_lab/
├── channels/           ← 채널 모델 (awgn.py; Phase 5에서 rayleigh 추가)
├── guidance/           ← 시맨틱/구조 가이드 추출기
│   ├── text_extractor.py       (BLIP2 캡션)
│   ├── edge_extractor.py       (MuGE Canny)
│   ├── depth_extractor.py
│   └── segmentation_extractor.py
├── models/             ← JSCC + Diffusion 모델 래퍼
│   ├── jscc_model.py           (VAE + Blind SNR + Canny TX)
│   ├── diffusion_wrapper.py    (MDTv2 + ControlNet + CLIP)
│   └── model_bundle.py
├── pipelines/          ← 추론 / 평가 파이프라인
│   ├── infer_pipeline.py
│   ├── eval_pipeline.py
│   └── regeneration_loop.py
├── evaluators/         ← 연구 지표 계산기
│   ├── quality.py
│   ├── clip_score.py
│   ├── object_preservation.py
│   ├── hallucination.py
│   └── semantic_reliability.py
└── utils/
```

---

## 환경 및 실행 명령

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest   # Python 3.9, PyTorch 2.1.0, CUDA 11.8

# 추론 (AWGN, 단일 이미지 또는 폴더)
python scripts/infer_images.py --config configs/default.yaml
python scripts/infer_images.py --config configs/composed.yaml \
    --input /path/imgs/ --output /path/out/ --snr 5 --device cuda:0

# 평가 (단일 SNR / SNR sweep / 픽셀 품질만)
python scripts/evaluate.py --config configs/composed.yaml --snr 10
python scripts/evaluate.py --config configs/composed.yaml --snr-list -5,0,5,10,15,20,25
python scripts/evaluate.py --config configs/composed.yaml --snr 10 --no-clip

# 테스트 (GPU 불필요)
python -m pytest tests/ -v
```

**체크포인트** (HuggingFace `murjun/SGDJSCC`에서 다운로드 → `sgdjscc_lab/checkpoints/`):
- `JSCC_model.pth`
- `diffusion_backbone.pth`
- `diffusion_controlnet.pth`
- `muge-epoch-19-checkpoint.pth`

---

## Phase별 개발 계획

| Phase | 상태 | 범위 |
|-------|------|------|
| 1 | ✅ 완료 | AWGN 단일 이미지 / 폴더 추론 CLI |
| 2 | ✅ 완료 | 모듈 분리 (channels/guidance/models/pipelines) + `_defaults_` config 합성 |
| 3 | ✅ 완료 | 전체 평가기 suite, SNR-sweep CSV, depth/seg 가이드, regeneration loop |
| 4 | ✅ 완료 | 패킷 인식 검증기 + 적응형 가이드 (4-A), 키프레임 / 시간적 파이프라인 (4-B) |
| 5 | 🔲 스캐폴딩 | Rayleigh fading, DiT/DiTJSCC, 강화된 시맨틱 평가 (5-A/B/C) |

---

## 개발 원칙

1. **알고리즘 경로 보존**: `SGDJSCC/inference_one.py`의 순전파 수치는 변경 금지 (VAE 스케일링 팩터 `15.45`, AWGN 노이즈 공식, blind SNR 추정, step matching 등)
2. **관심사 분리**: 각 모듈(`channels/`, `guidance/`, `models/`, `pipelines/`, `evaluators/`)은 독립적으로 교체 가능하도록 설계
3. **원본 저장소 읽기 전용**: 모든 신규 아이디어는 `SGDJSCC/`가 아닌 `sgdjscc_lab/`에 구현

---

## 관련 문서

- [README.md](./README.md) — 문서 인덱스
- [phases_1to3.md](./phases_1to3.md) — Phase 1~3 상세
- [phase4.md](./phase4.md) — Phase 4 상세
- [phase5.md](./phase5.md) — Phase 5 상세
- [limitation_reference_map.md](./limitation_reference_map.md) — 참고 논문 및 한계점 맵
- [framework_comparison.md](./framework_comparison.md) — 원본 vs lab 구조 비교

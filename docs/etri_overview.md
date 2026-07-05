> [← 문서 색인](./README.md)

# ETRI 과제 개요

## 과제 목표

생성 AI 기반 **시맨틱 미디어 전송 신뢰성(Semantic Media Transmission Reliability)**을
정량 평가하는 End-to-End 시뮬레이션 프레임워크 개발.

> 핵심: 선명한 복원(PSNR 최대화)이 아니라, **노이즈 채널을 통과한 뒤에도 원본의
> 의미(semantic intent)가 얼마나 신뢰성 있게 전달되는가**를 측정한다.

## 시스템 파이프라인

```
Original Image / Keyframe
  → [Tx]  JSCC 시맨틱 인코더 (VAE latent, scaling 15.45 + MuGE 구조 가이드 + L2-norm)
  → [Ch]  무선 채널 (Phase 1~4: AWGN / Phase 5: Rayleigh·fast-fading·packet-drop)
  → [Rx]  확산 복원 (MDTv2 + 선택적 ControlNet, blind SNR → step matching → 디노이징 → decode)
  → [Eval] 시맨틱 일관성 & 할루시네이션 평가
  → outputs/results.csv
```

## 핵심 지표: SRS (Semantic Reliability Score)

```python
SRS = ( 0.30 * clip_image_image
      + 0.25 * clip_text_image
      + 0.25 * object_preservation_rate
      - 0.10 * missing_object_rate
      - 0.10 * additional_object_rate )
```

`results.csv`에 이미지 × SNR별로 기록되는 컬럼: `psnr, ssim, lpips,
clip_image_image, clip_text_image, object_preservation_rate, missing_object_rate,
additional_object_rate, hallucination_score, semantic_reliability_score, fid`.
(패킷 평가 시 `srs_base, srs_packet, srs_v2` 등이 추가된다 — [phase4.md](./phase4.md).)

## 실험 설정

- **SNR 범위**: `[-5, 0, 5, 10, 15, 20, 25]` dB
- **비교 그룹**: WITT baseline(생성 복원 없음) / DiffJSCC·SGDJSCC baseline(가이드 없음)
  / 제안(SGDJSCC + 구조 가이드 + 시맨틱·할루시네이션 평가)
- **가이드 손상 규칙**: AWGN/Rayleigh는 JSCC latent·채널 심볼에만 적용한다.
  가이드는 직접 손상하지 않고 — edge=dropout/blur/erasing, seg=클래스 dropout/영역
  제거, 캡션=token dropout — 로 별도 손상한다.
- **입력 크기**: 128×128 패치 타일링, H·W를 128 배수로 리사이즈.

## 저장소 & 모듈 구조

```
Semantic/
├── sgdjscc_lab/        ← PRIMARY 개발 패키지
├── SGDJSCC/            ← 원본 READ-ONLY (논문 베이스라인, 런타임 재사용)
├── CLIP/ Deep-JSCC-PyTorch/ DiffJSCC/ WITT/ POPE/ diffusers/  ← 외부 베이스라인/참고
└── paper/
```

`sgdjscc_lab` 모듈 구성은 [문서 색인의 디렉터리 트리](./README.md#디렉터리-구성)와
[framework_file_roles.md](./framework_file_roles.md) 참조.

## 실행 명령

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest   # Python 3.9, PyTorch 2.1.0, CUDA 11.8

# 추론 (AWGN, 단일 이미지/폴더)
python scripts/infer_images.py --config configs/composed.yaml \
    --input /path/imgs/ --output /path/out/ --snr 5 --device cuda:0

# 평가 (단일 SNR / sweep / 픽셀 품질만)
python scripts/evaluate.py --config configs/composed.yaml --snr-list -5,0,5,10,15,20,25

# 테스트 (GPU 불필요)
python -m pytest tests/ -v
```

**체크포인트** (HuggingFace `murjun/SGDJSCC` → `sgdjscc_lab/checkpoints/`):
`JSCC_model.pth`, `diffusion_backbone.pth`, `diffusion_controlnet.pth`,
`muge-epoch-19-checkpoint.pth`.

## Phase별 현황

| Phase | 상태 | 범위 |
|-------|------|------|
| 1 | ✅ | AWGN 단일 이미지/폴더 추론 CLI |
| 2 | ✅ | 모듈 분리 + `_defaults_` config 합성 |
| 3 | ✅ | 평가기 suite, SNR-sweep CSV, depth/seg 가이드, regeneration loop |
| 4 | ✅ | 패킷 인식 검증기 + 적응형 가이드(4-A), 키프레임/시간적(4-B) |
| 5 | ✅ 스캐폴드 | Rayleigh/fast-fading/packet-drop 채널 조건화(5-A), 저지연/early-exit(5-B), SRS-v2/regeneration search(5-C) |

## 개발 원칙

1. **알고리즘 경로 보존** — `SGDJSCC/inference_one.py`의 순전파 수치 변경 금지
   (VAE scaling `15.45`, AWGN 공식, blind SNR, step matching 등).
2. **관심사 분리** — 각 모듈은 독립 교체 가능.
3. **원본 읽기 전용** — 신규 아이디어는 `sgdjscc_lab/`에만 구현.

## 관련 문서
- [phases_1to3.md](./phases_1to3.md) · [phase4.md](./phase4.md) · [phase5.md](./phase5.md)
- [etri_development_roadmap.md](./etri_development_roadmap.md) · [limitation_reference_map.md](./limitation_reference_map.md)
</content>

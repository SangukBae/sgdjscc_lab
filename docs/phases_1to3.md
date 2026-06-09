> [← 문서 색인](./README.md)

# Phase 1–3 요약

## Phase 1 요약

Phase 1은 최소 실행 가능한 패키지를 구축했다:

- AWGN 단일 이미지 / 폴더 추론
- config 기반 CLI
- 출력 이미지 저장 경로
- 원본 추론 경로 보존

완료 기준:

```bash
python scripts/infer_images.py --config configs/default.yaml
```

---

## Phase 2 요약

Phase 2는 단일 스크립트(monolithic) 구조를 모듈형 패키지로 변환했다.

### 주요 구조 변경

| 이전 | 이후 |
|---|---|
| 모델 내부의 인라인 AWGN 채널 | `channels/awgn.py` |
| 평평한(flat) 런타임 로더 | `models/jscc_model.py` + `models/diffusion_wrapper.py` + `runtime.py` |
| 평평한 파이프라인 | `pipelines/infer_pipeline.py` |
| 최상위 레벨 전처리 | `utils/preprocessing.py` |
| scripts/pipeline 내부의 seed·메모리 헬퍼 | `utils/seed.py`, `utils/memory.py` |
| fragment config 시스템 없음 | `config.py`의 `_defaults_` composition |

### Phase 2 완료 항목

- 모듈형 패키지 구조
- editable install 지원
- config composition
- config / I/O / AWGN 채널 단위 테스트

---

## Phase 3 요약

Phase 3은 실제 연구-평가 기반을 구축했다.

### 평가기 (Evaluators)

- `quality.py` — PSNR / SSIM / LPIPS
- `clip_score.py` — CLIP 이미지-이미지 및 텍스트-이미지 유사도
- `object_preservation.py` — 객체 보존율
- `hallucination.py` — 할루시네이션 점수
- `semantic_reliability.py` — Semantic Reliability Score (SRS)

### 가이드 확장

- `depth_extractor.py` — DPT 단안 깊이(monocular depth)
- `segmentation_extractor.py` — SegFormer 시맨틱 세그멘테이션

### 평가 파이프라인

- `eval_pipeline.py` — 단일 SNR 및 SNR-sweep 평가
- `regeneration_loop.py` — SRS 기반 재시도 경로
- `evaluate.py` — 평가 CLI

### 데이터셋 config

- Kodak
- COCO val2017
- ADE20K validation

### Semantic Reliability Score

```text
SRS = 0.30 × clip_image_image
    + 0.25 × clip_text_image
    + 0.25 × object_preservation_rate
    - 0.10 × missing_object_rate
    - 0.10 × additional_object_rate
```

### 그 시점의 Phase 3 휴리스틱 한계

- 객체 보존과 할루시네이션은 여전히 휴리스틱 CLIP 기반 지표다
- POPE 스타일 VQA는 아직 Phase 3 스택에 통합되지 않았다
- Depth / segmentation 모델은 최초 사용 시 외부 다운로드가 필요하다
- Regeneration loop는 경량 프로토타입이다

이후 phase들은 패킷 인식 검증, 시간적 지표, 로컬 VQA 기반 할루시네이션 점검,
SRS-v2, 채널 조건화 평가, 지연/early-exit 실험을 추가한다. 이 절은 의도적으로
현재 패키지의 최종 상태가 아니라 **Phase 3 스냅샷**이다.

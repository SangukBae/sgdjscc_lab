> [← 문서 색인](../README.md)

# Phase 1–3 요약

> 보관본. 이후 전체 전략은 [etri_strategy.md](../etri_strategy.md), 후속 기능은
> [phase4.md](../phase4.md) · [phase5.md](../phase5.md) 참조.

> 이 문서는 Phase 3 시점의 스냅샷이다. 이후 phase가 얹은 패킷/시간적/채널 조건화
> 기능은 [phase4.md](../phase4.md) · [phase5.md](../phase5.md) 참조.

## Phase 1 — AWGN 추론 CLI

최소 실행 가능한 패키지를 구축했다: config 기반 CLI로 AWGN 단일 이미지/폴더 추론,
출력 저장, 원본 추론 경로 보존.

```bash
python scripts/infer_images.py --config configs/default.yaml
```

## Phase 2 — 모듈화

단일 스크립트 구조를 모듈형 패키지로 재구성했다(알고리즘은 그대로, 구조만 분리).

| 이전 | 이후 |
|---|---|
| 모델 내부 인라인 AWGN | `channels/awgn.py` |
| 평평한 런타임 로더 | `models/jscc_model.py` + `models/diffusion_wrapper.py` + `runtime.py` |
| 평평한 파이프라인 | `pipelines/infer_pipeline.py` |
| script/util 혼재 전처리 | `utils/preprocessing.py` + `utils/{seed,memory}.py` |
| fragment config 없음 | `config.py`의 `_defaults_` 합성 |

editable install 지원, config/IO/AWGN 단위 테스트 추가.

## Phase 3 — 평가 프레임워크

실제 연구-평가 기반을 구축했다.

**평가기** (`evaluators/`)
- `quality.py` — PSNR / SSIM / LPIPS
- `clip_score.py` — CLIP 이미지-이미지 / 텍스트-이미지 유사도
- `object_preservation.py` — 객체 보존율
- `hallucination.py` — 할루시네이션 점수
- `semantic_reliability.py` — Semantic Reliability Score (SRS)

**가이드 확장** — `depth_extractor.py`(DPT 단안 깊이), `segmentation_extractor.py`(SegFormer).

**평가 파이프라인** — `eval_pipeline.py`(단일 SNR + SNR-sweep), `regeneration_loop.py`(SRS 기반 재시도), CLI `evaluate.py`.

**데이터셋 config** — Kodak / COCO val2017 / ADE20K validation.

### Semantic Reliability Score

```text
SRS = 0.30 × clip_image_image
    + 0.25 × clip_text_image
    + 0.25 × object_preservation_rate
    − 0.10 × missing_object_rate
    − 0.10 × additional_object_rate
```

### Phase 3 시점의 한계

- 객체 보존/할루시네이션은 휴리스틱 CLIP 기반 (POPE 스타일 VQA는 Phase 5-C에서 통합)
- depth/seg 모델은 최초 사용 시 외부 다운로드 필요
- regeneration loop는 경량 프로토타입

이후 phase가 패킷 인식 검증, 시간적 지표, VQA 할루시네이션, SRS-v2, 채널 조건화,
저지연/early-exit를 추가한다.
</content>

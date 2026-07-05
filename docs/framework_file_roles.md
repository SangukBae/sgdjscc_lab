> [← 문서 색인](./README.md)

# 파일별 프레임워크 역할

`sgdjscc_lab` 파일을 실행 흐름 순서로 정렬해 역할을 매핑한다. 핵심은 원본 SGDJSCC
알고리즘 경로를 보존하되 코드를 명시적 모듈로 재구성했다는 점이다.

> **역할**: 이 문서 = 파일별 **실행 흐름/역할 지도**(무엇이 언제 실행되는가).
> 원본·논문 **대비 차이/정합**은 [framework_comparison.md](./framework_comparison.md).

논문 Figure 1(b) 블록: `DeepJSCC Encoder` / `Semantic Extractor` /
`Semantic side info encoder·decoder` / `Wireless Channel` / `Diffusion Denoiser` /
`DeepJSCC Decoder`. 인프라 파일은 "Fig 1(b) 외부", 여러 블록을 조율하면 "오케스트레이션".

## 1. 추론 프레임워크 (실행 순서)

```
infer_images.py → config.py → runtime.py → 모델 빌더 → io/preprocessing
                → infer_pipeline.py → guidance → JSCC + AWGN + diffusion → save
```

| 파일 | 역할 | Fig 1(b) |
|---|---|---|
| `scripts/infer_images.py` | CLI 진입: 인자 파싱, config 로드, 모델 빌드, 배치 추론 | 외부 |
| `config.py` | YAML 로드, `_defaults_` 합성, 상대 경로 해석, CLI override | 외부 |
| `configs/{default,composed}.yaml` + `channel·model·infer` fragment | 추론 설정 | 외부 |
| `utils/seed.py` | 랜덤 시드 고정(재현성) | 외부 |
| `runtime.py` | device 해석 + 전체 모델 번들 조립 | 오케스트레이션 |
| `_sgdjscc.py` | `SGDJSCC/`를 `sys.path`에 주입(원본 모듈 재사용) | 외부 |
| `models/jscc_model.py` | VAE enc/dec, blind SNR 예측기, canny TX net | Encoder / Semantic side info / Decoder |
| `channels/awgn.py` | 교체 가능한 AWGN latent 전송 | Wireless Channel |
| `models/diffusion_wrapper.py` | MDTv2 backbone + 선택적 ControlNet + CLIP + 공유 VAE | Diffusion Denoiser |
| `models/model_bundle.py` | JSCC/확산/가이드/device를 런타임 번들로 패킹 | 외부 |
| `io.py` | 파일 나열, 텐서 로드, 결과 저장 | 외부 |
| `utils/preprocessing.py` | crop/resize, 128×128 패치 분할·병합 | Encoder 전단계 |
| `pipelines/infer_pipeline.py` | 메인 파이프라인: 이미지 순회, 패치별 forward | 오케스트레이션 |
| `guidance/text_extractor.py` | BLIP2 캡션(텍스트 가이드) | Semantic Extractor |
| `guidance/edge_extractor.py` | MuGE soft edge + uncertainty map | Semantic Extractor |

**코어 forward** (`infer_pipeline`): soft edge 전처리 → VAE encode/normalize → AWGN →
mask/power scalar → step matching → canny 재전송 → canny latent → 확산 디노이징 → 최종 decode.

## 2. 평가 프레임워크 (실행 순서)

```
evaluate.py → eval config → eval_pipeline.py → infer_pipeline.py
            → evaluators → csv_logger.py → metrics_io.py
```

| 파일 | 역할 |
|---|---|
| `scripts/evaluate.py` | 평가 CLI: SNR 옵션, eval context, 단일/sweep 실행 |
| `configs/eval/default.yaml`, `dataset/*.yaml` | 지표·SNR·CSV·SRS 가중치·데이터셋 설정 |
| `pipelines/eval_pipeline.py` | 추론을 감싸 데이터셋 순회·지표·CSV·sweep 제어 |
| `pipelines/infer_pipeline.py` | 평가 내부에서 실제 복원 수행(추론을 대체하지 않고 감쌈) |
| `evaluators/quality.py` | PSNR / SSIM / LPIPS |
| `evaluators/clip_score.py` | CLIP 이미지-이미지 / 텍스트-이미지 |
| `evaluators/object_preservation.py` | 객체 보존율 |
| `evaluators/hallucination.py` | 할루시네이션(추가 객체) |
| `evaluators/semantic_reliability.py` | SRS 종합 |
| `utils/csv_logger.py` | 이미지별 지표 행을 CSV로 스트리밍 |
| `utils/metrics_io.py` | 집계 + 콘솔 요약 표 |
| `pipelines/regeneration_loop.py` | SRS < 임계값이면 재복원, 최적 유지(선택) |

## 3. Phase 4 / 5 확장 모듈 (opt-in)

config로 명시 활성화하지 않는 한 기본 추론 경로의 일부가 아니다.

| 파일군 | 역할 |
|---|---|
| `guidance/{depth,segmentation}_extractor.py` | 선택적 depth/seg 가이드(Phase 3 확장) |
| `controllers/{adaptive_guidance,snr_guidance}_*.py` | SNR 인식 가이드/스텝 제어(4-A) |
| `guidance/{semantic_packet,object,relation}_extractor.py`, `importance_estimator.py` | 시맨틱 패킷 구성(4-A) |
| `evaluators/{semantic_packet_matcher,relation_consistency,attribute_consistency}.py` | 패킷 인식 검증(4-A) |
| `controllers/regeneration_policy.py` | 실패 양상 인식 재시도(4-A) |
| `video/{scene_change_detector,keyframe_extractor,semantic_delta,motion_residual,temporal_pipeline}.py` | 키프레임/GOP·델타·시간적 재사용(4-B) |
| `evaluators/temporal_consistency.py`, `scripts/evaluate_video.py` | 시간적 지표 + 비디오 CLI(4-B) |
| `channels/{rayleigh,fast_fading,packet_drop,measurement,complex_ops}.py` | 추가 채널 + 수신기 evidence(5-A) |
| `models/{channel_condition_encoder,reliability_head,diffusion_wrapper_channel}.py` | 채널 evidence 인코딩 + adapter 조건화(5-A) |
| `controllers/channel_condition_policy.py`, `pipelines/channel_conditioned_infer.py` | latent/joint/blind one-pass 추론(5-A) |
| `acceleration/{ddim_sampler,consistency_decoder,early_exit,latency_profiler,water_filling}.py` | step-budget·few-step·early-exit·지연(5-B) |
| `evaluators/{hallucination_vqa,vqa_backend,semantic_reliability_v2,regeneration_search}.py` | VQA·SRS-v2·다중전략 search(5-C) |
| `controllers/adaptive_search_policy.py` | regeneration 전략 순서(5-C) |
| `scripts/benchmark_{latency,sampling}.py` | 지연/샘플링 벤치마크 CLI |

## 4. 호환성 shim

`pipeline.py`, `preprocessing.py`는 Phase 2 재구성 이전 import 경로를 유지하는 하위
호환 re-export다(실제 로직은 `pipelines/`, `utils/`에 있음). 메인 실행 경로가 아니다.

## 한 줄 정리

`scripts/`(진입) · `configs/`(설정) · `runtime/models/channels/guidance/`(모델 조립)
· `pipelines/`(오케스트레이션) · `evaluators/`(지표) · `utils/`(I/O·전처리·로깅) ·
shim(레거시 import). 즉 단일 모델 파일이 아니라 진입·설정·추론·평가·확장을 분리한
실험 프레임워크다.
</content>

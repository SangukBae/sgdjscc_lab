# sgdjscc_lab 프레임워크 파일 역할

## 목적

이 문서는 `sgdjscc_lab` 파일들을 실제 실행 흐름 순서로 정렬하여 전체 프레임워크에서의
역할에 매핑한다.

기본 정렬 순서는 다음과 같다:

1. 추론 프레임워크
2. 평가 프레임워크
3. Phase 4 / 5 확장 모듈
4. 호환성 모듈

핵심은, `sgdjscc_lab`이 원본 `SGDJSCC` 알고리즘 경로를 보존하되 코드를 명시적인
모듈로 재구성한다는 점이다.

추가된 논문 매핑 컬럼에서, Figure 1(b)의 블록 이름은 다음과 같다:

- `DeepJSCC Encoder`
- `Semantic Extractor`
- `Semantic side information encoder`
- `Wireless Channel`
- `Semantic side information decoder`
- `Diffusion Denoiser`
- `DeepJSCC Decoder`

어떤 파일이 실험 실행을 위한 인프라일 뿐이면 `Figure 1(b) 외부`로 표시한다. 한 파일이
여러 블록을 함께 조율하면 `블록 간 오케스트레이션`으로 표시한다.

---

## 1. 추론 프레임워크 순서

| 순서 | 프레임워크 단계 | 주요 파일 | 프레임워크 내 모듈 역할 | Figure 1(b) 블록 |
|---|---|---|---|---|
| 1 | CLI 진입 | `scripts/infer_images.py` | 추론 시작, CLI 인자 파싱, config 로드, device 해석, 모델 빌드, 배치 추론 실행. | Figure 1(b) 외부 |
| 2 | Config 로딩 및 override | `src/sgdjscc_lab/config.py` | YAML config 로드, `_defaults_` fragment 병합, 상대 경로 해석, CLI override 적용. | Figure 1(b) 외부 |
| 3 | Config fragment | `configs/default.yaml` | 기본 AWGN 추론용 단일 파일 기본 config. | Figure 1(b) 외부 |
| 4 | Config fragment | `configs/composed.yaml` | channel/model/infer/eval fragment를 조립하는 composed config 진입점. | Figure 1(b) 외부 |
| 5 | Config fragment | `configs/channel/awgn.yaml` | `snr_db` 같은 채널 레벨 AWGN 설정 정의. | Wireless Channel |
| 6 | Config fragment | `configs/model/sgdjscc.yaml` | 모델/가이드/확산 관련 추론 옵션 정의. | 블록 간 오케스트레이션 |
| 7 | Config fragment | `configs/infer/awgn.yaml` | 입력 경로, 출력 경로, 런타임 device 기본값 정의. | Figure 1(b) 외부 |
| 8 | 재현성 설정 | `src/sgdjscc_lab/utils/seed.py` | Python, NumPy, PyTorch, cuDNN의 랜덤 시드 고정. | Figure 1(b) 외부 |
| 9 | Device 및 모델 조립 | `src/sgdjscc_lab/runtime.py` | device 문자열을 `torch.device`로 변환하고 추론용 전체 모델 번들을 조립. | 블록 간 오케스트레이션 |
| 10 | 외부 코드 브리지 | `src/sgdjscc_lab/_sgdjscc.py` | `SGDJSCC/`를 `sys.path`에 주입해 lab 패키지가 원본 베이스라인 모듈을 재사용하게 함. | Figure 1(b) 외부 |
| 11 | JSCC 코어 모델 | `src/sgdjscc_lab/models/jscc_model.py` | 원본 SGDJSCC forward 경로가 사용하는 VAE 인코더/디코더, blind SNR 예측기, canny 전송 네트워크를 빌드. | DeepJSCC Encoder / Semantic side information encoder / Semantic side information decoder / DeepJSCC Decoder |
| 12 | 채널 모듈 | `src/sgdjscc_lab/channels/awgn.py` | AWGN latent 전송 단계를 교체 가능한 채널 모듈로 구현. | Wireless Channel |
| 13 | 확산 시맨틱 파이프라인 | `src/sgdjscc_lab/models/diffusion_wrapper.py` | 시맨틱 복원을 위한 확산 backbone, 선택적 ControlNet, CLIP, 공유 VAE 로드. | Diffusion Denoiser |
| 14 | 모델 컨테이너 | `src/sgdjscc_lab/models/model_bundle.py` | JSCC 모델, 확산 파이프라인, 가이드 추출기, device/offload 설정을 하나의 런타임 번들로 패킹. | Figure 1(b) 외부 |
| 15 | 입력 파일 탐색 및 이미지 I/O | `src/sgdjscc_lab/io.py` | 이미지 파일 나열, 텐서로 로드, 복원 결과 저장. | Figure 1(b) 외부 |
| 16 | 패치 전처리 | `src/sgdjscc_lab/utils/preprocessing.py` | 필요 시 입력 crop/resize, 이미지를 `128x128` 패치로 분할, 복원 패치를 다시 병합. | DeepJSCC Encoder 이전 전(前)단계 지원 |
| 17 | 배치 오케스트레이션 | `src/sgdjscc_lab/pipelines/infer_pipeline.py` | 메인 추론 파이프라인: 이미지 순회, 패치 준비, 패치별 추론 실행, 출력 기록. | 블록 간 오케스트레이션 |
| 18 | 텍스트 가이드 추출 | `src/sgdjscc_lab/guidance/text_extractor.py` | `use_text=true`일 때 시맨틱 텍스트 가이드로 쓰이는 BLIP2 캡션 생성. | Semantic Extractor |
| 19 | 엣지 가이드 추출 | `src/sgdjscc_lab/guidance/edge_extractor.py` | 구조 가이드로 쓰이는 MuGE soft edge map과 uncertainty map 생성. | Semantic Extractor |
| 20 | 코어 forward pass | `src/sgdjscc_lab/pipelines/infer_pipeline.py` | 주요 순서 블록 실행: soft edge 전처리 → VAE encode/normalize → AWGN → mask/power scalar → step matching → canny 재전송 → canny latent → 확산 디노이징 → 최종 decode. | DeepJSCC Encoder → Semantic side information encoder → Wireless Channel → Semantic side information decoder → Diffusion Denoiser → DeepJSCC Decoder |
| 21 | 출력 저장 | `src/sgdjscc_lab/io.py` | 복원된 이미지 텐서를 출력 디렉터리에 이미지 파일로 저장. | Figure 1(b) 외부 |

### 추론 요약

추론 backbone은 사실상 다음과 같다:

`infer_images.py → config.py → runtime.py → 모델 빌더 → io/preprocessing → infer_pipeline.py → guidance → JSCC + AWGN + diffusion → save`

---

## 2. 평가 프레임워크 순서

| 순서 | 프레임워크 단계 | 주요 파일 | 프레임워크 내 모듈 역할 | Figure 1(b) 블록 |
|---|---|---|---|---|
| 1 | 평가 CLI 진입 | `scripts/evaluate.py` | 평가 시작, SNR 옵션 파싱, config 로드, eval context 빌드, 모델 빌드, 단일 SNR 또는 SNR-sweep 평가 실행. | Figure 1(b) 외부 |
| 2 | 평가 config fragment | `configs/eval/default.yaml` | 활성 지표, SNR sweep 리스트, CSV 경로, SRS 가중치, regeneration-loop 옵션 정의. | Figure 1(b) 외부 |
| 3 | 데이터셋 config fragment | `configs/dataset/kodak.yaml`, `configs/dataset/coco.yaml`, `configs/dataset/ade20k.yaml` | 평가 실험을 위한 데이터셋별 입력/참조/주석 설정 제공. | Figure 1(b) 외부 |
| 4 | 평가 파이프라인 | `src/sgdjscc_lab/pipelines/eval_pipeline.py` | 데이터셋 순회, 지표 계산, 선택적 CSV 로깅, SNR sweep 제어로 추론 파이프라인을 감쌈. | 블록 간 오케스트레이션 |
| 5 | 추론 코어 재사용 | `src/sgdjscc_lab/pipelines/infer_pipeline.py` | 평가 내부에서 실제 복원을 수행. 평가는 추론 알고리즘을 대체하지 않고 감싼다. | DeepJSCC Encoder → Semantic side information encoder → Wireless Channel → Semantic side information decoder → Diffusion Denoiser → DeepJSCC Decoder |
| 6 | 품질 지표 | `src/sgdjscc_lab/evaluators/quality.py` | PSNR, SSIM, LPIPS 계산. | Figure 1(b) 외부 |
| 7 | CLIP 시맨틱 지표 | `src/sgdjscc_lab/evaluators/clip_score.py` | CLIP 이미지-이미지 및 텍스트-이미지 유사도 계산. | Figure 1(b) 외부 |
| 8 | 객체 보존 지표 | `src/sgdjscc_lab/evaluators/object_preservation.py` | 원본에 존재하는 객체가 복원에서 얼마나 살아남는지 추정. | Figure 1(b) 외부 |
| 9 | 할루시네이션 지표 | `src/sgdjscc_lab/evaluators/hallucination.py` | 원본에는 없지만 복원에 나타나는 추가 객체를 추정. | Figure 1(b) 외부 |
| 10 | Semantic Reliability Score | `src/sgdjscc_lab/evaluators/semantic_reliability.py` | 시맨틱 지표들을 대표 지표인 SRS 점수로 결합. | Figure 1(b) 외부 |
| 11 | CSV 결과 로깅 | `src/sgdjscc_lab/utils/csv_logger.py` | 긴 평가 동안 이미지별 지표 행을 CSV 파일로 스트리밍. | Figure 1(b) 외부 |
| 12 | 요약 포매팅 | `src/sgdjscc_lab/utils/metrics_io.py` | 지표 행을 집계하고 콘솔 요약 표를 포맷. | Figure 1(b) 외부 |
| 13 | 선택적 재시도 경로 | `src/sgdjscc_lab/pipelines/regeneration_loop.py` | 시맨틱 신뢰도가 임계값 미만일 때 복원을 재실행하고 최적의 재시도 결과를 유지. | Diffusion Denoiser / DeepJSCC Decoder 주변의 블록 간 오케스트레이션 |

### 평가 요약

평가 backbone은 사실상 다음과 같다:

`evaluate.py → config.py/eval config → eval_pipeline.py → infer_pipeline.py → evaluators → csv_logger.py → metrics_io.py`

---

## 3. Phase 4 / 5 확장 모듈

### Phase 3 구조 가이드 확장

| 파일 | 역할 | Figure 1(b) 블록 |
|---|---|---|
| `src/sgdjscc_lab/guidance/depth_extractor.py` | 구조 조건화 및 평가를 위한 선택적 depth 가이드 추출기. | Semantic Extractor |
| `src/sgdjscc_lab/guidance/segmentation_extractor.py` | 영역 인식 가이드 및 분석을 위한 선택적 시맨틱 세그멘테이션 추출기. | Semantic Extractor |

### Phase 4 시맨틱 / 시간적 확장

| 파일 | 역할 | Figure 1(b) 블록 |
|---|---|---|
| `src/sgdjscc_lab/controllers/adaptive_guidance_controller.py`, `src/sgdjscc_lab/controllers/snr_guidance_policy.py` | 변경되지 않은 이미지 forward 경로 위에 얹은 SNR 인식 가이드/스텝 제어. | 블록 간 오케스트레이션 |
| `src/sgdjscc_lab/guidance/semantic_packet_extractor.py`, `object_extractor.py`, `relation_extractor.py`, `importance_estimator.py` | 캡션/객체/관계/속성/세그멘테이션/depth 단서로부터 시맨틱 패킷을 구성. | Semantic Extractor |
| `src/sgdjscc_lab/evaluators/semantic_packet_matcher.py`, `relation_consistency.py`, `attribute_consistency.py` | 원본 vs 복원 패킷을 비교하고 패킷 인식 시맨틱 일관성을 계산. | Figure 1(b) 외부 |
| `src/sgdjscc_lab/controllers/regeneration_policy.py` | 패킷/검증기 출력으로부터 실패 양상 인식 재시도 전략을 선택. | 블록 간 오케스트레이션 |
| `src/sgdjscc_lab/video/scene_change_detector.py`, `keyframe_extractor.py`, `semantic_delta.py`, `motion_residual.py`, `temporal_pipeline.py` | 키프레임/GOP 분할, 장면 전환 탐지, 패킷 델타 로직, 시간적 재사용, 단계적 prompt 스케줄링. | 블록 간 오케스트레이션 |
| `src/sgdjscc_lab/evaluators/temporal_consistency.py` | 시간적 SRS, 동일성(identity) 일관성, 시간적 세그멘테이션 IoU, 시간적 할루시네이션. | Figure 1(b) 외부 |
| `scripts/evaluate_video.py` | 비디오/키프레임 평가 CLI. | Figure 1(b) 외부 |

### Phase 5 채널 / 가속 / 검증기 확장

| 파일 | 역할 | Figure 1(b) 블록 |
|---|---|---|
| `src/sgdjscc_lab/channels/rayleigh.py`, `fast_fading.py`, `packet_drop.py`, `measurement.py` | 추가 채널 모델과, 채널 조건화 평가를 위한 수신기 evidence / 측정 추상화. | Wireless Channel |
| `src/sgdjscc_lab/models/channel_condition_encoder.py`, `reliability_head.py`, `diffusion_wrapper_channel.py` | 수신 채널 evidence를 조건 특징으로 인코딩하고, adapter 레벨의 채널 조건화 디코딩 정책을 적용. | Diffusion Denoiser / 블록 간 오케스트레이션 |
| `src/sgdjscc_lab/controllers/channel_condition_policy.py`, `src/sgdjscc_lab/pipelines/channel_conditioned_infer.py` | latent / joint / blind 모드를 갖는 one-pass 채널 조건화 추론 경로. | 블록 간 오케스트레이션 |
| `src/sgdjscc_lab/acceleration/ddim_sampler.py`, `consistency_decoder.py`, `early_exit.py`, `latency_profiler.py` | step-budget 제어, few-step 디코딩 인터페이스, 샘플러 내부 early exit, 지연 측정. | Diffusion Denoiser / Figure 1(b) 외부 |
| `src/sgdjscc_lab/evaluators/hallucination_vqa.py`, `vqa_backend.py`, `semantic_reliability_v2.py`, `regeneration_search.py` | 로컬 VQA backend, SRS-v2, 다중 전략 regeneration search를 통한 더 강한 시맨틱 검증. | Figure 1(b) 외부 |
| `src/sgdjscc_lab/controllers/adaptive_search_policy.py` | 실패 양상과 채널 상태에 따라 regeneration 전략 순서를 정함. | 블록 간 오케스트레이션 |
| `scripts/benchmark_latency.py`, `scripts/benchmark_sampling.py` | 지연 / 샘플링 트레이드오프 실험용 벤치마크 CLI. | Figure 1(b) 외부 |

이 파일들은 opt-in 연구 확장이다. config로 명시적으로 활성화하지 않는 한 기본
Phase 1–3 AWGN 추론 경로의 일부가 아니다.

---

## 4. 호환성 shim 모듈

| 파일 | 역할 | Figure 1(b) 블록 |
|---|---|---|
| `src/sgdjscc_lab/pipeline.py` | 구(舊) 최상위 파이프라인 API의 하위 호환 re-export. | Figure 1(b) 외부 |
| `src/sgdjscc_lab/preprocessing.py` | `utils/preprocessing.py`로 이동한 전처리 헬퍼의 하위 호환 re-export. | Figure 1(b) 외부 |
| `src/sgdjscc_lab/__init__.py` | 패키지 루트 마커 및 Python 패키지의 import 표면. | Figure 1(b) 외부 |

이 파일들은 그 자체로 메인 실행 경로가 아니다. Phase 2 모듈 재구성 이후 예전 import
경로가 깨지지 않도록 유지한다.

---

## 5. 한 줄 해석

`sgdjscc_lab`은 다음과 같이 구성된다:

- `scripts/`: 사용자 진입점
- `configs/`: 실험 설정
- `runtime/models/channels/guidance/`: 모델 조립 및 코어 구성 요소
- `pipelines/`: 실행 순서 및 오케스트레이션
- `evaluators/`: 연구 지표
- `utils/`: I/O, 전처리, 재현성, 로깅
- 호환성 shim: 예전 import 경로 지원

즉, 이 패키지는 단순한 "모델 파일 하나"가 아니라, 진입·설정·추론·평가·확장 지점을
서로 다른 모듈로 분리한 완전한 실험 프레임워크다.

> [← 문서 색인](./README.md)

# Phase 5 — 채널 조건화 · 저지연 · 강화 검증

> Phase 4와 달리 Phase 5는 확산 디코더 주변의 조건화 인터페이스와 AWGN을 넘어서는
> 채널 추상화를 수정할 수 있다. 모든 기능은 **기본값 off**의 opt-in 스캐폴드이며,
> 켜지 않으면 Phase 1~4 경로는 byte 단위로 변경되지 않는다.

## 마스터 스위치

```yaml
# configs/eval/default.yaml
use_phase5: false          # 기본 — 5-A/B/C 전체 비활성화

# Phase 5만 활성화
use_phase5: true
use_channel_conditioning: true
use_srs_v2: true
use_vqa_hallucination: false
use_regeneration_search: false
```

`use_phase5: false`이면 개별 플래그(`use_channel_conditioning`, `acceleration.*`,
`use_srs_v2`, `use_vqa_hallucination`, `use_regeneration_search`)는 무시된다.
Phase 4와 5는 독립 스위치다. 두 스위치 + 확장 플래그를 모두 켠 preset은
`configs/composed_phase5_full.yaml`. 런타임 체크는 `phase_gates.effective_flag(..., phase=5)`.

---

## 5-A — DiffCom 영감 채널 조건화

수신된 채널 신호 자체를 복원 조건으로 활용한다.

| 영역 | 모듈 | 상태 |
|---|---|---|
| 채널 | `channels/{rayleigh,fast_fading,packet_drop}.py` (+`build_channel`) | **구현** — AWGN 호환 `transmit()` + `observe()` |
| 측정 번들 | `channels/measurement.py` (`MeasurementBundle`) | **구현** — received/equalized/gain/noise_var/mask/SNR/reliability |
| 수신기 evidence 훅 | `pipelines/infer_pipeline.py` (`measurement_out`) | **구현** — 관측 전용 |
| 조건 인코더 | `models/channel_condition_encoder.py` | **구현(학습 불필요)** — `stats`(파라미터 없음) + `linear` 훅 |
| Reliability head | `models/reliability_head.py` | **구현(휴리스틱)** |
| 조건화 wrapper | `models/diffusion_wrapper_channel.py` | **adapter 레벨** — reliability가 guidance/steps 스케일, 조건 토큰 부착 |
| 정책/파이프라인 | `controllers/channel_condition_policy.py`, `pipelines/channel_conditioned_infer.py` | **구현 + `evaluate.py` 연결** — latent/joint/blind 모드 |

`use_channel_conditioning`이 켜지면 `evaluate_dataset`이
`OnePassChannelConditionedInference`를 빌드한다. 패치당 한 번 encode+transmit →
패치별 번들을 **이미지 레벨** 측정으로 집계 → 조건화된 cfg 결정 → **evidence를 만든
동일 received latent를 재사용**하며 디코딩(일회성 측정 forward 없음).

**근사인 부분**: 잡음 수신 신호는 (a) 실제 채널, (b) reliability로 스케일된
guidance/steps/blind-SNR, (c) 확산 init latent로 복원에 들어간다. 인코딩된 조건
토큰은 cfg에 부착되지만 **frozen SGD-JSCC denoiser가 소비하지 않는다** — FiLM /
cross-attention / posterior-gradient guidance는 재학습된 조건 인식 denoiser가 필요.

**Fast-fading water-filling (논문 Algorithm 4)** — `acceleration/water_filling.py`.
per-element 잡음레벨 `d_i=σ²/(g_i²+σ²)`(eq.12) 추적 → 공통 목표로 water-fill(eq.16)
→ DM 1-step f0 예측(eq.17) → 선택적 갱신. `infer_pipeline._run_water_filling_diffusion`이
`cfg.use_water_filling` + 패치별 `noise_level`이 있을 때 표준 decode 대신 라우팅.
알고리즘·배선·patch별 evidence는 CPU stub로 검증됨(`tests/test_water_filling.py`);
실제 **수치**만 MDTv2 체크포인트 의존.

---

## 5-B — 저지연 확산 & consistency

| 영역 | 모듈 | 상태 |
|---|---|---|
| Step-budget/DDIM | `acceleration/ddim_sampler.py` (`build_sampler_cfg`, `dynamic_step_budget`, `karras_schedule`) | **구현 + 연결** — `acceleration.sampler_steps`가 SNR별 디노이징 예산 변경 |
| Consistency 디코더 | `acceleration/consistency_decoder.py` | **인터페이스 + few-step 수식** — distilled student는 placeholder |
| Early exit | `acceleration/early_exit.py` | **checkpoint + 샘플러 내부** (아래) |
| 지연 프로파일러 | `acceleration/latency_profiler.py` | **구현** — total/decoder/per-step, CUDA 동기화 |
| 벤치마크 CLI | `scripts/benchmark_{latency,sampling}.py` | **구현** — 품질-대-지연 sweep |

**샘플러 내부 early-exit** — `early_exit_mode: intra_sampler`일 때 `_run_diffusion`이
`run_interruptible_sampling`을 사용한다. `check_interval`마다 채점하고 만족 시 루프를
**즉시 종료**(`generate_interruptible`가 SGD-JSCC 연속 DPM-Solver++(2M)를 재구동해
callback 훅 확보). `early_exit_metric`: `heuristic`(latent 수렴, 기본) 또는
`srs`/`srs_v2`(현재 clean latent를 디코딩해 원본 패치와 채점). 연속 샘플러만 커버,
discrete는 fallback.

**placeholder**: 학습된 consistency/distilled student. 디코더 API + few-step 수식 +
평가 경로는 완성되어 있어 학습된 모델을 끼워 넣을 수 있다.

---

## 5-C — 강화 검증기 & regeneration search

| 영역 | 모듈 | 상태 |
|---|---|---|
| VQA 할루시네이션 | `evaluators/hallucination_vqa.py` + `vqa_backend.py` | **구현 + 연결** — `mock`/`blip2`/`llava`/`mplug` backend, 실패 시 CLIP fallback(1회 경고) |
| SRS-v2 | `evaluators/semantic_reliability_v2.py` | **구현 + 연결** — base + packet + temporal + VQA |
| Regeneration search | `evaluators/regeneration_search.py` | **구현 + 연결** — 강/약 텍스트, 무조건, 채널 조건화 재시도; `verify_metric`(`srs`/`srs_v2`)로 최적 선택 |
| Search 순서 | `controllers/adaptive_search_policy.py` | **구현** — 실패 양상 + 채널 상태로 전략 순서 결정 |

VQA 기본 backend는 `Salesforce/blip2-opt-2.7b-coco`(캡션 추출기가 이미 로드하는
체크포인트). 무거운 모델은 지연 import, 로드 실패는 캐시. 기본 `type: none`(CLIP fallback).

---

## 통합 · config 표면

Phase 5 기능은 독립 모듈이 아니라 **메인 `scripts/evaluate.py` 경로**에서 도달 가능하다:
`use_channel_conditioning`(one-pass 라우팅), `acceleration.sampler/sampler_steps`,
`acceleration.early_exit`, `use_srs_v2`(+`vqa_backend`), `use_regeneration_search`,
`channel_condition` config 블록(encoder/policy가 소비).

config: `configs/channel/{rayleigh,fast_fading,packet_drop}.yaml`,
`model/channel_conditioned.yaml`, `acceleration/default.yaml`, `eval/phase5.yaml`,
composed 예시 `composed_phase5.yaml`.

---

## 참고 매핑 · 남은 한계

- **DiffCom** → `observe()`→`MeasurementBundle` 패턴, latent/joint/blind 조건화 모드.
  Posterior-gradient consistency 조건화는 재현 안 함(frozen denoiser).
- **LDM-enabled SemCom** → `karras_schedule` + consistency few-step 샘플링, 지연
  프로파일러. Consistency *학습*은 재현 안 함.

**남은 한계**
- 채널 조건 토큰은 cfg에 부착되지만 frozen SGD-JSCC denoiser가 소비하지 않음
  (조건화는 received-latent init + reliability 스케일 guidance/steps로 작동).
- 학습된 consistency/distilled student 없음(few-step은 결정론적 근사).
- 샘플러 내부 인터럽트는 연속 샘플러만, discrete는 fallback.
- Fading `signal_scale`/step-matching은 AWGN 공식을 재사용(fading에선 근사).
</content>

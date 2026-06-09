> [← 문서 색인](./README.md)

# Phase 5 — 계획 & 구현 현황

- [마스터 스위치](#master-switch)
- [Phase 5 계획](#phase-5-plan)
- [Phase 5 구현 현황](#phase-5-implementation-status)

---

<a id="master-switch"></a>

## 마스터 스위치

모든 Phase 5 기능은 **기본값 off**다. 최상위 플래그 `use_phase5` 하나로 phase
전체를 제어한다:

```yaml
# configs/eval/default.yaml (또는 composed config)
use_phase5: false   # 기본값 — 모든 Phase 5-A/B/C 기능 비활성화
```

**규칙**: `use_phase5: false`이면, 아래 나열된 개별 기능 플래그
(`use_channel_conditioning`, `acceleration.*`, `use_srs_v2`,
`use_vqa_hallucination`, `use_regeneration_search`)는 config에서 명시적으로
`true`로 설정되어 있어도 런타임에 무시된다.

**중요**: `use_phase5: true`는 Phase 4 기능을 **자동으로 켜지 않는다**. Phase 4와
Phase 5는 독립적인 마스터 스위치다.

### Phase 5만 활성화 (Phase 4 없이)

```yaml
use_phase4: false   # Phase 4-A/B는 꺼둠
use_phase5: true

use_channel_conditioning: true
use_srs_v2: true
use_vqa_hallucination: false
use_regeneration_search: false
```

### Phase 4 + Phase 5 활성화 (풀 스택)

```yaml
use_phase4: true
use_phase5: true
```

preset `configs/composed_phase5_full.yaml`은 두 스위치가 모두 true이고 확장 플래그가
전부 활성화되어 있다.

### 헬퍼 함수

모든 런타임 체크는 `sgdjscc_lab.phase_gates.effective_flag`를 거친다:

```python
from sgdjscc_lab.phase_gates import effective_flag, phase5_enabled

# use_phase5가 false이면 raw 플래그 값과 무관하게 False를 반환한다.
use_channel_cond = effective_flag(cfg, "use_channel_conditioning", phase=5)
```

---

<a id="phase-5-plan"></a>

# Phase 5 계획

Phase 5는 `모델 및 채널 연구 단계`다.
Phase 4와 달리, Phase 5는 확산 디코더 주변의 조건화 인터페이스를 수정하고 AWGN을
넘어서는 새 채널 추상화를 추가하는 것이 허용된다.

Phase 5는 두 개의 주요 트랙과 하나의 보조 평가 트랙으로 나뉜다:

1. `Phase 5-A`: DiffCom에서 영감받은 CSI-free 채널 조건화
2. `Phase 5-B`: 저지연 확산 및 consistency distillation
3. `Phase 5-C`: 더 강한 시맨틱 검증 및 다중 전략 regeneration

## Phase 5-A: DiffCom 영감 채널 조건화 확산

주요 참고:

- `paper/diffcom/README.md`
- `paper/diffcom/main_diffcom.py`
- `paper/diffcom/conditioning_method/diffcom.py`
- `paper/diffcom/guided_diffusion/measurement.py`

DiffCom에서 가져오는 것:

1. 수신된 채널 신호 자체가 확산 조건으로 유용하다
2. 채널 관측은 operator 스타일 추상화로 노출되어야 한다
3. 블라인드 복원을 명시적으로 평가해야 한다
4. 조건화는 다음을 지원해야 한다:
   - 표준 채널 인식 모드
   - 고충실도 joint 모드
   - 블라인드 모드

계획 파일:

```text
src/sgdjscc_lab/
├── channels/
│   ├── rayleigh.py
│   ├── fast_fading.py
│   └── packet_drop.py
├── models/
│   ├── channel_condition_encoder.py
│   ├── diffusion_wrapper_channel.py
│   └── reliability_head.py
├── controllers/
│   └── channel_condition_policy.py
└── pipelines/
    └── channel_conditioned_infer.py
```

구현 단계:

1. `channels/rayleigh.py`
   - 현재 AWGN 채널 추상화와 호환되는 slow Rayleigh fading API 구현
   - 출력:
     - 잡음 latent
     - 등화(equalized) latent
     - 채널 이득(gain)
     - 잡음 분산
2. `channels/fast_fading.py`
   - 설정 가능한 블록 길이로 심볼/블록 레벨 fading 구현
   - 현재 패치 레벨 이미지 추론과의 호환성 유지
3. 현재 SGD-JSCC 경로에서 수신기 측정값 노출
   - 내부 추론 반환 경로를 수정해 중간값을 선택적으로 수집 가능하게 함:
     - `encode_features_hat`
     - `mask_token`
     - `power_scalar`
     - 추정 SNR
     - 선택적 위상 추정값
4. `channel_condition_encoder.py`
   - raw 채널 관측을 조건 토큰으로 압축
   - 첫 조건 벡터:
     - 잡음 latent 특징
     - 정규화된 power scalar
     - mask 또는 reliability map
     - SNR 추정값
     - 위상 추정값
   - 이후 확장:
     - fast fading 왜곡 특징
     - 토큰별 reliability
5. `diffusion_wrapper_channel.py`
   - 현재 확산 backbone 주위에 채널 조건화 wrapper 구축
   - 첫 구현은 전체 denoiser를 교체하지 않고 경량 adapter 사용:
     - FiLM 스타일 조건화
     - 추가 context 토큰
     - 채널 토큰용 선택적 cross-attention 분기
6. `channel_conditioned_infer.py`
   - DiffCom 명명에서 영감받은 세 가지 실험 모드 지원:
     - `latent_conditioned`
     - `joint_conditioned`
     - `blind_conditioned`
7. 블라인드 실험 설계
   - 미지 SNR AWGN
   - CSI 없는 slow fading
   - 불완전 또는 무(無) CSI fast fading
   - 비교:
     - 추정 SNR 컨트롤러만
     - 채널 토큰 조건화
     - 채널 토큰 조건화 + regeneration

핵심 구현 노트:

DiffCom의 `operator.observe_and_transpose()` 패턴을 `sgdjscc_lab`에서는 내부 측정
번들 추상화로 미러링하여, 모든 채널 실험이 `reconstruction`과 `수신기 evidence`를
모두 반환하도록 한다. 이는 현재의 결정론적 이미지 파이프라인에서 DiffCom 스타일의
posterior-guided 복원으로 가는 가장 깔끔한 다리다.

## Phase 5-B: 저지연 확산 및 consistency distillation

주요 참고:

- `paper/LDM-enabled-SemCom-system/train_DIV2K/consistency_models.py`
- `paper/LDM-enabled-SemCom-system/train_DIV2K/consistency_model_training.py`
- `paper/LDM-enabled-SemCom-system/train_DIV2K/t_calculate.py`
- `paper/LDM-enabled-SemCom-system/train_MNIST/eavluation.py`

저지연 LDM 코드에서 가져오는 것:

1. 명시적 consistency 샘플링 유틸리티
2. consistency 학습 및 distillation 스캐폴딩
3. Karras 스케줄 기반 샘플링
4. EECD 스타일 지연 분석을 위한 타이밍 측정
5. one-step 및 few-step 샘플링 비교

계획 파일:

```text
src/sgdjscc_lab/
├── acceleration/
│   ├── ddim_sampler.py
│   ├── consistency_decoder.py
│   ├── early_exit.py
│   └── latency_profiler.py
└── scripts/
    ├── benchmark_latency.py
    └── benchmark_sampling.py
```

구현 단계:

1. 베이스라인 지연 벤치마크
   - 내부 실험에 사용한 동일 하드웨어 프로파일에서 현재 50-step 파이프라인 측정
   - 리포트:
     - end-to-end 지연
     - 디코더 지연
     - 스텝별 디노이징 비용
2. DDIM ablation
   - 다음에 대한 설정 가능한 스케줄 추가:
     - 50-step
     - 20-step
     - 10-step
     - 5-step
   - SRS와 LPIPS를 주요 품질 제약으로 유지
3. `consistency_decoder.py`
   - `ConsistencySamplingAndEditing`에서 영감받은 few-step 디코더 인터페이스 프로토타입
   - 초기 목표:
     - 1-step
     - 2-step
     - 5-step
4. `latency_profiler.py`
   - `t_calculate.py`의 아이디어 포팅
   - 리포트:
     - 디코더 지연
     - 총 지연
     - 50-step 베이스라인 대비 유효 speedup
5. `early_exit.py`
   - 디노이징 중 중간 복원 평가
   - 다음일 때 조기 종료:
     - SRS가 임계값 초과
     - 직전 스텝 대비 개선이 허용 오차 미만
6. 동적 라우팅 정책
   - 채널 상태와 예측 reliability 결합:
     - 고 SNR + 높은 예측 SRS → 1~5 step
     - 중 SNR → 10~20 step
     - 저 SNR 또는 블라인드 채널 → 30~50 step

기대 출력:

- 품질 vs 지연 곡선
- SRS vs 지연 곡선
- SNR 구간별 step 수 권장 표

## Phase 5-C: 더 강한 검증기 및 regeneration search

Phase 5-C는 Phase 5-A와 5-B를 출판 가능하고 방어하기 쉽게 만드는 보조 평가 레이어다.

계획 파일:

```text
src/sgdjscc_lab/
├── evaluators/
│   ├── hallucination_vqa.py
│   ├── semantic_reliability_v2.py
│   └── regeneration_search.py
└── controllers/
    └── adaptive_search_policy.py
```

구현 단계:

1. 순수 CLIP 휴리스틱의 더 강한 대안으로 VQA 스타일 할루시네이션 점검 추가
2. 패킷 인식 및 시간 인식 항을 포함한 `SRS-v2` 생성
3. 단발(one-shot) regeneration을 다중 전략 search로 교체:
   - 강한 텍스트 / 약한 엣지
   - 약한 텍스트 / 강한 엣지
   - 무조건(unconditional) fallback
   - 채널 조건화 재시도
4. 첫 성공 복원이 아니라 검증된 SRS가 가장 높은 출력을 최종 선택

## Phase 5 완료 기준

다음이 모두 참일 때 Phase 5를 완료로 간주한다:

1. AWGN과 Rayleigh 채널 설정이 모두 지원됨
2. 블라인드 또는 불완전 CSI 모드가 적어도 하나 구현·벤치마크됨
3. 채널 조건화 확산이 적어도 하나의 비자명한 미지 채널 설정에서 시맨틱 신뢰도를 개선
4. few-step 또는 distilled 디코딩이 제한된 SRS 열화로 명확한 지연 감소를 제공
5. 더 강한 검증기와 regeneration search가 실험 루프에 통합됨

## Phase 5 실험 설계

채널:

- 알려진/미지 SNR을 갖는 AWGN
- slow Rayleigh fading
- 설정 가능한 블록 길이의 fast fading
- packet drop / 시맨틱 패킷 손상

베이스라인:

1. `sgdjscc_lab`의 원본 SGD-JSCC 경로
2. 적응형 가이드만 적용한 Phase 4 모델
3. 블라인드 모드 없는 채널 조건화 모델
4. 블라인드 채널 조건화 모델
5. few-step consistency 디코더 변형들

지표:

- 품질:
  - PSNR
  - SSIM
  - LPIPS
- 시맨틱:
  - CLIP 이미지-이미지
  - CLIP 텍스트-이미지
  - 객체 보존
  - 관계 일관성
  - 속성 일관성
  - 세그멘테이션 일관성
- 신뢰도:
  - SRS
  - SRS-v2
  - 할루시네이션율
  - regeneration 성공률
- 효율:
  - sec/image
  - sec/frame
  - 디노이징 스텝 수
  - 50-step 베이스라인 대비 speedup

## 권장 실행 우선순위

남은 연구 우선순위의 실용적 구현 순서는 다음과 같다:

1. `FAST-GSC` 영감 Phase 4 패킷 및 키프레임 프레임워크
2. 저지연 LDM 영감 Phase 5-B 가속
3. Phase 5-C의 더 강한 검증기 / regeneration 작업
4. `DiffCom` 영감 Phase 5-A 채널 조건화

이 순서는 의도적이다:

- 먼저 ETRI를 직접 향하는 핵심 한계를 해결한다:
  명시적 시맨틱, 더 강한 가이드, 부가정보 제어
- 그다음 확산 지연을 줄여 프로토타입을 실용적으로 실행 가능하게 만든다
- 마지막으로 채널 적응과 fast-fading 견고성을 후속 트랙으로 심화한다

---

<a id="phase-5-implementation-status"></a>

# Phase 5 구현 현황

Phase 5는 **config 기반, opt-in 스캐폴드**로 구현된다: 모든 기능이 기본값 off이므로,
명시적으로 활성화하지 않는 한 Phase 1–4 이미지/비디오 경로는 byte 단위로 변경되지
않는다. (계획상) 목표는 학습된 SOTA 재현이 아니라 *실험적으로 실행 가능한 구조 +
테스트 가능한 최소 동작 + Phase-4 호환성*이었다.

## Phase 5-A — DiffCom 영감 채널 조건화 확산

| 영역 | 모듈 | 상태 |
|---|---|---|
| 채널 | `channels/rayleigh.py`, `channels/fast_fading.py`, `channels/packet_drop.py` (+`channels/__init__.build_channel`) | **구현됨** — AWGN 호환 `transmit()` + 풍부한 `observe()` |
| 측정 번들 | `channels/measurement.py` (`MeasurementBundle`) | **구현됨** — received/equalized/gain/noise_var/mask/SNR/reliability + 선택적 실제 수신기 evidence |
| 수신기-evidence 훅 | `pipelines/infer_pipeline.py` (`measurement_out`) | **구현됨** — 관측 전용; `encode_features_hat`/`mask_token`/`power_scalar`/SNR-est 채움 |
| 조건 인코더 | `models/channel_condition_encoder.py` | **구현됨 (학습 불필요)** — `stats` 모드(파라미터 없음) + `linear` 학습 가능 훅 |
| Reliability head | `models/reliability_head.py` | **구현됨 (휴리스틱)** — SNR/gain/mask 신뢰도 + 학습 가능 훅 |
| 이미지 레벨 측정 일관성 | `channels/measurement.py` (`aggregate_bundles`) + `pipelines/infer_pipeline.py` (`run_image_channel_conditioned`) | **구현됨** — 채널 evidence가 모든 패치에 걸쳐 집계되고, 디코더는 그 evidence를 만든 동일한 received latent를 재사용 |
| 조건화 wrapper | `models/diffusion_wrapper_channel.py` | **adapter 레벨** — 신뢰도가 guidance/steps를 스케일; blind→blind-SNR; 조건 토큰을 `cfg.channel_condition_tokens`로 부착; encoder/policy는 `channel_condition` config 블록에서 구성 |
| 정책 / 파이프라인 | `controllers/channel_condition_policy.py`, `pipelines/channel_conditioned_infer.py` | **구현됨** — `latent`/`joint`/`blind` 모드; config 게이트 `use_channel_conditioning`; **`scripts/evaluate.py`에 연결됨**(메인 eval 루프가 복원을 이를 통해 라우팅) |

통합: `use_channel_conditioning`이 설정되면 `evaluate_dataset`이
`OnePassChannelConditionedInference`를 빌드하고, 각 이미지를 현재의 **one-pass**
경로로 복원한다. `run_image_channel_conditioned`는 패치당 한 번 encode+transmit을
수행하고, 그 결과 패치별 번들을 `aggregate_bundles`로 하나의 **이미지 레벨** 측정으로
집계하며, 조건화된 cfg를 결정한 뒤, evidence를 생성한 동일한 received latent를
재사용하며 디코딩한다. 즉, 정책은 별도 forward pass의 일회성 측정을 보지 않는다.

**완전한 DiffCom이 아닌 근사**인 부분: 잡음 섞인 수신 신호는 (a) `JSCCModel.channel`
내부의 실제 채널, (b) reliability로 스케일된 `cfg`(guidance/steps/blind-SNR),
(c) 확산 init으로서의 received latent를 통해 복원에 들어간다. 인코딩된 조건 토큰은
cfg에 부착되지만 **frozen SGD-JSCC denoiser가 소비하지 않는다** — FiLM /
cross-attention / posterior-gradient guidance는 조건 인식(재학습된) denoiser가
필요하므로 다음 단계로 남겨둔다.

## Phase 5-B — 저지연 확산 & consistency distillation

| 영역 | 모듈 | 상태 |
|---|---|---|
| Step-budget / DDIM ablation | `acceleration/ddim_sampler.py` (`build_sampler_cfg`) | **구현 + 연결됨** — `evaluate_dataset`이 SNR별 cfg에 적용하므로 `acceleration.sampler_steps`가 실제로 디노이징 예산을 변경; 기본 off |
| 동적 라우팅 | `acceleration/ddim_sampler.py` (`dynamic_step_budget`) | **구현됨** — SNR/신뢰도 → step budget |
| Karras 스케줄 | `acceleration/ddim_sampler.py` (`karras_schedule`) | **구현됨** — LDM SemCom 코드에서 포팅 |
| Consistency 디코더 | `acceleration/consistency_decoder.py` | **인터페이스 + few-step 수식** — `baseline`/`fewstep`/`distilled_placeholder`; distilled student는 깨끗한 placeholder(few-step으로 fallback) |
| Early exit | `acceleration/early_exit.py` | **checkpoint 레벨 + 샘플러 내부**(아래 "해결된 한계" 참조) |
| 지연 프로파일러 | `acceleration/latency_profiler.py` | **구현됨** — total/decoder/per-step, CUDA 동기화 |
| 벤치마크 CLI | `scripts/benchmark_latency.py`, `scripts/benchmark_sampling.py` | **구현됨** — 품질-대-지연 sweep |

**placeholder**인 부분: 학습된 consistency/distilled student 디코더(teacher-student
학습은 재현되지 않음); 디코더 API + few-step 샘플링 수식 + 평가 경로는 완성되어 있어
distilled 모델을 끼워 넣을 수 있다.

## Phase 5-C — 더 강한 검증기 & 다중 전략 regeneration

| 영역 | 모듈 | 상태 |
|---|---|---|
| VQA 할루시네이션 | `evaluators/hallucination_vqa.py` + `evaluators/vqa_backend.py` | **구현 + 연결됨** — 주입 가능한 `vqa_fn`; 실제 로컬 backend(`mock`/`blip2`/`llava`/`mplug`) + CLIP fallback(아래 "해결된 한계" 참조) |
| SRS-v2 | `evaluators/semantic_reliability_v2.py` (+ `EvalContext._get_srs_v2`, `use_srs_v2`) | **구현 + 연결됨** — `_compute_metrics`가 shortcut이 아니라 전체 `SemanticReliabilityV2Evaluator`(base + packet + temporal + VQA) 실행; 계산된 base SRS 재사용 |
| Regeneration search | `evaluators/regeneration_search.py` (+ `eval_pipeline._run_regeneration_search`, `use_regeneration_search`) | **구현 + 연결됨** — 강/약 텍스트, 무조건, 채널 조건화 재시도(채널 조건화 경로를 실제로 재실행); **설정된** `regeneration_search.verify_metric`(`srs` 또는 `srs_v2`, 후자는 각 후보를 packet + VQA 레이어로 채점)로 최적 선택; `regeneration_strategy` 기록 |
| Search 순서 | `controllers/adaptive_search_policy.py` | **구현됨** — 실패 양상 + 채널 상태로 전략 순서 결정 |

## 참고 매핑 (Phase 5)

- **DiffCom** (`paper/diffcom/…`): `observe()`→`MeasurementBundle` 패턴이
  `operator.observe()`를 미러링; `latent`/`joint`/`blind` 모드가 DiffCom의 조건화
  모드를 미러링; 블라인드 경로가 `blind_diffcom`(미지 SNR / 불완전 CSI)을 미러링.
  Posterior-gradient consistency 조건화는 재현되지 *않음*(frozen denoiser).
- **LDM-enabled SemCom** (`paper/LDM-enabled-SemCom-system/…`): `karras_schedule`과
  consistency few-step 샘플링 루프는 `consistency_models.py` / `t_calculate.py`에서
  포팅; 지연 프로파일러는 `t_calculate.py` 타이밍 아이디어를 미러링. Consistency
  *학습*은 재현되지 않음.
- **FAST-GSC 연속성**: Phase-4 단계적 prompt 스케줄, 패킷 시맨틱, 시간적 파이프라인은
  손대지 않음; SRS-v2가 Phase-4의 `temporal_srs`를 소비하고, 채널 조건화 wrapper는
  `cfg.prompt_override` / `cfg.staged_prompts`를 보존한다.

## Phase 5 config 표면(surface)

`configs/channel/{rayleigh,fast_fading,packet_drop}.yaml`,
`configs/model/channel_conditioned.yaml`, `configs/acceleration/default.yaml`,
`configs/eval/phase5.yaml`, 그리고 composed 예시 `configs/composed_phase5.yaml`.
키: `channel`, `csi`, `use_channel_conditioning`, `condition_mode`,
`acceleration.sampler` / `sampler_steps`, early-exit 임계값, 동적 라우팅,
`use_srs_v2`, `use_regeneration_search`.

## 통합 현황 (리뷰 이후)

Phase 5 기능들은 (독립 모듈만이 아니라) **메인** `scripts/evaluate.py` 경로에서
도달 가능하다:

- `use_channel_conditioning` → 복원이 **one-pass** 채널 조건화 파이프라인으로 라우팅됨(아래 참조).
- `acceleration.sampler` / `sampler_steps` → SNR별 실행 config에 적용됨.
- `acceleration.early_exit` (`early_exit_mode: intra_sampler`) → 확산 루프가 중간에 멈춤(아래 참조).
- `use_srs_v2` (+ `use_vqa_hallucination` / `vqa_backend`) → 실제 로컬 VQA backend를 갖춘 전체 SRS-v2 평가기(아래 참조).
- `use_regeneration_search` → eval 루프에서 다중 전략 search, 설정된 `verify_metric`(`srs` / `srs_v2`)로 최적.
- `channel_condition` config 블록 → encoder/policy가 소비(죽은 config 없음).

## 해결된 한계 (이번 반복)

각 항목은 **구현됨 / 연결됨 / 근사됨 / fallback / 미구현**으로 태그된다.

**1. 실제 로컬 VQA backend** (이전: VQA가 외부 backend 필요)
- *구현됨*: `evaluators/vqa_backend.py` — `mock` / `blip2` / `llava` / `mplug`용
  `build_vqa_backend` adapter, 작은 `answer(image, question)->str` 계약; 무거운
  모델은 모두 **지연 import**. 기본 BLIP-2 모델은 `Salesforce/blip2-opt-2.7b-coco`
  (캡션 추출기가 이미 로드하는 known-good 체크포인트); backend 로드 실패는 캐시됨
  (반복 재로드 없음).
- *연결됨*: `vqa_backend` config → `EvalContext._get_srs_v2` →
  `VQAHallucinationEvaluator.from_config` → SRS-v2의 할루시네이션 레이어.
- *fallback*: `transformers`/weights 누락 또는 런타임 backend 오류 시, 경고를
  **한 번** 로그하고, 나머지 실행 동안 VQA를 비활성화하며, CLIP 할루시네이션
  휴리스틱으로 degrade(메서드 `clip_fallback` / `vqa_error_fallback`).
- *미구현*: BLIP-2/LLaVA/mPLUG weights는 번들되지 않음; 기본 `type: none`(CLIP
  fallback). LLaVA/mPLUG adapter는 best-effort.

**2. One-pass 채널 조건화** (이전: 이미지당 추가 측정 forward)
- *구현됨*: `infer_pipeline`이 `_encode_and_transmit`(encode → channel → mask/power
  → step-match, 패치별 `ForwardArtifacts` + 수신기 evidence 구성)와
  `_decode_diffusion`(canny → ControlNet latent → diffusion → decode)으로 분할됨.
  표준 single-call 경로는 수치적으로 변경되지 않음.
- *연결됨*: `run_image_channel_conditioned`가 **패치당 한 번** encode+transmit을
  실행하고, 번들을 **이미지 레벨** 측정으로 집계하며, 조건화된 cfg를 결정한 뒤,
  **동일한 received latent를 재사용**하며 디코딩 — 따라서 일회성 측정 forward(및
  record/replay tape)가 사라진다. 실제 적용된 조건화 cfg는 정직한 로깅을 위해
  `info["resolved_cfg"]` / 실행 payload의 `"cfg"`로 반환됨(더 이상 base cfg 아님).
- *구현됨 (조건 소스 == 디코더 init)*: 번들의 `best_estimate`(`ChannelConditionEncoder`가
  읽는 값)는 디코더의 확산 init latent `encode_features_hat / power_scalar`로
  설정됨 — `use_jscc_feature`가 켜졌을 때(조건화 경로가 강제) `_run_diffusion`이
  `latent_init`으로 쓰는 **동일** 텐서. 채널 descriptor(gain / noise / mask /
  reliability / SNR)는 realisation에서 복사되며 변형하지 않음.
- *근사됨*: `csi`/blind는 guidance와 전송된 latent에 영향을 줌; phase-1 step-match는
  여전히 base cfg의 `use_gt_csi`를 사용(조건화는 guidance/steps를 바꾸고 이미 계산된
  step은 아님). `ChannelTape`는 선택적 레거시 기능으로 남음(이 경로에서는 더 이상
  사용 안 함).

**3. 샘플러 내부 인터럽트 early-exit** (이전: checkpoint 재렌더만)
- *구현됨*: `acceleration/early_exit.run_interruptible_sampling` — `check_interval`
  스텝마다 채점하고 `EarlyExitController`가 만족되는 순간 **종료**하는 단일 루프
  (사용되지 않는 스텝은 실행되지 않음).
- *구현됨*: `models/diffusion_wrapper.generate_interruptible` — generator의 공개
  헬퍼로 SGD-JSCC **연속 DPM-Solver++(2M)** 업데이트를 재구동(원본 `generate`는
  callback 훅이 없고 `SGDJSCC/`는 read-only — `_INTERRUPTIBLE_REQUIRES`와 루프
  인용 참조), 진정한 루프 중간 종료 가능.
- *연결됨 (세 가지 지표 모두)*: `acceleration.early_exit` +
  `early_exit_mode: intra_sampler`일 때 `_run_diffusion`이 사용. `early_exit_metric`
  존중: `heuristic`(저렴한 latent 수렴, 기본) **또는** `srs` / `srs_v2` — 검증된
  지표는 루프의 현재 clean latent(`state["x0"]`)를 디코딩해 캐시된
  `SemanticReliability(/V2)Evaluator`로 **원본 패치**와 채점
  (`_resolve_early_exit_score_fn` → `_build_early_exit_score_fn`). 원본 패치가
  `_decode_diffusion`에 전달되므로 검증 채점은 API 전용이 아니라 실제다.
- *fallback*: `step_style != continuous`이거나 헬퍼가 없는 pipe는 `pipe.generate`로
  fallback(인터럽트 없음, 로깅됨); `srs`/`srs_v2` 검증기 부재(예: CLIP 없음)는 경고를
  로그하고 휴리스틱으로 degrade; `checkpoint_legacy` 모드(`evaluate_checkpoints`)는 유지.
- *미구현*: 재구동 루프 vs 원본 `generate`의 GPU 수치 동일성은 mock pipe로 오프라인
  검증(루프 구조 + early exit + x0 노출)되었으나 실제 체크포인트에서는 **아직** 아님;
  검증 지표는 점검마다 preview를 디코딩(추가 비용, opt-in).

## 남은 한계 (Phase 5 → 향후)

- 채널 조건 **토큰**은 cfg에 부착되지만 여전히 frozen SGD-JSCC denoiser가 소비하지
  않음(FiLM / cross-attention / posterior guidance는 재학습된 조건 인식 denoiser가
  필요). 조건화는 received-latent init + reliability로 스케일된 guidance/steps를
  통해 작동한다.
- 학습된 consistency/distilled student 없음; few-step은 결정론적 근사다.
- 샘플러 내부 인터럽트는 **연속** 샘플러만 커버; discrete는 fallback.
- Fading `signal_scale`/step-matching은 AWGN 공식을 재사용(fading에서는 근사).

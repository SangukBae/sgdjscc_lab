> [← 문서 색인](./README.md)

# PSSS/SKEM 준비 상태 — variable-length SKEM selector + SKIM/SFA vs SKEM/DSA 비교

이 문서는 `docs/lgvsc_1c_reproduction_readiness.md`가 명시했던 미완료 항목 —
**"keyframe 선택(SKIM vs SKEM)은 네 모드 전부 동일하다"** — 을 메운 작업을
정리한다. `docs/etri_strategy.md` "후속 딥러닝 4단계" 1단계의 연장(1C 다음
단계)이며, 이 저장소에서는 **PSSS/SKEM 단계**로 부른다.

**핵심 경고(반복): 이 작업으로도 LGVSC의 SKEM/PSSS를 faithful하게 재현한 것이
아니다.** 논문은 InternVL2-8B 기반 captioning + PSSS 프롬프트의 정확한
하이퍼파라미터·후처리를 공개하지 않았다. 이 저장소가 완성한 것은 (1) 논문
Eq.1-2 그대로의 **PSSS 수식**(yes/no 최종 텍스트 비교가 아니라 실제 다음-토큰
확률로 계산), (2) 그 PSSS로 keyframe을 자동회귀적으로 고르는 **SKEM selector**
(가변 길이 segment 생성), (3) 기존 fixed selector(SKIM에 가까운 근사)와
비교하는 batch driver/summary 확장이다. mock/proxy PSSS 백엔드는 구조 검증용일
뿐 실제 LGVSC PSSS 성능의 근거가 아니며, real 백엔드도 캡션 모델·MLLM 선택이
논문과 다를 수 있어 수치를 논문과 직접 비교할 수 없다.

> **코드 리뷰 반영(2026-07 후속):** 최초 구현에 대한 리뷰에서 High 1건 +
> Medium 4건 + Low 1건이 지적됐다 — 전부 실제로 수정하고 회귀 테스트로
> 확인했다. 자세한 내용은 아래 "코드 리뷰에서 발견·수정된 문제" 참조. 이
> 문서의 나머지 서술은 수정 이후 상태를 기준으로 한다.

## 무엇이 새로 생겼는가

| 구성 요소 | 파일 | 역할 |
|---|---|---|
| PSSS 스코어러 | `src/sgdjscc_lab/video/psss.py` | `PsssBackend` 인터페이스 + `mock`/`proxy`/`real` 3종 구현. `S_rel = P("No") - P("Yes")`, `S_abs = P("Yes")`를 실제 모델 다음-토큰 확률로 계산(real), 또는 명확히 태그된 근사(proxy/mock)로 계산 |
| SKEM selector | `src/sgdjscc_lab/video/skem_selector.py` | `PsssKeyframeSelector` — PSSS 임계값(`eta_th`, 기본 0.35) 기반 자동회귀 keyframe 삽입. 기존 `KeyframeExtractor`와 동일한 `.extract(frames) -> Dict` 인터페이스를 구현해 `TemporalPipeline`에 그대로 꽂힌다 |
| Selector factory | `src/sgdjscc_lab/video/keyframe_extractor.py::build_keyframe_extractor` / `build_caption_fn` | `keyframe.selector: fixed`(기본, 기존 동작 불변) \| `fixed_interval`(논문 문자 그대로의 SKIM, `FixedIntervalKeyframeSelector`) \| `psss`(SKEM)로 선택. 캡션 소스(`captions_file`/`model`/`mock`)도 함께 선택 |
| Segment 메타데이터 | `src/sgdjscc_lab/video/segment.py::SegmentRecord.keyframe_selection` | segment별 selector/backend/threshold/PSSS score(raw_logits/evidence 포함)/reason 저장. **`fixed`(기존 `KeyframeExtractor`, scene-change 기반) selector에서만** 이 키 자체가 `to_dict()` 출력에서 생략됨(기존 출력과 스키마까지 100% 동일) — `fixed_interval`도 `structure["selector"]`를 채우므로 `keyframe_selection`이 실제로 붙는다(`backend_kind: "not_applicable"`, `psss_score`/`reason`은 `None` — PSSS 근거가 없다는 뜻을 명시하는 provenance이지, 생략 대상이 아니다). |
| 비교 config 4종 | `configs/etri_lgvsc_1c_{skim_sfa_fixed,skem_dsa_psss,skem_dsa_mock_psss,skem_dsa_proxy_psss}.yaml` | 아래 "config 4종" 참조 |
| batch driver 확장 | `scripts/batch_lgvsc_1c_reproduce.py` | 새 4개 모드 등록 + `selector_backend`/`psss_backend_kind`/segment 길이 통계/PSSS score 통계/`n_start_only`·`n_bidirectional`·`n_fallback` segment 수/worker model_id 등 summary 컬럼 확장 + `build_aggregate_comparison()`(SKIM/SFA vs SKEM/DSA per-video + MEAN 표) |

## 코드 리뷰에서 발견·수정된 문제 (2026-07 후속)

| 심각도 | 문제 | 수정 |
|---|---|---|
| High | `MllmTokenProbPsssBackend`가 입력 텐서를 항상 CPU에 생성해, `device: cuda:0` config(`skem_dsa_psss.yaml` 등)에서 모델(GPU)과 입력(CPU)의 device mismatch로 실제 GPU 실행이 실패했다 | `_model_device()`를 추가해 모델의 input embedding layer(`get_input_embeddings().parameters()`, accelerate/device_map 모델도 우선 고려) → 일반 파라미터 → `self.device` 순으로 실제 device를 조회하고, `_forward_logits()`가 그 device로 입력 텐서를 생성하도록 수정. 실제 CUDA 디바이스에서의 회귀 테스트(`tests/test_psss.py::TestMllmTokenProbPsssBackendDevicePlacement::test_real_cuda_model_does_not_hit_a_device_mismatch`, `torch.cuda.is_available()`일 때만 실행, 이 세션 환경에서 실제로 실행·통과 확인)로 검증 |
| Medium | Yes/No continuation을 `tokenizer.encode(text)`로 그대로 토크나이즈해, 일부 실제 tokenizer가 continuation에도 BOS/EOS를 붙이면 `P("Yes")`가 아니라 `P(BOS, "Yes", EOS)`를 계산하는 문제 | `_encode(text, add_special_tokens=False)`를 continuation(`_variant_mass`)에서 명시적으로 사용하도록 수정(프롬프트 자체는 기본값`True` 유지, 특수 kwarg를 지원 안 하는 최소 fake tokenizer는 `TypeError` fallback으로 하위호환). BOS/EOS를 실제로 삽입하는 `_SpecialTokenAwareTokenizer` fake로 회귀 테스트 추가(`tests/test_psss.py::TestMllmTokenProbPsssBackendSpecialTokens`) — 이전 fake들은 애초에 special token을 넣지 않아 이 버그를 재현하지 못했었다 |
| Medium | `skim_sfa_fixed`가 기존 `KeyframeExtractor`(scene-change 반응)를 썼기 때문에, 논문의 "동일 길이 구간" SKIM 기준선이 아니었다 | 순수 `FixedIntervalKeyframeSelector`(`video/keyframe_extractor.py`, scene-change 신호 전혀 없음, `l_n = interval` 그대로)를 신규 추가하고 `skim_sfa_fixed.yaml`이 `keyframe.selector: fixed_interval` + `interval: 12`(= `skem_dsa_psss.yaml`의 `max_segment_length`와 동일)를 쓰도록 변경 — 두 비교선의 keyframe 개수/CBR이 같은 상한을 공유하게 됨. 회귀 테스트: `tests/test_skem_selector.py::TestFixedIntervalKeyframeSelector`(5개) |
| Medium | `PsssScoreResult`의 `raw_logits`/`evidence`/`model_id`/`proxy_of`/`notes`가 `skem_selector.py`의 `psss_scores` 레코드에 복사되지 않아 `keyframes.json`/`segments.json`에 남지 않음 | `record = score.to_dict()`로 `PsssScoreResult` 전체를 베이스로 삼고 selector 전용 필드(`index`/`compared_to_keyframe`/`threshold`)만 덧붙이도록 수정 — 이제 실제 산출물에 raw logits/evidence/notes가 전부 남는다. 회귀 테스트로 `segments.json`의 `keyframe_selection.psss_score`에 이 키들이 실제로 존재하는지 확인 |
| Medium | `summary_metrics.csv`의 `psss_score_mean/min/max`가 keyframe을 실제로 발생시킨 score만 집계해(`segments.json`의 `keyframe_selection`에는 트리거 score만 붙음) 전체 PSSS 분포처럼 오인될 수 있었다 | `keyframes.json`의 **전체** `psss_scores`(continue_segment 포함)를 읽어 population 통계(`psss_score_mean/min/max/n`)로 삼고, 기존 트리거-only 값은 `trigger_psss_score_mean/min/max`로 이름을 분리했다. CPU 스모크 재실행에서 `skem_dsa_proxy_psss`처럼 threshold를 한 번도 못 넘은 실행도 이제 의미 있는 `psss_score_mean`(예: 약 -0.90)을 보고한다 — 예전에는 트리거가 0건이라 빈 값이었다 |
| Low | PSSS가 꺼진(기존 `fixed` selector) 기본 경로에서도 `segments.json`에 `keyframe_selection: null`이 항상 추가돼 "결과 불변"이 엄밀하지 않았다 | `SegmentRecord.to_dict()`가 `keyframe_selection`이 `None`일 때 그 키 자체를 아예 생략하도록 수정 — 기존 fixed selector 실행의 `segments.json` 키 집합이 완전히 예전과 동일해졌다(값이 같다는 정도가 아니라 스키마 자체가 동일) |

전체 회귀: `conda run -n ptest python -m pytest tests/ -q` → **1090 passed, 0 failed**.

### 2차 코드 리뷰 반영 — CBR 매칭 + 문서 정정 (2026-07 후속)

1차 리뷰 수정 이후 재검토에서 Medium 1건 + Low 1건이 추가로 지적됐다 —
"blocking defect는 없다"는 평가였지만 둘 다 실제로 수정했다.

| 심각도 | 문제 | 수정 |
|---|---|---|
| Medium | `fixed_interval: 12`(SKIM)과 `max_segment_length: 12`(SKEM)는 **최대 길이만** 같다 — SKEM은 PSSS 판단에 따라 12프레임보다 훨씬 이전에도 keyframe을 추가하므로, 실제 keyframe 개수/CBR은 두 비교선이 다를 수 있다. | 최초 보정안은 `interval=ceil(n_frames/n_keyframes)`를 사용했지만, 10프레임/6 keyframe처럼 정수 interval로 표현할 수 없는 조합이 존재하므로 폐기했다. 현재 `--keyframe-count-match-from <mode>`는 완료된 source SKEM의 `keyframes.json`과 `temporal_metrics.csv`가 같은 clip 길이를 보고하는지 확인하고, target 길이도 같은 경우에만 `FixedCountKeyframeSelector`로 정확히 K개를 균등 배치한다. source keyframe/metrics/generated-config fingerprint를 기록하고 실행 후 source 불변성, target 프레임 수, 실제 keyframe 수를 다시 검증하며 불일치는 job 실패로 처리한다. summary의 `keyframe_match_status`/`requested_keyframes`/`actual_fixed_keyframes`/`keyframe_count_delta`가 이 검증을 담는다. 동일 keyframe 수는 동일 실제 CBR의 충분조건이 아니다. per-run channel-symbol accounting이 없거나 `proxy_fraction > 0`이면 `cbr_match_status=count_only`, exact accounting에서 symbol 수까지 같을 때만 `verified`, symbol 수가 다르면 `mismatch`로 기록한다. |
| Low | 문서가 "`fixed`/`fixed_interval` 둘 다 `keyframe_selection`을 생략한다"고 잘못 서술했다 — 실제로는 `FixedIntervalKeyframeSelector`도 `structure["selector"]`를 채우므로 `segment.py::_keyframe_selection_summary`가 non-None을 반환해 `keyframe_selection`이 실제로는 붙는다(`backend_kind: "not_applicable"`, `psss_score`/`reason`은 `None`) | 위 "무엇이 새로 생겼는가" 표의 서술을 "**`fixed`(기존 `KeyframeExtractor`)에서만** 생략, `fixed_interval`은 selector provenance를 기록"으로 정정. 회귀 테스트: `tests/test_skem_selector.py::test_fixed_interval_selector_segments_DO_have_keyframe_selection` |

이 후속 수정의 집중 회귀:
`conda run -n ptest python -m pytest tests/test_skem_selector.py tests/test_batch_lgvsc_1c_reproduce.py -q`
→ **90 passed, 0 failed**.
전체 회귀: `conda run -n ptest python -m pytest tests/ -q`
→ **1117 passed, 0 failed, 3 warnings**(기존 Transformer nested-tensor 경고).

## PSSS 계산 방식 (`video/psss.py`)

논문 Eq.1-2:

```
S_abs = P("Yes" | Info A, Info B, Semantic Focus)          ∈ [0, 1)
S_rel = P("No"  | ...) - P("Yes" | ...)                     ∈ (-1, 1)
```

`MllmTokenProbPsssBackend`(`backend_kind="real"`)는 이를 **문자열 비교가 아니라
실제 모델의 다음-토큰 로짓**으로 계산한다:

1. 논문의 프롬프트 템플릿("Info A, Info B. Determine whether they are similar
   from the perspective of Semantic Focus, use yes or no to answer.")을
   구성한다.
2. 프롬프트를 토크나이즈하고, 모델의 마지막 위치 로짓에 `log_softmax`를 적용한다.
3. `"Yes"`/`"No"`의 여러 표면형(대소문자·선행 공백: `"Yes"`, `" Yes"`, `"yes"`,
   `" yes"`, `"YES"`/`"No"`, `" No"`, `"no"`, `" no"`, `"NO"`)에 대해, 토크나이저가
   **몇 토큰으로 쪼개든** teacher-forcing으로 결합 시퀀스 확률을 정확히 계산한다
   (1토큰이면 즉시, 다토큰이면 각 단계를 순차적으로 forward). 동일 토큰 ID
   시퀀스로 귀결되는 표면형은 중복 계산하지 않는다.
4. `P(Yes)`/`P(No)`는 각 진영의 표면형 확률 합. `S_abs`/`S_rel`은 그 값으로
   Eq.1-2 그대로 계산하고, `p_yes_norm = P(Yes)/(P(Yes)+P(No))`도 별도로 저장한다
   (정규화된 값과 원시 값을 둘 다 기록).
5. raw logits(표면형별 첫 토큰의 로짓), 정규화 확률, 최종 `s_abs`/`s_rel`,
   사용된 프롬프트, 표면형별 토큰 ID/로그확률을 전부 `PsssScoreResult.raw_logits`
   / `.evidence`에 기록한다.

**Unavailable 처리:** `transformers`가 없거나, `model_id` 가중치를 로드하지
못하거나, 주입된 model/tokenizer가 다음-토큰 로짓을 못 주면(호출 자체가
실패하거나 `.logits`가 없는 출력을 반환하면) `PsssBackendUnavailableError`를
던진다 — mock 결과로 조용히 대체하지 않는다(`tests/test_psss.py::
TestMllmTokenProbPsssBackendUnavailable`로 검증).

## mock / proxy / real 구분

| backend | `backend_kind` | 실제로 하는 일 | 실제 PSSS인가 |
|---|---|---|---|
| `MockPsssBackend` | `mock` | 두 캡션의 어휘 Jaccard 겹침만 계산 — 결정적, 모델 없음 | ❌ 테스트/구조 검증 전용 |
| `ClipTextProxyPsssBackend` | `proxy` | 이 저장소의 기존 CLIP 텍스트 인코더로 두 캡션의 코사인 유사도를 계산해 같은 스키마로 사상 | ❌ yes/no 프롬프트도, 토큰 확률도 없음 — "이미 사용 가능한 모델"을 재사용한 근사일 뿐 |
| `MllmTokenProbPsssBackend` | `real` | 위 "PSSS 계산 방식" 그대로 — 실제 MLLM/causal-LM의 P(Yes)/P(No) | ✅ 논문 Eq.1-2를 그대로 구현 |

세 백엔드 모두 `PsssScoreResult.backend_kind`/`.notes`에 자신의 정체를 명시하며,
`proxy`는 `.proxy_of = "clip_text_similarity"`까지 못박는다 — 어떤 산출물에서도
proxy/mock 결과를 real PSSS로 착각할 수 없다.

## Fixed segment vs Variable-length segment

- **SKIM/SFA 모드(`skim_sfa_fixed`)**: `keyframe.selector: fixed_interval` —
  `FixedIntervalKeyframeSelector`(scene-change 신호 전혀 없음, 순수
  `interval`마다 keyframe, 논문 SKIM 정의 그대로). 이전 4개 1C 모드
  (`wan_skim_sfa` 등)가 쓰는 `keyframe.selector: fixed`(= 기존
  `KeyframeExtractor`, scene-change 거리 + `keyframe.max_gop` cap)와는
  **다른** selector다 — `fixed`는 여전히 "nearest reproducible" 근사일
  뿐(scene-change에도 반응) 논문의 문자 그대로의 SKIM이 아니라는
  `docs/lgvsc_1c_reproduction_readiness.md`의 기존 경고가 유효하므로,
  `skim_sfa_fixed`만 이 새 selector로 바꿨다.
- **SKEM/DSA 모드(`skem_dsa_psss` 등)**: `keyframe.selector: psss` —
  `PsssKeyframeSelector`가 K1=frame 0부터 자동회귀로 최신 keyframe과의 `S_rel`이
  `threshold`를 **초과(strict >)**할 때만 새 keyframe을 삽입한다. 이 저장소가
  논문에 없는 부분을 두 가지 추가했다(모두 정확히 이렇게 태그됨, PSSS 결정과
  섞이지 않음):
  - `min_segment_length`(기본 1): 최신 keyframe으로부터 이 프레임 수 이내는
    PSSS 평가 자체를 건너뛴다(실제 MLLM 호출 비용 절감 + 과도하게 짧은 segment
    방지).
  - `max_segment_length`(기본 12, `keyframe.max_gop`와 동일 철학): 이 길이에
    도달하면 PSSS 점수와 무관하게 강제로 keyframe을 삽입한다(CBR/지연 안전판).

  이 두 경계 안에서 segment 길이는 **실제로 가변적**이다 — CPU mock 스모크
  테스트에서 같은 `--max-frames 14` 조건으로 `01_person_walk`은 [1,3] 프레임
  segment 7개, `02_car_pass`는 [4,6] 프레임 segment 3개가 나왔다(둘 다 fixed
  selector로는 [2,12] 동일 패턴). 아래 "직접 검증 결과" 참조.

`SegmentGenerationRequest`/`generate_segment()`(1A) 계약은 애초에 GOP 길이에
대해 아무 가정도 하지 않으므로(각 GOP의 `target_indices`/`segment_length`는
호출마다 다를 수 있음), variable-length segment를 흘려보내기 위해
`video_generator.py`/`lgvsc_generate_worker.py`를 전혀 수정하지 않았다 — 이번
작업으로 확인한 것은 "이미 가변 길이를 지원하던 계약에 실제로 가변 길이
selector를 연결"이다.

## Wan start/end checkpoint 연결 방식 (변경 없음, 재확인만 함)

1B가 이미 구현한 대로: `scripts/lgvsc_generate_worker.py::run_wan_backend`가
segment마다 `end_keyframe_image` 존재 여부로 체크포인트를 자동 선택한다 —
있으면 `Wan2.1-FLF2V-14B-720P`(bidirectional), 없으면 `Wan2.1-I2V-14B-480P`
(start-only). PSSS selector가 만든 segment도 이 로직을 그대로 통과한다(코드
변경 없음). **마지막 open GOP**(다음 keyframe이 없는 마지막 segment)는 여전히
`end_keyframe_image`가 없으므로 자동으로 start-only 체크포인트로 라우팅되고,
mock 경로에서는 `BidirectionalInterpolationGenerator(missing_end_policy=
"fallback_start_only")`가 `conditioning_mode="start_only"`로 명확히 강등해
기록한다(`generation.conditioning_mode`가 실제 사용된 조건화를 정직하게
반영 — 요청이 bidirectional이었다는 사실은 잃지 않고
`n_fallback_segments`로 batch summary에 집계됨). 회귀 테스트:
`tests/test_skem_selector.py::TestTemporalPipelineWithPsssSelector::
test_variable_length_generate_branch_and_last_gop_fallback`.

## Config 4종

| Config | selector | PSSS backend | decoder | 실행 비용 |
|---|---|---|---|---|
| `etri_lgvsc_1c_skim_sfa_fixed.yaml` | `fixed_interval`(interval=12, 논문 문자 그대로의 SKIM) | 해당 없음 | Wan start-only(실제 GPU, 1B 검증된 worker 블록 그대로) | 실제 GPU — `wan_skim_sfa`와 동일 비용 |
| `etri_lgvsc_1c_skem_dsa_psss.yaml` | `psss` | `real`(MLLM, `model_id` 사용자 지정 필요) | Wan bidirectional(실제 GPU, 1B 검증된 worker 블록 그대로) | 실제 GPU + 실제 MLLM 가중치 — 가장 무거움 |
| `etri_lgvsc_1c_skem_dsa_mock_psss.yaml` | `psss` | `mock` | mock bidirectional interpolation | CPU, 다운로드 없음 — 항상 실행 가능 |
| `etri_lgvsc_1c_skem_dsa_proxy_psss.yaml` | `psss` | `proxy`(CLIP) | mock bidirectional interpolation | CPU, CLIP 가중치만 (경량) |

`skim_sfa_fixed`/`skem_dsa_psss`가 "최소 두 실험 모드" 요구사항이고, 나머지
둘은 진단용(mock/proxy)이다. 각 config 헤더에 정확히 이 표와 같은 구분을
반복해 적어 두었다 — config만 읽어도 실제/proxy/mock을 헷갈릴 수 없다.

## Batch driver + summary 확장

`scripts/batch_lgvsc_1c_reproduce.py`의 `MODES`가 8개로 늘었다(기존 4 +
신규 4). `collect_run_metrics()`가 (1) 생성된 per-video config를 다시 읽어
`selector_backend`/`psss_backend_kind`(`not_applicable` when 비-`psss`)를,
(2) `segments.json`의 `keyframe_selection`/`generation`을 집계해
`n_segments`/`segment_length_{min,max,mean,std}`/
`n_start_only_segments`/`n_bidirectional_segments`/`n_fallback_segments`/
`worker_model_id`를, (3) **`keyframes.json`의 전체 `psss_scores`**(PSSS가
평가한 모든 프레임 — keyframe을 실제로 발생시켰는지와 무관)를 읽어
population 통계 `psss_score_{mean,min,max,n}`을, 그리고 keyframe을 실제로
발생시킨 score만의 `trigger_psss_score_{mean,min,max}`을 **별도로** 채운다
(둘을 섞으면 편향 — 위 "코드 리뷰에서 발견·수정된 문제" 참조), (4) 실행
로그/결과 경로(`run_log_path`/`segments_json_path`/`keyframes_json_path`/
`recon_video_path`)를 채운다.
`build_aggregate_comparison(rows, modes_pair=("skim_sfa_fixed",
"skem_dsa_psss"))`가 영상별 + `MEAN` 행으로 두 모드를 나란히 놓은
`summary_aggregate_comparison.csv/.md/.json`을 만든다(영상이 없거나 두 모드
중 하나도 실행되지 않았으면 빈 리스트 — 파일을 쓰지 않는다).

**버그 수정(이번 작업 중 발견):** `--output-root`에 상대경로를 주면
생성 config의 절대경로 계산이 `_generated_configs/<mode>/` 기준으로 다시
해석되어 출력 경로가 중첩 이중화되는 문제가 있었다(`output_root` 자체가
상대경로였던 경우에만 발생 — 기본값은 항상 절대경로라 기존 4모드
문서화된 사용법에서는 드러나지 않았다). `main()`에서 `output_root =
Path(args.output_root).resolve()`로 고정해 근본 수정했다 — "생성 config는
영상별 절대 output 경로를 사용한다" 요구사항을 상대경로 CLI 입력에도
보장한다.

## 테스트

```bash
conda run -n ptest python -m pytest tests/test_psss.py tests/test_skem_selector.py \
    tests/test_batch_lgvsc_1c_reproduce.py -q
```

- `tests/test_psss.py`(30개): PSSS 수식(S_abs/S_rel), mock/proxy/real 각 백엔드,
  multi-token yes/no 처리(중복 표면형 이중 계산 방지 포함), model_id 실패·
  tokenizer 실패·logits 없는 출력 각각에 대한 `PsssBackendUnavailableError`,
  backend_kind 상호 구분, **continuation의 add_special_tokens=False 처리**
  (BOS/EOS를 실제로 붙이는 fake tokenizer로 회귀), **모델 device 배치**
  (embedding layer device 조회, 실제 CUDA 디바이스에서의 end-to-end 확인 —
  `torch.cuda.is_available()`일 때만 실행, 이 세션에서 실행·통과 확인).
- `tests/test_skem_selector.py`(34개): threshold 경계(`>` strict), 가변 길이
  segment, min/max segment length(경계 포함), 첫 프레임 항상 keyframe(PSSS 호출
  없음), 빈/1프레임/잘린 입력, **`FixedIntervalKeyframeSelector`**(순수
  고정 간격, scene-change 무관, 경계 케이스), `build_keyframe_extractor`/
  `build_caption_fn` factory(기본값 fixed 불변, `fixed_interval` 배선,
  에러 케이스), `TemporalPipeline` 전체 배선(segment의 `keyframe_selection`
  메타데이터에 **raw_logits/evidence/notes/model_id/proxy_of까지 포함되는지**,
  마지막 open GOP fallback), Rx-legal(`SegmentGenerationRequest`에 원본 프레임
  필드 없음 재확인), fixed selector 실행의 `segments.json`에
  `keyframe_selection` 키 자체가 없는지(스키마 불변) 확인.
- `tests/test_batch_lgvsc_1c_reproduce.py`(29개): 새 8-모드 config
  provenance(1B 검증 config와 worker 블록 일치, selector/psss.backend 값,
  `skim_sfa_fixed`가 `fixed_interval`을 쓰는지), **population-wide vs
  trigger-only PSSS score 통계가 실제로 다른 값을 내는지**, aggregate 비교
  표, dry-run/두 영상×두 모드 summary isolation.

`conda run -n ptest python -m pytest tests/ -q` → **1090 passed, 0 failed**
(회귀 없음).

## 직접 검증 결과 (이번 세션에서 실행)

### 1) CPU mock/proxy smoke (2개 영상, `--no-models`, 다운로드 없음 원칙 — proxy는 CLIP 최초 1회 다운로드)

```bash
conda run -n ptest python scripts/batch_lgvsc_1c_reproduce.py \
    --modes mock_baseline,skem_dsa_mock_psss,skem_dsa_proxy_psss \
    --videos 01_person_walk,02_car_pass \
    --output-root outputs/etri_video_eval/lgvsc_1c_reproduce_cpu_smoke \
    --max-frames 14 --no-models
```

→ 6/6 성공. 관측된 `summary_metrics.csv`(발췌):

| mode | video | selector_backend | psss_backend_kind | n_segments | segment_length min/max/mean | psss_score_mean (population) |
|---|---|---|---|---|---|---|
| mock_baseline | 01_person_walk | fixed | not_applicable | 2 | 2/12/7.0 | (해당 없음) |
| mock_baseline | 02_car_pass | fixed | not_applicable | 2 | 2/12/7.0 | (해당 없음) |
| skem_dsa_mock_psss | 01_person_walk | psss | mock | 7 | 1/3/2.0 | 0.562 (n=7) |
| skem_dsa_mock_psss | 02_car_pass | psss | mock | 3 | 4/6/4.67 | 0.320 (n=10) |
| skem_dsa_proxy_psss | 01_person_walk | psss | proxy | 2 | 2/12/7.0 | -0.901 (n=10) |
| skem_dsa_proxy_psss | 02_car_pass | psss | proxy | 2 | 2/12/7.0 | -0.907 (n=10) |

**같은 두 영상, 같은 `--max-frames 14`에서 fixed selector는 두 영상 모두
동일한 [2, 12] 분할을 냈지만, PSSS(mock) selector는 영상마다 실제로 다른
개수·길이의 segment를 만들었다(7개 vs 3개)** — variable-length 동작이 구조가
아니라 실행 결과로 확인됨. `skem_dsa_proxy_psss`가 fixed와 같은 분할 패턴을
보인 이유는, population 통계 수정(위 "코드 리뷰에서 발견·수정된 문제") 이후
정량적으로 확인됐다 — CLIP 텍스트 인코더가 이 저장소의 frame-statistics
placeholder 캡션("frame N stats ...")들을 일관되게 "매우 유사"(`s_rel` 평균
약 -0.90, 전부 `threshold=0.35`에 한참 못 미침)로 판단해 `max_segment_length`
강제 외에는 keyframe을 전혀 추가하지 않았기 때문이다. proxy 백엔드가 real
PSSS의 성능 대리 지표가 될 수 없다는 것을 보여주는 사례이지 버그가 아니다
(진짜 캡션이 있으면 CLIP 텍스트 유사도도 더 잘 갈릴 것으로 기대되지만, 이
또한 검증되지 않았다). 이 수정 전에는 트리거가 0건이라 `psss_score_mean`이
빈 값으로 나와 이 원인을 전혀 알 수 없었다는 점도 population 통계 수정의
실질적 효과다.

### 2) 실제 CUDA 디바이스에서의 PSSS device-placement 회귀 (fake model, 가중치 다운로드 없음)

```bash
conda run -n ptest python -m pytest tests/test_psss.py -k cuda -q
```

→ 1 passed (이 세션의 실제 GPU에서 실행됨, skip되지 않음). 모델을
`cuda:0`에 두고 `MllmTokenProbPsssBackend.score()`를 호출해 device mismatch
없이 완료되는지 확인 — High 심각도로 지적된 버그의 실제 CUDA 환경에서의
직접 재현·수정 확인이다. 실제 MLLM 가중치는 쓰지 않았으므로 "real PSSS의
실제 GPU 전체 실행"과는 다르다(아래 "아직 하지 않은 것" 참조).

### 3) 실제 GPU smoke — `skim_sfa_fixed`(이제 `fixed_interval` selector)를 실제 Wan 14B로 실행

```bash
conda run -n ptest python scripts/batch_lgvsc_1c_reproduce.py \
    --modes skim_sfa_fixed --videos 01_person_walk,02_car_pass \
    --max-frames 14 --no-models
```

이 머신에 이미 캐시된 `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`(1B에서 다운로드)를
사용해 두 영상 모두 실제 GPU(RTX 4080)에서 완주했다. `01_person_walk`의
`segments.json`에서 확인:

```
segment 0: conditioning_mode=start_only backend=external_segment_worker:wan:Wan-AI/Wan2.1-I2V-14B-480P-Diffusers mock=false
segment 1: conditioning_mode=start_only backend=external_segment_worker:wan:Wan-AI/Wan2.1-I2V-14B-480P-Diffusers mock=false
```

`recon.mp4`/`temporal_metrics.csv`/`n_generate=12` 모두 정상 산출 — 새 config가
1B에서 검증된 것과 동일하게 실제 GPU에서 동작함을 재확인했다(코드 재사용이라
당연하지만, config 헤더의 주장을 실행으로 검증했다는 의미).

### 아직 하지 않은 것 (실제 GPU/실제 MLLM 필요)

- `skem_dsa_psss.yaml`(backend: real)의 실제 실행 — 실제 MLLM(`model_id`
  placeholder: `Qwen/Qwen2.5-1.5B-Instruct`, 검증되지 않은 선택) 가중치가 이
  세션에서 다운로드되지 않았다. `keyframe.psss.real.model_id`를 실제 사용
  가능한 causal-LM/VLM으로 바꾸고 직접 실행해야 한다.
- 10개 영상 전체 × 8모드 배치, `summary_aggregate_comparison.*`을 실제
  `skim_sfa_fixed`/`skem_dsa_psss` 둘 다의 실제 GPU 결과로 채우는 것.
- PSSS `real` 백엔드가 실제로 논문이 보고한 것과 유사한 keyframe 선택
  패턴/CBR을 내는지에 대한 정량적 검증.

## 사용자가 실행할 다음 명령어

```bash
cd sgdjscc_lab

# (1) real MLLM PSSS + real Wan bidirectional (가장 무거움, 최초 준비 필요:
#     configs/experiments/lgvsc_1c/etri_lgvsc_1c_skem_dsa_psss.yaml의 keyframe.psss.real.model_id를
#     실제 보유한 MLLM으로 확정)
conda run -n ptest python scripts/batch_lgvsc_1c_reproduce.py \
    --modes skim_sfa_fixed,skem_dsa_psss --videos 01_person_walk \
    --max-frames 14 --device cuda:0

# (2) 10영상 전체 비교 (오래 걸림)
conda run -n ptest python scripts/batch_lgvsc_1c_reproduce.py \
    --modes skim_sfa_fixed,skem_dsa_psss --device cuda:0 --continue-on-error

# (3) 결과가 나온 뒤 비교표만 재생성
conda run -n ptest python scripts/batch_lgvsc_1c_reproduce.py --summary-only

# (4) keyframe-count-matched 비교 — SKEM을 먼저 돌린 뒤, 그 실제 keyframe
#     개수와 clip 길이에 맞춰 SKIM을 FixedCountKeyframeSelector로 다시 실행.
#     동일 channel-symbol CBR까지 자동 보장하는 것은 아니다. summary에서
#     keyframe_match_status와 cbr_accounting_kind를 먼저 확인하고,
#     cbr_match_status=count_only/verified/mismatch를 해석해야 한다.
conda run -n ptest python scripts/batch_lgvsc_1c_reproduce.py \
    --modes skem_dsa_psss --device cuda:0 --continue-on-error
conda run -n ptest python scripts/batch_lgvsc_1c_reproduce.py \
    --modes skim_sfa_fixed --device cuda:0 --continue-on-error \
    --keyframe-count-match-from skem_dsa_psss
```

`outputs/etri_video_eval/lgvsc_1c_reproduce/summary_aggregate_comparison.csv`의
`skem_dsa_psss.segment_length_std` > `skim_sfa_fixed.segment_length_std`(둘 다
0에 가깝지 않고, PSSS 쪽이 실제로 더 가변적)이면 "SKEM이 SKIM보다 의미
경계에 더 잘 맞춰 segment를 나눈다"는 구조적 신호로 볼 수 있다 — 이것이
LGVSC의 정성적 주장을 재현하는 것과 같은 뜻은 아니다(논문의 정량 결과와
직접 비교 불가, 위 "무엇이 새로 생겼는가" 서두 참조).

## 남은 제한사항 (숨기지 않고 명시)

1. **논문의 InternVL2-8B 캡셔너를 재현하지 않았다** — `caption_source: model`은
   이 저장소가 이미 쓰던 BLIP2/Qwen 캡셔너를 재사용한다. 캡셔너가 다르면
   PSSS 입력 텍스트가 달라지고, 그 결과 keyframe 선택도 논문과 달라진다.
2. **side-info 인코더는 여전히 없다** — 1B/1C 문서의 기존 한계 그대로,
   `side_infos`는 여전히 conditioning에 쓰이지 않는다.
3. **학습된 DSA adapter는 없다** — Wan은 segment 길이와 무관하게 동일 아키텍처를
   재사용한다(체크포인트 자동 선택은 "적절한 사전학습 체크포인트 선택"이지
   "가변 차원 latent를 위한 학습된 adapter"가 아니다).
4. **`eta_th` 자체의 CBR 캘리브레이션은 하지 않았다** — 논문은 `eta_th`를
   목표 CBR에 맞춰 보정한다고 설명하지만, 이 저장소는 `eta_th=0.35`(논문
   실험값)를 그대로 기본값으로 쓸 뿐 실제 이 데이터셋/모델 조합에서 CBR을
   계산해 `eta_th` 자체를 보정하지 않았다. (SKIM 쪽 keyframe 개수를 SKEM의
   실제 keyframe 수에 맞추는 `--keyframe-count-match-from`은 별개로
   구현했다 — 위 "2차 코드 리뷰 반영" 참조. 이는 keyframe 개수 매칭이고, 위
   한계는 SKEM 자신의 `eta_th`→CBR 보정이 없다는 뜻으로 여전히 유효하다.)
   또한 이 옵션은 인프라일 뿐 — 실제로 count-matched 상태에서
   `skim_sfa_fixed` vs `skem_dsa_psss`를 실행해 품질을 비교한 적은
   없다(사용자 몫).
5. **mock/proxy PSSS의 segment 패턴은 실제 의미 이해에 근거하지 않는다** — 위
   "직접 검증 결과"에서 보였듯 mock은 어휘 중복, proxy는 CLIP 임베딩에
   의존할 뿐이다. real 백엔드로 실제 실행하기 전에는 "의미 있는 keyframe
   선택"이라는 주장을 할 수 없다.
6. **`real` PSSS 백엔드는 실제 MLLM 가중치로는 이 세션에서 실행되지
   않았다** — CUDA device-placement 버그는 실제 GPU에서 fake model로
   재현·수정 확인했지만(위 "직접 검증 결과" 2번), 실제 MLLM 가중치를 로드해
   실제 keyframe 선택 결과를 관찰하는 것은 여전히 사용자 몫이다.

## 관련 문서

- [lgvsc_1c_reproduction_readiness.md](./lgvsc_1c_reproduction_readiness.md) — 이 작업 이전의 4-모드 1C 재현선(이 문서가 그 다음 단계)
- [lgvsc_1b_worker_readiness.md](./lgvsc_1b_worker_readiness.md) — Wan/SVD 실제 GPU 검증, 체크포인트 자동 선택
- [etri_strategy.md](./etri_strategy.md) — "후속 딥러닝 4단계" 전체 맥락
- [video_extension_lgvsc.md](./video_extension_lgvsc.md) — LGVSC 재현선/ETRI 개선선 설계 원안

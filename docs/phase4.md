> [← 문서 색인](./README.md)

# Phase 4 — 계획 & 구현 현황

- [마스터 스위치](#master-switch)
- [Phase 4 계획](#phase-4-plan)
- [Phase 4 구현 현황](#phase-4-implementation-status)

---

<a id="master-switch"></a>

## 마스터 스위치

모든 Phase 4 기능은 **기본값 off**다. 최상위 플래그 `use_phase4` 하나로 phase
전체를 제어한다:

```yaml
# configs/eval/default.yaml (또는 composed config)
use_phase4: false   # 기본값 — 모든 Phase 4-A/B 기능 비활성화
```

**규칙**: `use_phase4: false`이면, 아래 나열된 개별 기능 플래그
(`use_packet_eval`, `use_adaptive_guidance`, `use_packet_regeneration`)는
config에서 명시적으로 `true`로 설정되어 있어도 런타임에 무시된다. Phase 4-B
(`evaluate_video.py`)는 에러로 종료된다.

### Phase 4만 활성화

```yaml
use_phase4: true
use_phase5: false   # Phase 5는 꺼둠

use_packet_eval: true
use_adaptive_guidance: true
use_packet_regeneration: false
```

### Phase 4 + Phase 5 활성화

```yaml
use_phase4: true
use_phase5: true
```

Phase 5 전용 플래그는 [phase5.md](./phase5.md)를 참조한다.

### 헬퍼 함수

모든 런타임 체크는 `sgdjscc_lab.phase_gates.effective_flag`를 거친다:

```python
from sgdjscc_lab.phase_gates import effective_flag, phase4_enabled

# use_phase4가 false이면 raw 플래그 값과 무관하게 False를 반환한다.
use_packet = effective_flag(cfg, "use_packet_eval", phase=4)
```

---

<a id="phase-4-plan"></a>

# Phase 4 계획

Phase 4는 실제 프레임별 복원에는 현재 SGD-JSCC forward 경로를 그대로 유지하면서,
`sgdjscc_lab`을 단일 이미지 평가에서 `키프레임 지향 시맨틱 전송 프레임워크`로 확장한다.

Phase 4의 지도 원칙은 다음과 같다:

`SGD-JSCC 코어를 먼저 재학습하지 않는다. 기존 이미지 경로를 중심으로 시맨틱 패킷,
시간적 평가, 적응형 제어, 비디오 파이프라인을 구축한다.`

## Phase 4 목표

현재의 이미지 전용 프로토타입을 `RA-SGDJSCC-lite`로 변환한다:

1. 각 프레임에 대한 시맨틱 패킷 추출 및 캐싱
2. 키프레임 / 인터프레임 분할
3. 시간적 시맨틱 재사용 및 델타 전송 시뮬레이션
4. SNR 인식 적응형 가이드 제어
5. 더 강한 검증기 및 regeneration 로직
6. 시간적 지표 및 키프레임 레벨 리포팅

## Phase 4 구현 순서

### Phase 4-A: 신뢰도 우선 이미지 확장

이것이 첫 번째 구현 마일스톤이며, 비디오 전용 코드를 작성하기 전에 완료해야 한다.

범위:

1. 현재 이미지 파이프라인 위에 적응형 가이드 컨트롤러 추가
2. SRS를 순수 CLIP/객체 복합 지표에서 패킷 인식 검증기로 업그레이드
3. 실제 패킷 코딩 이전에도 시맨틱 패킷을 JSON 메타데이터로 저장
4. 탐지된 실패 양상으로 키(key)를 둔 구조화된 regeneration 정책 추가

주요 참고:

- `paper/FAST-GSC: Fast and Adaptive Semantic Transmission for Generative Semantic Communication/FAST_GSC.tex`
- 현재 `sgdjscc_lab` 모듈:
  - `src/sgdjscc_lab/evaluators/semantic_reliability.py`
  - `src/sgdjscc_lab/pipelines/regeneration_loop.py`
  - `src/sgdjscc_lab/pipelines/eval_pipeline.py`

계획 파일:

```text
src/sgdjscc_lab/
├── controllers/
│   ├── adaptive_guidance_controller.py
│   ├── snr_guidance_policy.py
│   └── regeneration_policy.py
├── guidance/
│   ├── semantic_packet_extractor.py
│   ├── object_extractor.py
│   ├── relation_extractor.py
│   └── importance_estimator.py
├── evaluators/
│   ├── semantic_packet_matcher.py
│   ├── relation_consistency.py
│   └── attribute_consistency.py
└── utils/
    └── packet_io.py
```

구현 단계:

1. `semantic_packet_extractor.py`
   - 다음으로부터 통합 시맨틱 패킷을 구성:
     - 캡션
     - 객체 목록
     - 장면 라벨
     - 관계(relation) triplet
     - 속성(attribute)
     - 엣지 요약
     - 세그멘테이션 요약
     - depth 요약
   - Phase 4-A는 아직 이 패킷을 채널로 전송하지 않는다
   - 분석을 위해 각 복원 이미지 옆에 `packet.json`을 직렬화한다
2. `adaptive_guidance_controller.py`
   - 현재 파이프라인 config/런타임 상태에서 추정 SNR을 읽음
   - 출력:
     - `guidance_scale`
     - `controlnet_scale`
     - `diffusion_step`
     - `use_text`
     - 선택적 `skip_diffusion`
   - 초기 정책:
     - `SNR <= 0 dB`: 강한 텍스트 + 엣지 가이드, 최대 확산 스텝
     - `0 < SNR < 8 dB`: 중간 가이드, 엣지 우선 정책
     - `SNR >= 8 dB`: 약한 가이드, 선택적 무조건(unconditional) 또는 skip 경로
3. `semantic_packet_matcher.py`
   - 원본 패킷 vs 복원 패킷 비교
   - 다음을 명시적으로 카운트:
     - 누락 객체
     - 추가 객체
     - 관계 오류
     - 속성 오류
     - 장면 불일치
4. `regeneration_policy.py`
   - 현재의 스칼라 재시도 정책을 오류 유형 인식 재시도로 교체
   - 예시 정책:
     - 객체 누락: 텍스트 가이드와 객체 우선 가이드 강화
     - 할루시네이션: 텍스트 CFG 감소, 엣지 가이드는 더 강하게 유지
     - 구조 왜곡: 제어 신호와 확산 스텝 증가
5. `semantic_reliability.py` 확장
   - 현재 SRS를 베이스라인 점수로 유지
   - 선택적 패킷 인식 항 추가:
     - 관계 일관성
     - 속성 일관성
     - 세그멘테이션 일관성
   - 둘 다 리포트:
     - `srs_base`
     - `srs_packet`

기대 출력:

- 이미지별 `packet.json`
- 이미지별 `error_report.json`
- `srs_base`, `srs_packet`, 오류 카운트를 포함한 SNR sweep CSV
- 고-SNR 열화에 대한 적응형 가이드 ablation 결과

### Phase 4-B: 키프레임 및 시간적 확장

이것이 실제 비디오/키프레임 마일스톤이며, Phase 4-A의 패킷 및 검증기 인프라 위에
구축된다.

주요 참고:

- `paper/FAST-GSC: Fast and Adaptive Semantic Transmission for Generative Semantic Communication/FAST_GSC.tex`
  - 시맨틱 유닛
  - 전송 순서
  - 시맨틱 차이 계산
  - 순차적 조건부 디노이징

계획 파일:

```text
src/sgdjscc_lab/
├── video/
│   ├── keyframe_extractor.py
│   ├── scene_change_detector.py
│   ├── semantic_delta.py
│   ├── temporal_pipeline.py
│   └── motion_residual.py
└── evaluators/
    └── temporal_consistency.py
```

구현 단계:

1. `scene_change_detector.py`
   - 실용적 휴리스틱 탐지기로 시작:
     - CLIP 이미지-이미지 거리
     - 연속 프레임 간 LPIPS
     - 선택적 색상 히스토그램 델타
   - 새 키프레임을 위한 장면 경계 표시
2. `keyframe_extractor.py`
   - GOP 형태의 그룹 생성
   - 출력:
     - 키프레임 인덱스
     - 인터프레임 범위
3. `semantic_delta.py`
   - 프레임 `t`의 패킷을 이전 키프레임 또는 이전 전송 프레임의 패킷과 비교
   - 델타 유닛 생성:
     - 신규 객체
     - 제거된 객체
     - 변경된 관계
     - 변경된 속성
     - 변경된 장면
4. `temporal_pipeline.py`
   - 키프레임:
     - 전체 이미지 파이프라인 실행
     - 전체 시맨틱 패킷 저장
   - 인터프레임:
     - 최신 키프레임 패킷 재사용
     - 시맨틱 델타만 적용
     - 변화량에 따라 가이드를 재사용하거나 약화
5. FAST-GSC에서 영감받은 순차 디노이징 스케줄
   - Phase 4-B는 FAST-GSC 학습을 직접 재구현하지 않는다
   - 대신, 디노이징 스케줄 동안 시맨틱 그룹을 단계적으로 주입해
     `시맨틱 유닛의 시간 경과 도착`을 모사한다
   - 실용적 첫 분할:
     - 초기 디노이징: 장면 + 주요 객체
     - 중간 디노이징: 관계 + 구조
     - 후기 디노이징: 속성 + 미세 보정
6. `temporal_consistency.py`
   - 리포트:
     - 시간적 SRS
     - 객체 동일성(identity) 일관성
     - 시간적 세그멘테이션 IoU
     - 시간적 할루시네이션율

기대 출력:

- 키프레임 목록 JSON
- 시퀀스별 시간적 지표 CSV
- 나란히 비교 리포트:
  - 프레임별 전체 시맨틱
  - 키프레임 재사용
  - 키프레임 + 시맨틱 델타 전송

## Phase 4 완료 기준

다음이 모두 참일 때 Phase 4를 완료로 간주한다:

1. `scripts/evaluate.py`가 이미지 모드에서 패킷 인식 SRS 평가를 실행할 수 있음
2. 새 비디오 평가 진입점이 정렬된 프레임 폴더를 처리할 수 있음
3. 키프레임 재사용과 시맨틱 델타 로직이 구조화된 로그를 생성함
4. 시간적 SRS와 시간적 할루시네이션 지표가 export됨
5. 리포트가 단순 프레임별 전체 가이드 대비 시맨틱 전송 오버헤드의 구체적 감소를 보임

## Phase 4 실험 설계

데이터셋:

- 이미지:
  - Kodak
  - COCO val2017
  - ADE20K validation
- 비디오:
  - 가능하면 ETRI 내부 키프레임/장면 전환 데이터
  - 없으면 정렬된 이미지 폴더로 추출한 공개 비디오 프레임

채널 설정:

- AWGN:
  - `-15, -10, -5, 0, 5, 10, 15 dB`
- 시맨틱 델타 실험용 선택적 packet-drop 시뮬레이션

주요 ablation:

1. 베이스라인 SGD-JSCC
2. SGD-JSCC + 적응형 가이드
3. SGD-JSCC + 적응형 가이드 + 패킷 인식 검증기
4. 키프레임 전용 전체 패킷
5. 키프레임 + 시맨틱 델타 재사용

---

<a id="phase-4-implementation-status"></a>

# Phase 4 구현 현황

Phase 4는 변경되지 않은 SGD-JSCC 이미지 forward pass 위에 얹은 **config 기반,
opt-in** 확장 세트로 구현된다. 모든 신규 기능은 기본값 *off*이므로, 명시적으로
활성화하지 않는 한 기존 `infer_images.py` / `evaluate.py` 경로는 Phase 3와 정확히
동일하게 동작한다.

## Phase 4-A (신뢰도 우선 이미지 확장) — 제공됨

| 영역 | 모듈 |
|---|---|
| 시맨틱 패킷 | `guidance/semantic_packet_extractor.py`, `guidance/object_extractor.py`, `guidance/relation_extractor.py`, `guidance/importance_estimator.py`, `utils/packet_io.py` |
| 적응형 가이드 | `controllers/snr_guidance_policy.py`, `controllers/adaptive_guidance_controller.py` |
| 패킷 인식 검증기 | `evaluators/semantic_packet_matcher.py`, `evaluators/relation_consistency.py`, `evaluators/attribute_consistency.py` |
| SRS 확장 | `evaluators/semantic_reliability.py` (`srs_base`, `srs_packet`, `score_packet`) |
| 구조화된 regeneration | `controllers/regeneration_policy.py` (실패 양상 키 기반 재시도) |
| 평가 통합 | `pipelines/eval_pipeline.py` (패킷 빌드/저장, 패킷 지표, CSV 컬럼) |

config에서 활성화 (`configs/eval/default.yaml` 참조):

```yaml
use_packet_eval: true          # 패킷 빌드, srs_base / srs_packet + 오류 카운트 출력
use_adaptive_guidance: true    # SNR-구간 가이드 스케일링 (강/중/약)
use_packet_regeneration: true  # 오류 유형 인식 재시도 (use_packet_eval 필요)
```

패킷 인식 이미지 평가 실행:

```bash
python scripts/evaluate.py --config configs/composed.yaml --snr 0 \
    -i ../inputs/   # 먼저 eval/default.yaml에서 use_packet_eval: true 설정
```

이미지별 출력 (`packet_dir` 아래): `<stem>.orig_packet.json`,
`<stem>.packet.json`, `<stem>.error_report.json`. 결과 CSV에는
`srs_base, srs_packet, object_match_rate, relation_consistency,
attribute_consistency, segmentation_consistency, scene_match,
missing_object_count, additional_object_count, relation_error_count,
attribute_error_count, guidance_regime`이 추가된다.

## Phase 4-B (키프레임 / 시간적 확장) — 제공됨

| 영역 | 모듈 |
|---|---|
| 장면 전환 | `video/scene_change_detector.py` (히스토그램 + 선택적 CLIP/LPIPS) |
| 키프레임 / GOP | `video/keyframe_extractor.py` |
| 시맨틱 델타 | `video/semantic_delta.py` |
| 모션 / residual | `video/motion_residual.py` |
| 시간적 파이프라인 | `video/temporal_pipeline.py` (키프레임 full / 인터프레임 재사용 + 델타; `build_staged_schedule`의 단계적 prompt를 `cfg.prompt_override`로 복원에 연결) |
| 시간적 지표 | `evaluators/temporal_consistency.py` (시간적 SRS, 객체 동일성 일관성, 시간적 세그멘테이션 IoU, 시간적 할루시네이션율) |
| CLI / config | `scripts/evaluate_video.py`, `configs/video/default.yaml`, `configs/composed_video.yaml` |

정렬된 프레임 폴더에 대해 키프레임/시간적 평가 실행:

```bash
# 전체 실행 (SGD-JSCC + CLIP/BLIP2 로드):
python scripts/evaluate_video.py --config configs/composed_video.yaml \
    --input /path/to/ordered_frames/ --snr 5 --device cuda:0

# 키프레임/델타/시간적 로직 dry run (체크포인트 없음). 캡션이 제공되지 않으면
# 패킷은 비어 있고, 캡션을 주면 시맨틱 델타/지표도 의미를 갖는다:
python scripts/evaluate_video.py --config configs/composed_video.yaml \
    --input /path/to/ordered_frames/ --no-models --captions /path/captions.txt
```

출력: `keyframes.json` (GOP 구조), `temporal_frames.csv` (프레임별 로그),
`temporal_metrics.csv` (시퀀스 지표 + `overhead_reduction`, 단순 프레임별 전체
전송 대비 시맨틱 유닛 절감). 이미지 평가기의 패킷/오류 JSON은 SNR별로
namespace가 부여되어(`packet_dir/snr_<snr>/…`) SNR sweep 시 서로 덮어쓰지 않는다.

## 참고 매핑 (어디서 왔는가)

- **FAST-GSC** → 시맨틱 유닛 패킷 설계(`semantic_packet_extractor`), 중요도 기반
  전송 순서(`importance_estimator`), 시맨틱 차이 계산(`semantic_delta`), 단계적
  조건부 디노이징 근사(`temporal_pipeline.build_staged_schedule`) — 단계적 prompt는
  샘플러 스텝별 주입이 아니라 확산 텍스트 조건(`cfg.prompt_override`)으로 공급된다.
- **SGD-JSCC** → 프레임별 복원 경로를 그대로 재사용; 키프레임은 기존 forward pass를
  호출하고, 적응형 가이드는 그 변경되지 않은 경로가 *어떤* config로 실행될지만 선택한다.

## 알려진 한계 / 다음 단계

- 패킷은 메타데이터로만 직렬화된다. 실제 시맨틱 패킷 채널 코딩 / drop 시뮬레이션은
  보류됨(Phase 5).
- 객체/장면/관계 추출은 CLIP/캡션 휴리스틱이다(아직 scene-graph나 POPE-VQA 모델
  없음); 관계/속성 파싱은 결정론적이지만 얕다.
- 단계적 디노이징은 **prompt** 레벨에서 연결된다(`cfg.prompt_override`가 패킷 유래
  단계적 prompt를 실제 복원에 공급). DPM-Solver 루프 *내부*의 진정한 스텝별 prompt
  전환은 SGD-JSCC 샘플러 수정이 필요하므로(알고리즘 보존 불변식) **구현되지 않았다**.
- 인터프레임 재사용은 키프레임 복원을 복사한다(GOP 키프레임이 단일한 일관 패킷+픽셀
  참조). 진정한 델타-워프 / 모션 보상 합성은 향후 작업이다.

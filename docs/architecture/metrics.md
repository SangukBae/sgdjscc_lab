---
status: active
updated: 2026-09-07
owner: ETRI SGD-JSCC 연구팀
source_commit: 8fbe6d98
supersedes: docs/etri_overview.md
---

> [← 문서 색인](../README.md)

# 평가 지표 정의

- 문서 범위
  - 공식 지표 정의
  - 구현·검증 상태: [current/status.md](../current/status.md)
  - 실행 절차: [protocols/evaluation.md](../protocols/evaluation.md)

## 이미지 지표 (`outputs/results.csv`)

- 기본 열
  - 품질: `psnr`, `ssim`, `lpips`, `fid`
  - 의미: `clip_image_image`, `clip_text_image`, `semantic_reliability_score`
  - 객체: `object_preservation_rate`, `missing_object_rate`, `additional_object_rate`
  - 할루시네이션: `hallucination_score`
  - 정의 위치: `src/sgdjscc_lab/utils/csv_logger.py::RESULT_COLUMNS`
- 패킷 평가 확장 열
  - `srs_base`
  - `srs_packet`
  - `srs_v2`

### Semantic Reliability Score (SRS)

- 과제의 헤드라인 지표. 가중치는 `configs/base/eval/default.yaml`에 있다.

```python
SRS = (0.30*clip_image_image + 0.25*clip_text_image + 0.25*object_preservation_rate
       - 0.10*missing_object_rate - 0.10*additional_object_rate)
```

- `srs_base` — 위 기본 SRS.
- `srs_packet` — semantic packet 기반 검증 결과를 blend한 확장 SRS.
- `srs_v2` — packet + temporal + VQA를 더한 SRS-v2(Phase 5-C 계열).

## 시간축(영상) 지표 (`temporal_metrics.csv`)

- `temporal_srs` — 시퀀스 전체의 평균 SRS.
- `srs_flicker` — 프레임 간 SRS 변동 폭(낮을수록 안정).
- `object_identity_consistency` — 같은 물체가 프레임이 넘어가도 동일하게 유지되는 정도.
- `temporal_hallucination_rate` — 영상 전체에서 없던 것이 지어내지는 비율.
- `PTC` (Packet-Temporal Consistency)
  - 전송 packet과 복원 packet의 시간축 일치도
- `SFR` (Semantic Flicker Rate)
  - 객체의 frame별 birth/death 비율
  - `srs_flicker`와 별도 보고
- `SDI` (Semantic Drift Index)
  - keyframe 거리 증가에 따른 의미 이탈
- `overhead_reduction` — 프레임별 전체 전송 대비 시맨틱 유닛(키프레임+델타) 절감률.

- `PTC`/`SFR`/`SDI` 검증 상태
  - 초기값: CLIP/packet 기반 잠정치
  - 보강값: OWLv2/VQA 기반 재측정
  - 근거: [OWLv2/VQA 보강 실험](../experiments/2026-07-28_owlv2_vqa_calibration.md)

## 지표 순환 분리 원칙 (loop-internal vs held-out)

- 분리 대상
  - 제어 지표: 재생성·선택 구동
  - 보고 지표: 결과 우위 검증
- 목적
  - 같은 지표를 최적화와 결과 주장에 재사용하는 순환 평가 방지

- **loop-internal** (재생성 구동): `srs_packet` / VQA hallucination 판정.
- **held-out**
  - 결과 보고용
  - 별도 GT 대조
  - 재생성에 관여하지 않은 temporal 지표

- 코드 규칙
  - `PacketVerifier` report에 `metric_role` 기록
  - 허용값: `loop_internal`, `held_out`
  - 구현: `pipelines/heldout_remeasurement.py`

## SAVER-JSCC 지표 계약 (`TARGET`, 미구현 포함)

SAVER 구조·gate 정의는
[saver_jscc_model_plan.md](../current/saver_jscc_model_plan.md)를 따른다. 아래 필드가
현재 evaluator에 모두 구현됐다는 뜻은 아니다. 구현 여부는 [status.md](../current/status.md)에
기록한다.

### Signed-control 지표

- `source_paired_h_add`
  - source GT에 없고 reconstruction에서 추가된 concept 수를 source-side opportunity로
    나눈 값.
  - selector가 negative 후보를 만들 때 사용한 detector와 독립된 evaluator로 계산한다.
- `false_suppression_rate`
  - source에서 present인데 negative/revocation action 이후 reconstruction에서 누락된
    entity 비율.
  - absolute 값과 positive-only 대비 paired delta를 모두 보고한다.
- `ghost_survival_auc`
  - GT EXIT/REVOKE event 이후 entity presence score를 사전 동결한 horizon에서 적분한 값.
  - horizon 전에 clip이 끝난 event는 censored로 표시하고 분모를 숨기지 않는다.
- `revocation_latency_frames`
  - valid REVOKE 수신 시점부터 held-out evaluator가 absence를 처음 확인할 때까지의
    frame 수. timeout/censoring을 별도 기록한다.

### Memory 지표

- `identity_consistency_no_memory`, `identity_consistency_append_only`,
  `identity_consistency_revocable`
- `identity_gain_retention`

```python
identity_gain_retention = (
    identity_consistency_revocable - identity_consistency_no_memory
) / max(
    identity_consistency_append_only - identity_consistency_no_memory,
    eps,
)
```

append-only가 no-memory보다 개선되지 않으면 위 retention으로 memory contribution을
주장하지 않고 해당 gate를 실패 처리한다.

### State/protocol 지표

```text
stale_resurrection_count
invalid_transition_count
duplicate_non_idempotent_count
cross_epoch_mutation_count
corrupt_negative_applied_count
memory_slot_swap_rate
```

앞의 다섯 violation count는 formal property test에서 0이어야 한다. 이는 안전 불변조건이며
생성 품질 우위 지표와 합성하지 않는다.

### Rate와 비용

```text
positive_payload_bytes
negative_payload_bytes
state_header_bytes
protection_bytes
feedback_bytes
retransmission_bytes
padding_bytes
total_on_wire_bytes
complex_channel_uses
trainable_parameter_count
peak_vram_bytes
latency_p50_ms
latency_p95_ms
```

- `saver_source`: exact bytes와 effective bpp가 공식 rate다.
- `saver_wireless`: actual complex channel uses가 공식 rate다.
- exact byte, estimated byte, proxy symbol, actual channel use는 서로 다른 열에 보존한다.
- trainable parameter, sampling step과 latency를 함께 맞추지 않은 비교는
  matched-compute로 부르지 않는다.

## 통합 리포트의 공식 축

- 결과 보존 원칙
  - 합성 점수 하나로 축약 금지
  - 아래 열을 독립 보존

```text
rate: exact_bundle_bytes, feedback_bytes, retransmission_bytes,
      effective_bits_per_frame, proxy_channel_symbols
quality: psnr, ssim, lpips, srs
hallucination: missing_rate, additional_rate, hallucination_score,
               temporal_hallucination_rate
cost: reconstruction_latency_ms, regeneration_latency_ms, retry_count,
      end_to_end_latency_ms
```

```python
effective_bits_per_frame = 8 * (
    exact_bundle_bytes + feedback_bytes + retransmission_bytes
) / evaluated_frames
```

- 집계 규칙
  - `proxy_channel_symbols`: 변조·FEC 가정을 포함한 참고값
  - exact byte와 proxy symbol: 합산 금지
  - 재생성: 추가 전송이 없어도 retry 수·지연 기록
  - baseline·paired 통계: [평가 프로토콜](../protocols/evaluation.md)

## Presence(객체 존재) 판정 backend

- 공통 인터페이스: `evaluators/presence_backends.py`
- backend
  - `clip`: 전역 유사도 + 고정 임계값
  - `owlv2`: zero-shot detector
  - `vqa`: BLIP-2 yes/no 질의
  - `gt`: 수작업 GT
  - `mock`: 테스트용
- 앙상블: `evaluators/presence_calibration.py`

- `ensemble_gt_filter`
  - closed-world 판정
  - GT object만 유지
  - object preservation 근거
- `ensemble_openworld_filter`
  - count·action·scene 잡음 제거
  - GT 밖 object 유지
  - hallucination·additional object 근거

- 실제 검증 수치는 [experiments/2026-07-28_owlv2_vqa_calibration.md](../experiments/2026-07-28_owlv2_vqa_calibration.md) 참고.

## 관련 문서
- [current/saver_jscc_model_plan.md](../current/saver_jscc_model_plan.md) — SAVER gate와 metric 사용 조건
- [system.md](./system.md) — 파이프라인 개요
- [tx_rx_contract.md](./tx_rx_contract.md) — 지표가 검증하는 Tx/Rx 설계
- [protocols/evaluation.md](../protocols/evaluation.md) — 지표를 계산하는 실행 절차

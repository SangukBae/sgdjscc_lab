---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: 63b7b23
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
- [system.md](./system.md) — 파이프라인 개요
- [tx_rx_contract.md](./tx_rx_contract.md) — 지표가 검증하는 Tx/Rx 설계
- [protocols/evaluation.md](../protocols/evaluation.md) — 지표를 계산하는 실행 절차

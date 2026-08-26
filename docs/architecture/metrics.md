---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: 63b7b23
supersedes: docs/etri_overview.md
---

> [← 문서 색인](../README.md)

# 평가 지표 정의

이 문서는 지표의 **공식 정의**만 다룬다. 지표별로 무엇이 이미 실측/보강됐고
무엇이 CLIP 기반 잠정치인지는 [current/status.md](../current/status.md), 평가를
어떻게 실행하는지는 [protocols/evaluation.md](../protocols/evaluation.md)를 따른다.

## 이미지 지표 (`outputs/results.csv`)

이미지 × SNR별 한 행: `psnr, ssim, lpips, clip_image_image, clip_text_image,
object_preservation_rate, missing_object_rate, additional_object_rate,
hallucination_score, semantic_reliability_score, fid`
(`src/sgdjscc_lab/utils/csv_logger.py::RESULT_COLUMNS`). 패킷 평가를 켜면
`srs_base, srs_packet, srs_v2` 등이 추가된다.

### Semantic Reliability Score (SRS)

과제의 헤드라인 지표. 가중치는 `configs/base/eval/default.yaml`에 있다.

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
- `PTC` (Packet-Temporal Consistency) — 전송 packet과 복원 영상 packet의 일치도가
  시간축에서 유지되는 정도.
- `SFR` (Semantic Flicker Rate) — 객체가 프레임마다 생겼다 사라지는 birth/death 비율.
  기존 `srs_flicker`와는 별도로 보고한다.
- `SDI` (Semantic Drift Index) — 키프레임에서 멀어질수록 의미가 원래 의미에서
  얼마나 이탈하는지.
- `overhead_reduction` — 프레임별 전체 전송 대비 시맨틱 유닛(키프레임+델타) 절감률.

`PTC`/`SFR`/`SDI`는 초기 구현이 CLIP/packet 기반 잠정치였고, 이후 OWLv2/VQA
보강 실험으로 재측정됐다 — 상세 수치는 [experiments/2026-07-28_owlv2_vqa_calibration.md](../experiments/2026-07-28_owlv2_vqa_calibration.md).

## 지표 순환 분리 원칙 (loop-internal vs held-out)

재생성/선택을 **구동하는 지표**와 결과 우위를 **보고하는 지표**를 분리한다.
같은 지표로 최적화와 승리 주장을 동시에 하면 순환 평가가 된다.

- **loop-internal** (재생성 구동): `srs_packet` / VQA hallucination 판정.
- **held-out** (결과 보고): 재생성 루프에 쓰지 않은 지표 — 별도 GT 대조,
  또는 재생성에 관여하지 않은 temporal 지표.

각 `PacketVerifier` report는 `metric_role`(`loop_internal`/`held_out`)을 태그해
둘을 코드 수준에서도 섞이지 않게 한다(`pipelines/heldout_remeasurement.py`).

## 통합 리포트의 공식 축

Rate–Reliability–Hallucination 결과는 합성 점수 하나로 축약하지 않는다. 최소한
다음 열을 독립적으로 보존한다.

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

`proxy_channel_symbols`는 변조·FEC 가정을 명시한 참고값이며 exact byte와 합치지
않는다. 재생성만 수행해 추가 전송이 없더라도 retry 수와 지연은 반드시 기록한다.
구체적인 baseline·paired 통계 절차는
[protocols/evaluation.md](../protocols/evaluation.md)를 따른다.

## Presence(객체 존재) 판정 backend

`object_preservation`/`hallucination`/packet verifier가 공통으로 의존하는
"이 객체가 있는가"라는 판정은 여러 backend로 교체 가능하다
(`evaluators/presence_backends.py`): `clip`(전역 유사도 + 고정 임계값, 기본값),
`owlv2`(zero-shot detector), `vqa`(BLIP-2 yes/no 질의), `gt`(수작업 GT), `mock`.
`evaluators/presence_calibration.py`가 여러 backend를 앙상블한다.

- `ensemble_gt_filter` — GT에 명시된 object만 남기는 closed-world 판정.
  **object preservation(의미 보존) 주장의 근거로 쓴다.**
- `ensemble_openworld_filter` — count/action/scene 잡음만 제거하고 GT에 없는
  object(할루시네이션 후보 포함)는 남긴다. **hallucination/additional object
  분석의 근거로 쓴다.**

실제 검증 수치는 [experiments/2026-07-28_owlv2_vqa_calibration.md](../experiments/2026-07-28_owlv2_vqa_calibration.md) 참고.

## 관련 문서
- [system.md](./system.md) — 파이프라인 개요
- [tx_rx_contract.md](./tx_rx_contract.md) — 지표가 검증하는 Tx/Rx 설계
- [protocols/evaluation.md](../protocols/evaluation.md) — 지표를 계산하는 실행 절차

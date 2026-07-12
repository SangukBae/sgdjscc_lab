> [← 문서 색인](./README.md)

# ETRI 전략 정리

이 문서는 기존 [개발계획서 보관본](./archive/etri_development_plan_v2.md),
[로드맵 보관본](./archive/etri_development_roadmap.md),
[한계점 지도 보관본](./archive/limitation_reference_map.md)을 합친 **ETRI 과제용 단일
전략 문서**다. 목적은 문서 수를 줄이면서도 "무엇이 핵심 문제이고, 무엇을 먼저
구현·평가해야 하는가"를 한 곳에서 보게 하는 것이다.

## 목표

`sgdjscc_lab`의 1차 목표는 최대 `PSNR`이 아니라 **무선 전송 후 시맨틱 의도의 신뢰성
있는 보존**이다. 즉, 수신 이미지가 자연스러워 보이는지보다 **원래 의도한 객체·관계·
장면 정보를 얼마나 정확히 보존했는가**를 더 중요하게 본다.

## 핵심 한계 3가지

| 핵심 한계 | 의미 | 현재 대응 |
|---|---|---|
| 할루시네이션과 객체/정보 왜곡 | 복원 이미지는 그럴듯하지만 없던 객체를 만들거나, 있어야 할 정보를 누락·왜곡할 수 있음 | hallucination evaluator, semantic packet verifier, regeneration, VQA 기반 검증 |
| 화질 중심 평가의 한계 | `PSNR`·`SSIM`만으로는 송신 의도와 수신 복원의 의미 일치를 설명하기 어려움 | SRS/srs_packet/srs_v2, CLIP, object preservation/missing/additional |
| 정지 이미지 중심 한계 | 단일 이미지에서는 시간 흐름, 장면 전환, 프레임 간 의미 일관성을 평가할 수 없음 | keyframe pipeline, scene change, temporal evaluator, temporal SRS |

우선순위는 **할루시네이션 완화/검출 → 의미 충실도 평가 체계 → 시간축 확장** 순서다.

## 권장 개발 순서

| 단계 | 초점 |
|---|---|
| 1~2 | 원본 SGD-JSCC 추론 경로 보존 + 모듈 구조 정리 |
| 3~4 | End-to-End 평가 골격과 시맨틱 우선 성공 기준 확립 |
| 5~6 | 의미 평가기 모음과 SRS 통합 |
| 7 | 할루시네이션 완화, 객체 추가·누락·왜곡 검출 |
| 8 | 화질 중심 평가에서 의미 충실도·신뢰성 평가 체계로 전환 |
| 9 | 정지 이미지에서 영상·장면 전환·시간 일관성으로 확장 |
| 10 | 가이드 손상·오버헤드 견고성 |
| 11 | 저지연 복원 |
| 12 | 블라인드/페이딩 채널 견고성 + 공정 비교 프로토콜 |

핵심 3축이 먼저이고, 오버헤드·저지연·채널 조건화는 그 위에 얹는 보조 연구 축이다.

## 현재 구현 상태

| 묶음 | 상태 | 요약 |
|---|---|---|
| 1~4 | 완료 | 원본 경로 보존, 모듈화, End-to-End 평가, 시맨틱 우선 철학 정리 |
| 5~6 | 완료 | 품질·CLIP·패킷·시간적·VQA 지표와 `srs_base/srs_packet/srs_v2` 연결 |
| 7 | 부분 | packet verifier, regeneration search, VQA, SRS-v2는 있으나 검증 일부는 휴리스틱 |
| 8 | 완료에 근접 | 의미 지표와 CSV 기록 경로가 구축됨 |
| 9 | 부분 | keyframe/scene-change/temporal evaluator는 있으나 motion residual 통합은 미완 |
| 10~12 | 부분/스캐폴드 | guide damage, edge codec, low-latency, channel conditioning은 연결됐지만 일부는 placeholder |

## 모듈 매핑

- 할루시네이션 완화·검출: `guidance/semantic_packet`, `evaluators/hallucination*`,
  `evaluators/packet_matcher`, `controllers/regeneration`
- 의미 충실도 평가: `evaluators/clip_score.py`,
  `evaluators/object_preservation.py`, `evaluators/semantic_reliability*.py`
- 시간축 확장: `video/keyframe.py`, `video/scene_change.py`,
  `video/temporal_pipeline.py`, `evaluators/temporal_consistency.py`
- 보조 축: `controllers/adaptive_guidance.py`, `models/diffusion_wrapper_channel.py`,
  `acceleration/`, `channels/`

## 관련 문서

- [etri_overview.md](./etri_overview.md)
- [phase4.md](./phase4.md)
- [phase5.md](./phase5.md)
- [training_scaffold.md](./training_scaffold.md)
- 보관본: [archive/etri_development_plan_v2.md](./archive/etri_development_plan_v2.md),
  [archive/etri_development_roadmap.md](./archive/etri_development_roadmap.md),
  [archive/limitation_reference_map.md](./archive/limitation_reference_map.md)

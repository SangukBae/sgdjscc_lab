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

## 한계별 원인과 해결 방안 (코드 검증 기반)

아래는 세 한계의 원인을 실제 코드 위치로 특정하고 해결 방안을 정리한 것이다.
모든 파일 경로는 `src/sgdjscc_lab/` 기준이다.

### 공통 병목 (한계 1·2)

`ObjectPreservationEvaluator._detect_objects()`
(`evaluators/object_preservation.py:98-115`)가 객체 존재를 **CLIP 전역
텍스트-이미지 유사도 + 고정 절대 임계값 0.25 + COCO-80 고정 어휘**로 판정한다.

전파 경로:

- `evaluators/hallucination.py:107-109` — 같은 `_detect_objects()`로
  `recon − orig` 계산 → **직접 상속**
- SRS 5개 항 중 3개(preservation/missing/additional,
  `evaluators/semantic_reliability.py:183-211`) → **직접 상속**
- `srs_packet` — base SRS를 `packet_blend=0.5`로 blend
  (`semantic_reliability.py:291-292`) → **부분 상속**; packet composite은 별도
  matcher 경로라 여기를 고쳐도 packet 추출(caption 기반) 오류원은 남음
- `srs_v2` — base·packet 층 경유 → **간접 상속**

이 지점을 고치면 한계 1·2의 **주 오류원**이 제거되지만, packet 계열 전체가
해결되는 것은 아니다.

**추가 병목 — 죽은 config 노브:** `configs/eval/default.yaml:73`의
`object_presence_threshold`는 `EvalContext`가 evaluator 생성 시 전달하지 않아
(`pipelines/eval_pipeline.py:137-146`) 어떤 값을 줘도 무효다. 판정 로직 개선
이전에 배선부터 필요하다.

### 한계 1: 할루시네이션과 객체/정보 왜곡

원인:

| # | 원인 | 근거 |
|---|---|---|
| 1 | 전역 CLIP 유사도 → 위치·개수·크기 정보 없음 ("2마리→3마리" 검출 불가) | `object_preservation.py:98-115` |
| 2 | 절대 임계값 구조라 score 분포 보정이 없음 — 실제 CLIP cosine 분포가 좁은 구간(경험적으로 0.2~0.3)에 몰리면 경계 객체가 깜빡이며 가짜 missing/additional 생성 | 코드상 구조적 취약점 + 경험적 관측 |
| 3 | 임계값이 config에 있으나 미배선 → 튜닝 자체가 불가능 | `configs/eval/default.yaml:73`, `eval_pipeline.py:137` |
| 4 | VQA 층은 구현됐으나 기본 `use_vqa_hallucination: false`라 층 자체가 빠짐; 켜도 backend가 `none`/실패면 CLIP fallback | `eval_pipeline.py:188`, `configs/eval/phase5.yaml:14-16`, `hallucination_vqa.py:100-104` |
| 5 | 재생성이 사후(reactive) 전용 — guidance_scale/step 조정만, 할루시네이션의 명시적 억제 없음 | `controllers/regeneration_policy.py:166` |
| 6 | negative prompt가 품질 단어로 하드코딩 — 의미 억제 훅 부재 | `pipelines/infer_pipeline.py:853-857` |

해결:

1. **[0순위] CLIP 판정 안정화 — config 배선 포함 한 묶음.**
   `EvalContext` → evaluator로 `object_presence_threshold` 전달 배선(기본값
   0.25 유지로 수치 보존되므로 phase gate 불필요). 히스테리시스 이중 임계값
   (presence ≥ 0.27 / absence < 0.23, 중간 밴드는 집계 제외) + uncertain band
   처리 정책을 config로 명시 — 이쪽은 opt-in config gate 뒤. 절대 임계값 →
   vocabulary softmax 상대 판정 또는 원본↔복원 Δsim 판정 옵션. 기본값에서
   기존 수치 재현을 회귀 테스트로 보증.
2. **OWLv2 grounded 검출기 백엔드.** `presence_backend: {clip|owlv2}` 플래그
   (transformers 4.49 pinned 환경 호환 확인). 박스+점수로 개수·위치 판정.
   검증 프로토콜 필수 동반: SNR 단조성, 합성 객체 주입 테스트, 고 SNR flip
   rate(저 SNR diffusion 아티팩트 오검출 대비).
3. **VQA 층 활성화 + POPE식 질문 샘플링(신규 구현).** 현재는 backend만 있고
   후보 선정은 `recon_packet.objects` 전달뿐
   (`semantic_reliability_v2.py:126-128`). adversarial/popular/random 네거티브
   샘플링은 루트 `POPE/`를 참조한 새 코드로 추가. 백엔드는
   `blip2-opt-2.7b-coco`(환경에서 known-good) 지정.
4. **사후 재생성 → 사전 억제.** `cfg.negative_prompt_extra` 훅 신설: 기본
   extra는 empty이고, 그때 최종 negative prompt는 기존 하드코딩 문자열과
   byte-identical(phase4 게이트 뒤). `additional_objects`를 재생성 negative
   prompt에 주입하되 "전송 packet에 없음 ∧ VQA가 복원에서 yes" 이중 조건
   시에만(오탐으로 실제 객체 삭제 방지). MDTv2에서의 억제 효과는 A/B 검증 후
   채택.

### 한계 2: 화질 중심 평가의 한계

원인:

| # | 원인 | 근거 |
|---|---|---|
| 1 | SRS 세 항이 CLIP 프로브에 직접 의존, srs_packet도 blend 0.5로 부분 상속 → 스칼라의 근거가 약함 | `semantic_reliability.py:183-211, 291` |
| 2 | GT 경로가 "예약만 됨" — `metadata` 인자가 시그니처에만 있고 본문 미사용, eval pipeline도 전달 안 함 → 모든 평가가 원본↔복원 자기참조 | `object_preservation.py:121`, `semantic_reliability.py:146`, `eval_pipeline.py`(전달 0건) |
| 3 | SRS 가중치(0.30/0.25/0.25/0.10/0.10)가 사람 판단과 정렬 검증 안 됨 | `semantic_reliability.py:45-51` |

해결:

1. **검출기 접지 공유.** 한계 1의 해결 1·2가 SRS 세 항의 신뢰도를 직접 올림.
   packet composite은 caption 기반 packet 추출 품질에 별도 의존하므로 "주
   오류원 제거"로 기술.
2. **GT 주석 경로 개통 — 2단계.** (a) eval pipeline이 dataset 주석을 읽어
   `metadata`로 전달하는 배선, (b) evaluator가 `metadata.objects`를 원본 검출
   대신 사용하는 분기. COCO/Visual Genome 주석으로 GT 대비 절대 preservation
   산출.
3. **VLM-judge 가중치 적합.** Qwen2.5-VL(기존 caption 환경 재활용)로 의도
   보존 0~1 점수화 → judge 점수를 타깃으로 SRS 5개 항 회귀 적합. 가중치는
   config 주입식(`semantic_reliability.py:103-106`)이라 적합 결과 반영은 코드
   무변경.
4. **오류유형 분해 리포트 — 현황 구분.** error_report는 이미 JSON으로
   저장되고(`eval_pipeline.py:579-581`) missing/additional/relation/attribute
   count 4종은 CSV 컬럼에 존재(`utils/csv_logger.py:136-139`). 추가 작업은
   객체 리스트(어떤 객체가 누락/추가됐는지)를 CSV 행으로 펼치는 부분과 SNR
   sweep 리포트에서의 집계·시각화다.
5. **Tx/Rx 정보 경계 명문화.** **Rx-legal**(전송 packet ↔ 복원 packet, packet
   기반 VQA, negative-prompt 재생성 = 자가 검증 수신기) vs **Eval-only**(원본
   이미지·GT 대비)를 문서·코드에서 구분. 배치 가능성 질문에 대한 답이 된다.

### 한계 3: 정지 이미지 중심의 한계

원인:

| # | 원인 | 근거 |
|---|---|---|
| 1 | `video/motion_residual.py`가 결정 경로에 미연결 — import는 `video/__init__.py`(docstring)와 `tests/test_video.py`뿐. 단위 테스트는 있으나 파이프라인이 소비하지 않음 | grep 검증 |
| 2 | reuse/recompute 결정이 의미 델타 단독(`reuse_threshold`) → 의미는 같고 픽셀만 움직이는 프레임(카메라 팬)을 잘못 재사용 | `video/temporal_pipeline.py:315-321` |
| 3 | 재사용 = keyframe 복원 전체 복사 → 느린 pan/drift가 누적돼도 인접 프레임 차이로는 안 잡힘 | `temporal_pipeline.py:320` |

해결:

1. **[1순위] motion gate — keyframe anchor 기준.** 인접 `(prev, curr)`가
   아니라 **`(keyframe_frame, curr)`** 대비로 `motion_residual.estimate()`
   계산(reuse가 keyframe 복원 복사이므로 드리프트는 keyframe 대비 누적량으로
   측정). 재사용 조건: "keyframe 대비 의미 델타 < t1 AND keyframe 대비
   block_max < t2" 이중 게이트. GOP anchor에 keyframe 원본 프레임 텐서 보관 +
   `FrameRecord.motion` 로깅 필드 추가.
2. **객체 깜빡임(flicker) 지표.** 프레임별 packet이 이미 보관되므로
   (`temporal_pipeline.py:141-142`) 원본에서 안정 존재하는 객체가 복원
   packet에서 나타났다 사라지는 birth/death rate 집계 = **시간적
   할루시네이션** 직접 측정. 새 모델 불필요,
   `evaluators/temporal_consistency.py`에 수십 줄.
3. **영역 선택적 재계산.** `block_motion`의 8×8 `block_map`으로 고모션 블록만
   식별 → 기존 128×128 patch tiling과 결합해 해당 블록만 재생성.
   `overhead_reduction`이 실제 모션에 정직해짐.
4. **(선택) 광류 백엔드.** `motion_residual.py:18` 예고 확장점대로 RAFT 옵션
   추가, keyframe 복원 flow-warp 기반 temporal SRS 정밀화.

### 실행 순서

| 순위 | 작업 | 범위 | 한계 | 규모 |
|---|---|---|---|---|
| 0 | CLIP 판정 안정화 | threshold 배선(게이트 불필요) + 히스테리시스·상대 판정(config gate) + uncertain band 정책 + 회귀 테스트 | 1·2 | 소~중 |
| 1 | motion gate | keyframe anchor 보관 + 이중 게이트 + `FrameRecord.motion` 로깅 | 3 | 소 |
| 2 | flicker/birth-death 지표 | packet 집계, 새 모델 불필요 | 3 | 소 |
| 3 | OWLv2 백엔드 | `presence_backend` 플래그 + 검증 프로토콜(단조성·주입·flip rate) | 1·2 | 중 |
| 4 | VQA/POPE + negative-prompt 훅 | POPE 샘플링 신규 구현 + cfg 훅(기본 empty→byte-identical) + 이중 안전장치 | 1 | 중 |
| 5 | GT metadata 경로 + VLM-judge 적합 | 배선 + 분기 2단계, judge 회귀 적합 | 2 | 중 |

**게이트 원칙:** 수치에 영향 없는 순수 배선(threshold 전달 등)은 게이트
불필요. 새 판정 로직(히스테리시스·상대 판정), 새 backend(OWLv2·VQA),
negative-prompt 재생성은 phase/config gate 뒤에 두고, 기본값에서 원본 SGD-JSCC
수치와 동일함을 회귀 테스트로 보증한다.

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

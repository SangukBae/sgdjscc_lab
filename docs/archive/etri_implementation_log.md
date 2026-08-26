> [← 문서 색인](../README.md)

# ETRI 구현 이력 (1차~6차, LGVSC 1A/1B/1C)

이 문서는 `etri_strategy.md`에 있던 상세 구현 로그를 2026-08-25에 그대로 옮긴
**과거 완료 기록**이다. 현재 상태 요약은 `etri_strategy.md`, 향후 계획은
`roadmap.md`를 참고한다. 아래 내용은 작성 당시 시점 기준이며 날짜/수치를
수정하지 않았다.

## 구현 실행 순서 (0~12)

이 표는 **개발자가 실제로 구현할 의존성 순서**다. 아래 월별 추진계획이 "언제 어떤
산출물을 보고할 것인가"라면, 이 표는 "그 산출물을 만들기 위해 코드에서 무엇을 먼저
붙여야 하는가"를 설명한다.

읽는 법:

- **PPT 블록**은 발표자료에 실제로 표시된 도형·기능 이름이다. 개발 작업이 어느 그림의
  어느 부분을 구현하는지 바로 확인하기 위한 열이다.
- **PPT 대응**은 발표자료에서 어느 한계와 슬라이드에 해당하는지 보여준다.
- **완료 기준**은 이 단계가 끝났다고 말할 수 있는 최소 산출물이다.
- 검출기에 의존하는 시간축 지표는 먼저 잠정 구현하고, OWLv2/VQA 보강 후 재측정한다.

| 순서 | PPT 블록 | 무엇을 구현하는가 | 해당 코드/모듈 | PPT 대응 | 월별 연결 | 완료 기준 |
|---|---|---|---|---|---|---|
| 0 ✅ | 슬라이드 6 `Packet Verifier` / 슬라이드 9 `Presence Calibration` | 객체 존재 판정 threshold가 실제 evaluator에 전달되게 하고, 히스테리시스·uncertain band를 옵션으로 둔다 | `object_preservation.py`, `hallucination.py`, `semantic_reliability*.py` | 한계 2·3, 슬라이드 6·9 | 7~8월 기반 작업 | 기존 기본값에서 결과가 깨지지 않고, threshold 변경이 실제 지표에 반영됨 — **완료** (`object_presence_threshold`/`object_presence_uncertain_band`가 EvalContext→SRS→하위 evaluator로 전달, 기본값 결과 불변) |
| 1 ✅ | 슬라이드 5 `입력 영상(mp4)` / `세그먼트 연결 → 복원 영상(mp4)` | mp4를 프레임으로 풀고, 복원 프레임을 다시 mp4로 저장한다 | 신규 `utils/video_io.py`, `evaluate_video.py` 확장 | 한계 1, 슬라이드 5 | 9월 | 테스트 영상 1개를 입력해 복원 mp4와 프레임별 로그가 생성됨 — **완료** (cv2/ffmpeg 백엔드, `--save-video`, `tests/test_video_io.py` 왕복 검증) |
| 2 ✅ | 슬라이드 5 `시간축 평가` / 슬라이드 9 `PTC·SFR·SDI` | temporal SRS와 별도로 packet consistency, 객체 깜빡임, semantic drift를 계산한다 | `evaluators/temporal_consistency.py`, `video/temporal_pipeline.py` | 한계 1·3, 슬라이드 5·9 | 8월 | `temporal_metrics.csv`에 `PTC`/`SFR`/`SDI` 초기값이 기록됨 — **완료(초기 지표)**. 2026-07-28 기준 OWLv2/VQA 보강 후 10개 영상 held-out 재측정까지 완료 |
| 3 ✅ | 슬라이드 5 `세그먼트 판단 게이트` / `의미 델타 + 모션 이중 게이트` | semantic delta만 보던 reuse/recompute 판단에 keyframe 대비 motion residual을 추가한다 | `video/semantic_delta.py`, `video/motion_residual.py`, `video/temporal_pipeline.py` | 한계 1, 슬라이드 5 | 9월 | 의미 변화는 작지만 카메라 이동이 큰 구간을 reuse하지 않음 — **완료** (`temporal.motion_threshold`/`motion_weight`/`motion_grid`, 기본 off = 기존 동작, decision 로그 기록) |
| 4 ✅ | 슬라이드 5 `키프레임` / `비-키프레임` / `세그먼트` 구조 | 프레임 단위 처리를 GOP/segment 단위 처리로 묶어 generate 분기를 붙일 수 있게 한다 | `video/keyframe_extractor.py`, 신규 `video/segment.py` | 한계 1, 슬라이드 5 | 9월 | 기존 frame-wise 결과와 segment 결과가 동등하게 재현됨 — **완료** (`segments.json`, frame-wise 로그와 병행 출력, `SegmentRecord.generation`은 generate 분기용 예약 인터페이스) |
| 5 ✅ 기초 | 슬라이드 5 `Generate (신규)` | 시작 keyframe, caption, side-info를 조건으로 세그먼트 생성 경로를 붙인다 | 신규 `video/video_generator.py`, `use_video_gen`/`video_generator.*` config | 한계 1·2, 슬라이드 5·6 | 9~10월 | `reuse/recompute/generate` 3-way 분기가 동작하고 생성 결과가 저장됨 — **완료(기초, 3차)**: mock(copy/interpolation) backend로 구조 구현, 실제 학습형 생성 모델(SVD/Open-Sora)은 후속 |
| 6 ✅ 기초 | 슬라이드 5 `Generate (start / start+end 양방향)` | 시작 keyframe과 끝 keyframe을 모두 조건으로 넣어 drift를 줄일 수 있는지 확인한다 | `video_generator` bidirectional mode, `pipelines/generation_mode_comparison.py` | 한계 1, 슬라이드 5 | 10월 이후 | start-only 대비 drift/flicker가 줄어드는지 비교 결과가 나옴 — **완료(기초, 4차)**: mock bidirectional backend + 비교 파이프라인 구조가 동작. 실제 drift/flicker 감소 여부에 대한 성능 주장은 실제 생성 모델 통합 후(5차+) 판단 |
| 7 ✅ 기초 | 슬라이드 6 `Packet Verifier` / `오류 유형별 재생성 Controller` | 전송 packet과 복원 packet을 비교하고, 추가·누락·왜곡별로 재생성 조건을 다르게 조정한다 | `evaluators/packet_verifier.py`, `controllers/verifier_controller.py`, `pipelines/packet_verification.py` | 한계 2, 슬라이드 6 | 9~10월 | 오류 유형별 report와 controller decision 로그가 생성됨 — **완료(기초, 2차)**: rule-based verifier/controller가 `TemporalPipeline` 결과에 옵션(기본 OFF)으로 연결됨. OWLv2/VQA 보강과 candidate action의 실제 sampler 반영은 5차·후속 |
| 8 ✅ 구현·실모델 재검증 완료 | 슬라이드 6 `Packet Verifier` 보강 / 슬라이드 9 `Presence Calibration` | CLIP 기반 객체 판정을 grounded detector와 VQA 질문으로 보강한다 | 신규 `evaluators/presence_backends.py`, `evaluators/presence_calibration.py`; 기존 `vqa_backend.py`/`hallucination_vqa.py` 재사용 | 한계 2·3, 슬라이드 6·9 | 9~10월 | verifier 판정의 오탐/미탐 사례가 줄어드는 정성·정량 결과가 나옴 — 공통 presence backend 인터페이스(clip/owlv2/vqa/gt/mock) + ensemble calibrator + `PacketVerifier` 보강 연결 완료. 이후 실제 OWLv2/VQA weight로 10개 영상 재측정 완료(`summary_metrics.csv`, 50 rows) |
| 9 ✅ 구현·실모델 재검증 완료 | 슬라이드 5 `시간축 평가` / 슬라이드 9 `held-out 최종 평가 지표` | 2단계에서 만든 CLIP 기반 temporal 지표를 OWLv2/VQA 보강 기준으로 다시 계산한다 | 신규 `pipelines/heldout_remeasurement.py`, `scripts/remeasure_video_metrics.py`, `scripts/batch_remeasure_owlv2_vqa_10videos.py` | 한계 1·3, 슬라이드 5·9 | 10~11월 | `PTC`/`SFR`/`SDI` 결과가 검출기 보강 전후로 비교됨 — clip_only vs calibrated 재측정 파이프라인과 10개 영상 batch가 동작하고 `summary_metrics.csv/md`가 생성됨 |
| 10 ✅ 코드 기반 스캐폴드 | 슬라이드 9 `Temporal SRS Calibration` / `held-out 최종 평가 지표` | GT 객체 주석과 VLM 판단을 이용해 SRS/Temporal SRS의 가중치를 보정한다 | 신규 `evaluators/temporal_srs_calibration.py` | 한계 3, 슬라이드 9 | 10~11월 | loop-internal 지표와 held-out 최종 지표가 분리되어 보고됨 — **완료(코드 기반 스캐폴드, 5차)**: 가중치 설정 로드/저장 + least-squares weight-fitting 함수 스텁이 동작(mock/synthetic target score 기준). 🟡 **실제 GT 주석이나 VLM judge 호출은 없음** — 실 데이터 연결은 후속 |
| 11 ✅ PoC 구현 완료 | 슬라이드 10 `1차 — 채널 심볼 절감 PoC` | 변화가 작은 latent/semantic 요소를 덜 보내고, 의미 보존 저하와 절감률의 관계를 본다 | 신규 `accounting/bit_accounting.py`, `pipelines/transmission_accounting.py` | 슬라이드 10·12 | 7~8월 | 절감률 vs SRS/PTC 곡선이 생성됨 — **완료(PoC, 6차)**: frame/segment 단위 bit/channel-symbol accounting + naive baseline 대비 절감률이 계산됨. 🟡 실제 bitstream/CBR 재현이 아니라 accounting PoC (아래 "6차 구현 결과" 참조) |
| 12 ✅ PoC 구현 완료 | 슬라이드 10 `2차 — 비트 기준 설계안` / 슬라이드 12 `평가 벤치마크` | 실제 bitrate/CBR 산정 방식, adaptive keyframe policy, DISTS/downstream 비교 지표를 정리한다 | 신규 `pipelines/rate_reliability_report.py`, `scripts/report_transmission_accounting.py` | 슬라이드 10·12·13 | 10~11월 | 최종 보고용 비교 프로토콜과 bit 기준 설계안이 정리됨 — **완료(PoC, 6차)**: rate/reliability trade-off 리포트(bits_per_frame/symbols_per_frame + PTC/SFR/SDI/severity) + append/merge 기반 비교 프로토콜이 동작. 🟡 실제 CBR/표준 bitstream 비교와 DISTS/downstream 지표 통합은 후속 |

PPT 블록 기준 큰 흐름은 다음과 같다.

| PPT 블록 묶음 | 포함 순서 | 의미 |
|---|---|---|
| 슬라이드 6·9 검증/평가 블록 | 0, 7~10 | 비디오 지표와 verifier가 의존할 최소 판정 기준을 맞추고, 최종 평가는 held-out으로 분리한다 |
| 슬라이드 5 비디오 확장 블록 | 1~6 | mp4 입출력, 시간축 지표, motion-aware 판단, segment 구조, generate 분기를 만든다 |
| 슬라이드 10 전송량 절감 블록 | 11~12 | ETRI 문의사항인 channel-symbol 절감과 bit 기준 설계안을 정리한다 |

**게이트 원칙:** 수치에 영향 없는 순수 배선(threshold 전달 등)은 게이트 불필요. 새 판정
로직, 새 backend(OWLv2·VQA), negative-prompt 재생성, generate 분기, 학습형 adapter/critic은
phase/config gate 뒤에 두고, 기본값에서 원본 SGD-JSCC 경로와 동일하게 동작하도록 관리한다.

## 1차~6차 구현 결과

### 1차 구현 결과 (2026-07)

> 상세 검증 기록(기준 커밋, 로컬/원격/실모델 실행 결과, 산출물)은
> [etri_stage1_validation.md](../etri_stage1_validation.md) 참조.

1차(순서 0~4)는 구현 완료됐다. 1차의 성격은 **최종 평가 체계 완성이 아니라 동작하는
비디오/시간축 평가 뼈대 확보**이며, OWLv2/VQA·Generate·Adapter/Critic·Temporal SRS
Calibration·bit accounting은 계획대로 후속 단계(2~6차)로 남겨뒀다.

구현 내용과 산출물 ↔ 코드 대응:

| 항목 | 코드 | 산출물 |
|---|---|---|
| Presence threshold 배선 + uncertain band(히스테리시스) | `evaluators/object_preservation.py`, `hallucination.py`, `semantic_reliability.py`, `pipelines/eval_pipeline.py::EvalContext`, `scripts/evaluate.py` — config 키 `object_presence_threshold` / `object_presence_uncertain_band` (band 기본 0.0 = 기존 결과 불변) | threshold 변경이 지표에 실제 반영 (`tests/test_evaluators.py::TestPresenceThresholdWiring`) |
| mp4/frame 비디오 IO | 신규 `utils/video_io.py` (cv2 → ffmpeg CLI 백엔드 자동 선택), `scripts/evaluate_video.py` mp4 입력 + `--save-video` | 복원 frame folder (`video_io.recon_frames_dir`) + 복원 mp4 (`video_io.recon_video`), mp4→frames 추출본 |
| `PTC`/`SFR`/`SDI` 시간축 의미 지표 | `evaluators/temporal_consistency.py` — packet consistency 유지(PTC), 원본 변화를 제외한 spurious birth/death 비율(SFR), keyframe 거리 대비 drift 기울기(SDI) | `temporal_metrics.csv`에 `ptc`/`sfr`/`sdi` 기록 |
| 의미 델타 + 모션 이중 게이트 | `video/temporal_pipeline.py` — keyframe-anchored `motion_residual` 점수, `temporal.motion_threshold`/`motion_weight`/`motion_grid`(+`semantic_delta_threshold` alias). 기본 `motion_threshold: null` = 기존 semantic-delta 단독 게이트와 동일 | `temporal_frames.csv`의 per-frame `decision`(`reuse`/`recompute_semantic`/`recompute_motion`)·`motion_score` = motion gate decision log, summary의 `n_recompute_motion` |
| Segment(GOP) 추상화 | 신규 `video/segment.py::SegmentRecord`/`build_segments` — segment_id, keyframe/inter 인덱스, frame_decisions, transmitted_units, delta/motion/temporal 요약. `generation` 필드는 3차 generate 분기용 예약 인터페이스(1차에서는 항상 null) | `segments.json` (frame-wise 로그는 그대로 유지, 병행 출력) |

주의: 1차의 `PTC`/`SFR`/`SDI`와 presence 판정은 처음에는 **CLIP/packet 기반 잠정 구현**이었다
(위 "후속 단계가 앞 단계의 잠정 구현을 마무리하는 방식" 표 참조). 2026-07-28 기준
5차 OWLv2/VQA 보강 후 10개 영상 held-out 재측정을 완료했으므로, 최종 해석에서는
`outputs/etri_video_eval/remeasure_10videos/summary_metrics.csv`의 filtered 모드를 기준으로 삼는다.

테스트 현황(정확한 표현): 1차 변경 관련 테스트(`tests/test_video.py`,
`tests/test_video_io.py`, `tests/test_evaluators.py`)는 통과. 전체 스위트 중
`tests/test_ddp.py::test_entrypoint_torchrun_dryrun` 1건은 1차 변경과 무관하게
**이전부터 실패**하던 테스트다(로그 문구 `"DDP: rank="` 기대와 실제 train.py 출력
`"DDP: world_size=…"` 불일치 — 별도 수정 대상).

### 2차 구현 결과 (2026-07)

2차(순서 7)는 구현 완료됐다. 2차의 성격은 **"복원이 전송 semantic packet과 맞는지
검증하고, 오류 유형별로 어떤 조치가 필요한지 결정·기록하는 기반을 만드는 것"**이며,
**최종 hallucination 문제 해결이 아니다.** OWLv2/VQA 기반 검출 보강(5차),
`video_generator`/Generate 분기 결합(3~4차 후속), 학습형 `Semantic Packet Fidelity
Adapter`·`Counterfactual Hallucination Critic`, 실제 diffusion sampler로의
negative-prompt/prompt-emphasis 주입은 계획대로 이번 범위에서 제외했다.

구현 내용과 산출물 ↔ 코드 대응:

| 항목 | 코드 | 산출물 |
|---|---|---|
| Packet Verifier(wrapper/service) | 신규 `evaluators/packet_verifier.py` — 기존 `evaluators/semantic_packet_matcher.compare()`(추가/누락/관계/속성/장면 오류 분리는 이미 Phase 4-A에서 구현됨)를 재사용하고, 하나의 severity 스칼라(`[0, 1]`, 오류가 커질수록 증가)를 덧붙임 | `PacketVerifier.verify()` → dict/JSON 직렬화 가능한 report (`missing_objects`/`additional_objects`/`relation_errors`/`attribute_errors`/`scene_match`/`severity`/`item_id`) |
| 오류 유형별 controller decision | 신규 `controllers/verifier_controller.py` — `VerifierController`가 severity + 오류 유형(추가/누락/관계·속성·장면)을 보고 `accept`/`suppress_extra`/`strengthen_missing`/`strengthen_structure_guidance`/`fallback_recompute`/`keyframe_fallback` 중 하나를 결정. negative prompt/prompt emphasis는 **candidate_actions로 로그만 남기고 실제 sampler에는 주입하지 않음**(2차 범위 제약) | `ControllerDecision.to_dict()` → `controller_decision`/`severity`/`triggered_modes`/`candidate_actions`/`reason` |
| Pipeline 연결(옵션, 기본 OFF) | 신규 `pipelines/packet_verification.py` (`maybe_run`), `scripts/evaluate_video.py`에서 `TemporalPipeline.run()` 결과 직후 호출 | config 게이트 `use_packet_verifier`(phase4-gated) + `verifier.enabled` — 기본 둘 다 false. 켜면 `temporal_frames.csv`에 `severity`/`controller_decision` 컬럼이, `segments.json`의 각 segment에 `verifier_summary`(`mean_severity`/`max_severity`/`decision_counts`/`worst_decision`)가 추가됨 |
| 출력 파일 | `pipelines/packet_verification.py::write_reports` | `packet_match_report.json`/`.csv`, `controller_decisions.json`/`.csv` (경로는 `configs/base/video/default.yaml`의 `verifier.report_json`/`report_csv`/`decisions_json`/`decisions_csv`) |

주의: 2차 controller는 여전히 **rule-based decision log 수준**이며, 오류 유형 판정은
1차와 마찬가지로 CLIP/캡션 기반 packet 추출기(`guidance/semantic_packet_extractor.py`)에
의존한다(위 "후속 단계가 앞 단계의 잠정 구현을 마무리하는 방식" 표의 "2차 `Packet
Verifier`와 regeneration controller" 행 참조). 강한 결론을 내리기 전에 5차 OWLv2/VQA
보강 후 재검증해야 한다.

테스트 현황: 2차 신규/확장 테스트(`tests/test_packet_matcher.py::TestPacketVerifier`,
`tests/test_controllers.py::TestVerifierController`,
`tests/test_video.py::TestPacketVerifierWiring`)와 기존 회귀 테스트
(`tests/test_packet_matcher.py`, `tests/test_regeneration_search.py`,
`tests/test_controllers.py`, `tests/test_video.py`, `tests/test_video_io.py`,
`tests/test_evaluators.py`)는 `ptest` conda 환경에서 통과 확인됨(정확한 pass/fail
수치는 저장소의 최신 테스트 실행 로그 참조 — 이 문서는 스냅샷이 아니라 구현 범위
설명이다).

### 3차 구현 결과 (2026-07)

3차(순서 5)는 구현 완료됐다. 3차의 성격은 **"`TemporalPipeline`에 reuse/recompute와
나란히 동작하는 세 번째 branch(generate)를 구조적으로 통과시키는 것"**이며,
**고성능 생성 모델 완성이나 LGVSC 수준 생성 품질 달성이 아니다.** 이번 범위의
generator backend(`copy`/`interpolation`)는 모두 mock이고, 실제 SVD/Open-Sora 같은
학습형 비디오 생성 모델 통합, bidirectional(start+end keyframe) conditioning,
OWLv2/VQA 기반 생성 결과 최종 검증은 계획대로 각각 후속(3~4차 이후, 4차, 5차)이다.

구현 내용과 산출물 ↔ 코드 대응:

| 항목 | 코드 | 산출물 |
|---|---|---|
| `video_generator` 인터페이스 | 신규 `video/video_generator.py` — `GenerationRequest`(start keyframe recon/index, target index, segment context, caption, packet, side-info, Rx-legal 경계가 분리된 reference 필드들, 4차 예약용 `end_keyframe_recon`) / `GenerationMetadata`(dict/JSON 직렬화) / `GenerationResult`. `VideoGenerator` 베이스 클래스 + `_BACKENDS` 레지스트리로 실제 backend(SVD/Open-Sora)를 나중에 꽂을 수 있는 확장점을 열어둠 | `CopyGenerator`(keyframe recon 그대로 복사), `InterpolationGenerator`(keyframe recon과 참조 프레임의 선형 보간) — 둘 다 `mock=True` |
| Rx-legal 경계 분리 | `InterpolationGenerator`는 기본적으로 `reference_prev_recon`(수신단이 실제로 가진 이전 복원)만 사용. `reference_target_frame`(원본 target — 평가 시 사용하면 데이터 누수)은 `allow_ground_truth_reference=True`를 명시적으로 켠 경우에만, 그리고 그 사실이 `notes`에 남는 경우에만 사용됨 | `tests/test_video_generator.py::TestInterpolationGenerator`(ground-truth reference가 기본값에서 무시됨을 검증) |
| reuse / recompute / generate 3-way 판단 | `video/temporal_pipeline.py::TemporalPipeline` — 기존 reuse 이중 게이트(semantic delta + motion)를 통과하지 못한 inter-frame 중, semantic delta가 `[generate_delta_min, generate_delta_max]`(기본값: `reuse_threshold` ~ `3×reuse_threshold`) 안에 있고 motion이 `generate_motion_max`(기본값: `motion_threshold`, motion gate 자체가 꺼져 있으면 무제한) 이하이면 `generate`, 그 외에는 기존과 동일하게 `recompute_semantic`/`recompute_motion` | `FrameRecord.decision == "generate"`, `TemporalPipeline._summarize()`의 `n_generate` |
| start-only generation 경로 | `TemporalPipeline._generate_frame()` — GOP의 start keyframe recon만 조건으로 사용(`conditioning_mode="start_only"`). 3차 시점에는 bidirectional(`GenerationRequest.end_keyframe_recon`/`video_generator.conditioning_mode: bidirectional`)을 `NotImplementedError`로 막아 4차 확장점으로 예약 — **4차에서 실제로 구현됨** (아래 "4차 구현 결과" 참조) | `tests/test_video_generator.py::TestStartOnlyBackendsRejectEndKeyframe`(start-only backend는 여전히 거부) |
| Segment 연결 | `video/segment.py::build_segments()`의 `_generation_summary()` — segment 내 generate 프레임을 집계(개수/target_indices/backend/conditioning_mode/mock 여부 + 프레임별 상세). generate가 없으면(기본값) `SegmentRecord.generation`은 1차와 동일하게 `None` | `segments.json`의 `generation` 필드 |
| Pipeline 연결(옵션, 기본 OFF) | `scripts/evaluate_video.py` — `use_video_gen`(phase4-gated) + `video_generator.enabled` 둘 다 true일 때만 `video_generator.build_generator(cfg)`로 backend를 만들고 `TemporalPipeline`에 주입. 끄면 `TemporalPipeline`은 `video_generator`를 만들지도, 호출하지도 않음 | `configs/base/video/default.yaml`의 `use_video_gen`/`video_generator.*` 블록 (기본 전부 OFF) |
| 출력 산출물 | `video/video_generator.py::save_generated_frames()` — `decision == "generate"`인 프레임만 별도 폴더에 저장(전체 복원 프레임은 기존과 동일하게 `recon_frames_dir`에도 저장됨) | `generated_frames_dir` 아래 `generated_{index:05d}.png`, `temporal_frames.csv`의 `decision=generate` 행, `temporal_metrics.csv`의 `n_generate`/`n_reused`/`n_recompute_semantic`/`n_recompute_motion` |

실제 CLI 경로(`scripts/evaluate_video.py --no-models --captions ...`)로 합성 4프레임
시퀀스(캡션으로 오브젝트 수를 다르게 부여)를 돌려 `n_generate=2`, `generated_frames_dir`에
2개 파일 저장, `segments.json`의 `generation` 필드 채워짐을 수동으로도 확인했다(단위
테스트와 별개의 엔드투엔드 스모크 확인).

주의: 3차의 generate 결과는 **품질을 주장하는 산출물이 아니다** — `copy`/`interpolation`
mock backend는 생성이 아니라 자리표시자다. 실제 생성 품질 비교, drift/flicker 측정,
OWLv2/VQA 기반 검증은 4~5차의 대상이다(아래 "후속 단계가 앞 단계의 잠정 구현을 마무리하는
방식" 표 참조).

테스트 현황: 3차 신규 테스트(`tests/test_video_generator.py`,
`tests/test_video.py::TestGenerateBranch`)와 기존 회귀 테스트
(`tests/test_video.py`, `tests/test_video_io.py`, `tests/test_evaluators.py`,
`tests/test_packet_matcher.py`, `tests/test_controllers.py`)는 `ptest` conda
환경에서 통과 확인됨(정확한 pass/fail 수치는 저장소의 최신 테스트 실행 로그 참조).

### 4차 구현 결과 (2026-07)

4차(순서 6)는 구현 완료됐다. 4차의 성격은 **"start keyframe만 쓰는 3차 generate
branch와 start+end keyframe을 함께 쓰는 bidirectional branch를 같은 파이프라인에서
구조적으로 비교 가능하게 만드는 것"**이며, **실제 SVD/Open-Sora 품질 검증이나
"bidirectional이 drift/flicker를 줄인다"는 성능 주장이 아니다.** 4차의
bidirectional backend(`bidirectional_interpolation`)도 3차와 마찬가지로 mock이다.

구현 내용과 산출물 ↔ 코드 대응:

| 항목 | 코드 | 산출물 |
|---|---|---|
| bidirectional `GenerationRequest` 확장 | `video/video_generator.py` — 3차에서 예약만 해뒀던 `end_keyframe_recon`을 `end_keyframe_index`와 함께 실제로 사용. `GenerationMetadata`에 `end_keyframe_index`/`relative_position` 필드 추가(start-only 결과에서는 항상 `None` — motion-gate 컬럼과 동일한 "항상 존재, 기본은 None" 컨벤션) | `GenerationRequest.end_keyframe_recon`/`end_keyframe_index`, `GenerationMetadata.to_dict()` |
| bidirectional mock backend | 신규 `BidirectionalInterpolationGenerator` — `relative_position = (target_index - start_keyframe_index) / (end_keyframe_index - start_keyframe_index)`로 시작/끝 keyframe recon을 선형 보간. 두 keyframe recon 모두 수신단이 실제로 복원한 것이므로 **Rx-legal**(3차 `InterpolationGenerator`의 ground-truth reference와 달리 mock 표시는 "학습형 생성이 아님"을 뜻할 뿐, 데이터 누수가 아님) | `tests/test_video_generator.py::TestBidirectionalInterpolationGenerator` |
| end keyframe 누락/범위 정책 | `video_generator.bidirectional_missing_end_policy`: `error`(기본, 명확한 예외) / `fallback_start_only`(해당 프레임만 `conditioning_mode="start_only"`로 강등, `notes`에 강등 사실 기록) — end keyframe이 없거나(마지막 GOP) `target_index`가 `[start, end]` 밖이면 적용 | `tests/test_video_generator.py`의 missing-end / out-of-range 테스트 |
| `TemporalPipeline`에 end keyframe recon 전달 | `TemporalPipeline._prepass_keyframe_recons()` — bidirectional 모드에서만(2안: prepass 추가) 모든 keyframe의 recon을 미리 계산·캐시하고, 메인 루프의 keyframe 분기는 그 캐시를 재사용(중복 계산 없음, `reconstruct_fn` 호출 횟수는 start-only와 동일하게 keyframe당 1회). 각 GOP의 inter-frame 생성 요청에는 **다음 GOP의 keyframe recon**을 end keyframe으로 전달. start-only 모드(`conditioning_mode` 기본값)는 이 prepass를 전혀 타지 않아 3차 단일 패스 동작이 그대로 유지됨 | `tests/test_video.py::TestBidirectionalGenerateBranch::test_keyframes_not_double_reconstructed`(재계산 없음 검증), `test_end_keyframe_recon_reaches_generator`(실제 전달 검증) |
| start-only vs bidirectional 비교 파이프라인 | 신규 `pipelines/generation_mode_comparison.py` — 동일 프레임 시퀀스를 `pipeline_factory`로 주입된 start-only/bidirectional 두 `TemporalPipeline`에 각각 통과시키고 `PTC`/`SFR`/`SDI`/`n_generate`/`n_reused`/`n_recompute_semantic`/`n_recompute_motion`을 diff. `scripts/evaluate_video.py`의 `video_generator.comparison_enabled`(기본 false)로 연결 — 켜면 2배 연산(두 모드 모두 실행) | `generation_mode_comparison.json`, `temporal_metrics_start_only.csv`, `temporal_metrics_bidirectional.csv` |
| 산출물 기록 | `FrameRecord.to_log()`에 `generation_conditioning_mode` 컬럼 추가(생성이 아니면 `None` — 모션 게이트 컬럼과 동일한 컨벤션), `segment.py::_generation_summary()`에 `end_keyframe_index` 집계 필드 추가 | `temporal_frames.csv`의 `generation_conditioning_mode` 컬럼, `segments.json`의 `generation.end_keyframe_index` |
| config | `configs/base/video/default.yaml`의 `video_generator.conditioning_mode: bidirectional` 허용 + `bidirectional_missing_end_policy`/`comparison_enabled`/`comparison_output`/`comparison_start_only_csv`/`comparison_bidirectional_csv` 추가(기본값 전부 기존 동작 보존 — `conditioning_mode: start_only`, `comparison_enabled: false`) | `config.py::_NESTED_PATH_KEYS`에 비교 산출물 경로 3종 등록 |

실제 CLI 경로(`scripts/evaluate_video.py --no-models --captions ...`)로 합성 8프레임
2-GOP 시퀀스(마지막 GOP는 end keyframe 없음)를 `conditioning_mode: bidirectional` +
`comparison_enabled: true`로 돌려 다음을 수동으로도 확인했다: 첫 GOP의 inter-frame이
진짜 bidirectional 보간(`relative_position=0.25`, `end_keyframe_index=4`)으로 기록됨,
마지막 GOP의 inter-frame은 `bidirectional_missing_end_policy: fallback_start_only`에
따라 `conditioning_mode=start_only`로 강등되어 기록됨, `generation_mode_comparison.json`
+ 두 개의 `temporal_metrics_*.csv`가 생성됨, generate를 아예 끈 기본 config
(`configs/recipes/video/composed_video.yaml`)는 동일 입력에서 `n_generate=0`으로 3차 이전과 동일하게
동작함(회귀 없음).

주의: 4차의 bidirectional 결과도 3차와 마찬가지로 **품질을 주장하는 산출물이 아니다** —
`bidirectional_interpolation`은 학습형 생성이 아니라 선형 보간 mock이다. 비교 파이프라인이
계산하는 `PTC`/`SFR`/`SDI` diff는 "두 모드를 같은 파이프라인에서 실행·비교할 수 있다"는
구조적 사실만 증명하며, "bidirectional이 더 낫다"는 근거로 인용해서는 안 된다. 실제
생성 품질·drift/flicker 감소 검증은 실제 backend 통합 이후(5차+) 대상이다(아래 "후속
단계가 앞 단계의 잠정 구현을 마무리하는 방식" 표 참조).

테스트 현황: 4차 신규 테스트(`tests/test_video_generator.py`의
`TestBidirectionalInterpolationGenerator`/`TestBidirectionalConditioningModeBuild`,
`tests/test_video.py`의 `TestBidirectionalGenerateBranch`/`TestGenerationModeComparison`)와
기존 회귀 테스트(`tests/test_video_generator.py`, `tests/test_video.py`,
`tests/test_video_io.py`, `tests/test_evaluators.py`, `tests/test_packet_matcher.py`,
`tests/test_controllers.py`)는 `ptest` conda 환경에서 통과 확인됨(정확한 pass/fail
수치는 저장소의 최신 테스트 실행 로그 참조).

### 5차 구현 결과 (2026-07)

5차(순서 8~10)는 먼저 **"CLIP-only verifier에 OWLv2/VQA/GT/VLM judge를 꽂을 수
있는 인터페이스와, 1~4차 결과를 재측정할 수 있는 held-out 파이프라인"**을 만들었고,
이후 실제 GPU recon frame을 재사용해 **10개 영상 × 5개 모드 실모델 재측정까지 완료**했다.
따라서 OWLv2/VQA presence calibration의 실제 weight 연결과 `PTC`/`SFR`/`SDI` held-out
재검증은 완료 처리한다. 단, `TemporalSRSCalibration`은 여전히 가중치 fitting
스캐폴드이며 실제 VLM judge 기반 가중치 보정 결과는 후속이다.

구현 내용과 산출물 ↔ 코드 대응:

| 항목 | 코드 | 산출물 |
|---|---|---|
| Presence backend 인터페이스 | 신규 `evaluators/presence_backends.py` — `PresenceResult`(object_name/present/confidence/backend/evidence) + `PresenceBackend` 공통 인터페이스. `MockPresenceBackend`(packet 기반, 의존성 없음), `ClipPresenceBackend`(기존 CLIP 판정을 인터페이스 뒤로 감쌈, 재구현 아님), `Owlv2PresenceBackend`/`VqaPresenceBackend`(lazy-load, 미설치 시 `PresenceBackendUnavailableError`로 명확히 실패 — VQA는 기존 `vqa_backend.py`/`hallucination_vqa.py` 재사용), `GtPresenceBackend`(주석 dict 조회) | `build_presence_backend()` 레지스트리; 테스트는 `mock`/`gt`/stub만 사용, 실제 OWLv2/VQA weight는 테스트 의존성 아님 |
| Presence ensemble / calibration | 신규 `evaluators/presence_calibration.py` — `PresenceCalibrator`가 `clip_only`/`owlv2_only`/`vqa_only`/`gt_only`/`ensemble_majority`/`ensemble_weighted` 6개 모드로 backend 결과를 조합. 기본값 `clip_only`(비활성 시 아무 backend도 안 만듦) | `CalibratedPresence`(final_present/final_confidence/contributing_backends/per_backend, dict/JSON 직렬화) |
| Packet Verifier 보강 | `evaluators/packet_verifier.py` — `PacketVerifier(presence_calibrator=None, metric_role="loop_internal")`. **기본값(`presence_calibrator=None`)에서는 결과가 2~4차와 완전히 동일**. calibrator + `reconstructed_image`가 모두 주어졌을 때만 missing/additional 판단을 재검증해 보정. report에 `raw_clip_result`(원본 스냅샷)와 `calibrated_presence_result`(보정 근거, 미보정 시 `None`)를 분리 저장, 모든 report에 `metric_role`(`loop_internal`/`held_out`) 태그 | `pipelines/packet_verification.py`가 `verifier.use_presence_calibration`(기본 false)로 연결 — 켜면 `packet_match_report`에 3개 컬럼(`metric_role`/`raw_clip_result`/`calibrated_presence_result`)이 추가되고, 실제 재구성 이미지가 calibrator에 전달됨 |
| Held-out 재측정 파이프라인 | 신규 `pipelines/heldout_remeasurement.py` — `items_from_temporal_records()`(TemporalPipeline 결과, 이미지 포함 → 모든 backend로 재검증 가능)/`items_from_saved_packets()`(디스크에 저장된 packet JSON 쌍, 이미지 없음 → `mock`/`gt`만 가능)/`items_from_recon_frame_dirs()`(완료된 실모델 run의 `extracted_frames`+`recon_frames` 재사용, 이미지 기반 OWLv2/VQA 검증 가능) 입력 경로. `PacketVerifier`/`PresenceCalibrator`가 `image=None`이어도 image-free backend는 정상 동작하도록 배선되어 있고, `RemeasurementItem.gt_metadata`가 실제로 `gt` backend에 전달됨(각 backend의 `check()`가 `gt_metadata`를 받는 공통 시그니처로 통일). `remeasure()`가 clip_only/calibrated 양쪽으로 `PacketVerifier` report + `PTC`/`SFR`/`SDI`(`evaluators/temporal_consistency.py` 재사용)를 계산하고 `metric_delta`(항목별 clip_only/calibrated/diff)를 생성 | `clip_only_metrics.json/csv`, `calibrated_metrics.json/csv`, `metric_delta.json/csv` — `scripts/remeasure_video_metrics.py`(`--from-run`/`--from-packets`/`--from-recon-frames`)와 `scripts/batch_remeasure_owlv2_vqa_10videos.py`(10개 영상 × 5개 모드 batch, summary CSV/MD) |
| Temporal SRS Calibration 스캐폴드 | 신규 `evaluators/temporal_srs_calibration.py` — `TemporalSRSCalibration`(SRS/temporal-SRS 가중치 로드/저장), `fit_weights_least_squares()`(GT/VLM `target_score`가 주어졌을 때의 least-squares weight fitting — **실제 VLM 호출은 없음**, 호출자가 target을 공급) | `save()`/`load()` JSON 포맷; `tests/test_temporal_srs_calibration.py`에서 synthetic target으로 정확한 weight 복원 확인 |
| config | `configs/base/video/default.yaml`의 `verifier.use_presence_calibration`/`presence_mode`/`presence_backends`/`presence_backend_weights`/`metric_role`, 신규 `heldout.*`, `temporal_srs_calibration.*` 블록 (기본값 전부 OFF/`clip_only`) | `config.py::_NESTED_PATH_KEYS`에 `heldout.*`/`temporal_srs_calibration.weights_output` 등록 |

실제 CLI 경로(`scripts/remeasure_video_metrics.py --from-run --no-models --captions ...`,
`--from-packets <dir>`, `--from-recon-frames <run_dir>`)로 합성/실모델 프레임 시퀀스를 돌려
`clip_only_metrics`/`calibrated_metrics`/`metric_delta` 파일이 생성되고, presence
calibration을 켠 `evaluate_video.py` 실행에서 `packet_match_report`에 새 컬럼이
추가되는 것을 확인했다. 최종 실모델 재측정은 `outputs/etri_video_eval_real_full_step50`
의 10개 baseline run을 재사용해 `outputs/etri_video_eval/remeasure_10videos/`에 저장했다.

10개 영상 × 5개 모드 batch 결과(`summary_metrics.csv`, 50 rows, 모든 `n_items=100`):

| mode | severity diff | PTC diff | SFR diff | missing diff | additional diff | hallucination diff |
|---|---:|---:|---:|---:|---:|---:|
| `owlv2` | -0.1300 | +0.2036 | -0.0337 | -211.6 | -18.6 | -0.0314 |
| `vqa` | -0.2065 | +0.3337 | -0.0219 | -364.6 | -14.9 | -0.0253 |
| `ensemble_nofilter` | -0.2065 | +0.3337 | -0.0219 | -364.6 | -14.9 | -0.0253 |
| `ensemble_gt_filter` | -0.1872 | +0.3120 | -0.0182 | -72.4 | +0.0 | +0.0000 |
| `ensemble_openworld_filter` | -0.1857 | +0.2875 | -0.0190 | -169.4 | -14.9 | -0.0598 |

해석 기준은 모드별로 분리한다. `ensemble_gt_filter`는 GT에 명시된 object만 남기는
closed-world object preservation 평가이며, hallucination/additional 주장의 근거로 쓰지
않는다. `ensemble_openworld_filter`는 count/action/scene 잡음만 제거하고 non-GT object를
남기므로 hallucination/additional object 분석에 사용한다. `ensemble_nofilter`는
`one`/`walking`/`sidewalk` 같은 caption-token 잡음이 섞이는 비교용 baseline일 뿐 최종
주장의 근거로 쓰지 않는다. 현재 설정에서는 `vqa`와 `ensemble_nofilter`가 같은 집계값을
보여 ensemble이 VQA 판단에 강하게 지배됨을 함께 명시한다.

테스트 현황: 5차 신규 테스트(`tests/test_presence_backends.py`,
`tests/test_heldout_remeasurement.py`, `tests/test_temporal_srs_calibration.py`,
`tests/test_packet_matcher.py::TestPacketVerifierPresenceCalibration`,
`tests/test_video.py::TestPresenceCalibrationWiring`)와 기존 회귀 테스트
(`tests/test_video_generator.py`, `tests/test_video.py`, `tests/test_video_io.py`,
`tests/test_evaluators.py`, `tests/test_packet_matcher.py`, `tests/test_controllers.py`)는
`ptest` conda 환경에서 통과 확인됨(정확한 pass/fail 수치는 저장소의 최신 테스트
실행 로그 참조).

### 6차 구현 결과 (2026-07)

6차(순서 11~12)는 **PoC 구현 완료**됐다. 6차의 성격은 **"ETRI가 문의한 'semantic
unit 절감뿐 아니라 channel-symbol/bit 절감도 되는가'라는 질문에 코드로 답할 수
있는 accounting 구조와 비교 프로토콜을 만드는 것"**이며, **실제 통신 표준
수준의 bitstream/CBR 구현이 아니다.** 이 모듈에는 엔트로피 코더도, 변조/부호율
모델도, 실제 channel-coded side-info 스트림도 없다 — 모든 숫자는 (a) 이 저장소가
이미 만들어내는 실제 산출물(패킷 JSON, caption 문자열)의 정확한 byte 길이이거나
(b) 실제 모델 아키텍처 상수(VAE `z_channels`/downsample, Canny 채널 인코더
출력 차원)에서 유도한 명시적 proxy다. 모든 component는 `proxy: true/false`
플래그와 근거 note를 함께 남긴다(`accounting/bit_accounting.py` 모듈
docstring 참조).

구현 내용과 산출물 ↔ 코드 대응:

| 항목 | 코드 | 산출물 |
|---|---|---|
| Bit/channel-symbol accounting 계산기 | 신규 `accounting/bit_accounting.py` — `keyframe_visual_latent_symbols`/`edge_side_info_symbols`/`caption_bits`/`semantic_packet_bits`/`motion_side_info_bits`/`generated_frame_symbols`/`reused_frame_symbols`/`recompute_frame_symbols` 8개 component. `caption_bits`/`semantic_packet_bits`는 실제 UTF-8 byte 길이(proxy 아님); `keyframe_visual_latent_symbols`(VAE `z_channels=16`, downsample=8, 128px patch → patch당 4096 latent element)/`edge_side_info_symbols`(Canny 채널 인코더 출력 320/4096 비율 proxy)/`motion_side_info_bits`(quantized block-map proxy)는 명시적 proxy | `Component.to_dict()`(`value`/`unit`/`proxy`/`note`), `TransmissionAccountingRecord` |
| 결정(decision)별 payload 모델 | `account_frame()` — `keyframe`/`recompute_semantic`/`recompute_motion`은 전체 visual latent + edge + packet을, `reuse`는 아무것도, `generate`는 caption + motion side-info만 전송한다고 가정(mock generate backend가 이미 받은 keyframe 복원 + caption/side-info만 조건으로 쓰는 것과 일치, `video/video_generator.py` 참조) | `TransmissionAccountingRecord.components`(8개 필드 모두 항상 존재, 해당 없는 항목은 0) |
| Naive baseline 2종 | `compute_baseline_record()` — `naive_full_frame_packet`(모든 프레임을 keyframe처럼 전체 전송한다고 가정, 상한선) / `keyframe_only_lgvsc_style`(실제 keyframe만 전체 전송, 나머지는 항상 side-info-only라고 가정, LGVSC 스타일 하한 근사) — 둘 다 이 프레임의 실제 packet/shape 데이터로 계산(전역 상수 아님) | `BASELINE_METADATA`(각 baseline의 가정과 `not_a_real_cbr: true` 표시) |
| Frame/segment/summary 파이프라인 | 신규 `pipelines/transmission_accounting.py::account_transmission()` — `TemporalPipeline.run()` 결과를 읽기만 하고 변경하지 않음(`result`/`FrameRecord` 불변, 테스트로 확인). semantic-unit 절감(`transmitted_units`/`naive_units`/`overhead_reduction`, 1~4차가 이미 계산)은 그대로 통과시키고, bit/channel-symbol 절감만 새로 계산해 **두 축을 분리 유지** | `frame_accounting.json/csv`, `segment_accounting.json/csv`, `accounting_summary.json`(`total_bits`/`total_channel_symbols`/`total_semantic_units`/`baseline_*`/`bit_reduction`/`symbol_reduction`/`semantic_unit_reduction`/`proxy_fraction`) |
| Rate/reliability trade-off 리포트 | 신규 `pipelines/rate_reliability_report.py` — accounting summary + `PTC`/`SFR`/`SDI`(`evaluators/temporal_consistency.py`) + packet verifier `mean_severity`를 한 행으로 결합. `append_rate_reliability_row()`/`merge_rate_reliability_curves()`로 여러 run(config/policy/SNR)의 점을 하나의 곡선 CSV로 누적(label 기준 dedup) | `rate_reliability_summary.json`, `rate_reliability_curve.csv` |
| Pipeline 연결(옵션, 기본 OFF) | `scripts/evaluate_video.py` — `accounting.enabled`/`rate_reliability.enabled`(둘 다 기본 false). 켜도 `TemporalPipeline`의 재구성/결정/기존 지표는 전혀 바뀌지 않고 산출물만 추가됨(수동 회귀 확인: 동일 입력에서 accounting 끈 실행과 켠 실행의 `temporal_metrics.csv`/`segments.json`이 byte-동일) | `configs/base/video/default.yaml`의 `accounting.*`/`rate_reliability.*` 블록, `config.py::_NESTED_PATH_KEYS`에 경로 등록(OmegaConf `${accounting.output_dir}` 보간 사용) |
| 독립 CLI | 신규 `scripts/report_transmission_accounting.py` — `--input`(from-run, 처음부터 재계산) / `--from-accounting-summary`(기존 `accounting_summary.json` + `temporal_metrics.csv` + 선택적 `packet_match_report.json`을 읽어 rate/reliability 리포트만 재생성, 재계산 없음) 두 모드 | 위와 동일 산출물; 재생성 모드는 원본 실행과 동일한 숫자를 재현함을 스모크로 확인 |

실제 CLI 경로(`scripts/evaluate_video.py --no-models --captions ...`, `accounting.enabled`/
`rate_reliability.enabled: true`)로 합성 6프레임 시퀀스 두 종류를 돌려 확인했다:
(1) 프레임마다 독립적인 랜덤 노이즈 입력 → 전부 keyframe으로 판정되어
`bit_reduction`/`symbol_reduction` 모두 0(naive baseline과 동일 — 절감할 게 없으니
0이 나오는 것 자체가 올바른 sanity-check), (2) 거의 동일한 프레임(노이즈만 미세하게
다름) + 동일 caption → `n_reused=5`, `semantic_unit_reduction`/`bit_reduction`/
`symbol_reduction` 모두 `0.833`으로 **세 지표가 서로 다른 계산 경로를 거치고도 일치**함을
확인했다(우연이 아니라 5개 reuse 프레임이 세 지표 모두에서 동일하게 "0 전송"으로
집계되기 때문). `accounting.enabled: false`(기본값) 실행은 `outputs/accounting/`
디렉터리 자체를 만들지 않고 `temporal_metrics.csv`/`segments.json` 등 기존 산출물이
byte-동일함을 확인했다(회귀 없음). `scripts/report_transmission_accounting.py`의
`--from-accounting-summary` 모드로 기존 산출물에서 rate/reliability 리포트를 다시
만들어도 동일한 숫자가 나옴을 확인했다.

주의: 6차 결과는 **논문급 CBR/bitrate 비교가 아니다.** `edge_cr`/
`symbols_per_bit_proxy`/`motion_side_info_bits`의 quantization 가정은 모두 튜닝
가능한 PoC 상수이며, 실제 변조·부호화·엔트로피 코딩을 반영하지 않는다.
`keyframe_visual_latent_symbols`도 실제 `encode_features` 텐서를 측정한 값이
아니라 프레임 shape + 고정 아키텍처 상수로부터의 추론이다(모듈 docstring 참조).
"semantic unit 절감"(1~4차, 이미 검증)과 "bit/channel-symbol 절감"(6차, 이번
PoC)은 항상 분리해서 인용해야 하며, 최종 논문급 CBR 비교·실제 DISTS/downstream
지표 통합은 6차 이후 후속 과제다.

테스트 현황: 6차 신규 테스트(`tests/test_transmission_accounting.py` — 38개,
per-component proxy/exact 계산, 결정별 payload 모델, baseline 2종, frame/segment/
summary 집계, accounting 비변형성, rate/reliability row 생성 및 append/merge)와
기존 회귀 테스트(`tests/test_presence_backends.py`, `tests/test_heldout_remeasurement.py`,
`tests/test_temporal_srs_calibration.py`, `tests/test_packet_matcher.py`,
`tests/test_video_generator.py`, `tests/test_video.py`, `tests/test_video_io.py`,
`tests/test_evaluators.py`, `tests/test_controllers.py`)는 `ptest` conda 환경에서
통과 확인됨(정확한 pass/fail 수치는 저장소의 최신 테스트 실행 로그 참조).


## 후속 딥러닝 1단계(LGVSC 재현 기반) 게이트별 상태

1단계는 다시 세 개의 독립 완료 게이트로 관리한다.

- **1A ✅ 완료(구조, 2026-07) — Segment API와 구조 변경:** 모델 weight 없이 완료 가능한
  코드 작업이다. segment 요청/결과 계약, 기존 mock 호환, Rx-legal 경계, 원본 inter-frame
  접근 방지, 기본값 수치 불변 및 전체 회귀 테스트까지 포함한다. 아래 "1A 구현 결과" 참조.
- **1B ✅ start-only + bidirectional 실제 GPU 검증 완료(2026-07) — 실제 생성
  backend 연결:** 완료 기준은 원래 "별도 환경에서 실제 weight를 로드해 최소 한
  segment를 GPU로 생성"이며, **Wan(`WanImageToVideoPipeline`)이 start-only와
  bidirectional 둘 다에서 이 기준을 실제로 충족**했다 — 실제 14B 모델이 RTX
  4080에서 caption(및 bidirectional segment에서는 end keyframe)을 조건으로
  실제 프레임을 생성했다(`docs/lgvsc_1b_worker_readiness.md`의 "1B Wan 검토 —
  실제 GPU 시도 결과" 및 "Wan bidirectional 수정 — 원인과 해결" 참조).
  bidirectional(last_image) 조건화는 원래 diffusers 버전 문제로 오인됐으나,
  실제 원인은 **체크포인트 선택**이었다 — 두 keyframe 조건화를 지원하려면
  `transformer/config.json`에 `pos_embed_seq_len`이 설정된 체크포인트가
  필요한데 기존에 쓰던 `Wan2.1-I2V-14B-480P`에는 이 값이 없었다. Wan의 공식
  first-last-frame 체크포인트 `Wan2.1-FLF2V-14B-720P`로 교체하고, 한 영상 안에
  섞여 있는 end-keyframe 유무에 따라 segment마다 체크포인트를 자동 선택하도록
  `run_wan_backend()`를 고쳐 실제 GPU에서 해결을 확인했다. 코드만 있고 실제
  생성 산출물이 없으면 완료가 아니라는 원래 기준을 그대로 유지했고, 이제
  start-only/bidirectional 둘 다 그 기준을 넘었다. 아래 "1B 구현 결과"와
  `docs/lgvsc_1b_worker_readiness.md` 참조.
- **1C 🟡 재현 준비 완료(구조, 2026-07) / 실제 검증 실행 대기 — LGVSC 재현선
  검증:** `SKIM+SFA`와 `SKEM+DSA`에 대응하는 재현 가능한 baseline(config
  4종 + batch driver + summary 생성기)을 완성했다 — 완료 기준은 여전히
  `SKIM+SFA`와 `SKEM+DSA` **결과**, 가변 길이 segment, seam/temporal
  지표와 재현 수준을 함께 보고하는 것이며, 이 결과 자체는 아직 없다(사용자가
  10개 영상 × 4모드 실제 GPU 실행을 완료해야 나온다). "구조 준비"와 "결과
  보고"를 혼동하지 않는다 — 아래 "1C 구현 결과"와
  `docs/lgvsc_1c_reproduction_readiness.md` 참조.

1A와 1B는 완료됐다(1B는 start-only와 bidirectional 조건화 둘 다 실제 GPU
검증까지 완료). **1C는 검증 준비(config/batch driver/summary 생성기)까지
완료**됐고, 실제 10영상×4모드 검증 실행 자체는 사용자가 진행한다.

**1C 다음 단계(PSSS/SKEM) — ✅ 코드/테스트/CPU 스모크 완료(2026-07), 실제
MLLM 검증 대기:** 1C의 "keyframe 선택(SKIM/SKEM)은 네 모드 전부 동일하다"는
한계를 메웠다 — 논문 Eq.1-2 그대로의 PSSS(실제 모델 다음-토큰 확률로
`S_rel = P(No) - P(Yes)` 계산, yes/no 최종 텍스트 비교 아님)와 그 PSSS로
자동회귀 keyframe을 선택하는 SKEM selector(`video/skem_selector.py`)를
추가했다. `keyframe.selector: fixed`(기본, 기존 4개 모드는 그대로 이 경로)
vs `psss`(신규)로 완전히 독립적인 selector를 골라 쓸 수 있고, PSSS는
mock/proxy(CLIP)/real(MLLM) 3개 backend로 명확히 구분된다. 비교 config 4종
(`skim_sfa_fixed`/`skem_dsa_psss`/`skem_dsa_mock_psss`/`skem_dsa_proxy_psss`)
과 batch driver/summary 확장까지 포함해 직접 실행·검증했다(CPU mock 스모크로
같은 두 영상이 fixed selector에서는 동일한 [2,12] 분할, PSSS(mock) selector
에서는 영상마다 다른 [1,3]/[4,6] 분할을 내는 것을 확인; `skim_sfa_fixed`는
실제 GPU Wan 14B로도 완주 확인). **완료로 인정되지 않는 부분**: PSSS
`real`(MLLM) 백엔드의 실제 GPU 실행, 실제 InternVL2급 캡셔너와의 정합,
side-info 인코더, 학습된 DSA adapter, CBR 캘리브레이션은 모두 아직 없다 —
자세한 내용은 `docs/lgvsc_psss_skem_readiness.md` 참조.

1B는 1A가 완성한
`SegmentGenerationRequest`/`generate_segment()`
계약을 그대로 구현 대상으로 삼았다(파이프라인 재작업 없음) — `TemporalPipeline`은
`ExternalSegmentWorkerGenerator`를 다른 mock backend와 동일하게
`video_generator.generate_segment()`로만 호출하므로, 실제 Open-Sora/Wan weight를
연결해도 `video/temporal_pipeline.py`를 다시 건드릴 필요가 없다.


## 1A/1B/1C 상세 구현 결과

### 1A 구현 결과 (2026-07)

1A(위 "1단계 — LGVSC 재현 기반"의 첫 번째 게이트)는 구조 완료됐다. 1A의 성격은
**"frame-wise mock generate 호출을 GOP/segment 단위 Rx-legal 생성 계약으로 바꾸고,
1B가 실제 Open-Sora/Wan backend를 붙일 때 `TemporalPipeline`을 다시 뜯지 않아도 되게
만드는 것"**이며, **1B(실제 backend 연결)나 1C(LGVSC 재현 검증)가 아니다.** Open-Sora/Wan
설치, weight 다운로드, 실제 GPU 생성, 생성 품질/drift 비교 주장은 이번 범위에 없다 — 3~4차와
마찬가지로 mock backend(`CopyGenerator`/`InterpolationGenerator`/
`BidirectionalInterpolationGenerator`)만 사용했고, 이번에 바뀐 것은 **호출 계약과
`TemporalPipeline`이 그 계약을 부르는 방식**이다.

구현 내용과 산출물 ↔ 코드 대응:

| 항목 | 코드 | 산출물 |
|---|---|---|
| Segment 요청/결과 계약 | `video/video_generator.py`의 신규 `SegmentGenerationRequest`(`segment_id`/`start_frame_index`/`end_frame_index`/`target_indices`/`start_keyframe_recon`/`start_keyframe_index`/`end_keyframe_recon`/`end_keyframe_index`/`fps`/`captions`/`packets`/`side_infos`/`reference_prev_recon`/`reference_prev_recons`, `segment_length` property) / `SegmentGenerationResult`(`frame_for`/`metadata_for`) + `validate_segment_request()`/`validate_segment_result()` | `tests/test_video_generator.py`의 `TestSegmentGenerationRequestRxLegal`/`TestValidateSegmentRequest`/`TestSegmentGenerationResult`/`TestValidateSegmentResult` |
| Rx-legal 경계 | `SegmentGenerationRequest`에는 원본/ground-truth target frame 필드가 **아예 존재하지 않음**(기존 frame-level `GenerationRequest.reference_target_frame`과 달리 opt-in 예외조차 없음) — `TemporalPipeline`도 이제 이 필드를 채울 코드 경로 자체가 없으므로 `allow_ground_truth_reference`는 pipeline 경로에서 사실상 no-op이 됨(하위호환을 위해 생성자 인자는 유지, docstring에 명시) | `tests/test_video_generator.py::TestSegmentGenerationRequestRxLegal::test_no_ground_truth_or_original_frame_field`(dataclass 필드 자체를 검사), `tests/test_video.py::TestSegmentGenerateBranch::test_rx_legal_segment_request_has_no_original_frame_field_or_leak`(실제 파이프라인 실행에서 spy backend가 받은 request에 원본 미전송 target 픽셀이 없음을 확인) |
| 기존 mock backend 호환 | `VideoGenerator.generate_segment()` 기본 구현이 `target_indices`마다 `generate()`를 순서대로 호출(내부에서 `reference_prev_recon`을 같은 호출 안에서 체인) — `CopyGenerator`/`InterpolationGenerator`/`BidirectionalInterpolationGenerator` 모두 override 없이 새 계약에서 그대로 동작 | `tests/test_video_generator.py::TestDefaultGenerateSegmentFallback` |
| `TemporalPipeline` segment 호출 배선 | `video/temporal_pipeline.py::run()` — 기존에는 `generate` 결정마다 즉시 `video_generator.generate()`를 불렀지만(구 `_generate_frame()`), 1A는 한 GOP의 `generate` 결정 프레임들을 `pending` 리스트로 모았다가 다음 keyframe을 만나는 시점(또는 시퀀스 끝)에 신규 `_flush_pending_generate()`로 **GOP당 정확히 한 번** `video_generator.generate_segment()`를 호출한다. `generate` 결정이 없는 GOP는 backend를 아예 부르지 않는다. reuse/recompute/keyframe 프레임은 이전과 동일하게 즉시 처리된다 | `tests/test_video.py::TestSegmentGenerateBranch::test_one_generate_segment_call_for_the_whole_gop`/`test_mixed_reuse_recompute_generate_in_one_segment_single_call`/`test_no_generate_targets_never_calls_backend` |
| FrameRecord/SegmentRecord 투영 | `_flush_pending_generate()`가 backend 결과를 `target_indices` 순서대로 각 `FrameRecord`에 되돌려 씀(`recon`/`generation`/`recon_packet`/`srs` — 기존 즉시-호출 경로와 동일한 필드, 동일한 스키마). `video/segment.py::build_segments()`는 무수정 — 이미 `records`를 읽어 집계하므로 자동으로 새 값을 반영 | `tests/test_video.py::TestSegmentGenerateBranch::test_frame_count_index_order_and_shape_preserved`/`test_generated_frames_projected_onto_frame_records_in_order`/`test_segment_record_generation_lists_all_generated_targets` |
| Bidirectional(4차) 및 마지막 GOP 정책 유지 | segment 배칭 후에도 4차의 end-keyframe 조건화·`bidirectional_missing_end_policy`(`error`/`fallback_start_only`)가 GOP 단위로 그대로 전달됨 — 마지막 GOP는 `end_keyframe_recon`/`end_keyframe_index`가 `None`인 segment request로 전달되고, `fallback_start_only`가 설정되면 해당 프레임만 `conditioning_mode="start_only"`로 강등됨(4차와 동일 결과) | `tests/test_video.py::TestSegmentGenerateBranch::test_bidirectional_segment_calls_carry_correct_end_keyframe_per_gop` |
| 잘못된 backend 결과에 대한 명시적 오류 | `validate_segment_result()`가 `TemporalPipeline._flush_pending_generate()`에서 항상 호출됨 — 요청과 다른 frame 수/순서/index/`segment_id`, `start_keyframe_recon`과 다른 tensor shape, 또는 metadata가 자신이 속한 frame과 어긋나는 경우(`target_indices`/`source_keyframe_index` 불일치, `GenerationMetadata`가 아닌 타입, 알 수 없는 `conditioning_mode`)를 반환하는 backend는 `ValueError`로 즉시 실패(조용한 오정렬 방지) | `tests/test_video_generator.py::TestValidateSegmentResult`, `tests/test_video.py::TestSegmentGenerateBranch::test_backend_returning_wrong_frame_count_raises`/`test_backend_returning_wrong_shape_raises`/`test_backend_returning_wrong_indices_raises` |
| 기본값 수치 불변 | `enable_generate=False`(기본값)에서는 `pending`이 결코 채워지지 않으므로 `video_generator.generate_segment()`가 한 번도 호출되지 않음 — 1~4차와 완전히 동일한 reuse/recompute-only 출력 | `tests/test_video.py::TestSegmentGenerateBranch::test_generate_disabled_never_touches_video_generator`, 그리고 기존 `TestGenerateBranch`/`TestBidirectionalGenerateBranch`/`TestGenerationModeComparison` 전체(수정 없이 그대로 통과) |
| Segment 생성 시간의 profiling 귀속 | `utils/profiling.py`의 신규 `RunProfiler.record_frame()` — enter/exit 타이머 없이 미리 계산된 elapsed를 직접 기록. `_flush_pending_generate()`가 `generate_segment()` 호출을 직접 시간 측정해 그 GOP의 generate 대상 프레임들에 균등 배분하고, `run()`의 flush 호출도 다음 keyframe의 `_prof.frame(i)` 타이머가 시작되기 **전에** 실행되도록 순서를 옮김(마지막 GOP는 루프 종료 후 flush) | `tests/test_speed_optimizations.py::TestSegmentGenerationProfilingAttribution` |

`fps`는 `TemporalPipeline(fps=...)` 생성자 인자로 추가되어 매 `SegmentGenerationRequest.fps`에
그대로 전달된다(기본 `None`, 수치 경로에 영향 없음) — 1B가 세그먼트의 실제 재생 길이를
알 필요가 있을 때를 위한 배선이다.

주의: 1A의 segment 배칭은 **구조적 개선일 뿐 생성 품질과 무관하다** — 여전히 mock
backend가 `generate()`를 프레임별로 순회 호출하는 기본 fallback이므로, 실제 출력 값은
3~4차와 바이트 단위로 동일하다(회귀 테스트로 확인).

### 1A 검토 반영 (2026-07)

1A 1차 구현 이후 두 차례의 코드 리뷰에서 총 4건의 보완 필요 사항(Medium 2건, Low 2건)이
지적됐다. 넷 다 이번 문서 갱신 이전에 수정하고 회귀 테스트로 확인했다 — 아래는 지적
사항과 정확히 무엇을 고쳤는지의 대응표다.

| 지적 (심각도) | 문제 | 수정 |
|---|---|---|
| 비연속 generate 대상의 prev_recon 오귀속 (Medium) | `_flush_pending_generate()`가 GOP의 **첫** generate 대상 시점의 `prev_recon`만 스냅샷하고, 기본 fallback은 이후 모든 대상을 "같은 호출 안에서 방금 생성한 프레임"으로 체인했다 — frame3=generate, frame4=recompute(recon=5.0), frame5=generate인 경우 frame5가 frame4의 실제 recon(5.0)이 아니라 frame3의 생성 결과를 참조해 수치가 달라졌다(probe로 확인: 수정 전 1.0, 기존 per-frame 의미상 예상 3.0) | `SegmentGenerationRequest`에 `reference_prev_recons`(target별 override 리스트) 추가. `TemporalPipeline`은 각 pending 항목을 만들 때 "직전 프레임 인덱스가 같은 GOP의 이전 pending 항목이 아니면"(= 그 프레임이 즉시 해결된 keyframe/reuse/recompute라면) 실제 `prev_recon`을 override로 채우고, 직전 프레임이 바로 앞의 pending 항목이면 `None`으로 두어 기본 fallback의 체인 로직에 맡긴다. `VideoGenerator.generate_segment()` 기본 구현은 override가 있으면 그것을, 없으면 체인 값을 사용 |
| Segment 생성 시간이 profiler에서 누락/오귀속 (Medium) | generate 프레임의 `_pctx`가 backend 호출 **전에** 종료되고(따라서 elapsed≈0), 중간 GOP의 실제 flush는 다음 keyframe의 `_pctx.__enter__()` **이후**(같은 반복문 안)에 실행돼 그 keyframe의 타이머가 flush 시간까지 함께 재는 문제였다. 마지막 GOP는 루프가 끝난 뒤 flush되므로 어떤 프레임 record에도 잡히지 않았다(80ms sleep backend로 실측: 실제 0.08초, profiler 합계 0.0003초) | `RunProfiler.record_frame()`(enter/exit 없이 elapsed를 직접 기록하는 API) 추가. `_flush_pending_generate()`가 `generate_segment()` 호출을 직접 `time.monotonic()`으로 측정해 그 호출이 커버하는 프레임 수만큼 균등 배분한 뒤 `record_frame()`으로 기록. **또한** `run()`에서 flush 호출을 다음 keyframe의 `_prof.frame(i)` 생성/`__enter__()` **이전**으로 옮겨, 다음 keyframe의 타이머가 flush 시간을 전혀 포함하지 않게 함 |
| `validate_segment_result()`가 metadata 정합성을 검사하지 않음 (Low) | `segment_id`, `metadata[i].target_indices`, `metadata[i].source_keyframe_index`, metadata 타입, `conditioning_mode` 값이 검사되지 않아 엉뚱한 request를 가리키는 metadata가 조용히 통과했다 | `validate_segment_result()`에 `result.segment_id == request.segment_id`, `metadata[i].target_indices == [target_indices[i]]`, `metadata[i].source_keyframe_index == request.start_keyframe_index`, `isinstance(metadata[i], GenerationMetadata)`, `conditioning_mode ∈ {start_only, bidirectional}` 검사를 추가 |
| Segment 생성 시간은 잘 귀속되지만 profiler **call-count** delta(`diffusion_calls`/`blip2_calls`/`clip_calls`)는 generate 프레임 record에 반영되지 않음 (Low) | `_flush_pending_generate()`가 `generate_segment()` 호출 전후로 wall-clock만 재고 `record_frame()`을 `elapsed_sec`만 채워 호출했다 — 1B의 실제 backend가 내부에서 `profiling.record_diffusion_call()` 등을 찍어도 run 전체 counter에는 쌓이지만 그 generate 프레임들의 `diffusion_calls`/`blip2_calls`/`clip_calls`는 항상 0으로 남는 구조였다 | `_flush_pending_generate()`가 `generate_segment()` 호출 **직전**에 `RunProfiler.counters`를 스냅샷하고, 호출 후 `diffusion_calls`/`blip2_calls`/(`clip_image_calls`+`clip_text_calls`) 각각의 delta를 계산해 신규 `_split_evenly()`(정수를 나머지까지 정확히 합이 보존되도록 프레임 수만큼 균등 분배)로 커버된 generate 프레임들에 나눠 `record_frame()`에 전달 |

수정 후 `SegmentGenerationRequest.reference_prev_recon`(단수)은 "target_indices[0] 이전
프레임의 재구성값"이라는 원래 의미를 유지하되, 여러 target을 정확히 다루려면
`reference_prev_recons`(target별 리스트)를 함께 채우는 쪽을 권장한다 — 1B 이후 실제
segment 단위 backend는 두 필드 모두 무시하고 두 keyframe만으로 조건화해도 무방하다
(모듈 docstring 참조).

테스트 현황: 1A 신규 테스트(`tests/test_video_generator.py`의
`TestSegmentGenerationRequestRxLegal`/`TestValidateSegmentRequest`/
`TestSegmentGenerationResult`/`TestValidateSegmentResult`/
`TestDefaultGenerateSegmentFallback`, `tests/test_video.py::TestSegmentGenerateBranch`,
검토 반영 후 추가된 `tests/test_speed_optimizations.py::TestSegmentGenerationProfilingAttribution`
— call-count delta 분배와 무-profiler sanity 케이스 포함)와 기존 회귀 테스트 전체가
`ptest` conda 환경에서 통과 확인됨: `conda run -n ptest python -m pytest
tests/test_video_generator.py tests/test_video.py tests/test_etri_batch_tools.py -q`
→ 202 passed; 전체 스위트 `conda run -n ptest python -m pytest tests/ -q` → **937 passed,
0 failed**(이번 변경으로 인한 회귀 없음; 과거 문서에 기록된 `tests/test_ddp.py::
test_entrypoint_torchrun_dryrun`의 로그 문구 불일치도 현재 재확인 결과 통과함 — 1A와
무관한 이전 상태였던 것으로 보임).

### 1B 구현 결과 (2026-07)

1B는 **구조·검증 준비 완료**됐다. 1B의 성격은 **"1A가 완성한
`SegmentGenerationRequest`/`generate_segment()` 계약 뒤에 실제 비디오 생성
모델을 안전하게(별도 conda 환경, subprocess 격리, 명확한 오류 처리로) 연결할 수
있는 구조를 만들고, 그 구조를 fake(mock) worker로 철저히 검증하는 것"**이며,
**"실제 GPU에서 실제 모델로 생성한 결과를 검증하는 것"이 아니다.** Open-Sora/Wan/
diffusers weight 다운로드, Hugging Face 로그인/라이선스 동의, 장시간 GPU
smoke test는 계획대로 이번 범위에서 실행하지 않았다 — 사용자가 직접 수행한다
(`docs/lgvsc_1b_worker_readiness.md`의 명령어 참조).

구현 내용과 산출물 ↔ 코드 대응:

| 항목 | 코드 | 산출물 |
|---|---|---|
| 외부 worker backend | `video/video_generator.py`의 신규 `ExternalSegmentWorkerGenerator`(1A의 `SegmentGenerationRequest`/`SegmentGenerationResult` 계약을 그대로 사용, `generate_segment()`만 override) + `SegmentWorkerError`. 요청을 `manifest.json` + start/end keyframe PNG로 저장하고 `python_bin`(다른 conda 환경의 인터프리터)으로 `worker_script`를 subprocess 실행, `result.json`/`error.json`을 읽어 결과 복원 | `tests/test_video_generator.py`의 `TestExternalSegmentWorkerGeneratorRoundTrip`/`TestExternalSegmentWorkerGeneratorManifestRxLegal`/`TestExternalSegmentWorkerGeneratorErrors` |
| Worker 스크립트 | 신규 `scripts/lgvsc_generate_worker.py` — `--backend mock`(PIL+numpy만, ptest가 실제로 실행하는 유일한 경로) / `--backend svd`(`diffusers.StableVideoDiffusionPipeline`, image만 조건화) / `--backend wan`(`diffusers.WanImageToVideoPipeline`, image+last_image+prompt 조건화 — 아래 "1B Wan 후속" 참조) / `--backend callable`(사용자 adapter 동적 import — 실제 Open-Sora 연동 지점). 무거운 모델 의존성은 각 backend 분기 안에서만 lazy import — `mock` 경로는 `diffusers`를 아예 import하지 않음(회귀 테스트로 확인) | `tests/test_lgvsc_generate_worker.py`(19개 mock/IO/dispatch + 7개 `TestRunWanBackendReferenceWiring` — manifest/이미지 IO, mock backend 동작, `generate()` 검증, `main()` end-to-end, 실제 subprocess 호출, diffusers-미import 가드) |
| Callable backend 템플릿 | 신규 `scripts/lgvsc_example_callable_backend.py` — 실제 Open-Sora 연동 시 복사해서 채울 어댑터 함수 템플릿(현재는 mock으로 fallback해 배선 자체를 스모크 테스트) | `tests/test_lgvsc_generate_worker.py::TestMainEndToEnd::test_callable_backend_via_example_template` 등 |
| Config 배선 | `configs/base/video/default.yaml`의 `video_generator.backend: external_segment_worker` + `video_generator.worker.*`(python_bin/worker_script/backend/backend_entrypoint/model_id/device/dtype/seed/height/width/num_inference_steps/decode_chunk_size/extra_json/timeout_sec/work_dir/cleanup_on_success/extra_env) — 기본값 전부 비활성/no-op. `build_generator()`가 `conditioning_mode` 분기보다 먼저 `external_segment_worker`를 확인(이 backend는 conditioning-mode-agnostic) | `tests/test_video_generator.py::TestBuildGeneratorExternalSegmentWorker`(+ `test_build_generator_constructs_wan_backend`) |
| 예시 config 5종 | `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_mock.yaml`(fake-worker, ptest에서 바로 실행 가능) / `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_svd.yaml`(실제 GPU, image만 조건화, 실제 GPU 검증 완료) / `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_start_only.yaml`(실제 GPU, image+prompt 조건화, 실제 GPU 검증 완료) / `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_bidirectional_fixed.yaml`(실제 GPU, image+last_image+prompt 조건화, **실제 GPU 검증 완료 — 지금부터 이걸 쓸 것**; segment별로 `Wan2.1-FLF2V-14B-720P`/`Wan2.1-I2V-14B-480P`를 자동 선택) / `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_bidirectional_experimental.yaml`(원래 실패를 재현하는 기록용, 더 이상 실사용하지 않음) — 전부 이 머신에 이미 있던 `semantic-diffusers` conda 환경의 python 경로를 가리킴 | `docs/lgvsc_1b_worker_readiness.md` |
| 오류 처리 | timeout / 0이 아닌 종료 코드(error.json 유무 각각) / result.json 없음·JSON 아님·status≠ok / frame 수·index 불일치 / frame shape 불일치 / 프레임 파일 없음 / 잘못된 python_bin / **metadata가 실제 frame과 어긋남**(잘못된 `segment_id`, `target_indices`/`source_keyframe_index` 불일치, 알 수 없는 `conditioning_mode`) — 전부 `SegmentWorkerError`로 work_dir 경로 포함해 즉시 실패, 실패 시 work_dir은 `cleanup_on_success`와 무관하게 항상 보존 | `tests/test_video_generator.py::TestExternalSegmentWorkerGeneratorErrors`(12개) |
| Rx-legal 경계 유지 | `SegmentGenerationRequest`에 원본 target frame 필드가 없다는 1A의 설계를 그대로 물려받음 — manifest에는 keyframe 이미지·caption·packet·side-info·run-config만 기록되고, 이를 검증하는 테스트가 실제 파일 내용을 읽어 확인 | `tests/test_video_generator.py::TestExternalSegmentWorkerGeneratorManifestRxLegal`, `tests/test_lgvsc_generate_worker.py::TestRunMockBackend::test_rx_legal_ignores_unreferenced_original_frame_on_disk` |
| TemporalPipeline 통합 + profiler 회귀 | 1A의 GOP당-1회 배치 호출이 실제 subprocess 기반 backend에서도 동일하게 동작(mixed reuse/recompute/generate, bidirectional, no-generate-segment 스킵)하고, 1A 검토에서 고친 profiler elapsed/call-count 귀속도 real subprocess에서 유지됨을 확인 | `tests/test_video.py::TestSegmentGenerateBranchExternalWorker`, `tests/test_speed_optimizations.py::TestSegmentGenerationProfilingAttributionExternalWorker` |

**1B 검토 반영 (2026-07)**: 1B 1차 구현 이후 코드 리뷰에서 Medium 1건이
지적됐다 — `generate_segment()`가 `_read_result()` 결과를 `validate_segment_result()`
호출 없이 그대로 반환해, fake worker가 `conditioning_mode="sideways"`/
`source_keyframe_index=999`/`target_indices=[999]`처럼 frame 수·shape는
맞지만 metadata가 거짓인 결과를 돌려줘도 `generate_segment()` 단독 호출은
통과했다(`TemporalPipeline` 경로는 그 함수를 항상 호출하므로 안전했지만,
외부 프로세스는 신뢰 경계이므로 backend 자체가 걸러야 한다). 수정:
`generate_segment()`가 `_read_result()` 직후 `validate_segment_result()`를
직접 호출하고, 실패를 `work_dir`을 포함한 `SegmentWorkerError`로 감싼다.
자세한 내용은 `docs/lgvsc_1b_worker_readiness.md`의 "1B 검토 반영" 참조.

**직접 실행한 검증(이 세션에서 완료)**:

1. `ptest` 유닛/통합 테스트: `conda run -n ptest python -m pytest
   tests/test_video_generator.py tests/test_video.py
   tests/test_speed_optimizations.py tests/test_lgvsc_generate_worker.py -q`
   → **228 passed**. 전체 스위트 `conda run -n ptest python -m pytest tests/ -q`
   → **983 passed, 0 failed**(회귀 없음).
2. fake-worker subprocess 왕복을 `ExternalSegmentWorkerGenerator` API로 직접
   실행 — bidirectional 조건화(상대 위치 0.25/0.5/0.75)가 정확히 블렌딩됨을 확인.
3. `scripts/evaluate_video.py` 실제 CLI 경로로 실제 ETRI 테스트 영상
   (`01_person_walk.mp4`, 100프레임)을 `--no-models`로 실행 — 기본
   threshold에서는 전부 reuse(정상 동작)였고, threshold를 임시로 완화한
   재실행에서는 91개 inter-frame이 전부 `generate` 분기로 라우팅되어 9개
   segment 각각 한 번씩 mock worker subprocess가 실제로 호출됐으며
   `generated_frames/`에 91개 PNG가 저장되고 `segments.json`의 `generation`
   필드가 정확히 채워짐을 확인(임시 설정/산출물은 검증 후 삭제).

**실행하지 않은 것(당시 계획대로, 사용자 몫)**: 실제 Open-Sora/Wan/SVD weight
다운로드, Hugging Face 인증/라이선스 동의, 실제 GPU에서의 생성 품질 확인.

### 1B Wan 후속 — Open-Sora/Wan external segment worker (2026-07)

1B의 미완료 항목이던 **"Open-Sora/Wan external segment worker"**를 후속
작업으로 구현했다. **Wan을 선택**했다 — 이유는 (1) 이 머신의
`semantic-diffusers` 환경에 이미 설치된 `diffusers` 0.39.0.dev0가
`WanImageToVideoPipeline`을 이미 포함하고 있었고, (2) 그 파이프라인이
`image`(start keyframe) + `last_image`(end keyframe) + `prompt`(caption)를
전부 받는 안정된 클래식 파이프라인이어서(Open-Sora는 이 환경에 없고, Wan2.2
5B의 I2V는 diffusers 자신이 "실험적"이라 명시한 modular pipeline API가
필요해 제외) LGVSC의 segment decoder 계약(start/end keyframe + caption)에
가장 가깝게 실제로 조건화할 수 있었기 때문이다.

| 항목 | 코드 | 산출물 |
|---|---|---|
| Wan backend | `scripts/lgvsc_generate_worker.py::run_wan_backend()` — `image`(항상)/`last_image`(end keyframe 있을 때만, 실제 bidirectional 조건화)/`prompt`(segment 내 첫 caption, 실제 텍스트 조건화). `side_infos`는 accepted이지만 **의도적으로 미사용**(수치형 delta/motion dict를 프롬프트로 바꿀 검증된 방법이 없음 — 한계로 명시, 과장하지 않음). Wan의 `(num_frames-1) % 4 == 0` 제약을 만족하도록 요청 프레임 수를 올림. `extra_json`의 `offload_mode`(`sequential`/`model`)로 VRAM/속도 트레이드오프 선택 | `tests/test_lgvsc_generate_worker.py::TestRunWanBackendReferenceWiring`(7개 — fake `WanImageToVideoPipeline`으로 image/last_image/prompt 인자 전달, shape 복원, metadata, frame-count 반올림, offload 배선, dependency-unavailable 검증) |
| Config | `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_start_only.yaml`(안전/기본, 16GB 카드용 저해상도/offload 기본값), `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_bidirectional_fixed.yaml`(**실제 GPU 검증 완료 — 지금부터 이걸 쓸 것**, segment별 체크포인트 자동 선택), `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_bidirectional_experimental.yaml`(원래 실패를 재현하는 기록용으로 유지, 코드 리뷰 피드백으로 단일 `wan.yaml`에서 분리) — 기존 `..._svd.yaml`도 이번 환경 수정을 반영해 `PYTHONNOUSERSITE`를 `"1"`→`"0"`으로 정정(아래 참조) | `docs/lgvsc_1b_worker_readiness.md` |
| Config wiring 테스트 | `worker.backend: wan`이 `build_generator()` → `ExternalSegmentWorkerGenerator` → subprocess argv(`--backend wan`)까지 정확히 전달되는지 확인 | `tests/test_video_generator.py::TestBuildGeneratorExternalSegmentWorker::test_build_generator_constructs_wan_backend` |
| 환경 수정 (`semantic-diffusers`, `ptest`는 무변경) | `pip install -U transformers`(4.51.3→5.14.1, huggingface-hub 버전 충돌 해결) → `pip install -U peft`(0.17.0→0.20.0, 업그레이드된 transformers가 제거한 `HybridCache`를 구버전 peft가 무조건 import하던 문제 해결) → `pip uninstall torchaudio`(torch 2.12.0+cu130과 ABI 불일치하던 torchaudio 2.6.0+cu124 제거 — 영상 생성에 불필요) | 세 수정 후 `WanImageToVideoPipeline`/`StableVideoDiffusionPipeline` 모두 실제 import 성공을 직접 확인 |
| `PYTHONNOUSERSITE` 정정 | 위 환경 수정 후 실측한 결과, 이 머신의 `semantic-diffusers`는 **user-site 패키지를 포함해야** (`huggingface_hub`의 `is_offline_mode`가 env 자체 설치본에는 없고 user-site 설치본에만 있음) import가 성공한다 — 이전에 문서화했던 `PYTHONNOUSERSITE: "1"`(제외)과 반대. 두 config 모두 `"0"`으로 정정 | `docs/lgvsc_1b_worker_readiness.md`의 "PYTHONNOUSERSITE" 절 |

테스트: `conda run -n ptest python -m pytest tests/test_lgvsc_generate_worker.py
tests/test_video_generator.py tests/test_video.py
tests/test_speed_optimizations.py -q` → **237 passed**; 전체 스위트 → **992
passed, 0 failed**(회귀 없음).

**실제 GPU 시도 결과 (직접 실행, 2026-07)**: `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`
전체 가중치(84GB)를 `semantic-diffusers` 환경에 실제 다운로드(24.5분)한 뒤
`ExternalSegmentWorkerGenerator`로 실제 GPU(RTX 4080) 세그먼트 생성을 두 번
시도했다. **start-only(image+caption prompt)는 성공** — 실제 14B 모델이
64×64 프레임을 63초에 생성했고 `validate_segment_result()`까지 통과했다
(mock이 아닌 최초의 실제 GPU 최종 검증). **bidirectional(+last_image)은
실패** — `diffusers/models/transformers/transformer_wan.py`의
`torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)`에서
tensor 크기 불일치가 발생했고, 원인은 체크포인트가 배포한
`image_encoder`(`CLIPVisionModelWithProjection`)가 사용 중인 diffusers
개발 브랜치(0.39.0.dev0)의 `WanImageToVideoPipeline`이 기대하는 클래스
(`CLIPVisionModel`)와 달라 두 keyframe을 함께 인코딩할 때 임베딩 shape이
어긋나는 것으로 확인됨(이 시점의 진단 — **이후 "체크포인트 선택" 문제로
재규명·해결됨, 아래 "Wan bidirectional 실제 GPU 문제 해결" 참조**). 정확한
로그·트레이스백은 `docs/lgvsc_1b_worker_readiness.md`의 "1B Wan 검토 —
실제 GPU 시도 결과" 참조.

**후속 코드 리뷰 반영 (2026-07)**: 세 가지를 수정했다. (1) `run_wan_backend()`의
프레임 매핑이 `target_indices` 리스트 내 순서(`enumerate` 인덱스)로 생성 clip
프레임을 골랐던 버그를 수정 — segment 안에서 generate target이 비연속(예:
`[1, 5, 8]`)이면 실제 시간 위치와 다른 프레임을 반환하던 문제였다. 이제는
`target_index - start_frame_index`(실제 segment 오프셋) 기준으로, 그리고
clip 길이는 target list 개수가 아니라 segment의 실제 span(`segment_length`/
`end_frame_index`, bidirectional이면 `end_keyframe_index`까지)으로 계산해
매핑한다. 회귀 테스트 2개 추가
(`test_non_contiguous_targets_map_to_correct_temporal_position`,
`test_bidirectional_targets_map_to_relative_temporal_position`). (2) 위처럼
Wan config를 `wan_start_only.yaml`/`wan_bidirectional_experimental.yaml`로
분리해 known-broken 모드가 기본 config가 되지 않도록 함. (3)
`docs/lgvsc_1b_worker_readiness.md`가 "SVD 실제 GPU 세그먼트 생성 미시도"라고
잘못 기록했던 것을 정정 — 실제로는 SVD도 실제 GPU에서 `n_generate=1`,
`generated_frames=1`로 성공했다. `tests/test_lgvsc_generate_worker.py
tests/test_video_generator.py -q` → **121 passed**(기존 119 + 회귀 테스트 2개),
전체 스위트 회귀 없음.

**Wan bidirectional 실제 GPU 문제 해결 (2026-07 후속)**: 위 review 직후 남은
마지막 항목 — "Wan bidirectional 실제 GPU 문제"를 실제로 해결했다. diffusers
소스 코드(`pipeline_wan_i2v.py`/`transformer_wan.py`)를 직접 읽어 원인을
재규명한 결과, 원래 의심했던 "diffusers 버전-체크포인트 `image_encoder`
호환성 문제"는 부정확했다 — 실제 원인은 **체크포인트 선택**이었다.
`last_image`가 있으면 두 이미지가 배치 2로 CLIP 인코딩되는데, 이를 배치
1의 doubled-sequence로 reshape하는 `WanImageEmbedding`의 학습된
`pos_embed` 파라미터는 체크포인트의 `transformer/config.json`에
`pos_embed_seq_len`이 있을 때만 생성된다. 기존에 쓰던
`Wan2.1-I2V-14B-480P`에는 이 값이 없어(단일 이미지 조건화만 학습됨)
reshape이 생략되고 배치 크기가 어긋나 크래시했던 것이다. 이전 로그의
"`Expected types for image_encoder: ... got CLIPVisionModelWithProjection`"
경고는 무관한 경고였음도 확인했다(수정 후 성공한 실행에서도 동일하게
출력됨). Wan의 공식 first-last-frame 체크포인트
`Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers`(`pos_embed_seq_len: 514`)로
바꾸자 해결됐다. 한 영상 안에 end keyframe이 있는/없는 segment가 섞여
있을 수 있으므로 `run_wan_backend()`가 **segment마다** 필요한 체크포인트를
`extra_json.bidirectional_model_id`로 자동 선택하도록 수정했고, 체크포인트와
요청된 조건화 모드가 맞지 않으면 파이프라인 호출 전에
`WorkerBackendUnavailableError`로 명확히 실패하는 preflight 체크도
추가했다. `configs/experiments/etri_video_eval/etri_video_eval_lgvsc_worker_wan_bidirectional_fixed.yaml`로
실제 GPU에서 `scripts/evaluate_video.py --max-frames 14`를 실행해
segment 0(`conditioning_mode=bidirectional`, `end_keyframe_index=12`,
`n_generated=11`, `Wan2.1-FLF2V-14B-720P` 사용)과 segment 1(마지막 GOP,
`conditioning_mode=start_only`, `Wan2.1-I2V-14B-480P` 사용, `n_generated=1`)
모두 성공을 직접 확인했다(exit code 0, `error.json` 없음, `n_generate=12`
전체, 12개 PNG 실제 픽셀 분산 확인). 신규 테스트 3개 추가
(`test_start_only_segment_uses_default_model_even_with_bidirectional_id_set`,
`test_bidirectional_segment_with_incapable_checkpoint_raises_clear_error`,
`test_start_only_segment_with_flf2v_only_checkpoint_raises_clear_error`),
기존 bidirectional 테스트도 실제 checkpoint 선택 로직을 검증하도록 갱신.
`tests/ -q` 전체 스위트 → **997 passed**, 0 failed(회귀 없음). **현재 1B
상태: "start-only + bidirectional 모두 실제 GPU 검증 완료"**가 정확한
요약이다 — 자세한 원인 분석·재현 로그는
`docs/lgvsc_1b_worker_readiness.md`의 "Wan bidirectional 수정 — 원인과
해결" 참조.

### 1C 구현 결과 (2026-07) — 재현 준비 완료, 실제 검증 실행은 사용자

**1C의 이번 작업 범위는 "검증 준비"이지 "검증 실행"이 아니다** — 실제
10영상 × 4모드 GPU 실행/결과 판정은 사용자가 직접 한다. 이번에 완성한 것:

- **재현 baseline 4모드**: `mock_baseline`(이 저장소 mock/interpolation,
  LGVSC 구성요소 아님) / `svd_start_only`(real SVD, image-only) /
  `wan_skim_sfa`(real Wan, start-keyframe+caption — LGVSC SFA에 대응하는
  nearest-reproducible 근사) / `wan_skem_dsa`(real Wan, start+end
  keyframe+caption — LGVSC DSA에 대응하는 근사). **SKIM/SKEM(keyframe
  선택) 자체는 네 모드 전부 이 저장소의 동일한 고정-간격+scene-change
  추출기를 쓴다 — 별도 재현하지 않았다**는 것을 문서와 config 헤더 모두에
  명시했다(과장 방지, 사용자 지시사항).
- **Config 4종**: `configs/experiments/lgvsc_1c/etri_lgvsc_1c_{mock_baseline,svd_start_only,
  wan_skim_sfa,wan_skem_dsa}.yaml` — `wan_skim_sfa`/`wan_skem_dsa`는
  각각 1B에서 실제 GPU 검증된 `..._wan_start_only.yaml`/
  `..._wan_bidirectional_fixed.yaml`의 `video_generator.worker` 블록을
  그대로 복사해, 테스트(`TestBasedOnVerifiedConfigs`)로 두 파일이 계속
  일치하는지 고정했다.
- **Batch driver**: `scripts/batch_lgvsc_1c_reproduce.py` —
  `--modes`/`--videos`/`--max-frames`/`--device`/`--no-models`/`--dry-run`/
  `--skip-existing`/`--continue-on-error`/`--summary-only`를 지원하고,
  `outputs/etri_video_eval/lgvsc_1c_reproduce/<mode>/<video_id>/`로 mode×video별
  output을 완전히 격리한다(구현 중 실제로 경로 격리 버그 하나를 직접
  발견·수정함 — 생성 config가 `_generated_configs/<mode>/`의 상대 경로를
  쓰면 같은 mode의 모든 영상이 파일을 공유해버리는 문제였고, 절대 경로로
  바꿔 해결). Summary 생성기가 `temporal_metrics.csv`/`segments.json`/
  `generated_frames/`를 모아 `summary_metrics.csv`/`.md`/`.json`으로
  묶는다(mode, video_id, status, PTC/SFR/SDI/SRS류 전 지표,
  generated_frame_count, conditioning_modes_observed, backends_observed,
  has_end_keyframe, error_log_path).
- **테스트**: `tests/test_batch_lgvsc_1c_reproduce.py` 신규 18개 — 모드→config
  선택, output 경로 mode/video별 격리, dry-run이 subprocess를 호출하지
  않음, 명령어에 `--max-frames`/`--device`/`--no-models` 반영,
  `--summary-only`가 디스크의 기존 결과에서 재생성, `--continue-on-error`로
  실패 job 후에도 다음 job 진행(과 그 반대: flag 없으면 첫 실패에서 중단),
  `wan_skim_sfa`/`wan_skem_dsa`/`svd_start_only` config가 각각의 1B 검증
  완료 config를 기반으로 함. `tests/test_batch_lgvsc_1c_reproduce.py
  tests/test_video_generator.py tests/test_video.py -q` → **197 passed**;
  전체 스위트 `tests/ -q` → **1015 passed**, 0 failed(회귀 없음).
- **직접 실행한 검증** (실제 GPU 전체 validation은 하지 않음 — 지시사항
  준수): `--dry-run`으로 wan_skim_sfa/wan_skem_dsa 두 영상의 명령어·경로
  생성을 직접 확인, `mock_baseline --no-models --max-frames 14`로 진짜
  GPU 없는 smoke 1건을 실제로 실행해 `temporal_metrics.csv`/
  `segments.json`/`generated_frames/`/`summary_metrics.csv`가 실제로
  올바르게 생성됨을 확인(`n_generate=12`, `conditioning_modes_observed=
  bidirectional;start_only`, `has_end_keyframe=True` — 마지막 GOP가
  자동으로 start_only로 fallback하는 것까지 정확히 관측됨), `--skip-existing`/
  `--summary-only`도 이 실제 산출물에 대해 직접 실행해 확인. 검증에 쓴
  임시 산출물은 삭제했다(사용자의 실제 검증과 섞이지 않도록).

자세한 모드 정의·config↔소스 대응·실행 명령어·결과 해석 주의사항은
`docs/lgvsc_1c_reproduction_readiness.md` 참조.


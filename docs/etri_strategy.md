> [← 문서 색인](./README.md)

# ETRI 전략 정리

이 문서는 기존 [개발계획서 보관본](./archive/etri_development_plan_v2.md),
[로드맵 보관본](./archive/etri_development_roadmap.md),
[한계점 지도 보관본](./archive/limitation_reference_map.md)을 합친 **ETRI 과제용 단일
전략 문서**다. 목적은 문서 수를 줄이면서도 "무엇이 핵심 문제이고, 무엇을 먼저
구현·평가해야 하는가"를 한 곳에서 보게 하는 것이다.

이 문서의 한계 번호와 개발 방향은 `ETRI_연구진행상황_1차공유_20260715_v2.pptx`의
구성에 맞춘다. 슬라이드 번호는 **2026-07-16 개정판 기준**이다: 통합 송신단·수신단
블록다이어그램이 슬라이드 7·8로 새로 들어가면서 기존 7~12번이 9~14번으로 밀렸다
(한계 3 = 슬라이드 9, 전송량 절감 = 슬라이드 10, 신규 연구 아이템·평가 벤치마크 =
슬라이드 12, 협의 필요사항 = 슬라이드 13).

## 목표

`sgdjscc_lab`의 1차 목표는 최대 `PSNR`이 아니라 **무선 전송 후 시맨틱 의도의 신뢰성
있는 보존**이다. 즉, 수신 이미지나 영상이 자연스러워 보이는지보다 **원래 의도한
객체·관계·장면 정보가 채널 잡음·페이딩·패킷 손실 이후에도 얼마나 정확히
보존되는가**를 더 중요하게 본다.

과제의 핵심 산출물은 다음 세 가지다.

- 송신단 인코딩, 무선 채널, 수신단 생성 복원, 평가·기록을 잇는 End-to-End
  시뮬레이션 프레임워크
- 화질 지표와 별개로 의미 보존을 측정하는 `SRS`, `srs_packet`, packet verifier,
  시간축 의미 지표
- 영상 확장과 전송량 절감 PoC를 통해 "의미 신뢰도와 전송량의 관계"를 분석할 수
  있는 실험 기반

## 핵심 한계 3가지

| 핵심 한계 | 의미 | 현재/계획 대응 |
|---|---|---|
| 한계 1: 시간축·영상 한계 | 단일 이미지 평가로는 시간 흐름, 장면 전환, 프레임 간 의미 일관성을 볼 수 없음. 현재 영상 파이프라인은 의미 변화만 보고 재사용해 카메라 이동 같은 모션을 놓칠 수 있음 | keyframe pipeline, scene change, temporal evaluator, semantic delta + motion gate, `PTC`/`SFR`/`SDI`, LGVSC-inspired generate 분기 |
| 한계 2: 수신단 생성 복원의 할루시네이션 | 복원 결과가 그럴듯해 보여도 없던 객체를 만들거나, 있어야 할 정보를 누락·왜곡할 수 있음. 비디오 generate 분기는 이 위험을 더 키움 | semantic packet verifier, packet matcher, 오류 유형별 regeneration controller, OWLv2/VQA 보강, 후속 `Semantic Packet Fidelity Adapter`와 `Counterfactual Hallucination Critic` |
| 한계 3: 평가 체계 신뢰도 한계 | `PSNR`·`SSIM`·CLIP·기존 SRS만으로는 시간축 의미 일관성, 객체 깜빡임, 의미 drift를 충분히 설명하기 어렵고, 재생성 판단 지표와 최종 평가 지표가 같으면 순환 평가가 생김 | loop-internal 지표와 held-out 최종 지표 분리, `PTC`/`SFR`/`SDI`, GT/VLM 기반 `Temporal SRS Calibration`, Presence Calibration, DISTS/downstream 후보 관리 |

실행 우선순위는 **최소 평가기 안정화 → 시간축 파이프라인 구축 → 검출·검증 강화
→ 비교 지표 고도화** 순서다. 시간축 파이프라인(`video_io`, segment abstraction,
generate)은 먼저 구축할 수 있지만, flicker·temporal hallucination·semantic verifier의
강한 결론은 객체 검출과 SRS 품질에 의존하므로 OWLv2/VQA/GT 기반 보강 후 재검증한다.

## 한계별 원인과 해결 방안

아래는 세 한계의 원인을 실제 코드 위치와 연결해 정리한 것이다. 모든 파일 경로는
`src/sgdjscc_lab/` 기준이다.

### 공통 병목: 객체 존재 판정

`ObjectPreservationEvaluator._detect_objects()`
(`evaluators/object_preservation.py`)는 객체 존재를 **CLIP 전역 텍스트-이미지
유사도 + 고정 절대 임계값 + 제한 어휘**로 판정한다.

전파 경로:

- `evaluators/hallucination.py` — 같은 객체 판정으로 `recon - orig` 계산
- SRS 항목 중 preservation/missing/additional — 객체 판정 오류를 직접 상속
- `srs_packet` — base SRS와 packet composite을 blend하므로 CLIP 기반 오류를 부분 상속
- temporal flicker, PTC, SFR, SDI — 프레임별 객체/packet 판정이 흔들리면 시간축 지표도 흔들림

따라서 비디오 확장보다 앞서 또는 병행해 **최소 판정 안정화**가 필요하다. 다만
OWLv2/VQA 같은 강한 검출기까지 모두 끝낸 뒤에야 비디오를 시작해야 한다는 뜻은 아니다.
`video_io`, segment 구조, generate 분기는 먼저 구현 가능하고, 검출 의존 지표는 초기
CLIP 기반 잠정치로 낸 뒤 재측정한다.

### 한계 1: 시간축·영상 한계

원인:

| # | 원인 | 근거 |
|---|---|---|
| 1 | 현재 기본 경로는 정지 이미지 중심이며, 비디오 입력·mp4 출력·세그먼트 단위 복원은 확장 항목 | `video_extension_lgvsc.md`, PPT 슬라이드 5 — **1차에서 mp4 입출력·segment 구조 구현됨** (`utils/video_io.py`, `video/segment.py`) |
| 2 | reuse/recompute 결정이 의미 델타 중심이라, 의미는 같지만 픽셀 모션이 큰 카메라 pan/zoom을 놓칠 수 있음 | `video/temporal_pipeline.py` — **1차에서 `motion_residual` 기반 이중 게이트 구현됨** (`temporal.motion_threshold`, 기본 off) |
| 3 | reuse는 keyframe 복원 결과를 재사용하므로, 긴 구간에서는 drift와 객체 깜빡임을 별도 지표로 봐야 함 | temporal evaluator — **1차에서 `PTC`/`SFR`/`SDI` 잠정 지표 구현됨** (`evaluators/temporal_consistency.py`) |

해결:

1. **키프레임 기반 영상 확장.** 키프레임은 기존 SGD-JSCC 송신/복원 경로를 그대로
   재사용하고, 비-키프레임에는 semantic delta와 motion side-info를 저용량으로 보낸다.
   이 방향은 LGVSC를 직접 재현하는 것이 아니라, LGVSC의 "키프레임 + 사이드 정보 +
   생성 복원" 구조를 `sgdjscc_lab`에 맞게 차용하는 것이다.
2. **semantic delta + motion 이중 게이트.** 재사용 조건을 "의미 변화가 작다" 하나로
   두지 않고, keyframe 대비 motion residual까지 함께 본다. 의미가 같아도 모션이 크면
   `reuse` 대신 `recompute` 또는 `generate`로 보낸다.
3. **3-way 복원 분기.** 비-키프레임 구간을 `reuse`, `recompute`, `generate`로 나눈다.
   `generate`는 start-only keyframe conditioning부터 시작하고, 가능하면 start+end
   bidirectional conditioning으로 drift를 줄인다. SVD/Open-Sora 같은 공개 비디오 생성
   모델은 별도 worker로 붙이는 PoC부터 시작한다.
4. **시간축 의미 지표.** 기존 temporal SRS에 더해 `PTC`, `SFR`, `SDI`를 정의한다.
   초기에는 CLIP/packet 기반 잠정치로 구현하고, OWLv2/VQA 보강 후 재측정한다.

### 한계 2: 수신단 생성 복원의 할루시네이션

원인:

| # | 원인 | 근거 |
|---|---|---|
| 1 | 확산 복원은 그럴듯한 이미지를 만들 수 있지만 전송 packet에 없는 객체를 추가할 수 있음 | 생성형 복원 구조 자체의 위험 |
| 2 | 기존 regeneration은 사후 점수 기반 재시도 성격이 강하고, 오류 유형별 억제·강조가 충분히 구조화되어 있지 않음 | `controllers/regeneration_policy.py`, `pipelines/infer_pipeline.py` |
| 3 | 비디오 generate 분기는 보내지 않은 중간 프레임까지 생성하므로 정지 이미지보다 hallucination, object drift, flicker 위험이 큼 | PPT 슬라이드 6 |

해결:

1. **Packet Verifier.** 전송 semantic packet과 복원 결과에서 재추출한 packet을 비교해
   추가 객체, 누락 객체, 관계/구조 왜곡을 직접 검출한다. 핵심은 원본 이미지가 아니라
   수신단이 보유한 전송 packet을 기준으로 판단하는 **Rx-legal self-verification**이다.
2. **오류 유형별 재생성 Controller.** 오류 종류에 따라 prompt, guidance, recompute
   전략을 다르게 적용한다. 추가 객체는 negative prompt로 억제하고, 누락 객체는 prompt나
   조건을 강화하며, 구조 왜곡은 edge/motion guide를 강화한다. 반복 실패 시
   recompute/keyframe fallback을 둔다.
3. **OWLv2/VQA 보강.** CLIP 전역 유사도만으로 객체 존재를 판단하면 오탐/미탐이 생기므로,
   OWLv2 같은 grounded detector와 VQA 질문 기반 확인을 verifier의 보강층으로 둔다.
4. **후속 학습형 모듈.** `Semantic Packet Fidelity Adapter`는 전송 packet을 diffusion
   복원의 조건 embedding으로 직접 주입하는 경량 adapter다. `Counterfactual
   Hallucination Critic`은 복원 객체가 전송 packet 관점에서 허용되는지 판별하는 critic이다.
   이 둘은 현 과제의 1차 필수 구현이 아니라 고도화/후속 항목으로 관리한다.

### 한계 3: 평가 체계 신뢰도 한계

원인:

| # | 원인 | 근거 |
|---|---|---|
| 1 | `PSNR`·`SSIM`은 프레임 화질 중심이라 객체·관계·장면 의미와 시간축 일관성을 설명하지 못함 | PPT 슬라이드 9 |
| 2 | 재생성 여부를 판단한 지표로 최종 성능까지 주장하면 metric gaming 또는 순환 평가가 생김 | closed-loop 구조의 평가 위험 |
| 3 | 기존 CLIP/SRS만으로는 객체 birth/death, semantic drift, temporal hallucination을 직접 설명하기 어려움 | 비디오 확장 평가 요구 |

해결:

1. **지표 분리.** loop-internal 지표와 held-out 최종 평가 지표를 분리한다.
   packet matcher, `srs_packet`, VQA는 재생성 여부 판단에 쓰고, GT 객체 보존과 신규
   시간축 지표는 최종 성능 주장에 사용한다.
2. **신규 시간축 의미 지표.**
   - `PTC`(Packet-Temporal Consistency): 전송 packet과 복원 영상 packet의 일치도가
     시간축에서 유지되는지 측정
   - `SFR`(Semantic Flicker Rate): 객체가 프레임마다 생겼다 사라지는 birth/death 비율 측정
   - `SDI`(Semantic Drift Index): 키프레임에서 멀어질수록 의미가 원래 의미에서 얼마나
     이탈하는지 측정
3. **Temporal SRS Calibration.** GT/VLM 판단 기준으로 시간축 SRS의 가중치를 보정한다.
   이는 1차 구현이 아니라 GT/VLM 연결 후 고도화 항목이다.
4. **비교 지표 후보 관리.** Presence Calibration은 객체 존재 판정 신뢰도 보강용으로,
   DISTS/downstream 지표는 최종 비교용 후보로 별도 관리한다.

## 전송량 절감 대응

ETRI 문의사항은 "semantic unit 수 절감뿐 아니라 채널 심볼 또는 비트 기준 전송량
축소가 가능한가"이다. 답은 **구조적으로 가능하되 단계적으로 검증**하는 것이다.

| 단계 | 내용 | 산출물 |
|---|---|---|
| 현행 | 프레임 간 의미 변화를 비교해 변화가 작으면 이전 복원을 재사용 | semantic unit 기준 절감 |
| 1차 PoC | 변화가 작은 latent/semantic 요소를 덜 보내고, 수신단이 이전 프레임 정보를 재사용 | 절감률 vs SRS/PTC 곡선 |
| 2차 설계 | 현재 구조는 아날로그 심볼 전송에 가까우므로, 비트 기준 전송에는 양자화·비트 매핑·채널 부호화 설계 필요 | bit 기준 설계안 |
| 후속 | 중요한 의미에 더 많은 심볼/비트를 배분하는 importance-aware allocation | 전송량-의미 신뢰도 체계 분석 |

이 문서에서는 **semantic unit 절감**, **channel-symbol 절감**, **bit 기준 절감**을 구분한다.
8월 진도점검 전까지는 channel-symbol 절감 PoC와 정량화 가능성 검증에 집중하고, bit 기준은
설계안으로 제시한다.

## 구현 실행 순서

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
| 2 ✅ | 슬라이드 5 `시간축 평가` / 슬라이드 9 `PTC·SFR·SDI` | temporal SRS와 별도로 packet consistency, 객체 깜빡임, semantic drift를 계산한다 | `evaluators/temporal_consistency.py`, `video/temporal_pipeline.py` | 한계 1·3, 슬라이드 5·9 | 8월 | `temporal_metrics.csv`에 `PTC`/`SFR`/`SDI` 초기값이 기록됨 — **완료(잠정 지표)** — CLIP/packet 기반 초기값, 5차 OWLv2/VQA 보강 후 재측정 필수 |
| 3 ✅ | 슬라이드 5 `세그먼트 판단 게이트` / `의미 델타 + 모션 이중 게이트` | semantic delta만 보던 reuse/recompute 판단에 keyframe 대비 motion residual을 추가한다 | `video/semantic_delta.py`, `video/motion_residual.py`, `video/temporal_pipeline.py` | 한계 1, 슬라이드 5 | 9월 | 의미 변화는 작지만 카메라 이동이 큰 구간을 reuse하지 않음 — **완료** (`temporal.motion_threshold`/`motion_weight`/`motion_grid`, 기본 off = 기존 동작, decision 로그 기록) |
| 4 ✅ | 슬라이드 5 `키프레임` / `비-키프레임` / `세그먼트` 구조 | 프레임 단위 처리를 GOP/segment 단위 처리로 묶어 generate 분기를 붙일 수 있게 한다 | `video/keyframe_extractor.py`, 신규 `video/segment.py` | 한계 1, 슬라이드 5 | 9월 | 기존 frame-wise 결과와 segment 결과가 동등하게 재현됨 — **완료** (`segments.json`, frame-wise 로그와 병행 출력, `SegmentRecord.generation`은 generate 분기용 예약 인터페이스) |
| 5 ✅ 기초 | 슬라이드 5 `Generate (신규)` | 시작 keyframe, caption, side-info를 조건으로 세그먼트 생성 경로를 붙인다 | 신규 `video/video_generator.py`, `use_video_gen`/`video_generator.*` config | 한계 1·2, 슬라이드 5·6 | 9~10월 | `reuse/recompute/generate` 3-way 분기가 동작하고 생성 결과가 저장됨 — **완료(기초, 3차)**: mock(copy/interpolation) backend로 구조 구현, 실제 학습형 생성 모델(SVD/Open-Sora)은 후속 |
| 6 ✅ 기초 | 슬라이드 5 `Generate (start / start+end 양방향)` | 시작 keyframe과 끝 keyframe을 모두 조건으로 넣어 drift를 줄일 수 있는지 확인한다 | `video_generator` bidirectional mode, `pipelines/generation_mode_comparison.py` | 한계 1, 슬라이드 5 | 10월 이후 | start-only 대비 drift/flicker가 줄어드는지 비교 결과가 나옴 — **완료(기초, 4차)**: mock bidirectional backend + 비교 파이프라인 구조가 동작. 실제 drift/flicker 감소 여부에 대한 성능 주장은 실제 생성 모델 통합 후(5차+) 판단 |
| 7 ✅ 기초 | 슬라이드 6 `Packet Verifier` / `오류 유형별 재생성 Controller` | 전송 packet과 복원 packet을 비교하고, 추가·누락·왜곡별로 재생성 조건을 다르게 조정한다 | `evaluators/packet_verifier.py`, `controllers/verifier_controller.py`, `pipelines/packet_verification.py` | 한계 2, 슬라이드 6 | 9~10월 | 오류 유형별 report와 controller decision 로그가 생성됨 — **완료(기초, 2차)**: rule-based verifier/controller가 `TemporalPipeline` 결과에 옵션(기본 OFF)으로 연결됨. OWLv2/VQA 보강과 candidate action의 실제 sampler 반영은 5차·후속 |
| 8 ✅ 코드 기반 스캐폴드 | 슬라이드 6 `Packet Verifier` 보강 / 슬라이드 9 `Presence Calibration` | CLIP 기반 객체 판정을 grounded detector와 VQA 질문으로 보강한다 | 신규 `evaluators/presence_backends.py`, `evaluators/presence_calibration.py`; 기존 `vqa_backend.py`/`hallucination_vqa.py` 재사용 | 한계 2·3, 슬라이드 6·9 | 9~10월 | verifier 판정의 오탐/미탐 사례가 줄어드는 정성·정량 결과가 나옴 — **완료(코드 기반 스캐폴드, 5차)**: 공통 presence backend 인터페이스(clip/owlv2/vqa/gt/mock) + ensemble calibrator + `PacketVerifier` 보강 연결. 🟡 **실제 OWLv2/VQA weight로 오탐/미탐이 실제로 줄었는지는 검증되지 않음** — mock backend 기준 구조 테스트만 통과 |
| 9 ✅ 코드 기반 스캐폴드 | 슬라이드 5 `시간축 평가` / 슬라이드 9 `held-out 최종 평가 지표` | 2단계에서 만든 CLIP 기반 temporal 지표를 OWLv2/VQA 보강 기준으로 다시 계산한다 | 신규 `pipelines/heldout_remeasurement.py`, `scripts/remeasure_video_metrics.py` | 한계 1·3, 슬라이드 5·9 | 10~11월 | `PTC`/`SFR`/`SDI` 결과가 검출기 보강 전후로 비교됨 — **완료(코드 기반 스캐폴드, 5차)**: clip_only vs calibrated 재측정 파이프라인이 동작하고 `metric_delta`가 생성됨. 🟡 계산 구조만 검증됨 — 실제 검출기 보강으로 지표가 개선됐다는 결과는 아직 없음(기본 설정에서는 calibrated == clip_only) |
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

### 단계별 구현 묶음과 완료 확인 기준

위 표의 0~12번을 그대로 직렬로만 처리하면 generate 분기보다 verifier가 늦게 붙는 문제가
생긴다. 실제 개발은 아래 묶음 단위로 진행한다. 핵심 원칙은 **비디오 기반을 먼저 만들고,
생성 분기를 붙이기 전에 packet verifier를 먼저 준비하는 것**이다.

| 단계 | 포함 순서 | 구현할 것 | 완료 확인 |
|---|---|---|---|
| 1차 ✅ 구현 완료 (2026-07) | 0~4 | Presence threshold 배선, mp4/frame IO, `PTC`/`SFR`/`SDI`(CLIP/packet 기반 잠정 지표 — 5차 재측정 필요), motion-aware gate(기본 OFF, 실데이터 튜닝 후속), segment 구조 | 테스트 영상 입력 후 복원 frame/mp4가 생성되고, `temporal_metrics.csv`에 `PTC`/`SFR`/`SDI`가 기록되며, motion이 큰 구간이 reuse되지 않는지 로그로 확인 — `tests/test_video.py`·`tests/test_video_io.py`·`tests/test_evaluators.py`로 검증됨 (아래 "1차 구현 결과" 참조) |
| 2차 ✅ 구현 완료 (2026-07) | 7 | `Packet Verifier`, 전송 packet과 복원 packet 비교, 오류 유형별 리포트, regeneration controller 기본 구조 | 추가/누락/왜곡 객체가 report에 분리 기록되고, 오류 유형별 controller decision 로그가 남음 — `tests/test_packet_matcher.py`·`tests/test_controllers.py`·`tests/test_video.py`로 검증됨 (아래 "2차 구현 결과" 참조) |
| 3차 ✅ 구현 완료 (2026-07) | 5 | `video_generator` 인터페이스, `reuse`/`recompute`/`generate` 3-way 분기, start-only generation 경로 | config에서 generate를 켰을 때 inter-frame 일부가 generate branch로 들어가고, 생성 프레임이 저장됨 — `tests/test_video.py::TestGenerateBranch`·`tests/test_video_generator.py`로 검증됨 (아래 "3차 구현 결과" 참조) |
| 4차 ✅ 구현 완료 (2026-07) | 6 | start+end keyframe 조건을 받는 bidirectional generation mode | start-only와 start+end 결과를 같은 영상에서 비교하고, `SFR`/`SDI`/flicker가 별도 CSV로 기록됨 — `tests/test_video_generator.py`·`tests/test_video.py::TestBidirectionalGenerateBranch`·`TestGenerationModeComparison`로 검증됨 (아래 "4차 구현 결과" 참조) |
| 5차 ✅ 코드 기반 스캐폴드 구현 완료 (2026-07) — 🟡 실제 모델 검증 필요 | 8~10 | OWLv2/VQA verifier 보강 인터페이스, held-out temporal metric 재측정 파이프라인, GT/VLM 기반 SRS 보정 스캐폴드 | CLIP-only 결과와 보강(calibrated) 결과가 비교 리포트로 나오고, loop-internal 지표와 held-out 지표가 분리되어 저장됨 — `tests/test_presence_backends.py`·`tests/test_heldout_remeasurement.py`·`tests/test_temporal_srs_calibration.py`·`tests/test_packet_matcher.py::TestPacketVerifierPresenceCalibration`로 검증됨 (아래 "5차 구현 결과" 참조) |
| 6차 ✅ PoC 구현 완료 (2026-07) — 🟡 실제 bitstream 검증 필요 | 11~12 | channel-symbol/bit accounting PoC, naive baseline 비교, rate-reliability trade-off 리포트 | 절감률 vs `SRS`/`PTC`/`SFR`/`SDI`/severity 곡선이 생성되고, symbol/bit 계산 로그와 baseline 비교 표가 생성됨 — `tests/test_transmission_accounting.py`로 검증됨 (아래 "6차 구현 결과" 참조) |

각 단계의 최소 산출물은 다음과 같이 둔다.

| 단계 | 최소 산출물 |
|---|---|
| 1차 | 복원 mp4 또는 frame folder, `temporal_frames.csv`, `temporal_metrics.csv`, keyframe/segment 구조 JSON, motion gate decision log |
| 2차 | `packet_match_report.json` 또는 CSV, 오류 유형별 additional/missing/distorted 기록, controller decision log |
| 3차 | `reuse`/`recompute`/`generate` 분기 로그, generated frames, generate ON/OFF 비교 metric CSV |
| 4차 | start-only vs bidirectional 비교 CSV, `SFR`/`SDI`/flicker 비교 결과, drift 감소 여부 리포트 |
| 5차 | CLIP-only vs calibrated verifier 비교 리포트(`metric_delta.json`), temporal metric 재측정 결과(`clip_only_metrics`/`calibrated_metrics`), Temporal SRS Calibration weight 설정/저장 포맷 — 실제 OWLv2/VQA weight 검증 결과는 미포함 |
| 6차 | `frame_accounting.json/csv`, `segment_accounting.json/csv`, `accounting_summary.json`(bit/symbol 절감률, naive baseline 대비), `rate_reliability_summary.json`/`rate_reliability_curve.csv` — 실제 CBR/표준 bitstream 검증 결과는 미포함 |

완료 기준은 "코드가 실행된다"가 아니라 **각 단계 결과가 파일로 남고, 이전 단계와 비교
가능한 로그/CSV가 생성되는지**로 판단한다.

### 1차 구현 결과 (2026-07)

> 상세 검증 기록(기준 커밋, 로컬/원격/실모델 실행 결과, 산출물)은
> [etri_stage1_validation.md](./etri_stage1_validation.md) 참조.

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

주의: 1차의 `PTC`/`SFR`/`SDI`와 presence 판정은 **CLIP/packet 기반 잠정 구현**이다
(위 "후속 단계가 앞 단계의 잠정 구현을 마무리하는 방식" 표 참조). 최종 주장 전에
5차 OWLv2/VQA 보강 후 재측정해야 한다.

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
| 출력 파일 | `pipelines/packet_verification.py::write_reports` | `packet_match_report.json`/`.csv`, `controller_decisions.json`/`.csv` (경로는 `configs/video/default.yaml`의 `verifier.report_json`/`report_csv`/`decisions_json`/`decisions_csv`) |

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
| Pipeline 연결(옵션, 기본 OFF) | `scripts/evaluate_video.py` — `use_video_gen`(phase4-gated) + `video_generator.enabled` 둘 다 true일 때만 `video_generator.build_generator(cfg)`로 backend를 만들고 `TemporalPipeline`에 주입. 끄면 `TemporalPipeline`은 `video_generator`를 만들지도, 호출하지도 않음 | `configs/video/default.yaml`의 `use_video_gen`/`video_generator.*` 블록 (기본 전부 OFF) |
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
| config | `configs/video/default.yaml`의 `video_generator.conditioning_mode: bidirectional` 허용 + `bidirectional_missing_end_policy`/`comparison_enabled`/`comparison_output`/`comparison_start_only_csv`/`comparison_bidirectional_csv` 추가(기본값 전부 기존 동작 보존 — `conditioning_mode: start_only`, `comparison_enabled: false`) | `config.py::_NESTED_PATH_KEYS`에 비교 산출물 경로 3종 등록 |

실제 CLI 경로(`scripts/evaluate_video.py --no-models --captions ...`)로 합성 8프레임
2-GOP 시퀀스(마지막 GOP는 end keyframe 없음)를 `conditioning_mode: bidirectional` +
`comparison_enabled: true`로 돌려 다음을 수동으로도 확인했다: 첫 GOP의 inter-frame이
진짜 bidirectional 보간(`relative_position=0.25`, `end_keyframe_index=4`)으로 기록됨,
마지막 GOP의 inter-frame은 `bidirectional_missing_end_policy: fallback_start_only`에
따라 `conditioning_mode=start_only`로 강등되어 기록됨, `generation_mode_comparison.json`
+ 두 개의 `temporal_metrics_*.csv`가 생성됨, generate를 아예 끈 기본 config
(`configs/composed_video.yaml`)는 동일 입력에서 `n_generate=0`으로 3차 이전과 동일하게
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

5차(순서 8~10 중 코드로 완결 가능한 범위)는 **구현 완료**됐다. 5차의 성격은
**"CLIP-only verifier에 OWLv2/VQA/GT/VLM judge를 꽂을 수 있는 인터페이스와, 1~4차
결과를 재측정할 수 있는 held-out 파이프라인을 만드는 것"**이며, **실제 OWLv2/VQA
모델의 성능(오탐/미탐 감소, 지표 개선)을 검증한 것이 아니다.** 🟡 표시된 항목은
모두 "구조는 동작하지만 실 모델로 검증되지 않음"을 뜻한다.

구현 내용과 산출물 ↔ 코드 대응:

| 항목 | 코드 | 산출물 |
|---|---|---|
| Presence backend 인터페이스 | 신규 `evaluators/presence_backends.py` — `PresenceResult`(object_name/present/confidence/backend/evidence) + `PresenceBackend` 공통 인터페이스. `MockPresenceBackend`(packet 기반, 의존성 없음), `ClipPresenceBackend`(기존 CLIP 판정을 인터페이스 뒤로 감쌈, 재구현 아님), `Owlv2PresenceBackend`/`VqaPresenceBackend`(lazy-load, 미설치 시 `PresenceBackendUnavailableError`로 명확히 실패 — VQA는 기존 `vqa_backend.py`/`hallucination_vqa.py` 재사용), `GtPresenceBackend`(주석 dict 조회) | `build_presence_backend()` 레지스트리; 테스트는 `mock`/`gt`/stub만 사용, 실제 OWLv2/VQA weight는 테스트 의존성 아님 |
| Presence ensemble / calibration | 신규 `evaluators/presence_calibration.py` — `PresenceCalibrator`가 `clip_only`/`owlv2_only`/`vqa_only`/`gt_only`/`ensemble_majority`/`ensemble_weighted` 6개 모드로 backend 결과를 조합. 기본값 `clip_only`(비활성 시 아무 backend도 안 만듦) | `CalibratedPresence`(final_present/final_confidence/contributing_backends/per_backend, dict/JSON 직렬화) |
| Packet Verifier 보강 | `evaluators/packet_verifier.py` — `PacketVerifier(presence_calibrator=None, metric_role="loop_internal")`. **기본값(`presence_calibrator=None`)에서는 결과가 2~4차와 완전히 동일**. calibrator + `reconstructed_image`가 모두 주어졌을 때만 missing/additional 판단을 재검증해 보정. report에 `raw_clip_result`(원본 스냅샷)와 `calibrated_presence_result`(보정 근거, 미보정 시 `None`)를 분리 저장, 모든 report에 `metric_role`(`loop_internal`/`held_out`) 태그 | `pipelines/packet_verification.py`가 `verifier.use_presence_calibration`(기본 false)로 연결 — 켜면 `packet_match_report`에 3개 컬럼(`metric_role`/`raw_clip_result`/`calibrated_presence_result`)이 추가되고, 실제 재구성 이미지가 calibrator에 전달됨 |
| Held-out 재측정 파이프라인 | 신규 `pipelines/heldout_remeasurement.py` — `items_from_temporal_records()`(TemporalPipeline 결과, 이미지 포함 → 모든 backend로 재검증 가능)/`items_from_saved_packets()`(디스크에 저장된 packet JSON 쌍, 이미지 없음 → `mock`/`gt`만 가능) 두 입력 경로. `PacketVerifier`/`PresenceCalibrator`가 `image=None`이어도 image-free backend는 정상 동작하도록 배선되어 있고, `RemeasurementItem.gt_metadata`가 실제로 `gt` backend에 전달됨(각 backend의 `check()`가 `gt_metadata`를 받는 공통 시그니처로 통일). `remeasure()`가 clip_only/calibrated 양쪽으로 `PacketVerifier` report + `PTC`/`SFR`/`SDI`(`evaluators/temporal_consistency.py` 재사용)를 계산하고 `metric_delta`(항목별 clip_only/calibrated/diff)를 생성 | `clip_only_metrics.json/csv`, `calibrated_metrics.json/csv`, `metric_delta.json/csv` — 신규 `scripts/remeasure_video_metrics.py`(`--from-run`/`--from-packets` 두 입력 모드; `--from-run`은 원본 run의 3~4차 generate/bidirectional 설정도 동일 config로 재구성하지만, 재구성 자체가 원 실행을 그대로 재생하는 것은 아니므로 특정 run의 byte-정확한 재현이 필요하면 `--from-packets`를 권장) |
| Temporal SRS Calibration 스캐폴드 | 신규 `evaluators/temporal_srs_calibration.py` — `TemporalSRSCalibration`(SRS/temporal-SRS 가중치 로드/저장), `fit_weights_least_squares()`(GT/VLM `target_score`가 주어졌을 때의 least-squares weight fitting — **실제 VLM 호출은 없음**, 호출자가 target을 공급) | `save()`/`load()` JSON 포맷; `tests/test_temporal_srs_calibration.py`에서 synthetic target으로 정확한 weight 복원 확인 |
| config | `configs/video/default.yaml`의 `verifier.use_presence_calibration`/`presence_mode`/`presence_backends`/`presence_backend_weights`/`metric_role`, 신규 `heldout.*`, `temporal_srs_calibration.*` 블록 (기본값 전부 OFF/`clip_only`) | `config.py::_NESTED_PATH_KEYS`에 `heldout.*`/`temporal_srs_calibration.weights_output` 등록 |

실제 CLI 경로(`scripts/remeasure_video_metrics.py --from-run --no-models --captions ...`,
`--from-packets <dir>` 둘 다)로 합성 프레임 시퀀스를 돌려 `clip_only_metrics`/
`calibrated_metrics`/`metric_delta` 파일이 생성되고, presence calibration을 켠
`evaluate_video.py` 실행에서 `packet_match_report`에 새 컬럼이 추가되는 것을
수동으로도 확인했다(단위 테스트와 별개의 엔드투엔드 스모크 확인).

주의(가장 중요): 5차의 모든 산출물은 **mock 또는 CLIP-only 기준**이다.
`presence_backends: ["mock"]`(또는 backend 미설정)로는 `calibrated` 결과가
`clip_only`와 항상 동일하다 — 이는 버그가 아니라 "보정할 실 데이터가 없으면 원본과
같아야 한다"는 의도된 sanity-check 동작이다. **실제 OWLv2/VQA weight를 연결해
오탐/미탐이 실제로 줄었는지, `PTC`/`SFR`/`SDI`가 실제로 달라지는지, Temporal SRS
Calibration이 실제 GT/VLM 판단과 맞는지는 전혀 검증되지 않았다** — 이는 5차의
의도된 범위 제한이며(코드 인터페이스 구축이 목표), 실 모델 통합·정량 검증은 후속
과제다.

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
| Pipeline 연결(옵션, 기본 OFF) | `scripts/evaluate_video.py` — `accounting.enabled`/`rate_reliability.enabled`(둘 다 기본 false). 켜도 `TemporalPipeline`의 재구성/결정/기존 지표는 전혀 바뀌지 않고 산출물만 추가됨(수동 회귀 확인: 동일 입력에서 accounting 끈 실행과 켠 실행의 `temporal_metrics.csv`/`segments.json`이 byte-동일) | `configs/video/default.yaml`의 `accounting.*`/`rate_reliability.*` 블록, `config.py::_NESTED_PATH_KEYS`에 경로 등록(OmegaConf `${accounting.output_dir}` 보간 사용) |
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

### 후속 단계가 앞 단계의 잠정 구현을 마무리하는 방식

각 단계는 독립된 새 기능만 추가하는 것이 아니다. 앞 단계에서 만든 구현 중 일부는 당시
사용 가능한 검출기와 지표에 의존한 **잠정 구현**이며, 뒤 단계에서 더 강한 verifier,
held-out 평가, 전송량 accounting을 붙이면서 재측정하고 최종화한다.

| 앞 단계의 잠정 구현 | 왜 잠정인가 | 어느 후속 단계에서 마무리하는가 | 마무리 방식 |
|---|---|---|---|
| 1차 `Presence threshold`와 객체 존재 판정 | CLIP probe 기반 threshold는 객체 오탐/미탐에 민감함 | 5차(8~10) | OWLv2/VQA 기반 Presence Calibration을 붙이고, 기존 CLIP-only 결과와 보강 결과를 비교한다 |
| 1차 `PTC`/`SFR`/`SDI` 초기 지표 | 프레임별 packet/object 판정이 흔들리면 시간축 지표도 흔들림 | 5차(8~10) | OWLv2/VQA 보강 후 temporal metric을 재계산하고, loop-internal 지표와 held-out 최종 지표를 분리한다 |
| 1차 motion-aware segment 판단 | residual 기반 motion은 optical flow보다 거칠고, semantic delta와 결합 threshold가 초기값임 | 4차(6), 5차(8~10) | bidirectional generation 결과의 drift/flicker를 보고 threshold를 조정하고, 보강 verifier 기준으로 segment decision 품질을 재검증한다 |
| 2차 `Packet Verifier`와 regeneration controller | 초기 controller는 rule-based decision log 수준이며, 검출기 신뢰도에 의존함 | 5차(8~10) | OWLv2/VQA verifier와 held-out 지표로 error-type 판단을 재검증하고, 필요 시 controller rule을 조정한다 |
| 3~4차 generate/bidirectional 결과 | 생성 결과는 open-loop로 좋아 보일 수 있고 hallucination이나 drift가 숨어 있을 수 있음 | 5차(8~10) | `Packet Verifier`, `PTC`/`SFR`/`SDI`, held-out 평가로 generate ON/OFF와 start-only/bidirectional 결과를 다시 비교한다 |
| 1~4차의 semantic-unit 절감 효과 | semantic unit 절감은 실제 channel-symbol 또는 bit 절감과 다름 | 6차(11~12) | channel-symbol 절감률, symbol/pixel, bpp 설계안을 붙여 전송량 기준 결과로 재정리한다 — **완료(PoC, 6차)**: `accounting/bit_accounting.py`/`pipelines/transmission_accounting.py`가 bit/channel-symbol 절감을 semantic-unit 절감과 분리해 계산(아래 "6차 구현 결과" 참조). 🟡 실제 CBR/표준 bitstream 검증은 아님 |

> **5차 진행 상태 참고**: 위 표에서 "5차(8~10)"가 마무리 단계로 적힌 항목들은 5차에서
> **인터페이스와 파이프라인 구조**(presence backend, calibrator, held-out
> remeasurement, SRS calibration 스캐폴드)까지는 구현·테스트됐다. 하지만 실제
> OWLv2/VQA weight를 연결해 "정말로 오탐/미탐이 줄었는지", "PTC/SFR/SDI가 실제로
> 달라지는지" 재검증하는 작업 자체는 **아직 수행되지 않았다** — mock backend 기준
> 구조 테스트만 통과한 상태다. 즉 위 "마무리 방식" 문장의 도구는 준비됐지만, 그
> 도구로 실제 마무리(재검증)를 수행하는 것은 실 모델 통합 이후의 후속 작업이다.

> **6차 진행 상태 참고**: channel-symbol/bit accounting과 rate/reliability
> trade-off 리포트는 **구조와 계산 자체는 완료·테스트**됐지만, 이는 실제
> 통신 표준 bitstream/CBR을 재현한 결과가 아니라 이 저장소 안에서 계산 가능한
> 실제 데이터(패킷 JSON, caption, 프레임 shape)와 명시적 proxy(latent 아키텍처
> 상수, CR 비율, quantization 가정)로 구성된 accounting PoC다. "절감률이
> 몇 %다"라는 숫자 자체는 나오지만, 그 숫자가 실제 무선 채널의 bit/symbol
> 소비량과 일치한다는 검증은 아니다.

따라서 1차 완료는 "최종 평가 체계 완성"이 아니라 **동작하는 비디오/시간축 평가 뼈대
확보**를 뜻한다. 최종 주장은 5차에서 검출기와 평가 지표를 보강하고, 6차에서 전송량
기준(PoC accounting)을 붙인 뒤에도, 실제 bitstream/CBR 재현과 실 모델 검증이라는
후속 단계를 거쳐야 한다.

## 월별 추진계획

| 시기 | 초점 | 산출물 |
|---|---|---|
| 7월 | SNR 스윕 본실험 + channel-symbol 절감 1차 PoC | SNR 결과 CSV/곡선, 심볼 절감 PoC 결과 |
| 8월 | 중간진도점검 | 절감 효과 1차 결과, `PTC`/`SFR`/`SDI` 정의와 초기 결과 |
| 9월 | 영상 파이프라인 고도화 + verifier 연동 착수 | motion gate, mp4 IO, segment 구조, VQA/OWLv2 연동 초기 결과 |
| 10월 | 페이딩 견고성 + bit 설계안 + verifier 고도화 | Rayleigh/페이딩 결과, bit 기준 설계안, packet verifier 고도화 결과 |
| 11월 | 공정 비교 + 최종 정리 | held-out 평가 기반 최종 실험, 비교 프로토콜, 최종 보고서 |

학습형 adapter/critic과 `Temporal SRS Calibration`은 1차 필수 구현이 아니라 고도화/후속
항목으로 관리한다.

## 신규 연구 아이템 확장 가능성

1. **전송률 적응형 영상 시맨틱 통신.** 채널 상태와 영상 변화량에 맞춰 꼭 필요한 만큼만
   전송한다. `semantic delta + channel-symbol` 절감과 `PTC/SFR/SDI` 기반
   rate-semantic reliability trade-off 분석이 핵심이다.
2. **수신단 신뢰성 제어형 생성 복원.** 원본 없이 수신단이 전송 packet 기준으로 복원 결과를
   자체 검증하고, hallucination critic/regeneration controller로 재복원을 제어한다.
3. **시맨틱 전송 평가 벤치마크.** `PTC/SFR/SDI`, VQA/OWLv2 검증, held-out 평가,
   SRS 고도화를 묶어 시맨틱 통신 연구용 공정 비교 프로토콜로 확장한다.

## ETRI 협의 필요사항

| 항목 | 협의 내용 |
|---|---|
| 전송량 평가 단위 | channel symbol 수, symbol/pixel 비율, bpp 중 우선 보고 단위 결정 |
| 영상 연구 범위 | keyframe 기반 PoC 수준인지, 실제 비디오 코덱/모션 보상까지 포함할지 결정 |
| 채널 범위 | AWGN 중심인지, Rayleigh/fast fading을 필수 범위로 포함할지 결정 |
| 비교 기준 모델 | WITT, DiffJSCC, Deep-JSCC 중 필수 비교군과 공정 비교 조건 결정 |
| 재복원 평가 기준 | oracle 상한, Rx-legal self-verification, held-out 최종 평가의 구분 방식 결정 |
| 평가 데이터셋·시간축 지표 | 장면 전환/객체 등장·소멸이 있는 영상 데이터와 `PTC/SFR/SDI` 보고 방식 결정 |
| 산출물·라이선스 | 최종 납품에 포함할 오픈소스, 모델 가중치, 라이선스 범위 결정 |

특히 **전송량 평가 단위**, **비교 조건**, **시간축 지표 보고 방식**은 8월 진도점검
결과물의 형태를 결정하므로 우선 협의 대상이다.

## 현재 구현 상태

| 묶음 | 상태 | 요약 |
|---|---|---|
| 원본 경로/모듈화 | 완료 | 원본 경로 보존, 모듈 구조 정리, End-to-End 평가 골격 |
| 의미 평가 | 완료에 근접 | 품질·CLIP·패킷·VQA 지표와 `srs_base/srs_packet/srs_v2` 연결; presence threshold/uncertain band 배선 완료(1차 순서 0, CLIP probe는 잠정) |
| 할루시네이션 검증 | 부분(코드 스캐폴드 확장, 5차) | packet verifier(이미지 경로 + `evaluators/packet_verifier.py` severity, 2차; presence-backend 보강 연결, 5차), 오류 유형별 controller(`controllers/verifier_controller.py`, 2차), presence backend 인터페이스(`evaluators/presence_backends.py`) + ensemble calibrator(`evaluators/presence_calibration.py`) + held-out 재측정(`pipelines/heldout_remeasurement.py`), regeneration search, VQA, SRS-v2는 있으나 일부 판단은 휴리스틱이고, 5차 presence backend는 아직 CLIP-only/mock 기준으로만 검증됨(실제 OWLv2/VQA weight 미검증) — candidate action도 아직 sampler에 미주입 |
| 비디오 확장 | 1~4차 완료(기초) | keyframe/scene-change/temporal evaluator + mp4 IO(`utils/video_io.py`), motion 이중 게이트, segment 구조(`video/segment.py`), `PTC`/`SFR`/`SDI`(잠정), reuse/recompute/generate 3-way 분기 + mock start-only/bidirectional video_generator(`video/video_generator.py`, 3~4차), start-only vs bidirectional 비교 파이프라인(`pipelines/generation_mode_comparison.py`, 4차) — 실제 생성 모델 통합·품질/drift 검증은 5차 이후 후속 |
| 전송량 절감 | 계획/PoC | semantic unit 절감은 가능하나 channel-symbol 절감은 1차 PoC 대상, bit 기준은 설계안 대상 |
| 채널/비교/저지연 | 부분/스캐폴드 | guide damage, edge codec, low-latency, channel conditioning은 연결됐지만 일부는 placeholder |

## 모듈 매핑

- 시간축·영상 확장: `video/keyframe_extractor.py`, `video/scene_change_detector.py`,
  `video/motion_residual.py`, `video/temporal_pipeline.py`, `video/segment.py`,
  `video/video_generator.py` (3차 mock start-only + 4차 mock bidirectional generate backend),
  `pipelines/generation_mode_comparison.py` (4차, start-only vs bidirectional 비교),
  `utils/video_io.py`, `evaluators/temporal_consistency.py`
- 할루시네이션 완화·검출: `guidance/semantic_packet`, `evaluators/hallucination*`,
  `evaluators/semantic_packet_matcher.py`, `evaluators/packet_verifier.py` (2차 + 5차 보강),
  `controllers/regeneration*`, `controllers/verifier_controller.py` (2차),
  `pipelines/packet_verification.py` (2차 + 5차 보강),
  `evaluators/presence_backends.py` / `evaluators/presence_calibration.py` (5차, CLIP/OWLv2/VQA/GT/mock 공통 인터페이스),
  `pipelines/heldout_remeasurement.py` / `scripts/remeasure_video_metrics.py` (5차, held-out 재측정),
  `evaluators/temporal_srs_calibration.py` (5차, SRS 가중치 스캐폴드)
- 의미 충실도 평가: `evaluators/clip_score.py`,
  `evaluators/object_preservation.py`, `evaluators/semantic_reliability*.py`
- 채널/전송량/보조 축: `channels/`, `controllers/adaptive_guidance.py`,
  `models/diffusion_wrapper_channel.py`, `acceleration/`

## 관련 문서

- [etri_overview.md](./etri_overview.md)
- [phase4.md](./phase4.md)
- [phase5.md](./phase5.md)
- [training_scaffold.md](./training_scaffold.md)
- [video_extension_lgvsc.md](./video_extension_lgvsc.md)
- 보관본: [archive/etri_development_plan_v2.md](./archive/etri_development_plan_v2.md),
  [archive/etri_development_roadmap.md](./archive/etri_development_roadmap.md),
  [archive/limitation_reference_map.md](./archive/limitation_reference_map.md)

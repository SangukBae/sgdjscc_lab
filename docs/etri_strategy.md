> [← 문서 색인](./README.md)

# ETRI 전략 정리

이 문서는 기존 [개발계획서 보관본](./archive/etri_development_plan_v2.md),
[로드맵 보관본](./archive/etri_development_roadmap.md),
[한계점 지도 보관본](./archive/limitation_reference_map.md)을 합친 **ETRI 과제용 단일
전략 문서**다. 목적은 문서 수를 줄이면서도 "무엇이 핵심 문제이고, 무엇을 먼저
구현·평가해야 하는가"를 한 곳에서 보게 하는 것이다.

이 문서의 한계 번호와 개발 방향은 `ETRI_연구진행상황_1차공유_20260715_v2.pptx`의
구성에 맞춘다.

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
| 1 | 현재 기본 경로는 정지 이미지 중심이며, 비디오 입력·mp4 출력·세그먼트 단위 복원은 확장 항목 | `video_extension_lgvsc.md`, PPT 슬라이드 5 |
| 2 | reuse/recompute 결정이 의미 델타 중심이라, 의미는 같지만 픽셀 모션이 큰 카메라 pan/zoom을 놓칠 수 있음 | `video/temporal_pipeline.py`, `video/motion_residual.py` 미통합 |
| 3 | reuse는 keyframe 복원 결과를 재사용하므로, 긴 구간에서는 drift와 객체 깜빡임을 별도 지표로 봐야 함 | temporal evaluator 필요 |

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
| 1 | `PSNR`·`SSIM`은 프레임 화질 중심이라 객체·관계·장면 의미와 시간축 일관성을 설명하지 못함 | PPT 슬라이드 7 |
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
| 0 | 슬라이드 6 `Packet Verifier` / 슬라이드 7 `Presence Calibration` | 객체 존재 판정 threshold가 실제 evaluator에 전달되게 하고, 히스테리시스·uncertain band를 옵션으로 둔다 | `object_preservation.py`, `hallucination.py`, `semantic_reliability*.py` | 한계 2·3, 슬라이드 6·7 | 7~8월 기반 작업 | 기존 기본값에서 결과가 깨지지 않고, threshold 변경이 실제 지표에 반영됨 |
| 1 | 슬라이드 5 `입력 영상(mp4)` / `세그먼트 연결 → 복원 영상(mp4)` | mp4를 프레임으로 풀고, 복원 프레임을 다시 mp4로 저장한다 | 신규 `video_io`, `evaluate_video.py` 확장 | 한계 1, 슬라이드 5 | 9월 | 테스트 영상 1개를 입력해 복원 mp4와 프레임별 로그가 생성됨 |
| 2 | 슬라이드 5 `시간축 평가` / 슬라이드 7 `PTC·SFR·SDI` | temporal SRS와 별도로 packet consistency, 객체 깜빡임, semantic drift를 계산한다 | `evaluators/temporal_consistency.py`, `video/temporal_pipeline.py` | 한계 1·3, 슬라이드 5·7 | 8월 | `temporal_metrics.csv`에 `PTC`/`SFR`/`SDI` 초기값이 기록됨 |
| 3 | 슬라이드 5 `세그먼트 판단 게이트` / `의미 델타 + 모션 이중 게이트` | semantic delta만 보던 reuse/recompute 판단에 keyframe 대비 motion residual을 추가한다 | `video/semantic_delta.py`, `video/motion_residual.py`, `video/temporal_pipeline.py` | 한계 1, 슬라이드 5 | 9월 | 의미 변화는 작지만 카메라 이동이 큰 구간을 reuse하지 않음 |
| 4 | 슬라이드 5 `키프레임` / `비-키프레임` / `세그먼트` 구조 | 프레임 단위 처리를 GOP/segment 단위 처리로 묶어 generate 분기를 붙일 수 있게 한다 | `video/keyframe_extractor.py`, 신규 segment record | 한계 1, 슬라이드 5 | 9월 | 기존 frame-wise 결과와 segment 결과가 동등하게 재현됨 |
| 5 | 슬라이드 5 `Generate (신규)` | 시작 keyframe, caption, side-info를 조건으로 세그먼트 생성 경로를 붙인다 | 신규 `video_generator`, `use_video_gen` config | 한계 1·2, 슬라이드 5·6 | 9~10월 | `reuse/recompute/generate` 3-way 분기가 동작하고 생성 결과가 저장됨 |
| 6 | 슬라이드 5 `Generate (start / start+end 양방향)` | 시작 keyframe과 끝 keyframe을 모두 조건으로 넣어 drift를 줄일 수 있는지 확인한다 | `video_generator` bidirectional mode | 한계 1, 슬라이드 5 | 10월 이후 | start-only 대비 drift/flicker가 줄어드는지 비교 결과가 나옴 |
| 7 | 슬라이드 6 `Packet Verifier` / `오류 유형별 재생성 Controller` | 전송 packet과 복원 packet을 비교하고, 추가·누락·왜곡별로 재생성 조건을 다르게 조정한다 | `evaluators/packet_matcher.py`, `controllers/regeneration*` | 한계 2, 슬라이드 6 | 9~10월 | 오류 유형별 report와 재생성 로그가 생성됨 |
| 8 | 슬라이드 6 `Packet Verifier` 보강 / 슬라이드 7 `Presence Calibration` | CLIP 기반 객체 판정을 grounded detector와 VQA 질문으로 보강한다 | OWLv2/VQA backend, `hallucination_vqa.py` | 한계 2·3, 슬라이드 6·7 | 9~10월 | verifier 판정의 오탐/미탐 사례가 줄어드는 정성·정량 결과가 나옴 |
| 9 | 슬라이드 5 `시간축 평가` / 슬라이드 7 `held-out 최종 평가 지표` | 2단계에서 만든 CLIP 기반 temporal 지표를 OWLv2/VQA 보강 기준으로 다시 계산한다 | temporal evaluator 재실행, report 비교 | 한계 1·3, 슬라이드 7 | 10~11월 | `PTC`/`SFR`/`SDI` 결과가 검출기 보강 전후로 비교됨 |
| 10 | 슬라이드 7 `Temporal SRS Calibration` / `held-out 최종 평가 지표` | GT 객체 주석과 VLM 판단을 이용해 SRS/Temporal SRS의 가중치를 보정한다 | GT metadata loader, VLM judge, SRS weight config | 한계 3, 슬라이드 7 | 10~11월 | loop-internal 지표와 held-out 최종 지표가 분리되어 보고됨 |
| 11 | 슬라이드 8 `1차 — 채널 심볼 절감 PoC` | 변화가 작은 latent/semantic 요소를 덜 보내고, 의미 보존 저하와 절감률의 관계를 본다 | latent/packet masking, symbol accounting PoC | 슬라이드 8·10 | 7~8월 | 절감률 vs SRS/PTC 곡선이 생성됨 |
| 12 | 슬라이드 8 `2차 — 비트 기준 설계안` / 슬라이드 10 `평가 벤치마크` | 실제 bitrate/CBR 산정 방식, adaptive keyframe policy, DISTS/downstream 비교 지표를 정리한다 | bit accounting, comparison profile, report scripts | 슬라이드 8·10·11 | 10~11월 | 최종 보고용 비교 프로토콜과 bit 기준 설계안이 정리됨 |

PPT 블록 기준 큰 흐름은 다음과 같다.

| PPT 블록 묶음 | 포함 순서 | 의미 |
|---|---|---|
| 슬라이드 6·7 검증/평가 블록 | 0, 7~10 | 비디오 지표와 verifier가 의존할 최소 판정 기준을 맞추고, 최종 평가는 held-out으로 분리한다 |
| 슬라이드 5 비디오 확장 블록 | 1~6 | mp4 입출력, 시간축 지표, motion-aware 판단, segment 구조, generate 분기를 만든다 |
| 슬라이드 8 전송량 절감 블록 | 11~12 | ETRI 문의사항인 channel-symbol 절감과 bit 기준 설계안을 정리한다 |

**게이트 원칙:** 수치에 영향 없는 순수 배선(threshold 전달 등)은 게이트 불필요. 새 판정
로직, 새 backend(OWLv2·VQA), negative-prompt 재생성, generate 분기, 학습형 adapter/critic은
phase/config gate 뒤에 두고, 기본값에서 원본 SGD-JSCC 경로와 동일하게 동작하도록 관리한다.

### 단계별 구현 묶음과 완료 확인 기준

위 표의 0~12번을 그대로 직렬로만 처리하면 generate 분기보다 verifier가 늦게 붙는 문제가
생긴다. 실제 개발은 아래 묶음 단위로 진행한다. 핵심 원칙은 **비디오 기반을 먼저 만들고,
생성 분기를 붙이기 전에 packet verifier를 먼저 준비하는 것**이다.

| 단계 | 포함 순서 | 구현할 것 | 완료 확인 |
|---|---|---|---|
| 1차 | 0~4 | Presence threshold 배선, mp4/frame IO, `PTC`/`SFR`/`SDI`, motion-aware gate, segment 구조 | 테스트 영상 입력 후 복원 frame/mp4가 생성되고, `temporal_metrics.csv`에 `PTC`/`SFR`/`SDI`가 기록되며, motion이 큰 구간이 reuse되지 않는지 로그로 확인 |
| 2차 | 7 | `Packet Verifier`, 전송 packet과 복원 packet 비교, 오류 유형별 리포트, regeneration controller 기본 구조 | 추가/누락/왜곡 객체가 report에 분리 기록되고, 오류 유형별 controller decision 로그가 남음 |
| 3차 | 5 | `video_generator` 인터페이스, `reuse`/`recompute`/`generate` 3-way 분기, start-only generation 경로 | config에서 generate를 켰을 때 inter-frame 일부가 generate branch로 들어가고, 생성 프레임이 저장됨 |
| 4차 | 6 | start+end keyframe 조건을 받는 bidirectional generation mode | start-only와 start+end 결과를 같은 영상에서 비교하고, `SFR`/`SDI`/flicker가 별도 CSV로 기록됨 |
| 5차 | 8~10 | OWLv2/VQA verifier 보강, temporal metric 재측정, GT/VLM 기반 SRS 보정 | CLIP-only 결과와 OWLv2/VQA 보강 결과가 비교 리포트로 나오고, loop-internal 지표와 held-out 지표가 분리되어 저장됨 |
| 6차 | 11~12 | channel-symbol 절감 PoC, bit accounting 설계, 비교 프로토콜 | 절감률 vs `SRS`/`PTC` 곡선이 생성되고, symbol/pixel 또는 bpp 계산 로그와 최종 비교 조건 표가 생성됨 |

각 단계의 최소 산출물은 다음과 같이 둔다.

| 단계 | 최소 산출물 |
|---|---|
| 1차 | 복원 mp4 또는 frame folder, `temporal_frames.csv`, `temporal_metrics.csv`, keyframe/segment 구조 JSON, motion gate decision log |
| 2차 | `packet_match_report.json` 또는 CSV, 오류 유형별 additional/missing/distorted 기록, controller decision log |
| 3차 | `reuse`/`recompute`/`generate` 분기 로그, generated frames, generate ON/OFF 비교 metric CSV |
| 4차 | start-only vs bidirectional 비교 CSV, `SFR`/`SDI`/flicker 비교 결과, drift 감소 여부 리포트 |
| 5차 | CLIP-only vs OWLv2/VQA verifier 비교 리포트, temporal metric 재측정 결과, held-out 평가 결과, Temporal SRS Calibration 설정/결과 |
| 6차 | channel-symbol 절감률 로그, 절감률 vs 의미 신뢰도 곡선, bit accounting 설계 문서, 최종 비교 프로토콜 |

완료 기준은 "코드가 실행된다"가 아니라 **각 단계 결과가 파일로 남고, 이전 단계와 비교
가능한 로그/CSV가 생성되는지**로 판단한다.

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
| 1~4차의 semantic-unit 절감 효과 | semantic unit 절감은 실제 channel-symbol 또는 bit 절감과 다름 | 6차(11~12) | channel-symbol 절감률, symbol/pixel, bpp 설계안을 붙여 전송량 기준 결과로 재정리한다 |

따라서 1차 완료는 "최종 평가 체계 완성"이 아니라 **동작하는 비디오/시간축 평가 뼈대
확보**를 뜻한다. 최종 주장은 5차에서 검출기와 평가 지표를 보강하고, 6차에서 전송량
기준을 붙인 뒤에 한다.

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
| 의미 평가 | 완료에 근접 | 품질·CLIP·패킷·VQA 지표와 `srs_base/srs_packet/srs_v2` 연결 |
| 할루시네이션 검증 | 부분 | packet verifier, regeneration search, VQA, SRS-v2는 있으나 일부 판단은 휴리스틱 |
| 비디오 확장 | 부분 | keyframe/scene-change/temporal evaluator는 있으나 mp4 IO, motion gate, generate 분기는 후속 |
| 전송량 절감 | 계획/PoC | semantic unit 절감은 가능하나 channel-symbol 절감은 1차 PoC 대상, bit 기준은 설계안 대상 |
| 채널/비교/저지연 | 부분/스캐폴드 | guide damage, edge codec, low-latency, channel conditioning은 연결됐지만 일부는 placeholder |

## 모듈 매핑

- 시간축·영상 확장: `video/keyframe.py`, `video/scene_change.py`,
  `video/motion_residual.py`, `video/temporal_pipeline.py`,
  `evaluators/temporal_consistency.py`
- 할루시네이션 완화·검출: `guidance/semantic_packet`, `evaluators/hallucination*`,
  `evaluators/packet_matcher`, `controllers/regeneration*`
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

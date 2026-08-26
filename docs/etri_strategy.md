> [← 문서 색인](./README.md)

# ETRI 전략 정리 — 현재 상태

이 문서는 **핵심 한계 3가지와 그에 대한 현재 대응 상태**만 다룬다. 각 단계의 상세
구현 로그(1차~6차, LGVSC 1A/1B/1C)는 [archive/etri_implementation_log.md](./archive/etri_implementation_log.md),
아직 진행 전인 향후 계획은 [roadmap.md](./roadmap.md)로 분리했다.

문서 내 "PPT 슬라이드 N" 표기는 `ETRI_연구진행상황_1차공유_20260715_v2.pptx`
(2026-07-16 개정판) 기준이다.

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

| 단계 | 내용 | 상태 |
|---|---|---|
| 현행 | 프레임 간 의미 변화를 비교해 변화가 작으면 이전 복원을 재사용 | 완료 — semantic unit 기준 절감 |
| 1차 PoC | 변화가 작은 latent/semantic 요소를 덜 보내고, 수신단이 이전 프레임 정보를 재사용 | 완료(PoC) — 절감률 vs SRS/PTC 곡선 |
| 실제 bit 전송 | 양자화·비트 매핑·checksum을 포함한 실제 binary packet 전송 | **완료(2026-08)** — `transmission/` 패키지, 4-bit 양자화가 화질 저하 없이 선택됨. 상세는 아래 "실제 bit 전송(transmission/)" |
| 후속 | 중요한 의미에 더 많은 심볼/비트를 배분하는 importance-aware allocation | 계획 — [roadmap.md](./roadmap.md) 참고 |

이 문서에서는 **semantic unit 절감**, **channel-symbol 절감(PoC, proxy 기반)**,
**실제 bit 절감(`transmission/`, 실측)**을 구분한다.

### 실제 bit 전송(`transmission/`)

`accounting/bit_accounting.py`의 PoC 이후, 실제 binary packet 경로가 추가됐다
(2026-08-17/18): `transmission/quantization.py`(4/6/8/16-bit 실제 bit-packing),
`transmission/wire_packet.py`(checksum 포함 결정적 packet), `transmission/byte_accounting.py`
(정확한 byte 수, proxy 아님), `channels/digital_packet.py`로 config에서 선택.
10개 영상 Pareto sweep 결과(`outputs/transmission_reduction_full_20260818_043425/summary.json`)
4-bit 양자화가 full-precision AWGN 대비 평균 ΔPSNR −0.13dB로 화질 저하 없이 선택됐다
(영상별 분산·SRS/할루시네이션 변화는 미확인 — 일반화에는 추가 검증 필요).


### 단계별 구현 묶음과 완료 확인 기준

세부 의존성 순서(0~12번, PPT 블록 대응)는
[archive/etri_implementation_log.md](./archive/etri_implementation_log.md)에
있다. 핵심 원칙은 **비디오 기반을 먼저 만들고, 생성 분기를 붙이기 전에 packet
verifier를 먼저 준비하는 것**이었고, 아래는 그 결과로 나온 단계별 완료 상태다.

| 단계 | 포함 순서 | 구현할 것 | 완료 확인 |
|---|---|---|---|
| 1차 ✅ 구현 완료 (2026-07) | 0~4 | Presence threshold 배선, mp4/frame IO, `PTC`/`SFR`/`SDI`(CLIP/packet 기반 잠정 지표 — 5차 재측정 필요), motion-aware gate(기본 OFF, 실데이터 튜닝 후속), segment 구조 | 테스트 영상 입력 후 복원 frame/mp4가 생성되고, `temporal_metrics.csv`에 `PTC`/`SFR`/`SDI`가 기록되며, motion이 큰 구간이 reuse되지 않는지 로그로 확인 — `tests/test_video.py`·`tests/test_video_io.py`·`tests/test_evaluators.py`로 검증됨 (상세: archive/etri_implementation_log.md의 "1차 구현 결과") |
| 2차 ✅ 구현 완료 (2026-07) | 7 | `Packet Verifier`, 전송 packet과 복원 packet 비교, 오류 유형별 리포트, regeneration controller 기본 구조 | 추가/누락/왜곡 객체가 report에 분리 기록되고, 오류 유형별 controller decision 로그가 남음 — `tests/test_packet_matcher.py`·`tests/test_controllers.py`·`tests/test_video.py`로 검증됨 (상세: archive/etri_implementation_log.md의 "2차 구현 결과") |
| 3차 ✅ 구현 완료 (2026-07) | 5 | `video_generator` 인터페이스, `reuse`/`recompute`/`generate` 3-way 분기, start-only generation 경로 | config에서 generate를 켰을 때 inter-frame 일부가 generate branch로 들어가고, 생성 프레임이 저장됨 — `tests/test_video.py::TestGenerateBranch`·`tests/test_video_generator.py`로 검증됨 (상세: archive/etri_implementation_log.md의 "3차 구현 결과") |
| 4차 ✅ 구현 완료 (2026-07) | 6 | start+end keyframe 조건을 받는 bidirectional generation mode | start-only와 start+end 결과를 같은 영상에서 비교하고, `SFR`/`SDI`/flicker가 별도 CSV로 기록됨 — `tests/test_video_generator.py`·`tests/test_video.py::TestBidirectionalGenerateBranch`·`TestGenerationModeComparison`로 검증됨 (상세: archive/etri_implementation_log.md의 "4차 구현 결과") |
| 5차 ✅ 구현·실모델 재검증 완료 (2026-07) | 8~10 | OWLv2/VQA verifier 보강 인터페이스, held-out temporal metric 재측정 파이프라인, GT/VLM 기반 SRS 보정 스캐폴드, 10개 영상 실제 OWLv2/VQA 재검증 | CLIP-only 결과와 보강(calibrated) 결과가 비교 리포트로 나오고, loop-internal 지표와 held-out 지표가 분리되어 저장됨. 이후 실제 GPU recon frame 10개 영상 × 5개 모드 batch가 `ok=50`으로 완료되어 `summary_metrics.csv/md` 생성 — `tests/test_presence_backends.py`·`tests/test_heldout_remeasurement.py`·`tests/test_temporal_srs_calibration.py`·`tests/test_packet_matcher.py::TestPacketVerifierPresenceCalibration`·`tests/test_batch_remeasure_owlv2_vqa_10videos.py`로 검증됨 (상세: archive/etri_implementation_log.md의 "5차 구현 결과") |
| 6차 ✅ PoC 구현 완료 (2026-07) — 🟡 실제 bitstream 검증 필요 | 11~12 | channel-symbol/bit accounting PoC, naive baseline 비교, rate-reliability trade-off 리포트 | 절감률 vs `SRS`/`PTC`/`SFR`/`SDI`/severity 곡선이 생성되고, symbol/bit 계산 로그와 baseline 비교 표가 생성됨 — `tests/test_transmission_accounting.py`로 검증됨 (상세: archive/etri_implementation_log.md의 "6차 구현 결과") |
| 6차 후속 ✅ 실제 bitstream 구현 완료 (2026-08-17/18) | 6차의 🟡 해소 | 신규 `transmission/` 패키지 — `quantization.py`(4/6/8/16-bit 실제 bit-packing, `bit_depth=32`는 lossless raw-float32), `wire_packet.py`(checksum 포함 결정적 binary packet), `packet_bundle.py`(latent+caption+edge+manifest 전체 프레임 번들), `byte_accounting.py`(정확한 packet byte 수 `proxy=False` vs. channel-symbol/FEC 추정치 `proxy=True` 명시 분리), `receiver_runtime.py`(수신측은 직렬화 bytes만 받고 out-of-band 객체 없음). `channels/digital_packet.py::DigitalPacketChannel`로 AWGN/Rayleigh와 동일하게 config 선택형 | `scripts/run_transmission_reduction_eval.py`의 SKEM×bit-depth Pareto sweep으로 10개 영상 전체 실행 완료 — `outputs/transmission_reduction_full_20260818_043425/summary.json`에서 4-bit 양자화(`skem_int4`)가 full-precision AWGN 대비 화질 저하 없이(mean PSNR 23.47dB, ΔPSNR −0.13dB) 선택됨. `tests/test_transmission_reduction_eval.py`·`tests/test_transmission_reduction_temporal_integration.py`로 검증됨. 이제 실제 bit-packed 전송량 기준 결과이며, `accounting/bit_accounting.py`의 proxy 추정과는 분리해서 읽어야 함 |

각 단계의 최소 산출물은 다음과 같이 둔다.

| 단계 | 최소 산출물 |
|---|---|
| 1차 | 복원 mp4 또는 frame folder, `temporal_frames.csv`, `temporal_metrics.csv`, keyframe/segment 구조 JSON, motion gate decision log |
| 2차 | `packet_match_report.json` 또는 CSV, 오류 유형별 additional/missing/distorted 기록, controller decision log |
| 3차 | `reuse`/`recompute`/`generate` 분기 로그, generated frames, generate ON/OFF 비교 metric CSV |
| 4차 | start-only vs bidirectional 비교 CSV, `SFR`/`SDI`/flicker 비교 결과, drift 감소 여부 리포트 |
| 5차 | CLIP-only vs calibrated verifier 비교 리포트(`metric_delta.json`), temporal metric 재측정 결과(`clip_only_metrics`/`calibrated_metrics`), 10개 영상 OWLv2/VQA/ensemble 재측정 summary(`outputs/etri_video_eval/remeasure_10videos/summary_metrics.csv/md`), Temporal SRS Calibration weight 설정/저장 포맷 |
| 6차 | `frame_accounting.json/csv`, `segment_accounting.json/csv`, `accounting_summary.json`(bit/symbol 절감률, naive baseline 대비), `rate_reliability_summary.json`/`rate_reliability_curve.csv` — 실제 CBR/표준 bitstream 검증 결과는 미포함 |

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
| 1~4차의 semantic-unit 절감 효과 | semantic unit 절감은 실제 channel-symbol 또는 bit 절감과 다름 | 6차(11~12) | channel-symbol 절감률, symbol/pixel, bpp 설계안을 붙여 전송량 기준 결과로 재정리한다 — **완료(PoC, 6차)**: `accounting/bit_accounting.py`/`pipelines/transmission_accounting.py`가 bit/channel-symbol 절감을 semantic-unit 절감과 분리해 계산(상세: archive/etri_implementation_log.md의 "6차 구현 결과"). 🟡 실제 CBR/표준 bitstream 검증은 아님 |

> **5차 진행 상태 참고**: 5차(8~10)의 presence backend, calibrator, held-out
> remeasurement 구조는 구현·테스트됐고, 이후 실제 OWLv2/VQA weight를 연결한 10개 영상
> 재검증도 완료됐다(`outputs/etri_video_eval/remeasure_10videos/summary_metrics.csv`,
> 50 rows). 따라서 "OWLv2/VQA 실제 weight 연결 및 재검증"은 완료 항목으로 본다. 남은
> 5차 계열 후속은 실제 VLM judge를 이용한 Temporal SRS 가중치 fitting과 보고서용 해석
> 정리다.

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

## 현재 구현 상태

| 묶음 | 상태 | 요약 |
|---|---|---|
| 원본 경로/모듈화 | 완료 | 원본 경로 보존, 모듈 구조 정리, End-to-End 평가 골격 |
| 의미 평가 | 완료에 근접 | 품질·CLIP·패킷·VQA 지표와 `srs_base/srs_packet/srs_v2` 연결; presence threshold/uncertain band 배선 완료(1차 순서 0, CLIP probe는 잠정) |
| 할루시네이션 검증 | 5차 실모델 재검증 완료, sampler 연동 후속 | packet verifier(이미지 경로 + `evaluators/packet_verifier.py` severity, 2차; presence-backend 보강 연결, 5차), 오류 유형별 controller(`controllers/verifier_controller.py`, 2차), presence backend 인터페이스(`evaluators/presence_backends.py`) + ensemble calibrator(`evaluators/presence_calibration.py`) + held-out 재측정(`pipelines/heldout_remeasurement.py`), `scripts/batch_remeasure_owlv2_vqa_10videos.py` 기반 10개 영상 실제 OWLv2/VQA 재검증 완료. `ensemble_gt_filter`는 object preservation, `ensemble_openworld_filter`는 hallucination/additional 분석용으로 분리한다. candidate action의 실제 sampler 주입은 아직 후속 |
| 비디오 확장 | 1~4차 완료(기초) + 후속 1A·1B 완료(1B는 start-only+bidirectional 실제 GPU 검증까지) + 1C 재현 준비 완료(실제 검증 실행은 사용자) + PSSS/SKEM 단계 코드/테스트/CPU 스모크 완료(실제 MLLM 검증은 사용자) | keyframe/scene-change/temporal evaluator + mp4 IO(`utils/video_io.py`), motion 이중 게이트, segment 구조(`video/segment.py`), `PTC`/`SFR`/`SDI`(잠정), reuse/recompute/generate 3-way 분기 + mock start-only/bidirectional video_generator(`video/video_generator.py`, 3~4차), start-only vs bidirectional 비교 파이프라인(`pipelines/generation_mode_comparison.py`, 4차), **1A**: Rx-legal segment-level 생성 계약(`SegmentGenerationRequest`/`SegmentGenerationResult`/`generate_segment()`)과 `TemporalPipeline`의 GOP당-1회 배치 호출(`video/video_generator.py`, `video/temporal_pipeline.py`), **1B**: 별도 conda 환경 subprocess worker(`scripts/lgvsc_generate_worker.py`, `mock`/`svd`/`wan`/`callable` backend) + `ExternalSegmentWorkerGenerator` — `wan`(`image`+`last_image`+`prompt`)의 start-only와 bidirectional 둘 다 실제 GPU 검증 완료(bidirectional은 segment별 체크포인트 자동 선택으로 해결). **1C**: `SKIM+SFA`/`SKEM+DSA`에 대응하는 재현 baseline 4종(`mock_baseline`/`svd_start_only`/`wan_skim_sfa`/`wan_skem_dsa`) config + `scripts/batch_lgvsc_1c_reproduce.py` batch driver + summary 생성기 — keyframe 선택(SKIM/SKEM)은 네 모드 공통, decoder 조건화만 차이(nearest-reproducible 근사, faithful reproduction 아님). **PSSS/SKEM 단계**: 실제 모델 다음-토큰 확률로 `S_rel=P(No)-P(Yes)`를 계산하는 PSSS(`video/psss.py`, mock/proxy/real 3-backend)와 그 위에서 자동회귀로 variable-length keyframe을 고르는 SKEM selector(`video/skem_selector.py`) — `keyframe.selector: fixed`(기존, 불변) vs `psss`(신규)로 완전히 독립적인 selector 선택, 비교 config 4종(`skim_sfa_fixed`/`skem_dsa_psss`/`skem_dsa_mock_psss`/`skem_dsa_proxy_psss`) + batch summary 확장(`selector_backend`/`psss_backend_kind`/segment 길이 통계/PSSS score 통계 + SKIM-vs-SKEM aggregate 비교표). 자세한 내용은 `docs/lgvsc_1b_worker_readiness.md`의 "1B Wan 검토" / `docs/lgvsc_1c_reproduction_readiness.md` / `docs/lgvsc_psss_skem_readiness.md` 참조 |
| 전송량 절감 | 계획/PoC | semantic unit 절감은 가능하나 channel-symbol 절감은 1차 PoC 대상, bit 기준은 설계안 대상 |
| 채널/비교/저지연 | 부분/스캐폴드 | guide damage, edge codec, low-latency, channel conditioning은 연결됐지만 일부는 placeholder |

## 모듈 매핑

- 시간축·영상 확장: `video/keyframe_extractor.py`, `video/scene_change_detector.py`,
  `video/motion_residual.py`, `video/temporal_pipeline.py`, `video/segment.py`,
  `video/video_generator.py` (3차 mock start-only + 4차 mock bidirectional generate backend
  + 1A segment-level `SegmentGenerationRequest`/`SegmentGenerationResult`/`generate_segment()` 계약
  + 1B `ExternalSegmentWorkerGenerator`), 신규 `scripts/lgvsc_generate_worker.py` /
  `scripts/lgvsc_example_callable_backend.py` (1B, 별도 conda 환경에서 실행되는 실제 생성 backend worker),
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

- [roadmap.md](./roadmap.md) — 향후 연구개발 계획
- [archive/etri_implementation_log.md](./archive/etri_implementation_log.md) — 1차~6차, LGVSC 1A/1B/1C 상세 구현 이력
- [etri_overview.md](./etri_overview.md), [phase4.md](./phase4.md), [phase5.md](./phase5.md)
- [video_extension_lgvsc.md](./video_extension_lgvsc.md) — LGVSC 매핑 설계
- [lgvsc_1b_worker_readiness.md](./lgvsc_1b_worker_readiness.md) / [lgvsc_1c_reproduction_readiness.md](./lgvsc_1c_reproduction_readiness.md) / [lgvsc_psss_skem_readiness.md](./lgvsc_psss_skem_readiness.md) — 개별 검증 리포트
- 더 오래된 보관본: [archive/etri_development_plan_v2.md](./archive/etri_development_plan_v2.md),
  [archive/etri_development_roadmap.md](./archive/etri_development_roadmap.md),
  [archive/limitation_reference_map.md](./archive/limitation_reference_map.md)

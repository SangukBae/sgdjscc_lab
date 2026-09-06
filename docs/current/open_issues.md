---
status: active
updated: 2026-09-04
owner: ETRI SGD-JSCC 연구팀
source_commit: aae9e26
supersedes: docs/etri_strategy.md, docs/phase4.md, docs/phase5.md
---

> [← 문서 색인](../README.md)

# 알려진 한계·기술 부채

- 문서 범위
  - 구현 내부의 근사
  - 미해결 지점
- 연결 문서
  - 구현 상태: [status.md](./status.md)
  - 신규 연구 계획: [roadmap.md](./roadmap.md)

## 시간축·영상

- **인터프레임 재사용은 키프레임 복사**
  - 진정한 델타-워프/모션 보상이 아니다.
  - 픽셀 모션이 크지만 의미가 같은 구간은 motion 이중 게이트로 걸러도 실데이터 threshold 튜닝은 아직 안 됐다.
- **motion 이중 게이트 threshold가 초기값**
  - `temporal.motion_threshold`는 기본 off이며, 켰을 때의 값도 실모션 비디오 표본으로 튜닝된 값이 아니다.
- **LGVSC 재현선의 keyframe 선택은 SKIM/SKEM만 구분**
  - decoder 조건화(단일 vs 두 keyframe)만 다르고, 그 외 재현 baseline 4모드(`mock_baseline`/ `svd_start_only`/`wan_skim_sfa`/`wan_skem_dsa`)는 keyframe 선택 로직을 공유한다(이후 `skim_sfa_fixed`/`skem_dsa_psss` 계열에서 실제로 분리됨 — [experiments/2026-07_lgvsc_psss_skem.md](../experiments/2026-07_lgvsc_psss_skem.md)).
- **PSSS `real` backend는 실제 MLLM 가중치로 실행된 적이 없다**
  - device placement 버그는 fake model로 GPU에서 재현·수정했지만, 실제 keyframe 선택 품질은 검증되지 않았다.
- **side-info(모션/캡션)는 실제 생성 조건화에 쓰이지 않는다**
  - Wan backend도 `side_infos`를 accept만 하고 조건화에 반영하지 않는다.
- **학습된 DSA adapter 없음**
  - Wan은 세그먼트 길이와 무관하게 동일 아키텍처를 재사용한다(체크포인트 자동 선택 ≠ 가변 차원 학습 adapter).

## 할루시네이션 완화

- **candidate action이 실제 sampler에 반영되지 않는다**
  - `verifier_controller`의 negative-prompt/prompt-emphasis 결정은 로그만 남기고, 실제 diffusion 샘플러 호출에 주입하는 배선이 없다.
- **단계적 디노이징은 prompt 레벨 연결뿐**
  - 샘플러 루프 내부의 스텝별 prompt 전환은 SGD-JSCC 샘플러 수정이 필요해(알고리즘 보존 불변식과 충돌) 구현하지 않았다.
- **패킷은 평가/제어 메타데이터일 뿐**
  - 실제 semantic packet의 채널 코딩/drop 시뮬레이션은 없다(패킷 자체가 채널을 통과하며 손상되는 시나리오는 미구현).

## 평가 체계

- **Temporal SRS Calibration은 synthetic target 스캐폴드뿐**
  - 실제 GT 주석/VLM judge 연결이 안 됐다.
- **DISTS/downstream task 지표 없음**
  - video captioning
  - action classification
  - depth estimation
  - 용도: LGVSC 직접 비교
- **object-track 기반 drift, flow-warp temporal consistency 없음**
  - 현재 `PTC`/`SFR`/`SDI`는 packet/object 판정에 의존하는 지표뿐이다.
- **`eta_th`(PSSS 임계값)의 CBR 캘리브레이션 없음**
  - 논문 실험값(0.35)을 그대로 쓸 뿐, 이 데이터셋/모델 조합에서 목표 CBR에 맞춰 보정하지 않았다.
- **통합 개발평가는 완료됐지만 폐루프 accounting은 미완**
  - pixel·semantic·hallucination·temporal·bundle byte·reconstruction elapsed의 120-pair
    통합과 paired 95% CI는 완료됐다. feedback/retransmission byte, regeneration 지연,
    retry 수를 같은 row에 기록하는 폐루프 accounting은 아직 없다.
- **별도 held-out 영상 결과 없음**
  - 개발 10영상에서 `few10 + both-omit`이 평균 gate를 통과했지만 open hallucination과
    additional-object CI 상한 0.0525/0.0570이 margin 0.05를 넘는다. 증가는
    `01_person_walk`, `02_car_pass`에 집중돼 최종 일반화 판정에는 미사용 영상의 paired
    검증이 필요하다.
  - 2026-08-29 기준 별도 데이터셋을 즉시 만들 수 없어 검증을 데이터 준비 시점까지
    연기했다. 취소 또는 통과로 간주하지 않으며, 그동안 `few10 + both-omit`은 opt-in
    개발 후보로만 유지하고 최종 operating point라는 표현을 사용하지 않는다.

## 전송량

- **channel-symbol/bit accounting PoC는 proxy 기반**
  - latent symbol 수는 실제 `encode_features` 텐서가 아니라 프레임 shape + 고정 아키텍처 상수에서 추론한 값이고, edge/motion side-info는 명시적 proxy 비율·quantization 가정이다.
- **실제 binary packet 전송(`transmission/`)도 채널 심볼/FEC 환산은 proxy**
  - byte 수는 정확하지만 실제 변조/채널부호 표준을 재현하지 않는다.
- **importance-aware bit allocation 없음**
  - 중요한 의미 요소에 더 많은 심볼/비트를 배분하는 정책이 아직 없다.
- **severity의 인과적 feedback 경로 없음**
  - 현재 프레임의 severity는 복원 후에 계산되므로 다음 프레임/GOP 제어나 재전송 feedback에 연결해야 한다.
  - 관련 byte와 왕복 지연 accounting도 미구현이다.
- **float32 full은 3개 core condition 범위이며 별도 held-out 검증이 아니다**
  - 기존 10영상 60dB 결과(PSNR `11.32 vs 23.34`, SSIM `0.081 vs 0.731`, LPIPS `0.739 vs 0.254`)는
    decoder step 계약이 잘못된 legacy 결과로 판정했다.
  - 10dB 3-GPU full 300프레임에서 digital wire `34.725/0.9351/0.1242`,
    AWGN `34.004/0.9296/0.1264`로 품질 격차가 해소됐고, in-process/wire 최대 PSNR 차이는
    `0.000752dB`, wire round-trip은 300/300 bit-exact였다.
  - 핵심 artifact registry와 리포트 계약 보정은 완료했지만, 이 300프레임은 원인 진단용
    core condition 표본이다. 최종 정책의 일반화 주장은 별도 held-out 검증 후에만 가능하다.
  - 수정된 10dB 정책의 fixed int16/int8/int6/int4 재평가는 10영상×100프레임에서 완료됐다.
    4개 bit-depth가 모두 품질 허용 기준을 통과했고 `fixed_int4`가 최소 bit-depth로 선택됐다.
  - [full 실측](../experiments/2026-08-28_float32_digital_step_normalization_full.md),
    [진단 프로토콜](../protocols/float32_digital_diagnostics.md).
- **신규 `fixed_int6` ETRI 운용점의 both-omit·공개 데이터 검증이 남았다**
  - 2026-09-04부터 신규 ETRI 기본 bit-depth를 `fixed_int6`로 정했지만 근거가 된
    양자화 비교는 기존 개발 영상 10개와 baseline guide 조건이다.
  - `int6 + both-omit`의 exact bundle byte, additional-object, ghost-track, latency는
    아직 실측하지 않았다. 기존 int4 절감률·hallucination 수치를 int6에 전용하면 안 된다.
  - G1 v1.1이 완료됐으므로 같은 OVIS Pilot·seed·diffusion step의 paired bridge
    코드·설정·테스트를 준비했다(**실행 준비 완료, GPU 실행은 미실시**). 상세:
    [int6 결정](../experiments/2026-09-04_int6_etri_operating_point_decision.md),
    [bridge 준비 기록](../experiments/2026-09-06_negative_semantics_int6_bridge_preparation.md).
- **VAE-direct는 통합 개발평가의 strict SSIM gate를 실패했다**
  - both-omit에서 23.4885s/video로 full50보다 4.60배 빠르고 PSNR·LPIPS 및 semantic
    지표는 양호했지만, 평균 SSIM 하락 0.01129가 사전 margin 0.01을 넘었다.
  - threshold를 사후 변경하지 않으며 primary held-out 후보에서는 제외한다. few10도
    hallucination CI 경고가 있어 최종 decoder 정책은 아직 확정할 수 없다.
- **proxy SKEM exact-rate 보정이 fixed schedule로 퇴화**
  - 10영상×10설정 full 결과는 actual transmitting count와 effective byte를 정확히
    맞췄지만, 10/10 영상에서 fixed/SKEM keyframe·transmitting index가 같았다.
  - PSNR/SSIM/LPIPS 차이 0은 SKEM 우위가 아니라 동일 schedule 결과다. raw 100
    byte/video 차이도 config label의 manifest 길이뿐이라 padding 후 완전 동률이다.
  - 이 결과는 selector를 fixed로 유지한다는 근거다. 당시 bit-depth는 `fixed_int4`였지만
    2026-09-04 이후 신규 ETRI bit-depth는 별도 Pareto 결정에 따라 `fixed_int6`다.
    다른 semantic schedule의 효용을 보려면 exact-count 후보 중 fixed와 다른 index를
    강제하거나 실제 MLLM PSSS로 재검증해야 한다. 상세:
    [실험 결과](../experiments/2026-08-28_fixed_skem_matched_rate_10db.md).
- **reliable-digital 경로에서 uncertainty payload를 decoder가 소비하지 않는다**
  - 16-profile full 결과에서 uncertainty q4/ds2/ds4/reuse2/omit이 10영상 각각의
    PSNR·SSIM·LPIPS를 baseline과 전부 정확히 같게 만들었다.
  - 수신기는 `edge_already_received=true`로 analog Canny 재전송을 건너뛰며,
    uncertainty는 건너뛴 `_retransmit_canny()`에서만 사용된다. 따라서 현재 packet은
    복원에 기여하지 않는 uncertainty를 baseline byte의 45.41%만큼 전송한다.
  - uncertainty를 digital packet에서 제거할지, 별도 decoder 조건으로 연결할지
    Tx/Rx 계약 결정이 필요하다.
- **edge guide 영향이 현재 checkpoint/config에서 수치적으로 매우 약하다**
  - `edge_omit`의 영상별 최대 절대 변화가 PSNR 0.0000747dB, SSIM 0.000000444,
    LPIPS 0.00000252뿐이었다. edge는 ControlNet latent로 연결되므로 uncertainty와
    달리 구조적 bypass는 아니지만, pixel metric만 보면 사실상 무감도에 가깝다.
  - ControlNet off/scale sensitivity와 SRS·hallucination·temporal 지표를 확인하기 전
    guide가 불필요하다고 일반화하면 안 된다.
- **guide 최소-byte 조합은 개발셋에서 닫혔지만 Tx/Rx 계약 정리가 남았다**
  - 통합 120-pair에서 `edge_ds4 + uncertainty_omit`과 `edge_omit + uncertainty_omit`을
    실행했다. both-omit은 219,459.7 bytes/video(-90.843%)이고 full50 baseline 대비
    pixel 변화가 수치 오차 수준이며 closed/open semantic 지표 차이는 0이었다.
  - 이는 현재 reliable-digital checkpoint/config의 개발 결과다. packet schema에서
    uncertainty/edge를 기본 제거할지, 향후 decoder 조건으로 다시 연결할지 결정하고
    held-out에서 확인하기 전 다른 채널·checkpoint로 일반화하면 안 된다.

## 채널·저지연

- **채널 조건 토큰을 frozen denoiser가 소비하지 않는다**
  - received-latent init + reliability 스케일 guidance/steps로만 조건화가 작동한다 ([architecture/tx_rx_contract.md](../architecture/tx_rx_contract.md) §4).
- **학습된 consistency/distilled student 없음**
  - few-step 샘플링은 결정론적 근사 수식만 있다.
- **sampler early-exit 제한**
  - 연속 sampler: 지원
  - discrete sampler: fallback
- **Fading `signal_scale`/step-matching은 AWGN 공식을 재사용**
  - fading 채널에서는 근사다.

## 데이터

- **대규모 외부 영상은 취득했지만 학습 적합성 검증은 아직 없음**
  - YouTube-VOS 2019 train 3,471개와 valid 507개를 취득·해시 고정했다.
  - 이 사실은 RSM/allocator 학습이 수행됐거나 temporal metric 분별력이 검증됐다는 뜻이 아니다.
- **negative-semantics G0/G1 데이터 gate 통과, effective-seed/source-paired 한계는 v1.2에서 별도 문서화**
  - ETRI 10영상은 반복 개발 이력 때문에 전부 Development only로 고정했다.
  - OVIS Pilot 40 / Train 3,471 / Development 10 / Validation 507 / DAVIS Held-out 30을
    실제 바이트와 hash로 동결했고 사용 승인·Held-out seal도 기록했다.
  - Pilot은 official GT 기반 40개 독립 event이며 ENTER/EXIT/OCCLUDE/REAPPEAR 각 10개다.
  - G0 자동 감사 13/13과 `--require-pass` exit 0을 확인했다.
  - G1은 240/240 정식 Pilot run을 완료했고 evaluator threshold(0.2)와 selector
    weight 분리를 동결해 declared-seed 기준 `gate_status: PASSED`를 얻었다.
  - 남은 한계: (1) 사람 지각 hallucination은 여전히 주장할 수 없다. (2) 이
    reconstruction 경로는 `seed`를 소비하지 않아 `effective_seed_count=1`이며,
    같은 gate threshold를 정직하게(seed 중복 제거) 재적용하면 few10 기준
    `NOT_PASSED`로 뒤집힌다. (3) source-carried additional-object를 제외한
    source-paired h_add는 raw 대비 약 47~51% 낮고 full50은 `h_add_min` 미달이다.
    (4) `results/…_derived_v1_2b`는 checksum-frozen이지만 그 결과를 만든
    derivation 코드(`derive_negative_semantics_g1_v1_2.py` 등)가 당시 git에
    commit되지 않은 code-dirty 상태였다 — 코드가 clean commit된 뒤
    `v1_2c` 경로로 재파생해야 provenance 한계가 해소된다(결론 자체는
    바뀔 것으로 예상하지 않음).
  - 근거: [G0 v1.2 기록](../experiments/2026-09-03_negative_semantics_g0_official_gt_amendment_v1_2.md),
    [G1 v1.1 결과·감사](../experiments/2026-09-06_negative_semantics_g1_v1_1_pilot_results.md),
    [G1 v1.2 amendment](../experiments/2026-09-06_negative_semantics_g1_v1_2_amendment.md)

## 결과 판정 의미

- **개발 요약기의 과거 `validation_passed`가 평균 gate만 반영함**
  - 2026-08-29 결과 JSON은 point mean을 기준으로 true였지만 hallucination CI 상한은
    margin을 넘었다. 해당 결과는 historical artifact로 보존하고 재작성하지 않는다.
  - 이후 요약기는 `run_integrity_passed`, mean screening, CI screening을 분리하고
    baseline 자체를 후보에서 제외한다. Development role에서는
    `validation_passed=null`이며 Validation manifest가 명시된 실행에서만 CI 기반 bool이다.

## 논문 정합성

- 논문 미공개 학습값
  - GAN weight
  - discriminator 세부값
  - `cfg_dropout_prob`
  - edge codec ViT 구성
- 처리
  - `paper_assumed_hparams` 가정치 사용
  - 상세: [논문 정합성](../reference/paper_alignment.md)
- Complex-phase transport·joint CSI
  - 현재: layer 수준 scaffold
  - 미지원: e2e faithful 재학습

## 관련 문서
- [status.md](./status.md) — 현재 구현 상태
- [roadmap.md](./roadmap.md) — 이 한계들을 해소하기 위한 향후 계획
- [architecture/tx_rx_contract.md](../architecture/tx_rx_contract.md) — 설계 차원의 근사 지점

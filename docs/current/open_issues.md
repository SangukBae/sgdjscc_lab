---
status: active
updated: 2026-08-28
owner: ETRI SGD-JSCC 연구팀
source_commit: 6d6c4ed
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
- **통합 평가 계약은 문서만 확정**
  - feedback/retransmission byte, regeneration 지연, retry 수를 한 row에 자동 기록하는 harness 배선은 아직 없다.
- **paired 통계·별도 held-out 영상 결과 없음**
  - 현재 10영상 결과에는 정책별 paired difference, 95% 신뢰구간, 별도 영상 split 재검증이 없다.

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
- **VAE-direct의 semantic·시간축 operating point 검증이 없다**
  - 20프레임에서 pixel fidelity와 receiver decode latency는 50-step diffusion보다 우수했지만,
    sender·bundle 전체 지연, semantic fidelity, 할루시네이션, 시간축 지표는 비교하지 않았다.
  - 따라서 VAE-direct/few-step/full diffusion 중 최종 정책은 아직 확정할 수 없다.
- **fixed–SKEM exact matched-rate full 결과 대기**
  - keyframe 수는 맞았지만 `recompute_semantic`으로 실제 transmitting frame 수가 달라졌다.
  - `rate_matching.csv` 통과: 35/50; selector 절감 효과는 잠정 결과로만 사용한다.
  - 새 실행기는 fixed max-GOP을 유지한 채 영상별 SKEM을 보정해 실제 visual 전송 수를
    exact match하고, raw byte 1% + padding 후 effective byte exact를 자동 검증하도록 준비됐다.
  - 10영상×10설정 3-GPU full 실행과 결과 registry 고정 전까지 SKEM 효과 결론은 여전히 잠정이다.
- **10dB fixed_int4 packet의 90.90%가 edge·uncertainty**
  - visual latent는 5.89%라 추가 bit-depth 축소 효과가 제한적이다.
  - 다음 전송량 절감 대상은 edge/uncertainty 압축·선택 전송이다.

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

- **대규모 영상 학습 데이터셋 없음**
  - 현재 영상 자산은 ETRI 10-영상 평가셋 (`data/etri_video_eval/`, GT는 10개 영상의 수작업 검증 샘플)뿐이다.
  - temporal SRS/flicker/drift의 baseline 분별력 검증에는 실제 모션이 있는 더 큰 영상 데이터셋이 필요하다.

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

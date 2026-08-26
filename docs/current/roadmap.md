---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: 63b7b23
supersedes:
---

> [← 문서 색인](../README.md)

# 향후 연구개발 계획

- 이 문서는 **아직 완료되지 않은 것**만 다룬다. 완료된 항목과 그 근거는
  [status.md](./status.md)(현재 상태)와
  [../archive/etri_implementation_log.md](../archive/etri_implementation_log.md)(상세 이력)를
  참고한다. 알려진 한계·기술 부채는 [open_issues.md](./open_issues.md). 아래는
  [architecture/system.md](../architecture/system.md)의 핵심 연구 문제 3가지(시간축·영상 /
  할루시네이션 / 평가 신뢰도) + 전송량 절감이라는 **연구 목표 기준**으로 남은 과제를
  정리한 것이며, Phase 번호나 1차~6차 같은 과거 구현 순서로 나누지 않는다.

> **연구개발 메인 문서:** 앞으로 구현할 작업과 우선순위는 이 문서를 기준으로
> 갱신한다. 구현이 끝나면 [status.md](./status.md)로 옮기고, 검증 결과는 날짜가
> 포함된 `docs/experiments/` 문서로 남긴다.

## 권장 실행 순서

| 순서 | 작업 | 완료 조건 |
|---:|---|---|
| 1 | 평가 계약 확정 | baseline, 컬럼, held-out 지표, feedback·지연 비용 정의 |
| 2 | 평가 harness 확장 | 동일 입력·seed로 설정 sweep과 paired 결과 생성 |
| 3 | verifier→sampler 배선 | action schema, 실제 prompt 반영, retry·중단 조건 구현 |
| 4 | 정적 정책 회귀·ablation | 기존 품질을 깨지 않고 regeneration 효과 분리 측정 |
| 5 | 동적 예산 controller | 다음 프레임/GOP 또는 명시적 feedback 기반 제어 |
| 6 | 정책 조합 비교 | static/SNR-only/severity-only/combined × regeneration ON/OFF |
| 7 | 최종 검증 | ETRI 10영상 paired 평가 후 별도 held-out 영상 재검증 |

- 하이퍼파라미터 조합을 반복 실행하는 책임은 **평가 harness**에 둔다. controller는
  한 시점의 관측값으로 예산·행동을 결정할 뿐, 스스로 실험 sweep을 수행하지 않는다.

## 1. 시간축·영상 신뢰성 고도화 (한계 1 후속)

- LGVSC 1A/1B는 검증 완료, 1C는 config·batch driver 준비 완료다. real MLLM PSSS와
  10영상×4모드 재현 실행은 아직 남아 있다. 그 위에 붙는 **ETRI 딥러닝 개선선**도
  후속 과제다.

- **Learned bidirectional keyframe adapter** — 현재는 mock 보간/Wan 등 고정 backend만
  사용. 통신 조건을 반영해 학습되는 decoder adapter 필요.
- **Semantic packet/side-info encoder** — 전송 packet과 motion/side-info를 diffusion
  조건 embedding으로 직접 주입하는 경량 encoder.
- **Channel reliability router** — 채널 상태에 따라 decoder 조건화 강도를 학습적으로
  조절.
- **Variable-length DSA 고도화** — 현재 `video/skem_selector.py`의 PSSS 기반 선택은
  mock/proxy 백엔드까지만 검증됨. 실제 MLLM(`real` 백엔드) 연결과 side-info 인코더가
  남음.

## 2. 할루시네이션 완화 고도화 (한계 2 후속)

- **Candidate action의 실제 sampler 주입** — `controllers/verifier_controller.py`는
  negative-prompt/prompt-emphasis 후보를 결정·로그만 하고 실제 샘플러에 반영하지
  않는다. 공통 action 스키마, 샘플러 입력(negative_prompt는 이미 지원됨,
  prompt-emphasis는 미지원), retry 상한/중단 조건 설계가 먼저 필요하다.
- **OWLv2/VQA semantic critic 기반 제한된 재생성** — 판정 이후 재생성/추가
  keyframe/fallback을 실제로 트리거하는 폐루프.
- **Temporal SRS Calibration 실 데이터 연결** — 현재 가중치 fitting은 synthetic
  target 기준 스캐폴드뿐. 실제 GT 주석/VLM judge 연결이 남음.
- **Semantic Packet Fidelity Adapter / Counterfactual Hallucination Critic** — 전송
  packet을 조건으로 직접 주입하는 학습형 adapter와, 복원 객체의 packet 정합성을
  판별하는 critic. 1차 필수 구현이 아닌 고도화 항목.

## 3. 전송량-신뢰도 공동 최적화 (한계 3 + 전송량)

- `transmission/` 패키지로 4~16bit 양자화와 직렬화 packet byte 집계까지는 완료됐다.
  물리 채널 symbol/FEC는 아직 proxy다. 남은 것은 "무엇을 얼마나 보낼지"를 신뢰도
  신호와 함께 동적으로 정하는 부분이다.

- **Learnable keyframe/side-info selector** — 현재 SKEM/비트뎁스 선택은 오프라인
  Pareto sweep으로 고정값을 고르는 방식. 채널·검증 신호를 함께 보는 학습형 selector로
  발전.
- **송신단·폐루프 지능화** — bit budget과 생성 실패 위험(verifier severity)을 공동
  고려하는 정책. severity는 수신 후에만 나오므로 동일 프레임이 아니라 다음
  프레임/GOP 예산 조정 또는 별도 feedback 채널 설계가 전제. 동일 프레임의 최초
  전송 결정에는 송신단에서 계산 가능한 motion·semantic-change risk proxy만 쓸 수
  있다. feedback을 사용하면 feedback byte, 왕복 지연, 재전송 bundle byte를 모두
  전송률·지연 결과에 포함한다.
- **Importance-aware allocation** — 중요한 의미 요소에 더 많은 심볼/비트를 배분.

## 4. 평가 벤치마크 완성 (한계 3)

- **모듈별 ablation** — 개선선의 각 구성요소(adapter/router/selector/critic)를 독립적으로
  끈 비교. 최소 조합은 static/SNR-only/severity-only/combined 정책과 regeneration
  ON/OFF의 교차 비교다.
- **실제 CBR/표준 bitstream 비교** — `transmission/`은 실제 bit-packing이지만 여전히
  표준 변조/FEC를 재현하지 않음(byte 수는 정확, 채널 심볼/FEC 환산은 proxy).
- **DISTS/downstream task 지표** — 현재 `evaluators/`에는 없음. LGVSC와 직접 비교하려면
  필요.
- **latency/VRAM 비교**, **LGVSC 재현선(1C) 실제 10영상×4모드 실행 및 재현 수준 판정**
  (config/batch driver는 준비 완료, 실행 자체가 남음).
- **paired 통계 검증** — 영상별 차이의 평균·표준편차·95% 신뢰구간을 보고하고,
  ETRI 10영상에서 선택한 설정은 별도 held-out 영상에서 한 번 더 검증한다.

## 일정

- 7~8월 계획(SNR 스윕, channel-symbol 절감 1차 PoC, `PTC`/`SFR`/`SDI` 초기 결과)은
  완료됐다 — [status.md](./status.md) 참고. 남은 일정:

| 시기 | 초점 | 산출물 |
|---|---|---|
| 9월 | 영상 파이프라인 고도화 + verifier 연동 | motion gate 실데이터 튜닝, VQA/OWLv2 연동 결과 반영 |
| 10월 | 페이딩 견고성 + verifier 고도화 | Rayleigh/페이딩 결과, packet verifier candidate action 실제 반영 |
| 11월 | 공정 비교 + 최종 정리 | held-out 평가 기반 최종 실험, 비교 프로토콜, 최종 보고서 |

## 신규 연구 아이템 확장 가능성

1. **전송률 적응형 영상 시맨틱 통신.** 채널 상태와 영상 변화량에 맞춰 꼭 필요한 만큼만
   전송한다. `semantic delta + channel-symbol/bit` 절감과 `PTC/SFR/SDI` 기반
   rate-semantic reliability trade-off 분석이 핵심.
2. **수신단 신뢰성 제어형 생성 복원.** 원본 없이 수신단이 전송 packet 기준으로 복원
   결과를 자체 검증하고, hallucination critic/regeneration controller로 재복원을
   제어한다.
3. **시맨틱 전송 평가 벤치마크.** `PTC/SFR/SDI`, VQA/OWLv2 검증, held-out 평가, SRS
   고도화를 묶어 시맨틱 통신 연구용 공정 비교 프로토콜로 확장한다.

## ETRI 협의 필요사항

| 항목 | 협의 내용 |
|---|---|
| 전송량 평가 단위 | channel symbol 수, symbol/pixel 비율, bpp, 실제 byte(`transmission/`) 중 우선 보고 단위 결정 |
| 영상 연구 범위 | keyframe 기반 PoC 수준인지, 실제 비디오 코덱/모션 보상까지 포함할지 결정 |
| 채널 범위 | AWGN 중심인지, Rayleigh/fast fading을 필수 범위로 포함할지 결정 |
| 비교 기준 모델 | WITT, DiffJSCC, Deep-JSCC 중 필수 비교군과 공정 비교 조건 결정 |
| 재복원 평가 기준 | oracle 상한, Rx-legal self-verification, held-out 최종 평가의 구분 방식 결정 |
| 평가 데이터셋·시간축 지표 | 장면 전환/객체 등장·소멸이 있는 영상 데이터와 `PTC/SFR/SDI` 보고 방식 결정 |
| 산출물·라이선스 | 최종 납품에 포함할 오픈소스, 모델 가중치, 라이선스 범위 결정 |

## 관련 문서

- [status.md](./status.md) — 현재 상태
- [open_issues.md](./open_issues.md) — 알려진 한계·기술 부채
- [../archive/etri_implementation_log.md](../archive/etri_implementation_log.md) — 상세 구현 이력
- [../architecture/tx_rx_contract.md](../architecture/tx_rx_contract.md) — LGVSC 시스템 구조 매핑
- [../experiments/2026-07_lgvsc_1c_reproduction.md](../experiments/2026-07_lgvsc_1c_reproduction.md) — 1C 실행 준비 상태

---
status: active
updated: 2026-08-28
owner: ETRI SGD-JSCC 연구팀
source_commit: 6d6c4ed
supersedes:
---

> [← 문서 색인](../README.md)

# 향후 연구개발 계획

- 문서 범위
  - 미완료 연구개발 과제
  - 기준: 시간축·영상, 할루시네이션, 평가 신뢰도, 전송량
  - 제외: 과거 Phase·차수별 구현 순서
- 연결 문서
  - 완료 상태: [status.md](./status.md)
  - 과거 이력: [etri_implementation_log.md](../archive/etri_implementation_log.md)
  - 한계·기술 부채: [open_issues.md](./open_issues.md)
  - 목표 정의: [system.md](../architecture/system.md)

- 관리 규칙
  - 메인 계획: 이 문서
  - 구현 완료 후: [status.md](./status.md)로 이동
  - 검증 완료 후: 날짜 기반 `docs/experiments/` 문서 추가

## 권장 실행 순서

fixed–SKEM exact matched-rate 재평가는 완료됐다. proxy SKEM이 10/10 영상에서 fixed와
동일 schedule로 수렴해 별도 selector 이득은 없었고 `fixed_int4`를 유지한다. 아래는
그 결과를 반영한 다음 실행 순서다.

| 순서 | 작업 | 완료 조건 |
|---:|---|---|
| 1 | edge·uncertainty 전송량 절감 | **실행 준비 완료** — 16-profile·3-GPU harness를 full 실행하고 품질 gate 내 최소-byte 후보를 확정·registry 보존 |
| 2 | 통합 평가 harness·복원 정책 비교 | Rate·품질·SRS·할루시네이션·시간축·전체 지연을 paired row로 기록하고 VAE-direct/few-step/full diffusion operating point를 비교 |
| 3 | verifier→sampler 배선 | 실제 prompt 반영, retry·중단 조건 구현 |
| 4 | 동적 예산 controller | 채널·uncertainty·verifier 위험도로 전송량과 복원 연산량을 결정하고 feedback/retransmission byte·RTT 포함 |
| 5 | 별도 held-out 최종 검증·문서 마감 | paired 통계·confidence interval, 최종 operating point, 표·그래프·재현성 registry 확정 |

- 역할 분리
  - 평가 harness: hyperparameter 조합 반복 실행
  - controller: 단일 시점의 예산·행동 결정

## 1. 시간축·영상 신뢰성 고도화 (한계 1 후속)

- 현재 기준
  - 1A·1B: 검증 완료
  - 1C: config·batch driver 준비 완료
- 남은 작업
  - real MLLM PSSS
  - 10영상 × 4모드 재현
  - ETRI 학습형 개선선

- **Learned bidirectional keyframe adapter**
  - 현재는 mock 보간/Wan 등 고정 backend만 사용.
  - 통신 조건을 반영해 학습되는 decoder adapter 필요.
- **Semantic packet/side-info encoder**
  - 전송 packet과 motion/side-info를 diffusion 조건 embedding으로 직접 주입하는 경량 encoder.
- **Channel reliability router**
  - 채널 상태에 따라 decoder 조건화 강도를 학습적으로 조절.
- **Variable-length DSA 고도화**
  - 현재 `video/skem_selector.py`의 PSSS 기반 선택은 mock/proxy 백엔드까지만 검증됨.
  - 실제 MLLM(`real` 백엔드) 연결과 side-info 인코더가 남음.

## 2. 할루시네이션 완화 고도화 (한계 2 후속)

- **Candidate action의 실제 sampler 주입**
  - `controllers/verifier_controller.py`는 negative-prompt/prompt-emphasis 후보를 결정·로그만 하고 실제 샘플러에 반영하지 않는다.
  - 공통 action 스키마, 샘플러 입력(negative_prompt는 이미 지원됨, prompt-emphasis는 미지원), retry 상한/중단 조건 설계가 먼저 필요하다.
- **OWLv2/VQA semantic critic 기반 제한된 재생성**
  - 판정 이후 재생성/추가 keyframe/fallback을 실제로 트리거하는 폐루프.
- **Temporal SRS Calibration 실 데이터 연결**
  - 현재 가중치 fitting은 synthetic target 기준 스캐폴드뿐.
  - 실제 GT 주석/VLM judge 연결이 남음.
- **Semantic Packet Fidelity Adapter / Counterfactual Hallucination Critic**
  - 전송 packet을 조건으로 직접 주입하는 학습형 adapter와, 복원 객체의 packet 정합성을 판별하는 critic.
  - 1차 필수 구현이 아닌 고도화 항목.

## 3. 전송량-신뢰도 공동 최적화 (한계 3 + 전송량)

- 완료
  - 4~16 bit 양자화
  - 직렬화 packet byte 집계
  - float32 reliable-digital baseline 포함 10영상 정상화 sweep
  - 수정된 10dB fixed-selector 양자화 재평가와 `fixed_int4` operating point 확정
- 현재 판정
  - float32 10dB baseline: full 300프레임에서 AWGN 동등 이상, transport bit-exact 확인 완료
  - `fixed_int4`: float32 대비 28.45% byte 절감, 세 품질 허용 기준을 모두 통과한 최소 bit-depth
  - exact matched-rate에서 proxy `skem_int4`는 fixed와 동일 schedule·품질로 수렴했다.
    raw 100 byte/video 차이도 manifest label 길이뿐이며 padding 후 동률이므로 운영점은
    `fixed_int4`를 유지한다.
- 근사
  - 물리 channel symbol·FEC는 proxy
- 목표
  - 신뢰도 기반 동적 전송 대상·예산 결정

- **Learnable keyframe/side-info selector**
  - 현재 SKEM/비트뎁스 선택은 오프라인 Pareto sweep으로 고정값을 고르는 방식.
  - 채널·검증 신호를 함께 보는 학습형 selector로 발전.
- **송신단·폐루프 지능화**
  - bit budget과 생성 실패 위험(verifier severity)을 공동 고려하는 정책.
  - severity는 수신 후에만 나오므로 동일 프레임이 아니라 다음 프레임/GOP 예산 조정 또는 별도 feedback 채널 설계가 전제.
  - 동일 프레임의 최초 전송 결정에는 송신단에서 계산 가능한 motion·semantic-change risk proxy만 쓸 수 있다.
  - feedback을 사용하면 feedback byte, 왕복 지연, 재전송 bundle byte를 모두 전송률·지연 결과에 포함한다.
- **Importance-aware allocation**
  - 중요한 의미 요소에 더 많은 심볼/비트를 배분.

## 4. 평가 벤치마크 완성 (한계 3)

- **모듈별 ablation**
  - 개선선의 각 구성요소(adapter/router/selector/critic)를 독립적으로 끈 비교.
  - 최소 조합은 static/SNR-only/severity-only/combined 정책과 regeneration ON/OFF의 교차 비교다.
- **실제 CBR/표준 bitstream 비교**
  - `transmission/`은 실제 bit-packing이지만 여전히 표준 변조/FEC를 재현하지 않음(byte 수는 정확, 채널 심볼/FEC 환산은 proxy).
- **DISTS/downstream task 지표**
  - 현재 `evaluators/`에는 없음.
  - LGVSC와 직접 비교하려면 필요.
- **latency·VRAM 비교**
- **LGVSC 1C 재현 실행**
  - 범위: 10영상 × 4모드
  - 현재: config·batch driver 준비 완료
  - 남은 작업: 실행·재현 수준 판정
- **paired 통계 검증**
  - 영상별 차이의 평균·표준편차·95% 신뢰구간을 보고하고, ETRI 10영상에서 선택한 설정은 별도 held-out 영상에서 한 번 더 검증한다.

## 일정

- 완료된 7~8월 범위
  - SNR sweep
  - channel-symbol 절감 1차 PoC
  - `PTC`, `SFR`, `SDI` 초기 결과
  - 전송 실험 정상화·3-GPU 10영상 sweep·결과 registry 고정
  - float32 digital 진단 harness·3-GPU 실행기·production 오류 집중 GPU 검증
  - 10dB decoder-step full 검증(3 core condition×100프레임) 및 raw output 원격–로컬 SHA 대조
  - 10dB fixed 양자화 10영상×6설정 full 재평가, 4-bit 운영점 확정 및 결과 registry 고정
- 근거: [status.md](./status.md)
- 남은 일정

| 시기 | 초점 | 산출물 |
|---|---|---|
| 9월 | 공정 selector 비교 기반 확정 | **완료** — proxy SKEM null 결과, `fixed_int4` 유지 |
| 10월 | 전송량 절감 + verifier 폐루프 | edge·uncertainty 절감 결과, 통합 평가 row, verifier candidate action 실제 sampler 반영 |
| 11월 | 동적 제어 + 최종 정리 | 예산 controller ablation, 별도 held-out 최종 검증, 비교 프로토콜·최종 보고서 |

## 신규 연구 아이템 확장 가능성

1. **전송률 적응형 영상 시맨틱 통신**
   - 입력: 채널 상태·영상 변화량
   - 목표: 필요한 의미 정보만 전송
   - 분석: rate–semantic reliability trade-off
2. **수신단 신뢰성 제어형 생성 복원**
   - 기준: 전송 packet
   - 제어: hallucination critic·regeneration controller
3. **시맨틱 전송 평가 벤치마크**
   - 지표: `PTC`, `SFR`, `SDI`, SRS
   - 검증: VQA·OWLv2·held-out 평가

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

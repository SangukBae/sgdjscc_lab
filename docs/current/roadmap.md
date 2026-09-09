---
status: active
updated: 2026-09-09
owner: ETRI SGD-JSCC 연구팀
source_commit: fd2426a
supersedes:
---

> [← 문서 색인](../README.md)

# 향후 연구개발 계획

- 문서 범위
  - 미완료 연구개발 과제
  - 기준: SAVER 구조, 시간축·영상, 할루시네이션, 평가 신뢰도, 전송량
  - 제외: 과거 Phase·차수별 구현 순서
- 연결 문서
  - 완료 상태: [status.md](./status.md)
  - SAVER 단일 기준: [saver_jscc_model_plan.md](./saver_jscc_model_plan.md)
  - 과거 이력: [etri_implementation_log.md](../archive/etri_implementation_log.md)
  - 한계·기술 부채: [open_issues.md](./open_issues.md)
  - 목표 정의: [system.md](../architecture/system.md)

## 공식 SAVER-JSCC 연구선 (2026-09-07 등록)

[SAVER-JSCC 모델 계획](./saver_jscc_model_plan.md)을 활성 모델·논문 단일 기준으로
등록했다. [negative semantic paper plan](./negative_semantic_paper_plan.md)은 SAVER의
SV0/SV1 진입을 판정하는 데이터·Oracle·packet/RSM 선행 계획으로 유지한다. 이전
[paper writing notes](../reference/paper_writing_notes.md)는 reliability-layer 연구선의
historical artifact이며 현재 논문의 claim 경계가 아니다.

SAVER 핵심 모델의 독립 tensor prototype, signed packet/ledger, fault channel,
prompt-only receiver-state bridge, dataset/loss/stage runner/checkpoint 형식은
`PROTOTYPE_IMPLEMENTED_UNTRAINED`이다. 이 선행 구현은 gate를 통과했다는 뜻이 아니며,
실험·claim 순서는 아래 표를 그대로 따른다. 실제 SAVER 학습 checkpoint와 formal 성능
근거는 없다.

G3 Oracle REVOKE의 세 receiver-memory arm manifest, frame-aligned Wan 주입, snapshot
전환 GOP 분할, ghost/identity evaluator와 read-only audit도 구현됐다. G2가
`NOT_PASSED`이므로 protocol은 `implementation_draft_waiting_for_g2` 상태를 유지하고
실행은 허가하지 않는다.

| 순서 | SAVER 단계 | 완료 조건 |
|---:|---|---|
| 완료·실패 | SV0 Oracle signed control | 40영상 320/320·감사 완료; 품질/false-suppression은 통과했지만 Oracle H_add 감소 0%로 `NOT_PASSED` |
| 구현 보존·formal 중단 | SV1 SAT + VREM | SV0 미통과. 현 구조로 학습하지 않고 versioned 재설계 여부부터 결정 |
| 구현 보존·formal 중단 | SV2 SM-DiT | SV0 미통과. 구조적 receiver-state injection 재설계 후보로만 유지 |
| 구현 보존·formal 중단 | SV3 JASR + channel codec | 상위 gate 미통과로 formal Pareto 실행 중단 |
| 구현 보존·formal 중단 | SV4 channel/packet robustness | 상위 gate 미통과로 formal protocol 실행 중단 |
| 봉인 유지 | SV5 held-out/generalization | 구조 재동결 전 Held-out 개봉 금지 |

SV0는 실제로 실패했다. 따라서 구현된 full SAVER prototype의 formal 학습·최종 채택을
중단하고 동결 계획의 Stop Track을 적용한다. 연구를 계속하려면 이번 Pilot을 본 데이터로
명시하고, prompt-only control을 구조적으로 바꾸는 별도 versioned 재설계와 새 gate가
필요하다.

### Negative-semantics 선행 gate 상태

G0 v1.2에서 프로토콜·실제 split·아카이브와 영상별 hash·사용 승인·Held-out 봉인을
완료하고 **`PASSED`**했다. G1은 정식 OVIS Pilot 240/240 run을 완료해
`gate_status: PASSED`(declared-seed 정의)를 얻었지만, v1.2 amendment가
이 reconstruction 경로의 `effective_seed_count=1`(seed 미소비)과
source-carried additional-object 혼입을 확인해 같은 gate를 정직하게
재적용하면 `NOT_PASSED`로 뒤집힌다
([v1.1 결과](../experiments/2026-09-06_negative_semantics_g1_v1_1_pilot_results.md),
[v1.2 amendment](../experiments/2026-09-06_negative_semantics_g1_v1_2_amendment.md)).
후속 fixed_int4–fixed_int6 bridge도 정식 완료했다. int6는 품질이 소폭
개선됐지만 int4 대비 bytes가 44.217% 증가했고 H_add의 유의한 개선은 없었다.
따라서 이후 primary development bit-depth는 `fixed_int4`다. 후속 G2 Oracle ABSENT
Pilot도 완료됐지만 primary few10의 Oracle source-paired H_add 감소가 0%여서
provisional gate가 `NOT_PASSED`다. 다음 우선순위는 G3 실행이 아니라 Stop Track 수용
또는 versioned receiver-control 구조 재설계 여부 결정이다.

| 순서 | G0 후속 작업 | 완료 조건 |
|---:|---|---|
| 완료 | 데이터 사용 권한 확정 | 프로젝트 비상업 연구 사용 승인과 raw-media 비재배포 경계 기록 |
| 완료 | 외부 split materialize | OVIS Pilot 40 / Train 3,471 / Validation 507 video ID·바이트·tree hash 고정 |
| 완료 | Pilot event annotation | official GT로 40개 독립 영상, 4 event type 각 10개 확보 |
| 완료 | Held-out 봉인 | DAVIS val 30개를 이전 split과 source-disjoint로 고정, method 개발 개봉 금지 |
| 완료 | G0 재감사 | `audit_negative_semantics_g0.py --require-pass` exit 0, v1.2 amendment 기록 |
| 완료 | G1 실행 준비 | source-only OWLv2 calibration, fixed_int4+both-omit, few10/full50, 3 seeds, resume 계약 구현 |
| 완료 | G1 정식 GPU run | Pilot 40영상 전체 완료(240/240, 실패 0), `g1_summary.json` gate: declared-seed `PASSED` |
| 완료 | G1 v1.2 amendment | effective-seed 재계산 결과 few10 gate `NOT_PASSED`; source-paired h_add로 full50 prevalence 미달 확인; code-clean v1_2c 결론 동일 |
| 완료 | int4–int6 bridge | OVIS Pilot 40영상 formal 완료, 실패 0, 품질 gate 통과, int6 bytes +44.217%, H_add 유의차 없음 |
| 완료·실패 | G2 Oracle ABSENT Pilot | clean commit `6f9593d`, 320/320·실패 0·runner audit `PASSED`; Oracle H_add 상대 감소 0%, 개선 CI 실패로 provisional `NOT_PASSED`; G1 dependency도 `NOT_PASSED` |

상세 동결값과 자동 annotation 규칙은
[G0 v1.2 기록](../experiments/2026-09-03_negative_semantics_g0_official_gt_amendment_v1_2.md)에 있다.
G1에서는 selector와 독립인 자동 evaluator를 먼저 고정한 뒤 GPU smoke와 정식 paired run을 구분한다.
명령과 결과 판독은 [G1 실행 준비 기록](../experiments/2026-09-03_negative_semantics_g1_preparation.md)을 따른다.

2026-09-07부터 논문 primary development 양자화 운용점은 `fixed_int4`다. 2026-09-04의
int6 결정은 후속 bridge 전 임시 판단으로 보존하되 현재 기준으로는 superseded다.
int6는 품질 민감도·robustness ablation과 명시적 시연 opt-in에만 사용한다. 근거는
[bridge 결과](../experiments/2026-09-07_negative_semantics_int6_bridge_results.md)와
[fixed_int4 결정](../experiments/2026-09-07_fixed_int4_primary_operating_point_decision.md)을 따른다.

G2의 상세 수치와 Stop Track 판정은
[2026-09-09 G2 결과](../experiments/2026-09-09_negative_semantics_g2_oracle_absent_pilot_results.md)에
고정했다. 같은 Pilot을 prompt/threshold 튜닝 후 confirmatory 데이터로 재사용하지 않는다.

- 관리 규칙
  - 메인 계획: 이 문서
  - 구현 완료 후: [status.md](./status.md)로 이동
  - 검증 완료 후: 날짜 기반 `docs/experiments/` 문서 추가

## 기존 backbone 개선선의 권장 실행 순서

통합 120-pair 개발평가까지 완료됐다. guide는 `candidate_both_omit`이 full50에서
semantic 지표를 유지하면서 baseline byte를 90.843% 줄였다. decoder는 few10이 평균
gate를 통과했지만 hallucination CI 상한이 margin을 넘었고, VAE-direct는 SSIM gate를
실패했다. 별도 held-out 데이터셋을 당장 만들 수 없어 최종 검증은 명시적으로
연기하고, guide 계약과 폐루프 구현을 먼저 진행한다.

### held-out 연기 결정 (2026-08-29)

- 별도 held-out 최종 검증은 취소하지 않고 **데이터셋 준비 시점까지 연기**한다.
- `few10 + candidate_both_omit`은 계속 개발셋 잠정 후보로만 유지한다.
- held-out 전에는 최종 operating point나 일반화 성능으로 표기하지 않는다.
- 기존 10영상으로 threshold를 추가 튜닝하지 않으며 baseline 경로를 보존한다.
- 새로운 독립 영상이 준비되면 `full50+baseline`, `full50+both-omit`,
  `few10+both-omit`을 같은 10dB 조건에서 paired 재검증한다.

| 순서 | 작업 | 완료 조건 |
|---:|---|---|
| 완료 | edge·uncertainty 전송량 절감 1차 | 16-profile·3-GPU full 완료·registry 보존. `combined_ds4` 조건부 최소 후보, uncertainty bypass 확인 |
| 완료 | guide 빈 조합 + 통합 평가 | 120/120 pair·real ensemble·paired CI 완료. both-omit guide와 few10 decoder를 개발셋 잠정 후보로 결정 |
| 1 | guide Tx/Rx 계약 정리 | both-omit을 opt-in 정책으로 명시하고 baseline·manifest·packet accounting·resume 계약 보존 |
| 2 | verifier→sampler 배선 | 실제 prompt 반영, retry·중단·fallback 조건 구현 및 추가 지연/시도 횟수 기록 |
| 3 | verifier 폐루프 ablation | OFF·로그 전용·retry/stop·prompt 제어 조건의 품질·hallucination·지연 비교 |
| 4 | 동적 예산 controller | 채널·uncertainty·verifier 위험도로 전송량과 복원 연산량을 결정하고 feedback/retransmission byte·RTT 포함 |
| 연기 | 별도 held-out 최종 검증 | 새 독립 데이터셋 준비 후 3조건 paired 검증; hallucination CI 포함 전 gate 통과 여부 판정 |
| 5 | 최종 문서 마감 | held-out 결과로 최종 operating point, 표·그래프·재현성 registry 확정 |

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
  - 수정된 10dB fixed-selector 양자화 재평가와 당시 rate-first `fixed_int4` operating point 확정
- 현재 판정
  - float32 10dB baseline: full 300프레임에서 AWGN 동등 이상, transport bit-exact 확인 완료
  - `fixed_int4`: float32 대비 28.45% byte 절감, 세 품질 허용 기준을 모두 통과한 최소 bit-depth이자 2026-09-07 이후 primary development 운용점
  - `fixed_int6`: float32에 더 가까운 품질의 secondary ablation/시연 opt-in 조건. OVIS bridge에서는 int4 대비 PSNR +0.100~0.146dB 대신 bytes +44.217%
  - exact matched-rate에서 proxy `skem_int4`는 fixed와 동일 schedule·품질로 수렴했다.
    raw 100 byte/video 차이도 manifest label 길이뿐이며 padding 후 동률이었다. 이 결과는
    selector 비교에서 fixed를 유지한다는 뜻이다. bit-depth는 후속 bridge 결과에 따라
    `fixed_int4`를 primary로 사용한다.
  - 통합 개발평가에서 `candidate_both_omit`은 full50 기준 semantic 지표를 유지하며
    baseline 대비 90.843% byte를 줄였다. `few10 + both-omit`은 평균 gate를 통과하고
    reconstruction 시간을 63.486% 줄였으나 hallucination/additional-object CI 상한이
    margin을 넘어 별도 held-out 전까지 잠정 후보다.
- 근사
  - 물리 channel symbol·FEC는 proxy
- 목표
  - 신뢰도 기반 동적 전송 대상·예산 결정

- **int6 bridge validation**
  - 동일 OVIS Pilot·both-omit·diffusion step에서 `fixed_int4`(G1 재사용)와
    `fixed_int6`(신규)를 비교한 **formal 실행을 완료**했다. 40영상×2정책,
    int6 실패 0, held-out 미접근, 산출물 10,589개 checksum 불일치 0이다.
  - int6는 두 정책 모두 품질 gate를 통과했지만 bundle bytes가 44.217% 증가했다.
    raw/source-paired H_add paired CI는 모두 0을 포함했고 ghost uncensored 표본은
    정책당 4개뿐이다.
  - G1 v1.2 amendment가 확인한 effective_seed_count=1을 반영해 처음부터
    seed 1개(2025)로 설계했다 — 3-seed 재부풀리기를 반복하지 않는다.
  - 기존 int4 reconstruction·accounting·detector 점수는 v1.1 run에서
    그대로 재사용하고(재계산 없음), 새로운 결과 root
    (`outputs/negative_semantics_int6_bridge_rtx4080`)만 사용한다.
  - 결과와 후속 결정:
    [bridge 결과](../experiments/2026-09-07_negative_semantics_int6_bridge_results.md),
    [fixed_int4 결정](../experiments/2026-09-07_fixed_int4_primary_operating_point_decision.md).

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
  - fixed_int4 edge·uncertainty 16-profile 3-GPU full ablation, 160/160 pair와 byte accounting 검증 및 결과 registry 고정
  - 통합 semantic·hallucination·temporal 120-pair 3-GPU full 평가와 개발셋 잠정 후보 선정
- 근거: [status.md](./status.md)
- 남은 일정

| 시기 | 초점 | 산출물 |
|---|---|---|
| 즉시 | Stop Track 결정 | G2 negative result와 최소 재현 artifact를 보존하고 현 prompt-only 연구선 종료 여부 결정 |
| 재설계 승인 시 | SV0 v2 설계 | Development split에서 receiver-state가 diffusion block에 직접 작용하는 구조와 새 protocol/version 동결 |
| 새 SV0 통과 후 | SAVER SV1/SV2 | SAT/VREM·SM-DiT 학습을 재개하고 prompt-only 대비 구조 효과 분리 |
| SV1/SV2 통과 후 | SAVER SV3/SV4 | JASR·codec·fault channel formal Pareto·robustness 실행 |
| 구조 동결 후 | SAVER SV5 | Validation·Held-out·독립 backbone; 현재는 봉인 유지 |

## 신규 연구 아이템 확장 가능성

아래 항목은 이제 SAVER-JSCC의 하위 후보이며, 각 SV gate를 통과하기 전에는 별도
완료 과제로 표현하지 않는다.

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

- [saver_jscc_model_plan.md](./saver_jscc_model_plan.md) — 활성 모델·논문 단일 기준
- [negative_semantic_paper_plan.md](./negative_semantic_paper_plan.md) — SV0/SV1 선행 gate
- [status.md](./status.md) — 현재 상태
- [open_issues.md](./open_issues.md) — 알려진 한계·기술 부채
- [../archive/etri_implementation_log.md](../archive/etri_implementation_log.md) — 상세 구현 이력
- [../architecture/tx_rx_contract.md](../architecture/tx_rx_contract.md) — LGVSC 시스템 구조 매핑
- [../experiments/2026-07_lgvsc_1c_reproduction.md](../experiments/2026-07_lgvsc_1c_reproduction.md) — 1C 실행 준비 상태

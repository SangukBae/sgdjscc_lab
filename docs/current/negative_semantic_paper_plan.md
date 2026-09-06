---
status: active
updated: 2026-09-06
owner: ETRI SGD-JSCC 연구팀
source_commit: aae9e26
primary_venue: IEEE Transactions on Multimedia
fallback_venue: IEEE Transactions on Circuits and Systems for Video Technology
g0_gate: PASSED
g1_gate: PASSED_DECLARED_SEED__NOT_PASSED_EFFECTIVE_SEED
supersedes: docs/reference/paper_writing_notes.md
---

> [← 문서 색인](../README.md) · [현재 로드맵](./roadmap.md) ·
> [구현 상태](./status.md) · [알려진 한계](./open_issues.md)

# 논문용 최종 개발 계획: Positive–Negative Semantic Source Coding과 Revocable Receiver Memory

## 0. 문서 목적과 최종 결정

이 문서는 다음 가제의 IEEE Transactions급 논문을 만들기 위한 **단일 실행 기준 문서**다.

> **Joint Positive–Negative Semantic Source Coding with Revocable Receiver Memory for Generative Video Communication**

최종 개발 결정은 **조건부 GO**다. 연구의 생존 여부는 구현량이나 일정이 아니라 아래
순차 gate의 실측 결과로 판정한다.

1. 현재 개발 backbone에서 additional-object hallucination이 반복되는가?
2. 완전한 `ABSENT` 정보를 주면 receiver가 실제로 이를 억제할 수 있는가?
3. Receiver Semantic Memory(RSM)가 identity/temporal consistency를 개선하는가?
4. append-only RSM의 ghost persistence를 assertion revocation이 줄이는가?
5. 같은 total rate와 compute에서 negative bits가 다른 용도보다 높은 효용을 가지는가?
6. 압축 packet과 online allocator가 oracle 이득의 충분한 부분을 유지하는가?
7. held-out 영상과 독립 generative backbone에서도 개선 방향이 유지되는가?

이 문서는 현재 [roadmap.md](./roadmap.md)의 일반적인 verifier 폐루프·동적 예산 제어와
구분되는 **별도 논문 연구선**을 정의한다. 완료된 단계는 날짜 기반
`docs/experiments/YYYY-MM-DD_<name>.md`에 결과를 고정하고, 구현 상태는
[status.md](./status.md)에 반영한다.

2026-09-03에 G0 v1.2 방법론·데이터 계약을 동결하고 **`PASSED`**했다.
OVIS official-GT Pilot 40 / Train 3,471 / Development 10 / Validation 507 /
DAVIS Held-out 30과 사용 승인·hash·seal을 확보했고 자동 감사 13/13을 통과했다.
사람 지각 hallucination은 claim하지 않으며 official-GT-anchored automatic
additional-object/ghost-track만 primary 표현으로 허용한다. 근거는
[G0 v1.2 기록](../experiments/2026-09-03_negative_semantics_g0_official_gt_amendment_v1_2.md)을 따른다.

### 0.1 연구 분기

| 분기 | 필수 조건 | 논문 범위 | 판단 |
|---|---|---|---|
| **Full Track** | P1·Oracle ABSENT·P2·Oracle REVOKE 모두 통과 | negative source coding + revocable RSM + joint allocator | TMM/TCSVT Transactions 목표 |
| **Reduced Track** | P1·Oracle ABSENT 통과, P2 또는 Oracle REVOKE 실패 | negative source coding + joint allocator, RSM 기여 제거 | 제목·기여 축소, conference 또는 범위 재검토 |
| **Stop Track** | Oracle ABSENT 실패 | 본 아이디어 중단 | packet·allocator·adapter 개발 금지 |

P2가 실패하면 현재 제목에서 `Revocable Receiver Memory`를 제거한다. Full Track의
결과가 없는데 Reduced Track 결과를 이용해 temporal-state revocation을 주장하지 않는다.

## 1. 현재 출발점과 동결 조건

### 1.1 개발 backbone

- ETRI 신규 기본 통신 backbone: `fixed_int6`
- G1 v1.1 현상 확인 backbone: `fixed_int4` stress configuration
- guide profile: `candidate_both_omit`
- reconstruction stress policy: `few10`
- 비교 기준: `full50 + baseline`, `full50 + both-omit`
- 주요 조건: reliable-digital, fixed-reference 10 dB, seed 2025

`fixed_int4 + both-omit + few10`은 **frozen development stress configuration**이다.
held-out 검증이 끝난 최종 operating point가 아니며 G1 v1.1의 비교 가능성을 위해
변경하지 않는다. 2026-09-04 이후 신규 ETRI 개발과 G2 이후 primary method는
`fixed_int6`를 후보로 사용하되, G1 통과 후 동일 조건의 int4/int6 paired bridge
validation을 먼저 통과해야 한다. 결정 근거는
[int6 ETRI 운용점 기록](../experiments/2026-09-04_int6_etri_operating_point_decision.md)을 따른다.

현재 개발셋 결과는 다음과 같다.

| 항목 | 현재 관측 |
|---|---:|
| 전송량 | 219,459.7 bytes/video, `full50 + baseline` 대비 -90.843% |
| reconstruction 시간 | 39.454 s/video, -63.486% |
| PSNR 변화 | -0.2435 dB |
| SSIM 변화 | -0.00188 |
| LPIPS 변화 | +0.01558 |
| open hallucination CI 상한 | 0.0525 |
| additional-object CI 상한 | 0.0570 |

위 결과는 연구 동기와 stress condition으로만 사용한다. 특히 기존 10영상은 threshold를
추가 튜닝하는 최종 검증셋으로 사용하지 않는다. 근거는
[통합 평가 결과](../experiments/2026-08-29_integrated_semantic_validation_10db.md)와
[open issues](./open_issues.md)를 따른다.

### 1.2 현재 구현에서 재사용할 자산

- exact binary packet과 byte accounting: `transmission/`
- packet verifier와 open/closed presence evaluation: `evaluators/`
- negative prompt를 받을 수 있는 내부 diffusion wrapper
- GOP/segment와 Rx-legal 생성 계약: `video/temporal_pipeline.py`,
  `video/generation/`
- 실제 Wan/SVD worker 실행 경로
- `PTC`, `SFR`, `SDI`, SRS와 video-level 결과 registry
- 3-GPU 실행, manifest, resume, checksum 검증

### 1.3 현재 구현의 필수 보완점

- Wan worker는 `side_infos`를 받지만 실제 conditioning에 사용하지 않는다.
- GOP 사이에서 track별 semantic state를 유지하는 persistent RSM이 없다.
- `verifier_controller`의 candidate action은 sampler에 실제 개입하지 않는다.
- semantic packet 자체의 drop/reorder/corruption protocol이 없다.
- positive/negative action을 공통 예산에서 고르는 allocator가 없다.
- 실제 FEC·modulation은 proxy이므로 첫 논문의 core claim으로 사용하지 않는다.

## 2. 연구 질문, 가설과 주장 경계

### 2.1 핵심 연구 질문

> 제한된 통신 예산에서 receiver가 생성하기 쉬운 잘못된 의미와 더 이상 유효하지 않은
> 의미를 송신단이 별도의 rate-bearing source로 부호화하고, versioned receiver memory를
> 갱신하면, 동일 total rate와 compute에서 positive-only 전송보다 hallucination과 ghost
> persistence를 낮출 수 있는가?

### 2.2 검증 가설

| ID | 가설 | 주 지표 | 실패 시 조치 |
|---|---|---|---|
| H1 | Oracle `ABSENT`는 추가 객체를 억제한다. | `H_add` | 전체 연구 중단 |
| H2 | append-only RSM은 No-RSM보다 identity/temporal consistency를 높인다. | `I_reid`, identity consistency | RSM 주기여 제거 |
| H3 | append-only RSM에는 exit 이후 ghost cost가 존재한다. | `H_ghost`, survival AUC | RSM 주기여 제거 또는 다른 stateful decoder 선택 |
| H4 | revocable RSM은 identity 이득을 유지하며 ghost를 줄인다. | `H_ghost`, `I_reid`, false suppression | RSM adapter 개발 중단 |
| H5 | negative action은 같은 비트의 positive/protection action보다 일부 조건에서 높은 marginal utility를 갖는다. | matched-rate Pareto | joint allocation 주장 제거 |
| H6 | 압축·online 방법은 oracle 개선량의 사전 기준 이상을 유지한다. | oracle gain retention | codec/selector 재설계 후 1회 재검증, 재실패 시 Reduced/Stop 판정 |

### 2.3 신규성의 중심

논문 기여는 다음 네 항목으로 제한한다.

1. positive와 negative semantics를 공통 rate budget에서 선택하는 rate-constrained
   semantic source-coding formulation
2. open-world absence와 counterfactual marginal utility per bit 기반 negative coding
3. versioned assertion-level RSM과 reorder/duplication-safe anti-resurrection,
   loss-mitigated revocation protocol
4. 같은 total rate와 compute에서 hallucination, ghost survival, false suppression,
   reappearance identity를 함께 검증하는 평가 체계

### 2.4 주장하지 않을 것

- negative prompting 또는 diffusion negation 자체의 최초 제안
- scene-graph delete/tombstone command 자체의 최초 제안
- semantic packet, HARQ 또는 retransmission 자체의 최초 제안
- 임의 packet loss에서도 revocation 전달을 보장한다는 주장
- prompt ledger만 구현하고 neural latent/KV state를 삭제했다는 주장
- `few10/full50`만 비교하고 독립 decoder에 일반화했다는 주장
- 현재 ETRI 10영상 결과를 최종 일반화 결과로 사용하는 것
- proxy FEC/channel symbol을 실제 on-wire coded bits로 부르는 것

Negative concept 선택·guidance는 NPC와 NEGATE 같은 선행 연구가 있고, MPEG-4 BIFS에는
scene-graph insert/delete command가 존재한다. 이들은 receiver-only 또는 protocol baseline으로
취급한다.

- [NPC: Automated Negative Prompting](https://openaccess.thecvf.com/content/WACV2026/papers/Park_Guiding_What_Not_to_Generate_Automated_Negative_Prompting_for_Text-Image_WACV_2026_paper.pdf)
- [NEGATE: Constrained Semantic Guidance for Text-to-Video Diffusion](https://arxiv.org/abs/2603.06533)
- [MPEG-4 BIFS 공식 설명](https://mpeg.chiariglione.org/standards/mpeg-4/scene-description-and-application-engine.html)
- [Semantic error detector와 HARQ 사례](https://ieeexplore.ieee.org/abstract/document/9955991/)

## 3. 연구 범위

### 3.1 Full Track 필수 모듈

1. RSM state machine
2. 최소 RSM-lite prompt/token conditioning
3. versioned negative packet codec
4. dependency-aware joint rate allocator
5. calibrated online selector
6. hallucination/ghost/false-suppression/reappearance evaluator

RSM conditioning adapter와 learned risk predictor는 각각 G8/G9 포함 gate를 통과한 경우에만
최종 방법에 넣는다. RSM-lite 또는 heuristic selector보다 개선되지 않는 학습 모듈은
개발 완료 여부와 무관하게 제거한다.

### 3.2 후반 extension

- `NOTREL(subject, relation, object)` packet
- learned entropy model, Bloom/XOR/Cuckoo filter
- 일반 FEC code-rate 최적화
- feedback ACK 기반 incremental redundancy
- 다중 receiver/model-version compatibility

`NOTREL`은 ABSENT와 revocation이 모두 통과한 후에만 구현한다. 첫 논문의 main table은
NOTREL 없이도 닫혀야 한다.

### 3.3 첫 논문에서 제외

- full PHY waveform과 표준 변조/FEC 재현
- multi-user/multicast
- online foundation-model fine-tuning
- cryptographic provenance·privacy
- 기존 비디오 코덱을 대체한다는 rate–distortion 우위 주장

## 4. 시스템 모델

### 4.1 시간 단위와 정보 집합

영상은 GOP 또는 segment 단위 $t$로 처리한다.

- $X_t$: 송신단 원본 GOP
- $S_t^+$: positive semantics와 visual/keyframe source
- $S_t^-$: confirmed-absent 또는 더 이상 유효하지 않은 assertion
- $C_t$: application-level protection action
- $Z_t=(Z_t^+,Z_t^-,Z_t^C)$: 실제 전송 binary packet
- $M_t$: receiver semantic memory
- $D$: frozen 또는 adapter-conditioned generative decoder
- $\hat X_t$: 복원 GOP

송신단은 현재와 과거 원본만 사용한다. 수신단은 수신 packet과 과거 receiver state만
사용한다. 미래 frame 또는 전송되지 않은 원본 target frame은 receiver 입력에 사용할 수
없다.

### 4.2 Negative Semantic Source

전체 vocabulary의 complement를 보내지 않는다. 후보 concept $c$는 다음 조건을
만족해야 한다.

\[
c \in \mathcal C_t^-
\quad\text{if}\quad
q_D(c\mid S_t^+,M_{t-1},D_{\mathrm{id}})\text{ is high}
\quad\land\quad
a(c\mid X_t)=\text{confirmed-absent}.
\]

absence 상태는 반드시 다음 세 가지로 구분한다.

\[
a(c\mid X_t)\in
\{\text{present},\text{confirmed-absent},\text{unknown}\}.
\]

- `present`: negative 전송 금지
- `confirmed-absent`: negative 후보 허용
- `unknown`: 보수적으로 negative 전송 금지

`not detected = absent`로 취급하지 않는다. absence confidence는 pilot/dev에서 calibration하고
held-out threshold를 다시 조정하지 않는다.

### 4.3 RSM 데이터 모델

RSM은 decoder 외부의 명시적 state로 시작한다.

```text
EntityRecord
  scene_epoch
  entity_id
  concept_id
  identity_embedding_ref
  created_version

AssertionRecord
  scene_epoch
  assertion_id
  entity_id / subject_id / object_id
  predicate
  state: ACTIVE | SUSPENDED | REVOKED
  version
  effective_frame
  valid_from / valid_until
  confidence
  evidence_embedding_ref
```

identity와 render visibility를 분리한다. occlusion이나 out-of-view가 identity 삭제로
이어져서는 안 된다.

### 4.4 상태 전이

```text
ASSERT  -> ACTIVE
ACTIVE  -> SUSPENDED     # 일시적 occlusion, identity 유지
SUSPENDED -> ACTIVE      # RESUME, 재등장
ACTIVE/SUSPENDED -> REVOKED
SCENE_RESET -> scene_epoch 증가, 이전 scene assertion 비활성화
```

상황별 의미는 다음과 같다.

| 상황 | 상태 처리 | identity 처리 |
|---|---|---|
| 객체 exit/out-of-view | visibility assertion revoke 또는 비활성화 | 재등장 가능성이 있으면 cold identity 유지 |
| 일시적 occlusion | `SUSPEND` | 유지 |
| reappearance | `RESUME` | 기존 identity 재사용 |
| 잘못된 semantic assertion | permanent `REVOKE` | assertion만 제거 |
| scene cut | `SCENE_RESET` | 이전 epoch state 비활성화 |

### 4.5 RSM 세 조건

P2는 다음 세 조건을 반드시 함께 비교한다.

1. `No-RSM`: 현재 GOP 조건만 사용
2. `Append-only RSM`: memory를 누적하지만 revocation을 적용하지 않음
3. `Revocable RSM`: versioned state operation을 적용

append-only RSM이 No-RSM 대비 identity/temporal consistency를 높이지 못하면, 그 RSM이
만든 ghost를 다시 제거한 결과는 기여로 인정하지 않는다.

### 4.6 RSM-lite와 RSM-adapter

- **RSM-lite**
  - RSM snapshot/delta를 deterministic prompt 또는 token sequence로 컴파일
  - Wan `side_infos`를 실제 prompt/token conditioning에 연결
  - 논문 표현: `receiver-side semantic ledger conditioning`
- **RSM-adapter**
  - entity/assertion token encoder
  - 작은 cross-attention 또는 FiLM-style adapter
  - 본체 decoder는 가능하면 frozen
  - 논문 표현: `adapter-conditioned receiver semantic memory`

RSM-lite만 구현된 상태에서 `neural state deletion`, `latent deletion`, `KV cache deletion`을
주장하지 않는다.

## 5. Packet protocol

### 5.1 최소 packet 종류

```text
ABSENT(scope_id, concept_id, confidence, valid_from, valid_until)
ENTITY_CREATE(scene_epoch, entity_id, concept_id, identity_ref, version)
ASSERT(scene_epoch, assertion_id, entity_id, predicate, state, version, validity)
STATE(scene_epoch, assertion_id, mode, effective_frame, version)
SCENE_RESET(new_epoch, effective_frame, version)
STATE_SNAPSHOT(scene_epoch, snapshot_version, active_assertions...)
```

`STATE.mode`는 최소 `SUSPEND`, `RESUME`, `REVOKE`를 지원한다. `NOTREL`은 후반
extension으로 유지한다.

### 5.2 공통 header

최소 header 필드는 다음과 같다.

```text
schema_version
packet_type
scene_epoch
sequence_number
payload_length
ontology_version_hash
checksum
```

필드 bit-width와 entropy coding은 Gate G4에서 동결한다. 문서에 없는 hidden metadata나
JSON label 길이가 결과 rate를 바꾸지 않도록 논문 실험은 binary wire packet을 기준으로 한다.

### 5.3 Ontology와 ID 계약

- shared concept ontology의 version/hash를 기록한다.
- codebook은 사전 공유 가능하지만 논문과 artifact에 공개한다.
- open-vocabulary concept은 UTF-8/string fallback 또는 reserved ID 경로를 사용하고 실제
  byte를 센다.
- `assertion_id`를 참조하는 STATE는 해당 ASSERT보다 먼저 도착해도 pending tombstone으로
  보존한다.
- epoch가 다른 assertion ID는 동일 ID여도 다른 객체로 취급한다.
- confidence quantization bit 수는 rate ablation에 포함한다.

### 5.4 Protocol 불변조건

1. **Idempotence**: 같은 packet을 여러 번 적용해도 state가 변하지 않는다.
2. **Version monotonicity**: 현재 version 이하의 operation은 무시한다.
3. **Reorder safety**: REVOKE를 먼저 받은 뒤 늦은 ASSERT가 와도 부활하지 않는다.
4. **Epoch isolation**: 이전 scene packet이 새 scene state를 바꾸지 않는다.
5. **Conservative corruption handling**: checksum이 실패한 negative packet은 적용하지 않는다.
6. **Rx legality**: RSM update에 원본 target frame을 사용하지 않는다.

### 5.5 Loss 한계와 완화

REVOKE 자체가 유실되면 receiver는 disappearance를 알 수 없으므로 임의 loss에서
anti-resurrection을 보장하지 않는다. 첫 논문에서는 다음 application-level 방법만 비교한다.

- fixed repetition
- 다음 packet/keyframe에 latest state piggyback
- 주기적 `STATE_SNAPSHOT`

모든 반복·snapshot·piggyback byte는 rate에 포함한다. ACK, incremental redundancy,
일반 FEC 최적화는 후속 TCOM/TWC 범위로 둔다.

## 6. Rate와 accounting 계약

### 6.1 첫 논문의 rate 범위

일반 채널 코드는 모든 비교군에 동일하게 고정한다. 최적화 예산은 다음과 같다.

\[
R_P + R_N + R_{\mathrm{rev\mbox{-}protect}} \le B.
\]

- $R_P$: positive semantic, visual latent, keyframe, positive header
- $R_N$: ABSENT/STATE/RSM negative payload와 header
- $R_{\mathrm{rev-protect}}$: repetition, piggyback, snapshot overhead

실제 FEC code rate까지 선택할 경우 본 문서 범위를 벗어나며 논문 제목과 formulation을
`joint source–channel` 또는 `protection-aware`로 다시 승인해야 한다.

### 6.2 공식 rate

```text
exact_initial_bytes
negative_packet_bytes
revocation_protection_bytes
feedback_bytes
retransmission_bytes
padding_bytes
total_on_wire_bytes
effective_bits_per_frame
bpp
```

\[
R_{\mathrm{eff}}
=
8\,\frac{
\text{initial}+\text{negative}+\text{protection}+\text{feedback}
+\text{retransmission}+\text{padding}
}{N_{\mathrm{frame}}}.
\]

`proxy_channel_symbols`는 별도 참고 열로 두고 exact bytes와 합산하지 않는다.

### 6.3 Matched-rate 원칙

- 모든 방법은 같은 $B$에서 positive action도 다시 최적화한다.
- 단순히 baseline packet에 padding을 추가해 budget을 맞춘 결과만으로 우위를 주장하지
  않는다.
- positive-only baseline은 negative에 사용한 bit를 extra keyframe, latent precision,
  entity/relation 또는 보호에 실제로 사용할 기회를 갖는다.
- 최소 4개 rate budget에서 Pareto frontier를 작성한다. budget 값은 Gate G0 pilot에서
  현재 실제 bpp 분포를 본 뒤 본실험 전에 동결한다.

## 7. Joint allocator

### 7.1 Action 집합

```text
Positive source actions
  entity assertion
  relation assertion
  latent precision level
  extra keyframe

Negative source actions
  ABSENT concept
  SUSPEND / RESUME / REVOKE
  negative confidence precision

Revocation-protection actions
  repetition count
  next-packet piggyback
  periodic state snapshot
```

### 7.2 의존성

- protection action은 보호 대상 packet이 선택된 경우만 유효하다.
- `RESUME`은 해당 entity/assertion이 receiver에 정의돼 있어야 한다.
- precision 증가는 base payload 선택 이후에만 가능하다.
- REVOKE/ASSERT의 version order를 깨는 action 조합은 금지한다.
- keyframe/latent action의 상호 배타성은 기존 transport 계약을 따른다.

일반 additive knapsack은 위 의존성과 concept 간 상호작용을 완전히 처리하지 못한다.

### 7.3 목적함수

주 결과는 임의의 단일 합성 점수보다 constrained optimization과 Pareto frontier로 보고한다.

\[
\min_{\mathcal P,\mathcal N,\mathcal C}
\quad H_{\mathrm{add}}+\alpha A_{\mathrm{ghost}}
\]

subject to

\[
R\le B,\qquad
D_{\mathrm{false}}\le\epsilon_F,\qquad
\Delta Q\le\epsilon_Q,\qquad
L_{\mathrm{e2e}}\le L_{\max}.
\]

여기서 $A_{\mathrm{ghost}}$는 exit 이후 ghost survival curve의 면적이다. 여러
$\alpha$와 $B$를 sweep해 rate–hallucination–suppression frontier를 보고한다.

### 7.4 Counterfactual marginal utility

Oracle에서는 candidate 추가 intervention의 실제 효과를 측정한다.

\[
\Delta(c\mid\mathcal N)
=L(\hat X(\mathcal N))-L(\hat X(\mathcal N\cup\{c\})).
\]

용어는 `counterfactual marginal utility per transmitted bit`로 통일한다. 정식 structural
causal model을 제시하지 않는 한 `causal utility`라고 부르지 않는다.

### 7.5 단계별 solver

| 단계 | solver | 용도 |
|---|---|---|
| Oracle-small | exhaustive search 또는 ILP/branch-and-bound | upper bound, action interaction 분석 |
| Oracle-large | conditional marginal을 재계산하는 dependency-aware greedy | 확장 데이터 upper bound 근사 |
| Online-heuristic | calibrated hallucination risk/bit greedy | 학습 전 deployable baseline |
| Online-learned | $q_\phi(c\mid S_t^+,M_{t-1},D_{id})$ | 최종 predictor 기반 allocator |

Oracle 값은 deployable 성능으로 보고하지 않는다. train/dev rollout으로 predictor를 만들고,
held-out reconstruction을 predictor 학습에 사용하지 않는다.

## 8. 데이터와 split

### 8.1 역할 분리

| split | 목적 | 허용 작업 |
|---|---|---|
| Pilot | phenomenon density, metric sanity, provisional threshold 결정 | 탐색 허용, 최종 주장 금지 |
| Train | RSM adapter와 risk predictor 학습 | 학습만 허용 |
| Development | packet, allocator, hyperparameter 개발 | 반복 실험 허용 |
| Validation | threshold·operating point 선택 | 제한된 선택만 허용 |
| Held-out Test | 최종 paired evaluation | 한 번만 개봉, 재튜닝 금지 |

기존 ETRI 10영상은 Development 전용 자산이다. Pilot이나 최종 test에 재사용하지 않는다.

### 8.2 공개 데이터 후보

- [DAVIS 2017](https://davischallenge.org/): 정밀한 multi-object mask와 짧은 controlled sequence
- [YouTube-VOS/VIS](https://youtube-vos.org/dataset/): 다양한 object track과 대규모 영상
- [OVIS](https://songbai.site/ovis/): Pilot에 선택; exhaustive category mask·instance identity·occlusion GT
- [TAO](https://taodataset.org/): open-world category와 긴 object track
- [BDD100K](https://bdd-data.berkeley.edu/): 사람·차량 중심 실제 주행 scene

최종 조합은 다음 기준으로 Gate G0 v1.2에서 선택했다.

- exit·occlusion·reappearance·scene cut event 밀도
- frame-level mask/track annotation 가용성
- 학습·평가·재배포 라이선스
- 저장 공간과 preprocessing 비용
- 사람·차량에 편중되지 않은 concept 다양성

### 8.3 Event annotation

각 event는 다음 필드를 갖는다.

```text
video_id
scene_epoch
entity_id
concept_id
event_type: ENTER | EXIT | OCCLUDE | REAPPEAR | SCENE_CUT
start_frame
end_frame
visibility_before / visibility_after
annotation_source
official_gt_verified
boundary_unit / source_frame_stride
```

Pilot gate는 provider GT로 검증된 최소 30~50개의 독립 event를 요구하며, G1 이후 생성
실험은 최소 3개 generation seed를 요구한다. 같은
영상의 인접 event를 완전히 독립 표본으로 취급하지 않는다. 최종 평가 목표는 최소 50개
독립 영상과 100개 이상의 검증 event이며, 실제 확보 가능성은 G0 데이터 audit에서 확정한다.

### 8.4 Leakage 방지

- selector와 최종 evaluator에 같은 detector weight를 사용하지 않는다.
- OWLv2가 selector에 쓰이면 최종 object 판정에는 weight를 공유하지 않는 별도 자동
  evaluator를 사용한다. threshold와 association rule은 Validation 전에 Pilot에서 동결한다.
- oracle rollout 결과는 oracle table과 predictor train에만 사용하며 held-out policy 선택에
  사용하지 않는다.
- concept별 split만 하지 않고 video/source별 split을 우선한다.

## 9. 공식 평가 지표

### 9.1 Primary endpoints

#### Additional-object hallucination

\[
H_{\mathrm{add}}
=\frac{\#\text{generated absent concept events}}
{\#\text{absent opportunities}}.
\]

`absent opportunity`는 평가 ontology와 event scope에서 해당 concept이 confirmed-absent인
경우로 정의한다. 단순 frame 수를 분모로 사용하지 않는다.

#### Ghost persistence

entity $i$의 GT exit 시점을 $t_i^{exit}$라 할 때

\[
T_{\mathrm{ghost}}^{(i)}
=\min\{\tau\ge0:\hat y_{i,t_i^{exit}+\tau}=0\}
\]

와 고정 horizon의 survival curve/AUC를 함께 보고한다. censoring과 scene end를 명시한다.

### 9.2 Safety endpoints

- false suppression: 실제 present 객체가 negative control 때문에 사라진 비율
- object recall 변화
- reappearance identity consistency
- 잘못된 relation suppression(NOTREL extension 시)

### 9.3 Quality와 semantic endpoints

- PSNR, SSIM, LPIPS
- SRS와 object preservation
- PTC, SFR, SDI
- temporal identity consistency
- 가능하면 DISTS/VMAF와 downstream task metric

기존 개발 screening margin을 provisional non-inferiority 기준으로 시작한다.

| 지표 | provisional margin |
|---|---:|
| PSNR 하락 | 0.5 dB |
| SSIM 하락 | 0.01 |
| LPIPS 증가 | 0.02 |
| false suppression 증가 | 0.01~0.02 absolute |

최종 margin은 Pilot split 결과와 application requirement로 G0에서 확정하며 Oracle 본실험
후 변경하지 않는다.

### 9.4 Rate와 compute

- exact total bytes, effective bits/frame, bpp
- initial/negative/protection/feedback/retransmission/padding byte breakdown
- diffusion step 수와 generation count
- detector/VLM/predictor 호출 수
- p50/p95 end-to-end latency
- GPU seconds와 peak VRAM
- retry와 regeneration latency

`matched-compute`는 generation count만 같다는 의미가 아니다. primary matched-compute table은
diffusion step/rollout budget을 맞추고, practical table은 실제 latency와 VRAM을 별도로 보고한다.

### 9.5 통계

- 기본 비교: 같은 video·seed·budget의 paired difference
- 상위 cluster: video
- 하위 반복: event와 seed
- 5,000회 deterministic hierarchical bootstrap 95% CI
- primary endpoints: `H_add`, ghost survival AUC
- secondary family: false suppression, identity, quality, latency
- 다수 secondary 검정은 Holm correction 또는 exploratory 표기로 구분
- 평균뿐 아니라 median, p95, failure count를 보고

primary 결과는 method·budget 정보를 받지 않는 frozen automatic evaluator로 산출한다.
따라서 사람 지각 기반 hallucination 또는 human-verified identity라는 표현은 쓰지 않는다.

## 10. 비교군과 ablation

### 10.1 필수 matched-rate 비교군

1. positive-only, 각 budget에서 재최적화
2. 같은 bit를 extra positive entity/relation에 사용
3. 같은 bit를 latent precision 또는 extra keyframe에 사용
4. receiver verifier + retry
5. receiver-only negative guidance(NPC/NEGATE 계열 또는 재현 가능한 동등 baseline)
6. raw UTF-8 negative text
7. fixed-rule binary negative packet
8. compressed negative packet
9. compressed negative + RSM-lite
10. compressed negative + revocable RSM-adapter
11. oracle negative semantics
12. oracle joint allocator

### 10.2 RSM ablation

| 조건 | memory | revocation | adapter |
|---|---:|---:|---:|
| No-RSM | off | off | off |
| Append-only RSM-lite | prompt/token | off | off |
| Revocable RSM-lite | prompt/token | on | off |
| Append-only adapter | token memory | off | on |
| Revocable adapter | token memory | on | on |

### 10.3 Packet/protocol ablation

- raw string vs concept ID
- confidence 없음 vs quantized confidence
- no protection vs repetition vs piggyback vs snapshot
- in-order vs reorder vs duplication vs packet loss
- stale ASSERT after REVOKE
- STATE-before-ASSERT
- scene reset 전후 late packet

### 10.4 Allocator ablation

- fixed $R_N/R$ ratio
- positive-only
- hallucination-risk only
- utility/bit without dependency handling
- dependency-aware heuristic
- learned predictor
- oracle

### 10.5 Backbone 일반화

- primary Full Track backbone: RSM conditioning이 가능한 Wan 기반 GOP generator
- additional-object control: 내부 SGD-JSCC diffusion `few10/full50`
- 독립 확인: 실제로 다른 generative backbone 최소 1개

두 번째 backbone에 RSM-lite만 이식했다면 protocol generalization만 주장한다. adapter를
이식하지 않았다면 adapter가 model-agnostic하다고 쓰지 않는다.

### 10.6 논문 전체 context baseline과 채널 조건

Negative-layer 내부 비교와 별도로 다음 전체 시스템 baseline을 가능한 범위에서 같은
source와 frame 구간으로 평가한다.

- original SGD-JSCC
- 재현 가능한 LGVSC 계열 또는 현재 LGVSC-inspired 4-mode
- H.264/H.265/AV1, 가능하면 VVC
- WITT/DeepJSCC/DiffJSCC 중 동일 데이터와 rate 정의로 재현 가능한 방법

표준 코덱의 rate는 실제 bitstream 크기를 사용하고, semantic system의 proxy symbol과
직접 합산하지 않는다. 주장은 표준 코덱 대비 pixel rate–distortion 우위가 아니라,
같은 전송 예산에서 제안 negative layer가 generative receiver의 semantic failure를 얼마나
줄이는지에 둔다.

채널 실험은 다음 두 층으로 분리한다.

1. **Primary source-coding table**: reliable-digital, 10 dB reference, 모든 방법에 같은
   channel protection
2. **Robustness table**: 기존 evaluation protocol의 AWGN SNR 중 사전 선택한 subset과
   packet loss/reorder/duplication sweep

packet-loss 후보값은 0%, 1%, 5%, 10%, reorder window는 0, 1, 3 GOP로 시작하되
G0 Pilot에서 실제 failure 분포를 확인한 뒤 본실험 전에 동결한다.

## 11. 단계별 실행 계획과 gate

아래 gate 중 G0의 rate·metric·margin·annotation 규칙과 실제 split은 2026-09-03
v1.2에서 동결하고 통과했다. 동결값을 바꿔야 한다면 영향받는 다음 gate 전에
versioned amendment를 남기며 기존 기록을 덮어쓰지 않는다.

### G0. Protocol freeze와 데이터 audit

**현재 판정: `PASSED`, 자동 감사 13/13 (2026-09-03 v1.2)**

- 완료: Full Track, TMM primary venue, wire-byte rate 단위, 4개 budget, metric/margin,
  ontology/state/event schema, 실제 5-way split·공식 archive·영상별 tree hash·사용 승인·Held-out seal
- 감사 통과: 13개 조건 중 13개
- Pilot: OVIS official-GT-derived 40개, 40개 독립 영상, event type별 10개
- 근거: [G0 v1.2 official-GT 기록](../experiments/2026-09-03_negative_semantics_g0_official_gt_amendment_v1_2.md)
- 강제 검사: `python scripts/audit_negative_semantics_g0.py --repo-root . --require-pass`

**작업**

- 연구 Track, primary venue와 rate 단위 확정
- dataset license·event density audit
- Pilot/Train/Dev/Validation/Test 목록을 파일로 고정
- concept ontology와 event annotation schema v1.2
- primary/secondary metric과 margin 사전 선언
- 최소 4개 rate budget과 compute budget 정의

**산출물**

- `configs/experiments/negative_semantics/g0_protocol.yaml`
- dataset split manifest와 SHA-256
- `docs/experiments/<date>_negative_semantics_g0_protocol.md`
- preregistered gate JSON

**통과 조건**

- Pilot에 30~50개 독립 event 확보
- held-out이 기존 ETRI 10영상과 source-level로 분리
- 라이선스와 재현 절차 확인

### G1. P1 additional-object phenomenon

**작업**

- frozen `fixed_int4 + both-omit + few10` 실행
- 최소 3 seeds
- absent opportunity와 additional object를 event/video 단위 집계
- `full50 + both-omit`도 병행해 sampler-step 의존성 확인
- G1 통과 후 같은 공개 데이터·seed·diffusion step에서
  `fixed_int6 + both-omit` bridge condition을 추가해 ETRI 기본 운용점에서도 현상과
  exact-byte 차이를 확인

**통과 조건**

- hallucination이 한 영상·한 seed에만 집중되지 않음
- 사전 정의한 최소 prevalence와 event count 충족
- frozen independent automatic evaluator의 calibration과 selector weight 분리 통과
- int6 bridge 결과가 없는 동안 G1 int4 prevalence를 int6 운용 성능으로 일반화하지 않음

**실패 시**

- 사전 허용된 범위에서 Pilot dataset만 1회 교체 가능
- 두 번째 dataset에서도 실패하면 Stop Track

**실행 결과 (2026-09-06)**

- `fixed_int4 + both-omit`, `few10`/`full50`, 3 declared seed로 Pilot
  40영상 전체(child run 6개 × 40영상 = 240/240)를 완료했다. 실패 0,
  held-out 미접근. declared-seed 정의로는 위 통과 조건을 모두 만족해
  `gate_status: PASSED`다.
- 그러나 v1.2 amendment가 이 reconstruction 경로는 `seed`를 소비하지 않아
  `effective_seed_count=1`임을 확인했다. 같은 threshold를 effective-seed
  지표에 재적용하면 "hallucination이 한 seed에만 집중되지 않음" 조건이
  **FAIL**로 뒤집혀 few10 기준 gate가 `NOT_PASSED`가 된다. 즉 3-seed
  통과는 실제로 존재하지 않았던 독립성에 의존했다.
- source-paired additional-object(원본 프레임이 이미 오탐이던 경우를 제외)로
  다시 계산하면 h_add가 약 47~51% 낮아지고, full50은 사전 정의한
  `h_add_min` prevalence 조건을 만족하지 못한다.
- 따라서 이 gate의 **정직한 판정은 "P1 현상이 존재한다(raw, 단일 effective
  seed, 24/23개 영상에 분산됨)"까지이며, "여러 독립 seed에 걸쳐 재현된다"와
  "reconstruction이 새로 만든 오탐이 prevalence 기준을 만족한다(특히
  full50)"는 아직 성립하지 않는다.** G2 진입 전 이 경계를 반영해야 한다.
- int6 bridge는 코드·설정·테스트를 완료해 실행 준비가 됐고 GPU 실행만
  남았다.
- 근거: [G1 v1.1 결과·감사](../experiments/2026-09-06_negative_semantics_g1_v1_1_pilot_results.md),
  [G1 v1.2 amendment](../experiments/2026-09-06_negative_semantics_g1_v1_2_amendment.md),
  [int6 bridge 준비](../experiments/2026-09-06_negative_semantics_int6_bridge_preparation.md)

### G2. Oracle ABSENT controllability

**작업**

- RSM 없이 완벽한 confirmed-absent 목록을 receiver negative condition으로 제공
- no negative, random negative, frequency negative, oracle negative 비교
- matched generation/step budget 유지

**provisional 통과 조건**

- $H_{\mathrm{add}}$ 상대 감소 25% 이상
- paired one-sided 95% CI가 개선 방향
- false suppression 증가 CI 상한 0.01~0.02 이내
- PSNR/SSIM/LPIPS non-inferiority 통과

**실패 시**

- RSM, packet codec, allocator 개발 중단
- receiver control mechanism을 바꾸는 별도 연구로 재정의하지 않는 한 본 계획 종료

### G3. RSM-lite와 P2/Oracle REVOKE

**작업**

- persistent RSM ledger와 state transition 구현
- RSM snapshot을 deterministic prompt/token으로 컴파일
- Wan side-info→conditioning 실제 연결
- No-RSM, append-only RSM, revocable RSM 3조건 비교
- exit, occlusion, reappearance, scene cut 분리

**provisional 통과 조건**

1. append-only RSM이 No-RSM보다 identity/temporal consistency를 유의하게 개선
2. append-only RSM에서 측정 가능한 ghost persistence 존재
3. oracle revocation이 ghost AUC를 25% 이상 감소
4. revocable RSM이 append-only RSM identity 이득의 90% 이상 유지
5. false suppression과 quality margin 통과

**실패 시**

- 1 실패: RSM 주기여 제거
- 2 실패: RSM 주기여 제거 또는 사전 등록한 다른 stateful decoder에서 1회 재검증
- 3~5 실패: RSM-adapter 진입 금지, Reduced Track 검토

### G4. Packet schema와 protocol property

**작업**

- schema/version/ontology/ID 계약 동결
- serializer/deserializer와 exact bit-length 함수 구현
- checksum과 conservative discard
- state snapshot/delta 구현
- reorder/duplication/stale packet property test

**통과 조건**

- 모든 packet round-trip bit-exact
- serializer 길이와 accounting byte 일치
- idempotence/version/epoch/reorder property test 통과
- corruption packet이 negative suppression을 실행하지 않음
- lost REVOKE test는 미보장 상태를 명시적으로 기록

### G5. Oracle joint allocator

**작업**

- G4 actual bit cost 사용
- 작은 후보 집합 exhaustive/ILP upper bound
- action interaction과 dependency 분석
- 최소 4개 rate budget에서 positive-only와 Pareto 비교

**provisional 통과 조건**

- 적어도 일부 실용 budget에서 negative action이 선택됨
- positive-only frontier보다 primary hallucination 지표 개선
- safety/quality/latency constraints 충족
- padding만으로 만들어진 rate equality가 아님

**실패 시**

- joint allocation 주장을 제거하고 negative packet의 고정 budget 연구로 축소
- Transactions 목표 재검토

### G6. Binary transport와 disorder/loss

**작업**

- existing `transmission/` bundle에 negative packet 통합
- packet loss, duplication, reorder simulation
- repetition, piggyback, snapshot 비교
- 모든 overhead를 exact rate에 포함

**통과 조건**

- loss 0 조건에서 G4/G5 결과 재현
- reorder/duplication에서 anti-resurrection property 유지
- loss sweep에서 보호 방법의 rate–ghost Pareto 확보
- failure case와 lost-REVOKE 한계를 결과에 포함

### G7. Online heuristic allocator

**작업**

- confirmed-absent calibration
- decoder rollout 없이 계산 가능한 risk proxy
- dependency-aware greedy와 fixed-ratio 비교

**통과 조건**

- compressed method가 oracle 개선량의 70% 이상 유지
- matched-rate positive-only보다 개선
- selector overhead를 포함한 latency constraint 통과

### G8. RSM-adapter

**작업**

- entity/assertion token encoder
- frozen decoder에 작은 conditioning adapter 연결
- RSM-lite와 matched-rate/compute 비교

**포함 조건**

- RSM-lite 대비 primary endpoint, false suppression, latency 또는 prompt/token overhead 중
  사전 정의한 한 축에서 material improvement
- 다른 핵심 축의 non-inferiority 유지

효과가 없으면 adapter를 최종 방법에서 제거하고 RSM-lite를 사용한다. 개발했다는 이유만으로
논문 모델에 포함하지 않는다.

### G9. Learned risk predictor

**작업**

- offline decoder rollout으로 target marginal utility 생성
- $q_\phi(c\mid S_t^+,M_{t-1},D_{id})$ 학습
- calibration, cross-decoder drift, inference cost 평가

**통과 조건**

- held-out validation에서 heuristic보다 utility ranking/calibration 개선
- end-to-end frontier 개선 또는 같은 성능에서 계산량 감소
- train video/concept leakage 없음

효과가 없으면 heuristic을 최종 deployable 방법으로 유지한다.

### G10. 최종 matched-rate·compute 평가

**작업**

- 모든 필수 baseline과 ablation
- 최소 3 seeds
- video-level paired hierarchical bootstrap
- p50/p95 latency, VRAM, failure count
- official-GT-anchored independent automatic evaluator

**통과 조건**

- Full Track이면 H1~H6 모두 통과
- exact total rate와 compute 비교 완전성 검증
- 최종 method row의 checksum/manifest 확보

### G11. Held-out와 독립 backbone

**작업**

- 동결된 config로 held-out test 한 번 실행
- 다른 generative backbone에서 효과 방향 확인
- threshold와 model 선택 재튜닝 금지

**최종 통과 조건**

- `H_add`와 ghost AUC 모두 개선 방향
- primary claim의 paired CI가 사전 기준 충족
- false suppression과 pixel/semantic quality non-inferiority
- compressed online method가 oracle gain retention 기준 충족
- 독립 backbone에서 최소 효과 방향 유지

실패한 결과를 숨기지 않고 test failure로 보고한다. held-out 실패 후 같은 test를 dev로
재분류해 재튜닝하지 않는다.

## 12. 구현 파일 계획

아래 경로는 제안이며 실제 구현 전 이름 충돌을 확인한다.

```text
src/sgdjscc_lab/negative_semantics/
  __init__.py
  ontology.py                 # concept ID와 version
  schema.py                   # ABSENT/ASSERT/STATE/RESET dataclass
  rsm.py                      # state machine과 invariant
  packet_codec.py             # binary encode/decode, checksum
  prompt_compiler.py          # RSM-lite conditioning
  candidate_selector.py       # heuristic/confirmed-absent
  risk_predictor.py           # 후반 learned predictor

src/sgdjscc_lab/controllers/
  joint_rate_allocator.py     # dependency-aware allocation

src/sgdjscc_lab/models/
  rsm_condition_adapter.py    # 후반 adapter

src/sgdjscc_lab/evaluators/
  negative_semantic_metrics.py
  ghost_survival.py
  reappearance_identity.py

src/sgdjscc_lab/pipelines/
  negative_semantic_eval.py

scripts/
  run_negative_semantic_gate.py
  build_negative_semantic_events.py
  summarize_negative_semantic_results.py

configs/experiments/negative_semantics/
tests/
  test_negative_packet_codec.py
  test_rsm_state_machine.py
  test_joint_rate_allocator.py
  test_negative_semantic_pipeline.py
```

### 12.1 필수 unit/property test

- 모든 packet type encode/decode round-trip
- corrupt checksum reject
- duplicate packet idempotence
- lower version ignore
- STATE-before-ASSERT tombstone
- late ASSERT after REVOKE anti-resurrection
- epoch reset 후 stale packet ignore
- snapshot/delta convergence
- lost REVOKE의 미보장 behavior 명시
- action dependency 위반 allocator reject
- actual serialized length와 rate ledger exact match
- receiver 입력에 GT target frame이 없다는 Rx-legal test
- resume와 seed에 따른 결과·manifest 계약

### 12.2 Config 원칙

- 모든 신규 기능 기본값 `off`
- 기존 SGD-JSCC forward 수치 불변
- oracle, heuristic, learned mode를 config에서 분리
- `metric_role: loop_internal | held_out` 유지
- schema/ontology/model ID를 resolved config와 manifest에 기록

## 13. 실행 순서와 예상 일정

일정은 gate 통과를 전제로 한 상대 주차이며, 실패한 gate 이후 작업은 수행하지 않는다.

| 주차 | 단계 | 핵심 산출물 |
|---:|---|---|
| 1~2 | G0 | split·ontology·metric·gate 동결 |
| 3~4 | G1 | P1 phenomenon report |
| 5 | G2 | Oracle ABSENT 판정 |
| 6~8 | G3 | RSM-lite, P2, Oracle REVOKE |
| 9~10 | G4 | binary schema, bit-cost, protocol tests |
| 11~12 | G5 | Oracle allocator와 Pareto |
| 13~14 | G6 | transport와 loss/reorder |
| 15~16 | G7 | heuristic online method |
| 17~19 | G8 | RSM-adapter와 ablation |
| 20~21 | G9 | learned predictor |
| 22~24 | G10 | full matched-rate·compute 평가 |
| 25~27 | G11 | held-out·독립 backbone·human 평가 |
| 28~30 | 논문 | 표·그림·원고·artifact audit |

Wan 14B worker는 메모리 요구량이 크므로 여러 Wan 프로세스를 같은 GPU에 동시에 올리지
않는다. 작은 pilot로 gate를 닫은 뒤 full run을 수행하고, 실패 시 이미 생성된 artifact를
결과 문서에 보존한다.

## 14. 결과 파일과 재현성 계약

각 gate run은 최소 다음을 저장한다.

```text
resolved_config.yaml
run_manifest.json
dataset_split_manifest.json
ontology_manifest.json
packet_schema.json
per_frame.csv
per_event.csv
per_video.csv
paired_effect.csv
bootstrap_ci.json
latency_summary.json
rate_breakdown.csv
failure_cases.json
artifact_sha256.json
REPORT.md
```

manifest 필수 항목은 다음과 같다.

- git commit과 dirty state
- checkpoint/model ID와 hash
- decoder profile ID
- dataset split와 annotation version
- ontology/schema version
- seed
- GPU/device/dtype
- exact command
- package/driver environment
- primary metric role

결과는 [results registry protocol](../protocols/results_registry.md)과
[reproducibility protocol](../protocols/reproducibility.md)을 따른다.

## 15. 논문 표·그림 계획

### 15.1 필수 그림

1. Tx positive/negative candidate → joint allocator → packet → RSM → decoder 전체 구조
2. versioned assertion state transition과 anti-resurrection 예시
3. No-RSM / append-only RSM / revocable RSM 시간축 사례
4. rate–hallucination–false-suppression Pareto frontier
5. ghost survival curve
6. qualitative exit/occlusion/reappearance reconstruction

### 15.2 필수 표

1. 선행 연구 대비 contribution matrix
2. P1/P2와 Oracle gate 결과
3. matched-rate primary comparison
4. RSM ablation
5. packet/protection ablation
6. allocator/oracle gap
7. held-out와 independent backbone
8. latency·VRAM·failure count

### 15.3 Claim–evidence 계약

| 최종 주장 | 반드시 필요한 근거 |
|---|---|
| negative source가 유효하다 | Oracle ABSENT + compressed online matched-rate 결과 |
| RSM이 필요하다 | No-RSM 대비 identity/temporal 이득 |
| revocation이 유효하다 | append-only 대비 ghost 감소와 identity 유지 |
| joint allocation이 유효하다 | 각 budget의 재최적화된 positive-only frontier 대비 우위 |
| protocol이 reorder-safe다 | property test와 disorder experiment |
| loss-resilient하다 | 보호 overhead 포함 loss–ghost Pareto, 보장 한계 명시 |
| 일반화된다 | held-out video와 independent backbone |

근거가 없는 claim은 conclusion에서 제거한다.

## 16. 위험과 대응

| 위험 | 조기 신호 | 대응 |
|---|---|---|
| Oracle negative도 decoder에 안 통함 | G2 효과 없음 | 즉시 Stop Track |
| RSM이 positive benefit 없이 ghost만 만듦 | G3 H2 실패 | RSM 기여 제거 |
| detector가 absence를 잘못 판정 | false suppression 증가 | 3-state absence, calibration, independent evaluator |
| action utility가 비가산적 | DP 예측과 exhaustive 차이 큼 | ILP/conditional greedy, interaction ablation |
| negative bits보다 keyframe이 항상 우수 | G5에서 negative 미선택 | joint allocation 주장 제거 |
| REVOKE loss가 catastrophic | loss sweep 급격한 ghost 증가 | repetition/piggyback/snapshot과 conservative decoder |
| adapter가 backbone-specific | 두 번째 decoder 실패 | protocol generalization만 주장 |
| compute overhead가 품질 이득보다 큼 | p95 latency 초과 | heuristic/RSM-lite 유지, learned module 제거 |
| ETRI 10영상 과적합 | held-out effect 소실 | test 실패 보고, 재튜닝 금지 |
| 라이선스/재배포 문제 | dataset/model audit 실패 | G0에서 대체 자산 선택 |

## 17. 최종 제출 준비 조건

다음 항목이 모두 충족돼야 TMM/TCSVT regular paper 제출 준비 완료로 판정한다.

- Full Track H1~H6 통과
- 최소 50개 독립 held-out 영상 또는 G0에서 승인한 동등 규모
- 최소 100개 검증 exit/occlusion/reappearance event 또는 승인된 동등 규모
- 최소 3 generation seeds
- official-GT-anchored independent automatic evaluator와 frozen threshold
- exact matched-rate 및 rate breakdown
- matched compute와 실제 p50/p95 latency
- RSM/packet/allocator 필수 ablation
- 독립 generative backbone 최소 1개
- 모든 primary result의 video-level paired CI
- code/config/checkpoint/data license audit
- 표·그림 자동 생성 script
- 결과 registry와 SHA-256 manifest
- 실패 case와 limitation 공개

Full Track이 아닌 경우 제출처와 제목을 다시 결정한다. 일정이 임박했다는 이유로 gate를
완화하거나 held-out을 생략하지 않는다.

## 18. 최종 한 문장

이 연구의 핵심은 negative prompt를 추가하는 것이 아니라, **어떤 positive·negative
assertion과 revocation 보호에 제한된 전송 bit를 사용할지 공동 결정하고, 수신된 versioned
assertion이 generative receiver memory의 유효 evidence를 안전하게 갱신하도록 만드는 것**이다.

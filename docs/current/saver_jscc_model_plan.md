---
status: stopped_under_frozen_plan
updated: 2026-09-09
owner: ETRI SGD-JSCC 연구팀
source_commit: fd2426a
implementation_status: PROTOTYPE_IMPLEMENTED_UNTRAINED
training_status: NOT_STARTED_STOP_TRACK
formal_evidence_status: NOT_CONFIRMATORY__SV0_G2_NOT_PASSED
sv0_gate: NOT_PASSED
track_decision: STOP_UNDER_FROZEN_PLAN
primary_venue: IEEE Transactions on Multimedia
stretch_venue: IEEE Transactions on Wireless Communications
supporting_plan: docs/current/negative_semantic_paper_plan.md
supersedes:
---

> [← 문서 색인](../README.md) · [현재 상태](./status.md) ·
> [로드맵](./roadmap.md) · [Negative-semantics 선행 게이트](./negative_semantic_paper_plan.md)

# SAVER-JSCC 모델 개발 단일 기준

## 0. 문서 역할과 현재 판정

이 문서는 다음 제안 모델의 구조, 구현 순서, 학습 계약, 실험 gate와 논문 claim
경계를 정의하는 **단일 설계·실행 기준**이다.

> **SAVER-JSCC: Signed Assertions and Versioned Entity Memory for
> Revocation-Aware Generative Video Joint Source–Channel Coding**

현재 판정은 다음과 같다.

- SAVER-JSCC 핵심 module, tensor pipeline, signed packet/fault channel, dataset/loss,
  stage runner와 checkpoint 계약은 **prototype으로 구현됐지만 학습되지 않았다**.
- SV0/G2 Oracle ABSENT 40-video Pilot는 clean commit `6f9593d`에서 320/320, 실패
  0으로 완료됐다. runner audit은 `PASSED`지만 Oracle H_add 감소 0%로 provisional
  mechanism gate는 **`NOT_PASSED`**다.
- prompt-only RSM은 receiver-applied ledger에서만 condition을 만들고 Wan의
  `prompt`/`negative_prompt`로 연결한다. SM-DiT는 generic Transformer block bridge까지
  구현됐지만 실제 Wan/MDTv2 가중치에 삽입·학습된 상태는 아니다.
- G3 Oracle REVOKE의 세 memory arm manifest, frame-aligned Wan injection, snapshot
  전환 GOP split, ghost/identity evaluator와 audit는 구현됐다. G2 미통과로 draft와
  `execution_authorized=false`를 유지하며 GPU 실행·성능 근거는 없다.
- 기존 negative-semantics G0/G1, int4/int6 bridge는 문제 설정과 운용점의 선행
  근거이며 SAVER의 성능 증거가 아니다.
- G2는 prompt-only Oracle controllability의 negative result이며 학습된 SAVER 모델의
  성능 결과가 아니다. 동결 계획상 Stop Track을 적용하고 formal 학습을 중단한다.
- 기존 `SGDJSCC/` baseline과 게이트-off 경로는 보존한다.
- 각 module은 해당 gate를 통과했을 때만 최종 모델과 논문 contribution에 포함한다.

문서 역할은 다음처럼 분리한다.

| 문서 | 역할 |
|---|---|
| 이 문서 | SAVER 모델 구조·학습·gate·claim의 기준 |
| [negative_semantic_paper_plan.md](./negative_semantic_paper_plan.md) | Oracle ABSENT/REVOKE, packet, RSM 및 데이터 gate의 선행 실험 계획 |
| [status.md](./status.md) | 실제 구현·학습·검증 상태만 기록 |
| [roadmap.md](./roadmap.md) | 다음 수행 작업과 순서 |
| `docs/experiments/YYYY-MM-DD_*.md` | 완료된 실행의 불변 결과 기록 |

## 1. 연구 가설과 주장 경계

### 1.1 핵심 연구 질문

> 제한된 전송 예산에서 positive semantics뿐 아니라 confirmed absence와 revocation을
> 학습 가능한 signed state operation으로 부호화하고, 이를 versioned receiver neural
> memory와 diffusion block 내부에 주입하면 positive-only 또는 append-only memory보다
> hallucination과 ghost persistence를 낮추면서 entity identity consistency를 유지할
> 수 있는가?

### 1.2 의도한 구조적 기여

1. **Signed Assertion Tokenizer(SAT)**
   - 현재 GOP와 이전 semantic state의 차이를 entity slot 단위로 표현한다.
   - `present`, `confirmed-absent`, `unknown`을 분리한다.
2. **Joint Assertion-Symbol Router(JASR)**
   - positive visual source, signed assertion과 protection action이 하나의 예산에서
     경쟁하도록 한다.
3. **Versioned Revocable Entity Memory(VREM)**
   - 장기 identity와 현재 render eligibility를 분리하고 versioned state transition을
     적용한다.
4. **Signed-Memory Diffusion Transformer(SM-DiT)**
   - active memory와 absent/revoked memory를 diffusion block 내부의 비대칭 연산으로
     사용한다.

단순 negative prompt, 외부 rule-based gate, 일반 cross-attention 하나, PSSS threshold
변경 또는 bit-depth 변경은 SAVER의 핵심 contribution으로 주장하지 않는다.

### 1.3 논문에서 허용되는 주장

다음 주장은 각각 대응 gate가 통과한 뒤에만 허용한다.

- signed assertion이 matched-rate positive-only 대비 source-paired hallucination을 줄인다.
- revocable dual-bank memory가 append-only memory의 identity 이득을 유지하면서 ghost
  persistence를 줄인다.
- SM-DiT가 prompt/token-only conditioning보다 matched-compute에서 우수하다.
- JASR가 fixed allocation보다 rate–semantic-reliability Pareto frontier를 개선한다.
- 두 dataset과 두 generative backbone에서 같은 방향이 재현될 때만 일반화 가능성을
  주장한다.

다음은 검증 전 금지한다.

- 세계 최초, SOTA, model-agnostic, real-time, neural deletion, guaranteed revocation
- 실제 변조/FEC가 없는데 physical-layer JSCC 또는 TWC 수준의 channel 최적화 주장
- 개발·Pilot·smoke 결과를 held-out 또는 final evidence로 표현

## 2. 최종 모델 구조

```text
Transmitter
  source GOP X_t ─┬─> existing visual/keyframe encoder ──────────────┐
                  └─> SAT(X_t, Q^tx_{t-1})                           │
                        └─> signed entity/event tokens               │
  CSI + budget ───────────────────────────────> JASR                 │
                                                   ├─ visual action ─┤
                                                   └─ state action   │
                                                        │            │
                                         action-conditioned codec    │
                                                        └────────────┤
                                                                     ▼
                                                        wireless channel
                                                                     │
Receiver                                                             ▼
  decoded visual latent ──────────────────────────────────────────────┐
  decoded state operation ─> version check ─> VREM                    │
                                            ├─ active bank ──────────┤
                                            └─ absent/revoked bank ──┤
  received-signal/reliability ─> channel-token projector ─────────────┤
                                                                     ▼
                                                            SM-DiT decoder
                                                                     │
                                                                     ▼
                                                            reconstructed GOP
```

### 2.1 기준 tensor 계약

초기값은 구현을 시작하기 위한 기준이며 pilot에서만 조정한다.

| 이름 | 기준 shape | 의미 |
|---|---|---|
| `F_t` | `[B,L,d]` | GOP의 flattened VAE/video feature |
| `Q_t` | `[B,K,d_m]` | entity slot, 초기 `K=16`, `d_m=512` |
| `p_state` | `[B,K,3]` | present/confirmed-absent/unknown |
| `p_action` | `[B,K,6]` | skip/assert/update/suspend/resume/revoke |
| `C_ch` | `[B,T,d_m]` | current `[B,T,8]` channel token의 learned projection |
| `M_id` | `[B,K,d_m]` | long-lived identity bank |
| `M_render` | `[B,K,d_m]` | active render-state bank |
| `M_negative` | `[B,K,d_m]` | absent/revoked evidence bank |

배치 내 entity 수가 `K`보다 작으면 masked slot을 사용한다. `K`보다 큰 경우의 eviction은
confidence만으로 정하지 않고 identity age, visibility와 version state를 함께 사용한다.

## 3. 모듈 계약

### 3.1 Signed Assertion Tokenizer

입력은 현재 GOP feature와 송신단의 과거 source state뿐이다. 미래 frame 또는 수신단의
미관측 생성 결과를 사용하지 않는다.

```text
F_t + Q^tx_{t-1}
  → temporal slot binding
  → entity correspondence
  → tri-state head
  → state-difference embedding
  → action candidate tokens
```

- `not detected`를 `confirmed-absent`로 바꾸지 않는다.
- `unknown` slot에는 `REVOKE` action mask를 적용한다.
- 학습 GT는 track/segmentation annotation에서 만들며 자동 detector label은 confidence와
  provenance를 보존한다.
- scene cut은 entity별 대량 revoke가 아니라 별도 `SCENE_RESET` operation으로 처리한다.

### 3.2 Joint Assertion-Symbol Router

action 집합은 다음과 같다.

```text
visual: latent precision, extra keyframe, skip/reuse
state: ASSERT, UPDATE, SUSPEND, RESUME, REVOKE
protection: repetition, piggyback, state snapshot
```

학습에서는 straight-through Gumbel-TopK 또는 동일한 hard-forward/soft-backward
estimator를 사용한다. inference와 formal evaluation에서는 hard action과 실제 rate만
사용한다.

```math
a_k \in \{SKIP, ASSERT, UPDATE, SUSPEND, RESUME, REVOKE\}
```

```math
R_{visual}+R_{positive}+R_{negative}+R_{protection} \le B
```

- positive-only baseline도 같은 예산으로 다시 최적화한다.
- baseline에 padding만 추가해 matched-rate라고 하지 않는다.
- dependency-invalid action 조합은 mask한다.
- learned router가 dependency-aware heuristic보다 개선되지 않으면 최종 구조에서 제거한다.

### 3.3 Action-Conditioned Semantic Channel Codec

초기 구현은 두 profile을 분리한다.

1. `saver_source`
   - 기존 `fixed_int4` binary bundle과 exact byte accounting을 사용한다.
   - 첫 모델·multimedia 논문의 primary profile이다.
2. `saver_wireless`
   - action payload를 learned complex-symbol representation으로 전송한다.
   - digital header, modulation/FEC와 visual symbols를 공통 channel-use 단위로 환산한다.
   - TWC/TCOM 주장은 이 profile의 실제 구현·검증 이후에만 허용한다.

두 profile의 byte와 proxy channel symbol을 하나의 숫자로 더하지 않는다.

### 3.4 Versioned Revocable Entity Memory

```text
Identity bank
  entity identity, appearance reference, long-horizon re-entry information

Render-state bank
  ACTIVE | SUSPENDED | REVOKED, visibility, confidence, version

Negative bank
  confirmed-absent and revoked evidence used by SM-DiT
```

- feature update는 GRU/SSM 또는 gated slot updater로 학습한다.
- `scene_epoch`, `entity_id`, `operation`, `version` 검사는 deterministic하다.
- 현재 version 이하 operation은 무시한다.
- 먼저 도착한 tombstone 이후 늦은 ASSERT가 entity를 재활성화하지 못한다.
- `SUSPEND`는 identity를 삭제하지 않는다.
- corrupt negative packet은 적용하지 않으며 loss를 숨기지 않는다.

### 3.5 Signed-Memory Diffusion Transformer

선택한 diffusion block의 연산 순서를 다음처럼 변경한다.

```math
H_l^0 = AdaLN(H_l,t,C_{ch})
```

```math
H_l^1 = H_l^0 + SelfAttn(H_l^0)
```

```math
P_l = CrossAttn(H_l^1,M_{active}),\qquad
N_l = tanh(CrossAttn(H_l^1,M_{negative}))
```

```math
H_{l+1}=FFN(H_l^1+g_l^+P_l-g_l^-N_l)
```

`g_l^+`, `g_l^-`는 timestep, channel reliability, assertion confidence에서 계산한다.
negative residual은 bounded gate와 zero initialization으로 시작한다. 초기 구현에서는 base
backbone을 freeze하고 전체 block의 약 1/4, 1/2, 3/4 지점에 bottleneck adapter를
삽입한다.

prompt-only RSM은 비교군이며 SM-DiT 구현으로 간주하지 않는다.

## 4. 학습과 gradient 계약

```math
L = L_{diff}+\lambda_sL_{state}+\lambda_mL_{memory}
  +\lambda_gL_{ghost}+\lambda_pL_{preserve}
  +\lambda_rL_{rate}+\lambda_cL_{channel}
```

| loss | 역할 |
|---|---|
| `L_diff` | diffusion noise/velocity prediction |
| `L_state` | tri-state와 operation 분류 |
| `L_memory` | track identity contrastive consistency와 state transition |
| `L_ghost` | absent/revoked entity의 재생성 억제 |
| `L_preserve` | 실제 present entity의 false suppression 방지 |
| `L_rate` | budget 초과, entropy 또는 channel-use 비용 |
| `L_channel` | AWGN/fading/loss 조건의 semantic reconstruction |

필수 gradient path는 다음과 같다.

```text
L_diff/L_ghost/L_preserve
  → SM-DiT
  → VREM learned updater
  → semantic decoder
  → differentiable channel
  → semantic encoder
  → SAT

L_rate
  → JASR
  → action/precision/protection selection
```

hard version/tombstone 검사는 invalid state transition만 차단한다. 이 gate만 구현하고
neural memory라고 부르지 않는다. 학습에 사용한 detector/critic과 최종 보고 evaluator는
분리하고 최종 evaluator의 gradient는 모델에 연결하지 않는다.

## 5. 구현 위치와 호환성

아래 SAVER 핵심 모델 경로는 현재 prototype으로 존재한다.

```text
src/sgdjscc_lab/models/saver/
  contracts.py
  signed_assertion_tokenizer.py
  joint_assertion_symbol_router.py
  semantic_channel_codec.py
  versioned_entity_memory.py
  signed_memory_dit.py
  backbone_bridge.py
  rsm_conditioning.py

src/sgdjscc_lab/pipelines/saver_video_pipeline.py
src/sgdjscc_lab/data/saver_states.py
src/sgdjscc_lab/data/saver_dataset.py
src/sgdjscc_lab/transmission/saver_packet.py
src/sgdjscc_lab/transmission/saver_channel.py
src/sgdjscc_lab/training/saver_losses.py
src/sgdjscc_lab/training/saver_stage_runner.py
src/sgdjscc_lab/guidance/saver_receiver_state.py
src/sgdjscc_lab/evaluators/negative_semantics_g3.py
src/sgdjscc_lab/video/receiver_state_conditioning.py
scripts/train_saver_jscc.py
scripts/prepare_negative_semantics_g3.py
scripts/evaluate_negative_semantics_g3.py
scripts/audit_negative_semantics_g3.py
configs/experiments/saver_jscc/
configs/experiments/negative_semantics/g3_oracle_revoke_protocol.yaml
```

구현 판정은 `PROTOTYPE_IMPLEMENTED_UNTRAINED`이다. CPU test는 state/action mask,
version/epoch monotonicity, stale ASSERT 차단, exact byte, loss/reorder/duplicate/corruption,
budget projection, gradient, checkpoint mismatch와 baseline import isolation을 검사한다.
실제 데이터 materialization, 학습 checkpoint, real-backbone GPU 연결과 formal 성능은
아직 없다.

SV0에 한해 다음 실행 경로가 구현되어 있다.

```text
src/sgdjscc_lab/guidance/negative_conditioning.py
src/sgdjscc_lab/evaluators/negative_semantics_g2.py
scripts/run_negative_semantics_g2.py
scripts/audit_negative_semantics_g2.py
configs/experiments/negative_semantics/g2_oracle_absent_protocol.yaml
```

이는 frozen baseline diffusion에 receiver-side negative text를 주입하는 Oracle
controllability 시험 경로다. SAVER의 SAT/VREM/SM-DiT 구현이나 rate-bearing negative
packet으로 계산하지 않는다.

G3 Oracle REVOKE 준비·평가 경로도 구현되어 있다. Oracle state는 packet 밖의
`ORACLE_EVAL_ONLY` 입력이며 rate에 포함하지 않는다. G2 미통과로 protocol은 draft이고
prepare 단계가 `execution_authorized=false`를 기록하므로 formal 실행 근거가 아니다.

호환성 불변조건:

- SAVER 전용 config에서만 `use_saver_jscc: true`를 명시하며 production default에는
  해당 key를 추가하지 않는다.
- gate off에서 기존 SGD-JSCC image/video 결과를 변경하지 않는다.
- checkpoint에는 architecture version, `K`, `d_m`, injection layers, action vocabulary와
  packet schema fingerprint를 저장한다.
- SAVER checkpoint를 기존 baseline checkpoint로 조용히 로드하지 않는다.
- prototype config를 production default나 baseline recipe에 합성하지 않는다.

## 6. 단계와 go/no-go gate

### SV0. Oracle signed-control feasibility

- 결과 상태: **`VALIDATED_NOT_PASSED`** — 40-video·4-arm·2-policy 320/320 완료,
  runner audit `PASSED`, provisional mechanism gate `NOT_PASSED`
- negative-semantics G2 Oracle ABSENT와 G3 Oracle REVOKE를 재사용한다.
- G3 harness는 구현됐지만 G2 미통과로 실행하지 않는다.
- no/random/frequency/oracle condition을 matched generation compute로 비교한다.
- G1 effective-seed가 `NOT_PASSED`인 동안 결과는 mechanism feasibility다.

제안 통과 기준:

- source-paired `H_add` 상대 감소 25% 이상
- video-paired one-sided 95% CI가 사전 정의한 방향을 지지
- false suppression 증가 CI 상한 `2%` 이내
- PSNR/SSIM/LPIPS non-inferiority budget 통과

실측 결과 primary few10의 Oracle source-paired H_add는 no-negative와 동일한
1.0493%(156/14,867)로 상대 감소 0%였다. false suppression·품질 조건은 통과했지만
효능 조건과 개선 CI는 실패했다. 따라서 구현된 prototype의 formal 학습·최종 모델
채택을 중단하고 Stop Track으로 기록한다. 상세는
[G2 결과](../experiments/2026-09-09_negative_semantics_g2_oracle_absent_pilot_results.md)를
따른다.

### SV1. SAT와 VREM

비교:

1. No memory
2. Append-only single-bank memory
3. Versioned dual-bank memory

Append-only가 No-memory 대비 identity/temporal consistency를 개선하지 않으면 memory
contribution은 성립하지 않는다. VREM은 append-only identity 이득의 90% 이상을
유지하면서 ghost survival을 상대 25% 이상 줄이는 것을 provisional 목표로 한다.

### SV2. SM-DiT

비교:

- deterministic prompt-only RSM
- token-only 일반 cross-attention
- active memory only
- active+negative memory, signed residual 없음
- full SM-DiT

matched rate, trainable parameter 수와 sampling step에서 full SM-DiT의 paired CI가
prompt/token-only 대비 개선 방향을 지지해야 한다.

### SV3. JASR와 channel codec

최소 네 개 budget에서 다음을 비교한다.

- positive-only re-optimized
- fixed positive/negative ratio
- dependency-aware heuristic
- learned JASR
- oracle allocator

learned JASR가 heuristic보다 rate–quality–hallucination Pareto frontier를 개선하지 않으면
learned router claim을 제거한다.

### SV4. Channel and packet robustness

- AWGN과 Rayleigh SNR subset
- packet loss `0/1/5/10%`
- reorder window `0/1/3 GOP`
- duplicate, corrupt packet, stale ASSERT after REVOKE
- no protection, repetition, piggyback, snapshot

stale packet의 재활성화 0건은 protocol invariant이지 품질 개선 claim이 아니다.

### SV5. Held-out and backbone generalization

- Development와 Validation threshold를 분리한다.
- DAVIS Held-out seal은 모든 architecture/hyperparameter 선택 후에만 연다.
- primary backbone 외에 실제로 다른 generative backbone 하나에서 재검증한다.
- adapter를 이식하지 않은 backbone 결과로 model-agnostic을 주장하지 않는다.
- stochastic model은 최소 3개의 실제 effective training/generation seed를 사용한다.

## 7. 필수 비교군과 ablation

### 7.1 전체 시스템 비교군

- original SGD-JSCC
- 현재 LGVSC-inspired fixed/SKEM 계열
- positive-only matched-rate
- H.264/H.265/AV1, 가능하면 VVC
- 동일 rate 정의로 실행 가능한 WITT/DeepJSCC/DiffJSCC 계열

### 7.2 SAVER 구조 ablation

| 조건 | SAT | VREM | signed DiT | learned router |
|---|---:|---:|---:|---:|
| positive-only | off | off | off | off |
| prompt-only negative | fixed | off | off | off |
| append-only memory | on | single | off | off |
| revocable memory | on | dual | off | off |
| regular memory attention | on | dual | regular | off |
| fixed-router SAVER | on | dual | signed | off |
| full SAVER-JSCC | on | dual | signed | on |

추가 ablation:

- channel token 사용/미사용
- negative gate bounded/unbounded
- injection layer 위치와 개수
- identity/render bank 결합/분리
- fixed int4/int6 sensitivity는 primary claim과 분리

## 8. 평가와 통계

필수 평가 축:

```text
rate
  total_on_wire_bytes, effective_bits_per_frame, complex_channel_uses,
  feedback/retransmission/protection/padding

quality
  PSNR, SSIM, LPIPS, DISTS, FVD or equivalent video perceptual metric

semantic and temporal
  source-paired H_add, false suppression, ghost survival AUC,
  object identity consistency, PTC, SFR, SDI

cost
  trainable parameters, FLOPs where available, VRAM, p50/p95 latency,
  diffusion steps, retry count
```

- 단일 합성 점수만으로 우위를 판정하지 않는다.
- video/event clustered paired bootstrap 95% CI를 보고한다.
- selector/training에 사용한 evaluator와 final evaluator를 분리한다.
- exact bytes와 proxy symbols를 합산하지 않는다.
- 실험 protocol과 threshold는 held-out을 열기 전에 동결한다.

## 9. 구현·문서 진행 순서

구조 prototype은 병렬 준비를 위해 먼저 구현됐지만 SV0가 실패했으므로 원래 실행 사슬은
중단한다.

1. 완료된 SV0/G2 negative result와 최소 재현 artifact를 보존한다.
2. G3 GPU matrix, source-only SAVER formal 학습, SV1~SV5 실행은 중단한다.
3. 연구를 계속하려면 이번 Pilot을 seen data로 선언하고 prompt-only control 대신
   diffusion block 내부의 구조적 receiver-state injection을 별도 versioned 설계로 만든다.
4. 새 Development protocol과 gate를 동결하고 독립 output root에서 새 SV0를 수행한다.
5. 새 SV0가 통과한 경우에만 SAT/VREM·SM-DiT 학습과 후속 JASR/codec 실험을 재개한다.
6. 구조·threshold를 다시 동결하기 전에는 Validation과 Held-out을 열지 않는다.

기존 완료 실험 문서를 새 해석에 맞춰 덮어쓰지 않는다. 상태 변화는 `status.md`, 다음
작업은 `roadmap.md`, 불변 실행 결과는 날짜 기반 실험 문서에 각각 기록한다.

## 10. 출판 판단

- **TMM/TCSVT track**: `saver_source` profile, signed representation, VREM, SM-DiT와
  video semantic reliability를 중심으로 한다.
- **TWC/TCOM track**: `saver_wireless` profile, fading/interference, 실제 complex channel
  uses, learned channel codec와 power/bandwidth allocation이 중심이어야 한다.
- SV0–SV2까지만 통과하면 memory-conditioned generation 논문으로 범위를 줄인다.
- SV3까지 통과해야 joint rate allocation을 main contribution에 포함한다.
- SV4 wireless profile이 없으면 제목과 초록에서 physical-layer 최적화 주장을 제거한다.

## 관련 문서

- [negative_semantic_paper_plan.md](./negative_semantic_paper_plan.md) — 선행 데이터·Oracle·packet/RSM gate
- [roadmap.md](./roadmap.md) — 실제 다음 작업
- [status.md](./status.md) — 구현·학습·검증 상태
- [open_issues.md](./open_issues.md) — 미해결 위험
- [../architecture/system.md](../architecture/system.md) — 장기 시스템 구조
- [../architecture/tx_rx_contract.md](../architecture/tx_rx_contract.md) — 송수신 불변 계약
- [../architecture/metrics.md](../architecture/metrics.md) — 지표 정의
- [../protocols/training.md](../protocols/training.md) — 학습 실행 계약
- [../protocols/evaluation.md](../protocols/evaluation.md) — 평가 실행 계약

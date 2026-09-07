---
status: implementation_complete_untrained
date: 2026-09-07
source_branch: codex/saver-jscc-implementation
source_commit: c96b538
evidence_scope: cpu_structure_and_wiring_only
formal_model_evidence: NOT_AVAILABLE
---

> [← 문서 색인](../README.md) · [SAVER 단일 기준](../current/saver_jscc_model_plan.md) ·
> [현재 상태](../current/status.md)

# SAVER-JSCC 구조 prototype 구현 기록

## 결론

SAVER-JSCC의 signed representation, joint router, channel codec, versioned memory,
signed-memory diffusion adapter와 학습/packet 기반을 독립 opt-in prototype으로 구현했다.
관련 집중 CPU 구조·회귀 검사는 371개가 통과했고 전체 suite는 1703 passed,
9 skipped였다. 이 기록은 **코드 구현 증거**이며 학습된
모델, GPU 품질, SV0~SV5 gate 통과 또는 논문 성능 증거가 아니다.

장시간 G2 Pilot가 commit `6f9593d`의 clean checkout에서 실행 중이므로 구현은 별도
`codex/saver-jscc-implementation` worktree/branch에서 수행했다. 실행 중인 Pilot의
provenance를 바꾸지 않기 위해 기존 checkout은 수정하지 않았다.

## 구현 단위

| 단위 | 구현 파일 | 검증 경계 |
|---|---|---|
| signed state 계약 | `models/saver/contracts.py`, `data/saver_states.py` | tri-state/action mask, stable slot, missing ≠ absent |
| signed wire state | `transmission/saver_packet.py` | deterministic CRC packet/ACK, exact byte, epoch/version monotonicity |
| fault channel | `transmission/saver_channel.py` | loss/reorder/duplicate/corruption/repetition, stale ASSERT 차단 |
| SAT | `models/saver/signed_assertion_tokenizer.py` | cross-attention slot binding, GRU state, state/action/confidence heads |
| JASR | `models/saver/joint_assertion_symbol_router.py` | action/rate/protection/visual common budget, hard projection·soft gradient |
| action codec | `models/saver/semantic_channel_codec.py` | learned IQ event codec, AWGN/Rayleigh/loss, source/wireless rate 분리 |
| VREM | `models/saver/versioned_entity_memory.py` | identity/render/negative bank, reset/version/tombstone deterministic guard |
| SM-DiT | `models/saver/signed_memory_dit.py`, `backbone_bridge.py` | active/negative 비대칭 attention, bounded subtraction, zero-init, block hook |
| prompt-only RSM | `models/saver/rsm_conditioning.py`, `scripts/lgvsc_generate_worker.py` | receiver ledger만 사용, SAVER schema만 Wan positive/negative prompt로 소비 |
| pipeline | `pipelines/saver_video_pipeline.py`, `models/saver/factory.py` | SAT→JASR→codec→VREM→SM-DiT, explicit state carry, disabled identity path |
| data/training | `data/saver_dataset.py`, `training/saver_losses.py`, `training/saver_stage_runner.py`, `scripts/train_saver_jscc.py` | checksummed source-only sequence, seven losses, stage freeze, resume/fingerprint |

## 안전·호환성 결정

- `PRESENT`, `CONFIRMED_ABSENT`, `UNKNOWN`을 분리하고 `UNKNOWN→REVOKE`를 거부한다.
- 높은 version의 REVOKE 이후 낮은 version ASSERT는 적용하지 않는다.
- `SCENE_RESET`은 epoch와 reset version을 함께 검사해 duplicate/stale reset을 거부한다.
- `saver_source` exact binary byte와 `saver_wireless` actual complex channel use를 합산하지
  않는다.
- receiver condition은 delivered-packet ledger에서만 생성한다. 한 Wan segment에 서로
  다른 snapshot이 들어오면 임의 선택하지 않고 실패한다.
- SAVER package는 lazy import하며 기존 `pipelines`/`training` package export와 production
  config를 변경하지 않는다.

## 논리 커밋

| commit | 내용 |
|---|---|
| `b40c330` | signed-state와 packet/Tx-Rx 계약 |
| `348f8eb` | SAT와 VREM |
| `393768e` | JASR, action codec, SM-DiT adapter |
| `3d427b3` | pipeline/loss/staged training 연결 |
| `7e52117` | provenance-safe dataset/CLI |
| `9395c96` | packet fault/reorder simulator |
| `b5e03f0` | state/action/reset hardening |
| `668036a` | baseline package import isolation |
| `d22a8b2` | receiver conditioning과 backbone bridge |
| `c96b538` | Wan receiver-state prompt 소비 |

## 실행한 검증

```bash
SGDJSCC_ROOT=/home/sangukbae/ETRI/Semantic/SGDJSCC \
PYTHONPATH="$PWD/src:$PWD/scripts" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/sangukbae/anaconda3/envs/ptest/bin/python -m pytest -q \
  tests/test_architecture.py tests/test_config.py \
  tests/test_packet.py tests/test_packet_bundle.py tests/test_wire_packet.py \
  tests/test_channels.py tests/test_channels_phase5.py \
  tests/test_channel_conditioning.py tests/test_train_ops.py \
  tests/test_train_stages.py tests/test_receiver_runtime.py \
  tests/test_lgvsc_generate_worker.py tests/test_saver_*.py
```

집중 결과: `371 passed`. Git-ignored checkpoint/ETRI video를 원본 checkout에서
읽기 전용으로 연결해 `tests/` 전체를 재실행한 결과는 `1703 passed, 9 skipped`였다.
warning 3개는 기존 PyTorch Transformer nested-tensor 경고다.

## 남은 실증 작업

1. 실행 중 G2 Pilot 종료·감사와 SV0 판정
2. 실제 OVIS/YouTube-VOS source-only tensor manifest materialization
3. SAT/VREM SV1 seed별 학습과 No/Append-only/Revocable 비교
4. real frozen Wan/MDTv2 block에 bridge를 붙인 SV2 학습·matched-compute ablation
5. JASR/codec 네 budget Pareto와 packet/channel formal robustness
6. Validation, sealed Held-out와 독립 backbone 재현

따라서 현재 허용되는 표현은 “SAVER-JSCC 구조 prototype과 학습 기반을 구현했다”이다.
“SAVER-JSCC가 hallucination을 줄였다”, “학습 완료”, “A급 논문 수준 성능”은 아직
허용되지 않는다.

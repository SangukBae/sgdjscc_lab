---
status: active
updated: 2026-09-08
owner: ETRI SGD-JSCC 연구팀
source_commit: 5520b90
supersedes: docs/training_scaffold.md, docs/dev/smoke_training.md
---

> [← 문서 색인](../README.md)

# 학습 지침

- 실행 도구: `scripts/train.py`
- 범위
  - 논문 3-stage 학습
  - 보조·확장 실험
  - SAVER-JSCC gate 통과 후의 단계별 학습 계약
- 비영향 범위
  - 추론
  - 평가

## Stage 구성

| 구분 | stage | 학습 대상 | 입력 |
|---|---|---|---|
| baseline | `jscc` | JSCC encoder/decoder | 이미지 |
| baseline | `text_dm` | text diffusion backbone | 이미지+캡션 |
| baseline | `controlnet` | ControlNet branch | 이미지+캡션+edge |
| supporting | `edge_codec` | edge 전송 codec | 이미지/edge |
| supporting | `csi_estimation` | SNR estimator | 이미지 |
| extension | `end_to_end_ft` | 선택한 JSCC/DM 모듈 | 이미지+캡션 |

- 권장 순서
  1. `jscc`
  2. `text_dm`
  3. `edge_codec`
  4. `controlnet`
- 확장 실험: baseline 이후 `end_to_end_ft`

## SAVER-JSCC 학습 계약 (`RUNNER IMPLEMENTED; FORMAL TRAINING NOT STARTED`)

모델 구조와 gate는
[saver_jscc_model_plan.md](../current/saver_jscc_model_plan.md)를 따른다. Stage 0은 학습이
없는 Oracle inference이며 `run_negative_semantics_g2.py`로 구현됐다. `sv1`, `sv2`,
`sv3`, `end_to_end` freeze 정책, seven-loss runner, sequence dataloader와 versioned
checkpoint는 구현됐다. 실제 source-only tensor manifest와 학습 결과는 없다.

| 순서 | 계획 stage | 학습 대상 | 시작 조건 |
|---:|---|---|---|
| 0 | `saver_sv0_oracle` | 학습 없음, Oracle condition 실험 — `IMPLEMENTED_UNVALIDATED` | negative-semantics G2 protocol 동결 |
| 1 | `sv1` | SAT + VREM updater/state heads | runner 구현; formal 시작은 SV0 통과 후 |
| 2 | `sv2` | SM-DiT zero-init adapters | runner 구현; formal 시작은 SV1 통과 후 |
| 3 | `sv3` | JASR + semantic codec | runner 구현; formal 시작은 SV2 통과 후 |
| 4 | `end_to_end` | SAT/VREM/SM-DiT/JASR/codec | runner 구현; formal 시작은 구조·rate 단위 동결 후 |

학습 순서를 건너뛰지 않는다.

1. Oracle action으로 decoder/memory 효과를 먼저 확인한다.
2. base SGD-JSCC/video diffusion backbone을 freeze한다.
3. SAT/VREM과 SM-DiT adapter만 학습한다.
4. SM-DiT 효과가 확인된 뒤 router와 channel codec을 연결한다.
5. full-backbone LoRA/unfreeze는 frozen-backbone ablation 이후 별도 승인한다.

필수 gradient audit:

- `L_diff/L_ghost/L_preserve → SM-DiT → VREM → channel codec → SAT`
- `L_rate → JASR → hard-forward/soft-backward action selection`
- frozen backbone parameter gradient는 0 또는 `None`
- held-out evaluator parameter gradient는 항상 `None`
- deterministic version/tombstone check를 neural deletion loss로 표현하지 않음

데이터 역할:

- OVIS/YouTube-VOS Train: training
- 사전 지정 Validation: early stopping·checkpoint selection
- OVIS Pilot/Development: mechanism 확인과 debugging; final selection에 재사용하지 않음
- DAVIS Held-out: architecture·threshold·checkpoint 동결 후 한 번만 개봉

각 stage의 formal 시작 전 필요한 smoke:

1. tensor shape와 action mask test
2. VREM state-transition property test
3. 1~2 step forward/backward에서 finite loss와 expected gradient path 확인
4. checkpoint save/resume 및 architecture fingerprint mismatch fail-closed 확인
5. 같은 seed 재현성과 서로 다른 effective seed 소비 확인

SAVER checkpoint 필수 metadata:

```text
model_family: saver_jscc
architecture_version
stage
K, d_m
action_vocabulary
packet_schema_version
injection_layers
base_checkpoint_sha256
trainable_parameter_names/hash
dataset_split_hash
loss_weights
rate_profile: saver_source | saver_wireless
```

기존 baseline checkpoint와 SAVER checkpoint 사이의 partial load는 명시적 migration
script 없이는 거부한다.

Prototype 실행 형식:

```bash
python scripts/train_saver_jscc.py \
  --config configs/experiments/saver_jscc/model_v1.yaml \
  --manifest /path/to/source_only_saver_tensor_manifest.jsonl \
  --output-dir outputs/checkpoints/saver/sv1/<run_id> \
  --stage sv1 --device cuda:0
```

`<run_id>`는 실제 이름으로 치환한다. loader는 tensor SHA-256, source-only provenance,
receiver/future leakage와 split 권한을 fail-closed 검사한다. 현 시점에는 공식 manifest가
없으므로 위 명령은 인터페이스 예시이지 완료된 formal run이 아니다.

## 기본 실행

```bash
# Stage 1
python scripts/train.py \
    --config configs/recipes/training/composed_train_jscc.yaml \
    --train-list /data/train --val-list /data/val --epochs 20 --device cuda:0

# Stage 2
python scripts/train.py \
    --config configs/recipes/training/composed_train_text_dm.yaml \
    --train-list /data/train --device cuda:0

# Supporting edge codec
python scripts/train.py \
    --config configs/recipes/training/composed_train_edge_codec.yaml \
    --train-list /data/train --val-list /data/val --epochs 50

# Stage 3
python scripts/train.py \
    --config configs/recipes/training/composed_train_controlnet.yaml \
    --train-list /data/train --device cuda:0
```

- 주요 옵션
  - `--stage`: config stage override
  - `--max-steps`, `--epochs`: 종료 조건
  - `--resume latest`: 최신 checkpoint 복원
  - `--no-models`: 모델 없이 설정·배선 검사

## 데이터와 Config

- caption: `sidecar`, `manifest`, `coco_json`, `multi_manifest`, `filename`
- edge: `canny`, `sidecar`, `muge_sidecar`
- 입력: 폴더 또는 file list

- config: `configs/recipes/training/composed_train_*.yaml`
- 데이터 기준: [datasets.md](./datasets.md)
- 사전 실패 조건
  - 잘못된 stage
  - caption·edge 누락
  - 학습 대상 parameter 0개

## Checkpoint와 Export

- 정확한 저장 위치는 config의 `checkpoint_dir`와 `train_log_path`가 결정한다.
  stage별 recipe의 기본 관례는 다음과 같다.

```text
outputs/
├── checkpoints/<stage>/
│   ├── latest.pth
│   └── best.pth
└── <stage>_train_log.jsonl
```

- 범용 `composed_train.yaml`은 각각 `outputs/checkpoints/`와
  `outputs/train_log.jsonl`을 사용한다. `--output-dir`는 `checkpoint_dir`만
  override하며 `train_log_path`를 checkpoint 하위로 자동 이동하지 않는다.

- 추론용 checkpoint로 변환할 때는 export 스크립트를 사용한다.

```bash
python scripts/export_checkpoint.py \
    --stage text_dm \
    --input outputs/checkpoints/text_dm/best.pth \
    --output checkpoints/diffusion_backbone.pth
```

| stage | 추론 파일 |
|---|---|
| `jscc` | `checkpoints/JSCC_model.pth` |
| `text_dm` | `checkpoints/diffusion_backbone.pth` |
| `controlnet` | `checkpoints/diffusion_controlnet.pth` |

- overwrite 규칙: `--force` 필요
- resume 기본 규칙
  - 저장 stage와 실행 stage가 다르면 즉시 실패
  - module key·shape·dtype·type fingerprint와 optimizer/scaler 집합이 모두 일치해야 함
  - stage tag가 없는 legacy checkpoint는 `train.resume_allow_legacy: true`, 부분
    복원은 `train.resume_allow_partial: true`를 명시한 감사용 migration에서만 허용
- checkpoint 구분: [reproducibility.md](./reproducibility.md)

## Multi-GPU

```bash
torchrun --standalone --nproc_per_node=3 scripts/train.py \
    --config configs/recipes/training/composed_train_text_dm.yaml \
    --train-list /data/train --val-list /data/val
```

- batch 규칙
  - `batch_size`: rank별 값
  - global batch: `batch_size × world_size × grad_accum_steps`
- 동기화 규칙
  - `text_dm`·`controlnet`: `DistributedDataParallel` wrapper
  - `jscc`·GAN discriminator·`edge_codec`·`csi_estimation`·`end_to_end_ft`:
    복합 forward가 submodule method를 직접 호출하므로 초기 state broadcast 후
    optimizer-step 경계에서 명시적으로 gradient 평균
- export·평가: 단일 process
- 현재 증거: CPU/Gloo 2-rank smoke. 실제 NCCL 장기학습 완료를 뜻하지 않는다.

## Smoke 검증

- 실제 가중치의 forward/backward와 저장·복원을 1~2 step으로 확인한다.

```bash
python scripts/make_tiny_dataset.py \
    --stage all --out ../data/tiny --n 6 --val 2 --size 128

python scripts/train.py \
    --config configs/recipes/training/composed_train_edge_codec.yaml \
    --train-list ../data/tiny/train --val-list ../data/tiny/val \
    --device cpu --max-steps 2 --log-every-steps 1 --save-every-steps 2
```

- 다른 stage
  - config·device만 교체
- 통과 조건

1. loss가 NaN/Inf 없이 기록된다.
2. 정확한 step에서 종료된다.
3. `latest.pth`와 `best.pth`가 생성된다.
4. resume 시 `global_step`이 복원된다.
5. freeze 로그가 stage 정책과 일치한다.

```bash
python -m pytest tests/test_train_stages.py -q
```

- 논문 대비 옵션·freeze 정책
  - [paper_alignment.md](../reference/paper_alignment.md)
  - [SAVER-JSCC 모델 계획](../current/saver_jscc_model_plan.md)

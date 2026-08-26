---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes: docs/training_scaffold.md, docs/dev/smoke_training.md
---

> [← 문서 색인](../README.md)

# 학습 지침

- 실행 도구: `scripts/train.py`
- 범위
  - 논문 3-stage 학습
  - 보조·확장 실험
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

- 학습 결과는 기본적으로 다음 위치에 저장된다.

```text
outputs/checkpoints/<stage>/
├── latest.pth
├── best.pth
└── train_log.jsonl
```

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
- export·평가: 단일 process

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

---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes: docs/training_scaffold.md, docs/dev/smoke_training.md
---

> [← 문서 색인](../README.md)

# 학습 지침

`scripts/train.py`는 논문 3-stage 학습과 보조 실험을 같은 CLI로 실행한다.
추론·평가 경로에는 영향을 주지 않는다.

## Stage 구성

| 구분 | stage | 학습 대상 | 입력 |
|---|---|---|---|
| baseline | `jscc` | JSCC encoder/decoder | 이미지 |
| baseline | `text_dm` | text diffusion backbone | 이미지+캡션 |
| baseline | `controlnet` | ControlNet branch | 이미지+캡션+edge |
| supporting | `edge_codec` | edge 전송 codec | 이미지/edge |
| supporting | `csi_estimation` | SNR estimator | 이미지 |
| extension | `end_to_end_ft` | 선택한 JSCC/DM 모듈 | 이미지+캡션 |

권장 순서는 `jscc` → `text_dm` → `edge_codec` → `controlnet`이다.
`end_to_end_ft`는 baseline 학습 이후의 확장 실험이다.

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

`--stage`로 config의 stage를 덮어쓰고, `--max-steps` 또는 `--epochs`로 종료
조건을 정한다. `--resume latest`는 최근 checkpoint를 찾고, `--no-models`는 모델을
로드하지 않고 설정과 배선만 검사한다.

## 데이터와 Config

- caption: `sidecar`, `manifest`, `coco_json`, `multi_manifest`, `filename`
- edge: `canny`, `sidecar`, `muge_sidecar`
- 입력: 폴더 또는 file list

stage별 composed config는 `configs/recipes/training/composed_train_*.yaml`에 있다.
데이터 형식과 생성 도구는 [datasets.md](./datasets.md)를 참고한다. 잘못된 stage,
누락된 caption/edge, 학습 대상 0개 설정은 checkpoint 로딩 전에 실패한다.

## Checkpoint와 Export

학습 결과는 기본적으로 다음 위치에 저장된다.

```text
outputs/checkpoints/<stage>/
├── latest.pth
├── best.pth
└── train_log.jsonl
```

추론용 checkpoint로 변환할 때는 export 스크립트를 사용한다.

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

기존 파일은 `--force` 없이는 덮어쓰지 않는다. baseline과 custom checkpoint를
구분하는 방법은 [reproducibility.md](./reproducibility.md)를 따른다.

## Multi-GPU

```bash
torchrun --standalone --nproc_per_node=3 scripts/train.py \
    --config configs/recipes/training/composed_train_text_dm.yaml \
    --train-list /data/train --val-list /data/val
```

`batch_size`는 rank별 값이며 전역 batch는
`batch_size × world_size × grad_accum_steps`다. export와 평가는 단일 프로세스로
실행한다.

## Smoke 검증

실제 가중치의 forward/backward와 저장·복원을 1~2 step으로 확인한다.

```bash
python scripts/make_tiny_dataset.py \
    --stage all --out ../data/tiny --n 6 --val 2 --size 128

python scripts/train.py \
    --config configs/recipes/training/composed_train_edge_codec.yaml \
    --train-list ../data/tiny/train --val-list ../data/tiny/val \
    --device cpu --max-steps 2 --log-every-steps 1 --save-every-steps 2
```

다른 stage는 config와 device만 바꿔 같은 방식으로 확인한다. 다음 조건을 모두
만족하면 배선 검증을 통과한 것으로 본다.

1. loss가 NaN/Inf 없이 기록된다.
2. 정확한 step에서 종료된다.
3. `latest.pth`와 `best.pth`가 생성된다.
4. resume 시 `global_step`이 복원된다.
5. freeze 로그가 stage 정책과 일치한다.

```bash
python -m pytest tests/test_train_stages.py -q
```

논문과 다른 학습 옵션 및 freeze 정책은
[paper_alignment.md](../reference/paper_alignment.md)를 참고한다.

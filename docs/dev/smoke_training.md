> [← 문서 색인](../README.md)

# Real-model smoke 학습

**실모델**로 각 학습 stage가 실제로 도는지를 1~2 optimizer step으로 빠르게 검증한다.
목적은 수렴이 아니라 **배선·gradient·체크포인트 저장/복원** 확인이다. stub 기반 단위
테스트(`tests/test_train_stages.py`)와 달리 실제 가중치·forward/backward를 거친다.

전체 설계는 [training_scaffold.md](../training_scaffold.md) 참조.

## 준비

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab && conda activate ptest

# stage 공유 tiny dataset (이미지 + 캡션 sidecar)
python scripts/make_tiny_dataset.py --stage all --out ../data/tiny --n 6 --val 2 --size 128
```

체크포인트(HuggingFace `murjun/SGDJSCC` → `checkpoints/`):

| stage | 필요 체크포인트 |
|-------|-----------------|
| `edge_codec` | **없음** (self-contained, scratch, CPU 가능 — 가장 먼저 확인) |
| `jscc` | `JSCC_model.pth` |
| `text_dm` | `JSCC_model.pth`, `diffusion_backbone.pth` (+ CLIP 자동 다운로드) |
| `controlnet` | 위 + `diffusion_controlnet.pth` |

tiny dataset: stage 1은 이미지만, stage 2/3은 이미지+`.txt`(edge는 on-the-fly Canny),
edge_codec은 캡션 불필요.

## Stage별 smoke 명령 (`--max-steps`)

각 명령은 1~2 step 후 종료하고 체크포인트를 남긴다.

```bash
# edge_codec (CPU 가능, 가장 먼저)
python scripts/train.py --config configs/composed_train_edge_codec.yaml \
    --train-list ../data/tiny/train/ --val-list ../data/tiny/val/ \
    --device cpu --max-steps 2 --log-every-steps 1 --save-every-steps 2

# jscc / text_dm / controlnet — config만 바꿔 동일 패턴 (device cuda:0)
python scripts/train.py --config configs/composed_train_jscc.yaml \
    --train-list ../data/tiny/train/ --device cuda:0 \
    --max-steps 2 --log-every-steps 1 --save-every-steps 2
```

- `--log-every-steps 1`: 매 step loss 출력(기본 config는 0이라 없으면 안 보임).
- `--save-every-steps 2`: step 체크포인트(이게 있어야 `best_metric`이 실제 loss로 기록).
- batch 크기는 config `train.batch_size`(기본 4).
- **controlnet**: 학습된 edge codec 없이도 동작(무작위 codec + 경고 → ablation급).
  실제 baseline 수치는 `edge_codec` 학습 후. 연결 검증은 stage 3 로그의 `trained_codec=True`.

## 저장 / resume 검증

```bash
# 2 step → latest/best.pth
python scripts/train.py --config configs/composed_train_edge_codec.yaml \
    --train-list ../data/tiny/train/ --device cpu \
    --max-steps 2 --log-every-steps 1 --save-every-steps 2
# resume: global_step 2 → 4까지만 돌고 멈춰야 함
python scripts/train.py --config configs/composed_train_edge_codec.yaml \
    --train-list ../data/tiny/train/ --device cpu \
    --max-steps 4 --log-every-steps 1 --save-every-steps 2 \
    --resume outputs/checkpoints/edge_codec/best.pth
```

로그에 `Resuming from … global_step 2` → `Reached max_steps=4`면 정상.

## 통과 판정

1. 예외 없이 종료 + `Training complete [stage=…]`.
2. step 로그 `loss`(stage별 항: `loss_mse` / `loss_dm_*` / `loss_bce`+`loss_dice` /
   `loss_recon`+`loss_diff`)가 **유한**(NaN/Inf 아님).
   > `best_metric`은 `--save-every-steps`/val 없는 step-mode에선 0.0 sentinel — 실제
   > loss는 step 로그/`train_log.jsonl`로 확인.
3. `Reached max_steps=N`으로 **정확히 N step**에서 정지.
4. `outputs/checkpoints/<stage>/{latest,best}.pth` 생성.
5. resume 시 `global_step` 복원되어 남은 step만 진행.
6. (edge_codec→controlnet) stage 3 로그 `trained_codec=True`.
7. freeze 정책 로그가 기대와 일치(stage 3: `frozen=base_diffusion`).

CPU 단위/통합 테스트도 통과해야 한다: `python -m pytest tests/test_train_stages.py -q`.

## full training 전 남길 short-run 로그

본 학습 전 각 stage에서 수백~수천 step short run을 남겨 두면 재현·리뷰에 유용하다:
`train_log.jsonl`의 loss 하강 추세, 시작 freeze 리포트, 첫 배치 shape,
`eval_edge_codec.py`의 BCE/Dice/IoU/F1, 하드웨어/설정 메타(device·batch·AMP·SNR·seed).
권장 길이 예: jscc/text_dm/controlnet `--max-steps 500`, edge_codec `--epochs 5`.

## 관련 문서
- [../training_scaffold.md](../training_scaffold.md) · [../etri_strategy.md](../etri_strategy.md)
</content>

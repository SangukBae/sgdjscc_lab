> [← docs index](./README.md)

# Real-model smoke training (최소 검증 경로)

이 문서는 **실모델(real-model)** 로 각 학습 stage가 실제로 도는지를 1~2
optimizer step으로 빠르게 검증하는 방법을 정리한다. 목적은 수렴이 아니라
**"배선이 맞는가 / gradient가 흐르는가 / 체크포인트가 저장·복원되는가"** 를
확인하는 것이다. 단위 테스트(`tests/test_train_stages.py`, stub 기반)와 달리, 여기서는
실제 모델 가중치와 실제 forward/backward를 거친다.

전체 학습 프레임워크 설계는 [training_scaffold.md](./training_scaffold.md) 참조.

---

## 0) 공통 준비

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest
```

### 미리 준비할 체크포인트 (stage별)

HuggingFace `murjun/SGDJSCC`에서 받아 `sgdjscc_lab/checkpoints/`에 둔다
(config `model_root: ../checkpoints/`).

| stage | 필요한 체크포인트 | 비고 |
|-------|------------------|------|
| `edge_codec` | **없음** | self-contained, scratch 학습 |
| `jscc` | `JSCC_model.pth` | `use_semantic:false` → JSCC만 로드 |
| `text_dm` | `JSCC_model.pth`, `diffusion_backbone.pth` (+ CLIP ViT-L/14 자동 다운로드) | DM + JSCC VAE 필요 |
| `controlnet` | `JSCC_model.pth`, `diffusion_backbone.pth`, `diffusion_controlnet.pth` | + ControlNet 브랜치 |
| `end_to_end_ft` (extension) | 위 3개 (+ 보통 `--resume`로 stage3 결과) | baseline 아님 |

> `edge_codec`는 체크포인트가 전혀 필요 없어 가장 먼저, 그리고 CPU로도 돌릴 수
> 있는 가장 가벼운 smoke 대상이다.

### Tiny dataset 생성

```bash
# 모든 stage가 공유하는 작은 train/val 트리 (이미지 + 캡션 sidecar)
python scripts/make_tiny_dataset.py --stage all --out ../data/tiny --n 6 --val 2 --size 128
```

생성 결과(디렉터리 구조 / 파일명 규칙):

```
../data/tiny/
├── train/
│   ├── sample_000.png      # 작은 RGB 이미지 (사각형 몇 개 → Canny edge 생성됨)
│   ├── sample_000.txt      # 캡션 sidecar: "a photo of sample 000"
│   ├── sample_001.png
│   ├── sample_001.txt
│   └── …
└── val/
    ├── sample_006.png
    ├── sample_006.txt
    └── …
```

- **stage 1 (jscc)**: 이미지만 사용 (`.txt` 무시).
- **stage 2 (text_dm), 3 (controlnet)**: 이미지 + `.txt` 캡션. edge는 Canny로
  on-the-fly 생성(별도 파일 불필요).
- **edge_codec**: 캡션 불필요. 위 트리를 그대로 써도 되고, 이미지만 있는 폴더면 충분.
  (precomputed edge가 필요하면 `--stage edge_codec --edges`로 `*_edge.png`도 생성.)

---

## 1) Stage별 smoke 명령어 (`--max-steps`)

각 명령은 **1~2 optimizer step** 후 종료하고 체크포인트를 남긴다.

### edge_codec (체크포인트 불필요 — CPU 가능, 가장 먼저 확인)

```bash
python scripts/train.py --config configs/composed_train_edge_codec.yaml \
    --train-list ../data/tiny/train/ --val-list ../data/tiny/val/ \
    --device cpu --max-steps 2 --log-every-steps 1 --save-every-steps 2
```
- `--log-every-steps 1`: **매 step loss를 출력**(smoke에서 loss가 유한한지 확인용).
  기본 config는 `log_every_steps: 0`이라 이 플래그가 없으면 loss 줄이 안 보인다.
- `--save-every-steps 2`: step 단위 체크포인트(이게 있어야 `best_metric`이 실제
  loss로 기록된다 — §3 참고).
- 배치 크기는 config의 `train.batch_size`(기본 4)로 조정한다(별도 CLI 플래그 없음).

### Stage 1 — jscc

```bash
python scripts/train.py --config configs/composed_train_jscc.yaml \
    --train-list ../data/tiny/train/ --device cuda:0 \
    --max-steps 2 --log-every-steps 1 --save-every-steps 2
```

### Stage 2 — text_dm

```bash
python scripts/train.py --config configs/composed_train_text_dm.yaml \
    --train-list ../data/tiny/train/ --device cuda:0 \
    --max-steps 2 --log-every-steps 1 --save-every-steps 2
```

### Stage 3 — controlnet (baseline = edge_jscc transport)

```bash
# smoke에서는 학습된 edge codec이 없어도 동작한다(무작위 codec + 경고). 단,
# 경고가 뜨면 그건 ablation급이라는 뜻 — 실제 baseline 수치는 edge_codec 학습 후.
python scripts/train.py --config configs/composed_train_controlnet.yaml \
    --train-list ../data/tiny/train/ --device cuda:0 \
    --max-steps 2 --log-every-steps 1 --save-every-steps 2
```

### (선택) edge_codec → controlnet 연결까지 한 번에 검증

```bash
# 1) edge codec 2-step 학습 (best.pth 생성)
python scripts/train.py --config configs/composed_train_edge_codec.yaml \
    --train-list ../data/tiny/train/ --device cuda:0 \
    --max-steps 2 --log-every-steps 1 --save-every-steps 2
# 2) stage3가 그 체크포인트를 로드하는지 (로그 trained_codec=True 확인)
python scripts/train.py --config configs/composed_train_controlnet.yaml \
    --train-list ../data/tiny/train/ --device cuda:0 \
    --max-steps 2 --log-every-steps 1 --save-every-steps 2
```

---

## 2) 체크포인트 저장 / 복원(resume) 검증

```bash
# 2 step 학습 → latest.pth/best.pth 생성
python scripts/train.py --config configs/composed_train_edge_codec.yaml \
    --train-list ../data/tiny/train/ --device cpu \
    --max-steps 2 --log-every-steps 1 --save-every-steps 2

# resume: global_step 2에서 이어 4까지만 돌고 멈춰야 함
python scripts/train.py --config configs/composed_train_edge_codec.yaml \
    --train-list ../data/tiny/train/ --device cpu \
    --max-steps 4 --log-every-steps 1 --save-every-steps 2 \
    --resume outputs/checkpoints/edge_codec/best.pth
```

로그에서 `Resuming from epoch … / global_step 2` → `Reached max_steps=4` 가
보이면 resume 정상.

---

## 3) "smoke 통과" 판정 기준

아래가 모두 충족되면 통과로 본다.

1. **예외 없이 종료**하고 마지막에 `Training complete [stage=…]` 가 찍힌다.
2. `--log-every-steps 1`로 출력된 `[step N] loss=…` 의 `loss`(stage별 항:
   `loss_mse` / `loss_dm_*` / `loss_bce`+`loss_dice` / `loss_recon`+`loss_diff`)가
   **유한한 실수**다(NaN/Inf 아님). > 참고: 마지막 줄의 `best_metric`은
   `--save-every-steps`(또는 val)를 켜지 않은 step-mode 실행에서는 실제 loss가
   아니라 **0.0 sentinel**이 찍힌다. 실제 loss는 step 로그 / `train_log.jsonl`로
   확인한다.
3. `Reached max_steps=N — stopping.` 으로 **정확히 N step**에서 멈춘다.
4. 체크포인트가 생성된다: `outputs/checkpoints/<stage>/{latest,best}.pth`.
5. resume 시 `global_step`이 복원되어 남은 step만 추가로 돈다(§2).
6. (edge_codec→controlnet) stage3 로그에 `trained_codec=True`.
7. freeze 정책 로그가 기대와 일치한다(예: stage3 `frozen=base_diffusion`,
   `trainable=en_inblocks_controlnet …`).

CPU 단위·통합 테스트도 함께 통과해야 한다(체크포인트/GPU 불필요):
```bash
python -m pytest tests/test_train_stages.py -q
```

---

## 4) 논문용 full training 전에 남겨둘 short-run 로그

본 학습을 돌리기 전, 각 stage에서 **수백~수천 step**짜리 short run을 한 번 남겨
다음을 캡처해 두면 디버깅·재현·리뷰에 유용하다.

- **`train_log.jsonl`** (stage별 `train_log_path`): `global_step`, `loss`와
  stage별 항, `val_*`, `lr`. → loss가 **하강 추세**인지(최소한 발산하지 않는지)
  확인.
- **시작 freeze 리포트**: `Freeze policy [stage=…]: trainable=… frozen=… params=N`
  한 줄. → 학습 대상이 의도대로인지(특히 stage3 base DM frozen).
- **데이터/배치 형상**: 첫 배치 `keys/shapes`(dry-run 또는 첫 step 로그).
- **edge_codec eval**: `scripts/eval_edge_codec.py`의 BCE/Dice/IoU/F1 한 줄
  (SNR 몇 점 스윕). → edge codec이 무작위 대비 실제로 학습됐는지의 증거.
- **하드웨어/설정 메타**: device, `batch_size`, `grad_accum_steps`,
  `mixed_precision`, SNR, seed. → 재현용.

권장 short-run 길이(예시): jscc/text_dm `--max-steps 500`, controlnet
`--max-steps 500`, edge_codec `--epochs 5`. full training은 이 로그들이
"정상 하강 + 올바른 freeze + 유효한 데이터 형상"을 보여준 뒤에 시작한다.

---

## 관련 문서

- [training_scaffold.md](./training_scaffold.md) — 학습 프레임워크 전체 설계
- [etri_development_roadmap.md](./etri_development_roadmap.md) — Phase별 개발 계획

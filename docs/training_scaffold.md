> [← 문서 색인](./README.md)

# 학습 CLI — Stage-Aware Training

`scripts/train.py`는 SGD-JSCC 논문의 **3-stage 학습 절차**를 구조적으로 재현하는
학습 진입점이다. 추론(`infer_pipeline`)/평가(`eval_pipeline`) 경로와 원본 SGD-JSCC
기준선은 건드리지 않는다. `train.stage`에 따라 dataset/forward/loss/freeze가 분기되며,
epoch·step 두 모드를 모두 지원하는 통합 루프(global-step 기준)를 공유한다.

```
[CORE BASELINE — 논문 3-stage]
  jscc        JSCC 인코더/디코더 (고정 AWGN SNR=10dB)
  text_dm     text-guided latent DM (f0 예측)
  controlnet  edge ControlNet 브랜치 (base DM frozen)

[supporting]  edge_codec       전용 edge JSCC codec을 BCE+Dice로 학습 → stage 3가 로드
[supporting]  csi_estimation   blind SNR 추정기 학습 → 추론 step-matching에 연결
[extension]   end_to_end_ft    3-stage 이후 JSCC↔DM 공동 미세조정 (baseline 아님)
```

| 구분 | stage | baseline 여부 |
|------|-------|---------------|
| core baseline | `jscc`, `text_dm`, `controlnet` | ✅ 이 3개가 baseline |
| supporting | `edge_codec`, `csi_estimation` | baseline의 *부품*을 학습 |
| extension | `end_to_end_ft` | ❌ 확장 실험 |

## 빠른 시작

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab && conda activate ptest

# Stage 1 — JSCC (이미지 전용, 고정 SNR=10dB, MSE[+patch-GAN])
python scripts/train.py --config configs/composed_train_jscc.yaml \
    --train-list /data/imagenet/train/ --val-list /data/imagenet/val/ --epochs 20

# Stage 2 — text-guided DM (sidecar .txt 캡션)
python scripts/train.py --config configs/composed_train_text_dm.yaml \
    --train-list /data/pairs/train/ --device cuda:0

# (supporting) edge codec — stage 3용 codec 선행 학습 (JSCC/DM 불필요, edge만)
python scripts/train.py --config configs/composed_train_edge_codec.yaml \
    --train-list /data/edges/train/ --epochs 50

# Stage 3 — edge ControlNet (BASELINE = edge_jscc transport)
#   train.controlnet.edge_jscc.checkpoint 가 위 edge_codec 결과를 가리켜야 함
python scripts/train.py --config configs/composed_train_controlnet.yaml \
    --train-list /data/pairs/train/ --device cuda:0

# 대규모 step-based / dry-run
python scripts/train.py --config configs/composed_train_text_dm.yaml \
    --train-list /data/pairs/train/ --max-steps 250000
python scripts/train.py --config configs/composed_train_jscc.yaml \
    --train-list /path/imgs/ --no-models --epochs 1   # 배선만 검증(GPU 불필요)
```

`--stage {jscc|text_dm|controlnet|edge_codec|csi_estimation|end_to_end_ft}`로
`train.stage`를, `--max-steps`로 step 모드를 덮어쓴다. 잘못된 stage 설정(누락 caption/edge,
잘못된 transport, 학습 대상 0개 등)은 **체크포인트 로딩 전에** `StageConfigError`로 실패한다.

> **권장 순서**: `jscc` → `text_dm` → `edge_codec` → `controlnet`
> (`edge_codec`는 JSCC/DM과 독립이라 순서 무관). `end_to_end_ft`는 그 뒤의 추가 실험.

## stage ↔ 코드 매핑

| 구분 | stage | Dataset | Forward | Loss | 학습 대상 |
|------|-------|---------|---------|------|-----------|
| baseline 1 | `jscc` | `ImageOnlyDataset` | VAE encode → AWGN(10) → decode | `JSCCStageLoss` = MSE(+λ·GAN, +LPIPS opt) | JSCC (DM/guidance frozen) |
| baseline 2 | `text_dm` | `TextImageDataset` | `f0=VAE(x)`, sigmoid schedule, masked/unmasked 예측 | `DiffusionF0Loss` | base DM (JSCC frozen) |
| baseline 3 | `controlnet` | `TextImageEdgeDataset` | stage2 + edge 조건 `c` | `DiffusionF0Loss` | **ControlNet 브랜치만** (base DM frozen 강제) |
| supporting | `edge_codec` | `EdgeOnlyDataset` | edge → enc → channel → projector → decoder | `EdgeCodecLoss` = BCE+Dice | 전용 edge codec (`EdgeJSCC`) |
| supporting | `csi_estimation` | image latents | `√α·f0+√(1-α)·n` → SNR 추정 | `SNREstimationLoss` | `SNREstimator`만 |
| extension | `end_to_end_ft` | `TextImage[Edge]` | encode→channel→1-step denoise→decode | `EndToEndFTLoss` = w_r·recon + w_d·diff | JSCC/DM/ctrl 선택 조합 |

**논문 근거**: Stage 1 `L=‖x−x̂‖²+λ·L_GAN`(eq.7, 이후 JSCC frozen). Stage 2 Algorithm 1
(`t~U(0,1)`, `β̄_t=S(t)`, `∇‖f0−ε(f_t)‖²`) + MDTv2 masked 항. Stage 3 text-DM frozen,
구조 시맨틱 DiT 블록만 갱신.

## 논문과 다른 점 (정직한 목록)

1. **Patch-GAN (stage 1)** — 구조는 표준 `NLayerDiscriminator`(모든 knob config화),
   `gan.enabled=false`(기본)면 순수 MSE, `true`면 G/D 교대 학습. 단 원본 LPIPS 결합/
   가중 스케줄은 미재현(perceptual 수치 미보장). `lpips.enabled=true`(공개코드
   `MSE_LPIPS`와 정렬)도 선택 가능.
2. **Stage 3 edge transport** — baseline은 `edge_jscc`(전용 edge encoder→채널→
   projector, `edge_codec`이 학습한 체크포인트 로드). `shared_vae`(이미지 VAE stand-in)는
   비교용 ablation. codec 학습 데이터/스케줄은 논문 수치 미보장.
3. **end_to_end_ft** — baseline 아님. 논문 부록의 순차 미세조정을 tractable하게
   공동(joint)으로, 전체 reverse 대신 1-step denoise로 근사.
4. **데이터셋 규모** — 논문의 ~14M pair·250k step 스케줄/데이터는 미번들. 폴더 기반
   dataset + step-based 학습 기능만 제공.

## 주요 config (`configs/train/default.yaml`)

```yaml
train:
  stage: jscc
  epochs: 10 ; batch_size: 4 ; lr: 1.0e-4    # batch_size는 per-rank
  max_steps: 0                # >0 → step 모드 (save/val/log_every_steps)
  grad_accum_steps: 1 ; mixed_precision: false
  resume: null                # 경로 | "latest"/"auto" (checkpoint_dir 자동 탐색)

  dataset:
    caption_source: null      # sidecar | manifest | coco_json | multi_manifest | filename
    caption_path: null ; val_caption_path: null ; caption_select: first
    edge_source: null         # canny | sidecar | muge_sidecar
    muge_repr: reduced        # reduced(1ch) | edge_uncertainty(2ch) | multi(11ch)
    input_mode: folder        # folder | file_list

  jscc: { snr_db: 10.0, gan: {enabled: false, weight: 0.5}, lpips: {enabled: false} }
  dm:   { use_masked_branch: true, cfg_dropout_prob: 0.1, cfg_null_mode: zero }  # zero|learned
  edge_codec: { base_ch: 64, snr_db: 10.0, bce_weight: 1.0, dice_weight: 1.0,
                multi_snr: {enabled: false, min_db: 0, max_db: 20} }
  controlnet:
    allow_unfrozen_base_dm: false     # ⚠ base DM unfreeze (논문 위반)
    edge_transport: edge_jscc         # BASELINE=edge_jscc | ablation=shared_vae
    edge_jscc: { checkpoint: null, base_ch: 64, snr_db: 10.0 }
  end_to_end_ft: { train_jscc: true, train_diffusion: true, snr_db: 10.0 }
```

stage별 composed config: `composed_train_{jscc,text_dm,edge_codec,controlnet,
controlnet_shared_vae,csi_estimation,end_to_end_ft}.yaml`.

## Freeze 정책 (stage가 최상위, `training/freeze.py`)

먼저 모든 모듈을 freeze한 뒤 stage가 학습 대상만 opt-in한다. 레거시
`trainable_modules.freeze_*`는 stage 허용 범위 안에서 *추가 freeze*만 가능하다.

- `controlnet`: base DM **무조건 frozen**, ControlNet 브랜치만 학습.
  `train.controlnet.allow_unfrozen_base_dm: true`로만 해제(경고, 논문 이탈).
- `end_to_end_ft`: `train_{jscc,diffusion,controlnet}`로 학습 조합 지정.

## CLI 필수 인자 검증

| stage | 필수 입력 |
|-------|-----------|
| `jscc`/`edge_codec`/`csi_estimation` | image-only (edge_codec은 +`edge_source`) |
| `text_dm` | + `caption_source` (manifest면 `caption_path`) |
| `controlnet` | + `caption_source` + `edge_source` + 유효한 `edge_transport` |
| `end_to_end_ft` | + `caption_source`, 학습 대상 ≥1, `train_controlnet`이면 edge 필요 |

`edge_codec`/`csi_estimation`은 self-contained라 JSCC/DM 번들 로딩을 자동 생략한다.
`--no-models` dry-run은 배선·shape만 확인하고 학습 없이 종료한다.

## 체크포인트 / 로그

```
outputs/checkpoints/<stage>/{latest,best,epoch_NNNN}.pth
```

각 `.pth`: `epoch`, `global_step`, `stage`, `model_state`(학습 모듈만),
`optimizer_state`, `best_metric`. resume는 epoch과 **global_step을 모두 복원**한다.
로그는 `train_log.jsonl`에 JSON 한 줄(`global_step`, `loss`, stage별 항, `val_*`, `lr`).

### 학습 산출물 → 추론 체크포인트 export

학습 결과(`.../best.pth`)는 가중치를 `runner_state.modules.<name>`에 담아 추론 로더와
포맷이 다르므로 `scripts/export_checkpoint.py`로 변환한다.

| stage | export 포맷 | 추론 파일 |
|-------|-------------|-----------|
| `jscc` | raw `state_dict` | `checkpoints/JSCC_model.pth` |
| `text_dm` | `{"model_ema": …}` | `checkpoints/diffusion_backbone.pth` |
| `controlnet` | `{"model_ema": …}` | `checkpoints/diffusion_controlnet.pth` |

```bash
python scripts/export_checkpoint.py --stage text_dm \
    --input outputs/checkpoints/text_dm/best.pth \
    --output checkpoints/diffusion_backbone.pth [--force] [--dry-run]
```

`--force` 없이는 기존 output을 덮어쓰지 않는다. bare state_dict 입력은 `jscc`에서만
허용(diffusion stage는 무관 체크포인트가 `model_ema`로 감싸지는 것을 막기 위해 거부).
**export 불필요 두 경로**: `edge_codec→controlnet`은 `edge_jscc.checkpoint`가 직접 로드,
`csi_estimation→inference`는 `snr_estimator_checkpoint` 경로 연결.

## Edge codec (supporting stage)

stage 3의 `edge_jscc` transport가 쓸 전용 codec을 BCE+Dice로 학습한다(입력=edge map,
출력=edge reconstruction 확률).

```bash
# self-supervised: caption·JSCC/DM 체크포인트 불필요 (CPU 가능)
python scripts/train.py --config configs/composed_train_edge_codec.yaml \
    --train-list /data/edges/train/ --val-list /data/edges/val/ --epochs 50
python scripts/eval_edge_codec.py --config configs/composed_train_edge_codec.yaml \
    --checkpoint outputs/checkpoints/edge_codec/best.pth --val-list /data/edges/val/
#   → BCE/Dice/IoU@0.5/F1@0.5 출력, --snr 로 edge-link SNR 스윕
```

- **edge source**: `edge_source: canny`면 이미지 폴더만으로 on-the-fly Canny(cv2 없으면
  Sobel), `sidecar`면 `<stem>_edge.png`(또는 `edge_dir`)에서 로드.
- 학습된 체크포인트(`.../edge_codec/best.pth`)를 stage 3의
  `train.controlnet.edge_jscc.checkpoint`가 가리키게 한다. `base_ch`/`norm`은 codec
  학습 때와 동일해야 로드된다(로그에 `trained_codec=True`).

**주장 가능/한계**: stage 3 edge 조건이 전용 링크(독립 인코더 + 명시적 채널 + latent
정렬)로 들어오고 codec이 BCE+Dice로 **실제 학습**된 가중치를 쓴다("엣지를 자체 링크로
전송" 논문 의도와 정렬). 단 논문 edge JSCC는 ViT 기반이고 이 `EdgeJSCC`는
conv/ViT(`arch` 선택)이며 추론 ViT canny 코덱과는 **별개 모듈**이다. baseline config는
`edge_jscc.checkpoint`가 채워져 있어 파일이 없으면 즉시 `FileNotFoundError`(의도된
fail-fast) — stage 3 전에 반드시 `edge_codec`을 먼저 학습해야 한다.

## 논문 정렬 보강 (요약)

- **Stage-1 손실** — MSE only / public-code-like(`+0.1·LPIPS(alex)`) / paper-like
  (patch-GAN) 조합 선택. 기본 전부 off.
- **CSI 추정** — `SNREstimator`(공개 `Prediction_Model` 미러, paper-like) +
  `csi_estimation` stage로 실제 학습·추론 연결. 추론이 `net²=α`를 쓰므로 net은 진폭
  `√α`를 출력(target 메타데이터로 자동 √-wrap). phase/joint(Alg.3)은 복소 채널 확장
  전까지 scaffold.
- **adaLN SNR 조건화 edge codec** — `EdgeJSCCViT`의 `vit.snr_cond`가 WITT
  `SNREmbedder`/modulate 패턴 미러(선형 SNR `10**(snr_db/10)` 주입). 위치·값은
  WITT-aligned이나 블록은 DiT식 adaLN. multi-SNR 학습 시 SNR-adaptive.

## 데이터 입력 확장

기존 `sidecar`/`canny` 경로와 완전 호환되는 확장:

- **caption 생성** — `scripts/generate_captions.py`로 caption-less 폴더(CelebA/CelebA-HQ 등)에
  `<stem>.txt` 생성(`fixed`/`filename`/`model`). `model` 모드는 Qwen2.5-VL-3B-Instruct
  (`guidance/qwen_caption`)를 사용하며 `transformers>=4.49` + 가중치 + (권장) GPU 필요.
  이는 inference/eval 파이프라인의 BLIP-2와 **별개**라 논문 충실 forward pass에는 영향 없음.
  ⚠️ 논문 캡션과 동일하지 않음(paper-like). `paper_mode`에서 차단.
- **COCO multi-caption** — `caption_source: coco_json`(이미지당 5캡션 보존).
  train/val JSON이 다르므로 `caption_path`+`val_caption_path`를 각각 지정.
  `caption_select: first|longest|random`(val은 재현성 위해 `first`로 강등).
- **file-list 입력** — `input_mode: file_list` + `file_list_path`(DiffJSCC식 train.list).
  모든 dataset type에서 동작, 상대경로는 리스트 파일 기준.
- **대규모 변환 스크립트**:
  - `scripts/prepare_cc3m.py` — CC3M-WDS `.tar` shard → jpg/txt sidecar pair.
    기본 shard별 하위폴더 레이아웃(full-scale inode 병목 회피). `--append` +
    `--delete-shard-on-success`로 디스크 압박 없이 순차 변환(원자적 commit,
    `.shard_done` 마커, 재실행 멱등). ⚠️ `--delete-shard-on-success`는 검증된 shard의
    원본 tar를 영구 삭제.
  - `scripts/prepare_sa1b.py` — SA-1B `.tar` → image-only(`.json` 마스크 드롭). 어떤
    학습 dataset도 마스크를 소비하지 않아 image stage(`jscc`/`csi_estimation`/`edge_codec`)
    용. 동일한 순차 변환·삭제 안전장치. ⚠️ tar가 root 소유라 **컨테이너 내부(root)**에서 실행.

### stage별 데이터 (요약)

| stage | 필요 데이터 |
|-------|-------------|
| `jscc`/`edge_codec`/`csi_estimation` | image-only (SA-1B는 `prepare_sa1b.py` 변환) |
| `text_dm`/`controlnet` | caption 필요 (COCO는 `coco_json`, CC3M은 변환, CelebA는 caption 생성) |

## 운영 안정성 & 메모리 토글

기존 인프라(step checkpointing, AMP, grad accum, exact resume)에 더한 기능. 모든 신규
토글 기본 **off**라 재현성에 영향 없음.

- **SIGINT/SIGTERM 안전 저장** (`training/interrupt.py`) — 시그널 시 안전 지점에서
  `interrupt_latest.pth` 저장 후 정상 종료(두 번째 시그널은 강제 종료). DDP는 rank 0만 기록.
- **Auto-resume** (`train.resume: latest`) — `checkpoint_dir`의 `latest.pth`(없으면
  `interrupt_latest.pth`) 자동 탐색. 없으면 fresh run으로 시작.
- **Validation image logging** (`train.val_images.*`) — val 첫 배치를 고정해 eval
  모드로 `input | edge | recon` 패널 저장(rank 0). runner의 1-step f0 예측 + 공유 VAE
  decode 재사용. 기본 off.
- **메모리 토글** (`training/perf.py`) — 실효 있는 것은 `use_8bit_adam`(bitsandbytes
  설치 시 optimizer state VRAM 절감) **하나**뿐. `gradient_checkpointing`은 현재
  core 모듈에 hook이 없어 **no-op**, `use_xformers`는 MDTv2가 이미 native memory-efficient
  attention을 호출해 **추가 최적화 아님**(둘 다 상태 보고용, 로그로 명시). 실질 권장은
  `mixed_precision: true` + `num_workers` 상향.

## Multi-GPU (DDP)

`torchrun`으로 PyTorch DDP 지원(단일 프로세스/CPU 경로는 불변). `batch_size`는
**per-rank**: `global_batch = batch_size × world_size × grad_accum_steps`.

```bash
torchrun --standalone --nproc_per_node=3 scripts/train.py \
    --config configs/paper_train_text_dm.yaml \
    --train-list data/coco/train2017 --val-list data/coco/val2017 --batch-size 21
```

- **Stage 2 (`text_dm`)**: 지원·검증(param + learned CFG null token grad sync,
  DistributedSampler, rank0 checkpoint; CPU Gloo smoke + 3×GPU NCCL).
- **Stage 3 (`controlnet`)**: 구조 준비(`find_unused_parameters=False` 기본, base DM
  frozen). 멀티-GPU end-to-end 검증은 미완.
- **Stage 1/`edge_codec`**: 배선은 generic이나 GAN/codec 경로 DDP 미검증.

export·evaluation은 single-process(DDP는 학습 전용). 상세 DDP 동작·검증은
[paper_gap_closure.md](./paper_gap_closure.md#multi-gpu-training-ddp).

## 관련 문서
- [smoke_training.md](./smoke_training.md) · [paper_training_alignment.md](./paper_training_alignment.md) · [dataset_status.md](./dataset_status.md) · [phase5.md](./phase5.md)
- `python scripts/report_datasets.py` — 현재 머신 데이터 보유 상태 리포트
</content>

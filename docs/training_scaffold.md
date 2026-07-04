> [← docs index](./README.md)

# 학습 CLI — Stage-Aware Training Framework

## 개요

`scripts/train.py`는 `sgdjscc_lab`의 학습 진입점이다. SGD-JSCC 논문
("Semantics-Guided Diffusion for Deep Joint Source-Channel Coding", Sec. VI
*Training Details*)의 **3-stage 학습 절차**를 구조적으로 재현한다. 이 3-stage가
**core baseline**이며, 그 외 두 개는 baseline이 아니다.

```
[CORE BASELINE — 논문 3-stage]
원본 이미지 → [stage 1: jscc]         JSCC 인코더/디코더 (고정 AWGN SNR=10dB)
            → [stage 2: text_dm]      text-guided latent DM (f0 예측)
            → [stage 3: controlnet]   edge ControlNet 브랜치 (base DM frozen)

[supporting — stage 3가 소비하는 부품을 학습]
  [edge_codec]      전용 edge JSCC codec을 BCE+Dice로 학습 → stage 3의
                    edge_jscc transport가 이 체크포인트를 로드한다.

[extension — core baseline이 아님, 별도 추가 실험]
  [end_to_end_ft]   3-stage 완료 후 JSCC↔DM 공동 미세조정 (부가 실험)
```

**용어 정리 (논문 표기 시 오해 방지):**

| 구분 | stage | 위치 | baseline 주장 |
|------|-------|------|---------------|
| core baseline | `jscc`, `text_dm`, `controlnet` | 메인 이미지 파이프라인 | ✅ 이 3개가 baseline |
| supporting | `edge_codec` | stage 3 부품(edge codec) 학습 | baseline의 *구성요소*를 학습 |
| extension | `end_to_end_ft` | 3-stage 이후 추가 실험 | ❌ baseline 아님 (확장) |

기존 추론(`infer_pipeline`) / 평가(`eval_pipeline`) 경로와 원본 SGD-JSCC
기준선은 **건드리지 않는다**. 학습은 `train.stage` 값에 따라 dataset / forward /
loss / freeze 정책이 모두 분기되며, 하나의 통합 루프(global-step 기준)만
공유한다(stage-agnostic). epoch 기준과 step 기준 종료/저장/검증을 모두 지원한다.

---

## 빠른 시작

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest

# Stage 1 — JSCC (이미지 전용, 고정 SNR=10dB, MSE[+patch-GAN])
python scripts/train.py --config configs/composed_train_jscc.yaml \
    --train-list /data/imagenet/train/ --val-list /data/imagenet/val/ \
    --device cuda:0 --epochs 20

# Stage 2 — text-guided DM (caption-image pair, sidecar .txt 캡션)
python scripts/train.py --config configs/composed_train_text_dm.yaml \
    --train-list /data/pairs/train/ --device cuda:0

# (supporting) edge codec — stage 3의 edge_jscc transport가 쓸 codec을 먼저 학습
#   self-contained: JSCC/DM 체크포인트 불필요, caption 불필요 (edge map만)
python scripts/train.py --config configs/composed_train_edge_codec.yaml \
    --train-list /data/edges/train/ --val-list /data/edges/val/ \
    --device cuda:0 --epochs 50
python scripts/eval_edge_codec.py --config configs/composed_train_edge_codec.yaml \
    --checkpoint outputs/checkpoints/edge_codec/best.pth --val-list /data/edges/val/

# Stage 3 — edge ControlNet (BASELINE = 전용 edge_jscc transport)
#   train.controlnet.edge_jscc.checkpoint 가 위 edge_codec 결과를 가리켜야 한다.
python scripts/train.py --config configs/composed_train_controlnet.yaml \
    --train-list /data/pairs/train/ --device cuda:0

# Stage 3 (ABLATION) — shared_vae 엣지 transport (전용 링크 없음, 비교용)
python scripts/train.py --config configs/composed_train_controlnet_shared_vae.yaml \
    --train-list /data/pairs/train/ --device cuda:0

# (extension, baseline 아님) end-to-end JSCC↔DM 미세조정 (step-based, grad-accum, AMP)
python scripts/train.py --config configs/composed_train_end_to_end_ft.yaml \
    --train-list /data/pairs/train/ --device cuda:0 \
    --resume outputs/checkpoints/controlnet/latest.pth

# 대규모 step-based 학습 (논문 DM stage ≈ 250k step)
python scripts/train.py --config configs/composed_train_text_dm.yaml \
    --train-list /data/pairs/train/ --device cuda:0 --max-steps 250000

# Dry-run — config/stage/dataset 배선만 검증(체크포인트·GPU 불필요)
python scripts/train.py --config configs/composed_train_jscc.yaml \
    --train-list /path/to/images/ --no-models --epochs 1
```

`--stage {jscc|text_dm|controlnet|edge_codec|end_to_end_ft}`로 `train.stage`를,
`--max-steps`로 `train.max_steps`를 덮어쓸 수 있다. 잘못된 stage 설정(누락된
caption/edge source, 잘못된 edge_transport, 학습 대상 0개 등)은 **체크포인트
로딩 이전에** 명확한 예외로 실패한다.

> **권장 학습 순서 (baseline 재현):**
> `jscc` → `text_dm` → `edge_codec` → `controlnet`.
> (`edge_codec`는 `text_dm` 전에 돌려도 무방하다 — JSCC/DM과 독립적이다.)
> `end_to_end_ft`는 baseline에 포함되지 않으며, 위 4개가 끝난 뒤의 *추가* 실험이다.

### 학습 산출물 → 추론 체크포인트 export

학습 결과(`outputs/checkpoints/<stage>/best.pth`)는 모듈 가중치를
`runner_state.modules.<name>`에 담는다. 추론/runtime 로더가 기대하는 포맷은
이와 다르므로 **수동 복사가 아니라 포맷 변환(export)** 이 필요하다.
`scripts/export_checkpoint.py`가 이를 자동화한다.

| stage | training 모듈 | export 포맷 | 추론 파일 |
|-------|---------------|-------------|-----------|
| `jscc` | `jscc_model` | raw `state_dict` | `checkpoints/JSCC_model.pth` |
| `text_dm` | `diffusion` | top-level `model_ema` | `checkpoints/diffusion_backbone.pth` |
| `controlnet` | `diffusion` | top-level `model_ema` | `checkpoints/diffusion_controlnet.pth` |

```bash
# Stage 1 JSCC → raw state_dict
python scripts/export_checkpoint.py --stage jscc \
    --input outputs/checkpoints/jscc/best.pth \
    --output checkpoints/JSCC_model.pth

# Stage 2 text_dm → {"model_ema": ...}
python scripts/export_checkpoint.py --stage text_dm \
    --input outputs/checkpoints/text_dm_coco_json/best.pth \
    --output checkpoints/diffusion_backbone.pth

# Stage 3 controlnet → {"model_ema": ...}
python scripts/export_checkpoint.py --stage controlnet \
    --input outputs/checkpoints/controlnet/best.pth \
    --output checkpoints/diffusion_controlnet.pth --force

# 어떤 키를 뽑는지 미리 확인 (파일 미작성)
python scripts/export_checkpoint.py --stage controlnet \
    --input outputs/checkpoints/controlnet/best.pth \
    --output checkpoints/diffusion_controlnet.pth --dry-run
```

`--force` 없이는 기존 output을 덮어쓰지 않는다. 입력 누락 / stage·체크포인트 키
불일치 / 추출 실패는 **저장 전에** 명확한 예외로 실패한다. 추출 위치는
`runner_state.modules.<name>`(현재 포맷) → `model_state.<name>`(legacy) 순으로
탐색한다.

> **bare state_dict 입력은 `jscc`에서만 허용한다.** bare(중첩 없는 raw
> state_dict)는 모듈 식별 정보가 없어 stage와 대조할 수 없는데, `jscc`의 export
> 결과는 그 자체가 raw state_dict라 1:1 pass-through로 안전하다. 반면
> `text_dm`/`controlnet`은 bare 입력을 **명시적으로 거부**한다 — 무관한 raw
> 체크포인트(예: `JSCC_model.pth`)가 조용히 `model_ema`로 감싸지는 것을 막기
> 위함이다. 따라서 diffusion stage에는 `runner_state.modules.diffusion`을 담은
> 학습 체크포인트(`.../<stage>/best.pth`)를 넘겨야 한다.

> **export 대상이 아닌 두 stage (포맷 변환 불필요):**
> - `edge_codec → controlnet` : stage 3 config의
>   `train.controlnet.edge_jscc.checkpoint`가
>   `outputs/checkpoints/edge_codec/best.pth`를 **직접 로드**한다(nested 추출은
>   로더가 수행). export하지 말고 경로만 연결하면 된다.
> - `csi_estimation → inference` : 포맷 변환이 아니라 **경로 연결** 대상이다.
>   stage 결과 경로를 `snr_estimator_checkpoint`에 넣으면
>   `models/csi_estimation.py`가 nested 가중치를 직접 추출한다.

---

## stage ↔ 코드 매핑

| 구분 | `train.stage` | Dataset | Forward (runner) | Loss | 학습 대상 (freeze 정책) |
|------|---------------|---------|------------------|------|------------------------|
| baseline 1 | `jscc` | `ImageOnlyDataset` | `JSCCStageRunner`: VAE encode → AWGN(SNR=10) → VAE decode | `JSCCStageLoss` = MSE (+ λ·GAN) | JSCC 학습, DM/guidance frozen |
| baseline 2 | `text_dm` | `TextImageDataset` | `TextDMStageRunner`: `f0=VAE(x)`, `t~U(0,1)`, `f_t`=sigmoid schedule, masked/unmasked 예측 | `DiffusionF0Loss` = `‖f0−ε(f_t)‖²` + `‖f0−ε(f̂_t)‖²` | base DM 학습, JSCC frozen |
| baseline 3 | `controlnet` | `TextImageEdgeDataset` | `ControlNetStageRunner`: stage2 forward + edge 조건 `c` (transport 선택) | `DiffusionF0Loss` (동일) | **ControlNet 브랜치만** 학습, base DM frozen (강제) |
| supporting | `edge_codec` | `EdgeOnlyDataset` | `EdgeCodecStageRunner`: edge → encoder → channel → projector → decoder → edge logits | `EdgeCodecLoss` = `BCE + Dice` | 전용 edge codec(`EdgeJSCC`) 학습. JSCC/DM 불필요 |
| extension | `end_to_end_ft` | `TextImage[Edge]` | `EndToEndFTStageRunner`: encode→channel→1-step DM denoise→decode | `EndToEndFTLoss` = `w_r·‖x−x̂‖²` + `w_d·‖f0−ε(f_t)‖²` | JSCC/DM/ctrl 선택적 공동 학습 |

논문 근거:
- Stage 1: 고정 잡음 채널 설정(AWGN, SNR=10dB)에서 학습,
  `L = ‖x−x̂‖² + λ·L_GAN` (eq. 7). 이 단계 이후 JSCC 모델은 고정(frozen)된다.
- Stage 2: Algorithm 1 — `t~U(0,1)`, `β̄_t=S(t)`, `f_t=√(1−β̄_t)f0+√β̄_t·n`,
  `∇‖f0−ε(f_t,β̄_t)‖²`. MDTv2의 masked-latent modeling으로 masked 항 추가.
- Stage 3: 원본 text-guided DM의 파라미터는 frozen, 구조적 시맨틱 특징을 다루는
  DiT 블록의 파라미터만 갱신한다.
- 확장 (end_to_end_ft): 논문 부록의 DM→JSCC-decoder 순차 미세조정 아이디어를
  **공동(joint)** 형태로 구조화한 것 (아래 "논문과 다른 점" 참고).

---

## 구현 수준 (full / scaffold)

| 항목 | 상태 | 비고 |
|------|------|------|
| stage 분기 (`train.stage`) | ✅ full | `training/stages.py` (4 stage) |
| stage별 config 검증 + 조기 실패 | ✅ full | caption/edge/transport/trainable-set 누락 시 명시적 예외 |
| Dataset 3종 + 공통 transform | ✅ full | `image / text_image / text_image_edge`, center/random crop + resize |
| **Stage 1 JSCC forward/loss** | ✅ full | VAE encode→AWGN→decode 미분 경로, 실모델로 gradient 검증됨 |
| Stage 1 고정 SNR=10dB | ✅ full | `train.jscc.snr_db`, runner가 `jscc.snr`에 강제 적용 |
| Stage 1 patch-GAN | ✅ 구조 full / ⚠️ 가중치 stand-in | NLayerDiscriminator(ndf/n_layers/norm/lr/weight config화), G/D 교대학습, AMP+grad-accum. 원본 LDM 정확 수치는 미보장 |
| **Stage 2 DM forward/loss** | ✅ full(구조) | masked+unmasked 두 항 모두 계산, sigmoid schedule는 추론과 동일 |
| Stage 2 text encode | ✅ full | DiffusionGenerator.encode_text(CLIP), no_grad+detach |
| **Stage 3 ControlNet** | ✅ full(구조) | control 브랜치만 학습, base DM frozen 강제 |
| Stage 3 freeze 강제 정책 | ✅ full | `train.controlnet.allow_unfrozen_base_dm`로만 해제 가능 |
| Stage 3 edge transport `edge_jscc` (**BASELINE**) | ✅ full | 전용 edge encoder→channel→projector (`models/edge_jscc.py`). `edge_codec` stage가 BCE+Dice로 **실제 학습**한 codec 체크포인트를 로드(`train.controlnet.edge_jscc.checkpoint`). 체크포인트 미지정 시 무작위 stand-in(=ablation급, 경고 출력) |
| Stage 3 edge transport `shared_vae` (ablation) | ✅ full(구조) | 이미지 VAE stand-in. 전용 전송 링크 없음 — 비교용 ablation |
| **edge_codec stage (BCE+Dice 학습형 edge codec)** | ✅ full | `EdgeJSCC` encoder+projector+**decoder**를 self-supervised로 학습. `models/edge_jscc.py::reconstruct`, `EdgeCodecLoss`, `EdgeCodecStageRunner`. JSCC/DM 체크포인트 불필요 |
| step-based training | ✅ full | `max_steps/save·val·log_every_steps`, global_step resume |
| grad accumulation | ✅ full | `grad_accum_steps`, optimizer-step 경계에서만 step |
| mixed precision (AMP) | ✅ full | `mixed_precision`, CPU에서 자동 비활성화 |
| **end_to_end_ft stage** | ✅ 구조 / ⚠️ 단순화 | JSCC+DM 공동 학습, 1-step denoise 기반 recon (아래 참고) |
| 대규모 외부 데이터셋(14M pair, SA-1B 등) | 🔲 미구현 | 폴더 기반 dataset 인터페이스만 제공 |
| 노이즈 스케줄러 `SigmoidNoiseScheduler` | ✅ full | 추론 코드의 `sigmoid_schedule`와 수식 일치 |

---

## 논문과 완전히 동일하지 않은 부분

1. **Patch discriminator (stage 1 GAN).** 구조는 표준 Pix2Pix/LDM
   `NLayerDiscriminator`와 동일하고 모든 knob(`ndf`, `n_layers`, `norm`, `mode`,
   `weight`, `lr`)이 config로 노출된다. 다만 원본 repo의 LPIPS 결합/가중 스케줄은
   재현하지 않아 perceptual 수치 일치는 보장하지 않는다.
   `train.jscc.gan.enabled=false`(기본)에서는 **순수 MSE-only**로 명확히 동작하며,
   `true`면 G/D 교대 학습이 AMP·grad-accum과 함께 돌아가고 로그에
   `loss_mse`/`loss_gan`/`loss_disc`가 남는다.
2. **Stage 3 edge transport.** 두 모드를 제공하며, **baseline은 `edge_jscc`**다.
   - `edge_jscc` (**BASELINE**): 전용 edge encoder → 무선 채널 → latent projector.
     논문의 "엣지를 자체 DeepJSCC 링크로 전송 후 latent 정렬" 구조와 일치한다
     (`models/edge_jscc.py`). codec 가중치는 더 이상 무작위 stand-in이 아니라
     **`edge_codec` stage가 BCE+Dice로 학습**한 체크포인트를 로드한다
     (`train.controlnet.edge_jscc.checkpoint`).
     단, **`checkpoint`를 지정하지 않으면** codec은 무작위 초기화 상태로 남고
     (경고 출력) 이때는 baseline이 아니라 ablation급으로만 취급해야 한다.
   - `shared_vae` (ablation): 엣지 맵을 이미지 VAE로 인코딩하는 간단 stand-in.
     전용 전송 링크가 없으므로 **baseline이 아닌 비교용 ablation**이다.

   남은 차이: edge codec의 학습 데이터·스케줄·정확한 손실 가중은 논문 수치와
   동일함을 보장하지 않는다(구조와 BCE+Dice 목적은 일치). decoder head는 codec
   학습용이며 stage 3 추론에는 `c`(condition latent)만 사용한다.
3. **end_to_end_ft (extension, baseline 아님).** 이는 core baseline(3-stage)에
   포함되지 않는 **별도 확장 실험**이다. 논문 부록은 DM을 먼저, 그다음 JSCC
   디코더를 **순차** 미세조정한다. 본 구현은 tractable하게 **공동(joint)** 으로
   하며, 전체 reverse diffusion 대신 **1-step denoise**(채널 SNR→noise level
   매핑)로 recon 항을 만든다. 즉 목적은 비슷하나 절차/스텝 수가 다르며, baseline
   비교표에는 넣지 않고 "추가 실험" 또는 "ablation/extension"으로만 보고한다.
4. **데이터셋 규모/구성.** 논문의 ~1,400만 text-image pair(SA-1B, JourneyDB,
   CC3M, Datacomp, CelebV-HQ) 및 250k step 학습 스케줄/데이터는 포함하지 않는다.
   폴더 기반 dataset과 caption/edge source 인터페이스, 그리고 그 스케줄을 돌릴 수
   있는 step-based 학습 기능을 제공한다.

---

## Config 주요 항목

```yaml
# configs/train/default.yaml (발췌)
train:
  stage: jscc                 # jscc | text_dm | controlnet | end_to_end_ft
  epochs: 10
  batch_size: 4
  lr: 1.0e-4

  # 대규모 step-based 학습 (0 = epoch 모드)
  max_steps: 0                # >0 → global optimizer step 기준 종료
  save_every_steps: 0
  val_every_steps: 0
  log_every_steps: 0
  grad_accum_steps: 1         # N micro-batch 누적 후 1 optimizer step
  mixed_precision: false      # torch.cuda.amp; CPU에서 자동 off

  dataset:
    type: auto                # auto → stage에서 유도
    caption_source: null      # sidecar | manifest | filename  (text_dm/controlnet/e2e)
    caption_path: null        # manifest 사용 시 JSON/CSV 경로
    edge_source: null         # canny | sidecar               (controlnet)
    edge_dir: null

  transforms:
    resize_to: 128            # int 또는 [H, W]
    crop_mode: center         # center | random | none

  jscc:
    snr_db: 10.0              # 고정 AWGN SNR (논문 10dB)
    gan:                      # patch-GAN (NLayerDiscriminator)
      enabled: false
      weight: 0.5             # λ
      mode: hinge             # hinge | vanilla
      lr: 1.0e-4
      ndf: 64                 # base 채널
      n_layers: 3             # stride-2 블록 수
      norm: batch             # batch | instance | none

  dm:
    use_masked_branch: true
    mask_weight: 1.0
    cfg_dropout_prob: 0.1       # CFG null-conditioning dropout (training only)
    scheduler: { start: 0.0, end: 3.0, tau: 0.7 }

  edge_codec:                 # supporting: 전용 edge codec 학습 (BCE+Dice)
    base_ch: 64               # controlnet.edge_jscc.base_ch 와 일치시킬 것
    norm: group
    snr_db: 10.0
    use_channel: true
    bce_weight: 1.0
    dice_weight: 1.0

  controlnet:
    allow_unfrozen_base_dm: false   # ⚠ base DM unfreeze (논문 위반) 위험 플래그
    edge_transport: edge_jscc       # BASELINE=edge_jscc | ablation=shared_vae
    edge_jscc:
      checkpoint: null              # edge_codec 학습 결과 경로(baseline 필수)
      base_ch: 64                   # checkpoint와 일치
      norm: group
      snr_db: 10.0
      use_channel: true

  end_to_end_ft:              # extension (baseline 아님)
    train_jscc: true
    train_diffusion: true
    train_controlnet: false
    snr_db: 10.0
    recon_weight: 1.0
    diff_weight: 1.0
```

stage별 composed config:
- `composed_train_jscc.yaml`, `composed_train_text_dm.yaml` — baseline stage 1/2.
- `composed_train_edge_codec.yaml` — supporting: edge codec 학습.
- `composed_train_controlnet.yaml` — **baseline stage 3 (edge_jscc transport)**.
- `composed_train_controlnet_edge_jscc.yaml` — 위 baseline의 명시적 이름 alias.
- `composed_train_controlnet_shared_vae.yaml` — **ablation** (shared_vae transport).
- `composed_train_end_to_end_ft.yaml` — extension (step-based + grad-accum + AMP).

---

## Edge codec (supporting stage) — 데이터 준비 & 학습 절차

`edge_codec` stage는 stage 3의 `edge_jscc` transport가 사용할 **전용 edge codec**을
실제로 학습한다. 입력은 edge map, 출력은 edge **reconstruction 확률**(logit)이며,
BCE + soft-Dice로 학습한다.

### 1) 데이터 준비

- **caption 불필요.** edge_codec은 self-supervised이다 (edge가 입력이자 타깃).
- 두 가지 edge source 중 선택:
  - `edge_source: canny` — **이미지 폴더만 있으면 된다.** edge map을 학습 중
    on-the-fly로 Canny(또는 cv2 미설치 시 Sobel)로 계산한다. 별도 edge 파일 불필요.
  - `edge_source: sidecar` — 미리 만든 edge map을 읽는다. `edge_dir`(또는
    이미지 옆 `<stem>_edge.png`)에서 파일명 매칭으로 찾는다. edge map은 흑백
    (밝을수록 edge)으로 저장하면 되며, 로드 시 단일 채널로 변환된다.
- 디렉터리 구조 / 파일명 규칙:

```
data/edges/
├── train/
│   ├── sample_000.png         # 원본 이미지 (canny면 이것만 있으면 됨)
│   ├── sample_000_edge.png    # (sidecar일 때만) 같은 stem + _edge.png
│   ├── sample_001.png
│   ├── sample_001_edge.png
│   └── …
└── val/
    ├── sample_100.png
    └── …
```
  - sidecar에서 `edge_dir`를 따로 둘 경우: `edge_dir/<stem>.<ext>`로 매칭한다
    (`<stem>`은 이미지 파일명에서 확장자 뺀 부분).
- 이미지 크기: 한 변이 128의 배수면 좋다(`transforms.resize_to: 128`이 처리).

### 2) 학습 명령어

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest

python scripts/train.py --config configs/composed_train_edge_codec.yaml \
    --train-list /data/edges/train/ --val-list /data/edges/val/ \
    --device cuda:0 --epochs 50
```
- JSCC/DM 체크포인트도, GPU도(원하면 `--device cpu`) 필수가 아니다 — codec은
  처음부터(scratch) 학습한다.
- 평가:
```bash
python scripts/eval_edge_codec.py --config configs/composed_train_edge_codec.yaml \
    --checkpoint outputs/checkpoints/edge_codec/best.pth --val-list /data/edges/val/
# → BCE / Dice / IoU@0.5 / F1@0.5 출력. --snr 로 edge-link SNR 스윕 가능.
```

### 3) 학습된 codec을 stage 3에 연결

체크포인트는 `outputs/checkpoints/edge_codec/{best,latest}.pth`에 생성된다.
이를 stage 3 baseline config의 `train.controlnet.edge_jscc.checkpoint`가 가리키게
한다(기본 baseline config는 `../outputs/checkpoints/edge_codec/best.pth`로 이미
설정되어 있다 — 경로만 맞으면 됨). `base_ch`/`norm`은 codec 학습 때와 동일해야
로드된다(encoder/projector를 strict=False로 로드, decoder 키는 무시).

```bash
python scripts/train.py --config configs/composed_train_controlnet.yaml \
    --train-list /data/pairs/train/ --device cuda:0
# 로그에 trained_codec=True 가 찍히면 정상.
```

### 4) 주장 가능한 수준 / 남는 한계 (과대표현 주의)

> 표현 가이드: 이 transport는 **"논문 완전 재현 baseline"이 아니라 "부분적으로
> 충실한 학습 scaffold + 학습된 edge codec 보강"** 으로 기술하는 것이 안전하다.
> 논문에 충실한 부분은 추론 경로이고, 이 학습 scaffold는 근사다.

- **주장 가능:** stage 3의 edge 조건이 *전용 edge 링크*(독립 인코더 + 명시적
  무선 채널 + latent 정렬)를 통해 들어오고, codec이 BCE+Dice로 **실제 학습**된
  가중치를 쓴다. "엣지를 자체 링크로 전송 후 latent 정렬"이라는 논문의 설계 *의도*
  와 정렬되며, `shared_vae` 대비 명확한 ablation 축을 제공한다.
- **남는 한계 (반드시 같이 보고):**
  1. **아키텍처 불일치:** 논문의 edge JSCC는 **ViT 기반**(SNR을 디코더 트랜스포머
     블록에 투영)인데 이 `EdgeJSCC`는 **conv encoder→channel→projector→decoder**다.
     또한 *추론* 경로가 쓰는 원본 ViT canny 코덱(`model_canny`)과도 **별개 모듈**
     이다 — 이 학습 codec은 추론 path를 대체하지 않는다.
  2. **운영 리스크 (fallback이 아니라 crash):** 기본 baseline config는
     `edge_jscc.checkpoint`에 경로가 **채워져** 있다. 그 파일이 없으면 graceful
     fallback이 아니라 **즉시 `FileNotFoundError`로 실패**한다(의도된 fail-fast).
     무작위 stand-in으로의 강등은 오직 `checkpoint: null`을 **명시**했을 때만
     일어난다. 따라서 stage 3 baseline을 돌리기 전 반드시 `edge_codec`을 먼저
     학습해 체크포인트를 만들어야 한다.
  3. codec의 학습 데이터 규모·스케줄·손실 가중이 논문 수치와 동일함을 보장하지 않는다.
  4. Canny 임계값·edge 정의가 논문과 다를 수 있다.
  5. decoder는 codec 학습용 head이며 stage 3 추론에는 condition latent `c`만
     쓴다(수신 edge map 시각화는 평가 CLI에만 있음).
  6. baseline 표에는 반드시 *학습된* 체크포인트로 얻은 결과만 넣는다.

---

## 논문 정렬 보강 — Stage-1 손실 / CSI 추정 / adaLN edge codec

세 항목을 논문(1차) + 공개 SGDJSCC 코드(2차) 기준으로 정렬했다. 각 조합의
충실도를 과장 없이 구분한다.

### (1) Stage-1 손실 조합 (`JSCCStageLoss`, `training/losses.py`)

- **논문 근거**: eq.7 `L = ‖x−x̂‖² + λ·max_D(L_GAN)` (patch discriminator).
- **공개 코드 근거**: `utils/loss_function.py::MSE_LPIPS` = `mse + 0.1·lpips(2X−1, alex)`
  (frozen). 공개 inference repo의 JSCC 손실엔 discriminator가 아니라 **LPIPS**가 쓰임.

| 조합 | config | 충실도 |
|------|--------|--------|
| MSE only | `gan.enabled=false`, `lpips.enabled=false` | 최소 baseline |
| **public-code-like** | `lpips.enabled=true, lpips.weight=0.1, lpips.net=alex` (GAN off) | 공개 코드 `MSE_LPIPS`와 정렬 |
| **paper-like** | `gan.enabled=true` (patch-GAN, λ=weight) | 논문 eq.7 형태(수치 미보장) |
| extension | GAN + LPIPS 동시, 또는 다른 net/weight | 논문/공개코드 둘 다 아님 |

> patch-GAN은 구조만 정렬(LPIPS 결합/스케줄·LDM 정확 수치 미보장). LPIPS는 `lpips`
> 패키지 필요(없으면 graceful skip). 기본값은 모두 **off**.

### (2) 블라인드 CSI 추정 (`models/csi_estimation.py`)

- **논문 근거**: Sec. IV-C. SNR 추정 eq.15 `min ‖ζ_P(√α f+√(1-α)n)−α‖²`, phase 추정
  `ξ_Q(…,α)→φ/π` (AF 모듈로 SNR 투영), joint 추정 Algorithm 3(φ=0 초기화 후 교대).
- **공개 코드 근거**: `snr_prediction_net.py::Prediction_Model`(16ch→4 resblock→FC→sigmoid→α)
  = **SNR 추정만 존재**. phase net·joint loop은 **공개 코드에 없음**.

| 구성 | 충실도 |
|------|--------|
| `SNREstimator` | **paper-like** (공개 `Prediction_Model` 구조 미러, 미학습) |
| `PhaseEstimator`(+`AFModule`), `joint_csi_estimate`(Alg.3) | **paper-inspired scaffold** — 공개 코드에 없고, sgdjscc_lab 채널은 실수 gain(복소 위상 없음)이라 phase 제거/추정은 구조적 placeholder. 단, **출력 계약은 정렬**: estimator는 √α(amplitude)를 내고 `joint_csi_estimate`가 `output_is_amplitude` 플래그로 **α(level)=out²** 변환 후 phase net(α 입력)에 넘김 |
| 손실 `SNREstimationLoss`/`PhaseEstimationLoss`, `synthesize_noisy_latent` | self-supervised 목적(`√α·f0+√(1-α)·n`) |
| **`csi_estimation` 학습 stage** | `CSIEstimationStageRunner` + `--stage csi_estimation` + `composed_train_csi_estimation.yaml`. **SNR 추정기만** 학습(image latents). phase/joint은 stage에서 학습 안 함(scaffold) |
| **추론 연결** | `snr_estimator_checkpoint` → `runtime.build_models`가 `jscc.snr_prediction_net`을 학습된 `SNREstimator`로 **교체**(`load_snr_estimator_into`). 학습 결과가 실제 blind step-matching 경로를 구동 |

> **출력 계약 정합**: 추론은 `snr_prediction_net(f̄)**2`로 signal level을 얻으므로
> 공개 net은 **진폭 √α**를 출력한다. 그래서 stage는 기본 `target: amplitude`(√α)로
> 학습 → 학습된 net이 `jscc.snr_prediction_net`의 **drop-in**이 된다(`net²=α`).
> `target: alpha`(논문 eq.15, α 직접 회귀)로 학습한 체크포인트는 **target 메타데이터를
> 체크포인트에 기록**(`runner_state.meta.csi_target`)하고, `load_snr_estimator_into`가
> 이를 읽어 **자동으로 √-wrap**(`_SqrtSNRAdapter`)하므로 squaring 런타임에서도
> `net²=α`가 성립한다(α² 오로딩 방지, 경고 로그). 메타데이터 없는 외부 state_dict는
> 기본값 amplitude로 가정.
>
> SNR 추정기는 `scripts/train.py --stage csi_estimation`으로 **실제 학습 + 추론 연결**
> 까지 가능(JSCC VAE로 image latent 생성). phase/joint은 복소채널 확장 전엔 학습
> 대상이 아니다(scaffold).

### (3) adaLN SNR 조건화 edge codec (`models/edge_jscc.py`)

- **논문/공개 코드 근거**: `model_canny.py`의 WITT ViT가 SNR을 adaLN으로 블록에 투영
  (`revised_witt/witt_modules.py`의 `SNREmbedder`/`modulate`/`Head_layer`).
- **구현**: `EdgeJSCCViT`에 `vit.snr_cond` 옵션 추가 — `SNREmbedder`+adaLN-Zero 블록
  (`_AdaLNBlock`)이 WITT의 `SNREmbedder`/`modulate` 패턴을 미러. 조건화 **값**도 공개
  WITT 규약을 따라 **선형 SNR `10**(snr_db/10)`** 를 넣는다(`model_canny.py`의
  `snr_scale=10**(snr/10)`와 일치, dB 아님). **위치+값은 WITT-aligned, 단 블록은 DiT식
  adaLN이라 Swin-window WITT-exact 아님.**
- **CAVEAT**: `edge_codec`은 고정 SNR로 학습 → 조건화가 **상수 modulation**. SNR-adaptive로
  만들려면 varying-SNR 학습 + per-forward SNR 주입(`EdgeJSCC._snr_tensor`). 기본 off.

---

## Freeze 정책 (stage 강제)

freeze는 stage 정책이 **최상위**다(`training/freeze.py`).

- 먼저 모든 모듈을 freeze한 뒤, stage가 학습 대상만 opt-in.
- 레거시 `trainable_modules.freeze_*` 플래그는 stage가 허용한 범위 안에서
  **추가로 freeze**만 가능하며, stage가 금지한 것을 unfreeze할 수는 없다.
- `stage=controlnet`: base text-guided DM은 **무조건 frozen**, ControlNet
  브랜치(`en_inblocks_controlnet`, `en_outblocks_controlnet`)만 학습.
  유일한 해제 방법은 `train.controlnet.allow_unfrozen_base_dm: true`(경고 출력,
  논문 절차에서 벗어남).
- `stage=end_to_end_ft`: `train.end_to_end_ft.{train_jscc, train_diffusion,
  train_controlnet}`로 학습 대상을 조합한다. `train_diffusion=true`면 denoiser
  전체, `false`+`train_controlnet=true`면 control 브랜치만 학습. param group에
  중복이 생기지 않도록 학습 집합을 먼저 정한 뒤 그룹을 만든다.

---

## CLI별 필수 인자 검증

| stage | 필수 입력 | 누락 시 |
|-------|-----------|---------|
| `jscc` | `train_input_path`, `train.jscc.snr_db` | `StageConfigError` |
| `text_dm` | + `caption_source` (manifest면 `caption_path`) | `StageConfigError` |
| `edge_codec` | + `edge_source` (sidecar면 `edge_dir`). **caption 불필요** | `StageConfigError` |
| `controlnet` | + `caption_source` + `edge_source` (sidecar면 `edge_dir`) + 유효한 `edge_transport` | `StageConfigError` |
| `end_to_end_ft` | + `caption_source`; 학습 대상 ≥1개; `train_controlnet=true`면 `dataset.type=text_image_edge`(auto면 자동 승격)+`edge_source` 필수 | `StageConfigError` |

`edge_codec`는 self-contained이므로 `scripts/train.py`가 JSCC/DM 번들 로딩을
자동 생략한다(체크포인트 불필요). `--no-models` 없이 그냥 실행하면 된다.

`--no-models` dry-run은 stage·dataset 배선을 검증하고 한 배치를 꺼내 shape를
확인한 뒤, **학습 없이** 종료한다(조용히 도는 가짜 학습 상태 제거).

---

## 체크포인트 / 로그

```
outputs/checkpoints/<stage>/
├── latest.pth      — 매 에폭 / step 저장
├── best.pth        — loss 개선 시
└── epoch_0005.pth  — save_every(epoch) 주기
```

각 `.pth`: `epoch`, `global_step`, `stage`, `model_state`(runner가 학습한 모듈만),
`optimizer_state`, `best_metric`. 재개(`restore_runner_state`)는 epoch과
**global_step을 모두 복원**해 step 모드에서도 정확히 이어서 학습한다. 로그는
`train_log.jsonl`에 JSON 한 줄(`global_step`, `epoch`, `stage`, `loss`,
stage별 항(`loss_mse`/`loss_gan`/`loss_disc` 또는 `loss_dm_*` 또는
`loss_recon`/`loss_diff`), `val_*`, `lr`).

---

## 새/변경 파일

| 파일 | 역할 |
|------|------|
| `scripts/train.py` | `--stage`(+`edge_codec`/e2e), `--max-steps`, 조기 검증, edge_codec 번들 생략 |
| `scripts/eval_edge_codec.py` | edge codec 평가 CLI (BCE/Dice/IoU/F1) |
| `scripts/make_tiny_dataset.py` | smoke 학습용 tiny dataset 생성기 |
| `pipelines/train_pipeline.py` | 통합 global-step 루프 (epoch+step 모드, AMP/accum, resume) |
| `training/stages.py` | stage 상수·resolve·검증 (5 stage), `CORE_STAGES` |
| `training/noise_schedule.py` | `SigmoidNoiseScheduler` |
| `training/losses.py` | `JSCCStageLoss`, `DiffusionF0Loss`, **`EdgeCodecLoss`(BCE+Dice)**, `EndToEndFTLoss`, `PatchDiscriminator`+`build_discriminator`, `GANLoss` |
| `training/edge_transport.py` | `build_edge_transport`(shared_vae/edge_jscc + 체크포인트 로드), **`build_edge_codec`** |
| `models/edge_jscc.py` | 전용 edge JSCC: encoder→channel→projector(→**decoder**), `reconstruct`, `load_codec_state` |
| `training/freeze.py` | `apply_stage_freeze_policy` (controlnet/edge_codec/e2e 정책) |
| `training/stage_runners.py` | JSCC/TextDM/ControlNet/**EdgeCodec**/EndToEndFT runner + AMP·grad-accum |
| `data/transforms.py`, `data/datasets.py` | 공통 transform + 4종 dataset(+`EdgeOnlyDataset`) |
| `configs/train/default.yaml` | step-based/GAN/edge_transport/edge_codec/e2e 필드 |
| `configs/composed_train_{jscc,text_dm,edge_codec,controlnet,controlnet_edge_jscc,controlnet_shared_vae,end_to_end_ft}.yaml` | stage별 composed config |
| `tests/test_train_stages.py` | stage/edge-transport/edge_codec/step/accum/e2e 단위 테스트 |
| `docs/smoke_training.md` | real-model smoke 학습 가이드 |

---

## 논문용 표현 제안 (오해 방지)

baseline / supporting / extension의 경계를 본문에서 명확히 하려면 아래 문장을
참고할 수 있다. **표현 원칙:** 추론 경로는 *paper-faithful*, 학습 scaffold는
*partially faithful (구조적 근사)* 로 기술해 과대주장을 피한다.

- **충실도 framing (권장):**
  > "We reuse the authors' original SGD-JSCC inference pipeline (paper-faithful
  > forward pass) and add a *partially faithful* training scaffold that
  > structurally reproduces the three-stage procedure; the scaffold approximates
  > rather than bit-exactly reproduces the paper's edge codec, GAN, and data scale."

- **Core stages 정의 (재현 *구조*):**
  > "Our training scaffold follows the three-stage SGD-JSCC procedure: (1) a
  > JSCC encoder/decoder under a fixed AWGN channel, (2) a text-guided latent
  > diffusion model, and (3) an edge-conditioned ControlNet branch on top of the
  > frozen stage-2 model."

- **Edge codec (supporting):**
  > "The structural (edge) guidance in Stage 3 is transmitted over a dedicated
  > edge JSCC link whose codec is trained separately with a BCE + Dice
  > objective; the received edge representation is aligned to the diffusion
  > latent and supplied to the ControlNet branch."

- **Shared-VAE ablation:**
  > "As an ablation we replace the dedicated edge link with a shared-VAE
  > encoding of the edge map, which removes the separate edge transmission path."

- **End-to-end fine-tuning (extension, NOT baseline):**
  > "Beyond the baseline, we additionally explore an optional end-to-end
  > fine-tuning step that jointly adapts the JSCC and diffusion models; this is an
  > extension reported separately from the core baseline."

## 데이터 입력 확장 — caption 생성 / multi-caption / file-list

기존 데이터 로더(`data/datasets.py`)에 **backward-compatible**한 3개 확장을 추가했다.
기존 `image` / `text_image` / `text_image_edge` / `edge` 경로와 `sidecar` 캡션,
`canny` 엣지는 그대로 동작한다.

### (1) caption 생성 — image-only 폴더를 text stage용으로 승격

CelebA / CelebA-HQ처럼 **캡션이 없는** 폴더는 text_dm/controlnet에 바로 못 쓴다.
`scripts/generate_captions.py`가 각 이미지 옆에 `<stem>.txt`를 만들어 기존
`caption_source: sidecar` 경로로 학습 가능하게 한다(로더 수정 없음).

```bash
# fixed 템플릿 (CPU) — 가장 단순
python scripts/generate_captions.py --input data/celeba/train --mode fixed \
    --text "a portrait photo of a person"
# filename 기반 (스모크용)
python scripts/generate_captions.py --input data/celeba/train --mode filename
# BLIP-2 모델 캡션 (GPU 권장) — guidance/text_extractor 재사용
python scripts/generate_captions.py --input data/celeba/train --mode model --device cuda:0
```

| mode | 의존성 | 충실도 |
|------|--------|--------|
| `fixed` | 없음(CPU) | **paper-like (not paper-faithful)** — placeholder 템플릿 |
| `filename` | 없음(CPU) | placeholder (스모크용) |
| `model` | transformers+weights(+GPU) | BLIP-2 근사 — 논문 원본 캡션과 동일하지 않음 |

> ⚠️ **충실도 경고**: 자동 생성 캡션은 논문의 CelebA-HQ 캡션과 동일하지 않다.
> 이 기능은 caption-less 데이터에서 **text stage를 가능하게** 하는 것이지,
> 논문 캡션을 bit-exact 재현하는 것이 아니다.

예시 config: [`composed_train_text_dm_celeba.yaml`](../configs/composed_train_text_dm_celeba.yaml).

### (2) COCO multi-caption — `coco_json` / `multi_manifest`

기존 단일 `sidecar` 캡션은 COCO JSON의 이미지당 5개 캡션 정보를 버린다. 새
`caption_source`로 multi-caption을 보존한다(원 SGD-JSCC 코드의 COCO 소비 방식에 정렬).

```yaml
train:
  dataset:
    caption_source:   coco_json   # 또는 multi_manifest (JSON {filename: [captions]})
    caption_path:     ../data/coco/annotations/captions_train2017.json  # train
    val_caption_path: ../data/coco/annotations/captions_val2017.json    # val (다른 파일!)
    caption_select:   random      # first | longest | random
```

- `caption_select`: `first`(caps[0]) / `longest`(가장 긴 캡션) / `random`(접근마다 무작위).
- **train/val 분리**: COCO는 train/val 캡션 JSON이 별도다. `caption_path`(train)와
  `val_caption_path`(val)를 따로 지정하면 val 이미지가 자기 캡션 JSON에서 해석된다.
  `val_caption_path` 미지정 시 val도 `caption_path`를 쓰므로(=train json), COCO처럼
  train/val이 다른 파일이면 **반드시 `val_caption_path`를 줘야** val 캡션이 맞는다.
- **재현성**: val 로더(`training=False`)는 `random`을 자동으로 `first`로 강등한다.
- 기존 `sidecar` / `manifest` / `filename`은 그대로 유지(단일 캡션). 충돌 없음.

예시 config: [`composed_train_text_dm_coco_json.yaml`](../configs/composed_train_text_dm_coco_json.yaml).

### (3) file-list 입력 — 대규모 데이터 운영

폴더 재귀탐색 대신 **이미지 경로 리스트**로 dataset을 구동(DiffJSCC식 train.list/val.list).
모든 dataset type(`image`/`text_image`/`text_image_edge`/`edge`)에서 동작하며,
caption/edge 매칭 규칙은 동일(sidecar `.txt`, `<stem>_edge.png`, on-the-fly canny).

```yaml
train:
  dataset:
    input_mode: file_list                       # folder(기본) | file_list
    file_list_path:     ../data/celeba/train.list
    val_file_list_path: ../data/celeba/val.list  # val 로더용(없으면 val 생략)
```

- 리스트는 한 줄당 이미지 경로(`#` 주석·빈 줄 무시). **상대경로는 리스트 파일 기준**,
  절대경로는 그대로. `--train-list` 없이도 동작(이 경우 `train_input_path` 불필요).
- 폴더 재귀탐색(기본, `input_mode: folder`)은 그대로 유지.

예시 config: [`composed_train_jscc_filelist.yaml`](../configs/composed_train_jscc_filelist.yaml).

### (4) CC3M WebDataset → training pair 변환 (`scripts/prepare_cc3m.py`)

대규모 text stage(stage 2/3) 학습 데이터로 `pixparse/cc3m-wds`를 쓴다. 다만
데이터 로더(`data/datasets.py`)는 **WebDataset `.tar` shard를 직접 읽지 못하고**
`이미지 + 같은 stem의 `.txt` 캡션 sidecar` 폴더만 소비한다. CC3M-WDS shard는 이미
`{key}.jpg` + `{key}.txt`(캡션) + `{key}.json`(메타) triple을 담고 있으므로,
**재캡셔닝 없이 추출 + stem 충돌 제거만** 하면 sidecar 포맷이 된다.

```bash
# 1) 어떤 결과가 나올지 먼저 dry-run (디스크에 아무것도 안 씀)
python scripts/prepare_cc3m.py --split train --limit-shards 2 --dry-run

# 2) train shard 일부만 변환 (현실적 부분 변환 — 전체 575 shard를 한 번에 X)
python scripts/prepare_cc3m.py --split train --limit-shards 8
python scripts/prepare_cc3m.py --split val   --limit-shards 1

# 3) 총 샘플 수 상한 (빠른 stage-2 smoke set 만들 때)
python scripts/prepare_cc3m.py --split train --limit-shards 8 --max-samples 2000

# 4) 경로/글롭/덮어쓰기 커스텀
python scripts/prepare_cc3m.py \
    --shard-dir data/cc3m_wds --output-dir data/cc3m_pairs \
    --split train --shard-glob 'cc3m-train-00*.tar' --overwrite
```

산출물(저장소 학습 포맷 그대로). **기본 레이아웃은 shard별 하위폴더**다 —
전체 CC3M(~290만 쌍 ≈ 580만 파일)을 한 폴더에 평평하게 쌓으면 inode/파일시스템
성능이 무너지므로, shard당 하위폴더(`<split>/<shard_tag>/`)로 분산한다. 로더는
재귀 스캔(`_list_images` → `rglob`)이라 split 루트만 `--train-list`로 주면
하위폴더를 투명하게 읽는다:

```
data/cc3m_pairs/
├── train/
│   ├── train0000/                       # shard별 하위폴더 (기본)
│   │   ├── train0000_000000000.jpg
│   │   └── train0000_000000000.txt      # 캡션 sidecar (stem 일치)
│   ├── train0000/.shard_done            # sequential 모드 완료 마커(로더엔 안 보임)
│   └── train0001/ …
└── val/
    └── val0000/ …
```

소수 shard만 다룰 땐 `--flat`으로 평평한 단일 폴더 출력도 가능하다
(`<split>/train0000_000000000.jpg`). full-scale에선 **쓰지 말 것.**

| 인자 | 의미 |
|------|------|
| `--shard-dir` | shard 디렉터리 (기본 `data/cc3m_wds`, 상대경로는 repo root 기준) |
| `--output-dir` | 출력 루트 (기본 `data/cc3m_pairs`; pair는 `<out>/<split>/`에) |
| `--split train\|val` | 기본 glob(`cc3m-train-*` / `cc3m-validation-*`)과 하위폴더 선택 |
| `--shard-glob` | glob 직접 지정 (split 기본값 덮어씀) |
| `--limit-shards N` | 앞에서 N개 shard만 처리 (부분 변환) |
| `--max-samples N` | 누적 N pair 작성 후 중단 (sequential 모드와는 비호환) |
| `--flat` | shard 하위폴더 대신 split 루트에 평평하게 출력 (소수 shard 전용; full-scale·sequential 금지) |
| `--overwrite` | 비어있지 않은 출력 split을 **재생성**(쓰기 전에 split 폴더를 비움) |
| `--append` | **sequential append**: split을 비우지 않고 새 shard만 이어서 추가(이미 변환된 shard는 skip). shard별 임시 디렉터리 staging → 원자적 commit |
| `--delete-shard-on-success` | 각 shard 출력이 **commit 검증된 뒤에만** 원본 `.tar` 삭제(디스크 절약). sequential 경로 활성화 |
| `--fail-on-existing` | sequential 모드에서 이미 변환된 shard(또는 마커 없는 기존 출력)를 만나면 skip 대신 **중단**(엄격 재실행 감지) |
| `--rebuild-unmarked` | sequential 모드에서 마커 없는 기존 출력(legacy/`--overwrite` 결과)을 adopt하지 않고 **재빌드**. 기본은 adopt(마커 스탬프 후 skip) |
| `--tmp-dir DIR` | sequential staging 위치(기본 `<output-dir>/.cc3m_tmp_<split>`; split 밖·동일 FS 권장) |
| `--dry-run` | 스캔/카운트만, 파일 미작성·tar 미삭제. **sequential `--dry-run`은 실제 run과 동일하게** 완료 shard를 skip으로 미리보기 |

- **shard별 하위폴더(기본)**: 디렉터리당 한 shard(~5k 쌍)만 두어 full-scale에서
  단일 폴더에 수백만 파일이 몰리는 inode/파일시스템 병목을 피한다. 로더가 재귀
  스캔하므로 split 루트 경로만 주면 된다.
- **stem 충돌 방지**: CC3M의 key(`000000000`…)는 **shard마다 reset**되어 모든
  shard에 `000000000`이 있다. shard 태그(`train0000_`)를 prefix해 `(shard,key)`를
  전역 유일하게 만든다 — 하위폴더든 `--flat`이든 이미지/캡션 stem은 일치한다.
- **건너뛰기 + 카운트**: 깨진 이미지(PIL `verify`)·이미지 없음·캡션 없음/공백
  샘플은 건너뛰고 요약 로그에 개수를 남긴다(치명적 에러 아님). `.txt`가 비면
  `.json`의 `caption` 필드로 폴백한다.
- **비파괴 + 재생성**: 비어있지 않은 출력 split은 `--overwrite` 없이는 거부한다.
  `--overwrite`는 "추가"가 아니라 **재생성**이다 — 쓰기 전에 `<output>/<split>/`을
  통째로 비우므로, 더 작은 `--max-samples`로 재실행하면 데이터셋이 그만큼 **줄고**,
  flat↔하위폴더 레이아웃이 한 split에 **혼재하지 않는다**(혼재 시 재귀 로더가
  같은 stem을 중복 카운트해 분포를 왜곡한다). 원본 shard는 절대 건드리지 않는다.

##### 세 가지 모드 — convert / regenerate / sequential append

| 모드 | 트리거 | 비어있지 않은 split | 파괴적 동작 | 용도 |
|------|--------|--------------------|-------------|------|
| **convert** (기본) | (없음) | **거부** | 없음 | 빈 split에 일괄 변환 |
| **regenerate** | `--overwrite` | **비우고** 재작성 | split 폴더 wipe | 전체 재생성·레이아웃 교체 |
| **sequential append** | `--append` / `--delete-shard-on-success` | **유지**하고 이어붙임 | (옵션 시) 성공한 shard의 `.tar` 삭제 | full-scale을 디스크 압박 없이 순차 변환 |

전체 CC3M를 풀면 tar(~수백 GB)와 변환 결과가 동시에 디스크를 점유한다.
**sequential append + delete**는 "shard 1개 변환 → 검증 → 그 tar 삭제"를 반복해
이 압박을 없앤다. 안전장치:

- **원자적 commit**: 각 shard는 split 바깥 staging
  (`<output>/.cc3m_tmp_<split>/<tag>.incoming/`)에 먼저 추출되고, 전체가 성공한
  뒤에만 완료 마커(`.shard_done`)와 함께 `<split>/<tag>/`로 **원자적 rename**된다.
  따라서 중간에 죽어도 split 안에는 **반쪽짜리 shard가 절대 안 남는다**(staging은
  split 밖이라 학습 로더에도 안 잡힌다). 다음 실행이 stale staging을 정리하고 그
  shard를 재시도한다.
- **tar 삭제는 commit 검증 후에만**: `.tar` 삭제는 출력이 원자적으로 commit된
  *뒤*에만 일어난다. 실패한 shard의 tar는 보존되고 요약에 `shards failed`로 집계,
  종료코드도 0이 아니다(성공한 shard는 그대로 유지).
- **재실행 멱등성**: `--append`는 `.shard_done`이 있는 shard를 skip하므로 같은
  명령을 반복하면 남은 shard만 이어서 처리한다. 이미 변환된 shard의 tar가 아직
  있으면(예: 재다운로드) `--delete-shard-on-success`가 그 tar도 정리한다.
- **마커 없는 기존 출력 = adopt**: legacy/`--overwrite`로 만든(=마커 없는) 출력
  디렉터리를 sequential이 만나면, 기본적으로 **adopt**한다 — 그 안의 pair 수를 세어
  `.shard_done` 마커를 스탬프하고 skip한다(좋은 데이터를 재빌드하지 않음). 원자적
  commit은 마커 없는 *비어있지 않은* 디렉터리를 절대 안 남기므로, 이런 디렉터리는
  다른 경로(legacy/overwrite)에서 만든 **완전한** 출력으로 간주해도 안전하다.
  강제로 재빌드하려면 `--rebuild-unmarked`를 준다. **dry-run도 동일하게** adopt/skip을
  미리보기한다(용량·재개 계획이 실제 run과 일치).
- **금지 조합**: sequential은 `--flat`(shard 단위 skip/commit 불가)·`--max-samples`
  (shard를 중간에 끊으면 부분 출력을 done으로 표시/삭제할 위험)와 비호환이며
  실행 전에 명확한 에러로 막는다. 순차 실행의 분량은 `--limit-shards`로 조절한다.

> ⚠️ **안전 주의**: `--delete-shard-on-success`는 원본 tar를 **영구 삭제**한다.
> 재다운로드 비용이 크거나 원본을 보존해야 하면 이 플래그 없이 `--append`만 써서
> 변환만 하고 tar는 수동으로 정리하라. `--tmp-dir`는 출력과 **같은 파일시스템**에
> 두어야 rename이 원자적이다(다른 FS면 복사 후 교체로 폴백 — 여전히 안전하지만
> 느리다).

검증(이 저장소에서 실제 수행): `--limit-shards 2 --dry-run` → shard당 5046 pair
(15138 member / 3). 기본 하위폴더 레이아웃으로 2 shard(6000 쌍) 변환 시
`train/train0000/`·`train/train0001/`로 분산되고, split 루트(`/.../train`)를 준
`TextImageDataset`가 재귀로 6000개를 모두 로드(`image [3,128,128]` + 중첩 경로의
sidecar 캡션). `--overwrite` 재생성도 확인: 20쌍 출력 후 `--max-samples 5
--overwrite` 재실행 → 정확히 5쌍만 남고(잔존 파일 없음), 하위폴더 출력에
`--flat --overwrite`로 재실행해도 하위폴더가 사라져 레이아웃 혼재가 없었다.
**sequential append + delete**(임시 sandbox, 원본 미사용): 2 shard commit 후
해당 tar 2개 삭제, 재실행 시 done shard skip + 남은 shard만 처리,
손상 tar는 FAILED로 표시되고 tar 보존·split에 부분 출력 없음·종료코드 1,
`.shard_done` 마커는 로더에 안 잡힘 확인. **dry-run 정확도**: 1 shard commit 후
`--append --dry-run`이 그 shard를 `would SKIP (already converted)`로, 다음 shard만
`would commit`으로 미리보기(실제 run과 일치). **adopt**: 마커를 지운 출력에
`--append` 실행 시 재빌드 대신 `adopted existing output (stamped marker)` 후 skip,
`--rebuild-unmarked`면 재빌드.
시드 세트 `data/cc3m_pairs/{train,val}`는 **sequential 규약대로 마이그레이션**됨 —
`train/train0000/`(5046쌍)·`val/val0000/`(840쌍) 모두 `.shard_done` 마커를 갖춘
full-shard commit이라 그대로 `--append` 가능(재빌드 안 함). 원본 tar는 보존.

#### 전체 흐름 예시 (이 머신 현재 상태 기준 → Stage 3까지)

> 이 머신엔 이미 `outputs/checkpoints/text_dm_coco_json/latest.pth`(진행 중인
> Stage 2 체크포인트)가 있다. 따라서 **주 흐름은 "이어서 학습(resume)"** 이다.
> 아래 (2)가 기본 경로이고, "처음부터 새로 시작"은 그 아래 선택지로 둔다.

```bash
cd /home/sangukbae/ETRI/Semantic/sgdjscc_lab
conda activate ptest

# (0) CC3M-WDS는 이미 data/cc3m_wds/ 아래에 다운로드되어 있다고 가정.

# (1) shard 일부 → training pair 변환 (train 8 shard ≈ 4만 쌍, val 1 shard)
#     기본 shard별 하위폴더 레이아웃 → split 루트만 --train-list로 넘기면 된다.
python scripts/prepare_cc3m.py --split train --limit-shards 8
python scripts/prepare_cc3m.py --split val   --limit-shards 1

# (2) [주 흐름] Stage 2 — 기존 체크포인트에서 이어서 학습 (resume)
python scripts/train.py --config configs/composed_train_text_dm.yaml \
    --train-list data/cc3m_pairs/train --val-list data/cc3m_pairs/val \
    --device cuda:0 --max-steps 250000 \
    --resume outputs/checkpoints/text_dm_coco_json/latest.pth
#    └ (선택) 처음부터 새로 시작하려면 --resume 없이 동일 명령을 쓴다.

# (3) Stage 2 산출물 → 추론용 backbone export
#     (resume 산출물 디렉터리 그대로: text_dm_coco_json/best.pth)
python scripts/export_checkpoint.py --stage text_dm \
    --input outputs/checkpoints/text_dm_coco_json/best.pth \
    --output checkpoints/diffusion_backbone.pth

# (4) Stage 3 — edge ControlNet 학습 (같은 pair 폴더, edge는 on-the-fly canny)
python scripts/train.py --config configs/composed_train_controlnet.yaml \
    --train-list data/cc3m_pairs/train --val-list data/cc3m_pairs/val \
    --device cuda:0
python scripts/export_checkpoint.py --stage controlnet \
    --input outputs/checkpoints/controlnet/best.pth \
    --output checkpoints/diffusion_controlnet.pth
```

> **데이터를 늘리는 두 방법**:
> - **재생성**: `--limit-shards`를 키워(또는 생략=전체) `--overwrite`로 split을
>   비우고 다시 쓴다 — shard 수를 늘린 명령이 곧 새 전체 데이터셋이 된다(잔존 파일·
>   레이아웃 혼재 없음). 디스크에 tar+출력을 동시에 둘 여유가 있을 때.
> - **순차 추가(권장, full-scale)**: 디스크가 빠듯하면 아래처럼 shard를 하나씩
>   변환하며 성공한 tar를 바로 지운다. 같은 명령을 반복하면 남은 shard만 이어서
>   처리하므로 중단/재개가 자유롭다.
>   ```bash
>   # 한 번에 N개씩 끊어서(원하면 --limit-shards 생략 → 전체) — 안전하게 재개 가능
>   python scripts/prepare_cc3m.py --split train --append \
>       --delete-shard-on-success --limit-shards 50
>   python scripts/prepare_cc3m.py --split val   --append --delete-shard-on-success
>   ```
>   `--delete-shard-on-success`는 **commit 검증된 shard의 tar만** 지운다(실패 tar 보존).
>   원본 tar를 남기고 싶으면 `--delete-shard-on-success` 없이 `--append`만 쓴다.
> full-scale 변환은 기본 shard별 하위폴더 레이아웃을 유지하라(`--flat`/`--max-samples`는
> sequential과 비호환).

### (5) SA-1B tar → image-only 변환 (`scripts/prepare_sa1b.py`)

SA-1B(Segment Anything, `data/sa1b/raw/sa_*.tar`)는 이미지 + per-image **세그멘테이션
마스크(`.json`)** 로 구성되며 **caption이 없다**. 이 저장소의 어떤 학습 dataset도
세그멘테이션 마스크를 소비하지 않으므로(근거: 이미지 로더 `data/image_dataset.py::_list_images`
가 `_IMG_EXTS` 확장자만 재귀 스캔, 마스크 전용 dataset 부재), SA-1B는 **image-only**
코퍼스로만 사용 가능하다 → stage `jscc` / `csi_estimation` / `edge_codec`
(edge는 Canny 온더플라이라 마스크 불필요). text stage(`text_dm`/`controlnet`)에는
caption이 없어 **그대로는 사용 불가**(쓰려면 `generate_captions.py`로 caption 생성 필요).

```bash
# 미리보기 — 아무것도 쓰지/지우지 않음
python scripts/prepare_sa1b.py --dry-run
# 1개만 실제 변환(원본 tar 보존)
python scripts/prepare_sa1b.py --limit-shards 1
# 디스크 압박 하에서 full-scale: tar 한 개씩 순차 변환 + commit 검증 후 그 tar만 삭제
python scripts/prepare_sa1b.py --delete-shard-on-success
```

출력: `data/sa1b_images/{train,val}/<shard_tag>/*.jpg` (JSON 마스크는 버림). 로더가
재귀 스캔하므로 `--train-list data/sa1b_images/train`(val은 `.../val`)로 전달한다.

**디스크-안전 순차 처리** (~11G tar × 68, 디스크 여유 부족 대응): tar 한 개씩만 처리한다 —
임시 디렉터리에 추출(`.jpg`만, `.json` 드롭, 깨진 이미지 PIL `verify` 스킵) → 검증(이미지
≥1개 + 표본 이미지 1장 full decode) → 최종 shard 디렉터리로 **원자적 `rename`** →
`.shard_done` 마커 기록 → **그 다음에야** `--delete-shard-on-success` 시 원본 `.tar` 삭제.
따라서 원본 tar와 추출 이미지가 **동시에 쌓이지 않는다**. 같은 명령을 다시 실행하면 **resume**:
`.shard_done`이 있는 shard는 skip(그리고 `--delete-shard-on-success`면 그 tar 삭제),
임시 staging 잔재(`*.incoming`/temp)는 자동 정리한다. 반면 **마커 없는 최종 출력
디렉터리**는 기본적으로 **건드리지 않고 skip(tar 보존)** 한다 — commit된 shard는 항상
마커를 갖기 때문에, 마커 없는 디렉터리는 모호한 잔재(legacy/외부/부분)로 보고 자동 삭제하지
않는다. 그것을 비우고 다시 만들려면 `--rebuild-unmarked`를 명시한다.

| 플래그 | 의미 |
|--------|------|
| `--shard-dir` | 입력 tar 디렉터리 (기본 `data/sa1b/raw`) |
| `--output-dir` | 출력 루트 (기본 `data/sa1b_images`; `train/`+`val/` 생성) |
| `--delete-shard-on-success` | shard 출력이 **commit 검증된 뒤에만** 원본 `.tar` 삭제 |
| `--limit-shards N` | 이번 실행에서 처리할 tar 개수 상한 |
| `--val-every K` | split 규칙: `shard_number % K == 0`이면 val (기본 10). shard **이름** 기준이라 tar 삭제에도 안정적 |
| `--all-train` | 모든 shard를 train으로 |
| `--dry-run` | 결정만 출력, 쓰기/삭제 없음 |
| `--rebuild-unmarked` | 마커 없는 기존 출력 디렉터리를 **비우고 재생성**(미지정 시 그 shard는 skip + tar 보존) |

> ⚠️ `--delete-shard-on-success`는 원본 tar를 **영구 삭제**한다. commit 검증된 shard의
> tar만 지우며(실패 shard의 tar는 보존), 검증 전에는 절대 삭제하지 않는다. 원본을 남기려면
> 이 플래그 없이 실행한다.
>
> ⚠️ **실행 주체**: SA-1B tar가 root 소유이므로 이 스크립트는 **컨테이너 내부(root)에서
> 실행**해야 한다 — 그래야 tar 읽기 + 출력 디렉터리(`.sa1b_tmp` staging 포함) 쓰기 + tar
> 삭제가 모두 된다. 일반 호스트 계정(`wilco`)으로 돌리면 root 소유 `data/sa1b/raw` 아래
> staging/삭제에서 `PermissionError`가 나고, 출력도 root 소유라 권한이 어긋난다. 원격 검증도
> 컨테이너 root 셸 기준이다(호스트 `wilco` 셸 아님).

### stage별 데이터 적용 (요약)

| stage | sidecar | coco_json/multi_manifest | file_list | caption 생성 필요? |
|-------|---------|--------------------------|-----------|--------------------|
| `jscc`/`edge_codec`/`csi_estimation` | — (image-only) | — | ✅ | 불필요 (SA-1B는 `prepare_sa1b.py` 변환만) |
| `text_dm`/`controlnet` | ✅ | ✅ | ✅ | celeba는 필요(generate_captions); CC3M은 `prepare_cc3m.py` 변환만; SA-1B는 caption 없어 그대로 불가 |

테스트: [`tests/test_data_extensions.py`](../tests/test_data_extensions.py) — sidecar 회귀,
coco_json first/longest/random, file_list(절대/상대), caption 생성, controlnet canny+sidecar.

## 관련 문서

- [smoke_training.md](./smoke_training.md) — real-model smoke 학습(최소 검증) 가이드
- [etri_development_roadmap.md](./etri_development_roadmap.md) — Phase별 개발 계획
- [phase5.md](./phase5.md) — Phase 5 (채널 조건화 + 저지연 + SRS-v2)
- [dataset_status.md](./dataset_status.md) — git 추적되는 데이터셋 역할/매핑 기준 문서
- `python scripts/report_datasets.py` — 현재 머신의 실제 데이터 보유 상태를 `data/_reports/dataset_status.md`로 생성

## 운영 안정성 & 메모리 최적화 기능 (operational stability)

기존 학습 인프라(step-based checkpointing, `latest.pth`, mixed precision, grad
accumulation, exact resume)는 그대로 두고, 장시간 학습의 **운영 안정성**과 **메모리
절약**을 위해 추가된 기능들. 모든 신규 토글의 기본값은 보수적으로 **off**라 기존
재현성에 영향을 주지 않는다.

### 1. SIGINT/SIGTERM 안전 저장 (`training/interrupt.py`)

학습 루프가 `InterruptHandler`로 SIGINT(Ctrl-C)/SIGTERM(스케줄러·docker stop)을
가로챈다. 시그널을 받으면 핸들러는 **플래그만 설정**(async-signal-safe)하고, 루프는
batch/step 경계의 안전한 지점에서 이를 폴링해 `interrupt_latest.pth`를 저장한 뒤 정상
종료한다. 두 번째 시그널이 오면 이전 핸들러를 복구하고 재발생시켜 강제 종료를 허용한다.

- **저장 포맷**은 기존 `latest/best/epoch` 저장과 동일(`restore_runner_state`로 그대로
  resume 가능). `interrupt_latest.pth`(명시적 기록) + `latest.pth`(갱신)를 함께 쓰므로
  이후 `--resume latest`가 자연스럽게 이어받는다.
- **DDP**: 모든 rank가 핸들러를 설치(torchrun이 그룹 전체에 시그널 전달)하되, 파일은
  **rank 0만** 기록한다.
- epoch 모드에서 epoch 중간 인터럽트는 현재 epoch을 저장 → resume 시 다음 epoch부터
  이어감(일반 end-of-epoch `latest.pth`와 동일 semantics). step 모드는 `global_step`이
  정확해 exact resume.

### 2. Auto-resume (`train.resume: latest`)

`train.resume`에 **명시 경로**를 주면 기존과 동일하게 그 체크포인트를 이어받는다. 추가로
`"latest"`(또는 `"auto"`)를 주면 `checkpoint_dir` 아래 `latest.pth`(없으면
`interrupt_latest.pth`)를 자동 탐색해 resume한다. **아무것도 없으면 명확한 로그와 함께
fresh run으로 시작**(안전한 기본값 — 첫 실행이 “이어받을 게 없다”는 이유로 죽지 않게).
CLI: `--resume latest` / `--resume /abs/path.pth` 모두 지원.

### 3. Validation image logging (`train.val_images.*`, `training/val_images.py`)

scalar val loss 외에 **실제 생성 샘플**을 주기적으로 저장(rank 0 전용). 우선순위는
`controlnet` → `text_dm`(+ 보너스로 `jscc`). 패널은 `input | edge(있으면) | recon`
한 줄/샘플 구성으로 `<checkpoint_dir>/<output_subdir>/{step_XXXXXXX|epoch_XXXX}.png`에
저장된다(중첩 깊이와 무관하게 명확하도록 base를 `checkpoint_dir`로 둠).

- **검증 데이터 + eval 모드**: `val_loader`의 **첫 배치를 고정**해 매 로깅 시점 동일 샘플을
  추적하며, `runner.set_mode(False)` + `torch.no_grad`로 렌더한다(dropout/CFG label-dropout
  등 train/eval 민감 동작을 끔). `val_loader`가 없으면 현재 training 배치로 **degraded
  fallback**하되 여전히 eval 모드로 렌더한다.
- 재구성 결과는 runner가 이미 wiring한 콜러블(`encode_latent_fn`/`encode_text_fn`/
  `encode_edge_fn`/`scheduler`/학습 대상 denoiser) + **공유 VAE decode**를 재사용한
  **1-step f0 예측**이다(새 추론 프레임워크를 만들지 않음; 학습 objective와 정확히 일치).
- `enabled: false`가 기본. `every_steps>0`(step 모드) 또는 `every_epochs>0`(epoch 모드)로
  켜며 `num_samples`로 패널당 샘플 수를 조절한다.

### 4. Optional 메모리/성능 토글 (`training/perf.py`)

기존 mixed precision/grad accumulation과 충돌 없이 동작하며 dependency가 없으면 graceful
fallback + 명확한 로그를 남긴다(조용히 무시하지 않음). 기본 전부 off.

| config | 동작 | fallback |
|--------|------|----------|
| `train.use_8bit_adam` | bitsandbytes `AdamW8bit`로 optimizer state VRAM 절감 | 패키지 없으면 경고 후 `torch.optim.AdamW` |
| `train.gradient_checkpointing` | 학습 대상 모듈에 `gradient_checkpointing_enable()`/flag 적용 | hook 없는 모듈은 “NOT applied” 경고 |
| `train.use_xformers` | memory-efficient attention 활성/검증 | xformers import 불가 시 경고. **MDTv2 backbone은 이미** `xformers.ops.memory_efficient_attention`을 기본 사용하므로 이 토글은 주로 importability 검증/보고용 |

> 핵심 파일: `training/interrupt.py`, `training/perf.py`, `training/val_images.py`,
> 그리고 이들을 wiring하는 `pipelines/train_pipeline.py`(resume 해석·인터럽트 저장·val
> 로깅) + `training/stage_runners.py`(optimizer 빌드·perf 토글 적용).

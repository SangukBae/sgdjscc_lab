> [← docs index](./README.md)

# 학습 CLI — Stage-Aware Training Framework (3 core stages + E2E FT)

## 개요

`scripts/train.py`는 `sgdjscc_lab`의 학습 진입점이다. SGD-JSCC 논문
("Semantics-Guided Diffusion for Deep Joint Source-Channel Coding", Sec. VI
*Training Details*)의 **3-stage 학습 절차**를 구조적으로 재현하고, 부가로
**end-to-end 미세조정** 단계를 제공한다.

```
원본 이미지 → [stage 1: jscc]         JSCC 인코더/디코더 (고정 AWGN SNR=10dB)
            → [stage 2: text_dm]      text-guided latent DM (f0 예측)
            → [stage 3: controlnet]   edge ControlNet 브랜치 (base DM frozen)
            → [확장: end_to_end_ft]   JSCC↔DM 공동 미세조정 (부가 단계)
```

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

# Stage 3 — edge ControlNet (shared_vae 엣지 transport)
python scripts/train.py --config configs/composed_train_controlnet.yaml \
    --train-list /data/pairs/train/ --device cuda:0

# Stage 3 (대안) — 전용 edge-JSCC transport로 비교 실험
python scripts/train.py --config configs/composed_train_controlnet_edge_jscc.yaml \
    --train-list /data/pairs/train/ --device cuda:0

# 확장 — end-to-end JSCC↔DM 미세조정 (step-based, grad-accum, AMP)
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

`--stage {jscc|text_dm|controlnet|end_to_end_ft}`로 `train.stage`를,
`--max-steps`로 `train.max_steps`를 덮어쓸 수 있다. 잘못된 stage 설정(누락된
caption/edge source, 잘못된 edge_transport, 학습 대상 0개 등)은 **체크포인트
로딩 이전에** 명확한 예외로 실패한다.

---

## stage ↔ 코드 매핑

| Stage | `train.stage` | Dataset | Forward (runner) | Loss | 학습 대상 (freeze 정책) |
|-------|---------------|---------|------------------|------|------------------------|
| 1 | `jscc` | `ImageOnlyDataset` | `JSCCStageRunner`: VAE encode → AWGN(SNR=10) → VAE decode | `JSCCStageLoss` = MSE (+ λ·GAN) | JSCC 학습, DM/guidance frozen |
| 2 | `text_dm` | `TextImageDataset` | `TextDMStageRunner`: `f0=VAE(x)`, `t~U(0,1)`, `f_t`=sigmoid schedule, masked/unmasked 예측 | `DiffusionF0Loss` = `‖f0−ε(f_t)‖²` + `‖f0−ε(f̂_t)‖²` | base DM 학습, JSCC frozen |
| 3 | `controlnet` | `TextImageEdgeDataset` | `ControlNetStageRunner`: stage2 forward + edge 조건 `c` (transport 선택) | `DiffusionF0Loss` (동일) | **ControlNet 브랜치만** 학습, base DM frozen (강제) |
| 확장 | `end_to_end_ft` | `TextImage[Edge]` | `EndToEndFTStageRunner`: encode→channel→1-step DM denoise→decode | `EndToEndFTLoss` = `w_r·‖x−x̂‖²` + `w_d·‖f0−ε(f_t)‖²` | JSCC/DM/ctrl 선택적 공동 학습 |

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
| Stage 3 edge transport `shared_vae` | ✅ full(구조) | 이미지 VAE stand-in (기존 방식) |
| Stage 3 edge transport `edge_jscc` | ✅ 구조 full / ⚠️ 가중치 stand-in | 전용 edge encoder→channel→projector (`models/edge_jscc.py`). codec 가중치는 논문 BCE/Dice 미학습 |
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
2. **Stage 3 edge transport.** 두 모드를 제공한다.
   - `shared_vae`: 엣지 맵을 이미지 VAE로 인코딩(간단 stand-in, 기존 방식).
   - `edge_jscc`: 전용 edge encoder → 무선 채널 → latent projector로,
     논문의 "엣지를 자체 DeepJSCC 링크로 전송 후 latent 정렬" 구조와 일치
     (`models/edge_jscc.py`). **다른 점**: 이 codec의 가중치는 무작위 초기화된
     구조적 stand-in이며, 논문의 BCE+Dice로 학습된 엣지 codec이 아니다. 교체
     지점(인코더/projector 또는 학습된 가중치 로드)은 코드로 분리되어 있다.
3. **end_to_end_ft (확장 단계).** 논문 부록은 DM을 먼저, 그다음 JSCC 디코더를
   **순차** 미세조정한다. 본 구현은 tractable하게 **공동(joint)** 으로 하며,
   전체 reverse diffusion 대신 **1-step denoise**(채널 SNR→noise level 매핑)로
   recon 항을 만든다. 즉 목적(최종 이미지 왜곡 최소화 + DM denoising 보존)은
   같지만 절차/스텝 수가 다르다.
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
    scheduler: { start: 0.0, end: 3.0, tau: 0.7 }

  controlnet:
    allow_unfrozen_base_dm: false   # ⚠ base DM unfreeze (논문 위반) 위험 플래그
    edge_transport: shared_vae      # shared_vae | edge_jscc
    edge_jscc: { base_ch: 64, norm: group, snr_db: 10.0, use_channel: true }

  end_to_end_ft:              # 확장 단계
    train_jscc: true
    train_diffusion: true
    train_controlnet: false
    snr_db: 10.0
    recon_weight: 1.0
    diff_weight: 1.0
```

stage별 composed config:
`configs/composed_train_jscc.yaml`, `composed_train_text_dm.yaml`,
`composed_train_controlnet.yaml`, `composed_train_controlnet_edge_jscc.yaml`
(edge-JSCC transport 비교), `composed_train_end_to_end_ft.yaml`(step-based +
grad-accum + AMP).

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
| `controlnet` | + `caption_source` + `edge_source` (sidecar면 `edge_dir`) + 유효한 `edge_transport` | `StageConfigError` |
| `end_to_end_ft` | + `caption_source`; 학습 대상 ≥1개; `train_controlnet=true`면 `dataset.type=text_image_edge`(auto면 자동 승격)+`edge_source` 필수 | `StageConfigError` |

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
| `scripts/train.py` | `--stage`(+e2e), `--max-steps`, 조기 검증 |
| `pipelines/train_pipeline.py` | 통합 global-step 루프 (epoch+step 모드, AMP/accum, resume) |
| `training/stages.py` | stage 상수·resolve·검증 (4 stage) |
| `training/noise_schedule.py` | `SigmoidNoiseScheduler` |
| `training/losses.py` | `JSCCStageLoss`, `DiffusionF0Loss`, `EndToEndFTLoss`, `PatchDiscriminator`+`build_discriminator`, `GANLoss` |
| `training/edge_transport.py` | `build_edge_transport` (shared_vae / edge_jscc) |
| `models/edge_jscc.py` | 전용 edge JSCC: encoder→channel→projector |
| `training/freeze.py` | `apply_stage_freeze_policy` (controlnet/e2e 정책) |
| `training/stage_runners.py` | JSCC/TextDM/ControlNet/**EndToEndFT** runner + AMP·grad-accum |
| `data/transforms.py`, `data/datasets.py` | 공통 transform + 3종 dataset |
| `configs/train/default.yaml` | step-based/GAN/edge_transport/e2e 필드 추가 |
| `configs/composed_train_{jscc,text_dm,controlnet,controlnet_edge_jscc,end_to_end_ft}.yaml` | stage별 composed config |
| `tests/test_train_stages.py` | stage/edge-transport/step/accum/e2e 단위 테스트 |

---

## 관련 문서

- [etri_development_roadmap.md](./etri_development_roadmap.md) — Phase별 개발 계획
- [phase5.md](./phase5.md) — Phase 5 (채널 조건화 + 저지연 + SRS-v2)

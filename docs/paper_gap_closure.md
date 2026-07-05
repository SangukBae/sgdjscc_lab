> [← 문서 색인](./README.md)

# Paper gap closure — `sgdjscc_lab`를 SGD-JSCC 논문에 정렬

대상 논문: **"Semantics-Guided Diffusion for Deep Joint Source-Channel Coding
in Wireless Image Transmission"** (Zhang et al., arXiv:2501.01138).

**무엇이 paper-faithful / paper-like / unsupported인지**, 그리고 `paper_mode`
guardrail이 재현 경로를 어떻게 강제하는지에 대한 single source of truth.

## 충실도 분류

- **paper-faithful** — *같은 데이터·checkpoint 하에서* 논문 / 공개 `SGDJSCC` 코드와
  (수치적으로 또는 구조적으로) 일치.
- **paper-like** — 의도 / 수식은 동일하나 명시·비명시 세부 하나가 다름(논문이 생략한
  값, 단순화한 모듈 등).
- **unsupported** — 여기서 재현 불가(비공개 데이터 / 가중치 / 세부); faithful로
  위장하지 못하도록 **guard**됨.

## `paper_mode` (핵심 축)

`paper_mode: true`(top-level config; 기본 `false`)는 `paper_mode.py`가 논문 config를
**강제**하고 non-faithful 대체물을 **차단**하게 하며, *어떤 checkpoint 로드보다 먼저*
`PaperModeError`를 raise한다. 두 CLI에 hook됨: `scripts/train.py`
(`paper_mode.enforce`), `scripts/evaluate.py`(`enforce_eval`). 강제 항목:

- **caption** — *알려진* auto-generated caption(`_AUTOCAPTION_PROVENANCE.json`
  sentinel 경유)과 `filename` source를 차단. 손으로 둔 `sidecar`/`manifest` `.txt`가
  논문 caption set인지는 **검증하지 않음**(경고와 함께 신뢰) — remaining gap 참조;
- MuGE soft edge (Canny 없음); `edge_jscc` transport (`shared_vae` 없음);
- learned CFG null token (zero-vector 없음); multi-SNR SNR-conditioned edge codec;
- **(eval)** 모든 확장 비활성(Phase 4/5, packet, regeneration, `shared_vae`), 그리고
  지표 set이 논문 보고 set과 **정확히** 일치해야 함(`PAPER_METRICS` =
  PSNR/LPIPS/CLIP×2/FID), `--profile`/`--no-clip` *이후* 강제 — 따라서 `paper_mode`
  하에서 축소 set이나 `--no-clip`은 거부됨.

`paper_mode`는 어떤 확장도 **삭제하지 않는다** — 모든 경로는 `paper_mode: false`에서
그대로 동작. 논문 경로는
`configs/paper_train_{jscc,text_dm,edge_codec,controlnet}.yaml` + `paper_eval_awgn.yaml`에
번들됨.

## 항목별 상태 (8개 task)

| # | 항목 | 상태 | 비고 |
|---|------|--------|-------|
| 1 | Stage-3 MuGE edge (Canny 아님) | **paper-like → faithful structure** | `edge_source: muge_sidecar`(precompute `scripts/prepare_muge_edges.py`). Repr: `reduced`(1ch), `edge_uncertainty`(2ch, inference에 가장 근접), `multi`(11ch). 논문 config 기본 `edge_uncertainty`; paper_mode에서 Canny 차단. |
| 2 | CelebA auto-caption 차단 | **partial (정직한 guardrail)** | provenance sentinel이 *알려진* auto-caption + `filename` 차단. 손으로 둔 sidecar는 논문 caption임을 *증명 불가* — 경고와 함께 신뢰. |
| 3 | Learned CFG null token | **done** | `dm.cfg_null_mode: zero\|learned`; `learned`은 학습 가능한 `LearnedNullToken` 추가(1회 등록, checkpoint화, resume-safe). paper_mode는 `learned` 요구. |
| 4 | Edge-JSCC structure 재사용 | **unsupported (guarded) + 재현 가능한 최근접** | 정확한 `model_canny.py` 재사용은 비공개 edge-codec 가중치 필요; `arch='paper'`는 `NotImplementedError`. 최근접은 `arch='vit'`(adaLN SNR-conditioned, WITT-location-faithful), 논문 config가 사용. |
| 5 | Edge-codec SNR conditioning | **done** | `edge_codec.multi_snr.{enabled,min_db,max_db}`가 edge-link SNR sampling → `EdgeJSCC.reconstruct(snr_db=…)` → adaLN. paper_mode가 요구. |
| 6 | Stage-3 train/infer edge 경로 통합 | **partial (논문 기본)** | paper_mode가 `edge_jscc` 강제, `shared_vae` 차단; codec `in_ch`는 `muge_repr`에서 유도(dataset→codec→transport 정렬). Bit-exact train≡infer는 **주장 안 함**(inference는 canny-transmission/VAE 경로 사용). |
| 7 | Complex phase / joint CSI (Alg. 3) | **partial (faithful layer, unsupported e2e)** | `channels/complex_ops.py`: complex 채널, 2-step equalization(`e^{-jφ̂}` 후 `/√(\|h\|²+σ²)`), phase/SNR 교대 loop. 공개 JSCC는 **real** latent 출력 → e2e complex는 비공개 retrain 필요. |
| 8 | Paper-only config 번들 | **done** | `configs/paper_*` + eval `paper_mode` 강제(`enforce_eval` + `enforce_eval_metrics`). |

## 남은 논문 비등가성 (정직하게)

- **Caption**: paper_mode는 *알려진* auto-caption + `filename`을 차단하나 손으로 둔
  sidecar는 검증 불가. 가장 강한 보장은 본질적으로 데이터셋 제공 source(COCO
  `coco_json`). 논문의 정확한 caption set은 재현하지 못함.
- **Edge JSCC**: 공개 가중치 없음 → 여기서 학습(`edge_codec`), `vit` adaLN 구조이며
  정확한 WITT 모듈 아님.
- **Stage-3 edge 경로**: `edge_uncertainty`가 더 가깝지만 원본 inference 경로와 정확히
  같지는 않음(`canny_transmission_net` + image-VAE를 training에서 재사용 안 함).
- **Complex transport**: complex layer + estimator는 올바르나 real-valued 공개 JSCC
  forward에 배선되지 않음. End-to-end complex는 unsupported.
- 논문이 생략한 **hyperparameter**(CFG dropout, LR schedule, GAN/LPIPS weight)는
  paper-like 기본값이며 config로 override 가능.

## 검증

```bash
cd sgdjscc_lab && conda activate ptest
python -m pytest tests/test_paper_mode.py -q      # 논문 config 로드 + smoke test
python -m pytest tests/ -q                          # 전체 suite

# 논문 config dry-run (GPU/checkpoint 불필요):
python scripts/train.py --config configs/paper_train_jscc.yaml \
    --train-list data/imagenet/train --no-models --epochs 1

# inference-aligned MuGE repr precompute (2ch edge+uncertainty):
python scripts/prepare_muge_edges.py --input data/coco/train2017 \
    --model-root ../checkpoints --repr edge_uncertainty

# eval paper_mode 강제 (확장 OFF → 하나라도 켜지면 hard exit):
python scripts/evaluate.py --config configs/paper_eval_awgn.yaml \
    --input data/kodak --snr 10
```

## Multi-GPU training (DDP)

`torchrun` 기반 PyTorch DDP; single-process / CPU 경로는 불변(모든 DDP helper는
no-op으로 degrade). `train.batch_size`는 **per-rank**:
`global_batch = batch_size × world_size × grad_accum_steps`(paper-like 64를 3 GPU에서
쓰려면 `batch_size≈21`).

```bash
torchrun --standalone --nproc_per_node=3 scripts/train.py \
    --config configs/paper_train_text_dm.yaml \
    --train-list data/coco/train2017 --val-list data/coco/val2017 --batch-size 21
```

**상태:**
- **Stage 2 (`text_dm`) — 지원 & 검증됨.** DDP-wrap된 denoiser + learned CFG null
  token gradient-sync; world_size=2 Gloo CPU smoke와 3×GPU box(NCCL)로 검증.
- **Stage 3 (`controlnet`) — 구조 준비됨.** 동일 배선;
  `find_unused_parameters=False`(기본; base DM frozen, ControlNet branch만 DDP-wrap
  되어 학습). multi-GPU end-to-end 검증은 미완.
- **Stage 1 (`jscc`) / `edge_codec` — DDP 미검증.** generic 배선이나 GAN 경로와
  self-contained codec은 DDP에서 미검증.

주요 파일: `distributed.py`(helper), `scripts/train.py`(torchrun init/cleanup),
`data/datasets.py`(DistributedSampler), `train_pipeline.py`(rank0 save/log,
`set_epoch`, sample-weighted validation 평균 + global `best.pth` 지표),
`stage_runners.py`(DDP-wrap denoiser, `no_sync` grad-accum, eager DDP-safe null
token). Export + evaluation은 single-process 유지. `DistributedSampler` padding
중복에서 잔여 bias가 남음. Test: `tests/test_ddp.py`.

## 파일 (요약)

- **신규**: `paper_mode.py`, `distributed.py`, `channels/complex_ops.py`,
  `prepare_muge_edges.py`, `configs/paper_*.yaml`, `tests/test_{paper_mode,ddp}.py`.
- **수정**: `training/{stages,stage_runners}.py`, `data/datasets.py`,
  `models/edge_jscc.py`, `training/edge_transport.py`, `pipelines/train_pipeline.py`,
  `scripts/{generate_captions,train,evaluate}.py`, `configs/train/default.yaml`.

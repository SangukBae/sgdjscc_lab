# Paper gap closure — aligning `sgdjscc_lab` with the SGD-JSCC paper

Target paper: **"Semantics-Guided Diffusion for Deep Joint Source-Channel Coding
in Wireless Image Transmission"** (Zhang et al., arXiv:2501.01138).

This document is the single source of truth for **what is paper-faithful, what is
paper-like, and what is unsupported** in `sgdjscc_lab`, and how the `paper_mode`
guardrails enforce the reproduction path.

## Fidelity taxonomy
- **paper-faithful** — matches the paper / public `SGDJSCC` code (numerically or
  structurally) *given the same data and checkpoints*.
- **paper-like** — same intent / formula family, but a stated-or-unstated detail
  differs (a value the paper does not give, a simplified module, …).
- **unsupported** — cannot be reproduced here (non-public data / weights /
  details); **guarded** so it cannot masquerade as faithful.

## `paper_mode` (the keystone)
`paper_mode: true` (top-level config; default `false`) makes
`src/sgdjscc_lab/paper_mode.py` **enforce** the paper config and **block**
non-faithful stand-ins, raising `PaperModeError` *before any checkpoint loads*.
It is hooked into **both** CLIs: `scripts/train.py` (training, after
`validate_stage_config` → `paper_mode.enforce`) and `scripts/evaluate.py`
(evaluation → `paper_mode.enforce_eval`). It enforces:
- **captions** — blocks *known* auto-generated captions (via the
  `_AUTOCAPTION_PROVENANCE.json` sentinel) and the `filename` pseudo-source.
  ⚠️ It does **not** verify that a hand-placed `sidecar`/`manifest` `.txt` is the
  paper's caption set (those are *trusted*, with a warning) — see item [2];
- MuGE soft edges (no Canny);
- `edge_jscc` transport (no `shared_vae` ablation);
- learned CFG null token (no zero-vector);
- multi-SNR, SNR-conditioned edge codec;
- **(eval)** every extension feature disabled (Phase 4/5, packet, regeneration,
  `shared_vae`) via `enforce_eval`, **and** the metric set must be **exactly**
  the paper's full reported set (`PAPER_METRICS` = PSNR/LPIPS/CLIP×2/FID),
  enforced via `enforce_eval_metrics` *after* `--profile`/`--no-clip` are applied
  — so `--profile extended`, `--no-clip`, *or a reduced set* (e.g.
  `metrics: [psnr, lpips]` missing CLIP/FID) under `paper_mode` is rejected.

`paper_mode` **does not delete** any extension — every non-faithful path still
works with `paper_mode: false`. The paper path is bundled in
`configs/paper_train_{jscc,text_dm,edge_codec,controlnet}.yaml` and
`configs/paper_eval_awgn.yaml`.

## Per-item status (the 8 tasks)

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Stage-3 edge input MuGE (not Canny) | **paper-like → faithful structure** | New `edge_source: muge_sidecar` (precompute via `scripts/prepare_muge_edges.py`, reusing the inference `guidance/edge_extractor`) + `muge_runtime`. MuGE representations are now explicit: `reduced` (1ch legacy), `edge_uncertainty` (2ch `[mean-edge, uncertainty]`, **closest to inference**), `multi` (11ch opt-in). paper configs default to `muge_sidecar + muge_repr=edge_uncertainty`; Canny is ablation-only and **blocked in paper_mode**. The 11→1 mean reduction still exists as the backward-compatible `reduced` path (`data/datasets.muge_reduce`). |
| 2 | Block CelebA auto-captions in paper_mode | **partial (honest guardrail)** | `generate_captions.py` writes an `_AUTOCAPTION_PROVENANCE.json` sentinel; paper_mode refuses to train a text stage on dirs containing it, and blocks `caption_source: filename`. **Limit (by design):** a hand-placed `sidecar`/`manifest` `.txt` cannot be proven to be the paper's captions — those are *trusted* (with a runtime WARNING), not verified. The log/`summary()` say exactly this (no "dataset-provided only" overclaim). Message on auto-captions: "CelebA-HQ auto-generated captions are not paper-faithful". |
| 3 | Learned CFG null token (not zero) | **done** | `train.dm.cfg_null_mode: zero\|learned`; `learned` adds a trainable `LearnedNullToken` (registered with the optimizer exactly once, checkpointed). **Resume-safe**: `LearnedNullToken._load_from_state_dict` materialises the token from the checkpoint before the lazy-create on first forward, and the optimizer registration is flag-driven so a *resumed* token is also optimised (tested). paper_mode requires `learned`. |
| 4 | Edge-JSCC original structure reuse | **unsupported (guardrail) + closest reproducible** | Exact reuse of `SGDJSCC/models/model_canny.py` (WITT Swin + QAM + a hard-coded non-public VAE path) is **unsupported**: the public HF release ships **no edge-codec weights** and the WITT interface doesn't match this edge-latent geometry. `arch='paper'` raises an explicit `NotImplementedError`. The closest reproducible structure is `arch='vit'` (adaLN SNR-conditioned transformer, WITT-location-faithful), which the paper configs use; `conv`/`vit` remain as paper-like/legacy. |
| 5 | Edge-codec SNR conditioning active | **done** | `train.edge_codec.multi_snr.{enabled,min_db,max_db}` samples the edge-link SNR per step and feeds it through `EdgeJSCC.reconstruct(..., snr_db=...)` → adaLN. paper_mode requires `multi_snr.enabled` (and `vit.snr_cond` for the ViT arch). Fixed-SNR remains supported (`multi_snr.enabled: false`). |
| 6 | Unify Stage-3 train/infer edge path | **partial (paper default)** | paper_mode forces `edge_transport: edge_jscc` (dedicated edge link, like inference) and blocks `shared_vae`. Training now defaults to the inference-carried **2ch** MuGE representation (`edge_uncertainty`: mean-edge + uncertainty), and the edge codec `in_ch` is derived from `train.dataset.muge_repr` so dataset → codec → transport stay aligned. Full bit-exact train≡infer alignment is still **not** claimed: inference uses the original canny-transmission/VAE path, while training feeds the chosen MuGE representation directly into the trainable edge codec. |
| 7 | Complex phase / joint CSI (Alg. 3) | **partial (faithful layer, unsupported end-to-end)** | New `channels/complex_ops.py`: the paper C/R maps, complex channel `y=h·z+n`, **two-step equalization** (phase removal `e^{-jφ̂}` then `/√(|h|²+σ²)`), and an **alternating phase/SNR loop** that actually rotates a complex latent (unlike the real-gain `joint_csi_estimate` no-op). **Unsupported end-to-end**: the *public* JSCC emits a **real** latent and its channels are real-gain, so routing this through the frozen JSCC forward needs a non-public complex-JSCC retrain. Smoke-tested (roundtrip + zero-noise recovery + loop shapes). |
| 8 | Paper-only config bundle | **done** | `configs/paper_train_{jscc,text_dm,edge_codec,controlnet}.yaml` + `configs/paper_eval_awgn.yaml`: `paper_mode: true`, fixed AWGN 10 dB JSCC, continuous-timestep sigmoid DM, learned CFG null, MuGE edges, `edge_jscc` transport, extensions disabled. The eval config's `paper_mode` is now **enforced** by `scripts/evaluate.py`: `enforce_eval` rejects any enabled extension / `shared_vae`, and `enforce_eval_metrics` (called *after* `--profile`/`--no-clip`) rejects non-paper metrics and `--no-clip` — so "paper eval" actually runs the paper baseline with the paper metric set. |

## Remaining paper non-equivalence (honest list)
- **Captions**: the paper's exact CelebA-HQ / training captions are its own.
  paper_mode blocks *known* auto-captions + the `filename` source, but **cannot
  verify** that a hand-placed `sidecar`/`manifest` is the paper's caption set
  (trusted with a warning). For the strongest guarantee use an intrinsically
  dataset-provided source (COCO `coco_json`). It does not reproduce the paper's
  exact caption set.
- **Edge JSCC**: no public edge-codec weights → the edge link is trained here
  (`edge_codec` stage) with the `vit` adaLN structure, not the exact WITT module.
- **Stage-3 edge path**: closer than before (`edge_uncertainty` preserves the
  inference-carried uncertainty and keeps dataset/codec channel counts aligned),
  but still not the exact original inference path (`canny_transmission_net` +
  image-VAE encode is not reused in training).
- **Complex transport**: implemented as a correct complex layer + estimators, but
  not wired through the real-valued public JSCC forward (needs a complex-JSCC
  retrain). End-to-end complex transmission is unsupported.
- **Hyperparameters** the paper does not publish (CFG dropout prob, exact LR
  schedule, GAN/LPIPS weights) are paper-like defaults, config-overridable.

## Validation commands
```bash
cd sgdjscc_lab && conda activate ptest

# 1) every paper config loads + passes stage/paper-mode validation, plus the
#    new smoke tests (MuGE source/repr, learned null token, edge_jscc, complex CSI):
python -m pytest tests/test_paper_mode.py -q

# 2) full suite (no regressions):
python -m pytest tests/ -q

# 3) paper config dry-run (config/stage/paper-mode wiring; no GPU/checkpoints):
python scripts/train.py --config configs/paper_train_jscc.yaml \
    --train-list data/imagenet/train --no-models --epochs 1

# 4) paper_mode guardrail demo (a Canny/shared_vae controlnet config is REJECTED):
python scripts/train.py --config configs/composed_train_controlnet.yaml \
    --train-list data/coco/train2017 --no-models --epochs 1 \
    && echo "(this is the extension path; paper configs would block canny)"

# 5) precompute the inference-aligned MuGE training repr (2ch edge+uncertainty):
python scripts/prepare_muge_edges.py --input data/coco/train2017 \
    --model-root ../checkpoints --repr edge_uncertainty

# 6) eval paper_mode is enforced (extensions OFF) — the clean paper eval passes:
python scripts/evaluate.py --config configs/paper_eval_awgn.yaml \
    --input data/kodak --snr 10        # logs "paper_mode=ON (eval)"; any enabled
                                       # extension or shared_vae → hard exit.
```

## Multi-GPU training (DDP)

`sgdjscc_lab` supports **PyTorch DistributedDataParallel** via `torchrun`. The
single-process / CPU path is unchanged (every DDP helper degrades to a no-op).

**Status (honest):**
- **Stage 2 (`text_dm`) — supported & validated.** DDP-wrapped denoiser + the
  learned CFG null token are gradient-synced; verified by a world_size=2 Gloo CPU
  smoke (`tests/test_ddp.py`: param + null-token sync, DistributedSampler, rank0
  checkpoint) and on the remote 3×GPU box (NCCL).
- **Stage 3 (`controlnet`) — structure-ready.** Same runner plumbing
  (DDP-wrapped denoiser; `edge_transport` left unwrapped as fixed `no_grad` side
  info). It sets `find_unused_parameters=True` **conservatively** because the base
  DM is frozen and only the ControlNet branches train, so some wrapped-module
  params may receive no grad on a step (DDP would otherwise hang). This can be set
  False if profiling shows all trainable params always get a grad — not yet
  validated end-to-end on multi-GPU, hence "structure-ready", not "validated".
- **Stage 1 (`jscc`) / `edge_codec` — not DDP-validated.** The plumbing is generic
  but the GAN path (Stage 1) and the self-contained codec are untested under DDP.

**What changed for DDP:** `src/sgdjscc_lab/distributed.py` (helpers:
`setup_distributed`/`is_rank0`/`unwrap_module`/`reduce_metric_sums`/`all_reduce_grads`/
`maybe_set_epoch`); `scripts/train.py` (torchrun init/cleanup, `cuda:{LOCAL_RANK}`);
`data/datasets.py` (DistributedSampler when distributed); `train_pipeline.py`
(rank0-only checkpoint + JSONL log, `sampler.set_epoch`, validation metric via
sum+count all-reduce); `stage_runners.py` (DDP-wrapped denoiser called in forward;
grad-accum uses `no_sync` on non-boundary micro-steps + a grad all-reduce at the
epoch-boundary flush; **learned CFG null token rebuilt EAGER** — a real `nn.Module`
created at runner construction with the probed label shape, DDP-wrapped, and
optimizer-registered once, so it is no longer a rank-local lazy parameter).

**Batch size:** `train.batch_size` is **per-rank**.
`global_batch = batch_size × world_size × grad_accum_steps`. To keep a paper-like
global batch (e.g. 64) on 3 GPUs use `batch_size≈21–22` (or raise `grad_accum_steps`).

**Run (3 GPUs):**
```bash
torchrun --standalone --nproc_per_node=3 scripts/train.py \
    --config configs/paper_train_text_dm.yaml \
    --train-list data/coco/train2017 --val-list data/coco/val2017
# single-process is unchanged:  python scripts/train.py --config … --device cuda:0
# torchrun --nproc_per_node=1 … behaves exactly like the single-process path.
```
Export + evaluation stay single-process (DDP is training-only here).

**Remaining DDP limitations:** validation `DistributedSampler` pads the last
batch to a multiple of world_size (duplicate samples) — the sum+count all-reduce
makes this a negligible bias, not exact. `prepare_muge_edges.py` is not
DDP-parallelised (split the input folder for manual parallelism).

## Files changed (summary)
- **new**: `src/sgdjscc_lab/paper_mode.py`, `src/sgdjscc_lab/distributed.py`,
  `src/sgdjscc_lab/channels/complex_ops.py`, `tests/test_ddp.py`,
  `scripts/prepare_muge_edges.py`, `configs/paper_train_{jscc,text_dm,edge_codec,controlnet}.yaml`,
  `configs/paper_eval_awgn.yaml`, `tests/test_paper_mode.py`, this doc.
- **edited**: `training/stages.py` (MuGE edge sources + `muge_repr` validation),
  `data/datasets.py` (MuGE loading + extractor reuse + multi-channel reprs +
  DistributedSampler), `training/stage_runners.py` (learned null token →
  DDP-safe eager, multi-SNR edge codec, DDP-wrapped denoiser + no_sync grad-accum),
  `models/edge_jscc.py` (per-step SNR override, `arch='paper'` guardrail +
  multi-channel edge I/O), `training/edge_transport.py` (edge-codec `in_ch` from
  `muge_repr`), `pipelines/train_pipeline.py` (rank0 save/log, set_epoch, val
  all-reduce), `scripts/generate_captions.py` (provenance sentinel),
  `scripts/train.py` (paper_mode hook + torchrun init/cleanup/device),
  `scripts/evaluate.py` (paper_mode **eval** hook), `configs/train/default.yaml`
  (`paper_mode`, `cfg_null_mode`, `edge_codec.multi_snr`, `dataset.muge_repr`,
  per-rank `batch_size` note).

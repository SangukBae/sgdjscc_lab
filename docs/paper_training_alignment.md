# Paper training alignment — confirmed vs assumed hyperparameters

Scope: how `sgdjscc_lab` sets the SGD-JSCC training / inference hyperparameters,
separating **what is confirmed by the public `SGDJSCC/` code** from **what is a
paper-table value** and **what is an unpublished assumption**. This is the
companion to [paper_gap_closure.md](./paper_gap_closure.md) (which covers the
structural fidelity taxonomy and the `paper_mode` guardrails).

Ground-truth priority for this repo: **public `SGDJSCC/` code first**, paper
table second. Where they conflict, we keep the public-code value for
reproducibility and say so explicitly.

## Legend

- **confirmed (public code)** — a literal value in `SGDJSCC/` source/config.
- **paper table** — a value stated in the paper's table/text but NOT found in the
  public code (so treated as a target/scale, not a reproducible constant).
- **assumed / unpublished** — neither in the public code nor pinned by the paper;
  a reasonable default chosen here.

## 1. Confirmed from the public SGDJSCC code

| Item | Value | Source (public code) |
|---|---|---|
| `diffusion_step` | 50 | `SGDJSCC/configs/inference.yaml` |
| `guidance_scale` | 4.0 | `SGDJSCC/configs/inference.yaml` |
| `controlnet_scale` | 0.3 | `SGDJSCC/configs/inference.yaml` |
| `cfg_method` | `pcs_1.0` | `SGDJSCC/configs/inference.yaml` |
| backbone `hidden_size` | 512 | `SGDJSCC/inference_config.py:274` `MDTv2(depth=12, hidden_size=512, patch_size=1, num_heads=8)` |
| backbone `depth` / `num_heads` / `patch_size` | 12 / 8 / 1 | same line |
| ControlNet `copy_blocks_num` | 6 | `SGDJSCC/inference_config.py:278` `MDTv2_ControlNet(..., copy_blocks_num=6, hidden_size=512)` |
| timestep `frequency_embedding_size` | 256 | `SGDJSCC/models/test_advanced_network/mask_diffusion.py:153` `TimestepEmbedder(hidden_size, frequency_embedding_size=256)` |
| JSCC training SNR | 10 dB | paper Sec. VI + repo default (fixed AWGN) |

Where these live in `sgdjscc_lab`:
- `configs/model/sgdjscc.yaml` — `diffusion_step`, `guidance_scale`,
  `controlnet_scale`, `cfg_method` (kept at the public-code values).
- `src/sgdjscc_lab/models/diffusion_wrapper.py` — `MDTv2(depth=12,
  hidden_size=512, patch_size=1, num_heads=8)` and
  `MDTv2_ControlNet(..., copy_blocks_num=6, hidden_size=512)`. **Do not change**
  (checkpoint compatibility).

## 2. Conflicts between the public code and the paper table

| Item | Public code | Paper table | Repo choice |
|---|---|---|---|
| CFG scalar / `guidance_scale` | **4.0** | **4.5** | Keep **4.0** (public code) for reproducibility. Documented in `configs/model/sgdjscc.yaml`. |
| "embedding size" | backbone `hidden_size = 512`; timestep `frequency_embedding_size = 256` | "embedding size = 256" | Keep backbone **512**. The table's 256 most plausibly refers to the **timestep/noise embedding** (`frequency_embedding_size=256`), a *separate* quantity — so 256 is **not** evidence for a 256-d backbone. Documented in `diffusion_wrapper.py`. |

These are the two items where a naive reading of the paper table would silently
diverge from the runnable public code. We keep the code values.

## 3. Assumed / unpublished (reasonable defaults, NOT paper-confirmed)

| Item | Repo default | Status |
|---|---|---|
| `lr` | 1e-4 | assumed (typical AdamW latent-DM/DiT lr); unpublished |
| `weight_decay` | 1e-5 | assumed; unpublished |
| `cfg_dropout_prob` | 0.1 | assumed (PixArt convention); unpublished |
| CFG null token | `learned` (paper configs) / `zero` (default) | paper-like intent; the paper does not publish its null token |
| edge codec arch / dims (`vit`, embed 128, depth 4, heads 4) | as listed | closest-reproducible structure; exact WITT edge-codec reuse is **unsupported** |
| edge codec multi-SNR range | 0–20 dB | assumed range; unpublished |
| JSCC GAN weight λ | 0.5 (`paper_train_jscc_gan.yaml`) | paper-LIKE objective (MSE + λ·GAN); λ **assumed/unpublished** |
| DM stage step count | 250k (`paper_train_*`) | paper-table-scale target; not code-confirmed |
| batch size | 64 (`paper_train_*`) | paper-table-scale target; not code-confirmed |

These are set so the pipeline runs and is *structurally* faithful; they are
labelled honestly in the configs (`assumed default (unpublished in the paper)`)
and must not be cited as paper-confirmed.

## 4. Data scope: COCO-only vs the paper's multi-dataset setup

`paper_train_text_dm.yaml` and `paper_train_controlnet.yaml` use a **COCO-only**
caption source. The paper's Stage-2 DM is trained on a much larger (~14M-image)
**multi-dataset** corpus. The repo configs are therefore a **smaller practical
reproduction**, not the paper's exact data setup. This is called out in each
config header. (The `paper_mode` guardrails still enforce *dataset-provided*
captions — no auto-captions — but they cannot make COCO equal to the paper's 14M
corpus.)

## 5. DDP global batch on 3 GPUs

`train.batch_size` is **per-rank (per GPU)**. Under
`torchrun --nproc_per_node=N`:

```
global_batch = batch_size * world_size * grad_accum_steps
```

To approximate the paper-scale global batch of **64** on **3 GPUs**, use per-rank
`--batch-size 21` (21 × 3 = **63 ≈ 64**). Raise `grad_accum_steps` if you need to
hit exactly 64 or to fit memory. Single-process runs: `global_batch = batch_size`.

## 6. Final 3-GPU training command set

Per-rank `--batch-size 21` → global ≈ 63 (≈ paper 64). Adjust data paths to your
machine. Stage 2 → edge precompute → edge codec → ControlNet, with JSCC optional.

### Stage 2 — text-guided DM
```bash
torchrun --standalone --nproc_per_node=3 scripts/train.py \
  --config configs/paper_train_text_dm.yaml \
  --train-list data/coco/train2017 \
  --val-list data/coco/val2017 \
  --batch-size 21
```

### MuGE edge precompute (edge + uncertainty; run once per split)
```bash
python scripts/prepare_muge_edges.py \
  --input data/coco/train2017 \
  --model-root checkpoints \
  --repr edge_uncertainty \
  --device cuda:0

python scripts/prepare_muge_edges.py \
  --input data/coco/val2017 \
  --model-root checkpoints \
  --repr edge_uncertainty \
  --device cuda:0
```

### Edge codec (supporting step; produces the edge_jscc checkpoint)
```bash
torchrun --standalone --nproc_per_node=3 scripts/train.py \
  --config configs/paper_train_edge_codec.yaml \
  --train-list data/coco/train2017 \
  --val-list data/coco/val2017 \
  --batch-size 21
```

### Stage 3 — edge ControlNet (loads the Stage-2 DM + edge codec checkpoint)
```bash
torchrun --standalone --nproc_per_node=3 scripts/train.py \
  --config configs/paper_train_controlnet.yaml \
  --train-list data/coco/train2017 \
  --val-list data/coco/val2017 \
  --batch-size 21
```

### Optional — Stage 1 JSCC (MSE-only base; or the MSE+GAN variant)
```bash
torchrun --standalone --nproc_per_node=3 scripts/train.py \
  --config configs/paper_train_jscc.yaml \
  --train-list data/imagenet/train \
  --val-list data/imagenet/val \
  --batch-size 21

# closer-to-paper MSE + λ·GAN variant (λ assumed/unpublished):
torchrun --standalone --nproc_per_node=3 scripts/train.py \
  --config configs/paper_train_jscc_gan.yaml \
  --train-list data/imagenet/train \
  --val-list data/imagenet/val \
  --batch-size 21
```

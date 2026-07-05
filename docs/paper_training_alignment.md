> [← 문서 색인](./README.md)

# Paper training alignment — confirmed vs assumed hyperparameter

`sgdjscc_lab`이 SGD-JSCC hyperparameter를 어떻게 설정하는지를 **공개 `SGDJSCC/`
코드로 confirmed**, **논문 table 값**, **미공개 assumption**으로 구분한다.
[paper_gap_closure.md](./paper_gap_closure.md)(구조적 충실도 + `paper_mode`)의
동반 문서. Ground-truth 우선순위: **공개 코드 우선**, 논문 table 차순; 충돌 시
재현성을 위해 공개 코드 값을 유지한다.

## 1. 공개 코드에서 confirmed

| 항목 | 값 | 출처 |
|---|---|---|
| `diffusion_step` / `guidance_scale` / `controlnet_scale` / `cfg_method` | 50 / 4.0 / 0.3 / `pcs_1.0` | `SGDJSCC/configs/inference.yaml` |
| backbone `depth`/`hidden_size`/`num_heads`/`patch_size` | 12 / 512 / 8 / 1 | `inference_config.py` `MDTv2(…)` |
| ControlNet `copy_blocks_num` | 6 | `inference_config.py` `MDTv2_ControlNet(…)` |
| timestep `frequency_embedding_size` | 256 | `mask_diffusion.py` `TimestepEmbedder(…)` |
| JSCC training SNR | 10 dB | 논문 §VI + repo 기본값 |

`sgdjscc_lab`에서는: `configs/model/sgdjscc.yaml`(scalar)과
`models/diffusion_wrapper.py`(`MDTv2` dim). dim은 **변경 금지**(checkpoint 호환성).

## 2. 공개 코드 vs 논문 table 충돌

| 항목 | 공개 코드 | 논문 table | Repo 선택 |
|---|---|---|---|
| `guidance_scale` | **4.0** | 4.5 | **4.0** 유지 (재현성) |
| "embedding size" | backbone 512; timestep 256 | 256 | backbone **512** 유지. 256은 256-d backbone이 아니라 *timestep* `frequency_embedding_size`(별개 값)를 가리킬 가능성이 가장 높음. |

## 3. Assumed / 미공개 (합리적 기본값)

| 항목 | 기본값 | 상태 |
|---|---|---|
| `lr` / `weight_decay` | 1e-4 / 1e-5 | assumed (전형적 AdamW latent-DM); 미공개 |
| `cfg_dropout_prob` | 0.1 | assumed (PixArt 관례); 미공개 |
| CFG null token | `learned`(논문 cfg) / `zero`(기본) | paper-like intent |
| edge codec (`vit`, embed 128, depth 4, heads 4) | 표기대로 | 재현 가능한 최근접; 정확한 WITT 재사용은 **unsupported** |
| edge codec multi-SNR range | 0–20 dB | assumed; 미공개 |
| JSCC GAN weight λ | 0.5 | paper-LIKE objective(MSE+λ·GAN); λ 미공개 |
| DM stage step / batch size | 250k / 64 | 논문 table-scale target; 코드 미확인 |

pipeline이 돌고 *구조적으로* faithful하도록 설정했으며, config에 정직하게
표기(`assumed default (unpublished in the paper)`)되고 paper-confirmed로 인용하면
안 된다.

## 4. 데이터 범위

`paper_train_text_dm.yaml` / `paper_train_controlnet.yaml`은 **COCO-only** caption
source를 사용한다. 논문 Stage-2 DM은 훨씬 큰(~14M-image) multi-dataset corpus로
학습하므로 repo config는 **더 작은 실용 재현**이다(각 config header에 명시).
`paper_mode`는 여전히 데이터셋 제공 caption을 강제하나, COCO를 논문의 14M corpus와
동등하게 만들 수는 없다.

## 5. 3-GPU training

`train.batch_size`는 **per-rank**:
`global_batch = batch_size × world_size × grad_accum_steps`. paper-scale 64를 3 GPU에서
쓰려면 `--batch-size 21`(≈63). 전체 command sequence(Stage 2 → MuGE precompute →
edge codec → ControlNet, JSCC 선택)는
[paper_gap_closure.md](./paper_gap_closure.md#multi-gpu-training-ddp)에 있다. 각 stage:

```bash
torchrun --standalone --nproc_per_node=3 scripts/train.py \
    --config configs/paper_train_<stage>.yaml \
    --train-list data/coco/train2017 --val-list data/coco/val2017 --batch-size 21
```

MuGE precompute(split당 1회): `scripts/prepare_muge_edges.py --input <split>
--model-root checkpoints --repr edge_uncertainty --device cuda:0`.

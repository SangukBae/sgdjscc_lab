# Dataset Status

`data/README.md` lives under the repo-root `data/` directory, which is ignored by
git via `/data/`. That makes it a poor place for documentation that must
propagate across machines.

This file is the tracked replacement for the parts that should travel with the
repo:

- canonical dataset roles
- stage-to-dataset mapping
- conversion workflow references
- operational notes that should not depend on one machine's local disk state

Machine-specific inventory such as "which datasets currently exist on this box",
"how many SA-1B shards are left", or "how large is `cc3m_pairs/` right now"
should be generated locally with:

```bash
python scripts/report_datasets.py
```

By default that writes an untracked markdown report to:

```text
data/_reports/dataset_status.md
```

## Canonical Dataset Roles

| Dataset path | Role | Loader type | Notes |
|---|---|---|---|
| `data/imagenet/` | image-only corpus | `image` | Typical Stage 1 baseline |
| `data/coco/` | text-image corpus | `text_image(_edge)` | Can use sidecar captions or COCO JSON captions |
| `data/journey_pairs/` | training-ready text-image pairs | `text_image(_edge)` | Prebuilt jpg/txt pairs |
| `data/cc3m_wds/` | raw WebDataset shards | — | Convert first with `scripts/prepare_cc3m.py` |
| `data/cc3m_pairs/` | converted text-image pairs | `text_image(_edge)` | Output of `prepare_cc3m.py` |
| `data/sa1b/raw/` | raw SA-1B tar shards | — | Not read directly by loaders |
| `data/sa1b_images/` | converted SA-1B image-only dataset | `image` | Output of `prepare_sa1b.py`; captions do not exist |
| `data/celeba/` | image-only CelebA | `image` | Text stages require generated sidecar captions |

## Stage Mapping

| Stage | Datasets | Notes |
|---|---|---|
| `jscc` | `imagenet`, `celeba`, `sa1b_images` | image-only |
| `csi_estimation` | `imagenet`, `celeba`, `sa1b_images` | image-only |
| `edge_codec` | `imagenet`, `celeba`, `sa1b_images` | Canny edges on the fly |
| `text_dm` | `coco`, `journey_pairs`, `cc3m_pairs`, captioned `celeba` | captions required |
| `controlnet` | `coco`, `journey_pairs`, `cc3m_pairs`, captioned `celeba` | captions + edges required |

## Conversion Workflows

- `scripts/prepare_cc3m.py`
  - converts raw `cc3m_wds/*.tar` shards into loader-ready jpg/txt pairs
  - supports sequential append and delete-on-success for disk-bounded conversion
- `scripts/prepare_sa1b.py`
  - converts raw `sa1b/raw/sa_*.tar` shards into `sa1b_images/{train,val}/<shard>/`
  - image-only output; `.json` masks are dropped
  - designed for one-shard-at-a-time conversion with optional tar deletion after
    verified commit

## Operational Rule

Tracked docs belong under `docs/`.

Untracked, machine-specific state belongs in generated reports under ignored
paths such as `data/_reports/`.

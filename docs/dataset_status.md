# Dataset Status

`data/README.md`는 repo-root의 `data/` 디렉터리에 있는데, 이 경로는 `/data/`로
git ignore된다. 그래서 머신 간에 전파되어야 할 문서를 두기에는 부적합하다.

이 파일은 repo와 함께 이동해야 하는 부분에 대한 tracked 대체본이다:

- canonical dataset 역할
- stage-to-dataset 매핑
- 변환 workflow 참조
- 특정 머신의 로컬 디스크 상태에 의존하면 안 되는 운영 노트

"이 머신에 지금 어떤 데이터셋이 있는가", "SA-1B shard가 몇 개 남았는가",
"`cc3m_pairs/`가 지금 얼마나 큰가" 같은 머신별 inventory는 로컬에서 생성한다:

```bash
python scripts/report_datasets.py
```

기본적으로 다음 경로에 untracked markdown report를 쓴다:

```text
data/_reports/dataset_status.md
```

## Canonical Dataset Roles

| Dataset path | 역할 | Loader type | 비고 |
|---|---|---|---|
| `data/imagenet/` | image-only corpus | `image` | 일반적인 Stage 1 baseline |
| `data/coco/` | text-image corpus | `text_image(_edge)` | sidecar caption 또는 COCO JSON caption 사용 가능 |
| `data/journey_pairs/` | training-ready text-image pair | `text_image(_edge)` | 사전 구축된 jpg/txt pair |
| `data/cc3m_wds/` | raw WebDataset shard | — | `scripts/prepare_cc3m.py`로 먼저 변환 |
| `data/cc3m_pairs/` | 변환된 text-image pair | `text_image(_edge)` | `prepare_cc3m.py` 출력 |
| `data/datacomp_pairs/` | 변환된 text-image pair | `text_image(_edge)` | DataComp jpg/txt pair |
| `data/sa1b/raw/` | raw SA-1B tar shard | — | loader가 직접 읽지 않음 |
| `data/sa1b_images/` | 변환된 SA-1B image-only 데이터셋 | `image` | `prepare_sa1b.py` 출력; caption 없음 |
| `data/celeba/` | image-only CelebA | `image` | text stage는 생성된 sidecar caption 필요 |
| `data/celeba_hq/` | image-only CelebA-HQ | `image` | text stage는 생성된 sidecar caption 필요 (`scripts/generate_captions.py`) |

## Stage Mapping

| Stage | Datasets | 비고 |
|---|---|---|
| `jscc` | `imagenet`, `celeba`, `sa1b_images` | image-only |
| `csi_estimation` | `imagenet`, `celeba`, `sa1b_images` | image-only |
| `edge_codec` | `sa1b_images`, `journey_pairs`, `cc3m_pairs`, `datacomp_pairs`, `celeba_hq` | MuGE edge sidecar (`edge_source: muge_sidecar`); on-the-fly Canny도 지원 |
| `text_dm` | `coco`, `journey_pairs`, `cc3m_pairs`, `datacomp_pairs`, caption된 `celeba`/`celeba_hq`, `sa1b_images` | caption 필요 (sidecar 또는 COCO JSON) |
| `controlnet` | `coco`, `journey_pairs`, `cc3m_pairs`, `datacomp_pairs`, caption된 `celeba`/`celeba_hq`, `sa1b_images` | caption + MuGE edge sidecar 필요 |

tracked **paper-like multi-GPU** workflow는 Stage 2/3를 결합 file-list
(`data/_lists/paper_like_multi/stage23_{train,val}.list`)로 구동하는데, 이는
`sa1b_images`, `journey_pairs`, `cc3m_pairs`, `datacomp_pairs`, `celeba_hq`를
풀링하고 edge/ControlNet 경로에는 이미지별 MuGE edge sidecar를 사용한다.
`scripts/prepare_paper_like_stage23_data.sh`와
`configs/custom_paper_like/paper_train_{text_dm,edge_codec,controlnet}_multi.yaml`
config 참조.

## Conversion Workflows

- `scripts/prepare_cc3m.py`
  - raw `cc3m_wds/*.tar` shard를 loader-ready jpg/txt pair로 변환
  - 디스크 제약 변환을 위한 순차 append + delete-on-success 지원
- `scripts/prepare_sa1b.py`
  - raw `sa1b/raw/sa_*.tar` shard를 `sa1b_images/{train,val}/<shard>/`로 변환
  - image-only 출력; `.json` 마스크는 드롭
  - 검증된 commit 이후 선택적 tar 삭제와 함께 one-shard-at-a-time 변환용 설계
- `scripts/generate_captions.py`
  - caption 없는 이미지 폴더(`celeba`, `celeba_hq`)를 `<stem>.txt` sidecar를 써서
    text-image pair로 승격(`fixed` / `filename` / Qwen2.5-VL `model` 모드; 후자는
    `transformers>=4.49` 필요)
- `scripts/prepare_paper_like_stage23_data.sh`
  - 결합 Stage 2/3 file-list를 구축하고 `sa1b_images`, `journey_pairs`,
    `cc3m_pairs`, `datacomp_pairs`, `celeba_hq` 전반에 이미지별 MuGE edge sidecar 생성

## Operational Rule

tracked 문서는 `docs/` 아래에 둔다.

untracked·머신별 상태는 `data/_reports/` 같은 ignored 경로의 생성 report에 둔다.

---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes:
---

> [← 문서 색인](../README.md)

# 데이터셋 운영 기준

- 문서 역할
  - canonical dataset 역할
  - stage–dataset 매핑
  - 변환 workflow
  - 머신 독립 운영 규칙
- 저장 원칙
  - `data/`: git ignore 대상
  - `docs/protocols/datasets.md`: tracked 기준 문서

- 머신별 inventory
  - 설치된 dataset
  - 남은 SA-1B shard
  - `cc3m_pairs/` 용량
  - 생성 명령

```bash
python scripts/report_datasets.py
```

- 기본적으로 다음 경로에 untracked markdown report를 쓴다:

```text
data/_reports/dataset_status.md
```

## 표준 데이터셋 역할

| 데이터 경로 | 역할 | loader 형식 | 비고 |
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

## Stage별 데이터 매핑

| Stage | 데이터셋 | 비고 |
|---|---|---|
| `jscc` | `imagenet`, `celeba`, `sa1b_images` | image-only |
| `csi_estimation` | `imagenet`, `celeba`, `sa1b_images` | image-only |
| `edge_codec` | `sa1b_images`, `journey_pairs`, `cc3m_pairs`, `datacomp_pairs`, `celeba_hq` | MuGE edge sidecar (`edge_source: muge_sidecar`); on-the-fly Canny도 지원 |
| `text_dm` | `coco`, `journey_pairs`, `cc3m_pairs`, `datacomp_pairs`, caption된 `celeba`/`celeba_hq`, `sa1b_images` | caption 필요 (sidecar 또는 COCO JSON) |
| `controlnet` | `coco`, `journey_pairs`, `cc3m_pairs`, `datacomp_pairs`, caption된 `celeba`/`celeba_hq`, `sa1b_images` | caption + MuGE edge sidecar 필요 |

- paper-like multi-GPU workflow
  - stage: 2·3
  - file list: `data/_lists/paper_like_multi/stage23_{train,val}.list`
  - dataset pool: `sa1b_images`, `journey_pairs`, `cc3m_pairs`, `datacomp_pairs`, `celeba_hq`
  - 구조 가이드: 이미지별 MuGE edge sidecar
  - 준비 스크립트: `scripts/prepare_paper_like_stage23_data.sh`
  - config: `configs/experiments/paper_reproduction/custom_paper_like/`

## 변환 절차

- `scripts/prepare_cc3m.py`
  - raw `cc3m_wds/*.tar` shard를 loader-ready jpg/txt pair로 변환
  - 디스크 제약 변환을 위한 순차 append + delete-on-success 지원
- `scripts/prepare_sa1b.py`
  - raw `sa1b/raw/sa_*.tar` shard를 `sa1b_images/{train,val}/<shard>/`로 변환
  - image-only 출력; `.json` 마스크는 드롭
  - 검증된 commit 이후 선택적 tar 삭제와 함께 one-shard-at-a-time 변환용 설계
- `scripts/generate_captions.py`
  - 대상: `celeba`, `celeba_hq`
  - 출력: `<stem>.txt` sidecar
  - 모드: `fixed`, `filename`, Qwen2.5-VL `model`
  - model 요구사항: `transformers>=4.49`
- `scripts/prepare_paper_like_stage23_data.sh`
  - Stage 2·3 결합 file-list 생성
  - 대상: `sa1b_images`, `journey_pairs`, `cc3m_pairs`, `datacomp_pairs`, `celeba_hq`
  - 출력: 이미지별 MuGE edge sidecar

## 운영 규칙

- tracked 문서는 `docs/` 아래에 둔다.

- untracked·머신별 상태는 `data/_reports/` 같은 ignored 경로의 생성 report에 둔다.

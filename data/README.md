---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes:
---

# 로컬 학습 데이터

현재 머신에 준비된 학습 데이터의 용도와 형식을 정리한다. 논문이 사용한 전체
규모의 말뭉치가 아니라 학습 경로 검증과 축소 실험용 데이터다. 상세한 loader 설정은
[데이터셋 지침](../docs/protocols/datasets.md)을 따른다.

## 데이터 구성

| 폴더 | 용도 | 형식 |
|---|---|---|
| `imagenet/` | JSCC·CSI·edge codec | 이미지 |
| `coco/` | text DM·ControlNet | 이미지+캡션 |
| `journey_pairs/` | text DM·ControlNet | 이미지+sidecar 캡션 |
| `celeba/` | JSCC 계열 | 이미지 |
| `cc3m_wds/` | 변환 전 CC3M | WebDataset tar |
| `cc3m_pairs/` | text DM·ControlNet | 변환된 이미지+캡션 |
| `sa1b/raw/` | 변환 전 SA-1B | tar |
| `sa1b_images/` | JSCC 계열 | 변환된 이미지 |
| `JourneyDB_subset/` | 원본 보관 | tgz/jsonl |
| `celeba_raw/` | 원본 보관 | zip |

`cc3m_wds/`와 `sa1b/raw/`의 tar는 학습 loader가 직접 읽지 않는다. 먼저 변환한다.

```bash
# CC3M → 이미지·캡션 pair
python scripts/prepare_cc3m.py --split train --limit-shards 8

# SA-1B → image-only
python scripts/prepare_sa1b.py --limit-shards 1
```

원본 tar를 변환 후 지우려면 각 스크립트의 `--delete-shard-on-success`를 명시한다.
검증된 output이 생성된 뒤에만 원본을 삭제하며, 먼저 `--dry-run`으로 대상을 확인하는
것이 좋다. 이 문서 정리 과정에서는 데이터 파일을 삭제하지 않았다.

## Stage별 선택

| Stage | 권장 데이터 |
|---|---|
| `jscc` | `imagenet`, `celeba`, `sa1b_images` |
| `csi_estimation` | `imagenet`, `celeba`, `sa1b_images` |
| `edge_codec` | 이미지 데이터+Canny |
| `text_dm` | `coco`, `journey_pairs`, `cc3m_pairs` |
| `controlnet` | 캡션 pair+Canny |

CelebA를 `text_dm`이나 `controlnet`에 사용하려면 먼저 caption sidecar를 만든다.

```bash
python scripts/generate_captions.py \
    --input data/celeba/train --mode fixed
```

대규모 데이터는 폴더 전체 탐색 대신 `input_mode: file_list`와 `file_list_path`를
사용할 수 있다. 학습 명령은 [학습 지침](../docs/protocols/training.md)을 참고한다.

## 보관 파일

다음은 loader가 사용하지 않는 원본 archive다. 공간이 필요해도 자동 삭제하지 말고,
변환 결과와 백업 여부를 확인한 뒤 별도로 정리한다.

- `imagenet/ILSVRC2012_img_*.tar`
- `JourneyDB_subset/`
- `celeba_raw/`
- `celeba/raw_extracted/` 아래의 중복 압축본

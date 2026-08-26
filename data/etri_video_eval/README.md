---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes:
---

# ETRI 영상 평가 데이터셋

영상·시간축 파이프라인 평가에 사용하는 10개 영상 데이터셋이다.

## 폴더 구성

```text
raw/        원본 10초 영상
processed/  실험 규격으로 변환한 영상
frames/     processed 영상을 PNG 프레임으로 추출한 폴더
gt/         사람이 검증한 구간별 객체 존재 GT(JSON)
captions/   no-models 검증용 선택적 캡션
```

`gt/<video_id>.json`은 `manual_verified_sampled_frames` 상태의 구간별 객체 존재
정보다. `gt` presence backend와 `ensemble_gt_filter` 보정에 사용한다. 형식과 사용
근거는 [지표 정의](../../docs/architecture/metrics.md#presence객체-존재-판정-backend)와
[OWLv2/VQA 보정 실험](../../docs/experiments/2026-07-28_owlv2_vqa_calibration.md)을
참고한다.

## 영상 규격

`processed/`의 영상은 모두 MP4(H.264, `yuv420p`), 512×256, 10fps, 10초,
100프레임이며 오디오는 제거했다. 가로·세로를 128의 배수로 맞춰 불필요한 겹침
타일 생성을 피한다.

## 사용법

```bash
# MP4 입력
python scripts/evaluate_video.py \
    --config configs/recipes/video/composed_video.yaml \
    --input data/etri_video_eval/processed/01_person_walk.mp4 \
    --snr 5 --save-video

# 프레임 폴더 입력
python scripts/evaluate_video.py \
    --config configs/recipes/video/composed_video.yaml \
    --input data/etri_video_eval/frames/01_person_walk \
    --snr 5 --save-video

# 모델을 로드하지 않는 파이프라인 검증
python scripts/evaluate_video.py \
    --config configs/recipes/video/composed_video.yaml \
    --input data/etri_video_eval/processed/01_person_walk.mp4 \
    --captions data/etri_video_eval/captions/01_person_walk.txt \
    --no-models
```

`composed_video.yaml`은 공개 baseline 체크포인트를 사용한다. checkpoint 선택 기준은
[재현성 지침](../../docs/protocols/reproducibility.md), 10개 영상 일괄 평가는
[평가 절차](../../docs/protocols/evaluation.md#etri-10-영상-배치-평가)를 참고한다.
영상 목록과 객체·이벤트 범주는 `manifest.csv`에 있다.

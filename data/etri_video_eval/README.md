# ETRI Video Evaluation Dataset

This folder contains the 10-video evaluation set for the ETRI video/temporal
pipeline experiments.

## Folder Layout

```text
raw/        Original 10-second generated videos.
processed/ Normalized experiment videos.
frames/    PNG frame folders extracted from processed videos.
gt/         Reserved for frame/segment-level object GT JSON.
captions/   Optional no-models dry-run captions, one txt file per video.
```

## Processed Video Format

All files under `processed/` were converted from `raw/` with the same settings:

- container/codec: `mp4`, H.264, `yuv420p`
- resolution: `512x256`
- fps: `10`
- duration: `10` seconds
- frames: `100`
- audio: removed

The resolution is intentionally a multiple of 128 in both dimensions. This avoids
the overlapping-patch explosion described in `src/sgdjscc_lab/utils/preprocessing.py`.

## Usage

Use either the processed mp4 directly:

```bash
python scripts/evaluate_video.py \
  --config configs/composed_video_paper_like_multi.yaml \
  --input data/etri_video_eval/processed/01_person_walk.mp4 \
  --snr 5 --save-video
```

Or use the extracted frame folder:

```bash
python scripts/evaluate_video.py \
  --config configs/composed_video_paper_like_multi.yaml \
  --input data/etri_video_eval/frames/01_person_walk \
  --snr 5 --save-video
```

For pipeline-only validation without model loading:

```bash
python scripts/evaluate_video.py \
  --config configs/composed_video_paper_like_multi.yaml \
  --input data/etri_video_eval/processed/01_person_walk.mp4 \
  --captions data/etri_video_eval/captions/01_person_walk.txt \
  --no-models
```

See `manifest.csv` for the video list and intended object/event categories.

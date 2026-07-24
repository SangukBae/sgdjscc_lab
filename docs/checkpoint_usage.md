# Checkpoint 사용 기준

이 문서는 로컬/원격에서 서로 다른 가중치를 실수로 섞어 쓰지 않기 위한 기준이다.

## 핵심 결론

- `configs/composed.yaml`, `configs/composed_video.yaml`은 공개 baseline checkpoint를 쓰는 기본 config다.
- `configs/composed_paper_like_multi.yaml`, `configs/composed_video_paper_like_multi.yaml`은 원격에서 학습한 paper-like multi-stage checkpoint를 쓰는 custom config다.
- baseline과 custom 가중치를 같은 `checkpoints/` 폴더에서 덮어써서 관리하지 않는다.

## 디렉터리 역할

| 경로 | 역할 |
|---|---|
| `checkpoints/` | 공개 SGD-JSCC baseline inference checkpoint |
| `outputs/checkpoints/*/best.pth` | stage별 학습 checkpoint. optimizer/scaler/epoch를 포함한 학습 스냅샷 |
| `checkpoints_custom/paper_like_multi/` | `outputs/checkpoints/paper_*_multi/best.pth`에서 export한 custom inference checkpoint |

## Baseline 실행

공개 baseline 가중치를 사용하려면 기존 config를 쓴다.

```bash
python scripts/infer_images.py --config configs/composed.yaml --snr 5
python scripts/evaluate_video.py --config configs/composed_video.yaml --input /path/to/video_or_frames --snr 5
```

이 경로는 `model_root: ../checkpoints/`를 사용한다.

## Custom paper-like multi-stage 실행

원격에서 학습한 multi-stage 가중치를 inference/evaluation에 반영하려면 새 config를 쓴다.

```bash
python scripts/infer_images.py --config configs/composed_paper_like_multi.yaml --snr 5
python scripts/evaluate_video.py --config configs/composed_video_paper_like_multi.yaml --input /path/to/video_or_frames --snr 5
```

이 경로는 다음을 사용한다.

```yaml
model_root: "../checkpoints_custom/paper_like_multi/"
snr_estimator_checkpoint: "../outputs/checkpoints/csi_estimation/best.pth"
```

## Custom inference checkpoint 생성 방법

`text_dm`과 `controlnet` 학습 checkpoint는 그대로 inference loader에 넣을 수 없다. 반드시
`scripts/export_checkpoint.py`로 변환해야 한다.

```bash
python scripts/export_checkpoint.py \
  --stage text_dm \
  --input outputs/checkpoints/paper_text_dm_multi/best.pth \
  --output checkpoints_custom/paper_like_multi/diffusion_backbone.pth \
  --force

python scripts/export_checkpoint.py \
  --stage controlnet \
  --input outputs/checkpoints/paper_controlnet_multi/best.pth \
  --output checkpoints_custom/paper_like_multi/diffusion_controlnet.pth \
  --force
```

`JSCC_model.pth`와 `muge-epoch-19-checkpoint.pth`는 baseline과 동일 파일을 복사해서 쓴다.

```bash
cp checkpoints/JSCC_model.pth checkpoints_custom/paper_like_multi/
cp checkpoints/muge-epoch-19-checkpoint.pth checkpoints_custom/paper_like_multi/
```

## 주의

- `outputs/checkpoints/*/best.pth`는 학습 재개 또는 export 입력용이다.
- `checkpoints/*.pth`와 `checkpoints_custom/*/*.pth`는 inference loader가 직접 읽는 파일이다.
- 로컬과 원격에서 같은 결과를 비교하려면 반드시 같은 config를 써야 한다.
- 특히 `configs/composed.yaml`과 `configs/composed_paper_like_multi.yaml`은 같은 입력을 줘도 서로 다른 diffusion 가중치를 쓰므로 결과가 달라질 수 있다.

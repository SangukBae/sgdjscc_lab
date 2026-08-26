---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes: docs/checkpoint_usage.md
---

> [← 문서 색인](../README.md)

# 재현 절차 — checkpoint 선택과 `paper_mode`

이 문서는 로컬/원격에서 서로 다른 가중치를 실수로 섞어 쓰지 않고, 논문 재현
경로와 확장 경로를 구분해 실행하기 위한 기준이다. 코드 수준의 충실도 분류
(paper-faithful/paper-like/scaffold/ETRI 확장)는 [reference/paper_alignment.md](../reference/paper_alignment.md)에 있다.

## Checkpoint 핵심 결론

- `configs/recipes/inference/composed.yaml`, `configs/recipes/video/composed_video.yaml`은
  공개 baseline checkpoint를 쓰는 기본 config다.
- `configs/recipes/inference/composed_paper_like_multi.yaml`,
  `configs/recipes/video/composed_video_paper_like_multi.yaml`은 원격에서 학습한
  paper-like multi-stage checkpoint를 쓰는 custom config다.
- baseline과 custom 가중치를 같은 `checkpoints/` 폴더에서 덮어써서 관리하지 않는다.

### 디렉터리 역할

| 경로 | 역할 |
|---|---|
| `checkpoints/` | 공개 SGD-JSCC baseline inference checkpoint |
| `outputs/checkpoints/*/best.pth` | stage별 학습 checkpoint. optimizer/scaler/epoch를 포함한 학습 스냅샷 |
| `checkpoints_custom/paper_like_multi/` | `outputs/checkpoints/paper_*_multi/best.pth`에서 export한 custom inference checkpoint |

### Baseline 실행

공개 baseline 가중치를 사용하려면 기존 config를 쓴다.

```bash
python scripts/infer_images.py --config configs/recipes/inference/composed.yaml --snr 5
python scripts/evaluate_video.py --config configs/recipes/video/composed_video.yaml --input /path/to/video_or_frames --snr 5
```

이 경로는 `model_root: ../checkpoints/`를 사용한다.

### Custom paper-like multi-stage 실행

원격에서 학습한 multi-stage 가중치를 inference/evaluation에 반영하려면 새 config를 쓴다.

```bash
python scripts/infer_images.py --config configs/recipes/inference/composed_paper_like_multi.yaml --snr 5
python scripts/evaluate_video.py --config configs/recipes/video/composed_video_paper_like_multi.yaml --input /path/to/video_or_frames --snr 5
```

이 경로는 다음을 사용한다.

```yaml
model_root: "../checkpoints_custom/paper_like_multi/"
snr_estimator_checkpoint: "../outputs/checkpoints/csi_estimation/best.pth"
```

### Custom inference checkpoint 생성 방법

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

### 주의

- `outputs/checkpoints/*/best.pth`는 학습 재개 또는 export 입력용이다.
- `checkpoints/*.pth`와 `checkpoints_custom/*/*.pth`는 inference loader가 직접 읽는 파일이다.
- 로컬과 원격에서 같은 결과를 비교하려면 반드시 같은 config를 써야 한다.
- 특히 `configs/recipes/inference/composed.yaml`과 `configs/recipes/inference/composed_paper_like_multi.yaml`은
  같은 입력을 줘도 서로 다른 diffusion 가중치를 쓰므로 결과가 달라질 수 있다.
- 학습 CLI와 stage별 export 매핑 전체는 [training.md](./training.md) 참고.

## `paper_mode` — 논문 재현 경로 강제

`paper_mode: true`는 확장 기능을 지우는 게 아니라 **논문 재현 실험과 ETRI 확장
실험이 섞이지 않게 guardrail을 거는 것**이다. 켜면:

- auto-caption, `filename` caption source 차단
- Canny stand-in 차단, MuGE sidecar 요구
- `shared_vae` edge transport 차단, `edge_jscc` 경로 요구
- Stage 3 `edge_jscc`는 학습된 edge codec checkpoint를 요구
- Stage 1 JSCC는 MSE-only가 아니라 MSE + patch-GAN 구조를 요구
- zero-vector CFG null 차단, learned null token 요구
- 논문 미공개값은 `paper_assumed_hparams`에 명시하고 실제 `train.*` 값과 일치해야 함
- 확장 기능(Phase 4/5, packet, regeneration 등) 비활성 요구
- eval metric set을 논문 보고 set에 맞춤

무엇이 paper-faithful/paper-like/scaffold/ETRI 확장으로 분류되는지, 하이퍼파라미터
출처(공개 코드 vs 논문 표 vs assumption)는 [reference/paper_alignment.md](../reference/paper_alignment.md)의
전체 표를 기준으로 삼는다 — 이 문서에서 다시 나열하지 않는다.

## 관련 문서
- [reference/paper_alignment.md](../reference/paper_alignment.md) — 충실도 분류, 하이퍼파라미터 출처
- [training.md](./training.md) — 학습 CLI, export, smoke 검증
- [datasets.md](./datasets.md) — 데이터셋 역할·stage 매핑

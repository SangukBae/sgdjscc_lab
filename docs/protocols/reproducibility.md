---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes: docs/checkpoint_usage.md
---

> [← 문서 색인](../README.md)

# 재현 절차 — checkpoint 선택과 `paper_mode`

- 목적
  - 로컬·원격 checkpoint 혼용 방지
  - 논문 재현·확장 경로 분리
- 충실도 분류
  - paper-faithful
  - paper-like
  - scaffold
  - ETRI 확장
  - 기준: [paper_alignment.md](../reference/paper_alignment.md)

## Checkpoint 핵심 결론

- 공개 baseline config
  - `configs/recipes/inference/composed.yaml`
  - `configs/recipes/video/composed_video.yaml`
- paper-like multi-stage config
  - `configs/recipes/inference/composed_paper_like_multi.yaml`
  - `configs/recipes/video/composed_video_paper_like_multi.yaml`
- baseline과 custom 가중치를 같은 `checkpoints/` 폴더에서 덮어써서 관리하지 않는다.

### 디렉터리 역할

| 경로 | 역할 |
|---|---|
| `checkpoints/` | 공개 SGD-JSCC baseline inference checkpoint |
| `outputs/checkpoints/*/best.pth` | stage별 학습 checkpoint. optimizer/scaler/epoch를 포함한 학습 스냅샷 |
| `checkpoints_custom/paper_like_multi/` | `outputs/checkpoints/paper_*_multi/best.pth`에서 export한 custom inference checkpoint |

### Baseline 실행

- 공개 baseline 가중치를 사용하려면 기존 config를 쓴다.

```bash
python scripts/infer_images.py --config configs/recipes/inference/composed.yaml --snr 5
python scripts/evaluate_video.py --config configs/recipes/video/composed_video.yaml --input /path/to/video_or_frames --snr 5
```

- 이 경로는 `model_root: ../checkpoints/`를 사용한다.

### Custom paper-like multi-stage 실행

- 원격에서 학습한 multi-stage 가중치를 inference/evaluation에 반영하려면 새 config를 쓴다.

```bash
python scripts/infer_images.py --config configs/recipes/inference/composed_paper_like_multi.yaml --snr 5
python scripts/evaluate_video.py --config configs/recipes/video/composed_video_paper_like_multi.yaml --input /path/to/video_or_frames --snr 5
```

- 이 경로는 다음을 사용한다.

```yaml
model_root: "../checkpoints_custom/paper_like_multi/"
snr_estimator_checkpoint: "../outputs/checkpoints/csi_estimation/best.pth"
```

### Custom inference checkpoint 생성 방법

- export 대상: `text_dm`, `controlnet`
- 변환 도구: `scripts/export_checkpoint.py`

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

- `JSCC_model.pth`와 `muge-epoch-19-checkpoint.pth`는 baseline과 동일 파일을 복사해서 쓴다.

```bash
cp checkpoints/JSCC_model.pth checkpoints_custom/paper_like_multi/
cp checkpoints/muge-epoch-19-checkpoint.pth checkpoints_custom/paper_like_multi/
```

### 주의

- `outputs/checkpoints/*/best.pth`는 학습 재개 또는 export 입력용이다.
- `checkpoints/*.pth`와 `checkpoints_custom/*/*.pth`는 inference loader가 직접 읽는 파일이다.
- 로컬과 원격에서 같은 결과를 비교하려면 반드시 같은 config를 써야 한다.
- config 비교 주의
  - `composed.yaml`: baseline diffusion weight
  - `composed_paper_like_multi.yaml`: custom diffusion weight
  - 같은 입력이어도 결과가 달라질 수 있음
- 학습 CLI와 stage별 export 매핑 전체는 [training.md](./training.md) 참고.

## `paper_mode` — 논문 재현 경로 강제

- `paper_mode: true`
  - 목적: 논문 재현·ETRI 확장 실험 분리
  - 적용 항목

- auto-caption, `filename` caption source 차단
- Canny stand-in 차단, MuGE sidecar 요구
- `shared_vae` edge transport 차단, `edge_jscc` 경로 요구
- Stage 3 `edge_jscc`는 학습된 edge codec checkpoint를 요구
- Stage 1 JSCC는 MSE-only가 아니라 MSE + patch-GAN 구조를 요구
- zero-vector CFG null 차단, learned null token 요구
- 논문 미공개값은 `paper_assumed_hparams`에 명시하고 실제 `train.*` 값과 일치해야 함
- 확장 기능(Phase 4/5, packet, regeneration 등) 비활성 요구
- eval metric set을 논문 보고 set에 맞춤

- 상세 기준
  - 구현 충실도 분류
  - hyperparameter 출처
  - 기준 문서: [paper_alignment.md](../reference/paper_alignment.md)

## 관련 문서
- [reference/paper_alignment.md](../reference/paper_alignment.md) — 충실도 분류, 하이퍼파라미터 출처
- [training.md](./training.md) — 학습 CLI, export, smoke 검증
- [datasets.md](./datasets.md) — 데이터셋 역할·stage 매핑

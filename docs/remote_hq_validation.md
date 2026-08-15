> [← 문서 색인](./README.md)

# 원격 3×RTX 4090 고품질 검증

저해상도 smoke, mock, `--no-models`를 거치지 않고 실제 체크포인트로 최종
산출물을 만드는 실행 경로다. 기본 설정은 이미지 Kodak 24장×7 SNR, ETRI
10영상×100프레임, SGD-JSCC diffusion 50 step, SVD 512×256/25 step, Wan
start-only·bidirectional 512×256/30 step이다.

## 환경

컨테이너 안에서 절대경로를 사용한다. 컨테이너의 `$HOME`은 `/root`이므로
`~/SangukBae/...`는 원격 호스트 경로를 가리키지 않는다.

```bash
cd /home/wilco/SangukBae/Semantic/sgdjscc_lab
export SGDJSCC_LGVSC_WORKER_PYTHON=/home/wilco/SangukBae/Semantic/.venvs/lgvsc_gen/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2
```

## 전체 실행

```bash
mkdir -p outputs/remote_hq_4090_20260816
python scripts/run_remote_hq_validation.py \
  --output-root outputs/remote_hq_4090_20260816 \
  2>&1 | tee outputs/remote_hq_4090_20260816/console.log
```

장시간 SSH 연결을 유지해야 하므로 실제 실행은 `tmux` 안에서 하는 것을 권장한다.
중단 후에는 같은 명령에 `--skip-existing`을 추가한다. 각 하위 배치도 완료된
video/mode만 건너뛰며, 최상위 상태는 `hq_validation_status.json`에 남는다.

```bash
mkdir -p outputs/remote_hq_4090_20260816
python scripts/run_remote_hq_validation.py \
  --output-root outputs/remote_hq_4090_20260816 \
  --skip-existing \
  2>&1 | tee -a outputs/remote_hq_4090_20260816/console.log
```

## 단계별 재실행

`--phases`는 `image,video,remeasure,svd,wan,quality` 중 쉼표 목록을 받는다.
예를 들어 Wan 실패분부터 이어서 실행하고 프레임 품질 지표를 다시 만들려면:

```bash
python scripts/run_remote_hq_validation.py \
  --output-root outputs/remote_hq_4090_20260816 \
  --phases wan,quality --skip-existing
```

## 중요한 실행 의미

- `run_etri_video_eval.py`는 영상 단위 3-worker 병렬이다. GPU마다 독립적인
  SGD-JSCC 작업 하나가 실행된다.
- Wan은 세 작업을 병렬로 돌리지 않는다. 한 14B pipeline을
  `device_map=balanced`로 GPU 0/1/2에 분산하고 영상·모드는 순차 실행한다.
- SVD는 GPU 0 한 장을 사용한다.
- 생성 batch에 `--no-models`를 전달하지 않으므로 keyframe 복원과 생성 worker
  모두 실제 모델이다.
- worker resolution/step CLI override가 생성된 video별 YAML에 기록되므로 결과
  provenance에서 512×256/25·30 step을 재확인할 수 있다.

## 산출물

```text
outputs/remote_hq_4090_20260816/
  preflight.json
  hq_validation_plan.json
  hq_validation_status.json
  image/kodak_snr_sweep.csv
  video_real_step50/baseline/<video>/
  remeasure/summary_metrics.{csv,md}
  generation/{svd_start_only,wan_skim_sfa,wan_skem_dsa}/<video>/
  quality/video_{frames,summary}.csv
  quality/generation_{frames,summary}.csv
  quality/{video,generation}_summary.json
```

`quality/generation_frames.csv`는 temporal packet 지표와 별도로 실제 원본 대응
프레임과 generated/recon frame의 PSNR·SSIM·LPIPS·CLIP image similarity를
계산한다. 후보 프레임 크기가 원본과 다르면 bicubic으로 원본 크기에 맞추고
`resized_for_metric=true`로 기록해 숨기지 않는다.

## 다운로드/실패 주의

Wan I2V/FLF2V, BLIP-2, OWLv2, SVD는 2026-08-16 원격 컨테이너 캐시에 모두
다운로드했고 `local_files_only=True` 조회까지 확인했다. 캐시가 삭제된 뒤 SVD
접근 오류가 나면 Hugging Face에서 모델 사용 조건을 승인하고 컨테이너 안에서
`hf auth login`한 뒤 `--skip-existing`으로 재개한다. 호스트 RAM은 62GiB이고
swap이 없으므로 Wan 작업을 GPU별 독립 프로세스 세 개로 동시에 실행하지 않는다.

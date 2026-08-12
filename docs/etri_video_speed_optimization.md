# ETRI 10-영상 실모델 검증 속도 분석 및 가속화 (2026-07-24)

## 0. 요약

`evaluate_video.py`(Phase 4-B 키프레임/시간축 파이프라인)의 실모델 검증이
느린 근본 원인은 **비디오 프레임이 128×128 패치로 타일링되어, 512×256 영상
1프레임당 8개 패치가 각각 독립적으로 전체 diffusion 샘플링을 돈다**는 것이다.
여기에 diffusion_step=50(기본값), 프레임마다 반복되는 BLIP2/CLIP 호출,
진행률을 알 수 없는 구조가 겹쳐 체감 속도를 더 늦춘다.

이번 작업으로:
- 프레임 단위 diffusion/BLIP2/CLIP 호출 카운터 + 진행률 스트리밍(JSON)을 추가했고,
- diffusion_step sweep, keyframe-only 실모델 모드, CLIP/캡션 경량화 옵션을
  전부 **opt-in config/CLI 플래그**로 추가했으며,
- CLIP 텍스트 임베딩 캐시(프로세스 내) + 원본 프레임 packet 디스크 캐시(옵션)를 추가하고,
- 배치 드라이버에 멀티-GPU 병렬 실행 + thread 제한 + GPU 사용률 로깅을 추가했고,
- 그 과정에서 **실모델 배치 실행을 막고 있던 버그 2건**을 실제로 발견·수정했다
  (§4 참조 — `--no-models`만 검증되어 있던 코드 경로라 지금까지 드러나지 않았음).
- 원격 GPU(RTX 4090 ×3)에서 1개 영상(01_person_walk, 100프레임)에 대해
  `real_keyframe_only_step10`/`real_all_frames_step10`을 **전체 길이로 실행 완료**했다.

## 1. 병목 원인 분석

코드 경로: `scripts/evaluate_video.py` → `video/temporal_pipeline.py` →
`pipelines/infer_pipeline.py` (`_run_diffusion`) / `guidance/semantic_packet_extractor.py`
/ `evaluators/clip_score.py` / `runtime.py`.

| # | 원인 | 근거 |
|---|---|---|
| 1 | **프레임당 diffusion 호출이 8배** | ETRI 영상은 512×256 → `prepare_patches`가 128×128 타일 8개로 분할. `TemporalPipeline`이 diffusion을 부르는 매 프레임(키프레임 또는 recompute)마다 `_process_patches`가 패치 8개를 **순차적으로** `run_single_image`에 넣는다 — 배치화되어 있지 않음. |
| 2 | `diffusion_step=50` 고정 | `configs/base/model/sgdjscc.yaml`의 기본값. DPM-Solver++ 2M 루프 스텝 수에 선형적으로 비례하는 비용. |
| 3 | recon(복원) 프레임 packet 추출 시 BLIP2가 매번 새로 호출됨 | `evaluate_video.py`의 `_caption_for()`: `--captions`를 줘도 **원본** 프레임만 그 캡션을 쓰고, **복원된** 프레임의 packet(캡션→오브젝트/할루시네이션 판정)은 실모델 모드에서 항상 `caption=None`을 넘겨 BLIP2를 다시 호출했다 — recompute/generate/키프레임마다 반복. (reuse 프레임만 예외: 키프레임의 recon_packet을 그대로 재사용.) |
| 4 | Scene-label/오브젝트 어휘 CLIP 텍스트 인코딩이 프레임마다 재계산됨 | `SemanticPacketExtractor._probe_scene()`이 고정된 15개 scene label을 **매 프레임** `clip.tokenize`+`encode_text`로 다시 인코딩. `ObjectExtractor`의 COCO-80 어휘도 동일 패턴. |
| 5 | 진행률을 알 수 없음 | 산출물이 `pipeline.run()` 전체가 끝난 뒤에만 파일로 써짐 — 100프레임 실행 중 몇 번째 프레임인지, 왜 느린지 알 방법이 없었음. |
| 6 | (신규 발견) `model_root` 상대경로가 배치 드라이버 출력 폴더 깊이와 안 맞음 | `configs/base/model/sgdjscc.yaml: model_root: "../checkpoints/"`는 config 파일이 `configs/` 바로 아래 있다고 가정. `run_etri_video_eval.py`가 생성하는 config.yaml은 `<output_root>/<stage>/<video>/`(2단계 아래)에 있어 `../checkpoints`가 엉뚱한 경로로 풀림 → **실모델(`--no-models` 아닌) 배치 실행은 지금까지 한 번도 성공한 적이 없었다.** `--no-models`만 `build_models()`를 건너뛰어 증상이 드러나지 않았던 것. §4에서 수정. |
| 7 | (신규 발견) SGDJSCC 원본의 `.cuda()` 하드코딩 | `SGDJSCC/models/test_advanced_network/diffusion_element_wise.py::encode_text()`가 `clip.tokenize(...).cuda()`로 **항상 프로세스 기본 CUDA 디바이스**를 씀 (`.to(self.device)`가 아님). `--device cuda:1`로 돌리면 텍스트 토큰(cuda:0)과 모델(cuda:1)이 어긋나 `RuntimeError`. `cuda:0`에서만 우연히 맞았던 것. SGDJSCC는 읽기 전용이라 원본은 못 고침 — §4에서 `CUDA_VISIBLE_DEVICES` 우회로 해결. |

원인 1·2(diffusion 자체 비용)이 압도적이고, 3·4는 부가 비용, 6·7은 애초에
실모델 배치/병렬 실행 자체를 막고 있던 잠재 버그였다.

## 2. 추가/수정한 파일

**신규**
- `src/sgdjscc_lab/utils/profiling.py` — 프레임 단위 diffusion/BLIP2/CLIP 호출 카운터
  + 스트리밍 진행률(JSON) + 최종 요약. 활성화하지 않으면(`set_active` 안 함) 완전 no-op.
- `src/sgdjscc_lab/utils/gpu_logger.py` — 백그라운드 스레드로 `nvidia-smi`를 주기적으로
  샘플링해 `gpu_util.csv`에 기록. `nvidia-smi` 없으면 조용히 비활성화.
- `src/sgdjscc_lab/guidance/packet_cache.py` — **원본** 프레임 packet(캡션/오브젝트/장면)만
  디스크 캐시. 복원 프레임 packet은 절대 캐시하지 않음(§3 설명).
- `scripts/run_speed_experiment.py` — 4-모드 비교 러너 + markdown 리포트 생성.
- `tests/test_speed_optimizations.py`, `tests/test_speed_experiment_runner.py` — 신규 기능 테스트(28개).

**수정**
- `src/sgdjscc_lab/pipelines/infer_pipeline.py` — `_run_diffusion()` 진입 시 프로파일링 카운터 1줄 추가(수치 영향 없음).
- `src/sgdjscc_lab/guidance/text_extractor.py` — BLIP2 호출 카운터 1줄 추가.
- `src/sgdjscc_lab/evaluators/clip_score.py` — CLIP 호출 카운터 + **텍스트 임베딩 프로세스 캐시**
  (`_encode_texts`가 동일 텍스트 리스트를 다시 받으면 모델 재호출 없이 캐시된 텐서를 clone 반환 — 결과 수치는 완전히 동일, 속도만 개선).
- `src/sgdjscc_lab/video/temporal_pipeline.py` — `force_interframe_reuse` 옵션 추가
  (기본 False = 기존 동작과 동일), 프로파일러 프레임 훅 배선.
- `scripts/evaluate_video.py` — 아래 §3의 CLI 플래그 전부 + progress/summary 파일 출력 +
  `temporal_frames.csv`에 `elapsed_sec/diffusion_calls/blip2_calls/clip_calls` 컬럼 추가.
- `scripts/run_etri_video_eval.py` — 위 플래그들의 배치 패스스루, `model_root` 절대경로
  수정(버그 6), `--parallel/--devices`(버그 7 우회 포함) + `--gpu-log-interval`.
- `tests/test_etri_batch_tools.py` — `model_root` 회귀 테스트 추가.

## 3. 새로 추가한 실행 옵션 (`evaluate_video.py`, 전부 opt-in)

| 플래그 | 효과 | 품질 검증 의미 변화 |
|---|---|---|
| `--diffusion-step N` | `cfg.diffusion_step` 오버라이드 | step<50이면 paper-comparable 아님 |
| `--max-frames N` | 앞 N프레임만 처리(처리량 추정용) | recon.mp4/temporal_metrics.csv가 부분 클립을 기술 |
| `--force-interframe-reuse` | 모든 inter-frame을 강제로 keyframe reuse (diffusion은 키프레임에서만) | inter-frame drift/hallucination이 전혀 검증 안 됨 — `real_all_frames_*`보다 약한 검증 |
| `--no-clip` | CLIP 평가자 자체를 안 만듦 (scene probe, CLIP object 검출, SRS의 CLIP 성분 없음) | CLIP 기반 지표 전부 무의미 |
| `--recon-caption-mode {own,skip}` | `own`(기본, 기존 동작 동일) / `skip`(복원 프레임 BLIP2 캡션 생략, caption="") | `skip`은 캡션 기반 recon 오브젝트/관계/속성이 사라짐 — hallucination/SRS 수치를 `own`과 비교 불가 |
| `--packet-cache-dir DIR` | **원본** 프레임 packet만 디스크 캐시(재실행 재사용) | 없음 — 캐시 키에 video mtime/size, caption source, CLIP 모델명, packet 스키마 버전 포함, 하나라도 다르면 전체 무효화 |
| `--profile-out PATH` | progress.json/profiling_summary.json 경로 지정 | 없음 |

`run_etri_video_eval.py`에는 동일 플래그가 패스스루로 추가됐고(`heldout` 스테이지 제외),
추가로:

| 플래그 | 효과 |
|---|---|
| `--parallel N --devices cuda:0,cuda:1,cuda:2` | N개 워커를 라운드로빈으로 여러 GPU에 분배(기본 `--parallel 1`=기존 순차 동작과 완전 동일). SGDJSCC의 `.cuda()` 하드코딩 버그를 `CUDA_VISIBLE_DEVICES` 리매핑으로 우회. |
| `--gpu-log-interval SEC` | `<output-root>/gpu_util.csv`에 GPU 사용률/메모리 주기 기록(기본 10초, `nvidia-smi` 없으면 자동 비활성화) |

## 4. 발견하고 고친 버그 (실모델 검증을 실제로 막고 있던 것들)

1. **`model_root` 상대경로 깊이 불일치** (`run_etri_video_eval.py::build_run_config`) —
   생성된 `config.yaml`이 `<output_root>/<stage>/<video>/`에 있는데 `../checkpoints/`는
   `configs/` 바로 아래를 가정 → 실모델 배치 실행이 전부 `FileNotFoundError`로 죽었음.
   `model_root`를 절대경로로 오버라이드해서 수정. (`run_speed_experiment.py`도 동일 수정.)
2. **SGDJSCC 원본의 `.cuda()` 하드코딩** (읽기 전용이라 원본은 수정 불가) —
   `--device cuda:1`에서 크로스-디바이스 크래시. `run_etri_video_eval.py`의 병렬 디스패치에서
   `CUDA_VISIBLE_DEVICES=<물리 인덱스>` + 프로세스 내부에는 `--device cuda:0`을 넘기는 방식으로 우회.
   `batch_status.json`의 `device` 필드는 물리 GPU를 그대로 기록(추적성 유지).

두 버그 모두 원격 GPU에서 직접 재현 후 수정 → 재검증까지 완료했다(§6).

## 5. 로컬 검증 결과

```
python -m pytest tests/ -q
```
- 수정 전: 724 passed
- 최종: **753 passed**, 0 failed (신규 28개 테스트 포함: profiling/force_interframe_reuse/
  packet_cache/CLIP 텍스트 캐시/gpu_logger/run_speed_experiment 결과 병합)
- `--no-models` 스모크: `evaluate_video.py --no-models --max-frames 10 --recon-caption-mode skip
  --packet-cache-dir ...` 및 `run_speed_experiment.py --modes no_models_captions` 정상 동작,
  `progress.json`/`profiling_summary.json`/캐시 파일 정상 생성 확인.
- GPU 없는 로컬 환경 특성상 실모델 경로는 로컬에서 검증 불가 → 원격에서 수행(§6).

## 6. 원격 GPU 검증 결과 (155.230.15.67, RTX 4090 × 3, 컨테이너 `sgdjscc`)

원격 컨테이너에서도 pytest 753 passed 확인 후, `01_person_walk`(512×256, 100프레임,
10fps) 1개 영상에 대해 두 모드를 **전체 100프레임** 실행:

| 모드 | 총 소요시간 | 파이프라인 순수 처리시간 | n_keyframes | n_reused | n_recompute | diffusion 호출 | BLIP2 호출 | ptc | sfr |
|---|---|---|---|---|---|---|---|---|---|
| `real_keyframe_only_step10` | 253.9s (~4.2분) | 168.1s | 9 | 91 | 0 | 72 (=9×8패치) | 9 | 0.504 | 0.037 |
| `real_all_frames_step10` | 311.6s (~5.2분) | 264.3s | 9 | 74 | 17 | 208 (=26×8패치) | 26 | 0.523 | 0.111 |

프레임 유형별 평균 소요시간(step10 기준):
- 키프레임(8패치 diffusion): ~14–18s/프레임 (첫 프레임은 CUDA warm-up으로 더 느림)
- recompute 프레임(8패치 diffusion): ~8.2s/프레임 (warm-up 이후 정상 속도, 패치당 ~1.0s)
- reuse 프레임(diffusion 없음): ~0.03s/프레임 — 사실상 무료

**해석**: 이 영상(사람 한 명이 걷는 저모션 클립)은 대부분의 inter-frame이 원래도
`reuse` 판정을 받아서, keyframe-only 강제가 절약하는 프레임이 17개뿐이라 속도 차이가
~19%에 그쳤다. 반대로 `sfr`(semantic flicker rate)은 `real_all_frames`가 0.111로
`keyframe_only`(0.037)보다 3배 높다 — 매 recompute가 독립적인 diffusion 샘플이라
프레임 간 미세한 흔들림이 더 생기는 것으로 보인다(참고 관찰이며, 정량적 결론을
내리기엔 표본 1개로 부족).

**병렬 실행 검증**: `run_etri_video_eval.py --parallel 2 --devices cuda:0,cuda:1
--diffusion-step 10 --max-frames 3`으로 2개 영상을 동시 처리 — `02_car_pass`가
`CUDA_VISIBLE_DEVICES=1 → cuda:0`으로 리매핑되어 물리 GPU 1에서 정상 완료(버그 2 수정
확인). `gpu_util.csv`에 두 GPU가 동시에 최대 98%/88% 사용률을 찍은 로그도 확보.

산출물(두 모드 모두 정상 생성): `config.yaml`, `recon.mp4`, `recon_frames/*.png`,
`temporal_frames.csv`(호출수/시간 포함), `temporal_metrics.csv`, `keyframes.json`,
`segments.json`, `profiling_summary.json`, `progress.json`, `run.log`.

## 7. step50/전체 10개 영상 예상 시간 (외삽, 실측 아님)

recompute 프레임의 warm-up 이후 patch당 비용(~1.0s @ step10)을 5배 스케일링(선형 가정,
DPM-Solver++ 스텝 수 비례)하면 step50에서 프레임당(8패치) ~41s.

| 시나리오 | 계산 | 추정 시간(1영상) |
|---|---|---|
| `real_keyframe_only_step50` | 9키프레임×41s + 모델로딩 ~80s | ~7–8분 |
| `real_all_frames_step50` (이 영상 기준, recompute 26/100) | 26×41s + 모델로딩 ~80s | ~19분 |
| `real_all_frames_step50` (최악, 전 프레임 recompute) | 100×41s + 모델로딩 | ~70분 |

10개 영상 전체(모션이 큰 05/06 등은 recompute 비율이 훨씬 높을 것으로 예상):
- `real_keyframe_only_step50` × 10영상, 순차: ~70–80분 / 3-GPU 병렬: ~25–30분
- `real_all_frames_step50` × 10영상, 순차: 최소 ~3시간, 모션 많은 영상 비중에 따라
  **6시간 이상**도 가능 / 3-GPU 병렬이어도 2시간+ 예상

## 8. 최종 응답에 포함할 요약

### 추천 실행 명령

```bash
# 1) 1개 영상, 빠른 처리량 추정 (모델 로딩 + 몇 프레임만)
python scripts/run_speed_experiment.py --videos 01_person_walk \
    --modes real_keyframe_only_step10,real_all_frames_step10 \
    --device cuda:0 --max-frames 8

# 2) 1개 영상 전체 길이, 4모드 비교 (오늘 실행한 것과 동일)
python scripts/run_speed_experiment.py --videos 01_person_walk --device cuda:0

# 3) 10개 영상 전체, keyframe-only step10 (가장 현실적인 1차 실모델 검증)
python scripts/run_etri_video_eval.py --stages baseline \
    --diffusion-step 10 --force-interframe-reuse \
    --parallel 3 --devices cuda:0,cuda:1,cuda:2 --gpu-log-interval 10

# 4) step50 전체 실모델은 위 결과로 시간을 먼저 가늠한 뒤, 모션이 큰 영상(05,06)만
#    우선 real_all_frames_step50으로 돌리고 나머지는 keyframe_only_step50으로 절충 추천
```

### 10개 전체 실모델 검증을 현실적으로 돌릴 때 추천 모드

**1차: `real_keyframe_only_step10`을 10개 영상 전체, 3-GPU 병렬로 먼저 돌릴 것을 추천.**
~25–30분 안에 전체 파이프라인(키프레임 복원 품질 + 시간축 구조)을 실모델로 검증할 수
있고, `--force-interframe-reuse`가 명시적으로 "inter-frame drift는 검증 안 됨"을
표시하므로 결과를 오독할 위험이 낮다. 이후 `real_all_frames_step10`을 모션이 큰
2–3개 영상(05_camera_pan_person, 06_handheld_sign 등)에 한정해 추가로 돌려
inter-frame recompute 품질을 표본 검증하고, step50 전체 배치는 그 결과를 보고
필요성이 확인된 영상에 한해서만 실행할 것을 권장한다. `--no-models` 결과는 어떤
경우에도 실화질 성능으로 인용하지 않는다.

## 9. 코드 리뷰 반영 (운영 안정성 수정, 2026-07-25)

1차 구현에 대한 리뷰에서 지적된 4건 — 전부 로컬 771 tests passed(신규 18개 포함)로
검증 완료:

1. **CUDA 디바이스 리매핑이 `--parallel` 경로에서만 적용되던 문제** —
   `remap_device_for_cuda_visible()`를 `run_etri_video_eval.py::_dispatch()`에서
   병렬 여부와 무관하게 항상 호출하도록 수정(`--parallel 1 --device cuda:1`도
   이제 커버). `run_speed_experiment.py::run_mode_video()`에도 동일 리매핑을
   새로 적용(이전엔 아예 우회 로직이 없어 `--device cuda:1` 실험이 취약했음).
2. **`run_speed_experiment.py`의 `results.json` merge가 진짜 동시성-안전하지 않던 문제** —
   `_merge_write_results_locked()`를 추가해 `fcntl.flock`으로 read-merge-write
   임계구역 전체를 프로세스 간 배타적으로 잠금. 실제 스레드 동시성 테스트
   (`test_concurrent_threads_do_not_lose_updates`, 12개 스레드가 동시에 서로 다른
   행을 write)로 손실 없음을 검증.
3. **`packet_cache.py`의 temp 파일명이 `<video>.json.tmp` 고정이던 문제** —
   `<video>.json.tmp<pid>`로 변경(`os.getpid()` 기반). 같은 `--packet-cache-dir`를
   여러 모드/threshold가 동시에 쓰는 상황(예: motion_sweep)에서 temp 파일 충돌 없음.
4. **profiling이 사실상 항상 켜져 있어 "opt-in" 설명과 실제 동작이 어긋나던 문제** —
   `evaluate_video.py`에 `--profile`(신규) / `--profile-out`(있으면 `--profile` 자동
   활성화) 플래그를 추가하고, 기본값 OFF로 변경. 플래그가 없으면 `progress.json`/
   `profiling_summary.json`이 전혀 생성되지 않고 `temporal_frames.csv`도 기존
   컬럼 집합 그대로다 — "기본 동작 불변"이 파일 단위로도 엄밀히 성립한다.
   `run_speed_experiment.py`(리포트가 profiling_summary.json에 의존)와
   `run_etri_video_eval.py`(신규 `--profile` 패스스루)는 각각 필요할 때 명시적으로
   `--profile`을 전달하도록 수정.

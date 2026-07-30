> [← 문서 색인](./README.md)

# OWLv2/VQA 실제 presence calibration 검증 준비 및 완료 결과

이 문서는 `docs/etri_strategy.md` 5차(순서 8~10)가 만든 presence-backend
스캐폴드를 **실제 OWLv2/VQA weight로 검증**하기 위해 무엇이 연결돼 있는지와,
2026-07-28에 수행한 10개 영상 batch 재검증 결과를 함께 정리한다. 초기 버전은
"실행 가능한 상태"까지의 준비 문서였지만, 현재는 실제 GPU recon frame을 재사용한
OWLv2/VQA/ensemble 50-job batch가 완료되어 결과 파일까지 남아 있다.

## 무엇이 이미 되어 있었는가 (변경 없이 확인만 함)

`evaluators/presence_backends.py`, `presence_calibration.py`,
`packet_verifier.py`, `pipelines/heldout_remeasurement.py`,
`scripts/remeasure_video_metrics.py`의 config 배선은 검증 결과 **이미
정확했다**:

```
verifier.use_presence_calibration
verifier.presence_mode
verifier.presence_backends
verifier.presence_backend_weights
verifier.presence_backend_cfg.owlv2.model_id
verifier.presence_backend_cfg.owlv2.score_threshold
verifier.presence_backend_cfg.vqa.vqa_backend.type
verifier.presence_backend_cfg.vqa.vqa_backend.model_id
verifier.presence_backend_cfg.vqa.vqa_backend.device
```

`evaluators/presence_calibration.py::build_presence_calibrator()` →
`evaluators/presence_backends.py::build_presence_backend()`로 이어지는 경로가
`verifier.presence_backend_cfg.<backend>.<key>`를 `OmegaConf.select()`로 정확히
읽고, VQA는 `verifier.presence_backend_cfg.vqa.vqa_backend`를 그대로
`evaluators/vqa_backend.py::build_vqa_backend()`에 넘긴다. 이는
`tests/test_presence_backends.py::TestBuildPresenceCalibrator::
test_owlv2_and_vqa_config_keys_route_through_without_loading_weights`가
(가중치 로딩 없이) 생성자 속성 검사로 증명한다 — `Owlv2PresenceBackend`/
`Blip2VQABackend`는 **생성 시점에는 가중치를 로드하지 않는다**
(`.check()`/`.answer()` 최초 호출 시 lazy load), 그래서 이 테스트는 네트워크나
GPU 없이도 config 전체 경로를 검증한다.

**결론: config 배선 자체는 고칠 것이 없었다.** 이번 준비 작업의 실제 산출물은
① 이 키들을 실제로 채운 예시 config, ② 기존 실모델 산출물을 재사용하는 새
CLI 경로, ③ 그 경로의 한계를 명시한 이 문서다.

## 준비된 예시 config 3종

`configs/video/default.yaml`의 `verifier.presence_backend_cfg`는 기본값이
빈 dict(`{}`, 전부 주석)로 남아 있다 — **기존 결과는 바뀌지 않는다**
(`verifier.use_presence_calibration: false`가 기본값이므로
`build_presence_calibrator()`는 여전히 `None`을 반환한다).

실제로 채워서 바로 실행 가능한 3개 config:

| Config | `presence_mode` | 용도 |
|---|---|---|
| `configs/etri_video_eval_owlv2.yaml` | `owlv2_only` | OWLv2 zero-shot detector 단독 검증 |
| `configs/etri_video_eval_vqa.yaml` | `vqa_only` | BLIP-2 VQA 단독 검증 |
| `configs/etri_video_eval_ensemble.yaml` | `ensemble_weighted` | CLIP+OWLv2+VQA 가중 앙상블 |

셋 다 `use_phase4: true` / `use_packet_verifier: true` /
`verifier.use_presence_calibration: true`가 이미 켜져 있고, 출력은
`outputs/etri_video_eval/manual_{owlv2,vqa,ensemble}/`로 격리된다. 필요하면
`presence_backend_weights`/`presence_backend_cfg`만 고쳐서 재사용하면 된다.

## 실행자가 바로 쓸 명령

### 1개 영상 OWLv2 sanity check (재구성 없이, 실모델 recon 재사용)

```bash
cd sgdjscc_lab
conda activate ptest
python scripts/remeasure_video_metrics.py --config configs/etri_video_eval_owlv2.yaml \
    --from-recon-frames outputs/etri_video_eval_real_full_step50/baseline/01_person_walk \
    --captions data/etri_video_eval/captions/01_person_walk.txt \
    --device cuda:0
```

### 1개 영상 VQA sanity check (동일 방식)

```bash
python scripts/remeasure_video_metrics.py --config configs/etri_video_eval_vqa.yaml \
    --from-recon-frames outputs/etri_video_eval_real_full_step50/baseline/01_person_walk \
    --captions data/etri_video_eval/captions/01_person_walk.txt \
    --device cuda:0
```

첫 실행 시 OWLv2(`transformers` Owlv2 클래스 + `google/owlv2-base-patch16-ensemble`
가중치)와 BLIP-2(`Salesforce/blip2-opt-2.7b-coco`)가 lazy-load되며, 캐시가 없으면
다운로드가 필요하다(장시간 GPU 배치는 아니지만 네트워크 접근은 필요). 가중치나
`transformers` 자체가 없으면 `PresenceBackendUnavailableError`로 명확히
실패한다(조용히 CLIP로 대체되지 않음, `*_only` 모드 한정 — 앙상블 모드는
가용한 backend만으로 계속 진행).

`--from-recon-frames` 모드의 packet 재추출용 CLIP은 `--device`를 명시하면 그
디바이스를 쓰고, 생략하면 CUDA 사용 가능 여부를 감지해 `cuda:0` → `cpu` 순서로
자동 선택한다. 실험 로그를 명확히 남기려면 위 예시처럼 `--device cuda:0`를 권장한다.

### 10개 영상 OWLv2/VQA/ensemble 배치 재측정 — `scripts/batch_remeasure_owlv2_vqa_10videos.py`

`scripts/run_etri_video_eval.py`의 `heldout` stage는 항상 `--from-run`(재구성)
경로만 쓰고, `configs/etri_video_eval_{owlv2,vqa,ensemble}.yaml`은 영상마다
직접 `--config`/heldout 출력 경로를 바꿔주지 않으면 모든 영상이 같은
`outputs/etri_video_eval/manual_{owlv2,vqa,ensemble}/heldout/`을 덮어쓴다.
이 문제를 해결하는 전용 배치 드라이버가 준비돼 있다:
`scripts/batch_remeasure_owlv2_vqa_10videos.py`. 이 스크립트는 실제 10개
영상 전체 실행을 대신 해주지 않는다(실제 GPU/네트워크 사용은 실행자가 직접
트리거) — 명령어 조립, 영상별/모드별 config 생성, 경로 충돌 방지, 결과 요약
집계까지만 담당한다.

**5개 모드** (`MODE_SPECS`, 각 모드가 왜 필요한지는 아래 "GT-object-only vs
open-world filter — 해석 차이" 참고):

| 모드 | base config | 목적 |
|---|---|---|
| `owlv2` | `configs/etri_video_eval_owlv2.yaml` | OWLv2-only detector calibration |
| `vqa` | `configs/etri_video_eval_vqa.yaml` | VQA-only (BLIP-2) calibration |
| `ensemble_nofilter` | `configs/etri_video_eval_ensemble.yaml` (`object_vocabulary_filter.enabled=false`) | 필터 적용 전 baseline — **비교용, 최종 주장에 쓰지 말 것** |
| `ensemble_gt_filter` | 〃 (`enabled=true`, `use_gt_vocabulary=true`) | GT object-only 보존 평가 — **object preservation 주장용** |
| `ensemble_openworld_filter` | 〃 (`enabled=true`, `use_gt_vocabulary=false`) | count/action/scene 잡음만 제거, non-GT object는 유지 — **hallucination/additional object 분석용** |

각 영상 × 모드 조합의 출력은
`outputs/etri_video_eval/remeasure_10videos/<mode>/<video_id>/heldout/`
아래로 자동 격리된다(경로 충돌 없음). 영상 목록은
`outputs/etri_video_eval_real_full_step50/baseline/`에서 `recon_frames/`가
있는 디렉터리를 자동 탐색하고, caption/GT는 각각
`data/etri_video_eval/captions/<video_id>.txt` /
`data/etri_video_eval/gt/<video_id>.json`을 자동 연결한다(파일이 없는
영상은 그 플래그만 빠진 채로 계속 진행 — 전체가 죽지 않는다).

```bash
cd sgdjscc_lab
conda activate ptest

# 무엇이 실행될지만 확인 (GPU/네트워크 사용 없음)
python scripts/batch_remeasure_owlv2_vqa_10videos.py --dry-run

# 실제 10개 영상 × 5개 모드 = 50개 job 전체 실행 (실 GPU + 최초 1회 가중치 다운로드)
python scripts/batch_remeasure_owlv2_vqa_10videos.py --device cuda:0

# 일부 영상/모드만
python scripts/batch_remeasure_owlv2_vqa_10videos.py \
    --videos 01_person_walk,03_dog_walk --modes owlv2,ensemble_gt_filter

# 중단된 배치를 이어서 (이미 끝난 (mode, video)는 건너뜀)
python scripts/batch_remeasure_owlv2_vqa_10videos.py --skip-existing

# 실행 없이 summary_metrics.csv/.md만 다시 생성
python scripts/batch_remeasure_owlv2_vqa_10videos.py --summary-only
```

기본은 **fail-fast**다 — job 하나가 실패하면(예: OWLv2 가중치 다운로드
실패, OOM) 그 자리에서 배치를 멈춘다. 50개 job을 끝까지 밀어붙이고 실패한
것만 나중에 골라내고 싶으면 `--continue-on-error`를 추가한다. 각 job의
stdout/stderr는 `<output-root>/<mode>/<video_id>/run.log`에 남는다.

배치가 끝나면(또는 `--summary-only`로) 다음 두 파일이 생성/갱신된다:

- `outputs/etri_video_eval/remeasure_10videos/summary_metrics.csv`
- `outputs/etri_video_eval/remeasure_10videos/summary_metrics.md`

컬럼: `video_id, mode, n_items, {mean_severity,ptc,sfr,sdi,
total_missing_objects,total_additional_objects,
temporal_hallucination_rate}_{clip_only,calibrated,diff}` — 각
`heldout/metric_delta.json`(`pipelines/heldout_remeasurement.py::_delta()`)을
그대로 반영한다.

#### 2026-07-28 완료된 10개 영상 실모델 재측정 결과

다음 명령으로 10개 영상 × 5개 모드 = 50개 job을 실행했고, 모든 job이
성공했다.

```bash
python scripts/batch_remeasure_owlv2_vqa_10videos.py --device cuda:0
```

완료 로그:

```text
Batch finished: 50/50 job(s) attempted — ok=50 skipped=0 failed=0 dry_run=0
Summary (50 row(s)) → outputs/etri_video_eval/remeasure_10videos/summary_metrics.csv
                        → outputs/etri_video_eval/remeasure_10videos/summary_metrics.md
```

모든 row는 `n_items=100`이며, 10개 영상 모두 동일한 실모델 baseline
`outputs/etri_video_eval_real_full_step50/baseline/<video_id>`의
`extracted_frames/` + `recon_frames/`를 재사용했다.

모드별 10개 영상 평균 diff는 다음과 같다. `severity`, `SFR`,
`missing/additional`, `temporal_hallucination_rate`는 낮을수록 좋고,
`PTC`는 높을수록 좋다.

| mode | severity diff | PTC diff | SFR diff | missing diff | additional diff | hallucination diff |
|---|---:|---:|---:|---:|---:|---:|
| `owlv2` | -0.1300 | +0.2036 | -0.0337 | -211.6 | -18.6 | -0.0314 |
| `vqa` | -0.2065 | +0.3337 | -0.0219 | -364.6 | -14.9 | -0.0253 |
| `ensemble_nofilter` | -0.2065 | +0.3337 | -0.0219 | -364.6 | -14.9 | -0.0253 |
| `ensemble_gt_filter` | -0.1872 | +0.3120 | -0.0182 | -72.4 | +0.0 | +0.0000 |
| `ensemble_openworld_filter` | -0.1857 | +0.2875 | -0.0190 | -169.4 | -14.9 | -0.0598 |

검증 결론:

- OWLv2-only는 10개 영상 전부에서 `mean_severity` 감소와 `PTC` 증가를 보였다.
- VQA-only는 OWLv2보다 더 큰 보정폭을 보였지만, object vocabulary 잡음에 더 민감하다.
- `ensemble_nofilter`는 VQA-only와 같은 집계값을 보여 현재 가중치/threshold에서는
  VQA 판단이 ensemble을 지배한다.
- `ensemble_gt_filter`는 GT object-only preservation 주장의 근거로 사용한다.
- `ensemble_openworld_filter`는 hallucination/additional object 분석의 근거로 사용한다.
- `ensemble_nofilter`는 caption-token contamination 확인용 baseline이며 최종 주장에 쓰지 않는다.

#### GT-object-only vs open-world filter — 해석 차이 (읽지 않고 결과만 보면 결론을 잘못 낸다)

`ensemble_gt_filter`(`use_gt_vocabulary=true`)는
`evaluators/object_vocabulary_filter.py::ObjectVocabularyFilter`가 GT
metadata가 있을 때 **GT의 `objects[].label`만** 그 프레임의 object
vocabulary로 쓰고 나머지는 전부 제외하는 closed-world 모드다(count/action/
scene 잡음뿐 아니라, GT가 언급하지 않은 진짜 hallucination 후보 object까지
같이 걸러진다 — `evaluators/object_vocabulary_filter.py`의
`ObjectVocabularyFilter` 클래스 docstring 참고). 그래서:

- **`ensemble_gt_filter`** → "GT에 있다고 명시된 object가 얼마나 잘
  보존됐는가"를 묻는다. **object preservation(의미 보존) 주장의 근거로
  쓴다.**
- **`ensemble_openworld_filter`**(`use_gt_vocabulary=false`)는 count/action/
  scene 잡음만 제거하고 GT에 없는 object(진짜 hallucination 후보 포함)는
  그대로 남긴다. **additional/hallucination object 분석(재구성이 없던
  object를 만들어내는지)의 근거로 쓴다.**
- **`ensemble_nofilter`**는 필터를 아예 끈 원래 상태다 — `one`/`walking`/
  `sidewalk` 같은 caption-noun 잡음이 그대로 missing/additional object로
  잡혀 severity/PTC/SFR 개선폭이 과대평가된다(자세한 배경은
  `evaluators/object_vocabulary_filter.py` 모듈 docstring 및 이 리포의
  이전 세션에서 `01_person_walk`로 직접 확인한 실측 diff 참고). **비교용
  baseline일 뿐, 최종 주장의 근거로 쓰면 안 된다.**

세 모드를 같은 영상에 대해 나란히 보고 싶다면 `summary_metrics.csv`를 `mode`
컬럼으로 필터링하거나 pivot하면 된다.

#### 단일 영상/모드를 수동으로 돌리고 싶다면

배치 드라이버 없이 그때그때 한 영상만 확인하고 싶으면 여전히
`scripts/remeasure_video_metrics.py`를 직접 호출할 수 있다(아래 "실행자가
바로 쓸 명령" 절 참고). `--gt-metadata` 파일 포맷은 다음 절에서 설명한다.

#### `--gt-metadata` format (읽기 전에 확인)

`data/etri_video_eval/gt/<video>.json`은 **segment-level 원본 GT**
(`{"n_frames": ..., "segments": [{"start_frame", "end_frame", "objects":
[{"label", "presence"}]}]}`)다 — `gt` presence backend가 기대하는
`{item_id: {object_name: bool}}` 형태가 아니다. `remeasure_video_metrics.py`는
이제 `--gt-metadata` 파일을 읽을 때 이 원본 형식을 자동 감지해
`convert_gt_to_presence()`로 변환한다(`looks_like_segment_level_gt()` —
`"segments"`/`"n_frames"` 키가 있으면 원본으로 판단). 그래서 위 명령처럼
`data/etri_video_eval/gt/${v}.json`을 그대로 넘겨도 안전하다.

이미 변환된 파일을 쓰고 싶다면 `scripts/run_etri_video_eval.py --stages
heldout`이 영상별로 저장하는 `outputs/etri_video_eval/heldout/<video>/
gt_presence.json`(`{"frame_00000": {"person": true}, ...}` 형식)을 그대로
써도 된다 — 두 파일 다 `item_id`가 `"frame_00000"` 문자열이고,
`--from-recon-frames`가 만드는 item은 정수 인덱스(`0, 1, 2, ...`)이지만
`pipelines/heldout_remeasurement.py::_lookup_gt_metadata()`가 둘 사이를
정규화해서 찾아준다(정수 인덱스 ↔ `frame_{idx:05d}` 양방향, `str`/`int` 혼용
모두 허용). 아무 쪽으로도 매칭이 안 되면 `None`으로 남고
`GtPresenceBackend`가 그 객체에 대해서만
`PresenceBackendUnavailableError`를 던진다(조용히 틀린 값으로 판정하지
않음).

`configs/etri_video_eval_owlv2.yaml`을 손으로 직접 호출할 때는 heldout 출력
경로가 모든 영상에 대해 같은 `outputs/etri_video_eval/manual_owlv2/heldout/`
로 덮어써진다는 점에 주의한다 — 영상별로 결과를 보존하려면 영상마다
`--config` 또는 heldout 출력 경로를 다르게 줘야 한다. 이 경로 충돌 문제를
자동으로 해결해주는 것이 바로 위 "10개 영상 OWLv2/VQA/ensemble 배치
재측정" 절의 `scripts/batch_remeasure_owlv2_vqa_10videos.py`다 — 10개
영상을 반복 실행하려면 수동 for문 대신 그 드라이버를 쓰는 것을 권장한다.

### pytest 환경에서 의존성/스모크 테스트

```bash
conda run -n ptest pytest tests/test_presence_backends.py \
    tests/test_heldout_remeasurement.py \
    tests/test_remeasure_video_metrics_cli.py \
    tests/test_packet_matcher.py \
    tests/test_batch_remeasure_owlv2_vqa_10videos.py -v
```

이 테스트들은 실제 OWLv2/VQA 가중치를 요구하지 않는다(mock/stub 기준). 단
`test_owlv2_backend_unavailable_when_transformers_lacks_owlv2`류 테스트는
존재하지 않는 model_id로 `transformers.from_pretrained()`를 호출해 실패
경로를 확인하므로 네트워크 접근이 있는 환경을 가정한다(실패해도 테스트
자체는 "정상적으로 실패했는가"만 확인하므로 무관하다).

## 기존 실모델 결과 재사용 가능 여부 — `outputs/etri_video_eval_real_full_step50`

**결론: 이미지는 byte-exact로 재사용 가능하다. 패킷은 아니다.**

이 배치는 `baseline` stage만 실행됐고, 각 영상 디렉터리
(`outputs/etri_video_eval_real_full_step50/baseline/<video>/`)에는:

- `extracted_frames/<video_stem>/frame_*.png` — 원본 입력 프레임
- `recon_frames/recon_*.png` — **실제 GPU 실모델(diffusion) 복원 결과**
- `temporal_frames.csv` — 프레임별 `role`(`keyframe`/`inter`)/decision/motion 등
- `segments.json`, `temporal_metrics.csv`, `keyframes.json`, `config.yaml`,
  `profiling_summary.json`, `progress.json`, `run.log`

는 있지만, **packet JSON(`<stem>.packet.json`/`<stem>.orig_packet.json`,
`utils/packet_io.py` 규약)은 어디에도 저장되지 않았다** — `evaluate_video.py`의
`baseline` stage가 애초에 packet dump를 호출하지 않기 때문이다(직접 확인:
`temporal_frames.csv`/`segments.json`에 `objects`/`relations` 같은 packet
필드가 전혀 없음).

이 때문에 `scripts/remeasure_video_metrics.py --from-packets`는 이 배치에
쓸 수 없다(애초에 packet 파일이 없으므로). 그래서 이번 준비 작업에서 새
경로를 추가했다:

- **`pipelines/heldout_remeasurement.py::items_from_recon_frame_dirs()`** +
  **`scripts/remeasure_video_metrics.py --from-recon-frames <run_dir>`** —
  `extracted_frames/`/`recon_frames/`를 직접 읽어 `reconstructed_image`
  텐서는 그 실모델 run이 실제로 만든 PNG 바이트 그대로 쓰고(재구성 재실행
  없음), 두 packet(`reference_packet`/`reconstructed_packet`)만
  `SemanticPacketExtractor`(CLIP + caption)로 **새로 재추출**한다.

### `--from-run`이 byte-exact 재사용인가? — **아니다**

`scripts/remeasure_video_metrics.py --from-run`(즉 `--input`을 주는 기본
모드, `_build_items_from_run()`)은 **처음부터 다시 재구성한다** —
`TemporalPipeline`을 새로 만들어 프레임을 다시 돌린다. 이는:

1. 그 config로 재구성이 여전히 가능한지 확인하는 것이지, 특정 과거 run이
   실제로 만든 결정/픽셀을 재생하는 것이 아니다.
2. diffusion 샘플링이 확률적이면(고정 seed가 없으면) 같은 config라도 매번
   다른 픽셀이 나올 수 있다.
3. `outputs/etri_video_eval_real_full_step50`가 만든 config.yaml은 원격
   GPU 서버(`/home/wilco/SangukBae/Semantic/sgdjscc_lab/...`) 경로를
   담고 있어, 이 로컬 환경에서 그 config.yaml을 그대로 다시 로드해도
   `model_root`/`input_path` 같은 절대경로가 이 머신의 실제 위치와 다르다
   (재구성 자체가 그대로 되지 않는다 — 새 config로 다시 잡아줘야 한다).

`scripts/remeasure_video_metrics.py`의 `_build_items_from_run()` docstring에
이미 이 사실이 명시돼 있다("does not replay a specific prior run's actual
decisions… prefer --from-packets for byte-for-byte remeasurement"). 이번에
추가한 `--from-recon-frames`는 그 문장의 "byte-for-byte" 조건을 **이미지에
한해** 만족시키는 세 번째 경로다 — 패킷까지 완전히 byte-exact로 만들려면
아래 후속 작업이 필요하다.

### 완전한 byte-exact 재검증을 원한다면 (후속 작업, 이번 범위 아님)

패킷까지 원본 run과 완전히 동일하게 재사용하려면 `evaluate_video.py`(또는
`video/temporal_pipeline.py`)가 각 프레임의 `orig_packet`/`recon_packet`을
`utils/packet_io.py::save_packet()`로 실제로 디스크에 dump하도록 배선해야
한다. 이는 `pipelines/eval_pipeline.py`/`video/temporal_pipeline.py`의 실행
경로를 건드리는 더 큰 변경이라 이번 "준비 단계" 범위에서는 하지 않았다(요청
사항의 "대규모 리팩터링은 하지 않는다"에 해당). 필요해지면:

1. `configs/video/default.yaml`에 `packet_dump.enabled`/`packet_dump.dir`류
   게이트를 하나 추가하고,
2. `video/temporal_pipeline.py`가 각 프레임 처리 후
   `save_packet(orig_packet, orig_packet_path(dir, frame_id))` /
   `save_packet(recon_packet, packet_path(dir, frame_id))`를 호출하도록
   (기본 OFF로) 연결한 뒤,
3. 다음 실모델 배치부터 `--from-packets`로 완전한 byte-exact 재검증을 한다.

## 정리 — 재현 또는 추가 검증할 때 쓸 명령

```bash
cd sgdjscc_lab
conda activate ptest   # 또는 실제 OWLv2/VQA weight를 내려받을 GPU 환경

# (A) 기존 실모델 recon 재사용 — GPU 재구성 없이 presence backend만 검증
python scripts/remeasure_video_metrics.py --config configs/etri_video_eval_owlv2.yaml \
    --from-recon-frames outputs/etri_video_eval_real_full_step50/baseline/01_person_walk \
    --captions data/etri_video_eval/captions/01_person_walk.txt \
    --device cuda:0

python scripts/remeasure_video_metrics.py --config configs/etri_video_eval_vqa.yaml \
    --from-recon-frames outputs/etri_video_eval_real_full_step50/baseline/01_person_walk \
    --captions data/etri_video_eval/captions/01_person_walk.txt \
    --device cuda:0

python scripts/remeasure_video_metrics.py --config configs/etri_video_eval_ensemble.yaml \
    --from-recon-frames outputs/etri_video_eval_real_full_step50/baseline/01_person_walk \
    --captions data/etri_video_eval/captions/01_person_walk.txt \
    --device cuda:0

# (B) 완전한 실모델 재구성부터 다시 검증하고 싶다면 (GPU, 시간 소요)
python scripts/evaluate_video.py --config configs/etri_video_eval_owlv2.yaml \
    --input data/etri_video_eval/processed/01_person_walk.mp4 \
    --captions data/etri_video_eval/captions/01_person_walk.txt \
    --snr 5 --device cuda:0 --save-video

# (C) 10개 영상 전체 × 5개 모드(owlv2/vqa/ensemble_nofilter/ensemble_gt_filter/
#     ensemble_openworld_filter) 배치 재측정. 2026-07-28 기준 완료된 결과가
#     outputs/etri_video_eval/remeasure_10videos/ 아래에 있다. 재현 또는 재실행 시 사용.
python scripts/batch_remeasure_owlv2_vqa_10videos.py --dry-run
python scripts/batch_remeasure_owlv2_vqa_10videos.py --device cuda:0

# 완료된 metric_delta.json들에서 summary만 다시 만들 때
python scripts/batch_remeasure_owlv2_vqa_10videos.py --summary-only
```

## 관련 문서

- [etri_strategy.md](./etri_strategy.md) — 5차(순서 8~10) 구현 결과, 무엇이
  "코드 기반 스캐폴드"이고 무엇이 실제 검증인지의 전체 맥락
- [etri_video_speed_optimization.md](./etri_video_speed_optimization.md) —
  `outputs/etri_video_eval_real_full_step50` 실모델 배치가 어떻게 만들어졌는지

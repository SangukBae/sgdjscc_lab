---
status: active
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: 076a26d
supersedes:
---

> [← 문서 색인](../README.md)

# 재현성 결과 저장소 (`results/`)

- 연결 문서
  - 저장소 색인: [../../results/README.md](../../results/README.md)
  - checkpoint/`paper_mode` 재현: [reproducibility.md](./reproducibility.md)
  - 평가 실행 절차: [evaluation.md](./evaluation.md)
  - manifest 유틸리티: [src/sgdjscc_lab/utils/run_manifest.py](../../src/sgdjscc_lab/utils/run_manifest.py)

## 배경

- 문제
  - `outputs/`는 전부 git 비추적(`.gitignore`) — 로그·프레임·packet·mp4 포함
  - 대용량 실행 결과와 핵심 지표가 분리되지 않아 재현 판단이 어려움
- 해결
  - `results/`를 git 추적 디렉터리로 신설
  - CSV/JSON/README/config만 복사 보존, 이미지·packet·mp4는 원본 경로만 기록
  - run마다 `manifest.json`으로 재현 조건을 명시

## `results/` 구조

- `results/README.md`: 저장소 전체 설명
- `results/registry.csv`: run 색인(1행 = 1 run)
- `results/<run_id>/`: run별 사본 + `manifest.json`

## run manifest 원칙

- 기록 항목
  - git commit / dirty 여부 (`include_git=False`로 명시 생략 가능)
  - 실행 명령 — `{text, argv, source}`. `source`는 `captured`(실행 시점 실측) /
    `reconstructed`(문서·로그에서 사후 재구성) / `unknown` 중 하나만 허용.
    **재구성한 명령을 실제 실행된 명령처럼 표현 금지** — 반드시 `reconstructed`로 표기
  - config — 아래 "resolved vs config_source" 참고
  - seed — 실제 값 / `"unknown"`(미확인) / `"not_set"`(seed 없음이 확인됨).
    **`null` 금지** — `None` 전달 시 `ValueError`로 즉시 실패(코드가 강제)
  - dataset sha256
  - checkpoint 상태·파일별 sha256
    - `recorded`: 파일 검증 완료
    - `unknown`: 사용 파일 미확인
    - `not_set`: checkpoint 미사용 확인
  - Python·CUDA·GPU 환경 (`include_environment=False`로 명시 생략 가능)
  - 평가기(evaluator) 버전
  - 원본 `outputs/` artifact 경로
  - **artifacts** — `results/`로 복사한 파일마다 사본 경로+sha256, 원본
    경로+sha256, 일치 여부(`matches`: 둘 다 구해졌을 때만 `true`/`false`,
    아니면 `"unknown"`)
  - exact(실측) vs proxy(추정) 필드 구분
  - NaN·실패 프레임 수
- **미확인 값은 추측하지 않고 문자열 `"unknown"`으로 기록** — 스키마 필드는 항상 존재하되 값만 `unknown`

### resolved config vs config_source — 절대 혼동 금지

- `_defaults_` fragment 목록만 있는 root config 파일은 **resolved config가 아니다**
  (fragment 병합 결과, CLI override가 전혀 반영 안 됨)
- `resolved_config_path`/`resolved_config`: 실행 당시 최종 merge+override 결과를
  **확실히 복원할 수 있을 때만** 사용 → `resolved_config.status: "resolved"`
- `config_source_path`: 확신이 없으면 이걸 사용 → `status: "config_source_only"`,
  `resolved`는 `"unknown"`으로 고정, `config_source_note`에 왜 resolved가 아닌지 기록
- 경로 기반 config는 파일이 없거나 sha256 계산에 실패하면 `ValueError`
- 과거 run 예시: `results/transmission_20260818/config_source/composed.yaml`은
  9개 `--configs` variant(fixed/skem × awgn/int16/int8/int6/int4)가 공유한 root
  config일 뿐 — variant별 실제 채널/bit_depth override는 이 파일에 없어
  `config_source_only`로 기록

### 과거 run(사후 이관) 예시: `results/transmission_20260818/manifest.json`

- 실행 시점(2026-08-18, 원격 GPU 서버) git/환경 정보가 당시 기록되지 않아 `unknown`
  (`include_git=False`, `include_environment=False`로 생성)
- 명령은 `docs/experiments/2026-08-18_transmission_reduction.md`에서 재구성
  → `command.source: "reconstructed"`
- seed는 스크립트에 `--seed` 인자가 없음을 grep으로 확인 → `"not_set"`(unknown 아님)
- `manifest.json`의 `extra.*_note` 필드에 각 판단 근거를 남김 — 현재 HEAD로
  되짚어 추측하지 않음

## 앞으로 새 run을 등록하는 절차

1. 평가/실험 스크립트 실행 후 `outputs/<run_dir>/`에서 핵심 CSV/JSON/README/config만
   `results/<run_id>/`로 복사 (이미지·packet·mp4 등 대용량 바이너리는 복사하지 않음)
2. `sgdjscc_lab.utils.run_manifest.build_run_manifest()` 호출
   - 가능하면 실행 스크립트 안에서 직접 호출 — git/Python/CUDA/GPU는 자동 수집됨
   - 사후 이관 시에는 실행 당시 값을 확인할 수 없는 필드를 `unknown`으로 명시
3. `write_run_manifest(path, manifest)`로 `results/<run_id>/manifest.json` 저장
4. `results/registry.csv`에 한 행 추가
5. `docs/experiments/YYYY-MM-DD_<name>.md`에 실험 해설이 있다면 `doc_link`로 연결
6. `results/README.md`의 "현재 등록된 run" 표 갱신

## 예시 — 사후 이관(과거 run)

```python
from sgdjscc_lab.utils.run_manifest import build_run_manifest, write_run_manifest, NOT_SET

manifest = build_run_manifest(
    run_id="transmission_20260818",
    command="python scripts/run_transmission_reduction_eval.py --configs ... --output-root outputs/...",
    command_source="reconstructed",   # 실행 시점 캡처가 아니라 문서에서 재구성한 값
    seed=NOT_SET,             # 스크립트에 --seed 인자 없음(grep으로 확인) — unknown 아님
    config_source_path="results/transmission_20260818/config_source/composed.yaml",
    # resolved_config_path가 아님: 이 파일은 9개 variant가 공유하는 root config일 뿐,
    # variant별 실제 채널/bit_depth override가 반영된 최종본이 아니므로 resolved 불가
    exact_fields=["packet_components.csv:total_bundle_bytes (실제 .sgbundle 파일 크기)"],
    proxy_fields=["keyframe_selection.csv:estimated_wire_bytes (proxy, FEC 모델 없음)"],
    nan_or_failure_counts={"fixed_int16": 52, "fixed_int4": 0},
    artifacts={
        "aggregate.csv": {
            "copied_path": "results/transmission_20260818/aggregate.csv",
            "original_path": "outputs/transmission_reduction_full_20260818_043425/aggregate.csv",
        },
    },
    include_git=False,           # 실행 당시 commit 미기록 → unknown 유지(현재 HEAD로 추측 금지)
    include_environment=False,   # 실행 당시 환경 미기록 → unknown 유지
)
write_run_manifest("results/transmission_20260818/manifest.json", manifest)
```

## 예시 — 앞으로의 실측 run (환경 자동 수집)

```python
from sgdjscc_lab.utils.run_manifest import build_run_manifest, write_run_manifest, UNKNOWN, NOT_SET

raw_seed = cfg.get("seed")  # None이면 "설정 안 됨"이 확인된 것 — unknown이 아니라 not_set
seed = NOT_SET if raw_seed is None else raw_seed

manifest = build_run_manifest(
    run_id="my_new_run",
    command_argv=sys.argv,          # argv 배열 보존 + shlex.join 기반 표시 문자열 생성
    command_source="captured",   # 실행 스크립트 안에서 직접 호출 → 실측값
    seed=seed,
    resolved_config=OmegaConf.to_container(cfg, resolve=True),  # 이미 merge+override 반영된 cfg
    checkpoints={"jscc": cfg.model_root + "/JSCC_model.pth"},
    original_artifact_paths={"output_root": str(output_root)},
    # include_git/include_environment 기본값 True → commit/dirty/python/torch/cuda/gpu 자동 수집
)
write_run_manifest(output_root / "manifest.json", manifest)
```

## 테스트

- `tests/test_run_manifest.py` (35개) — git 상태 감지(clean/dirty/non-repo),
  sha256, `unknown`/`not_set` 구분, `seed=None` 거부, `command_source` 검증,
  argv 보존, config·checkpoint 경로 검증, `resolved`/`config_source_only` 분기,
  artifact 쌍 해시·`matches` 판정, `include_git=False` 격리, round-trip 검증
- `get_cuda_env()` mock 테스트 — torch 미설치, 깨진 CUDA 런타임(`OSError`),
  GPU 이름 조회 실패(`RuntimeError`)를 각각 monkeypatch로 주입해 예외 없이
  `unknown`으로 저하되는지 확인 + `sys.meta_path`를 이용한 실제 import 단계
  `OSError` 시뮬레이션 1건(mock이 아닌 end-to-end 확인)
- GPU/torch 불필요 — 전부 CPU에서 실행

> [← docs index](./README.md)

# ETRI Development Roadmap

This document reorganizes the ETRI task into a practical development order.
It combines:

- the eight ETRI task goals from [etri_overview.md](./etri_overview.md)
- the prioritized SGD-JSCC limitations `A`, `B`, `C`, `D` from
  [limitation_reference_map.md](./limitation_reference_map.md)

The ordering principle is simple: build the measurement and software baseline
first, then add research improvements on top of that baseline.

## Phase 마스터 스위치

Phase 4와 Phase 5의 모든 확장 기능은 **기본값 `false`**로 꺼져 있다.
두 개의 상위 게이트 플래그로 한 번에 제어한다.

```yaml
# configs/eval/default.yaml (또는 composed config)
use_phase4: false   # Phase 4-A/B 전체 비활성화 (기본값)
use_phase5: false   # Phase 5-A/B/C 전체 비활성화 (기본값)
```

| 조합 | 동작 |
|------|------|
| 둘 다 `false` | Phase 1~3 기본 경로만 실행 |
| `use_phase4: true` | Phase 4-A/B 개별 플래그가 효력 발생 |
| `use_phase5: true` | Phase 5-A/B/C 개별 플래그가 효력 발생 |
| 둘 다 `true` | 전체 확장 기능 활성 가능 |

**중요**: `use_phase5: true`여도 `use_phase4`는 자동으로 켜지지 않는다.

개별 플래그(`use_packet_eval`, `use_channel_conditioning` 등)는 상위 마스터 스위치가
`true`일 때만 실제로 반영된다. 마스터 스위치가 `false`이면 개별 플래그 값은 무시된다.

자세한 사용법: [phase4.md#master-switch](./phase4.md#master-switch) / [phase5.md#master-switch](./phase5.md#master-switch)

---

## Recommended Order

| Step | Development item | Why this comes here | Main source |
|---|---|---|---|
| 1 | Preserve the original SGD-JSCC path and extension rule | Baseline reproducibility must be fixed first; otherwise later comparisons are not trustworthy. | [etri_overview.md](./etri_overview.md) |
| 2 | Organize the modular software structure | `channels/`, `guidance/`, `models/`, `pipelines/`, `evaluators/` must be stable before research changes accumulate. | [etri_overview.md](./etri_overview.md) |
| 3 | Build the end-to-end evaluation framework skeleton | Input -> channel -> reconstruction -> evaluation -> `results.csv` logging should exist before optimization work starts. | [etri_overview.md](./etri_overview.md) |
| 4 | Fix the evaluation philosophy around semantic intent | The project should define success around semantic preservation, not just pixel fidelity. | [etri_overview.md](./etri_overview.md) |
| 5 | Implement the evaluator suite including hallucination metrics | CLIP, object preservation, missing/additional objects, hallucination, and quality metrics are prerequisites for meaningful experiments. | [etri_overview.md](./etri_overview.md) |
| 6 | Integrate SRS as the headline metric | SRS should be finalized only after its component metrics are stable. | [etri_overview.md](./etri_overview.md) |
| 7 | Improve limitation `A`: reduce hallucination and semantic inconsistency under guidance | The core ETRI problem is not pixel fidelity alone, but preventing plausible-yet-wrong reconstructions and improving verified semantic reliability. | [limitation_reference_map.md](./limitation_reference_map.md) |
| 8 | Improve limitation `B`: make semantic side information robust and lightweight | Caption and edge guidance must remain useful under corruption while avoiding unnecessary transmission overhead. | [limitation_reference_map.md](./limitation_reference_map.md) |
| 9 | Improve limitation `C`: reduce diffusion reconstruction latency | Few-step decoding and consistency-style acceleration should be evaluated after semantic reliability metrics and verifier loops are already in place. | [limitation_reference_map.md](./limitation_reference_map.md) |
| 10 | Improve limitation `D`: add blind / fading channel robustness beyond strong CSI assumptions | Channel-conditioned and blind reconstruction should be expanded after the core semantic reliability and latency path is stable. | [limitation_reference_map.md](./limitation_reference_map.md) |
| 11 | Separate guide corruption models from channel noise | Once richer guidance exists, guide-specific corruption rules become necessary for realistic experiments. | [etri_overview.md](./etri_overview.md) |
| 12 | Lock the fair comparison protocol | Final protocol fixing should come last, after the pipeline, metrics, and improved methods are all settled. | [etri_overview.md](./etri_overview.md) |

## Current Implementation Status

| Step | Status | What is implemented now | Remaining gap |
|---|---|---|---|
| 1 | Complete | The original SGD-JSCC forward path is preserved and all new features are opt-in extensions layered around it. | None at the framework-rule level. |
| 2 | Complete | The package is already split into modular `channels/`, `guidance/`, `models/`, `pipelines/`, `evaluators/`, `controllers/`, `acceleration/`, and `video/` areas. | None for the main software structure. |
| 3 | Complete | `scripts/evaluate.py` and `pipelines/eval_pipeline.py` provide an end-to-end evaluation path from input to reconstruction, metrics, and CSV logging. | None for the baseline image evaluation loop. |
| 4 | Complete | The evaluation stack is semantic-first: CLIP, object preservation, hallucination, and SRS are first-class metrics beyond PSNR/SSIM/LPIPS. | None at the evaluation-policy level. |
| 5 | Complete | Quality metrics, CLIP metrics, packet-aware metrics, temporal metrics, and VQA-based hallucination evaluation are implemented. | Some advanced semantic evaluators are still heuristic rather than fully learned. |
| 6 | Complete | `srs_base`, `srs_packet`, and `srs_v2` are integrated into the evaluation path and can be enabled from config. | Final paper-level weight tuning may still change. |
| 7 | Partial | Packet-aware metrics, regeneration search, adaptive guidance, VQA-style hallucination checks, and `SRS-v2` support the semantic reliability path. | Guidance errors can still produce plausible-but-wrong outputs, and verification remains partly heuristic. |
| 8 | Partial | Semantic delta transmission simulation, packet reuse, caption/guide corruption hooks, and `overhead_reduction` reporting are implemented. | True semantic side-information coding, stronger corruption robustness, and drop-aware control remain incomplete. |
| 9 | Partial / Scaffolded | DDIM step-budget control, dynamic routing, early exit, latency profiling, and benchmark CLIs are implemented. | A trained distilled consistency decoder is still a placeholder rather than a finished model. |
| 10 | Partial | Rayleigh / fast-fading / packet-drop channels, channel measurement bundles, and channel-conditioned inference modes are implemented. | Blind robustness is still limited and stronger non-AWGN validation is still needed. |
| 11 | Limited | Supporting pieces exist, such as `packet_drop` channel support and segmentation-region dropout as an intended corruption mechanism. | A full guide-specific corruption framework matching the ETRI overview is not finished yet. |
| 12 | Partial | The main evaluation loop already supports packet eval, channel conditioning, SRS-v2, regeneration search, and video evaluation under shared configs. | The final fixed comparison protocol across all baselines and ablations is not fully locked down yet. |

## Short Version

```text
baseline preservation
-> modular structure
-> end-to-end pipeline
-> semantic-first evaluation philosophy
-> evaluator suite
-> SRS integration
-> limitation A improvement
-> limitation B improvement
-> limitation C improvement
-> limitation D improvement
-> guide corruption model
-> fair comparison protocol
```

## Why Limitations A, B, C, D Appear in This Order

- `A` comes first because hallucination and semantic inconsistency are the most
  direct failure modes against the ETRI semantic-reliability objective.
- `B` follows because side information is only useful if it remains robust
  under corruption and does not consume unreasonable transmission budget.
- `C` comes after the core semantic-reliability work because latency reduction
  must be judged against verified semantic quality, not runtime alone.
- `D` comes last among the limitation-improvement steps because blind/fading
  robustness is important, but follows the core reliability and latency path.

## Practical Interpretation

- Steps `1` to `6` establish the ETRI evaluation and software baseline.
- Steps `7` to `10` implement the priority SGD-JSCC limitation improvements.
- Steps `11` to `12` finalize realistic experiment design and paper-quality
  comparison protocol.

## Training Reproducibility (paper 3-stage + extension)

A stage-aware training framework now backs paper reproduction (see
[training_scaffold.md](./training_scaffold.md)):

- **Stage 1 / 2 / 3** (`jscc`, `text_dm`, `controlnet`) are implemented as
  separate runners with stage-specific datasets, losses, and a hard freeze
  policy. Stage 3 supports two edge transports: `shared_vae` and a dedicated
  `edge_jscc` link (`models/edge_jscc.py`).
- **Operational scale**: step-based training (`max_steps`,
  `save/val/log_every_steps`), gradient accumulation, and AMP enable the
  paper's ~250k-step DM schedule on real data; `global_step` is checkpointed
  and resumed.
- **Extension**: an `end_to_end_ft` stage jointly fine-tunes JSCC + DM.

Remaining gaps: the patch-GAN and `edge_jscc` codec weights are structural
stand-ins (not the paper's LDM-GAN / BCE-Dice-trained edge codec), the
`end_to_end_ft` recon path uses a single-step denoise rather than the full
reverse process, and the ~14M-pair open dataset is not bundled (only the loader
interface is provided).

## Related Documents

- [etri_overview.md](./etri_overview.md) — ETRI task goals and framework scope
- [limitation_reference_map.md](./limitation_reference_map.md) — prioritized SGD-JSCC limitations and references
- [phase4.md](./phase4.md) — Phase 4 status and design
- [phase5.md](./phase5.md) — Phase 5 status and design

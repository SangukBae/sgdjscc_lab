> [← docs index](./README.md)

# Phase 5 — Plan & Implementation Status

- [Master Switch](#master-switch)
- [Phase 5 Plan](#phase-5-plan)
- [Phase 5 Implementation Status](#phase-5-implementation-status)

---

## Master Switch

All Phase 5 features are **off by default**. A single top-level flag
`use_phase5` controls the entire phase:

```yaml
# configs/eval/default.yaml (or your composed config)
use_phase5: false   # default — all Phase 5-A/B/C features disabled
```

**Rule**: when `use_phase5: false`, every per-feature flag listed below
(`use_channel_conditioning`, `acceleration.*`, `use_srs_v2`,
`use_vqa_hallucination`, `use_regeneration_search`) is ignored at runtime,
even if the flag is explicitly set to `true` in the config.

**Important**: `use_phase5: true` does **NOT** automatically enable Phase 4
features. Phase 4 and Phase 5 are independent master switches.

### Enabling Phase 5 only (no Phase 4)

```yaml
use_phase4: false   # Phase 4-A/B stays off
use_phase5: true

use_channel_conditioning: true
use_srs_v2: true
use_vqa_hallucination: false
use_regeneration_search: false
```

### Enabling Phase 4 + Phase 5 (full stack)

```yaml
use_phase4: true
use_phase5: true
```

The preset `configs/composed_phase5_full.yaml` has both switches set to true
with all extended flags enabled.

### Helper function

All runtime checks go through `sgdjscc_lab.phase_gates.effective_flag`:

```python
from sgdjscc_lab.phase_gates import effective_flag, phase5_enabled

# Returns False when use_phase5 is false, regardless of the raw flag value.
use_channel_cond = effective_flag(cfg, "use_channel_conditioning", phase=5)
```

---

# Phase 5 Plan

Phase 5 is the `model and channel research phase`.
Unlike Phase 4, Phase 5 is allowed to modify the conditioning interface around
the diffusion decoder and add new channel abstractions beyond AWGN.

Phase 5 is split into two main tracks and one supporting evaluation track:

1. `Phase 5-A`: DiffCom-inspired CSI-free channel conditioning
2. `Phase 5-B`: low-latency diffusion and consistency distillation
3. `Phase 5-C`: stronger semantic verification and multi-strategy regeneration

## Phase 5-A: DiffCom-inspired channel-conditioned diffusion

Primary references:

- `paper/diffcom/README.md`
- `paper/diffcom/main_diffcom.py`
- `paper/diffcom/conditioning_method/diffcom.py`
- `paper/diffcom/guided_diffusion/measurement.py`

What we take from DiffCom:

1. the received channel signal itself is useful as a diffusion condition
2. channel observation should be exposed through an operator-style abstraction
3. blind reconstruction should be evaluated explicitly
4. conditioning should support:
   - standard channel-aware mode
   - high-fidelity joint mode
   - blind mode

Planned files:

```text
src/sgdjscc_lab/
├── channels/
│   ├── rayleigh.py
│   ├── fast_fading.py
│   └── packet_drop.py
├── models/
│   ├── channel_condition_encoder.py
│   ├── diffusion_wrapper_channel.py
│   └── reliability_head.py
├── controllers/
│   └── channel_condition_policy.py
└── pipelines/
    └── channel_conditioned_infer.py
```

Implementation steps:

1. `channels/rayleigh.py`
   - implement slow Rayleigh fading API compatible with the current AWGN
     channel abstraction
   - output:
     - noisy latent
     - equalized latent
     - channel gain
     - noise variance
2. `channels/fast_fading.py`
   - implement symbol- or block-level fading with configurable block length
   - keep compatibility with current patch-level image inference
3. expose receiver-side measurements from the current SGD-JSCC path
   - modify the internal inference return path so intermediate values can be
     optionally collected:
     - `encode_features_hat`
     - `mask_token`
     - `power_scalar`
     - estimated SNR
     - optional phase estimate
4. `channel_condition_encoder.py`
   - compress raw channel observations into condition tokens
   - first condition vector:
     - noisy latent feature
     - normalized power scalar
     - mask or reliability map
     - SNR estimate
     - phase estimate
   - later extension:
     - fast fading distortion feature
     - per-token reliability
5. `diffusion_wrapper_channel.py`
   - build a channel-conditioned wrapper around the current diffusion backbone
   - first implementation will use lightweight adapters instead of replacing
     the whole denoiser:
     - FiLM-style conditioning
     - extra context tokens
     - optional cross-attention branch for channel tokens
6. `channel_conditioned_infer.py`
   - support three experimental modes inspired by DiffCom naming:
     - `latent_conditioned`
     - `joint_conditioned`
     - `blind_conditioned`
7. blind experiment design
   - unknown SNR AWGN
   - slow fading without CSI
   - fast fading with imperfect or no CSI
   - compare:
     - estimated-SNR controller only
     - channel-token conditioning
     - channel-token conditioning + regeneration

Key implementation note:

The `operator.observe_and_transpose()` pattern from DiffCom will be mirrored in
`sgdjscc_lab` as an internal measurement bundle abstraction so that every
channel experiment returns both `reconstruction` and `receiver evidence`.
This is the cleanest bridge from the current deterministic image pipeline to
DiffCom-style posterior-guided reconstruction.

## Phase 5-B: low-latency diffusion and consistency distillation

Primary references:

- `paper/LDM-enabled-SemCom-system/train_DIV2K/consistency_models.py`
- `paper/LDM-enabled-SemCom-system/train_DIV2K/consistency_model_training.py`
- `paper/LDM-enabled-SemCom-system/train_DIV2K/t_calculate.py`
- `paper/LDM-enabled-SemCom-system/train_MNIST/eavluation.py`

What we take from the low-latency LDM code:

1. explicit consistency sampling utilities
2. consistency training and distillation scaffolding
3. Karras schedule-based sampling
4. timing measurement for EECD-style latency analysis
5. one-step and few-step sampling comparison

Planned files:

```text
src/sgdjscc_lab/
├── acceleration/
│   ├── ddim_sampler.py
│   ├── consistency_decoder.py
│   ├── early_exit.py
│   └── latency_profiler.py
└── scripts/
    ├── benchmark_latency.py
    └── benchmark_sampling.py
```

Implementation steps:

1. baseline latency benchmark
   - measure the current 50-step pipeline on the same hardware profile used for
     our internal experiments
   - report:
     - end-to-end latency
     - decoder latency
     - per-step denoising cost
2. DDIM ablation
   - add configurable schedules for:
     - 50-step
     - 20-step
     - 10-step
     - 5-step
   - keep SRS and LPIPS as the main quality constraints
3. `consistency_decoder.py`
   - prototype a few-step decoder interface inspired by
     `ConsistencySamplingAndEditing`
   - initial target:
     - 1-step
     - 2-step
     - 5-step
4. `latency_profiler.py`
   - port the idea of `t_calculate.py`
   - report:
     - decoder latency
     - total latency
     - effective speedup over the 50-step baseline
5. `early_exit.py`
   - evaluate intermediate reconstructions during denoising
   - stop early when:
     - SRS exceeds a threshold
     - improvement over the previous step is below a tolerance
6. dynamic routing policy
   - combine channel state and predicted reliability:
     - high SNR + high predicted SRS -> 1 to 5 steps
     - mid SNR -> 10 to 20 steps
     - low SNR or blind channel -> 30 to 50 steps

Expected outputs:

- quality vs latency curve
- SRS vs latency curve
- step-count recommendation table by SNR regime

## Phase 5-C: stronger verifier and regeneration search

Phase 5-C is the supporting evaluation layer that makes Phase 5-A and 5-B
publishable and easier to defend.

Planned files:

```text
src/sgdjscc_lab/
├── evaluators/
│   ├── hallucination_vqa.py
│   ├── semantic_reliability_v2.py
│   └── regeneration_search.py
└── controllers/
    └── adaptive_search_policy.py
```

Implementation steps:

1. add VQA-style hallucination checks as a stronger alternative to purely CLIP
   heuristics
2. produce `SRS-v2` with packet-aware and temporal-aware terms
3. replace one-shot regeneration with multi-strategy search:
   - strong text / weak edge
   - weak text / strong edge
   - unconditional fallback
   - channel-conditioned retry
4. choose final output by highest verified SRS, not only by first successful
   reconstruction

## Phase 5 completion criteria

Phase 5 will be considered complete when all of the following are true:

1. AWGN and Rayleigh channel settings are both supported
2. at least one blind or imperfect-CSI mode is implemented and benchmarked
3. channel-conditioned diffusion improves semantic reliability in at least one
   nontrivial unknown-channel setting
4. few-step or distilled decoding provides a clear latency reduction with
   bounded SRS degradation
5. stronger verifier and regeneration search are integrated into the
   experiment loop

## Phase 5 experimental design

Channels:

- AWGN with known and unknown SNR
- slow Rayleigh fading
- fast fading with configurable block length
- packet drop / semantic packet corruption

Baselines:

1. original SGD-JSCC path in `sgdjscc_lab`
2. adaptive-guidance-only Phase 4 model
3. channel-conditioned model without blind mode
4. blind channel-conditioned model
5. few-step consistency decoder variants

Metrics:

- quality:
  - PSNR
  - SSIM
  - LPIPS
- semantics:
  - CLIP image-image
  - CLIP text-image
  - object preservation
  - relation consistency
  - attribute consistency
  - segmentation consistency
- reliability:
  - SRS
  - SRS-v2
  - hallucination rate
  - regeneration success rate
- efficiency:
  - sec/image
  - sec/frame
  - denoising steps
  - speedup over the 50-step baseline

## Recommended execution priority

The practical implementation order for the remaining research priorities is:

1. `FAST-GSC`-inspired Phase 4 packet and keyframe framework
2. low-latency LDM-inspired Phase 5-B acceleration
3. stronger verifier / regeneration work in Phase 5-C
4. `DiffCom`-inspired Phase 5-A channel conditioning

This order is intentional:

- first solve the core ETRI-facing limits:
  explicit semantics, stronger guidance, side-information control
- then reduce diffusion latency so the prototype is practically runnable
- finally deepen channel adaptation and fast-fading robustness as a follow-up
  track

---

# Phase 5 Implementation Status

Phase 5 is implemented as a **config-driven, opt-in scaffold**: every feature is
off by default, so the Phase 1–4 image/video paths are byte-for-byte unchanged
unless explicitly enabled. The goal (per plan) was *experimentally runnable
structure + testable minimal behaviour + Phase-4 compatibility*, not trained SOTA
reproduction.

## Phase 5-A — DiffCom-inspired channel-conditioned diffusion

| Area | Module(s) | Status |
|---|---|---|
| Channels | `channels/rayleigh.py`, `channels/fast_fading.py`, `channels/packet_drop.py` (+`channels/__init__.build_channel`) | **implemented** — AWGN-compatible `transmit()` + rich `observe()` |
| Measurement bundle | `channels/measurement.py` (`MeasurementBundle`) | **implemented** — received/equalized/gain/noise_var/mask/SNR/reliability + optional real receiver evidence |
| Receiver-evidence hook | `pipelines/infer_pipeline.py` (`measurement_out`) | **implemented** — observation-only; fills `encode_features_hat`/`mask_token`/`power_scalar`/SNR-est |
| Condition encoder | `models/channel_condition_encoder.py` | **implemented (training-free)** — `stats` mode (no params) + `linear` learnable hook |
| Reliability head | `models/reliability_head.py` | **implemented (heuristic)** — SNR/gain/mask confidence + learnable hook |
| Image-level measurement consistency | `channels/measurement.py` (`aggregate_bundles`) + `pipelines/infer_pipeline.py` (`run_image_channel_conditioned`) | **implemented** — channel evidence is aggregated over all patches and the decoder reuses the same received latent that produced that evidence |
| Conditioned wrapper | `models/diffusion_wrapper_channel.py` | **adapter-level** — confidence scales guidance/steps; blind→blind-SNR; condition tokens attached as `cfg.channel_condition_tokens`; encoder/policy built from the `channel_condition` config block |
| Policy / pipeline | `controllers/channel_condition_policy.py`, `pipelines/channel_conditioned_infer.py` | **implemented** — `latent`/`joint`/`blind` modes; config-gated `use_channel_conditioning`; **wired into `scripts/evaluate.py`** (the main eval loop routes reconstruction through it) |

Integration: when `use_channel_conditioning` is set, `evaluate_dataset` builds a
`OnePassChannelConditionedInference` and reconstructs each image through the current
**one-pass** path. `run_image_channel_conditioned` performs encode+transmit once
per patch, aggregates the resulting per-patch bundles with `aggregate_bundles`
into one **image-level** measurement, resolves the conditioned cfg, and then
decodes while reusing the same received latent that generated the evidence. In
other words, the policy does not see a throwaway measurement from a separate
forward pass.

What is **approximation, not full DiffCom**: the noisy received signal enters
reconstruction via (a) the real channel inside `JSCCModel.channel`, (b)
reliability-scaled `cfg` (guidance/steps/blind-SNR), and (c) the received latent
as the diffusion init. The encoded condition tokens are attached to the cfg but
**not consumed by the frozen SGD-JSCC denoiser** — FiLM / cross-attention /
posterior-gradient guidance need a condition-aware (retrained) denoiser and are
left as the next step.

## Phase 5-B — low-latency diffusion & consistency distillation

| Area | Module(s) | Status |
|---|---|---|
| Step-budget / DDIM ablation | `acceleration/ddim_sampler.py` (`build_sampler_cfg`) | **implemented + wired** — `evaluate_dataset` applies it to the per-SNR cfg, so `acceleration.sampler_steps` actually changes the denoising budget; off-by-default |
| Dynamic routing | `acceleration/ddim_sampler.py` (`dynamic_step_budget`) | **implemented** — SNR/confidence → step budget |
| Karras schedule | `acceleration/ddim_sampler.py` (`karras_schedule`) | **implemented** — ported from the LDM SemCom code |
| Consistency decoder | `acceleration/consistency_decoder.py` | **interface + few-step math** — `baseline`/`fewstep`/`distilled_placeholder`; distilled student is a clean placeholder (falls back to few-step) |
| Early exit | `acceleration/early_exit.py` | **checkpoint-level + intra-sampler** (see "Resolved limitations" below) |
| Latency profiler | `acceleration/latency_profiler.py` | **implemented** — total/decoder/per-step, CUDA-synced |
| Benchmark CLIs | `scripts/benchmark_latency.py`, `scripts/benchmark_sampling.py` | **implemented** — quality-vs-latency sweep |

What is **placeholder**: a trained consistency/distilled student decoder (teacher-
student training is not reproduced); the decoder API + few-step sampling math +
evaluation path are complete so a distilled model can be dropped in.

## Phase 5-C — stronger verifier & multi-strategy regeneration

| Area | Module(s) | Status |
|---|---|---|
| VQA hallucination | `evaluators/hallucination_vqa.py` + `evaluators/vqa_backend.py` | **implemented + wired** — injectable `vqa_fn`; real local backends (`mock`/`blip2`/`llava`/`mplug`) with CLIP fallback (see "Resolved limitations") |
| SRS-v2 | `evaluators/semantic_reliability_v2.py` (+ `EvalContext._get_srs_v2`, `use_srs_v2`) | **implemented + wired** — `_compute_metrics` runs the full `SemanticReliabilityV2Evaluator` (base + packet + temporal + VQA), not a shortcut; reuses the computed base SRS |
| Regeneration search | `evaluators/regeneration_search.py` (+ `eval_pipeline._run_regeneration_search`, `use_regeneration_search`) | **implemented + wired** — strong/weak text, unconditional, channel-conditioned retry (genuinely re-runs the channel-conditioned path); selects best by the **configured** `regeneration_search.verify_metric` (`srs` or `srs_v2`, the latter scoring each candidate with packet + VQA layers); writes `regeneration_strategy` |
| Search ordering | `controllers/adaptive_search_policy.py` | **implemented** — orders strategies by failure mode + channel state |

## Reference mapping (Phase 5)

- **DiffCom** (`paper/diffcom/…`): the `observe()`→`MeasurementBundle` pattern
  mirrors `operator.observe()`; the `latent`/`joint`/`blind` modes mirror DiffCom's
  conditioning modes; the blind path mirrors `blind_diffcom` (unknown-SNR /
  imperfect-CSI). Posterior-gradient consistency conditioning is *not* reproduced
  (frozen denoiser).
- **LDM-enabled SemCom** (`paper/LDM-enabled-SemCom-system/…`): `karras_schedule`
  and the consistency few-step sampling loop are ported from
  `consistency_models.py` / `t_calculate.py`; the latency profiler mirrors the
  `t_calculate.py` timing idea. Consistency *training* is not reproduced.
- **FAST-GSC continuity**: the Phase-4 staged-prompt schedule, packet semantics and
  temporal pipeline are untouched; SRS-v2 consumes the Phase-4 `temporal_srs`, and
  the channel-conditioned wrapper preserves `cfg.prompt_override` / `cfg.staged_prompts`.

## Phase 5 config surface

`configs/channel/{rayleigh,fast_fading,packet_drop}.yaml`,
`configs/model/channel_conditioned.yaml`, `configs/acceleration/default.yaml`,
`configs/eval/phase5.yaml`, and the composed example `configs/composed_phase5.yaml`.
Keys: `channel`, `csi`, `use_channel_conditioning`, `condition_mode`,
`acceleration.sampler` / `sampler_steps`, early-exit thresholds, dynamic routing,
`use_srs_v2`, `use_regeneration_search`.

## Integration status (post-review)

The Phase 5 features are reachable from the **main** `scripts/evaluate.py` path
(not just standalone modules):

- `use_channel_conditioning` → reconstruction routed through the **one-pass**
  channel-conditioned pipeline (see below).
- `acceleration.sampler` / `sampler_steps` → applied to the per-SNR run config.
- `acceleration.early_exit` (`early_exit_mode: intra_sampler`) → the diffusion loop
  stops mid-run (see below).
- `use_srs_v2` (+ `use_vqa_hallucination` / `vqa_backend`) → full SRS-v2 evaluator
  with a real local VQA backend (see below).
- `use_regeneration_search` → multi-strategy search in the eval loop, best by the
  configured `verify_metric` (`srs` / `srs_v2`).
- `channel_condition` config block → consumed by the encoder/policy (no dead config).

## Resolved limitations (this iteration)

Each item is tagged **implemented / wired / approximated / fallback / not-yet**.

**1. Real local VQA backend** (was: VQA needed an external backend)
- *implemented*: `evaluators/vqa_backend.py` — `build_vqa_backend` adapters for
  `mock` / `blip2` / `llava` / `mplug` with a tiny `answer(image, question)->str`
  contract; all heavy models **lazy-imported**. Default BLIP-2 model is
  `Salesforce/blip2-opt-2.7b-coco` (the known-good checkpoint the caption extractor
  already loads); a failed backend load is cached (no repeated reloads).
- *wired*: `vqa_backend` config → `EvalContext._get_srs_v2` →
  `VQAHallucinationEvaluator.from_config` → SRS-v2's hallucination layer.
- *fallback*: missing `transformers`/weights, or a runtime backend error, logs a
  warning **once**, disables VQA for the rest of the run, and degrades to the CLIP
  hallucination heuristic (method `clip_fallback` / `vqa_error_fallback`).
- *not-yet*: BLIP-2/LLaVA/mPLUG weights are not bundled; default `type: none`
  (CLIP fallback). LLaVA/mPLUG adapters are best-effort.

**2. One-pass channel conditioning** (was: an extra measurement forward per image)
- *implemented*: `infer_pipeline` is split into `_encode_and_transmit` (encode →
  channel → mask/power → step-match, building per-patch `ForwardArtifacts` +
  receiver evidence) and `_decode_diffusion` (canny → ControlNet latent →
  diffusion → decode). The standard single-call path is numerically unchanged.
- *wired*: `run_image_channel_conditioned` runs encode+transmit **once per patch**,
  aggregates the bundles to an **image-level** measurement, decides the conditioned
  cfg, then decodes **reusing the same received latent** — so the throwaway
  measurement forward (and the record/replay tape) are gone. The conditioned cfg
  actually applied is returned in `info["resolved_cfg"]` / the run payload's
  `"cfg"` (no longer the base cfg) for honest logging.
- *implemented (condition source == decoder init)*: the bundle's
  `best_estimate` (what `ChannelConditionEncoder` reads) is set to the decoder's
  diffusion-init latent `encode_features_hat / power_scalar` — the **same** tensor
  `_run_diffusion` uses as `latent_init` when `use_jscc_feature` is on (which the
  conditioned path forces). Channel descriptors (gain / noise / mask / reliability
  / SNR) are copied from the realisation without mutating it.
- *approximated*: `csi`/blind affects guidance & the transmitted latent; the
  phase-1 step-match still uses the base cfg's `use_gt_csi` (conditioning changes
  guidance/steps, not the already-computed step). `ChannelTape` remains as an
  optional legacy capability (no longer used by this path).

**3. Intra-sampler interrupt early-exit** (was: checkpoint re-render only)
- *implemented*: `acceleration/early_exit.run_interruptible_sampling` — a single
  loop that scores every `check_interval` steps and **terminates** the moment the
  `EarlyExitController` is satisfied (unused steps never run).
- *implemented*: `models/diffusion_wrapper.generate_interruptible` — re-drives the
  SGD-JSCC **continuous DPM-Solver++(2M)** update from the generator's public
  helpers (the original `generate` has no callback hook and `SGDJSCC/` is
  read-only — see `_INTERRUPTIBLE_REQUIRES` and the loop citation), enabling true
  mid-loop exit.
- *wired (all three metrics)*: `_run_diffusion` uses it when
  `acceleration.early_exit` + `early_exit_mode: intra_sampler`.
  `early_exit_metric` is honoured: `heuristic` (cheap latent-convergence, default)
  **or** `srs` / `srs_v2` — the verified metrics decode the loop's current clean
  latent (`state["x0"]`) and score it against the **original patch** with a
  cached `SemanticReliability(/V2)Evaluator` (`_resolve_early_exit_score_fn` →
  `_build_early_exit_score_fn`). The original patch is threaded into
  `_decode_diffusion`, so verified scoring is real, not API-only.
- *fallback*: `step_style != continuous` or a pipe lacking the helpers falls back
  to `pipe.generate` (no interrupt, logged); a missing `srs`/`srs_v2` verifier
  (e.g. CLIP absent) logs a warning and degrades to the heuristic;
  `checkpoint_legacy` mode (`evaluate_checkpoints`) is retained.
- *not-yet*: GPU numeric parity of the re-driven loop vs the original `generate`
  is validated offline with a mock pipe (loop structure + early exit + x0
  exposure) but **not** yet on real checkpoints; verified metrics decode a preview
  per check (extra cost, opt-in).

## Remaining limitations (Phase 5 → future)

- Channel condition **tokens** are attached to the cfg but still not consumed by
  the frozen SGD-JSCC denoiser (FiLM / cross-attention / posterior guidance need a
  retrained, condition-aware denoiser). Conditioning acts via the received-latent
  init + reliability-scaled guidance/steps.
- No trained consistency/distilled student; few-step is a deterministic approximation.
- Intra-sampler interrupt covers the **continuous** sampler only; discrete falls back.
- Fading `signal_scale`/step-matching reuse the AWGN formula (approximate under fading).

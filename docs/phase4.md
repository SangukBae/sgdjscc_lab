> [← docs index](./README.md)

# Phase 4 — Plan & Implementation Status

- [Phase 4 Plan](#phase-4-plan)
- [Phase 4 Implementation Status](#phase-4-implementation-status)

---

# Phase 4 Plan

Phase 4 extends `sgdjscc_lab` from single-image evaluation into a
`keyframe-oriented semantic transmission framework` while keeping the current
SGD-JSCC forward path intact for actual per-frame reconstruction.

The guiding principle of Phase 4 is:

`do not retrain the SGD-JSCC core first; build the semantic packet, temporal
evaluation, adaptive control, and video pipeline around the existing image
path.`

## Phase 4 objective

Transform the current image-only prototype into `RA-SGDJSCC-lite`:

1. semantic packet extraction and caching for each frame
2. keyframe / inter-frame split
3. temporal semantic reuse and delta transmission simulation
4. SNR-aware adaptive guidance control
5. stronger verifier and regeneration logic
6. temporal metrics and keyframe-level reporting

## Phase 4 implementation order

### Phase 4-A: reliability-first image extension

This is the first implementation milestone and should be completed before any
video-specific code.

Scope:

1. add an adaptive guidance controller on top of the current image pipeline
2. upgrade SRS from a pure CLIP/object composite into a packet-aware verifier
3. store semantic packets as JSON metadata even before real packet coding
4. add structured regeneration policies keyed by detected failure mode

Primary references:

- `paper/FAST-GSC: Fast and Adaptive Semantic Transmission for Generative Semantic Communication/FAST_GSC.tex`
- current `sgdjscc_lab` modules:
  - `src/sgdjscc_lab/evaluators/semantic_reliability.py`
  - `src/sgdjscc_lab/pipelines/regeneration_loop.py`
  - `src/sgdjscc_lab/pipelines/eval_pipeline.py`

Planned files:

```text
src/sgdjscc_lab/
├── controllers/
│   ├── adaptive_guidance_controller.py
│   ├── snr_guidance_policy.py
│   └── regeneration_policy.py
├── guidance/
│   ├── semantic_packet_extractor.py
│   ├── object_extractor.py
│   ├── relation_extractor.py
│   └── importance_estimator.py
├── evaluators/
│   ├── semantic_packet_matcher.py
│   ├── relation_consistency.py
│   └── attribute_consistency.py
└── utils/
    └── packet_io.py
```

Implementation steps:

1. `semantic_packet_extractor.py`
   - build a unified semantic packet from:
     - caption
     - object list
     - scene label
     - relation triplets
     - attributes
     - edge summary
     - segmentation summary
     - depth summary
   - Phase 4-A does not transmit this packet over the channel yet
   - it serializes `packet.json` beside each reconstructed image for analysis
2. `adaptive_guidance_controller.py`
   - read estimated SNR from the current pipeline config/runtime state
   - output:
     - `guidance_scale`
     - `controlnet_scale`
     - `diffusion_step`
     - `use_text`
     - optional `skip_diffusion`
   - initial policy:
     - `SNR <= 0 dB`: strong text + edge guidance, max diffusion steps
     - `0 < SNR < 8 dB`: moderate guidance, edge-priority policy
     - `SNR >= 8 dB`: weak guidance, optional unconditional or skip path
3. `semantic_packet_matcher.py`
   - compare original packet vs reconstructed packet
   - explicitly count:
     - missing objects
     - additional objects
     - relation errors
     - attribute errors
     - scene mismatch
4. `regeneration_policy.py`
   - replace the current scalar retry policy with error-type-aware retries
   - example policies:
     - missing object: strengthen text guidance and object-priority guidance
     - hallucination: reduce text CFG and keep edge guidance stronger
     - structural distortion: increase control signal and diffusion steps
5. extend `semantic_reliability.py`
   - keep the current SRS as the baseline score
   - add optional packet-aware terms:
     - relation consistency
     - attribute consistency
     - segmentation consistency
   - report both:
     - `srs_base`
     - `srs_packet`

Expected outputs:

- per-image `packet.json`
- per-image `error_report.json`
- SNR sweep CSV with `srs_base`, `srs_packet`, and error counts
- ablation results for adaptive guidance on high-SNR degradation

### Phase 4-B: keyframe and temporal extension

This is the actual video/keyframe milestone and builds on Phase 4-A packet and
verifier infrastructure.

Primary reference:

- `paper/FAST-GSC: Fast and Adaptive Semantic Transmission for Generative Semantic Communication/FAST_GSC.tex`
  - semantic units
  - transmission order
  - semantic difference calculation
  - sequential conditional denoising

Planned files:

```text
src/sgdjscc_lab/
├── video/
│   ├── keyframe_extractor.py
│   ├── scene_change_detector.py
│   ├── semantic_delta.py
│   ├── temporal_pipeline.py
│   └── motion_residual.py
└── evaluators/
    └── temporal_consistency.py
```

Implementation steps:

1. `scene_change_detector.py`
   - start with a practical heuristic detector:
     - CLIP image-image distance
     - LPIPS between consecutive frames
     - optional color histogram delta
   - mark scene boundaries for new keyframes
2. `keyframe_extractor.py`
   - create GOP-like groups
   - output:
     - keyframe indices
     - inter-frame ranges
3. `semantic_delta.py`
   - compare packet at frame `t` with packet at the previous keyframe or
     previous transmitted frame
   - produce delta units:
     - new object
     - removed object
     - changed relation
     - changed attribute
     - changed scene
4. `temporal_pipeline.py`
   - keyframe:
     - run full image pipeline
     - save full semantic packet
   - inter-frame:
     - reuse latest keyframe packet
     - apply only semantic delta
     - reuse or attenuate guidance depending on change magnitude
5. FAST-GSC-inspired sequential denoising schedule
   - Phase 4-B will not reimplement FAST-GSC training directly
   - instead, it will emulate `semantic unit arrival over time` by injecting
     semantic groups in stages during the denoising schedule
   - practical first split:
     - early denoising: scene + major objects
     - middle denoising: relations + structure
     - late denoising: attributes + fine corrections
6. `temporal_consistency.py`
   - report:
     - temporal SRS
     - object identity consistency
     - temporal segmentation IoU
     - temporal hallucination rate

Expected outputs:

- keyframe list JSON
- per-sequence temporal metrics CSV
- side-by-side reports:
  - full per-frame semantics
  - keyframe reuse
  - keyframe + semantic delta transmission

## Phase 4 completion criteria

Phase 4 will be considered complete when all of the following are true:

1. `scripts/evaluate.py` can run packet-aware SRS evaluation in image mode
2. a new video evaluation entry point can process a folder of ordered frames
3. keyframe reuse and semantic delta logic produce structured logs
4. temporal SRS and temporal hallucination metrics are exported
5. the report shows a concrete reduction in semantic transmission overhead
   relative to naive per-frame full guidance

## Phase 4 experimental design

Datasets:

- image:
  - Kodak
  - COCO val2017
  - ADE20K validation
- video:
  - ETRI internal keyframe/scene-change data if available
  - otherwise public video frames extracted into ordered image folders

Channel settings:

- AWGN:
  - `-15, -10, -5, 0, 5, 10, 15 dB`
- optional packet-drop simulation for semantic delta experiments

Main ablations:

1. baseline SGD-JSCC
2. SGD-JSCC + adaptive guidance
3. SGD-JSCC + adaptive guidance + packet-aware verifier
4. keyframe-only full packet
5. keyframe + semantic delta reuse

---

# Phase 4 Implementation Status

Phase 4 is implemented as a set of **config-driven, opt-in** extensions layered on
top of the unchanged SGD-JSCC image forward pass.  Every new feature defaults to
*off*, so the legacy `infer_images.py` / `evaluate.py` paths behave exactly as in
Phase 3 unless explicitly enabled.

## Phase 4-A (reliability-first image extension) — delivered

| Area | Module(s) |
|---|---|
| Semantic packet | `guidance/semantic_packet_extractor.py`, `guidance/object_extractor.py`, `guidance/relation_extractor.py`, `guidance/importance_estimator.py`, `utils/packet_io.py` |
| Adaptive guidance | `controllers/snr_guidance_policy.py`, `controllers/adaptive_guidance_controller.py` |
| Packet-aware verifier | `evaluators/semantic_packet_matcher.py`, `evaluators/relation_consistency.py`, `evaluators/attribute_consistency.py` |
| SRS extension | `evaluators/semantic_reliability.py` (`srs_base`, `srs_packet`, `score_packet`) |
| Structured regeneration | `controllers/regeneration_policy.py` (failure-mode-keyed retries) |
| Eval integration | `pipelines/eval_pipeline.py` (packet build/save, packet metrics, CSV columns) |

Enable from config (see `configs/eval/default.yaml`):

```yaml
use_packet_eval: true          # build packets, emit srs_base / srs_packet + error counts
use_adaptive_guidance: true    # SNR-regime guidance scaling (strong/moderate/weak)
use_packet_regeneration: true  # error-type-aware retries (needs use_packet_eval)
```

Run packet-aware image evaluation:

```bash
python scripts/evaluate.py --config configs/composed.yaml --snr 0 \
    -i ../inputs/   # set use_packet_eval: true in eval/default.yaml first
```

Outputs per image (under `packet_dir`): `<stem>.orig_packet.json`,
`<stem>.packet.json`, `<stem>.error_report.json`.  The results CSV gains
`srs_base, srs_packet, object_match_rate, relation_consistency,
attribute_consistency, segmentation_consistency, scene_match,
missing_object_count, additional_object_count, relation_error_count,
attribute_error_count, guidance_regime`.

## Phase 4-B (keyframe / temporal extension) — delivered

| Area | Module(s) |
|---|---|
| Scene change | `video/scene_change_detector.py` (histogram + optional CLIP/LPIPS) |
| Keyframe / GOP | `video/keyframe_extractor.py` |
| Semantic delta | `video/semantic_delta.py` |
| Motion / residual | `video/motion_residual.py` |
| Temporal pipeline | `video/temporal_pipeline.py` (keyframe full / inter-frame reuse + delta; staged prompt from `build_staged_schedule` wired into reconstruction via `cfg.prompt_override`) |
| Temporal metrics | `evaluators/temporal_consistency.py` (temporal SRS, object identity consistency, temporal segmentation IoU, temporal hallucination rate) |
| CLI / config | `scripts/evaluate_video.py`, `configs/video/default.yaml`, `configs/composed_video.yaml` |

Run keyframe/temporal evaluation on an ordered frame folder:

```bash
# Full run (loads SGD-JSCC + CLIP/BLIP2):
python scripts/evaluate_video.py --config configs/composed_video.yaml \
    --input /path/to/ordered_frames/ --snr 5 --device cuda:0

# Dry run of the keyframe/delta/temporal logic (no checkpoints). Packets are
# empty unless captions are supplied, in which case semantic delta/metrics are
# meaningful too:
python scripts/evaluate_video.py --config configs/composed_video.yaml \
    --input /path/to/ordered_frames/ --no-models --captions /path/captions.txt
```

Outputs: `keyframes.json` (GOP structure), `temporal_frames.csv` (per-frame log),
`temporal_metrics.csv` (sequence metrics + `overhead_reduction`, the semantic-unit
saving vs naive per-frame full transmission).  Packet/error JSON from the image
evaluator are namespaced per SNR (`packet_dir/snr_<snr>/…`) so SNR sweeps do not
overwrite one another.

## Reference mapping (what came from where)

- **FAST-GSC** → semantic-unit packet design (`semantic_packet_extractor`),
  importance-based transmission ordering (`importance_estimator`), semantic
  difference calculation (`semantic_delta`), and staged conditional denoising
  approximation (`temporal_pipeline.build_staged_schedule`) — the staged prompt is
  fed to the diffusion text condition (`cfg.prompt_override`), not injected
  per-sampler-step.
- **SGD-JSCC** → the per-frame reconstruction path is reused verbatim; keyframes
  call the existing forward pass, and adaptive guidance only chooses *which*
  config that unchanged path runs with.

## Known limitations / next steps

- Packets are serialised as metadata only; real semantic-packet channel coding /
  drop simulation is deferred (Phase 5).
- Object/scene/relation extraction is CLIP/caption-heuristic (no scene-graph or
  POPE-VQA model yet); relation/attribute parsing is deterministic but shallow.
- Staged denoising is wired at the **prompt** level (`cfg.prompt_override` feeds
  the packet-derived staged prompt into the real reconstruction); true per-step
  prompt switching *inside* the DPM-Solver loop is **not** implemented because it
  would require modifying the SGD-JSCC sampler (algorithm-preservation invariant).
- Inter-frame reuse copies the keyframe reconstruction (the GOP keyframe is the
  single consistent packet+pixel reference); true delta-warp / motion-compensated
  synthesis is future work.

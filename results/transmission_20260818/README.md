# transmission_reduction run — transmission_reduction_full_20260818_043425

> **Annotated tracked copy.** This file is a copy of
> `outputs/transmission_reduction_full_20260818_043425/README.md` with the
> baseline/provenance wording below corrected for accuracy; no numeric result
> was changed (see `manifest.json`'s `artifacts."README.md".matches: false`
> and `extra.correction_note` for exactly what changed and why). The original,
> untouched file remains at the `outputs/` path above.

Real packet-bundle (all visual patches + per-patch captions + edge/uncertainty + manifest) transmission-size
accounting, full-video quality via the real `TemporalPipeline` path, and
SKEM/PSSS keyframe selection (optionally scene-change-combined). See the
module docstring of `scripts/run_transmission_reduction_eval.py` for the
exact-vs-estimate accounting boundaries and the digital bundle-only receiver boundary.

Pareto baseline: `fixed_awgn`, used only as a **provisional analog quality
baseline** — not a "no digital baseline was run" situation:
- `fixed_int16`/`skem_int16` **did run** (see `configs_run` in `summary.json`)
  but are excluded from baseline consideration because 52/1000 frames came
  back NaN/Inf (`total_nan_or_inf_frames` in `aggregate.csv`; see
  `manifest.json`'s `nan_or_failure_counts`).
- `float32` digital was **not run at all** this sweep (not in `configs_run`).
- NaN-clean digital candidates did exist (`fixed_int4`/`skem_int4`), but no
  pre-designated reliable-digital baseline was valid: float32 was not run and
  int16 was incomplete. `fixed_awgn` (analog) was therefore used as a
  temporary stand-in. Quality-gate comparisons against it mix quantization
  loss with AWGN noise and should be treated cautiously.
- The `pareto_frontier.csv`-selected `skem_int4` is therefore a **provisional
  candidate**, not a finalized operating point — re-judge once a reliable
  NaN-clean digital baseline (or float32) is run.

- `per_video_metrics.csv` / `aggregate.csv` — full-video quality (PSNR/SSIM/
  LPIPS over every reconstructed frame, not just keyframes) + exact
  transmission-bundle bytes per (video, config). `aggregate.csv`'s
  `mean_total_bundle_bytes` is in **bytes/video** — the mean, across the 10
  videos, of each video's total transmission-bundle bytes (100 frames/video).
  The "bytes/frame" figures in
  `docs/experiments/2026-08-18_transmission_reduction.md`'s result table are
  this value divided by 100; no separate bytes/frame column exists in this CSV.
- `keyframe_selection.csv` — per transmitting frame: decision, structured
  `force_reason` (`first_frame`|`scene_change`|`max_segment_length`|`psss`|
  `selected` — never inferred from prose), the 5-field measurement schema
  (`latent_elements`/`analog_channel_symbols`/`source_packet_bits`/
  `estimated_digital_channel_symbols`/`estimated_wire_bytes`), and
  `psss_backend_kind` (`mock`|`proxy`|`real` — never conflated).
- `packet_components.csv` — exact per-frame bundle byte breakdown (caption/
  edge/edge-uncertainty/visual/manifest payloads + container overhead), with
  `total_bundle_bytes` equal to the actual `.sgbundle` file size.
- `packets/<video>/<config>/frame_NNNNN.sgbundle` — the actual serialized
  transmission bundles (visual+caption+edge+manifest) a receiver would parse.
- `recon_videos/<video>/<config>/recon.mp4` + `frame_*.png` — the FULL
  reconstructed video (every frame, not just keyframes).
- `keyframe_sweep.csv` — PSSS threshold x max_segment_length grid (selection
  only, no reconstruction); reports `psss_backend_kind` per row.
- `pareto_frontier.csv` — smallest-bytes config meeting the quality gate
  (PSNR drop <= 0.5 dB, SSIM drop <= 0.01,
  LPIPS rise <= 0.02) against the provisional analog (`fixed_awgn`)
  baseline above — **not** a reliable digital baseline (see above); if none
  qualify, the nearest candidates are still listed.
- `source_size_report.csv` — exact source MP4 sizes only (see note above).
- `summary.json` — run configuration + selected config + baseline used.

Known limitations:
- `--psss-backend proxy` was used for keyframe selection this
  run — only `real` (with `--psss-model-id`) is genuine PSSS (an actual
  causal-LM/VLM's yes/no token probability); `mock`/`proxy` are explicitly
  NOT real PSSS (see `video/psss.py`'s module docstring) and every CSV/JSON
  in this run tags rows with `psss_backend_kind` so this is never conflated.
- `estimated_digital_channel_symbols`/`estimated_wire_bytes` are labeled
  proxy estimates (`unavailable — no --bits-per-symbol given`)
  — no real modulator/FEC coder exists in this codebase.
- Digital configs reconstruct from the exact `.sgbundle` bytes saved in this
  run. AWGN cannot be reconstructed from a byte bundle because its visual
  waveform is analog; for AWGN, `analog_channel_symbols_total` and the exact
  digital caption/edge/manifest bytes are reported as separate domains.
- **Known numerical fragility at coarse digital quantization** (found via GPU
  verification, not this feature's own bug): `jscc.snr_prediction_net` (the
  blind SNR predictor `_compute_step()` uses, `pipelines/infer_pipeline.py`,
  untouched by this feature) was only ever trained on AWGN-shaped
  degradation. On some real frames, bit_depth=8 quantization pushes its
  predicted signal scale to >= 1, making `10*log10(1/cur_step - 1)` evaluate
  `log10` of a non-positive number -> NaN, which then propagates through the
  (otherwise correct) diffusion decode. This driver detects it
  (`n_nan_or_inf_frames` in per_video_metrics.csv/aggregate.csv). Any such
  config unconditionally fails the quality gate, cannot be Pareto-selected,
  and does not get a misleading `recon.mp4`; finite-frame diagnostic means
  remain available together with `valid_frame_ratio`.

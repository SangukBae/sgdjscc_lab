---
status: frozen
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes:
---

# ETRI Research Progress Update — Detailed Slide Notes

> Frozen report snapshot. Legacy document paths in the body refer to the layout at
> presentation time; use [the current documentation index](../../README.md) now.

- Source deck: `ETRI_연구진행상황공유_20260816_ETRI용_업데이트.pptx`
- Reporting date: August 2026
- Number of slides: 9
- Purpose: Expand the compact slide text into a detailed explanation of the research rationale, data flow, implementation status, validation scope, metric definitions, and presentation caveats.

- In these notes, “implemented” means that the relevant code path and artifacts exist and have been exercised within the stated scope. It does not automatically mean that performance superiority has been demonstrated. Actual bitstream/CBR validation, full-SNR and fading experiments, fair comparisons with other models, full-video superiority of the generation modes, and GT/VLM-based Temporal SRS weight learning are explicitly treated as follow-on work.

## Narrative of the Deck

- The deck follows this argument:

1. Define the goal as preserving transmitted semantic intent, not merely maximizing visual quality.
2. Separate what has been implemented from the level at which it has been validated.
3. Present the baseline SGDJSCC image path and mark three key limitations.
4. Address Limitation 1, temporal/video processing, with a keyframe-based extension.
5. Address Limitation 2, generative hallucination, with packet-based receiver self-verification.
6. Address Limitation 3, evaluation credibility, with held-out evaluation and PTC/SFR/SDI.
7. Summarize completed, PoC, partial, and remaining work.
8. Turn the current results into three follow-on research themes.

## Shared Terminology

- **SGD-JSCC/SGDJSCC**: The baseline system that transmits latent image information and structural guidance through a simulated wireless channel and reconstructs the image with a diffusion model at the receiver.
- **Semantic intent**: The information that should survive transmission, such as objects, relations, attributes, and scene identity.
- **Semantic packet**: A structured representation of objects, relations, attributes, and scene information that can be used for verification at the receiver.
- **SRS**: Semantic Reliability Score, combining image-image/text-image similarity and object preservation, omission, and addition terms rather than relying only on pixel fidelity.
- **PTC**: Packet-Temporal Consistency, the temporal mean of reference-to-reconstruction packet consistency. Higher is better.
- **SFR**: Semantic Flicker Rate, the rate of reconstructed-object births/deaths not explained by real changes in the reference sequence. Lower is better.
- **SDI**: Semantic Drift Index, the least-squares slope of packet drift versus distance from the most recent keyframe. A larger positive value indicates stronger drift.
- **CBR**: Channel Bandwidth Ratio. It is required for a true matched-rate comparison and is not equivalent to the current accounting PoC.
- **Held-out evaluation**: A final evaluation path kept separate from the metrics used to control regeneration or adjust reconstruction.

---

## Slide 1. Generative AI-Based Semantic Media Transmission Reliability Enhancement Technology Development

### Main Point

- The title establishes that the project is not simply about using generative AI to reconstruct media. It is about making semantic media transmission that uses generative reconstruction more reliable and measurable. The deck is an August 2026 progress update, so it reports both completed implementation and work that still requires performance validation.

### Visual and Conceptual Structure

- The simple blue bars and white background frame the deck as a technical progress report.
- The title contains three important ideas:
  - **Generative AI-based**: the receiver performs diffusion-based generative restoration rather than only deterministic decoding.
  - **Semantic media transmission**: preserving objects, relations, and scene meaning is more important than reproducing every source pixel.
  - **Reliability enhancement**: the system must detect omissions, additions, distortions, temporal drift, and flicker instead of judging only visual plausibility.
- “Research Progress Update” signals that the slides distinguish working infrastructure from final comparative claims.

### Suggested Presentation Emphasis

- Open with the statement that success is not defined by the highest PSNR. The real question is whether the intended meaning remains trustworthy after transmission through a noisy channel. A generative model can improve reconstruction, but it can also invent content; therefore generation and verification must be developed together.

---

## Slide 2. Project Objective and Problem Definition

### Main Point

- The objective is to build an end-to-end simulation framework that quantifies how accurately the semantic intent of the source image—its objects, relations, and scene—survives noise, fading, and packet loss in a wireless channel. The table maps three ETRI concerns to the corresponding research response.

### Objective Box

- The blue box defines the full research scope:

- **Input**: a source image, later extended to video keyframes and non-keyframes.
- **Channel**: AWGN is the currently established baseline; Rayleigh fading, fast fading, and packet loss are managed as extensions.
- **Receiver output**: an image or video reconstructed with a diffusion model.
- **Evaluation target**: agreement between transmitted intent and reconstructed meaning, not only visual sharpness.
- **Deliverable form**: a modular Python end-to-end framework with reproducible CSV, JSON, and Markdown artifacts.

### Concern 1 — Hallucination in Generative Restoration

- Diffusion restoration can produce a plausible image at low SNR, but plausibility is not the same as faithfulness. It may add an object absent from the source, omit a required object, or change relations and attributes. The response has two stages:

1. Detect added, missing, and distorted semantic content per image or frame.
2. When reliability is below a threshold, adjust the reconstruction condition, retry, or select a safer fallback.

- Slide 6 turns this concept into the Packet Verifier, error-specific Controller, and OWLv2/VQA presence enhancement.

### Concern 2 — Limits of PSNR/SSIM-Centric Evaluation

- PSNR and SSIM measure pixel or structural fidelity. A generated result can preserve the same scene semantics with different pixels, or appear visually strong while losing a crucial object. The project therefore retains PSNR/SSIM/LPIPS but adds CLIP similarity, object preservation/missing/additional terms, hallucination analysis, SRS, and packet consistency to measure “Tx intent ↔ Rx reconstruction” directly.

- The proposal is not to discard image-quality metrics. It is to introduce a separate semantic-reliability axis.

### Concern 3 — Limits of Still-Image Evaluation

- Still-image metrics cannot show whether an object flickers, whether meaning drifts over time, or whether a scene transition is handled correctly. The response is a keyframe-based video pipeline, segment-level decisions, reconstructed MP4 output, and temporal semantic metrics. Slides 5 and 7 explain these elements.

### Repository Connections

- Project objective and base SRS: `docs/etri_overview.md`
- Three limitations and the implementation strategy: `docs/etri_strategy.md`
- Video extension design and scope caveats: `docs/video_extension_lgvsc.md`

### Interpretation Caveat

- Noise, fading, and packet loss all appear in the objective, but they are not validated to the same depth. The AWGN baseline path has been implemented and exercised; the final fair comparison matrix across all fading and packet-loss conditions remains future work. Present the target scope separately from the current validation scope.

---

## Slide 3. Implementation Status and Current Validation

### Main Point

- This slide separates “what has been built” from “how strongly it has been validated.” The upper table lists implementation status by requirement. The lower table separates structural, functional, and performance validation. The central conclusion is that structural and functional validation are complete, while comprehensive performance validation is only partial.

### Upper Table — Implementation by Requirement

#### 1) End-to-End Pipeline

- Image inference, video processing, evaluation, and stage-aware training have been integrated modularly while preserving the original SGD-JSCC inference path. Extension features are off by default. If Phase 4/5, video, or generation options are not enabled, the baseline path should remain unchanged.

#### 2) Hallucination Mitigation and Detection

- The implementation includes a Packet Verifier comparing transmitted and reconstructed semantic packets, a Controller producing actions for different error types, and OWLv2/VQA paths that strengthen object-presence decisions. Remeasurement has been completed, but the final closed-loop injection of Controller decisions into the diffusion sampler remains follow-on work.

#### 3) Quantitative Tx/Rx Intent Agreement

- PTC, SFR, and SDI extend image SRS into the temporal domain. Real SGD-JSCC reconstructions from 10 videos were evaluated in five presence modes, producing 50 jobs in total. Each job contains 100 items, and the aggregate results are stored in `outputs/remote_hq_4090_20260816/remeasure/summary_metrics.csv` and `.md`.

- The five modes are OWLv2-only, VQA-only, unfiltered ensemble, GT-object-only ensemble, and open-world ensemble. GT-object-only and open-world results support different claims. The unfiltered ensemble is a diagnostic baseline because caption-noun contamination can affect it.

#### 4) Keyframe-Based Video Extension

- MP4 input/output, keyframe and segment structures, a semantic/motion dual gate, and reuse/recompute/generate decisions have been implemented. Keyframes reuse the existing SGD-JSCC path; non-keyframes vary transmission and reconstruction behavior according to change.

#### 5) Reproducible Python Prototype

- The repository includes isolated per-video output, batch execution, CSV/JSON/Markdown summaries, status tracking, and report automation. The slide cites 1,133 passing automated tests as evidence of structural stability and regression coverage. This does not by itself prove algorithmic superiority; it shows that the expected code paths and invariants operate correctly.

### Lower Table — Levels of Validation

- **Structural validation — complete**: the software runs, the default settings preserve the original path, and automated tests pass.
- **Functional validation — complete**: real-model reconstructions and OWLv2/VQA remeasurement produce the required metrics and artifacts.
- **Performance validation — partial**: video baselines and calibrated evaluation are available, but a full video SNR sweep, fading tests, matched comparisons with WITT/DiffJSCC/other models, and actual-CBR/bitstream comparisons remain.

### Suggested Presentation Emphasis

- Use this slide as the boundary against overclaiming. “The code runs,” “the intended quantity is measured,” and “the method outperforms alternatives” are different claims. The first two have been established within the stated scope; only part of the third has been established.

---

## Slide 4. SGDJSCC Block Diagram

### Main Point

- This slide presents the baseline system data flow. The transmitter separates the source into latent, edge, and caption paths. These signals cross or condition the wireless link. The receiver combines text and structural conditions with MDTv2 diffusion restoration. Image and video outputs are then evaluated for semantic reliability.

### Colors and Line Styles

- **Blue**: the image-latent path, carrying the principal continuous visual payload.
- **Green**: the edge/structure path used as a ControlNet condition.
- **Orange**: the caption/text path used as a semantic text condition.
- **Gray dashed lines**: source-to-output evaluation or an error-free side-information assumption.
- **Red markers 1–3**: the three limitations developed in the next slides—video/time, generative hallucination, and evaluation reliability.

### Input and Patch Tiling

- The source image is split into 128×128 patches. The frame dimensions are prepared for this tiling convention, each `[3,128,128]` RGB patch is processed through the existing path, and the reconstructed patches are merged at the receiver.

### Three Transmitter Paths

#### 1) Image Latent

- The VAE encoder maps a patch into latent space and applies L2 normalization. Under the fixed architecture, one patch has a `16×16×16 = 4,096`-element latent. The slide’s “symbols/pixel ≈ 1/12” gives an intuitive latent-to-RGB-element ratio. The repository preserves the baseline VAE scaling factor of 15.45, AWGN computation, blind-SNR process, and diffusion-step matching.

#### 2) Edge/Structure

- MuGE extracts the edge representation, and a separate Edge JSCC path transmits structural guidance. At the receiver the edge is restored and encoded into a ControlNet condition. The slide’s CR=0.2 label communicates low-rate side information. The actual accounting code derives its default proxy from the Canny channel encoder constant `320/4096`; therefore the conceptual CR label on the slide should not be treated as identical to every accounting configuration.

#### 3) Caption/Text

- BLIP-2 extracts the caption, and the CLIP text encoder converts it into a diffusion condition. The dashed line represents an error-free side-information assumption. BLIP-2 is used for inference and evaluation captions, while Qwen2.5-VL-3B is reserved for offline caption generation.

### Wireless Channel

- AWGN is the implemented baseline channel. Rayleigh fading, fast fading, and packet loss are extension paths and scaffolds; their presence in the diagram does not mean the final all-condition comparison is complete. The Edge JSCC path simulates its own channel behavior within that network path.

### Receiver Reconstruction

1. Estimate the received latent’s SNR blindly.
2. Match the diffusion step to the estimated SNR.
3. Convert the caption to a CLIP text condition.
4. Convert the restored edge to a ControlNet condition.
5. Restore the latent with the fixed MDTv2 diffusion backbone.
6. Decode RGB patches with the VAE and merge them into the output image.

- The receiver is doing more than decoding a corrupted latent: it uses semantic text and structural guidance to generatively fill in missing information. That capability also creates hallucination risk, marked by red limitation 2.

### Evaluation Outputs

- Image evaluation records PSNR, SSIM, LPIPS, CLIP, object preservation/missing/additional, hallucination, and SRS in `results.csv`.
- Video evaluation records PTC, SFR, SDI, temporal SRS, and related fields in `temporal_metrics.csv`.
- The dashed source-to-output comparison emphasizes semantic agreement with the source, not only perceived quality.

### Interpretation Caveat

- Do not imply that caption and edge information experience exactly the same corruption as the principal latent. The slide explicitly distinguishes the error-free caption-side-information assumption and the separate Edge JSCC link. Also, video keyframe/segment/generate features are implemented but off by default, keeping the baseline image path distinct from extensions.

---

## Slide 5. Temporal and Video Implementation Status

### Main Point

- If video reuse is based only on small semantic change, the system can miss camera motion, zoom, or scene transitions. The current implementation therefore uses both semantic delta and motion residual to select reuse, recompute, or generate.

### Legend and Status Encoding

- White/gray blocks reuse the existing SGD-JSCC path.
- Yellow blocks are extensions around the existing path.
- Red dashed blocks are new generation paths.
- Numbered markers highlight the decision gate, segment concatenation, and three-way output branch.

### Input and Keyframe Extraction

- The MP4 I/O path extracts frames and writes reconstructed frames back to MP4. Keyframe selection can respond to scene transitions and optionally use PSSS. `keyframes.json` and `segments.json` preserve the selections and segment decisions for reproducibility and post-hoc analysis.

### Keyframe Path

- Keyframes use the original SGD-JSCC transmitter to carry latent, edge, and caption information. The baseline diffusion path reconstructs the keyframe. This reconstructed keyframe then becomes the reference or generation condition for subsequent non-keyframe segments. The video extension is therefore layered on top of the baseline image restorer rather than replacing it.

### Non-Keyframes and the Dual Gate

- For a non-keyframe, the pipeline computes semantic delta and motion residual relative to the keyframe.

- A large semantic delta indicates an object or scene change and increases the need for reconstruction.
- Even if semantic delta is small, a large motion residual indicates camera or spatial motion that should prevent naive reuse.
- Reuse is safe only when both forms of change are sufficiently small.

- The motion gate is off by default to preserve baseline behavior. When enabled, the threshold and decision reason are recorded. This prevents a semantically similar but spatially moving segment from being reused blindly.

### Three-Way Decision

- **Reuse**: reuse a reconstructed keyframe or prior output in a low-change segment. It offers the largest transmission and compute reduction, but overly long reuse can cause staleness or drift.
- **Recompute**: retransmit/reconstruct a frame through SGD-JSCC when semantic or motion change is large. This is more expensive but safer.
- **Generate**: synthesize a middle segment from a start keyframe or start+end keyframes, caption, and side information. Actual SVD and Wan GPU paths have been exercised. A full-video quality, drift, and flicker advantage has not yet been demonstrated and requires separate comparison.

### Output and Temporal Evaluation

- The segment outputs are concatenated in time into a reconstructed MP4. PTC, SFR, SDI, and temporal SRS are computed at the same time. Ten real-model videos were remeasured under five OWLv2/VQA modes. The set covers different semantic and motion conditions, including walking people, passing/exiting cars, a dog, camera pan, a handheld sign, a scene cut, and a busy sidewalk.

### Transmission-Rate Perspective

- A bit/channel-symbol accounting PoC exists for reuse, recompute, and generate. It combines directly countable artifacts such as semantic-packet JSON byte lengths with explicit proxies for latent, edge, and motion costs. Because it does not yet include a real entropy-coded bitstream, modulation, or channel coding, it must not be presented as proof of superiority at matched CBR.

### Relevant Artifacts

- Video extension scope: `docs/video_extension_lgvsc.md`
- Ten-video report: `outputs/etri_video_eval/final_report.md`
- Remote high-quality run status: `outputs/remote_hq_4090_20260816/hq_validation_status.json`
- Video/generation quality summaries: `outputs/remote_hq_4090_20260816/quality/`

### Transition to Slide 6

- The generate branch synthesizes intermediate frames that were not directly transmitted. This increases the risk of hallucination, drift, and flicker, motivating receiver-side verification in the next slide.

---

## Slide 6. Hallucination Verification Status

### Main Point

- Generative restoration can add, omit, or distort content, and video generation adds drift and flicker. The current receiver checks the output against the transmitted semantic packet, strengthens object-presence judgments with OWLv2/VQA, and produces error-specific control decisions. The trainable Adapter and Critic are explicitly follow-on modules.

### Meaning of Rx-Legal Self-Verification

- The Packet Verifier does not assume that the receiver can inspect the original source image. The transmitted semantic packet is information the receiver can legally possess, so comparing it with a packet re-extracted from the output provides “Rx-legal self-verification.” The packet includes objects, relations, attributes, and scene information.

### Upper Data Flow

1. The transmitted semantic packet reaches the receiver.
2. The current path performs fixed-MDTv2 diffusion restoration using text/edge and packet conditions.
3. A semantic packet is re-extracted from the reconstructed image or frame.
4. The transmitted and reconstructed packets are compared.

- The red-dashed **Semantic Packet Fidelity Adapter** is a future trainable component intended to inject the packet condition directly into the generator. Its presence in the diagram does not indicate completion.

### Packet Verifier

- The verifier reports object omissions, additional objects, relation mismatch, attribute drift, and scene mismatch separately, then folds them into a 0–1 severity value for the Controller. Default weights are missing 0.30, additional 0.25, relation 0.20, attribute 0.15, and scene 0.10. Zero means a perfect match; larger values indicate more severe semantic damage.

- Because CLIP/caption-derived packet object lists can contain false positives and negatives, the reconstructed frame can be rechecked with an OWLv2 grounded detector and BLIP-2 VQA. Raw and calibrated results are stored separately, making the source of any decision change traceable.

### Error-Specific Controller

- The verifier output becomes a candidate action rather than a single unconditional retry command.

- Additional object: strengthen negative prompting or suppression.
- Missing object: emphasize the object in the prompt or packet condition.
- Relation/structural distortion: strengthen edge, motion, or structural guidance.
- Repeated failure or high severity: choose recompute or keyframe fallback.

- Condition-adjustment decisions and logs are implemented. Injecting those actions into actual sampler prompt/guidance/step settings and measuring closed-loop improvement remains future work.

### Hallucination Critic and Safeguards

- The red-dashed **Hallucination Critic** is a future trainable module that would decide whether a reconstructed object is permissible from the transmitted packet’s perspective. It is intended to complement difficult cases rather than simply replace the rule-based verifier. Recompute/keyframe fallback prevents indefinite acceptance of uncertain generated output.

### Separating In-Loop and Held-Out Roles

- If packet matching or VQA is used to change reconstruction and then the same metric is reported as the final result, the evaluation becomes circular and vulnerable to metric gaming. Reports therefore distinguish `loop_internal` and `held_out` roles, and final evaluation is rerun through a separate path. Slide 7 explains this principle.

### Reading the Actual Remeasurement

- All 50 remote jobs completed successfully, but the evaluation modes support different interpretations.

- `ensemble_gt_filter`: a closed-world, GT-object-only mode for object-preservation claims.
- `ensemble_openworld_filter`: retains non-GT candidates for additional-object and hallucination analysis.
- `ensemble_nofilter`: diagnostic only because caption-noun contamination may be present.
- The observed severity decrease and PTC increase after OWLv2/VQA calibration reflect a change in presence judgment. They should not be described as an improvement to the underlying reconstruction model itself.

### Repository Connections

- Verifier implementation: `src/sgdjscc_lab/evaluators/packet_verifier.py`
- Presence calibration and interpretation: `docs/etri_owlv2_vqa_readiness.md`
- Fifty-job summary: `outputs/remote_hq_4090_20260816/remeasure/summary_metrics.md`

---

## Slide 7. Temporal Semantic Evaluation Status

### Main Point

- Credible video-semantic evaluation requires separating frame quality, in-loop control metrics, and final performance metrics. The implementation separates in-loop and held-out roles and uses PTC, SFR, and SDI to measure temporal preservation, flicker, and drift. Fifty remeasurement jobs—10 real-model videos in five modes—have been completed.

### Left Column — Why Existing Evaluation Is Insufficient

#### Limits of PSNR/SSIM

- PSNR and SSIM emphasize per-frame pixel or structural quality. They do not directly show whether an object disappears and returns, or whether meaning deteriorates as a frame moves farther from its keyframe.

#### Circular Evaluation Risk

- Using a metric to decide regeneration and then using the same metric for the final claim lets the system optimize directly for its evaluator. The score can improve without a corresponding independent improvement in semantic reliability.

#### Limits of CLIP/SRS Alone

- Global CLIP similarity and average SRS do not clearly identify the type and timing of object birth/death, long-term semantic drift, or temporal hallucination. Ordered packet/object comparisons are needed.

### Middle Column — Evaluation Principle

- **In-loop metrics**: packet matcher, packet SRS, and VQA support regeneration, stronger guidance, or fallback decisions.
- **Break the loop**: do not optimize and evaluate with the same signal.
- **Held-out final evaluation**: rerun evaluation independently with OWLv2/VQA. Interpret GT-object-only and open-world modes separately.

### Right Column — PTC

- PTC computes reference-to-reconstruction packet consistency for every frame and averages it over time. Per-frame packet consistency uses object 0.5, relation 0.2, attribute 0.2, and scene 0.1 weights. A value near 1 means that the transmitted meaning remains present throughout the sequence.

- PTC should not be read alone. A severe error in a small number of frames can be diluted by an average, so per-frame logs, severity, SFR, and SDI should also be inspected.

### Right Column — SFR

- SFR measures births and deaths of reconstructed objects between consecutive frames, but removes changes that also occur in the reference packets. It is therefore a rate of spurious semantic flicker rather than a raw object-set change rate. A value near zero is desirable.

### Right Column — SDI

- SDI regresses packet drift, `1 - consistency`, against distance from the most recent keyframe. A positive slope indicates that semantic disagreement tends to grow with keyframe distance. A value near zero or below means no distance-dependent deterioration was observed, but short or low-variation sequences may provide limited discrimination.

### Temporal SRS Calibration

- The code provides configuration loading/saving and a least-squares weight-fitting scaffold. It has not yet learned final weights from real GT annotations or an independent VLM judge. The dashed border communicates that the infrastructure exists but real-data calibration is still future work.

### Solid and Dashed Borders

- Solid: implementation and real remeasurement completed.
- Dashed: interface/scaffold exists, but final GT/VLM-based training or validation remains.

### Suggested Presentation Emphasis

- OWLv2/VQA remeasurement demonstrates detector/evaluator calibration, not retraining of the transmission or generative model. Also, the 50 jobs are 10 videos multiplied by five evaluation modes—not 50 distinct videos.

---

## Slide 8. Achievements to Date and Next Steps

### Main Point

- This slide classifies the project into complete, PoC complete, partially complete, and next-step work. It is the clearest single-slide distinction between deliverables already available and work required for a final performance table.

### Complete ① — Video End-to-End Path

- MP4 I/O, segment structures, the semantic/motion gate, and PTC/SFR/SDI are complete. Evidence includes a real-model 10-video baseline, reconstructed MP4s, frame/temporal CSVs, and keyframe/segment JSON files. These artifacts show that the path runs from input video through reconstruction to evaluation.

### Complete ② — Packet Verification and Held-Out Enhancement

- The Packet Verifier, OWLv2/VQA presence enhancement, and in-loop/held-out separation are implemented. The 10 videos × 5 modes = 50 remeasurements have completed. Completion here means that the evaluation infrastructure and real run are available; it does not mean every detector is perfect or that every hallucination has been eliminated.

### PoC Complete — Transmission Accounting

- The system estimates bit, channel-symbol, and semantic-unit costs for reuse/recompute/generate and produces rate–reliability tables and curves. Direct measurements such as semantic-packet JSON bytes coexist with explicitly labeled latent/edge/motion proxies. A real codec bitstream, entropy coding, modulation, and channel-coding CBR experiment remain future work.

### Partially Complete — SVD/Wan Generation

- Actual SVD and Wan segment-generation paths and start-only/start+end GPU execution have been confirmed. A full comparison across PSNR/LPIPS/DISTS, object preservation, drift, flicker, latency, and VRAM is still needed. The defensible claim is that the path can run, not yet that it is superior.

### Next Steps

1. Evaluate the full SNR range for image and video paths.
2. Include fading conditions in addition to AWGN.
3. Compare other models under matched input, channel, rate, resolution, and evaluator conditions.
4. Replace proxy-only accounting with actual bitstream/CBR measurement.
5. Connect Packet Verifier/Controller decisions to the diffusion sampler and test the closed loop.
6. Produce the final performance table, ablations, and comparison protocol.

### Suggested Closing Sentence

- “The end-to-end structure and semantic-reliability measurement system now operate on real models and videos; the next phase is to demonstrate gains under matched channel and transmission-rate conditions.”

---

## Slide 9. Follow-on Research Based on Current Results

### Main Point

- The existing work is organized into three independent but compatible research directions: transmission efficiency, receiver trustworthiness, and evaluation fairness.

### ① Rate-Adaptive Video Semantic Communication

#### Current Foundation

- The system already selects reuse, recompute, or generate based on video change and provides a transmission-accounting PoC. This is the basis for sending less information in low-change segments rather than allocating a fixed amount to every frame.

#### Follow-on Work

- Define a real bitstream for latent and side information.
- Add quantization, entropy coding, modulation, and channel coding to calculate actual CBR.
- Adapt keyframe and decision thresholds to a reliability target or transmission budget.
- Allocate bits/symbols according to scene transitions, rapid motion, and object importance.

#### Goal

- Find the minimum transmission rate that satisfies a semantic-reliability target and report the rate–reliability Pareto frontier.

### ② Self-Verifying Generative Restoration at the Receiver

#### Current Foundation

- The Packet Verifier compares transmitted and reconstructed packets, OWLv2/VQA strengthens presence decisions, and the Controller proposes actions for additions, omissions, and distortions. The receiver can inspect its own output without accessing the original source image.

#### Follow-on Work

- Connect Controller actions to prompt, negative prompt, ControlNet weight, guidance scale, and diffusion step.
- Limit retry count, transmission cost, and compute cost.
- Train the Semantic Packet Fidelity Adapter to inject packet conditions directly into generation.
- Train a Hallucination Critic to distinguish acceptable background content from a true packet-inconsistent hallucination.

#### Goal

- Build a receiver that does not accept every generated result blindly, but readjusts or falls back when confidence is inadequate.

### ③ Semantic Transmission Evaluation Benchmark

#### Current Foundation

- Image metrics, video PTC/SFR/SDI, ten-video held-out remeasurement, and separate GT-object-only/open-world modes provide a basis for comparing different transmission and generation strategies in one semantic-reliability language.

#### Follow-on Work

- Define a fixed evaluation matrix across the full SNR range and fading/packet-loss conditions.
- Compare WITT, DiffJSCC, SGDJSCC variants, and the proposed method under matched inputs, rates, and channels.
- Match rate using actual CBR or bitstream size.
- Reduce detector/VQA/VLM bias with GT and independent judging.
- Report per-frame, per-video, and category-level results with uncertainty.

#### Goal

- Create a fair protocol that asks not which method produces the most attractive video, but which method preserves the source meaning most reliably under the same channel and transmission rate.

### Relationship Among the Three Directions

- The rate-adaptive policy decides what to transmit under a budget, the self-verifying receiver decides whether the output can be trusted, and the benchmark measures both decisions independently. They can therefore become one closed-loop semantic communication system.

### Final Takeaway

- The main contribution is not merely attaching a video front end to an image restoration model. The project preserves the baseline path while connecting temporal transmission decisions, packet-based receiver verification, held-out semantic evaluation, and rate accounting in a reproducible experimental framework. The remaining work is to turn that framework into a matched-rate, fair-comparison demonstration of quantitative performance gains.

---

## Key Evidence Files

- Project overview and base pipeline: `README.md`, `docs/etri_overview.md`
- Three limitations and staged strategy: `docs/etri_strategy.md`
- Video extension and scope: `docs/video_extension_lgvsc.md`
- Stage-1 video/temporal validation: `docs/etri_stage1_validation.md`
- Interpretation of the real OWLv2/VQA 50-job run: `docs/etri_owlv2_vqa_readiness.md`
- Ten-video temporal/accounting report: `outputs/etri_video_eval/final_report.md`
- 2026-08-16 remote GPU execution status: `outputs/remote_hq_4090_20260816/hq_validation_status.json`
- Real OWLv2/VQA/ensemble summary: `outputs/remote_hq_4090_20260816/remeasure/summary_metrics.md`
- Video codec/rate artifacts: `outputs/remote_hq_4090_20260816/rate_benchmark/`
- Checkpoint separation and reproducibility rules: `docs/checkpoint_usage.md`

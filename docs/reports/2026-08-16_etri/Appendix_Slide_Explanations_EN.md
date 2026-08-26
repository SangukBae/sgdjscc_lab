---
status: frozen
updated: 2026-08-26
owner: ETRI SGD-JSCC 연구팀
source_commit: d0d3bfb
supersedes:
---

# Appendix Slide Explanations for ETRI Progress Meeting

> Frozen report snapshot. Legacy document paths in the body refer to the layout at
> presentation time; use [the current documentation index](../../README.md) now.

- Target slides: Appendix A, Appendix B-1, Appendix B-2
- Target deck: `ETRI_연구진행상황공유_20260816_ETRI용_업데이트_부록추가.pptx`
- Purpose: explain the added appendix slides in clear English for ETRI's requested items: evaluation inputs, metric definitions, and calculation methods.

---

## Appendix A. Evaluation Inputs and Selection Criteria

This slide explains what data we used for evaluation and why those inputs were selected. The goal is to make clear that the evaluation is not based on arbitrary examples, but on inputs that cover both still-image reconstruction and video-specific semantic reliability.

### Image Evaluation Inputs

For still-image evaluation, we used the Kodak image dataset. Kodak is a commonly used image set for image restoration and compression evaluation, so it is suitable for checking whether the image reconstruction pipeline produces stable and reproducible results.

The evaluation was performed across multiple SNR values:

```text
-5, 0, 5, 10, 15, 20, 25 dB
```

This SNR sweep is important because the system is designed for wireless semantic media transmission. If we only evaluate one channel condition, we cannot tell whether the method is robust when the channel becomes noisy. By sweeping SNR from low to high, we can observe how pixel quality and semantic reliability change as the channel condition improves.

The image evaluation records both conventional image-quality metrics and semantic metrics. In other words, we do not only ask whether the reconstructed image is visually close to the original. We also ask whether the intended objects and semantic content are preserved after transmission and generative reconstruction.

The main output file for this image evaluation is:

```text
outputs/remote_hq_4090_20260816/image/kodak_snr_sweep.csv
```

### Why Kodak Was Selected

Kodak was selected because it provides a reproducible image benchmark for checking reconstruction quality under different SNR values. It is useful for still-image experiments because the same input images can be repeatedly evaluated across channel conditions.

However, Kodak alone cannot evaluate temporal behavior. It cannot show whether objects flicker, disappear, reappear, or drift over time. That is why the video evaluation set is added separately.

### Video Evaluation Inputs

For video evaluation, we used ten 100-frame video sequences. The purpose of this set is to cover typical temporal cases that are important for semantic media transmission.

The ten videos are:

```text
01 person walk
02 car pass
03 dog walk
04 person + car
05 camera pan person
06 handheld sign
07 person enter
08 car exit
09 scene cut chair car
10 busy sidewalk
```

Each video was selected to test a different semantic or temporal scenario.

`person walk` checks whether a single human object remains semantically consistent over time. `car pass` and `dog walk` test moving objects. `person + car` checks whether multiple object categories can be preserved together. `camera pan person` is important because the object identity may remain the same while the camera motion is large. This tests whether a reuse-only strategy would miss motion changes. `handheld sign` includes camera shake and structural/text-like information. `person enter` and `car exit` test object birth and death. `scene cut chair car` tests scene transition handling. `busy sidewalk` tests a more complex background with multiple possible object candidates.

### Held-Out Remeasurement

After running the real-model video baseline, the reconstructed videos were remeasured using five OWLv2/VQA/ensemble modes:

```text
10 videos x 5 modes = 50 remeasurement jobs
```

This is important because the final evaluation should be separated from the internal metrics used during reconstruction or regeneration. In the presentation, this is referred to as held-out remeasurement.

The key point is that we separate two different claims:

1. The pipeline runs and produces video reconstruction outputs.
2. The reconstructed outputs can be independently re-evaluated using stronger object-presence checks.

This prevents us from using the same metric both to control the reconstruction loop and to claim final performance.

### Scope Limitation

This slide also states the current reporting boundary. The current appendix reports that the evaluation path, output files, and remeasurement procedure are available. It does not yet claim that the method is superior under the same true bitstream or CBR condition against all competing methods.

The fair comparison under identical CBR, fading channels, and other baseline models is listed as follow-up work.

---

## Before Appendix B-1: Newly Introduced Metrics for This Project

Before explaining the detailed formulas in Appendix B-1, it is useful to first clarify which metrics are newly introduced for this project. PSNR, SSIM, LPIPS, and CLIP are existing metrics that are already widely used. The main metrics newly defined or extended in this project are the SRS family for semantic transmission reliability and the PTC/SFR/SDI metrics for video-level temporal semantic stability.

The first key metric is **SRS, Semantic Reliability Score**. SRS is designed to measure not only visual quality, but whether the original or transmitted semantic intent remains in the reconstructed output. It combines CLIP-based global semantic similarity and object preservation, while penalizing missing objects and additional objects. In other words, SRS focuses less on “does the image look good?” and more on “was the transmitted meaning preserved?”

The next newly added metrics are **PTC, SFR, and SDI**, which were introduced for the video extension. These metrics capture temporal problems that cannot be observed from still-image evaluation alone.

- **PTC, Packet-Temporal Consistency**, measures whether the transmitted packet and reconstructed packet remain consistent across the video timeline.
- **SFR, Semantic Flicker Rate**, measures abnormal object birth and death events that cannot be explained by actual changes in the original video.
- **SDI, Semantic Drift Index**, measures whether semantic mismatch increases as frames get farther away from the nearest keyframe.

In summary, PSNR, SSIM, LPIPS, and CLIP are existing baseline metrics. The metrics to emphasize as newly introduced in this project are **SRS**, and the video-oriented extensions **PTC/SFR/SDI**. Appendix B-1 first explains the existing image/single-frame metrics and SRS, while Appendix B-2 explains the Packet Verifier and the temporal video metrics PTC, SFR, and SDI.

---

## Appendix B-1. Image and Single-Frame Metric Calculation Methods

This slide explains the metrics used for still images and individual video frames. These metrics can be divided into three groups:

1. Pixel-level quality metrics
2. Perceptual and global semantic similarity metrics
3. Object-level semantic reliability metrics

The reason for using all three groups is that generative reconstruction can look natural even when it changes the original semantic content. Therefore, conventional image-quality metrics alone are not enough.

### PSNR: Pixel Distortion

PSNR measures pixel-level distortion between the original image and the reconstructed image. It is computed from MSE, the mean squared error between the two images.

The basic idea is:

```text
MSE = average squared pixel error
PSNR = 10 log10(MAX^2 / MSE)
```

If MSE is small, PSNR becomes high. Therefore, higher PSNR means the reconstructed pixels are closer to the original pixels.

PSNR is useful for measuring signal distortion, but it does not directly measure semantic preservation. For example, a generative model may produce a visually plausible image with a lower PSNR, or it may achieve reasonable PSNR while missing an important object.

### SSIM: Structural Similarity

SSIM measures structural similarity between the original and reconstructed images. It compares luminance, contrast, and structure.

The slide shows the standard SSIM formula. The important interpretation is:

```text
higher SSIM = better structural similarity
```

SSIM is closer to perceived image structure than PSNR, but it is still mainly a frame-level visual-quality metric. It does not explicitly check whether the correct objects, relations, or scene meaning are preserved.

### LPIPS: Perceptual Distance

LPIPS measures distance in a learned feature space. Instead of comparing raw pixels, it compares deep neural network features extracted from the original and reconstructed images.

The basic idea is:

```text
LPIPS = weighted distance between deep features
```

Lower LPIPS means the reconstructed image is perceptually closer to the original. This is useful because human perception is often better aligned with feature-space differences than raw pixel differences.

However, LPIPS still does not fully answer whether the transmitted semantic intent is preserved. It is a perceptual distance metric, not a structured semantic verification metric.

### CLIP Image-Image Similarity

CLIP image-image similarity compares the CLIP image embedding of the original image with the CLIP image embedding of the reconstructed image.

The basic idea is:

```text
CLIP I-I = cosine similarity between original image embedding and reconstructed image embedding
```

Higher similarity means the two images are close in CLIP's semantic embedding space. This helps measure global semantic similarity beyond pixel matching.

### CLIP Text-Image Similarity

CLIP text-image similarity compares the text embedding of the caption with the image embedding of the reconstruction.

The basic idea is:

```text
CLIP T-I = cosine similarity between caption embedding and reconstructed image embedding
```

This measures whether the reconstructed image matches the transmitted or extracted textual description. It is useful because the receiver uses text/caption information as part of the generative reconstruction condition.

### Object Preservation, Missing, and Additional Object Rates

To evaluate semantic reliability more directly, we compare object sets from the reference packet and the reconstructed packet.

Let:

```text
O_ref = object set in the original/transmitted packet
O_rec = object set in the reconstructed packet
```

Then:

```text
OPR = object preservation rate
MR  = missing object rate
AR  = additional object rate
```

`OPR` measures how many reference objects are preserved in the reconstruction. `MR` measures how many reference objects are missing. `AR` measures how many objects appear in the reconstruction but were not in the reference packet.

This is important for hallucination analysis. A reconstructed image may look realistic, but if it contains an additional object that was not transmitted, it is semantically unreliable for this task.

### SRS: Semantic Reliability Score

SRS combines global semantic similarity and object-level preservation into one score:

```text
SRS = 0.30*Cii + 0.25*Cti + 0.25*OPR - 0.10*MR - 0.10*AR
```

Where:

```text
Cii = CLIP image-image similarity
Cti = CLIP text-image similarity
OPR = object preservation rate
MR  = missing object rate
AR  = additional object rate
```

The score increases when the reconstruction is semantically similar and preserves objects. It decreases when objects are missing or additional objects appear.

The key message is that SRS does not replace PSNR, SSIM, or LPIPS. Instead, it adds a semantic reliability axis that is more aligned with the goal of semantic media transmission.

---

## Appendix B-2. Packet Verifier and Temporal Video Metric Calculation Methods

This slide explains the metrics that are specific to packet verification and video temporal consistency.

The motivation is simple: image-level metrics are not enough for video. A video reconstruction may look acceptable frame by frame, but still have temporal semantic errors. For example, an object may flicker, disappear without reason, or gradually drift away from the original meaning as the frame gets farther from the keyframe.

### Packet Verifier Severity

The Packet Verifier compares the transmitted semantic packet with the semantic packet re-extracted from the reconstructed image or frame.

It separates errors into five categories:

```text
missing objects
additional objects
relation errors
attribute errors
scene mismatch
```

These errors are combined into one severity score:

```text
severity = 0.30*e_miss
         + 0.25*e_add
         + 0.20*e_rel
         + 0.15*e_attr
         + 0.10*e_scene
```

The interpretation is:

```text
severity near 0 = packet match is good
larger severity = stronger semantic mismatch
```

The weights reflect the relative importance of error types. Missing objects and additional objects receive high weights because they directly affect semantic intent and hallucination risk.

### Packet Consistency

Packet consistency is the frame-level agreement score between the transmitted packet and the reconstructed packet.

The slide defines:

```text
C_pkt(t) = 0.50*C_obj(t)
         + 0.20*C_rel(t)
         + 0.20*C_attr(t)
         + 0.10*C_scene(t)
```

Where:

```text
C_obj(t)   = object consistency at frame t
C_rel(t)   = relation consistency at frame t
C_attr(t)  = attribute consistency at frame t
C_scene(t) = scene consistency at frame t
```

This score is used as the basic frame-level consistency value for PTC and SDI.

### PTC: Packet-Temporal Consistency

PTC measures whether the transmitted semantic meaning is maintained across the entire video.

The simplified calculation shown on the slide is:

```text
PTC = mean_t C_pkt(t)
```

In words, we compute packet consistency for each frame and average it over time.

The interpretation is:

```text
higher PTC = transmitted meaning is preserved more consistently over the video
```

PTC is useful because it summarizes semantic consistency over the full sequence, not only for one frame.

### SFR: Semantic Flicker Rate

SFR measures abnormal object flicker. In this context, flicker means that an object appears or disappears in the reconstructed video even though the original/reference video does not explain that change.

The simplified slide formula is:

```text
SFR = mean_t spurious_birth_death(t) / union_objects(t)
```

The important detail is the word `spurious`. If an object actually enters or exits in the original video, that change should not be counted as an error. SFR only counts object birth/death events that are not explained by the original semantic packet sequence.

The interpretation is:

```text
lower SFR = fewer unexplained object flickers
```

This metric is especially important for generative video reconstruction because generated frames can introduce unstable object appearances.

### SDI: Semantic Drift Index

SDI measures whether semantic mismatch grows as the frame moves farther away from the nearest keyframe.

The simplified slide formula is:

```text
SDI = slope_LS(1 - C_pkt(t) vs keyframe_distance(t))
```

Here:

```text
1 - C_pkt(t) = packet drift at frame t
keyframe_distance(t) = distance from the most recent keyframe
slope_LS = least-squares regression slope
```

If SDI is positive and large, it means semantic error tends to increase as the frame gets farther from the keyframe. If SDI is near zero, it means there is no strong drift trend over keyframe distance.

The interpretation is:

```text
SDI near 0 = semantically stable over time
large positive SDI = semantic drift increases away from keyframes
```

### Temporal SRS

Temporal SRS summarizes semantic reliability across the video timeline.

The simplified slide formula is:

```text
Temporal SRS = mean_t SRS(t) + packet score summary
```

In practice, this means that frame-level SRS and packet-based reliability values are aggregated over time to provide a video-level semantic reliability summary.

Temporal SRS is useful as an overall score, but it should be interpreted together with PTC, SFR, and SDI. A single average score can hide short but serious errors, while PTC/SFR/SDI explain different temporal failure modes.

### Held-Out Interpretation

The last note on the slide says that final performance claims should be based on held-out remeasurement, not on loop-internal metrics.

This matters because if a metric is used inside the reconstruction loop to decide regeneration or guidance adjustment, using that same metric again as the final evaluation can lead to circular evaluation. Therefore, final claims should use a separate held-out evaluation path, including OWLv2/VQA remeasurement and GT object-only/open-world interpretation.

### Key Takeaway

The three most important new temporal metrics are:

```text
PTC: whether transmitted meaning is preserved over time
SFR: whether unexplained object flicker occurs
SDI: whether semantic drift increases away from keyframes
```

Together, these metrics extend the evaluation from still-image semantic reliability to video-level semantic reliability.

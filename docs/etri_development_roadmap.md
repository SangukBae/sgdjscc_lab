> [← docs index](./README.md)

# ETRI Development Roadmap

This document reorganizes the ETRI task into a practical development order.
It combines:

- the eight ETRI task goals from [etri_overview.md](./etri_overview.md)
- the prioritized SGD-JSCC limitations `1`, `2`, `5`, `6` from
  [limitation_reference_map.md](./limitation_reference_map.md)

The ordering principle is simple: build the measurement and software baseline
first, then add research improvements on top of that baseline.

## Recommended Order

| Step | Development item | Why this comes here | Main source |
|---|---|---|---|
| 1 | Preserve the original SGD-JSCC path and extension rule | Baseline reproducibility must be fixed first; otherwise later comparisons are not trustworthy. | [etri_overview.md](./etri_overview.md) |
| 2 | Organize the modular software structure | `channels/`, `guidance/`, `models/`, `pipelines/`, `evaluators/` must be stable before research changes accumulate. | [etri_overview.md](./etri_overview.md) |
| 3 | Build the end-to-end evaluation framework skeleton | Input -> channel -> reconstruction -> evaluation -> `results.csv` logging should exist before optimization work starts. | [etri_overview.md](./etri_overview.md) |
| 4 | Fix the evaluation philosophy around semantic intent | The project should define success around semantic preservation, not just pixel fidelity. | [etri_overview.md](./etri_overview.md) |
| 5 | Implement the evaluator suite including hallucination metrics | CLIP, object preservation, missing/additional objects, hallucination, and quality metrics are prerequisites for meaningful experiments. | [etri_overview.md](./etri_overview.md) |
| 6 | Integrate SRS as the headline metric | SRS should be finalized only after its component metrics are stable. | [etri_overview.md](./etri_overview.md) |
| 7 | Improve limitation `1`: make semantics explicit instead of hidden only in latent transport | Explicit semantic units are needed for packet extraction, semantic verification, and controllable reconstruction. | [limitation_reference_map.md](./limitation_reference_map.md) |
| 8 | Improve limitation `2`: upgrade weak `caption + canny` guidance to richer semantic guidance | Stronger guidance is needed for object, attribute, relation preservation and hallucination control. | [limitation_reference_map.md](./limitation_reference_map.md) |
| 9 | Improve limitation `6`: reduce semantic side-information overhead | After semantic units/guides exist, selective and compact transmission can be designed realistically. | [limitation_reference_map.md](./limitation_reference_map.md) |
| 10 | Improve limitation `5`: reduce diffusion reconstruction latency | Few-step decoding and consistency-style acceleration should be evaluated only after semantic quality can already be measured. | [limitation_reference_map.md](./limitation_reference_map.md) |
| 11 | Separate guide corruption models from channel noise | Once richer guidance exists, guide-specific corruption rules become necessary for realistic experiments. | [etri_overview.md](./etri_overview.md) |
| 12 | Lock the fair comparison protocol | Final protocol fixing should come last, after the pipeline, metrics, and improved methods are all settled. | [etri_overview.md](./etri_overview.md) |

## Short Version

```text
baseline preservation
-> modular structure
-> end-to-end pipeline
-> semantic-first evaluation philosophy
-> evaluator suite
-> SRS integration
-> limitation 1 improvement
-> limitation 2 improvement
-> limitation 6 improvement
-> limitation 5 improvement
-> guide corruption model
-> fair comparison protocol
```

## Why Limitations 1, 2, 6, 5 Appear in This Order

- `1` comes first because explicit semantic units are the foundation for later
  semantic control and verification.
- `2` follows because richer guidance depends on having a clearer semantic
  representation than the current latent-only baseline.
- `6` comes after `1` and `2` because selective transmission only makes sense
  after the transmitted semantic content has been defined.
- `5` comes after the semantic improvements because latency reduction must be
  evaluated against semantic quality, not only against runtime.

## Practical Interpretation

- Steps `1` to `6` establish the ETRI evaluation and software baseline.
- Steps `7` to `10` implement the priority SGD-JSCC limitation improvements.
- Steps `11` to `12` finalize realistic experiment design and paper-quality
  comparison protocol.

## Related Documents

- [etri_overview.md](./etri_overview.md) — ETRI task goals and framework scope
- [limitation_reference_map.md](./limitation_reference_map.md) — prioritized SGD-JSCC limitations and references
- [phase4.md](./phase4.md) — Phase 4 status and design
- [phase5.md](./phase5.md) — Phase 5 status and design

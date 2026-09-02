# Negative-semantics G0 data contract

This directory freezes the G0 split, ontology, and event-annotation contract.

- `dataset_split_manifest.json`: exact current split lists and SHA-256 for every assigned input/GT file
- `dataset_split_manifest.sha256`: detached hash of the split manifest
- `ontology_manifest.json`: concept IDs and the `present` / `confirmed_absent` / `unknown` state contract
- `event_schema.schema.json`: JSON Schema for `ENTER`, `EXIT`, `OCCLUDE`, `REAPPEAR`, and `SCENE_CUT`
- `etri_legacy_event_annotations.jsonl`: segment-boundary-derived legacy evidence; none is human-verified or eligible for the Pilot gate

Empty Pilot/Train/Validation/Held-out lists are deliberate. They make the missing
data visible and auditable; no runtime sampler may fill them implicitly.

Run the dependency-free audit from the repository root:

```bash
python scripts/audit_negative_semantics_g0.py --repo-root .
python scripts/audit_negative_semantics_g0.py --repo-root . --require-pass
```

The first command validates and reports the expected `NOT_PASSED` state. The
second exits with status 4 until every preregistered G0 pass condition is met.

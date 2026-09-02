# Negative-semantics G0 data contract

This directory freezes the G0 v1.1 split, ontology, acquisition, and
event-annotation contract.

- `dataset_split_manifest.json`: exact current split lists and SHA-256 for every assigned input/GT file
- `dataset_split_manifest.sha256`: detached hash of the split manifest
- `ontology_manifest.json`: concept IDs and the `present` / `confirmed_absent` / `unknown` state contract
- `event_schema.schema.json`: JSON Schema for `ENTER`, `EXIT`, `OCCLUDE`, `REAPPEAR`, and `SCENE_CUT`
- `etri_legacy_event_annotations.jsonl`: segment-boundary-derived legacy evidence; none is human-verified or eligible for the Pilot gate
- `youtube_vos_category_mapping.json`: frozen dataset-native to core-ontology mapping
- `youtube_vos_pilot_event_candidates.jsonl`: 33 deterministic candidates; none is human-verified yet
- `pilot_review/`: independent annotator and adjudicator CSV templates
- `data_use_approval.json`: project approval and accepted non-redistribution boundary
- `heldout_seal.json`: DAVIS held-out seal tied to the split-manifest hash
- `audit_report_v1_1.json`: current 12/13 machine-audit evidence

The current split has Pilot 33, Train 3,471, Development 10, Validation 474,
and source-disjoint Held-out Test 30. Raw external assets stay git-ignored.
No runtime sampler may alter these lists.

Run the dependency-free audit from the repository root:

```bash
python scripts/audit_negative_semantics_g0.py --repo-root .
python scripts/audit_negative_semantics_g0.py --repo-root . --require-pass
```

The first command currently validates 12 of 13 conditions and reports
`NOT_PASSED`. The second exits with status 4 until two independent human reviews
and required adjudication yield 30--50 accepted Pilot clusters.

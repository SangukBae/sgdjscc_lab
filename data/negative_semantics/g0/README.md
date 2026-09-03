# Negative-semantics G0 data contract

This directory freezes the G0 v1.2 split, ontology, acquisition, and
event-annotation contract.

- `dataset_split_manifest.json`: exact current split lists and SHA-256 for every assigned input/GT file
- `dataset_split_manifest.sha256`: detached hash of the split manifest
- `ontology_manifest.json`: concept IDs and the `present` / `confirmed_absent` / `unknown` state contract
- `event_schema.schema.json`: JSON Schema for `ENTER`, `EXIT`, `OCCLUDE`, `REAPPEAR`, and `SCENE_CUT`
- `etri_legacy_event_annotations.jsonl`: segment-boundary-derived legacy evidence; none is human-verified or eligible for the Pilot gate
- `youtube_vos_category_mapping.json`: frozen dataset-native to core-ontology mapping
- `ovis_category_mapping.json`: frozen OVIS-native to core-ontology mapping
- `ovis_official_gt_event_annotations.jsonl`: 40 official-GT-derived Pilot events from 40 videos
- `youtube_vos_pilot_event_candidates.jsonl`, `pilot_review/`: superseded v1.1 historical inputs
- `data_use_approval.json`: project approval and accepted non-redistribution boundary
- `heldout_seal.json`: DAVIS held-out seal tied to the split-manifest hash
- `audit_report_v1_1.json`: historical 12/13 human-review gate evidence
- `audit_report_v1_2.json`: current 13/13 official-GT audit evidence

The current split has OVIS Pilot 40, Train 3,471, Development 10, Validation 507,
and source-disjoint Held-out Test 30. Raw external assets stay git-ignored.
No runtime sampler may alter these lists.

Run the dependency-free audit from the repository root:

```bash
python scripts/audit_negative_semantics_g0.py --repo-root .
python scripts/audit_negative_semantics_g0.py --repo-root . --require-pass
```

Both commands validate 13 of 13 conditions and report `PASSED`; the second exits
with status 0. This completes the data/protocol gate, not GPU training. Human
perception claims remain out of scope: the paper may claim only official-GT-
anchored automatic additional-object and ghost-track endpoints unless a separate
human study is later preregistered.

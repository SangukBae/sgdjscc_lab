# float32 digital diagnostics — integrated run (full profile)

- generated: 2026-08-27T18:27:26.662313+00:00
- output_root: outputs/f32dig_20260827_164017
- stage_failures: 0
- execution_mode: three_gpu_parallel
- physical_devices: 0,1,2

**This section consolidates each stage's own verdict (see docs/protocols/float32_digital_diagnostics.md for the classification criteria); it is NOT itself a new judgment, only a rollup of what each stage's own summary.json/verdicts.jsonl already recorded. Only `ablation == "baseline"` AND `status == "final"` rows feed any dominant-verdict tally below -- auxiliary edge-equalizing ablations and still-provisional baseline rows are listed separately, never summed into it. When stages overlap, the richest evidence wins (`baseline_with_vae_direct` > `baseline_only` > provisional); only disagreements at the same evidence level are conflicts.**

## per-stage verdict summary (baseline, final only)

| stage | n_frames_processed | dominant_verdict | baseline verdict counts | evidence levels | provisional | auxiliary | failed_cases |
|---|---:|---|---|---|---:|---:|---:|
| stage3_single_frame_paths | 1 | inconclusive | inconclusive=1 | baseline_only=1 | 0 | 0 | 0 |
| stage4_single_frame_ablations | 1 | inconclusive | inconclusive=1 | baseline_with_vae_direct=1 | 0 | 2 | 0 |
| stage5_paired_frames | 20 | inconclusive | inconclusive=20 | baseline_with_vae_direct=20 | 0 | 0 | 0 |
| stage6_core_conditions/worker_00_normal_motion | 100 | inconclusive | (none) | (none) | 0 | 0 | 0 |
| stage6_core_conditions/worker_01_semantic_change | 100 | inconclusive | (none) | (none) | 0 | 0 | 0 |
| stage6_core_conditions/worker_02_scene_cut | 100 | inconclusive | (none) | (none) | 0 | 0 | 0 |

## overall (baseline, final only, deduplicated across stages by (video, frame))

- dominant_verdict: `inconclusive`
  - `inconclusive`: 20

## auxiliary evidence (serialized_raw_edge / awgn_edge_retransmit, never summed into the overall count)

- `packet_tx_rx_issue`: 2

## conflicts

- none detected.

## run-level artifact hashes (sha256)

- `execution_plan.json`: `075fcbf589317fe199276bbdaca82fe191f3f7d707e3de89455e4fcae13fe86a`

## per-stage artifact hashes (sha256)

### stage3_single_frame_paths
- `run_manifest.json`: `016f839316df6650beaf80ebe4ba2c0d482ecfad9ae165265c08f7b04d70b96d`
- `summary.json`: `0809e6cef30a0056ae9915457404fcc320f2f93cc3086c4730e7d3fb71bffe71`
- `REPORT.md`: `8067ce8afcfb08256fc1a02e1339100ef8aafd06f181a1bc4a5cdb0b3d3f8d8e`
- `path_comparison.csv`: `7eb90e044fdb685a4f5e66ddafbc052be929ddfbe872ef441f3aa5d6ff7376ef`
- `verdicts.jsonl`: `939eb01d9f54d82dd0b343fa278976c57aeaf902cc6072cecde87ab0bd2440a0`

### stage4_single_frame_ablations
- `run_manifest.json`: `53c77d4d3c37b3e0038f23850bc2f855630aa5535087eb7df8f91d1d8185bca2`
- `summary.json`: `0d885f00afec21a91da3c2cfa2765aa8eb79c88e5926209494c859c093e0bce0`
- `REPORT.md`: `6b45b67b280bdffdfd9e69eb34b4ad1c8b0e1f5c6036d902ac7e043d2acaae27`
- `path_comparison.csv`: `bf7ec65b132480c83ce6f27a9c99acd4185188f77cae3bbb97c84172cde036a3`
- `verdicts.jsonl`: `cda9f3c8d6a7852148f88d6a1443ca6839085f41828a693999ff4bf5f64dc0a6`

### stage5_paired_frames
- `run_manifest.json`: `d560d5d86c189c28e3b05d5c43d7b835c525317995317b9efd0548c5c1c18d29`
- `summary.json`: `4da73ce18bf344b1f5de293f690795935fcb5eb0b56fe18b032cb5a916788906`
- `REPORT.md`: `e2e639b2cc486a40a178f7297900e9dd051303029f7b3429a800d5f17a771c06`
- `path_comparison.csv`: `d3b4323d90faaf6324312929ee3f19237b321e1bc0971c5ea082a1c665bcebcf`
- `verdicts.jsonl`: `809a0e4dfb6495b171d744cd6b86536df528880305b20204cf62119f05aafef4`

### stage6_core_conditions/worker_00_normal_motion
- `run_manifest.json`: `6fb7250e10671989fd96841765f231a7a8c45ffcda05dce31f26eca11cca68dc`
- `summary.json`: `215c910868c774ab2472fc389bca6126ac5b01551f3f8be59858f18eb060a82e`
- `REPORT.md`: `45a09b2f150313df859ecdeb691696eb7d2df7fe3e18b4f9379f3f1dbb090e19`
- `path_comparison.csv`: `23dfb00bea781fa84b101cf713454bfb120830024b150105325204e4e41e9f37`

### stage6_core_conditions/worker_01_semantic_change
- `run_manifest.json`: `5175b55504d485002b6b607bf4f8b45d89291e532dac937b06fd72b4252da680`
- `summary.json`: `8acd7550da99a0fb92b8c1a5d594a83c4af49b96309665314f5900f8249319eb`
- `REPORT.md`: `45a09b2f150313df859ecdeb691696eb7d2df7fe3e18b4f9379f3f1dbb090e19`
- `path_comparison.csv`: `3f5b6b3792ac490182a7302113804ac980603cc5463885c7d8f03f675b780076`

### stage6_core_conditions/worker_02_scene_cut
- `run_manifest.json`: `15e8ffa83040929c99a75e49125f9aa57780ff927472191c104befd4828437f8`
- `summary.json`: `ce0c032405b22708a0124983c835913fc543a0d887d699b306d9477c8e304635`
- `REPORT.md`: `45a09b2f150313df859ecdeb691696eb7d2df7fe3e18b4f9379f3f1dbb090e19`
- `path_comparison.csv`: `f77f99609cad8b04f478b22498ff23e23a6150f243313306b0392238cf15c631`

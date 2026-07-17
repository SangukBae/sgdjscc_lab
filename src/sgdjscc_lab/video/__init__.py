"""video/ – Phase 4-B keyframe-oriented temporal extension.

Modules
-------
scene_change_detector – heuristic scene-boundary detection between frames.
keyframe_extractor    – GOP-like grouping into keyframes + inter-frame ranges.
semantic_delta        – packet-level change units between frames.
motion_residual       – lightweight pixel-motion / residual energy estimate.
temporal_pipeline     – keyframe/inter-frame orchestration + staged denoising
                        (semantic delta + motion dual reuse gate).
segment               – GOP/segment records aggregating frame results
                        (generate-branch attachment point, 1차 scope).
"""

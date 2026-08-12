"""Backward-compatible facade for :mod:`sgdjscc_lab.video.generation`.

New code should import the responsibility-specific modules under
``video.generation``. Existing scripts retain this import path unchanged.
"""

from .generation import (
    CONDITIONING_MODE_BIDIRECTIONAL,
    CONDITIONING_MODE_START_ONLY,
    MISSING_END_POLICY_ERROR,
    MISSING_END_POLICY_FALLBACK_START_ONLY,
    BidirectionalInterpolationGenerator,
    CopyGenerator,
    ExternalSegmentWorkerGenerator,
    GenerationMetadata,
    GenerationRequest,
    GenerationResult,
    InterpolationGenerator,
    SegmentGenerationRequest,
    SegmentGenerationResult,
    SegmentWorkerError,
    VideoGenerator,
    build_generator,
    save_generated_frames,
    validate_segment_request,
    validate_segment_result,
)

__all__ = [name for name in globals() if not name.startswith("_")]

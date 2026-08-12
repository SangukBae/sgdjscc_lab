"""Request, result, metadata, and validation contracts for video generation."""

from ._implementation import (
    CONDITIONING_MODE_BIDIRECTIONAL,
    CONDITIONING_MODE_START_ONLY,
    GenerationMetadata,
    GenerationRequest,
    GenerationResult,
    SegmentGenerationRequest,
    SegmentGenerationResult,
    validate_segment_request,
    validate_segment_result,
)

__all__ = [
    "CONDITIONING_MODE_BIDIRECTIONAL",
    "CONDITIONING_MODE_START_ONLY",
    "GenerationMetadata",
    "GenerationRequest",
    "GenerationResult",
    "SegmentGenerationRequest",
    "SegmentGenerationResult",
    "validate_segment_request",
    "validate_segment_result",
]

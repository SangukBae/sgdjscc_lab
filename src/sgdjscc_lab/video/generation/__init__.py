"""Video generation API grouped by responsibility.

The implementation remains behavior-compatible while callers can depend on a
narrow contracts, backend, worker, or factory module.
"""

from .backends import (
    MISSING_END_POLICY_ERROR,
    MISSING_END_POLICY_FALLBACK_START_ONLY,
    BidirectionalInterpolationGenerator,
    CopyGenerator,
    InterpolationGenerator,
)
from .base import VideoGenerator
from .contracts import (
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
from .external_worker import ExternalSegmentWorkerGenerator, SegmentWorkerError
from .factory import build_generator, save_generated_frames

__all__ = [name for name in globals() if not name.startswith("_")]

"""Out-of-process segment generation backend."""

from ._implementation import ExternalSegmentWorkerGenerator, SegmentWorkerError

__all__ = ["ExternalSegmentWorkerGenerator", "SegmentWorkerError"]

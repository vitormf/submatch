from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("submatch")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"  # pragma: no cover

from submatch.pipeline import PipelineConfig, run, run_batch
from submatch.types import MatchState, MatchResult, BatchPairResult, SegmentResult

__all__ = [
    "run", "run_batch", "PipelineConfig",
    "MatchState", "MatchResult", "BatchPairResult", "SegmentResult",
    "__version__",
]

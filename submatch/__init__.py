from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("submatch")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"  # pragma: no cover

# PipelineConfig.from_toml() is intentionally not re-exported; call it via PipelineConfig directly.
from submatch.pipeline import PipelineConfig, run, run_batch
from submatch.types import MatchState, MatchResult, BatchPairResult, SegmentResult

__all__ = [
    "run", "run_batch", "PipelineConfig",
    "MatchState", "MatchResult", "BatchPairResult", "SegmentResult",
    "__version__",
]

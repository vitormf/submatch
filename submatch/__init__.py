from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("submatch")
except PackageNotFoundError:
    __version__ = "unknown"

from submatch.pipeline import PipelineConfig, run, run_batch

__all__ = ["run", "run_batch", "PipelineConfig", "__version__"]

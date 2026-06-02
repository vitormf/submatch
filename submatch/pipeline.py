from __future__ import annotations
import dataclasses
import threading
from collections.abc import Callable
from pathlib import Path

from submatch import output


@dataclasses.dataclass
class PipelineConfig:
    model: str = "base"
    threshold: float = 0.35
    cross_threshold: float | None = None
    segments: int | None = None
    language: str | None = None
    sync: bool = True
    drift_threshold: float = 2.0
    device: str = "auto"
    audio_track: str | None = None
    workers: int | None = None
    use_cache: bool = True
    cache_dir: Path | None = None
    cache_ttl_days: int | None = None
    cache_max_mb: int | None = None
    resync: bool = False
    verbose: bool = False
    on_segment: Callable[[int, int], None] | None = None
    on_pair_complete: Callable[[output.BatchPairResult], None] | None = None


_model_local = threading.local()
_embed_local = threading.local()


def run(
    video: Path,
    subtitle: Path,
    config: PipelineConfig | None = None,
) -> output.MatchResult:
    raise NotImplementedError


def run_batch(
    pairs: list[tuple[Path, Path]],
    config: PipelineConfig | None = None,
) -> list[output.BatchPairResult]:
    raise NotImplementedError

from __future__ import annotations
import concurrent.futures
import dataclasses
import os
import shutil
import threading
from collections.abc import Callable
from pathlib import Path

from submatch import telemetry, transcribe
from submatch import cache as _cache_module
from submatch import scoring as _scoring
from submatch.types import BatchPairResult, MatchResult, MatchState


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
    pass_unsure: bool = False
    keep_synced: bool = False
    delete_failures: bool = False
    verbose: bool = False
    on_segment: Callable[[int, int], None] | None = None
    on_pair_complete: Callable[[BatchPairResult], None] | None = None

    @classmethod
    def from_toml(cls) -> "PipelineConfig":
        from submatch import config as _config
        cfg = _config.load_config()
        return cls(
            model=cfg.get("model", "base"),
            threshold=cfg.get("threshold", 0.35),
            cross_threshold=cfg.get("cross_threshold"),
            segments=cfg.get("segments"),
            language=cfg.get("language"),
            sync=not cfg.get("no_sync", False),
            drift_threshold=cfg.get("drift_threshold", 2.0),
            device=cfg.get("device", "auto"),
            audio_track=cfg.get("audio_track"),
            workers=cfg.get("workers"),
            use_cache=not cfg.get("no_cache", False),
            cache_dir=(
                Path(cfg["cache_dir"]).expanduser()
                if cfg.get("cache_dir") else None
            ),
            cache_ttl_days=cfg.get("cache_ttl_days"),
            cache_max_mb=cfg.get("cache_max_mb"),
            resync=cfg.get("resync", False),
            pass_unsure=cfg.get("pass_unsure", False),
            keep_synced=cfg.get("keep_synced", False),
            delete_failures=cfg.get("delete_failures", False),
        )


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_workers(requested: int | None, device: str) -> int:
    if requested is not None:
        return requested
    if device == "cuda":
        return 1
    return min(4, os.cpu_count() or 1)


_model_local = threading.local()


def _get_model(model_name: str, device: str):
    if not hasattr(_model_local, "model"):
        _model_local.model = transcribe.load_model(model_name, device=device)
    return _model_local.model


def _apply_postprocessing(
    result: MatchResult,
    subtitle: Path,
    config: PipelineConfig,
) -> MatchResult:
    passed = result.state == MatchState.PASS or (
        result.state == MatchState.UNSURE and config.pass_unsure
    )
    result = dataclasses.replace(result, passed=passed)
    if result.sync and result.sync.synced_srt_path:
        if config.keep_synced:
            kept = subtitle.with_stem(subtitle.stem + ".synced")
            shutil.copy(result.sync.synced_srt_path, kept)
        result.sync.synced_srt_path.unlink(missing_ok=True)
    if config.delete_failures and result.state == MatchState.FAIL:
        subtitle.unlink(missing_ok=True)
    return result


def _score_group_parallel(
    video: Path,
    subs: list[Path],
    config: PipelineConfig,
    model_name: str,
    device: str,
) -> list[BatchPairResult]:
    """Process all subtitles for one video in a single thread, sharing transcriptions."""
    model = _get_model(model_name, device)
    pair_config = dataclasses.replace(config, on_segment=None)
    results = []
    cache: _cache_module.VideoCache | None = None
    for sub in subs:
        try:
            result, new_cache = _scoring._score_pair(video, sub, pair_config, model, video_cache=cache)
            if result.state == MatchState.DRIFT and result.sync and config.resync:
                synced_path = result.sync.synced_srt_path
                try:
                    shutil.copy(synced_path, sub)
                finally:
                    synced_path.unlink(missing_ok=True)
                resync_config = dataclasses.replace(pair_config, sync=False)
                result, _ = _scoring._score_pair(video, sub, resync_config, model, video_cache=new_cache)
                result.state = _scoring._determine_state(result)
                result.resynced = True
            result = _apply_postprocessing(result, sub, pair_config)
            if cache is None:
                cache = new_cache
            pair_result = BatchPairResult(video=video, subtitle=sub, result=result, error=None)
        except Exception as exc:
            telemetry.capture(exc)
            pair_result = BatchPairResult(video=video, subtitle=sub, result=None, error=str(exc))
        results.append(pair_result)
        if config.on_pair_complete:
            config.on_pair_complete(pair_result)
    return results


def run(
    video: Path,
    subtitle: Path,
    config: PipelineConfig | None = None,
) -> MatchResult:
    if config is None:
        config = PipelineConfig()
    device = _resolve_device(config.device)
    model = _get_model(config.model, device)
    result, _ = _scoring._score_pair(video, subtitle, config, model)
    if result.state == MatchState.DRIFT and result.sync and config.resync:
        synced_path = result.sync.synced_srt_path
        try:
            shutil.copy(synced_path, subtitle)
        finally:
            synced_path.unlink(missing_ok=True)
        resync_config = dataclasses.replace(config, sync=False)
        result, _ = _scoring._score_pair(video, subtitle, resync_config, model)
        result.state = _scoring._determine_state(result)
        result.resynced = True
    result = _apply_postprocessing(result, subtitle, config)
    return result


def run_batch(
    pairs: list[tuple[Path, Path]],
    config: PipelineConfig | None = None,
) -> list[BatchPairResult]:
    if config is None:
        config = PipelineConfig()
    device = _resolve_device(config.device)
    workers = _resolve_workers(config.workers, device)

    results: list[BatchPairResult] = []

    if workers == 1:
        model = _get_model(config.model, device)
        video_caches: dict[Path, _cache_module.VideoCache] = {}
        for video, sub in pairs:
            try:
                cache = video_caches.get(video)
                match_result, new_cache = _scoring._score_pair(video, sub, config, model, video_cache=cache)
                if match_result.state == MatchState.DRIFT and match_result.sync and config.resync:
                    synced_path = match_result.sync.synced_srt_path
                    try:
                        shutil.copy(synced_path, sub)
                    finally:
                        synced_path.unlink(missing_ok=True)
                    resync_config = dataclasses.replace(config, sync=False)
                    match_result, _ = _scoring._score_pair(video, sub, resync_config, model, video_cache=new_cache)
                    match_result.state = _scoring._determine_state(match_result)
                    match_result.resynced = True
                match_result = _apply_postprocessing(match_result, sub, config)
                if cache is None:
                    video_caches[video] = new_cache
                pair_result = BatchPairResult(video=video, subtitle=sub,
                                             result=match_result, error=None)
            except Exception as exc:
                telemetry.capture(exc)
                pair_result = BatchPairResult(video=video, subtitle=sub,
                                              result=None, error=str(exc))
            results.append(pair_result)
            if config.on_pair_complete:
                config.on_pair_complete(pair_result)
    else:
        video_groups: dict[Path, list[Path]] = {}
        video_order: list[Path] = []
        for video, sub in pairs:
            if video not in video_groups:
                video_groups[video] = []
                video_order.append(video)
            video_groups[video].append(sub)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_video: dict[concurrent.futures.Future, Path] = {}
            for video in video_order:
                future = executor.submit(
                    _score_group_parallel, video, video_groups[video], config, config.model, device,
                )
                future_to_video[future] = video
            for future in concurrent.futures.as_completed(future_to_video):
                video = future_to_video[future]
                try:
                    group = future.result()
                    results.extend(group)
                except Exception as exc:
                    telemetry.capture(exc)
                    for sub in video_groups[video]:
                        pair_result = BatchPairResult(video=video, subtitle=sub,
                                                     result=None, error=str(exc))
                        results.append(pair_result)
                        if config.on_pair_complete:
                            config.on_pair_complete(pair_result)

    return results

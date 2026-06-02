# Architecture Cleanup: types.py, PipelineConfig Completion, args.py

## Goal

Three related improvements in one change:
1. Move result types out of `output.py` into `types.py` so `pipeline.py` no longer depends on the presentation module.
2. Complete `PipelineConfig` with `pass_unsure`, `keep_synced`, `delete_failures`, and a `from_toml()` classmethod so library users get full execution control without going through argparse.
3. Move `parse_args()` to `args.py` so `cli.py` is pure dispatch.

## Architecture

```
language.py ──┐
sync.py ───────┤
               ▼
            types.py ◄─── output.py  (formatting only)
               │
               ▼
            pipeline.py  (also imports config.py for from_toml())
               ▲
         cli.py / watch.py / report.py

args.py ──► cli.py
```

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `submatch/types.py` | **Create** | `MatchState`, `SegmentResult`, `MatchResult`, `BatchPairResult` |
| `submatch/args.py` | **Create** | `parse_args()` only |
| `submatch/output.py` | **Modify** | Remove type defs; `from submatch.types import …`; all formatting unchanged |
| `submatch/pipeline.py` | **Modify** | Import types from `types.py`; add 3 fields + `from_toml()` to `PipelineConfig`; handle new fields in `run()`, `run_batch()`, `_score_group_parallel()` |
| `submatch/cli.py` | **Modify** | `from submatch.args import parse_args`; remove `_should_fail`; remove manual sync/delete handling now owned by pipeline |
| `submatch/watch.py` | **Modify** | Remove manual `synced_srt_path.unlink()` in `_score_and_print` (pipeline handles it) |
| `submatch/__init__.py` | **Modify** | Re-export types: `from submatch.types import MatchState, MatchResult, BatchPairResult, SegmentResult` |
| `tests/test_output.py` | **Modify** | Update type imports: `submatch.output` → `submatch.types` |
| `tests/test_cli.py` | **Modify** | Update type imports; update `parse_args` import source |
| `tests/test_pipeline.py` | **Modify** | Update type imports |
| `tests/test_report.py` | **Modify** | Update type imports |
| `tests/test_args.py` | **Create** | Tests for `parse_args()` (moved from `test_cli.py`) |

## types.py

Exact contents moved from `output.py` (lines 1–49 of current `output.py`):

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from submatch.language import LanguageResult
from submatch.sync import SyncResult


class MatchState(str, Enum):
    PASS = "PASS"
    DRIFT = "DRIFT"
    FAIL = "FAIL"
    UNSURE = "UNSURE"


@dataclass
class SegmentResult:
    index: int
    start_ms: int
    score: float
    wer: float
    subtitle_text: str
    transcription: str


@dataclass
class MatchResult:
    confidence: float
    passed: bool
    threshold: float
    language: LanguageResult
    sync: SyncResult | None
    segments: list[SegmentResult]
    model: str
    cross_language: bool = False
    subtitle_language: str | None = None
    state: MatchState = MatchState.FAIL
    resynced: bool = False
    audio_track_index: int = 0
    audio_track_lang: str | None = None


@dataclass
class BatchPairResult:
    video: Path
    subtitle: Path
    result: MatchResult | None
    error: str | None
```

## PipelineConfig additions

Three new fields added to the dataclass (all default `False`):

```python
pass_unsure: bool = False
keep_synced: bool = False
delete_failures: bool = False
```

**`pass_unsure`**: After `_score_pair()`, if `result.state == MatchState.UNSURE` and `config.pass_unsure`, set `result.passed = True`. Applied in `run()` and the per-pair loop in `run_batch()`.

**`keep_synced`**: Controls whether the synced subtitle copy is preserved. After scoring, if sync occurred and `config.keep_synced`, copy `result.sync.synced_srt_path` to `subtitle.with_stem(subtitle.stem + ".synced")` before deleting the temp. Applied in `run()`, `run_batch()`, and `_score_group_parallel()`.

**`delete_failures`**: After scoring, if `result.state == MatchState.FAIL` and `config.delete_failures`, unlink the subtitle file. Applied in `run()` and per-pair in `run_batch()` / `_score_group_parallel()`.

The `_args_to_config()` function in `cli.py` adds these three fields:
```python
pass_unsure=getattr(args, "pass_unsure", False),
keep_synced=getattr(args, "keep_synced", False),
delete_failures=getattr(args, "delete_failures", False),
```

## PipelineConfig.from_toml()

```python
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
        cache_dir=Path(cfg["cache_dir"]).expanduser() if cfg.get("cache_dir") else None,
        cache_ttl_days=cfg.get("cache_ttl_days"),
        cache_max_mb=cfg.get("cache_max_mb"),
        resync=cfg.get("resync", False),
        pass_unsure=cfg.get("pass_unsure", False),
        keep_synced=cfg.get("keep_synced", False),
        delete_failures=cfg.get("delete_failures", False),
    )
```

`config.py` is unchanged. `_CONFIGURABLE_KEYS` already includes all three new fields plus `"telemetry"` (recently added).

## cli.py simplifications

- `_should_fail()` is removed. Its callers replace it with `not result.passed` (which now incorporates `pass_unsure` via PipelineConfig).
- The `if args.keep_synced` block in `main()` is removed (pipeline handles the copy).
- The `result.sync.synced_srt_path.unlink()` call in `main()` is removed (pipeline handles cleanup).
- The `if args.delete_failures` block in `main()` and `_run_batch()` is removed (pipeline handles deletion).
- A `"Deleted: …"` print is added to the `on_pair_complete` callback in `_run_batch` for CLI visibility when `delete_failures` is set.

## watch.py simplification

Remove from `_score_and_print`:
```python
if result.sync and result.sync.synced_srt_path:
    result.sync.synced_srt_path.unlink(missing_ok=True)
```

Pipeline handles this now based on `config.keep_synced`.

## args.py

`parse_args()` moves verbatim from `cli.py`. `cli.py` replaces the definition with `from submatch.args import parse_args`. `tests/test_args.py` is created with the `parse_args` tests extracted from `test_cli.py`.

## Testing

- `test_pipeline.py`: tests for `pass_unsure`, `keep_synced`, `delete_failures` behaviour; test for `from_toml()` (mocking `config.load_config()`).
- `test_args.py`: `parse_args` tests moved from `test_cli.py`.
- All files: type imports updated from `submatch.output` to `submatch.types`.
- Coverage target: ≥ 95%.

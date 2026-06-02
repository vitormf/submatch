# Pipeline Split & args_to_config Relocation

## Goal

Two structural cleanups: move `_args_to_config` from `cli.py` to `args.py` (fixing a `watch.py → cli.py` dependency), and split `pipeline.py` into `scoring.py` (single-pair engine) and a slimmer `pipeline.py` (orchestration + public API).

## Architecture

```
args.py          → parse_args(), _args_to_config()
scoring.py       → _score_pair(), _audio_driven_transcribe(), helpers
pipeline.py      → PipelineConfig, _get_model, run(), run_batch()

cli.py           → imports _args_to_config from args; _get_model/_resolve_device from pipeline
watch.py         → imports _args_to_config from args; _get_model/_resolve_device from pipeline
__init__.py      → unchanged (run, run_batch, PipelineConfig still from pipeline)
```

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `submatch/args.py` | **Modify** | Add `_args_to_config()` (moved from `cli.py`) |
| `submatch/scoring.py` | **Create** | `_get_embed_model`, `_is_cross_language`, `_determine_state`, `_cache_config`, `_audio_driven_transcribe`, `_score_pair` |
| `submatch/pipeline.py` | **Modify** | Remove scoring internals; keep `PipelineConfig`, `_resolve_device`, `_resolve_workers`, `_get_model`, `_score_group_parallel`, `_apply_postprocessing`, `run`, `run_batch` |
| `submatch/cli.py` | **Modify** | Import `_args_to_config` from `args` instead of defining it |
| `submatch/watch.py` | **Modify** | Import `_args_to_config` from `args` instead of `cli` |
| `tests/test_pipeline.py` | **Modify** | Update patches from `submatch.pipeline._score_pair` → `submatch.scoring._score_pair` where needed |
| `tests/test_scoring.py` | **Create** | Smoke test confirming `_score_pair` is importable from `submatch.scoring` |

## scoring.py

Contains all code that scores a single (video, subtitle) pair:

- `_get_embed_model()` — lazy-load embedding model
- `_is_cross_language(audio_lang, subtitle_lang)` — cross-language detection
- `_determine_state(result)` — PASS / DRIFT / FAIL / UNSURE classification
- `_cache_config(config)` — build cache key dict
- `_audio_driven_transcribe(...)` — Whisper transcription loop (~74 lines)
- `_score_pair(video, subtitle, config, model)` — full per-pair scoring (~187 lines)

`PipelineConfig` is only needed as a type annotation in `_score_pair` and `_cache_config`. Resolved without circular import via `from __future__ import annotations` + `TYPE_CHECKING`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from submatch.pipeline import PipelineConfig
```

All annotations are lazy strings at runtime (`from __future__ import annotations`), so `PipelineConfig` is never actually imported during execution.

## pipeline.py (after split)

Keeps orchestration and the public API (~250 lines):

- `PipelineConfig` dataclass (with `from_toml`)
- `_resolve_device`, `_resolve_workers`, `_get_model` — setup/infrastructure
- `_score_group_parallel` — parallel batch worker (calls `_score_pair` from scoring)
- `_apply_postprocessing` — post-run result mutation
- `run(video, subtitle, config)` — public single-pair entry point
- `run_batch(pairs, config)` — public batch entry point

## args.py (after addition)

`_args_to_config(args: argparse.Namespace) -> PipelineConfig` moves here verbatim from `cli.py`. Both `cli.py` and `watch.py` import it from `args`.

## Testing

- `tests/test_scoring.py` — smoke test: `from submatch.scoring import _score_pair` works; `_determine_state` returns expected states for known inputs
- `tests/test_pipeline.py` — patches of `submatch.pipeline._score_pair` updated to `submatch.scoring._score_pair`; all existing tests otherwise unchanged
- No new behavioural tests needed (pure refactor — no logic changes)

## Constraints

- `submatch/__init__.py` unchanged — `run`, `run_batch`, `PipelineConfig` remain importable from `submatch.pipeline`
- `cli.py` and `watch.py` import paths for `_resolve_device` and `_get_model` unchanged (still from `pipeline`)
- Coverage must stay ≥ 95%

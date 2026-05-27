# Cross-Language Subtitle Matching Design

**Goal:** Give `submatch` a meaningful confidence score when the subtitle is a translation of the audio (different language), rather than returning near-zero from a same-language token comparison.

---

## Behaviour

### Detection

Cross-language mode activates automatically when the audio language (detected by Whisper from the first transcribed segment) differs from the subtitle language (detected from the filename tag, e.g. `.pt.srt`, or from langdetect on the first 50 subtitle lines). Language comparison uses the 2-letter ISO prefix so `pt` and `pt-BR` are treated as the same language.

If either language is unknown (detection failed), the tool falls back to same-language token F1 mode.

### Scoring

**Same-language:** token F1 (existing behaviour, unchanged).

**Cross-language:** multilingual sentence embeddings. Model: `paraphrase-multilingual-MiniLM-L12-v2` from `sentence-transformers`. Both the subtitle segment text and Whisper transcription are embedded into a shared semantic space and compared with cosine similarity. The raw cosine score is normalized to remove the noise floor:

```
normalized = max(0.0, (cosine - 0.15) / 0.85)
```

The baseline of 0.15 represents typical cosine similarity between unrelated multilingual texts. After normalization: cosine 0.44 → 0.34 (marginal pass at default threshold), cosine 0.65 → 0.59 (solid pass), cosine 0.15 or below → 0.0 (definite fail).

The normalized score is returned as `SegmentScore.f1` so `aggregate()` and all downstream pipeline code work unchanged. `SegmentScore.wer` is set to 0.0 (not meaningful cross-language).

### Thresholds

- **`--threshold`** (default 0.35) — applies to same-language pairs and to cross-language pairs when `--cross-threshold` is not set.
- **`--cross-threshold`** (default: None → falls back to `--threshold`) — overrides pass/fail threshold for cross-language pairs only.

The effective threshold used for a pair is stored in `MatchResult.threshold` so output always shows the threshold that was actually applied.

### Model loading

The embedding model is downloaded on first use (~90 MB, cached by sentence-transformers) and held in thread-local storage (`_embed_local = threading.local()`), mirroring the existing Whisper model pattern. In parallel batch mode each worker thread gets its own instance. The model is only loaded when a cross-language pair is actually encountered — same-language runs incur zero overhead.

If `sentence-transformers` is not installed and cross-language is detected, the tool prints a clear install instruction and exits with code 2.

---

## Architecture

### `submatch/embeddings.py` (new)

```
load_embedding_model(model_name) → Any
normalize_cross_score(cosine: float) → float
cross_language_score(subtitle_text, transcription, model) → SegmentScore
```

Pure module, no CLI imports. `sentence_transformers` is imported lazily inside `load_embedding_model` so tests can mock it without the library installed.

### `submatch/output.py`

`MatchResult` gains two optional fields with defaults (backward-compatible):

```python
cross_language: bool = False
subtitle_language: str | None = None
```

`print_human` shows a different content-check header for cross-language results:
```
Content check — cross-language  (en audio → pt subtitle, 8 segments, base model)
```

### `submatch/cli.py`

New helpers:

```python
def _is_cross_language(audio_lang: str | None, subtitle_lang: str | None) -> bool
_embed_local = threading.local()
def _get_embed_model() -> Any  # lazy load, thread-local, ImportError → sys.exit(2)
```

`_score_pair` is refactored into two phases:
1. **Transcription phase** — iterates segments, calls Whisper, collects `(index, seg, trans_text)` tuples. After loop, `audio_lang` is known.
2. **Scoring phase** — detects cross-language, loads embed model if needed, scores all segments with either `token_f1` or `embeddings.cross_language_score`.

This avoids double-looping: Whisper runs once, scoring is a cheap second pass.

New flag: `--cross-threshold FLOAT` (dest: `cross_threshold`, default: `None`).

### `pyproject.toml`

Add `sentence-transformers>=2.2` to `dependencies`.

---

## Testing

- `tests/test_embeddings.py` — unit tests for all three functions; `sentence_transformers` mocked via `patch.dict(sys.modules, ...)`
- `tests/test_output.py` — new fields serialise to JSON; `print_human` shows cross-language header
- `tests/test_cli.py` — `_is_cross_language` unit tests; `--cross-threshold` parsing; cross-language scoring path calls `embeddings.cross_language_score`; same-language path does not load embed model; `_get_embed_model` missing-import error

# Cross-Language Subtitle Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a subtitle is in a different language than the audio (e.g. English movie + Portuguese translation), use multilingual sentence embeddings to score semantic similarity instead of token F1, so translated subtitles get a meaningful confidence score.

**Architecture:** A new `submatch/embeddings.py` module wraps `sentence-transformers` to compute normalized cosine similarity between multilingual text pairs. `_score_pair` in `cli.py` is refactored into two phases: transcription (collect all results first) then scoring (choose token F1 or embedding cosine based on language detection). `MatchResult` gains `cross_language` and `subtitle_language` fields. A `--cross-threshold` flag allows separate tuning of the cross-language pass/fail cutoff.

**Tech Stack:** `sentence-transformers>=2.2` (model: `paraphrase-multilingual-MiniLM-L12-v2`), `numpy` (already present via openai-whisper), existing `threading.local` pattern for per-thread model caching.

---

## File map

| File | Change |
|------|--------|
| `submatch/embeddings.py` | Create: `load_embedding_model`, `normalize_cross_score`, `cross_language_score` |
| `submatch/output.py` | Add `cross_language: bool = False` and `subtitle_language: str \| None = None` to `MatchResult`; update `print_human` header |
| `submatch/cli.py` | Add `embeddings` import; add `_is_cross_language`, `_embed_local`, `_get_embed_model`; refactor `_score_pair` into transcription + scoring phases; add `--cross-threshold` to `parse_args` |
| `pyproject.toml` | Add `sentence-transformers>=2.2` to dependencies |
| `README.md` | Add `--cross-threshold` to options table; add cross-language section |
| `tests/test_embeddings.py` | Create: tests for all functions in `embeddings.py` |
| `tests/test_output.py` | Add tests for new `MatchResult` fields and cross-language header in `print_human` |
| `tests/test_cli.py` | Add tests for `_is_cross_language`, `--cross-threshold` parsing, cross-language scoring path, `_get_embed_model` missing-import error; update `test_parse_args_defaults` and `test_parse_args_all_flags` |

---

## Task 1: `submatch/embeddings.py`

**Files:**
- Create: `submatch/embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_embeddings.py`:

```python
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def test_normalize_cross_score_at_baseline():
    from submatch.embeddings import normalize_cross_score
    assert normalize_cross_score(0.15) == pytest.approx(0.0)


def test_normalize_cross_score_at_one():
    from submatch.embeddings import normalize_cross_score
    assert normalize_cross_score(1.0) == pytest.approx(1.0)


def test_normalize_cross_score_below_baseline_clamps_to_zero():
    from submatch.embeddings import normalize_cross_score
    assert normalize_cross_score(0.0) == 0.0
    assert normalize_cross_score(-0.5) == 0.0


def test_normalize_cross_score_midpoint():
    from submatch.embeddings import normalize_cross_score
    # (0.575 - 0.15) / 0.85 == 0.5
    assert normalize_cross_score(0.575) == pytest.approx(0.5)


def test_load_embedding_model_calls_sentence_transformer():
    mock_st = MagicMock()
    mock_model = MagicMock()
    mock_st.SentenceTransformer.return_value = mock_model
    with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
        from submatch import embeddings
        result = embeddings.load_embedding_model()
    mock_st.SentenceTransformer.assert_called_once_with(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    assert result is mock_model


def test_cross_language_score_identical_vectors():
    from submatch.embeddings import cross_language_score
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0]])
    score = cross_language_score("hello", "olá", mock_model)
    # cosine = 1.0 → normalized = (1.0 - 0.15) / 0.85 = 1.0
    assert score.f1 == pytest.approx(1.0)
    assert score.wer == 0.0
    assert score.subtitle_tokens == 1


def test_cross_language_score_orthogonal_vectors():
    from submatch.embeddings import cross_language_score
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
    score = cross_language_score("hello", "xyz", mock_model)
    # cosine = 0.0 → normalized = max(0, (0.0 - 0.15) / 0.85) = 0.0
    assert score.f1 == 0.0
    assert score.wer == 0.0


def test_cross_language_score_both_empty():
    from submatch.embeddings import cross_language_score
    mock_model = MagicMock()
    score = cross_language_score("", "", mock_model)
    assert score.f1 == 1.0
    mock_model.encode.assert_not_called()


def test_cross_language_score_subtitle_empty():
    from submatch.embeddings import cross_language_score
    mock_model = MagicMock()
    score = cross_language_score("", "hello world", mock_model)
    assert score.f1 == 0.0
    mock_model.encode.assert_not_called()


def test_cross_language_score_transcription_empty():
    from submatch.embeddings import cross_language_score
    mock_model = MagicMock()
    score = cross_language_score("olá mundo", "", mock_model)
    assert score.f1 == 0.0
    mock_model.encode.assert_not_called()


def test_cross_language_score_subtitle_token_count():
    from submatch.embeddings import cross_language_score
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0]])
    score = cross_language_score("três palavras aqui", "three words here", mock_model)
    assert score.subtitle_tokens == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/vitor/resilio/Dev/submatch && pytest tests/test_embeddings.py -v
```
Expected: `ModuleNotFoundError` — `submatch.embeddings` does not exist yet.

- [ ] **Step 3: Create `submatch/embeddings.py`**

```python
from __future__ import annotations
from typing import Any
import numpy as np

from submatch.compare import SegmentScore

_BASELINE = 0.15
_SCALE = 1.0 - _BASELINE


def load_embedding_model(
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> Any:
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def normalize_cross_score(cosine: float) -> float:
    return max(0.0, (cosine - _BASELINE) / _SCALE)


def cross_language_score(
    subtitle_text: str,
    transcription: str,
    model: Any,
) -> SegmentScore:
    subtitle_tokens = len(subtitle_text.split())
    if not subtitle_text.strip() and not transcription.strip():
        return SegmentScore(f1=1.0, wer=0.0, subtitle_tokens=subtitle_tokens)
    if not subtitle_text.strip() or not transcription.strip():
        return SegmentScore(f1=0.0, wer=0.0, subtitle_tokens=subtitle_tokens)
    vecs = model.encode([subtitle_text, transcription])
    a, b = vecs[0], vecs[1]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / denom) if denom > 0.0 else 0.0
    return SegmentScore(
        f1=normalize_cross_score(cosine),
        wer=0.0,
        subtitle_tokens=subtitle_tokens,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/vitor/resilio/Dev/submatch && pytest tests/test_embeddings.py -v
```
Expected: all 11 tests pass.

- [ ] **Step 5: Run full suite**

```bash
cd /Users/vitor/resilio/Dev/submatch && pytest --ignore=tests/integration -q
```
Expected: all tests pass, coverage ≥ 95%.

- [ ] **Step 6: Commit**

```bash
cd /Users/vitor/resilio/Dev/submatch && git add submatch/embeddings.py tests/test_embeddings.py && git commit -m "feat: add embeddings module for cross-language scoring"
```

---

## Task 2: `MatchResult` fields + `output.py`

**Files:**
- Modify: `submatch/output.py`
- Modify: `tests/test_output.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_output.py`:

```python
# ── cross-language fields ─────────────────────────────────────────────────────

def test_format_json_cross_language_fields():
    result = _make_result()
    result.cross_language = True
    result.subtitle_language = "pt"
    data = json.loads(format_json(result))
    assert data["cross_language"] is True
    assert data["subtitle_language"] == "pt"


def test_format_json_cross_language_defaults_false():
    data = json.loads(format_json(_make_result()))
    assert data["cross_language"] is False
    assert data["subtitle_language"] is None


def test_print_human_cross_language_shows_header(capsys):
    result = _make_result()
    result.cross_language = True
    result.subtitle_language = "pt"
    result.language = LanguageResult(
        audio="en", subtitle_detected="pt", subtitle_filename="pt",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    print_human(result)
    out = capsys.readouterr().out
    assert "cross-language" in out
    assert "en" in out
    assert "pt" in out


def test_print_human_same_language_no_cross_header(capsys):
    print_human(_make_result())
    out = capsys.readouterr().out
    assert "cross-language" not in out
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/vitor/resilio/Dev/submatch && pytest tests/test_output.py -v -k "cross"
```
Expected: `test_format_json_cross_language_fields` fails (`KeyError: cross_language`), `test_print_human_cross_language_shows_header` fails.

- [ ] **Step 3: Add fields to `MatchResult` in `submatch/output.py`**

Replace the `MatchResult` dataclass (lines 36–43):

```python
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
```

- [ ] **Step 4: Update `print_human` content-check header in `submatch/output.py`**

Replace this line in `print_human`:

```python
    print(f"{_BOLD}Content check ({len(result.segments)} segments, {result.model} model){_RESET}")
```

with:

```python
    if result.cross_language:
        audio_lbl = result.language.audio or "?"
        sub_lbl = result.subtitle_language or "?"
        print(
            f"{_BOLD}Content check — cross-language{_RESET}"
            f"  ({audio_lbl} audio → {sub_lbl} subtitle,"
            f" {len(result.segments)} segments, {result.model} model)"
        )
    else:
        print(
            f"{_BOLD}Content check"
            f" ({len(result.segments)} segments, {result.model} model){_RESET}"
        )
```

- [ ] **Step 5: Run all output tests**

```bash
cd /Users/vitor/resilio/Dev/submatch && pytest tests/test_output.py -v
```
Expected: all tests pass.

- [ ] **Step 6: Run full suite**

```bash
cd /Users/vitor/resilio/Dev/submatch && pytest --ignore=tests/integration -q
```
Expected: all tests pass, coverage ≥ 95%.

- [ ] **Step 7: Commit**

```bash
cd /Users/vitor/resilio/Dev/submatch && git add submatch/output.py tests/test_output.py && git commit -m "feat: add cross_language and subtitle_language fields to MatchResult"
```

---

## Task 3: `cli.py` — detection, scoring, and `--cross-threshold`

**Files:**
- Modify: `submatch/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_cli.py`, update `test_parse_args_defaults` — add after the `assert args.workers is None` line:

```python
    assert args.cross_threshold is None
    assert args.delete_failures is False
```

In `test_parse_args_all_flags`, add `"--cross-threshold", "0.5"` to the argv list and at the end:

```python
    assert args.cross_threshold == pytest.approx(0.5)
```

Append new tests at the bottom of `tests/test_cli.py`:

```python
# ── cross-language detection ──────────────────────────────────────────────────

def test_is_cross_language_different():
    assert cli._is_cross_language("en", "pt") is True


def test_is_cross_language_same():
    assert cli._is_cross_language("en", "en") is False


def test_is_cross_language_prefix_match():
    # pt and pt-BR share the same prefix — not cross-language
    assert cli._is_cross_language("pt", "pt-BR") is False
    assert cli._is_cross_language("pt-BR", "pt") is False


def test_is_cross_language_none_audio():
    assert cli._is_cross_language(None, "pt") is False


def test_is_cross_language_none_subtitle():
    assert cli._is_cross_language("en", None) is False


def test_is_cross_language_both_none():
    assert cli._is_cross_language(None, None) is False


def test_parse_args_cross_threshold_default(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s)]):
        args = cli.parse_args()
    assert args.cross_threshold is None


def test_parse_args_cross_threshold_explicit(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s), "--cross-threshold", "0.5"]):
        args = cli.parse_args()
    assert args.cross_threshold == pytest.approx(0.5)


def test_get_embed_model_missing_import(capsys):
    """_get_embed_model exits with code 2 when sentence_transformers not installed."""
    # Clear any cached model first
    import threading
    cli._embed_local.__dict__.clear() if hasattr(cli._embed_local, '__dict__') else None
    # Ensure thread-local has no 'model' attribute
    if hasattr(cli._embed_local, 'model'):
        del cli._embed_local.model

    with patch.dict(sys.modules, {"sentence_transformers": None}), \
         pytest.raises(SystemExit) as exc:
        cli._get_embed_model()
    assert exc.value.code == 2
    assert "sentence-transformers" in capsys.readouterr().err


def test_score_pair_cross_language_uses_embeddings(tmp_path):
    """Audio='en', subtitle detected as 'pt': embeddings scoring is used."""
    video = tmp_path / "movie.mkv"
    video.touch()
    sub = tmp_path / "movie.pt.srt"
    sub.write_text(SAMPLE_SRT)

    subs_parsed = [Subtitle(1, 1_000, 3_500, "Olá mundo")]
    segs = [Segment(60_000, 90_000, "Olá mundo", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="pt", subtitle_filename="pt",
        video_metadata=None, expected=None, mismatch=True,
        mismatch_details=["audio=en but subtitle text detected as pt"],
    )
    mock_embed_score = MagicMock(f1=0.72, wer=0.0)
    mock_cross_fn = MagicMock(return_value=mock_embed_score)

    with patch("sys.argv", ["submatch", str(video), str(sub),
                            "--no-sync", "--threshold", "0.5"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.cli.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.cli.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.cli.subtitle.parse", return_value=subs_parsed), \
         patch("submatch.cli.sampler.select_segments", return_value=segs), \
         patch("submatch.cli.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.cli.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.cli.language.detect_from_text", return_value="pt"), \
         patch("submatch.cli.language.detect_from_filename", return_value="pt"), \
         patch("submatch.cli.language.detect_from_video", return_value=None), \
         patch("submatch.cli.language.build_result", return_value=lang), \
         patch("submatch.cli._get_embed_model", return_value=MagicMock()), \
         patch("submatch.embeddings.cross_language_score", mock_cross_fn), \
         pytest.raises(SystemExit):
        cli.main()

    mock_cross_fn.assert_called_once()


def test_score_pair_same_language_skips_embeddings(tmp_path):
    """Audio='en', subtitle 'en': token_f1 is used; embeddings not loaded."""
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--threshold", "0.01"])
    mock_get_embed = MagicMock()

    with patch("submatch.cli._get_embed_model", mock_get_embed):
        [c.__enter__() for c in ctx]
        try:
            with pytest.raises(SystemExit):
                cli.main()
        finally:
            for c in reversed(ctx):
                c.__exit__(None, None, None)

    mock_get_embed.assert_not_called()


def test_cross_threshold_used_for_cross_language_pair(tmp_path, capsys):
    """--cross-threshold 0.9 causes a pair with score 0.72 to fail."""
    video = tmp_path / "movie.mkv"
    video.touch()
    sub = tmp_path / "movie.pt.srt"
    sub.write_text(SAMPLE_SRT)

    subs_parsed = [Subtitle(1, 1_000, 3_500, "Olá mundo")]
    segs = [Segment(60_000, 90_000, "Olá mundo", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="pt", subtitle_filename="pt",
        video_metadata=None, expected=None, mismatch=True,
        mismatch_details=["audio=en but subtitle text detected as pt"],
    )
    # score 0.72 < cross-threshold 0.9 → should FAIL
    mock_embed_score = MagicMock(f1=0.72, wer=0.0)

    with patch("sys.argv", ["submatch", str(video), str(sub),
                            "--no-sync", "--threshold", "0.5",
                            "--cross-threshold", "0.9"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.cli.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.cli.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.cli.subtitle.parse", return_value=subs_parsed), \
         patch("submatch.cli.sampler.select_segments", return_value=segs), \
         patch("submatch.cli.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.cli.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.cli.language.detect_from_text", return_value="pt"), \
         patch("submatch.cli.language.detect_from_filename", return_value="pt"), \
         patch("submatch.cli.language.detect_from_video", return_value=None), \
         patch("submatch.cli.language.build_result", return_value=lang), \
         patch("submatch.cli._get_embed_model", return_value=MagicMock()), \
         patch("submatch.embeddings.cross_language_score",
               return_value=mock_embed_score), \
         pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1  # failed
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/vitor/resilio/Dev/submatch && pytest tests/test_cli.py -v -k "cross or is_cross or embed"
```
Expected: `AttributeError: module 'submatch.cli' has no attribute '_is_cross_language'` and flag-parsing failures.

- [ ] **Step 3: Add `embeddings` to imports in `submatch/cli.py`**

Replace the existing import line:

```python
from submatch import audio, compare, language, output, sampler, subtitle, sync, transcribe
```

with:

```python
from submatch import audio, compare, embeddings, language, output, sampler, subtitle, sync, transcribe
```

- [ ] **Step 4: Add `_is_cross_language`, `_embed_local`, and `_get_embed_model` to `submatch/cli.py`**

After the `_get_model` function (around line 109), add:

```python
def _is_cross_language(audio_lang: str | None, subtitle_lang: str | None) -> bool:
    if not audio_lang or not subtitle_lang:
        return False
    return audio_lang.split("-")[0].lower() != subtitle_lang.split("-")[0].lower()


_embed_local = threading.local()


def _get_embed_model():
    if not hasattr(_embed_local, "model"):
        try:
            _embed_local.model = embeddings.load_embedding_model()
        except ImportError:
            print(
                "Error: sentence-transformers not installed. "
                "Required for cross-language subtitle matching. "
                "Install with: pip install sentence-transformers",
                file=sys.stderr,
            )
            sys.exit(2)
    return _embed_local.model
```

- [ ] **Step 5: Add `--cross-threshold` to `parse_args` in `submatch/cli.py`**

After the `--delete-failures` argument, add:

```python
    parser.add_argument(
        "--cross-threshold", type=float, default=None, dest="cross_threshold",
        help="pass/fail threshold for cross-language pairs (default: same as --threshold)",
    )
```

- [ ] **Step 6: Refactor `_score_pair` in `submatch/cli.py`**

Replace the entire `_score_pair` function with:

```python
def _score_pair(
    video: Path,
    subtitle_path: Path,
    args: argparse.Namespace,
    model,
    show_progress: bool = True,
    bar=None,
) -> output.MatchResult:
    subtitles = subtitle.parse(subtitle_path)
    subtitle_sample = " ".join(s.text for s in subtitles[:50])
    subtitle_lang = (language.detect_from_filename(subtitle_path) or
                     language.detect_from_text(subtitle_sample))

    sync_result = None
    _sync_tmp: Path | None = None
    try:
        if not args.no_sync:
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
                _sync_tmp = Path(tmp.name)
                tmp.close()
                if bar is not None:
                    bar.set_description(f"sync  [{video.name} / {subtitle_path.name}]")
                sync_result = sync.sync_subtitle(video, subtitle_path, _sync_tmp)
                subtitles = subtitle.parse(sync_result.synced_srt_path)
            except RuntimeError as exc:
                print(f"Warning: ffsubsync failed ({exc}), proceeding without sync",
                      file=sys.stderr)

        duration_ms = audio.get_duration_ms(video)
        segments = sampler.select_segments(subtitles, duration_ms, n=args.segments)

        # Phase 1: transcribe all segments
        transcription_pairs: list[tuple[int, sampler.Segment, str]] = []
        audio_lang: str | None = None

        for i, seg in enumerate(segments):
            if bar is not None:
                bar.set_description(
                    f"[{i + 1}/{len(segments)}] {video.name} / {subtitle_path.name}"
                )
            elif show_progress and not args.json:
                print(f"  Transcribing segment {i + 1}/{len(segments)}...", end="\r")
            try:
                wav_path = audio.extract_segment(video, seg.start_ms, 30_000)
                try:
                    trans = transcribe.transcribe_segment(model, wav_path)
                    if i == 0:
                        audio_lang = trans.language
                    transcription_pairs.append((i + 1, seg, trans.text))
                finally:
                    wav_path.unlink(missing_ok=True)
            except Exception as exc:
                print(f"\nWarning: segment {i + 1} failed: {exc}", file=sys.stderr)

        if show_progress and not args.json:
            print()

        # Phase 2: determine scoring mode
        cross_lang = _is_cross_language(audio_lang, subtitle_lang)
        embed_model = _get_embed_model() if cross_lang else None

        # Phase 3: score segments
        segment_results: list[output.SegmentResult] = []
        for idx, seg, trans_text in transcription_pairs:
            if cross_lang:
                score = embeddings.cross_language_score(seg.subtitle_text, trans_text, embed_model)
            else:
                score = compare.token_f1(seg.subtitle_text, trans_text)
            segment_results.append(output.SegmentResult(
                index=idx,
                start_ms=seg.start_ms,
                score=score.f1,
                wer=score.wer,
                subtitle_text=seg.subtitle_text,
                transcription=trans_text,
            ))

        lang_result = language.build_result(
            audio=audio_lang,
            subtitle_detected=language.detect_from_text(subtitle_sample),
            subtitle_filename=language.detect_from_filename(subtitle_path),
            video_meta=language.detect_from_video(video),
            expected=args.language,
        )

        seg_scores = [
            compare.SegmentScore(
                f1=sr.score,
                wer=sr.wer,
                subtitle_tokens=len(sr.subtitle_text.split()),
            )
            for sr in segment_results
        ]
        confidence = compare.aggregate(seg_scores)

        effective_threshold = (
            args.cross_threshold
            if (cross_lang and args.cross_threshold is not None)
            else args.threshold
        )

        return output.MatchResult(
            confidence=confidence,
            passed=confidence >= effective_threshold,
            threshold=effective_threshold,
            language=lang_result,
            sync=sync_result,
            segments=segment_results,
            model=args.model,
            cross_language=cross_lang,
            subtitle_language=subtitle_lang,
        )
    finally:
        if _sync_tmp is not None:
            _sync_tmp.unlink(missing_ok=True)
```

- [ ] **Step 7: Run all cli tests**

```bash
cd /Users/vitor/resilio/Dev/submatch && pytest tests/test_cli.py -v
```
Expected: all tests pass.

- [ ] **Step 8: Run full suite**

```bash
cd /Users/vitor/resilio/Dev/submatch && pytest --ignore=tests/integration -q
```
Expected: all tests pass, coverage ≥ 95%.

- [ ] **Step 9: Commit**

```bash
cd /Users/vitor/resilio/Dev/submatch && git add submatch/cli.py tests/test_cli.py && git commit -m "feat: cross-language scoring via multilingual embeddings"
```

---

## Task 4: `pyproject.toml` + `README.md`

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`

- [ ] **Step 1: Add `sentence-transformers` to dependencies in `pyproject.toml`**

In the `dependencies` list, add after `"tqdm>=4.64"`:

```toml
    "sentence-transformers>=2.2",
```

So the full list becomes:

```toml
dependencies = [
    "openai-whisper>=20231117",
    "ffsubsync>=0.4.22",
    "langdetect>=1.0.9",
    "pysubs2>=1.6",
    "tqdm>=4.64",
    "sentence-transformers>=2.2",
]
```

- [ ] **Step 2: Update `README.md`**

Add `--cross-threshold` to the options table, after `--threshold`:

```markdown
| `--cross-threshold` | same as `--threshold` | Pass/fail threshold for cross-language pairs |
```

Add a new section after the options table (or after the batch mode section):

```markdown
### Cross-language matching

When the subtitle language differs from the audio language (e.g. English audio with Portuguese subtitles), `submatch` automatically switches from token F1 scoring to multilingual semantic similarity using `paraphrase-multilingual-MiniLM-L12-v2`. The score is normalized so the same `--threshold` applies to both same-language and cross-language pairs.

Use `--cross-threshold` to tune the pass/fail cutoff for translated subtitles independently:

```bash
submatch movie.mkv movie.pt.srt --cross-threshold 0.5
```

The model is downloaded on first use (~90 MB) and cached by sentence-transformers.
```

- [ ] **Step 3: Commit**

```bash
cd /Users/vitor/resilio/Dev/submatch && git add pyproject.toml README.md && git commit -m "feat: add sentence-transformers dependency and document cross-language matching"
```

---

## Self-review

**Spec coverage:**
- ✅ Multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) — Task 1
- ✅ Normalized cosine score `max(0, (cosine - 0.15) / 0.85)` — Task 1
- ✅ Cross-language detection (audio lang ≠ subtitle lang by prefix) — Task 3
- ✅ Subtitle language from filename or langdetect — Task 3
- ✅ Audio language from Whisper first segment — Task 3 (existing, used)
- ✅ Thread-local embedding model (`_embed_local`) — Task 3
- ✅ `--cross-threshold` flag (default: falls back to `--threshold`) — Tasks 3 + 3
- ✅ `MatchResult.cross_language` and `MatchResult.subtitle_language` — Task 2
- ✅ Human output cross-language header — Task 2
- ✅ JSON output includes new fields (via `dataclasses.asdict`) — Task 2
- ✅ Graceful ImportError message for missing sentence-transformers — Task 3
- ✅ `sentence-transformers>=2.2` dependency — Task 4
- ✅ README update — Task 4

**Type consistency:**
- `cross_language_score` returns `SegmentScore` — same type as `token_f1`. Used identically in Phase 3. ✅
- `_is_cross_language(str|None, str|None) -> bool` — used in `_score_pair` as `cross_lang: bool`. ✅
- `MatchResult.cross_language: bool = False` — default preserves all existing tests. ✅
- `MatchResult.subtitle_language: str | None = None` — default preserves all existing tests. ✅
- `effective_threshold` assigned from `args.cross_threshold` (float) or `args.threshold` (float) — both float, passed to `MatchResult.threshold`. ✅

**Placeholder scan:** None found.

**Backward compatibility:** All new `MatchResult` fields have defaults. All existing tests that create `MatchResult` directly will continue to work without modification.

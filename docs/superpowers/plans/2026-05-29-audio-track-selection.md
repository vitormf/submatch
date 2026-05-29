# Audio Track Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--audio-track` flag that lets users pick which audio stream to transcribe, supporting both integer index and language preference lists.

**Architecture:** New functions `list_audio_tracks` and `resolve_audio_track` in `audio.py` handle track discovery and selection. `extract_segment` and `sync_subtitle` gain an `audio_track` parameter. `_score_pair` in `cli.py` resolves the track once per video (caching the result in `_VideoCache`) and threads it through to both ffmpeg and ffs. `MatchResult` and `print_human` in `output.py` expose the selected track.

**Tech Stack:** Python 3.10+, ffprobe (track listing), ffmpeg (`-map 0:a:N`), ffsubsync (`--reference-stream a:N`), pytest + unittest.mock.

---

## File Map

| File | Change |
|---|---|
| `submatch/audio.py` | Add `_ISO_639_1_TO_2`, `_lang_match`, `list_audio_tracks`, `resolve_audio_track`; update `extract_segment` |
| `submatch/sync.py` | Update `sync_subtitle` signature |
| `submatch/output.py` | Update `MatchResult` dataclass; update `print_human` |
| `submatch/cli.py` | Update `_VideoCache`; update `_score_pair`; update `parse_args` |
| `tests/test_audio.py` | New tests for `list_audio_tracks`, `resolve_audio_track`, `extract_segment` audio_track |
| `tests/test_sync.py` | New tests for `sync_subtitle` audio_track |
| `tests/test_output.py` | New tests for `MatchResult` fields and `print_human` track line |
| `tests/test_cli.py` | Update `test_parse_args_defaults` and `test_parse_args_all_flags`; new tests for audio_track plumbing |

---

### Task 1: `list_audio_tracks`, `resolve_audio_track`, and ISO 639 helpers in `audio.py`

**Files:**
- Modify: `submatch/audio.py`
- Test: `tests/test_audio.py`

- [ ] **Step 1: Write failing tests for `list_audio_tracks`**

Add to `tests/test_audio.py`:

```python
import json
from unittest.mock import patch, MagicMock
from submatch.audio import list_audio_tracks, resolve_audio_track


def _ffprobe_audio_response(streams: list[dict]) -> str:
    return json.dumps({"streams": streams})


def test_list_audio_tracks_two_streams(tmp_path):
    video = tmp_path / "v.mkv"
    video.touch()
    response = _ffprobe_audio_response([
        {"tags": {"language": "eng"}},
        {"tags": {"language": "jpn"}},
    ])
    with patch("submatch.audio.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=response, returncode=0)
        tracks = list_audio_tracks(video)
    assert tracks == [{"index": 0, "lang": "eng"}, {"index": 1, "lang": "jpn"}]


def test_list_audio_tracks_no_language_tag(tmp_path):
    video = tmp_path / "v.mkv"
    video.touch()
    response = _ffprobe_audio_response([{"codec_type": "audio"}])
    with patch("submatch.audio.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=response, returncode=0)
        tracks = list_audio_tracks(video)
    assert tracks == [{"index": 0, "lang": None}]


def test_list_audio_tracks_empty(tmp_path):
    video = tmp_path / "v.mkv"
    video.touch()
    response = _ffprobe_audio_response([])
    with patch("submatch.audio.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=response, returncode=0)
        tracks = list_audio_tracks(video)
    assert tracks == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_audio.py::test_list_audio_tracks_two_streams tests/test_audio.py::test_list_audio_tracks_no_language_tag tests/test_audio.py::test_list_audio_tracks_empty -v
```

Expected: FAIL with `ImportError: cannot import name 'list_audio_tracks'`

- [ ] **Step 3: Write failing tests for `resolve_audio_track`**

Add to `tests/test_audio.py` (below the `list_audio_tracks` tests):

```python
def _two_track_video(tmp_path):
    """Returns (video_path, mock_list_audio_tracks context)."""
    video = tmp_path / "v.mkv"
    video.touch()
    tracks = [{"index": 0, "lang": "eng"}, {"index": 1, "lang": "jpn"}]
    return video, tracks


def test_resolve_audio_track_integer_valid(tmp_path):
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "1")
    assert idx == 1
    assert lang == "jpn"


def test_resolve_audio_track_integer_zero(tmp_path):
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "0")
    assert idx == 0
    assert lang == "eng"


def test_resolve_audio_track_integer_out_of_range(tmp_path):
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        with pytest.raises(SystemExit) as exc:
            resolve_audio_track(video, "5")
    assert exc.value.code == 2


def test_resolve_audio_track_language_exact_match(tmp_path):
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "jpn")
    assert idx == 1
    assert lang == "jpn"


def test_resolve_audio_track_language_iso_639_1_to_2(tmp_path):
    """'jp' should match 'jpn' via the ISO 639-1 mapping."""
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "jp")
    assert idx == 1
    assert lang == "jpn"


def test_resolve_audio_track_language_en_matches_eng(tmp_path):
    """'en' should match 'eng'."""
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "en")
    assert idx == 0
    assert lang == "eng"


def test_resolve_audio_track_preference_list_first_match(tmp_path):
    """First preference that matches wins."""
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "jp,en")
    assert idx == 1  # jp matched first


def test_resolve_audio_track_preference_list_fallback_to_second(tmp_path):
    """Falls through to second preference when first not present."""
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "fr,en")
    assert idx == 0  # fr not found, en matched


def test_resolve_audio_track_no_match_falls_back_to_track_0(tmp_path, capsys):
    """No preference matches → track 0 with a warning."""
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "fr,de")
    assert idx == 0
    assert lang == "eng"
    captured = capsys.readouterr()
    assert "Warning" in captured.err


def test_resolve_audio_track_case_insensitive(tmp_path):
    """'JP' should match 'jpn'."""
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "JP")
    assert idx == 1


def test_resolve_audio_track_ffprobe_failure_falls_back(tmp_path, capsys):
    """ffprobe failure returns (0, None) with a warning."""
    video = tmp_path / "v.mkv"
    video.touch()
    with patch("submatch.audio.list_audio_tracks", side_effect=RuntimeError("ffprobe fail")):
        idx, lang = resolve_audio_track(video, "jp")
    assert idx == 0
    assert lang is None
    captured = capsys.readouterr()
    assert "Warning" in captured.err
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
pytest tests/test_audio.py -k "resolve_audio_track" -v
```

Expected: FAIL with `ImportError: cannot import name 'resolve_audio_track'`

- [ ] **Step 5: Implement `_ISO_639_1_TO_2`, `_lang_match`, `list_audio_tracks`, and `resolve_audio_track` in `audio.py`**

Replace the full contents of `submatch/audio.py`:

```python
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_ISO_639_1_TO_2: dict[str, str] = {
    "ar": "ara", "bg": "bul", "ca": "cat", "cs": "ces", "cy": "wel",
    "da": "dan", "de": "deu", "el": "ell", "en": "eng", "es": "spa",
    "et": "est", "eu": "eus", "fa": "fas", "fi": "fin", "fr": "fra",
    "gl": "glg", "he": "heb", "hi": "hin", "hr": "hrv", "hu": "hun",
    "hy": "hye", "id": "ind", "is": "isl", "it": "ita", "ja": "jpn",
    "jp": "jpn",
    "ka": "kat", "kk": "kaz", "kn": "kan", "ko": "kor", "lt": "lit",
    "lv": "lav", "mk": "mkd", "ml": "mal", "mn": "mon", "mr": "mar",
    "ms": "msa", "ne": "nep", "nl": "nld", "no": "nor", "pa": "pan",
    "pl": "pol", "pt": "por", "ro": "ron", "ru": "rus", "sk": "slk",
    "sl": "slv", "sq": "sqi", "sr": "srp", "sv": "swe", "sw": "swa",
    "ta": "tam", "te": "tel", "th": "tha", "tr": "tur", "uk": "ukr",
    "ur": "urd", "uz": "uzb", "vi": "vie", "zh": "zho",
}


def _lang_match(pref: str, track_lang: str | None) -> bool:
    """Return True if pref matches track_lang, handling ISO 639-1 ↔ 639-2 equivalence."""
    if not track_lang:
        return False
    p = pref.lower()
    t = track_lang.lower()
    return _ISO_639_1_TO_2.get(p, p) == _ISO_639_1_TO_2.get(t, t)


def get_duration_ms(video_path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    duration_s = float(json.loads(result.stdout)["format"]["duration"])
    return int(duration_s * 1_000)


def has_audio_track(video_path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "a", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return len(json.loads(result.stdout).get("streams", [])) > 0


def list_audio_tracks(video_path: Path) -> list[dict]:
    """Return one dict per audio stream: {"index": int, "lang": str | None}."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "a", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    return [
        {"index": i, "lang": s.get("tags", {}).get("language")}
        for i, s in enumerate(streams)
    ]


def resolve_audio_track(video_path: Path, spec: str) -> tuple[int, str | None]:
    """Parse spec and return (track_index, track_lang).

    spec is either an integer string ("1") or a comma-separated language preference
    list ("jp,en,pt"). Falls back to track 0 with a warning if no preference matches.
    Exits 2 if an integer index is out of range.
    """
    try:
        tracks = list_audio_tracks(video_path)
    except Exception:
        print(
            f"Warning: could not list audio tracks for {video_path.name}, using track 0",
            file=sys.stderr,
        )
        return 0, None

    spec = spec.strip()

    # Integer path
    try:
        idx = int(spec)
        if idx < 0 or idx >= len(tracks):
            print(
                f"Error: audio track {idx} does not exist in {video_path.name} "
                f"({len(tracks)} track(s) available)",
                file=sys.stderr,
            )
            sys.exit(2)
        return idx, tracks[idx]["lang"]
    except ValueError:
        pass

    # Language preference list
    for pref in [p.strip() for p in spec.split(",") if p.strip()]:
        for track in tracks:
            if _lang_match(pref, track["lang"]):
                return track["index"], track["lang"]

    track_desc = ", ".join(f"a:{t['index']} ({t['lang'] or '?'})" for t in tracks) if tracks else "none"
    print(
        f"Warning: no audio track matches '{spec}' in {video_path.name} "
        f"(available: {track_desc}), using track 0",
        file=sys.stderr,
    )
    return 0, tracks[0]["lang"] if tracks else None


def extract_segment(video_path: Path, start_ms: int, duration_ms: int, audio_track: int = 0) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = Path(tmp.name)
    tmp.close()
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_ms / 1_000),
        "-i", str(video_path),
        "-t", str(duration_ms / 1_000),
        "-ar", "16000",
        "-ac", "1",
        "-vn",
    ]
    if audio_track > 0:
        cmd += ["-map", f"0:a:{audio_track}"]
    cmd.append(str(out_path))
    subprocess.run(cmd, capture_output=True, check=True)
    return out_path
```

- [ ] **Step 6: Run all audio tests to verify they pass**

```bash
pytest tests/test_audio.py -v
```

Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add submatch/audio.py tests/test_audio.py
git commit -m "feat: add list_audio_tracks and resolve_audio_track to audio.py"
```

---

### Task 2: `extract_segment` audio_track tests

**Files:**
- Modify: `tests/test_audio.py`

The `extract_segment` function was already updated in Task 1. This task adds the tests that verify the `-map` flag behavior.

- [ ] **Step 1: Write failing tests for `extract_segment` audio_track**

Add to `tests/test_audio.py`:

```python
def test_extract_segment_default_track_no_map_flag(tmp_path):
    """audio_track=0 must NOT add a -map flag."""
    video = tmp_path / "v.mp4"
    video.touch()
    captured_cmds = []

    def fake_run(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        # Write a stub WAV so the function returns without error
        return MagicMock(returncode=0)

    with patch("submatch.audio.subprocess.run", side_effect=fake_run):
        import submatch.audio as _audio
        # Patch tempfile to return a known path
        import tempfile as _tf
        with patch.object(_tf, "NamedTemporaryFile") as mock_ntf:
            mock_ntf.return_value.name = str(tmp_path / "out.wav")
            mock_ntf.return_value.close = lambda: None
            (tmp_path / "out.wav").write_bytes(b"")
            _audio.extract_segment(video, 0, 3_000, audio_track=0)

    assert captured_cmds
    cmd = captured_cmds[0]
    assert "-map" not in cmd


def test_extract_segment_nonzero_track_has_map_flag(tmp_path):
    """audio_track=2 must add '-map' '0:a:2'."""
    video = tmp_path / "v.mp4"
    video.touch()
    captured_cmds = []

    def fake_run(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        return MagicMock(returncode=0)

    with patch("submatch.audio.subprocess.run", side_effect=fake_run):
        import submatch.audio as _audio
        import tempfile as _tf
        with patch.object(_tf, "NamedTemporaryFile") as mock_ntf:
            mock_ntf.return_value.name = str(tmp_path / "out.wav")
            mock_ntf.return_value.close = lambda: None
            (tmp_path / "out.wav").write_bytes(b"")
            _audio.extract_segment(video, 0, 3_000, audio_track=2)

    assert captured_cmds
    cmd = captured_cmds[0]
    assert "-map" in cmd
    assert cmd[cmd.index("-map") + 1] == "0:a:2"
```

- [ ] **Step 2: Run tests to verify they pass (implementation is already in place from Task 1)**

```bash
pytest tests/test_audio.py::test_extract_segment_default_track_no_map_flag tests/test_audio.py::test_extract_segment_nonzero_track_has_map_flag -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_audio.py
git commit -m "test: verify extract_segment -map flag behavior for audio_track param"
```

---

### Task 3: `sync_subtitle` audio_track parameter

**Files:**
- Modify: `submatch/sync.py`
- Test: `tests/test_sync.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_sync.py`:

```python
def test_sync_subtitle_default_track_no_reference_stream(tmp_path):
    """audio_track=0 must NOT add --reference-stream to the ffs command."""
    video = tmp_path / "video.mp4"
    video.touch()
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text(SAMPLE_SRT)
    output_path = tmp_path / "synced.srt"
    output_path.write_text(SAMPLE_SRT)

    with patch("submatch.sync.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        sync_subtitle(video, subtitle, output_path, audio_track=0)

    called_cmd = mock_run.call_args[0][0]
    assert "--reference-stream" not in called_cmd


def test_sync_subtitle_nonzero_track_has_reference_stream(tmp_path):
    """audio_track=1 must add '--reference-stream' 'a:1' to the ffs command."""
    video = tmp_path / "video.mp4"
    video.touch()
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text(SAMPLE_SRT)
    output_path = tmp_path / "synced.srt"
    output_path.write_text(SAMPLE_SRT)

    with patch("submatch.sync.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        sync_subtitle(video, subtitle, output_path, audio_track=1)

    called_cmd = mock_run.call_args[0][0]
    assert "--reference-stream" in called_cmd
    ref_idx = called_cmd.index("--reference-stream")
    assert called_cmd[ref_idx + 1] == "a:1"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_sync.py::test_sync_subtitle_default_track_no_reference_stream tests/test_sync.py::test_sync_subtitle_nonzero_track_has_reference_stream -v
```

Expected: FAIL — the function doesn't have an `audio_track` parameter yet.

- [ ] **Step 3: Implement `audio_track` parameter in `sync_subtitle`**

Replace the `sync_subtitle` function in `submatch/sync.py`:

```python
def sync_subtitle(
    video_path: Path,
    subtitle_path: Path,
    output_path: Path | None = None,
    drift_threshold: float = DRIFT_THRESHOLD_SECONDS,
    audio_track: int = 0,
) -> SyncResult:
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
        output_path = Path(tmp.name)
        tmp.close()

    cmd = ["ffs", str(video_path), "-i", str(subtitle_path), "-o", str(output_path)]
    if audio_track > 0:
        cmd += ["--reference-stream", f"a:{audio_track}"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffsubsync failed: {result.stderr.strip()}")

    offset = _compute_offset(subtitle_path, output_path)
    return SyncResult(
        synced_srt_path=output_path,
        offset_seconds=offset,
        drift_detected=abs(offset) > drift_threshold,
    )
```

- [ ] **Step 4: Run all sync tests**

```bash
pytest tests/test_sync.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add submatch/sync.py tests/test_sync.py
git commit -m "feat: add audio_track param to sync_subtitle"
```

---

### Task 4: `MatchResult` audio track fields and `print_human` track line

**Files:**
- Modify: `submatch/output.py`
- Test: `tests/test_output.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_output.py`:

```python
def test_match_result_default_audio_track_fields():
    result = _make_result()
    assert result.audio_track_index == 0
    assert result.audio_track_lang is None


def test_format_json_includes_audio_track_fields():
    result = _make_result()
    result.audio_track_index = 1
    result.audio_track_lang = "jpn"
    data = json.loads(format_json(result))
    assert data["audio_track_index"] == 1
    assert data["audio_track_lang"] == "jpn"


def test_print_human_omits_track_line_when_default(capsys):
    result = _make_result()
    # audio_track_index=0 and audio_track_lang=None → no track line
    print_human(result)
    out = capsys.readouterr().out
    assert "track" not in out


def test_print_human_shows_track_line_with_lang(capsys):
    result = _make_result()
    result.audio_track_index = 1
    result.audio_track_lang = "jpn"
    print_human(result)
    out = capsys.readouterr().out
    assert "track" in out
    assert "a:1" in out
    assert "jpn" in out


def test_print_human_shows_track_line_no_lang(capsys):
    result = _make_result()
    result.audio_track_index = 2
    result.audio_track_lang = None
    print_human(result)
    out = capsys.readouterr().out
    assert "track" in out
    assert "a:2" in out


def test_print_human_shows_track_line_when_lang_known_but_index_zero(capsys):
    """Show track line if lang is known even when index is 0 (explicit track 0 selection)."""
    result = _make_result()
    result.audio_track_index = 0
    result.audio_track_lang = "eng"
    print_human(result)
    out = capsys.readouterr().out
    assert "track" in out
    assert "a:0" in out
    assert "eng" in out
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_output.py::test_match_result_default_audio_track_fields tests/test_output.py::test_format_json_includes_audio_track_fields tests/test_output.py::test_print_human_omits_track_line_when_default tests/test_output.py::test_print_human_shows_track_line_with_lang tests/test_output.py::test_print_human_shows_track_line_no_lang tests/test_output.py::test_print_human_shows_track_line_when_lang_known_but_index_zero -v
```

Expected: FAIL with `AttributeError: ... has no attribute 'audio_track_index'`

- [ ] **Step 3: Add fields to `MatchResult` in `output.py`**

In `submatch/output.py`, replace the `MatchResult` dataclass:

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
    state: MatchState = MatchState.FAIL
    resynced: bool = False
    audio_track_index: int = 0
    audio_track_lang: str | None = None
```

- [ ] **Step 4: Run field tests to verify they pass, print_human tests still fail**

```bash
pytest tests/test_output.py::test_match_result_default_audio_track_fields tests/test_output.py::test_format_json_includes_audio_track_fields -v
```

Expected: PASS

```bash
pytest tests/test_output.py::test_print_human_shows_track_line_with_lang -v
```

Expected: FAIL (no track line printed yet)

- [ ] **Step 5: Add track line to `print_human` in `output.py`**

In `print_human`, after the sync block (after the `else: print(f"sync  no drift  ...")` line), add:

```python
    if result.audio_track_index > 0 or result.audio_track_lang is not None:
        lang_part = f" ({result.audio_track_lang})" if result.audio_track_lang else ""
        print(f"track  a:{result.audio_track_index}{lang_part}")
```

The full sync + track block in `print_human` should look like:

```python
    if result.sync is None:
        print("sync  skipped")
    elif result.sync.drift_detected:
        sign = "+" if result.sync.offset_seconds >= 0 else ""
        print(f"sync  {sign}{result.sync.offset_seconds:.1f}s  {_YELLOW}⚠{_RESET}")
    else:
        print(f"sync  no drift  {_GREEN}✓{_RESET}")

    if result.audio_track_index > 0 or result.audio_track_lang is not None:
        lang_part = f" ({result.audio_track_lang})" if result.audio_track_lang else ""
        print(f"track  a:{result.audio_track_index}{lang_part}")
```

- [ ] **Step 6: Run all output tests**

```bash
pytest tests/test_output.py -v
```

Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add submatch/output.py tests/test_output.py
git commit -m "feat: add audio_track_index/lang fields to MatchResult and track line to print_human"
```

---

### Task 5: `_VideoCache` audio track fields and `_score_pair` integration in `cli.py`

**Files:**
- Modify: `submatch/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_cli.py`:

```python
def test_score_pair_resolve_audio_track_called_once_per_video(tmp_path):
    """resolve_audio_track is called exactly once even for two subtitles sharing a video."""
    video = tmp_path / "video.mp4"
    video.touch()
    sub1 = tmp_path / "sub.en.srt"
    sub2 = tmp_path / "sub.jp.srt"
    sub1.write_text(SAMPLE_SRT)
    sub2.write_text(SAMPLE_SRT)

    subs = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(video), str(sub1), str(sub2),
                             "--no-sync", "--compact", "--audio-track", "1"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.cli.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.cli.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.cli.audio.resolve_audio_track", return_value=(1, "jpn")) as mock_resolve, \
         patch("submatch.cli.subtitle.parse", return_value=subs), \
         patch("submatch.cli.sampler.select_segments", return_value=segs), \
         patch("submatch.cli.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.cli.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.cli.language.detect_from_text", return_value="en"), \
         patch("submatch.cli.language.detect_from_filename", return_value="en"), \
         patch("submatch.cli.language.detect_from_video", return_value=None), \
         patch("submatch.cli.language.build_result", return_value=lang):
        with pytest.raises(SystemExit):
            cli.main()

    assert mock_resolve.call_count == 1


def test_score_pair_passes_audio_track_to_extract_segment(tmp_path):
    """extract_segment is called with audio_track=1 when --audio-track 1 is used."""
    video = tmp_path / "video.mp4"
    video.touch()
    sub = tmp_path / "sub.srt"
    sub.write_text(SAMPLE_SRT)

    subs = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(video), str(sub), "--no-sync", "--audio-track", "1"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.cli.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.cli.audio.extract_segment", return_value=tmp_path / "seg.wav") as mock_extract, \
         patch("submatch.cli.audio.resolve_audio_track", return_value=(1, "jpn")), \
         patch("submatch.cli.subtitle.parse", return_value=subs), \
         patch("submatch.cli.sampler.select_segments", return_value=segs), \
         patch("submatch.cli.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.cli.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.cli.language.detect_from_text", return_value="en"), \
         patch("submatch.cli.language.detect_from_filename", return_value="en"), \
         patch("submatch.cli.language.detect_from_video", return_value=None), \
         patch("submatch.cli.language.build_result", return_value=lang):
        with pytest.raises(SystemExit):
            cli.main()

    assert mock_extract.call_count >= 1
    for call in mock_extract.call_args_list:
        assert call.kwargs.get("audio_track") == 1 or (len(call.args) > 3 and call.args[3] == 1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli.py::test_score_pair_resolve_audio_track_called_once_per_video tests/test_cli.py::test_score_pair_passes_audio_track_to_extract_segment -v
```

Expected: FAIL — either because `--audio-track` flag doesn't exist yet (Step 6 adds it) or because `audio_track` isn't threaded through.

Note: Run this after Task 6 adds the `--audio-track` flag to `parse_args`. If running tasks in order, come back to run this check after Task 6 Step 3. For now proceed to the implementation.

- [ ] **Step 3: Add audio track fields to `_VideoCache` in `cli.py`**

In `submatch/cli.py`, replace the `_VideoCache` dataclass:

```python
@dataclasses.dataclass
class _VideoCache:
    """Transcriptions from a video's first subtitle pass, reused for subsequent subtitles."""
    segment_starts: list[int]
    transcriptions: list[str]
    audio_lang: str | None
    audio_track_index: int = 0
    audio_track_lang: str | None = None
```

- [ ] **Step 4: Add audio track resolution to `_score_pair` in `cli.py`**

In `_score_pair`, insert a block right after `def _phase(...)` and before `sync_result = None`. The insertion point is around line 274. Add:

```python
    # Resolve audio track once per video; subsequent subtitles reuse from cache.
    if video_cache is not None:
        audio_track_index = video_cache.audio_track_index
        audio_track_lang = video_cache.audio_track_lang
    else:
        audio_track_index = 0
        audio_track_lang: str | None = None
        _at_spec = getattr(args, 'audio_track', None)
        if _at_spec:
            audio_track_index, audio_track_lang = audio.resolve_audio_track(video, _at_spec)
```

- [ ] **Step 5: Pass `audio_track_index` to `sync_subtitle` in `_score_pair`**

Change the `sync.sync_subtitle` call (currently `sync.sync_subtitle(video, subtitle_path, _sync_tmp, drift_threshold=args.drift_threshold)`) to:

```python
sync_result = sync.sync_subtitle(
    video, subtitle_path, _sync_tmp,
    drift_threshold=args.drift_threshold,
    audio_track=audio_track_index,
)
```

- [ ] **Step 6: Pass `audio_track_index` to `extract_segment` in `_score_pair`**

Change the `audio.extract_segment` call inside the segment loop (currently `wav_path = audio.extract_segment(video, seg.start_ms, 30_000)`) to:

```python
wav_path = audio.extract_segment(video, seg.start_ms, 30_000, audio_track=audio_track_index)
```

- [ ] **Step 7: Store track in `new_cache` and `MatchResult` in `_score_pair`**

In the `if video_cache is None` branch, update `new_cache` construction:

```python
new_cache = _VideoCache(
    segment_starts=[seg.start_ms for _, seg, _ in transcription_pairs],
    transcriptions=[t for _, _, t in transcription_pairs],
    audio_lang=audio_lang,
    audio_track_index=audio_track_index,
    audio_track_lang=audio_track_lang,
)
```

In the `MatchResult` construction (around line 399), add the two new fields:

```python
match_result = output.MatchResult(
    confidence=confidence,
    passed=confidence >= effective_threshold,
    threshold=effective_threshold,
    language=lang_result,
    sync=sync_result,
    segments=segment_results,
    model=args.model,
    cross_language=cross_lang,
    subtitle_language=subtitle_lang,
    audio_track_index=audio_track_index,
    audio_track_lang=audio_track_lang,
)
```

- [ ] **Step 8: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/integration
```

Expected: all tests PASS (the two new cli tests may still fail if Task 6 hasn't been done — that's OK, run them again after Task 6)

- [ ] **Step 9: Commit**

```bash
git add submatch/cli.py tests/test_cli.py
git commit -m "feat: resolve and thread audio_track through _score_pair and _VideoCache"
```

---

### Task 6: `--audio-track` CLI flag and `parse_args` tests

**Files:**
- Modify: `submatch/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_cli.py`, update `test_parse_args_defaults` to add the new assertion (the test already exists — add one line):

```python
    assert args.audio_track is None
```

Update `test_parse_args_all_flags` to add `"--audio-track", "jp,en"` to the argv list and add:

```python
    assert args.audio_track == "jp,en"
```

Also add a dedicated test:

```python
def test_parse_args_audio_track_integer(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s), "--audio-track", "2"]):
        args = cli.parse_args()
    assert args.audio_track == "2"


def test_parse_args_audio_track_language_preference(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s), "--audio-track", "jp,en,pt"]):
        args = cli.parse_args()
    assert args.audio_track == "jp,en,pt"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli.py::test_parse_args_defaults tests/test_cli.py::test_parse_args_audio_track_integer tests/test_cli.py::test_parse_args_audio_track_language_preference -v
```

Expected: FAIL with `AttributeError: Namespace object has no attribute 'audio_track'`

- [ ] **Step 3: Add `--audio-track` to `parse_args` in `cli.py`**

In `parse_args`, after the `--timing` argument (before `--version`), add:

```python
    parser.add_argument(
        "--audio-track", default=None, dest="audio_track",
        help="audio track to use: integer index (0-based) or comma-separated language preference list (e.g. jp,en,pt)",
    )
```

- [ ] **Step 4: Run all new parse_args tests**

```bash
pytest tests/test_cli.py::test_parse_args_defaults tests/test_cli.py::test_parse_args_all_flags tests/test_cli.py::test_parse_args_audio_track_integer tests/test_cli.py::test_parse_args_audio_track_language_preference -v
```

Expected: all PASS

- [ ] **Step 5: Run the full cli test suite including the Task 5 tests**

```bash
pytest tests/test_cli.py -v
```

Expected: all tests PASS (including `test_score_pair_resolve_audio_track_called_once_per_video` and `test_score_pair_passes_audio_track_to_extract_segment`)

- [ ] **Step 6: Run the complete unit test suite**

```bash
pytest tests/ --ignore=tests/integration -v
```

Expected: all tests PASS, coverage ≥ 95%

- [ ] **Step 7: Verify `--help` output**

```bash
submatch --help
```

Verify `--audio-track` appears with the correct description.

- [ ] **Step 8: Commit**

```bash
git add submatch/cli.py tests/test_cli.py
git commit -m "feat: add --audio-track flag to CLI"
```

---

### Task 7: Update README and ROADMAP

**Files:**
- Modify: `README.md`
- Modify: `docs/ROADMAP.md`

- [ ] **Step 1: Add `--audio-track` to the options table in `README.md`**

Find the options table in `README.md` and add a row for `--audio-track`. The flag belongs in the same section as `--model`, `--segments`, etc. (content options, not output options).

The row to add:

```markdown
| `--audio-track TEXT` | Audio track to use: integer index (0-based) or comma-separated language preference list (`jp,en,pt`). Default: track 0. |
```

- [ ] **Step 2: Update ROADMAP.md to mark Multiple Audio Track Selection as completed**

In `docs/ROADMAP.md`, replace the "Multiple audio track selection" section:

```markdown
## Multiple audio track selection ✓

Released in v0.X.X. `--audio-track` accepts either an integer index or a comma-separated language preference list (e.g. `--audio-track jp,en,pt`). If no track matches, falls back to track 0 with a warning.
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/ROADMAP.md
git commit -m "docs: document --audio-track flag and mark roadmap item complete"
```

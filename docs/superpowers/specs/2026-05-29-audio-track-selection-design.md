# Audio Track Selection Design

## Problem

Foreign films and dubbed content often ship with multiple audio tracks (e.g., the original Japanese track and an English dub). `submatch` currently always transcribes the first audio track (`a:0`). If the subtitle being checked matches a non-default track, the score will be artificially low, making submatch unusable for that content.

## Goal

Let users specify which audio track to transcribe so submatch can correctly verify subtitles for multi-track media.

---

## CLI Interface

```
--audio-track TEXT    audio track to use: integer index (0-based) or
                      comma-separated language preference list (e.g. jp,en,pt).
                      Default: track 0.
```

Two input forms accepted:

| Input | Meaning |
|---|---|
| `--audio-track 1` | Use track at index 1 (0-based) |
| `--audio-track jp` | Prefer Japanese; fall back to track 0 with a warning |
| `--audio-track jp,en,pt` | Prefer Japanese, then English, then Portuguese; fall back to 0 |

If no `--audio-track` is given, behavior is unchanged (track 0, no output change).

---

## New Functions in `audio.py`

### `list_audio_tracks(video_path: Path) -> list[dict]`

Calls ffprobe with `-show_streams -select_streams a` and returns one dict per audio stream:

```python
[
    {"index": 0, "lang": "eng"},   # lang from stream tags["language"], may be None
    {"index": 1, "lang": "jpn"},
]
```

### `resolve_audio_track(video_path: Path, spec: str) -> tuple[int, str | None]`

Parses `spec` and returns `(track_index, track_lang_or_None)`.

**Integer path:** `spec.strip()` is a valid integer → return `(int(spec), tracks[int(spec)]["lang"])`. Error (exit 2) if index is out of range.

**Language path:** split `spec` on `,`, strip each token, lowercase. For each preference in order, check tracks for a match. Matching rules:
- Case-insensitive
- ISO 639-1 two-letter code matches its ISO 639-2 three-letter equivalent (e.g. `jp` matches `jpn`, `en` matches `eng`) using a small lookup table
- If match found: return `(track_index, track_lang)`
- If no preference matched: print a warning to stderr and return `(0, tracks[0]["lang"] if tracks else None)`

---

## Changes to Existing Code

### `audio.extract_segment`

Add `audio_track: int = 0` parameter. When `audio_track > 0`, add `-map 0:a:{audio_track}` to the ffmpeg command (the default ffmpeg behavior selects `a:0` implicitly so no change for track 0).

### `sync.sync_subtitle`

Add `audio_track: int = 0` parameter. When `audio_track > 0`, append `--reference-stream a:{audio_track}` to the `ffs` command.

### `cli.py` — `_VideoCache`

Add two fields:
```python
audio_track_index: int = 0
audio_track_lang: str | None = None
```

### `cli.py` — `_score_pair`

- Resolve the track once at the top (called only when `video_cache is None`), store result in `new_cache`
- Pass `audio_track=cache.audio_track_index` through to `audio.extract_segment` and `sync.sync_subtitle`
- Populate `MatchResult.audio_track_index` and `MatchResult.audio_track_lang`

### `output.py` — `MatchResult`

Add two fields:
```python
audio_track_index: int = 0
audio_track_lang: str | None = None
```

### `output.py` — `print_human`

When `result.audio_track_index > 0` or `result.audio_track_lang` is not None, print a `track` line after the `sync` line:

```
track  a:1 (jpn)
```

When track is 0 and no language is known, omit the line entirely (no change in default behavior).

---

## Data Flow

```
parse_args  →  --audio-track TEXT stored as args.audio_track (str | None)
                ↓
_score_pair (first subtitle for a video):
  audio.list_audio_tracks(video)
  audio.resolve_audio_track(video, args.audio_track)  →  (index, lang)
  stored in _VideoCache
                ↓
  sync.sync_subtitle(..., audio_track=index)
  audio.extract_segment(..., audio_track=index)   [per segment]
                ↓
  MatchResult(audio_track_index=index, audio_track_lang=lang)
```

Subsequent subtitles for the same video reuse `_VideoCache` (track resolved once per video).

---

## Batch Mode

No special handling needed. `_score_pair` receives `args.audio_track` for every pair. Since a video's track is resolved once and cached in `_VideoCache`, the ffprobe call is not repeated for additional subtitles of the same video.

---

## Error Handling

| Condition | Behavior |
|---|---|
| Integer index out of range | Print error to stderr, exit 2 |
| Language not found in any track | Print warning to stderr, use track 0 |
| ffprobe fails to list tracks | Print warning, use track 0 |
| `--audio-track` with value that is neither a valid integer nor a valid language spec | Treat as language code (will warn if no match) |

---

## Testing

Unit tests cover:
- `list_audio_tracks`: parses ffprobe JSON with 0, 1, and 2+ streams; handles missing `language` tag
- `resolve_audio_track`: integer happy path; integer out of range; single language code match; preference list first match; preference list fallback; ISO 639-1 to 639-2 matching; case-insensitive matching; no tracks available
- `extract_segment`: with `audio_track=0` (no `-map` flag added); with `audio_track=2` (`-map 0:a:2` present)
- `sync_subtitle`: with `audio_track=0` (no `--reference-stream`); with `audio_track=1` (flag present)
- `print_human`: track line appears when index > 0; omitted when index == 0 and lang is None
- CLI: `--audio-track` flag is parsed; passed through to `_score_pair`

Integration tests are not added (existing integration tests cover the default track 0 path).

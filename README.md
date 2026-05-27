# submatch

Verify that a subtitle file matches the audio content of a video.

Subtitle download tools (like [subliminal](https://github.com/Diaoul/subliminal)) sometimes return correctly-timed but wrong-content subtitles — a different episode, a different release, or the wrong language track. `submatch` catches this by transcribing short audio segments with [Whisper](https://github.com/openai/whisper) and comparing against the subtitle text using token F1 scoring.

```
submatch video.mkv subtitle.en.srt
✔  Match confidence: 0.61  [████████████░░░░░░░░] PASS
   Audio language : en
   Subtitle lang  : en
   Drift          : 0.2 s
```

## Install

```bash
pip install -e .
```

System dependencies (must be on PATH):

```bash
# macOS
brew install ffmpeg
pip install ffsubsync   # provides the 'ffs' command
```

Whisper model weights download automatically on first run.

## Usage

**Single file:**
```bash
submatch video.mkv subtitle.en.srt
submatch video.mkv subtitle.pt.srt --model small --threshold 0.4 --verbose
submatch video.mkv subtitle.en.srt --no-sync --json
```

**Batch mode — directory of paired files:**
```bash
submatch /media/movies/            # pairs each video with its same-stem subtitle
submatch /media/movies/ --compact  # one line per pair
submatch /media/movies/ --json     # machine-readable JSON array
```

**Batch mode — one video against a subtitle directory:**
```bash
submatch movie.mkv subs/           # scores every subtitle in subs/ against movie.mkv
```

**Recursive — walk nested directory trees:**
```bash
submatch /media/series/ --recursive          # Plex/Kodi library layout
submatch movie.mkv subs/ -r                  # recurse into subs/ subdirectories
```

**Filtering — process only specific subtitles:**
```bash
submatch /media/natal/ --sub-lang pt          # matches pt.srt, pt-BR.srt, pt-PT.srt
submatch /media/natal/ --sub-lang en --sub-lang pt-BR   # multiple codes
submatch movie.mkv subs/ --filter "*.en.*"    # glob on subtitle filename
submatch /media/natal/ --sub-lang pt --filter "*.srt"   # both must pass
```

### Cross-language matching

When the subtitle language differs from the audio language (e.g. English audio with Portuguese subtitles), `submatch` automatically switches from token F1 scoring to multilingual semantic similarity using [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). The score is normalized so the same `--threshold` applies to both same-language and cross-language pairs.

Use `--cross-threshold` to tune the pass/fail cutoff for translated subtitles independently:

```bash
submatch movie.mkv movie.pt.srt --cross-threshold 0.5
```

The model is downloaded on first use (~90 MB) and cached by sentence-transformers.

## Supported subtitle formats

SRT, WebVTT, ASS/SSA (and any other format supported by [pysubs2](https://github.com/tkarabela/pysubs2)).

## Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` |
| `--threshold` | `0.35` | Pass/fail confidence cutoff (0–1) |
| `--cross-threshold` | same as `--threshold` | Pass/fail threshold for cross-language pairs |
| `--segments` | auto | Number of audio segments to sample |
| `--language` | auto | Expected audio language (e.g. `en`, `pt`) |
| `--no-sync` | off | Skip ffsubsync timing drift check |
| `--keep-synced` | off | Save timing-corrected subtitle to disk |
| `--recursive`, `-r` | off | Walk nested directories in batch mode |
| `--sub-lang CODE` | off | Keep only subtitles whose filename language code starts with CODE (repeatable; infers from text for untagged files) |
| `--filter GLOB` | off | Keep only subtitles whose filename matches the glob (e.g. `*.en.*`) |
| `--json` | off | Machine-readable JSON output |
| `--compact` | off | One-line-per-pair summary in batch mode |
| `--verbose` | off | Show subtitle and transcription text per segment |
| `--device` | `auto` | Whisper inference device: `cpu`, `mps` (Apple Silicon), `cuda` (NVIDIA), `auto` |
| `--workers` | `auto` | Parallel pairs in batch mode; auto selects 1 for GPU, up to 4 for CPU |

Segment count auto-selection: `< 30 min` → 5, `30–90 min` → 8, `> 90 min` → 12.

## How it works

1. **Sync** — runs `ffs` (ffsubsync) to correct timing drift; flags offsets > 2 s
2. **Sample** — divides the video into N zones (skipping first/last 5%), picks the 30-second window with the most subtitle words per zone
3. **Transcribe** — extracts each window as a 16 kHz mono WAV and transcribes with Whisper
4. **Score** — normalises both texts (lowercase, strip punctuation, remove fillers), computes token F1 per segment, returns a weighted average
5. **Report** — prints confidence, language signals, and drift; exits 0/1/2

The default threshold of 0.35 is intentionally low — subtitle text often paraphrases rather than quoting verbatim.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Confidence at or above threshold — subtitles match |
| `1` | Confidence below threshold — subtitles likely wrong |
| `2` | Error (missing dependency, unreadable file, no audio track, etc.) |

## Limitations

- Same-language only. Token F1 requires the subtitle and audio to be in the same language; translated subtitles will score near zero even if correct.
- Requires a local Whisper install (`pip install openai-whisper`). No API key needed.

# submatch

[![PyPI version](https://img.shields.io/pypi/v/submatch)](https://pypi.org/project/submatch/)
[![Python versions](https://img.shields.io/pypi/pyversions/submatch)](https://pypi.org/project/submatch/)
[![License](https://img.shields.io/github/license/vitormf/submatch)](LICENSE)

Verify that a subtitle file matches the audio content of a video.

Subtitle download tools (like [subliminal](https://github.com/Diaoul/subliminal)) sometimes return correctly-timed but wrong-content subtitles — a different episode, a different release, or the wrong language track. `submatch` catches this by transcribing short audio segments with [Whisper](https://github.com/openai/whisper) and comparing against the subtitle text using token F1 scoring.

```
submatch video.mkv subtitle.en.srt

PASS ✓  0.61  (thr 0.35 · base · 5 segs)
lang  audio=en  ·  sub=en
sync  no drift  ✓
  #1  00:04:12  0.68  ██████░░
  #2  00:18:44  0.55  ████░░░░
```

## Install

```bash
pip install submatch
```

ffmpeg is bundled automatically. Whisper model weights download on first run.

## Usage

**Single pair:**
```bash
submatch video.mkv subtitle.en.srt
submatch video.mkv subtitle.pt.srt --model small --threshold 0.4 --verbose
submatch video.mkv subtitle.en.srt --no-sync --json report.json
```

**Auto-discover — pass what you have:**
```bash
submatch video.mkv              # find all subtitles alongside the video
submatch subtitle.en.srt        # find the video alongside the subtitle
submatch v1.mkv v2.mkv          # each video finds its own subtitles
submatch s1.srt s2.srt          # each subtitle finds its own video
submatch video.mkv s1.srt s2.srt  # explicit subtitles for one video
```

**Batch mode — directory of paired files:**
```bash
submatch /media/movies/            # recursive by default; pairs each video with its subtitles
submatch /media/movies/ --compact  # one line per pair
submatch /media/movies/ --json results.json  # machine-readable JSON array
submatch /media/movies/ --no-recursive  # flat directory only
```

**Batch mode — one video against a subtitle directory:**
```bash
submatch movie.mkv subs/           # scores every subtitle in subs/ against movie.mkv
```

**Embedded subtitles — score subtitle tracks in the video container:**
```bash
submatch --embedded movie.mkv
submatch --embedded /path/to/library/
```

**Watch mode — monitor a directory for new pairs:**
```bash
submatch --watch /media/movies/
submatch --watch /media/movies/ --sub-lang en --delete-failures
submatch --watch /media/movies/ --poll             # for network mounts (NFS, SMB)
submatch --watch /media/movies/ --poll --interval 30
```

**Filtering — process only specific subtitles:**
```bash
submatch /media/shows/ --sub-lang pt          # matches pt.srt, pt-BR.srt, pt-PT.srt
submatch /media/shows/ --sub-lang en --sub-lang pt-BR   # multiple codes
submatch movie.mkv subs/ --filter "*.en.*"    # glob on subtitle filename
submatch /media/shows/ --sub-lang pt --filter "*.srt"   # both must pass
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

## Language support

✓ = confirmed by integration tests · ~ = supported by underlying tools, not yet integration-tested

| Language | Audio | Subtitle |
|---|---|---|
| Arabic | ~ | ✓ |
| Chinese (Simplified) | ✓ | ✓ |
| Czech | ~ | ✓ |
| Danish | ~ | ✓ |
| Dutch | ~ | ✓ |
| English | ✓ | ✓ |
| Finnish | ~ | ✓ |
| French | ✓ | ✓ |
| German | ✓ | ✓ |
| Greek | ~ | ✓ |
| Hebrew | ~ | ✓ |
| Hindi | ✓ | ✓ |
| Hungarian | ~ | ✓ |
| Indonesian | ~ | ✓ |
| Italian | ✓ | ✓ |
| Japanese | ~ | ✓ |
| Korean | ~ | ~ |
| Malayalam | ~ | ✓ |
| Neapolitan | ✓ | ✓ |
| Polish | ~ | ✓ |
| Portuguese | ✓ | ✓ |
| Portuguese (Brazil) | ✓ | ✓ |
| Romanian | ~ | ✓ |
| Russian | ~ | ✓ |
| Spanish | ✓ | ✓ |
| Swedish | ~ | ✓ |
| Thai | ~ | ✓ |
| Turkish | ✓ | ✓ |
| Ukrainian | ~ | ✓ |
| Vietnamese | ~ | ✓ |

**Audio** — Whisper can transcribe the spoken language. Chinese (Simplified) is tested via Shanghainese and Guiyangese speakers; standard Mandarin is expected to work. Thai audio is supported by Whisper but our integration tests use the tiny model which does not reliably transcribe Thai. **Subtitle** — submatch can score a subtitle in that language using token F1 (same-language) or multilingual sentence embeddings (cross-language, via [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)). Korean subtitle scoring is supported but no test fixture is available.

## Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` |
| `--threshold` | `0.35` | Pass/fail confidence cutoff (0–1) |
| `--cross-threshold` | same as `--threshold` | Pass/fail threshold for cross-language pairs |
| `--segments` | auto | Number of audio segments to sample |
| `--audio-track` | `0` | Audio track to use: integer index (0-based) or comma-separated language preference list (`jp,en,pt`). Default: track 0. |
| `--embedded` | off | Score embedded subtitle tracks in the video container instead of external files |
| `--language` | auto | Expected audio language (e.g. `en`, `pt`) |
| `--drift-threshold` | `2.0` | Seconds of timing offset before flagging as drift |
| `--no-sync` | off | Skip ffsubsync timing drift check |
| `--keep-synced` | off | Save timing-corrected subtitle to disk |
| `--no-recursive` | off | Do not recurse into subdirectories when expanding directories (default: recursive) |
| `--sub-lang CODE` | off | Keep only subtitles whose filename language code starts with CODE (repeatable; infers from text for untagged external files; always includes untagged embedded tracks) |
| `--filter GLOB` | off | Keep only subtitles whose filename matches the glob (e.g. `*.en.*`) |
| `--json FILE` | off | Write JSON report to FILE |
| `--csv FILE` | off | Write CSV report to FILE |
| `--html FILE` | off | Write self-contained HTML report to FILE |
| `--compact` | off | One-line-per-pair summary in batch mode |
| `--verbose` | off | Show subtitle and transcription text per segment |
| `--device` | `auto` | Whisper inference device: `cpu`, `mps` (Apple Silicon), `cuda` (NVIDIA), `auto` (CUDA > CPU; use `--device mps` explicitly on Apple Silicon) |
| `--workers` | `auto` | Parallel pairs in batch mode; auto selects up to 4 |
| `--delete-failures` | off | Delete subtitle files that fail the match check |
| `--resync` | off | On DRIFT (drift detected), copy synced subtitle over original and re-score |
| `--pass-unsure` | off | Exit 0 for UNSURE results (not enough transcription data) |
| `--no-cache` | off | Disable transcription cache and use subtitle-driven segment selection |
| `--clear-cache` | off | Delete all cached transcriptions and exit |
| `--timing` | off | Print per-phase timing breakdown (single-pair mode only) |
| `--watch` | off | Monitor a directory for new video/subtitle pairs and score them as they appear |
| `--poll` | off | Use polling instead of native filesystem events (required for network mounts) |
| `--interval N` | `10` | Seconds between directory scans in `--poll` mode |

Segment count auto-selection: `< 30 min` → 5, `30–90 min` → 8, `> 90 min` → 12.

**Breaking change:** `--json` now requires a filename. Bare `--json` is a parse error. Update scripts from `--json` to `--json output.json`. The same applies to `--csv` and `--html`.

## Configuration

`submatch` reads defaults from two TOML config files, merged in order:

1. `~/.config/submatch/config.toml` — personal defaults applied everywhere
2. `./submatch.toml` — directory-level defaults (overrides user config)

CLI flags always override both.

**Example `~/.config/submatch/config.toml`:**

```toml
model = "small"
threshold = 0.40
language = "en"
workers = 2
```

**Configurable flags:** `model`, `threshold`, `segments`, `language`, `no_sync`, `keep_synced`, `no_recursive`, `sub_lang`, `filter`, `device`, `workers`, `delete_failures`, `cross_threshold`, `resync`, `pass_unsure`, `drift_threshold`, `audio_track`, `cache_ttl_days`, `cache_max_mb`, `cache_dir`

> **Note:** Boolean flags set to `true` in config (e.g. `no_sync = true`) cannot be overridden back to `false` via the CLI — remove the line from your config instead.
>
> **Warning:** `delete_failures = true` will silently delete subtitle files on every run. Use with care.

## Telemetry

submatch reports crashes and unexpected pipeline errors to Sentry to help improve
reliability. No file paths or personal data are transmitted — all path strings are
replaced with `<path>` before sending.

To opt out, set an environment variable:

```
export SUBMATCH_NO_TELEMETRY=1
```

Or add to `~/.config/submatch/config.toml` or `./submatch.toml`:

```toml
telemetry = false
```

## How it works

1. **Sync** — runs `ffs` (ffsubsync) to correct timing drift; flags offsets > 2 s
2. **Sample** — divides the video into N zones (skipping first/last 5%). By default, uses ffmpeg `silencedetect` to find speech-rich windows (audio-driven mode); picks the best 30-second window per zone based on speech coverage, with a quality gate that retries if Whisper confidence is low. Use `--no-cache` for the original subtitle-driven path (highest word-count window per zone).
3. **Cache** — validated transcriptions are stored in `~/.cache/submatch/` keyed by video path, mtime, model, and segment count. Subsequent runs on the same video skip audio extraction and transcription entirely. Cache is evicted after 30 days or when it exceeds 200 MB (LRU). Use `--no-cache` to bypass or `--clear-cache` to wipe.
4. **Transcribe** — extracts each window as a 16 kHz mono WAV and transcribes with Whisper
5. **Score** — normalises both texts (lowercase, strip punctuation, remove fillers), computes token F1 per segment, returns a weighted average
6. **Report** — prints confidence, language signals, and drift; exits 0/1/2

The default threshold of 0.35 is intentionally low — subtitle text often paraphrases rather than quoting verbatim.

## States and exit codes

Each pair is assigned one of four states:

| State | Meaning | Exit code |
|---|---|---|
| `PASS` | Content matches, no timing drift | `0` |
| `DRIFT` | Content matches, but timing drift detected | `1` (use `--resync` to fix in place) |
| `FAIL` | Content does not match | `1` |
| `UNSURE` | Not enough transcription data to decide | `1` (use `--pass-unsure` to exit `0`) |
| — | Error (missing dependency, unreadable file, no audio track) | `2` |

## Acknowledgements

`submatch` is a complement to the existing subtitle ecosystem, not a replacement for it. It wouldn't exist without:

- [openai/whisper](https://github.com/openai/whisper) — the speech recognition engine that powers transcription
- [smacke/ffsubsync](https://github.com/smacke/ffsubsync) — timing drift correction used before scoring
- [tkarabela/pysubs2](https://github.com/tkarabela/pysubs2) — multi-format subtitle parsing (SRT, VTT, ASS/SSA)
- [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) — multilingual embeddings for cross-language scoring
- [Diaoul/subliminal](https://github.com/Diaoul/subliminal) and [morpheus65535/bazarr](https://github.com/morpheus65535/bazarr) — the subtitle download tools that `submatch` is designed to work alongside

## Limitations

- Runs Whisper locally — no API key needed. Model weights download on first run.
- Cross-language scoring uses multilingual sentence embeddings and is less precise than same-language token F1 — consider lowering `--cross-threshold` if you get too many false negatives.

# submatch

Verify that a subtitle file matches the audio content of a video.

## Usage

```bash
submatch video.mkv subtitle.srt
submatch video.mkv subtitle.srt --model small --threshold 0.4
submatch video.mkv subtitle.srt --json
```

## Install

```bash
pip install -e .
```

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) (system install)
- Whisper model downloads automatically on first run

## Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `base` | Whisper model: tiny, base, small, medium, large |
| `--threshold` | `0.35` | Pass/fail confidence cutoff |
| `--segments` | auto | Override number of sampled segments |
| `--language` | auto | Expected audio language (e.g. `en`, `pt`) |
| `--no-sync` | off | Skip ffsubsync drift check |
| `--keep-synced` | off | Save timing-corrected `.srt` to disk |
| `--json` | off | Output JSON instead of human-readable text |
| `--verbose` | off | Show subtitle and transcription text per segment |

## Exit codes

- `0` — confidence above threshold
- `1` — confidence below threshold
- `2` — error (missing dependency, bad input, etc.)

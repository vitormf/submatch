from __future__ import annotations
import json
import subprocess
import tempfile
from pathlib import Path


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


def extract_segment(video_path: Path, start_ms: int, duration_ms: int) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = Path(tmp.name)
    tmp.close()
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start_ms / 1_000),
            "-i", str(video_path),
            "-t", str(duration_ms / 1_000),
            "-ar", "16000",
            "-ac", "1",
            "-vn",
            str(out_path),
        ],
        capture_output=True,
        check=True,
    )
    return out_path

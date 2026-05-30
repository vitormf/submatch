from __future__ import annotations
import json
import os
import signal
import subprocess
from pathlib import Path


def list_subtitle_tracks(video: Path) -> list[dict]:
    """Return one dict per subtitle stream: {"index": int, "lang": str | None, "title": str | None}."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "s", str(video)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    return [
        {
            "index": i,
            "lang": s.get("tags", {}).get("language"),
            "title": s.get("tags", {}).get("title"),
        }
        for i, s in enumerate(streams)
    ]


def extract_subtitle_track(video: Path, index: int, dest: Path) -> None:
    """Extract subtitle stream at stream index `index` to `dest` as SRT."""
    cmd = ["ffmpeg", "-y", "-i", str(video), "-map", f"0:s:{index}", "-c:s", "srt", str(dest)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
    try:
        proc.communicate()
    except KeyboardInterrupt:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
        raise
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

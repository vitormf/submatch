from __future__ import annotations
from pathlib import Path

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m4v"}
SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".ass", ".ssa", ".sub"}


def find_pairs(directory: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for video in sorted(directory.iterdir()):
        if video.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        for sub in sorted(directory.iterdir()):
            if sub.suffix.lower() not in SUBTITLE_EXTENSIONS:
                continue
            if sub.stem == video.stem or sub.stem.startswith(video.stem + "."):
                pairs.append((video, sub))
    return pairs


def find_subtitle_candidates(subtitle_dir: Path) -> list[Path]:
    return sorted(
        p for p in subtitle_dir.iterdir()
        if p.suffix.lower() in SUBTITLE_EXTENSIONS
    )

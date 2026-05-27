from __future__ import annotations
from pathlib import Path

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m4v"}
SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".ass", ".ssa", ".sub"}


def find_pairs(directory: Path) -> list[tuple[Path, Path]]:
    videos = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    subtitles = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in SUBTITLE_EXTENSIONS
    )
    pairs: list[tuple[Path, Path]] = []
    for sub in subtitles:
        matches = [
            v for v in videos
            if sub.stem == v.stem or sub.stem.startswith(v.stem + ".")
        ]
        if matches:
            best = max(matches, key=lambda v: len(v.stem))
            pairs.append((best, sub))
    return sorted(pairs)


def find_subtitle_candidates(subtitle_dir: Path) -> list[Path]:
    return sorted(
        p for p in subtitle_dir.iterdir()
        if p.suffix.lower() in SUBTITLE_EXTENSIONS
    )

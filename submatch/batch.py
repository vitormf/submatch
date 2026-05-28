from __future__ import annotations
import fnmatch
import os
import re
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


def find_pairs_recursive(directory: Path) -> list[tuple[Path, Path]]:
    """Walk directory tree (symlinks not followed) and return all video/subtitle pairs."""
    all_pairs: list[tuple[Path, Path]] = []
    for dirpath, _dirnames, _filenames in os.walk(directory):
        all_pairs.extend(find_pairs(Path(dirpath)))
    return sorted(all_pairs)


def find_subtitle_candidates_recursive(subtitle_dir: Path) -> list[Path]:
    return sorted(
        p for p in subtitle_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUBTITLE_EXTENSIONS
    )


def classify_inputs(
    inputs: list[Path],
    recursive: bool = True,
) -> tuple[list[Path], list[Path]]:
    videos: list[Path] = []
    subtitles: list[Path] = []
    for path in inputs:
        if path.is_dir():
            if recursive:
                files = sorted(f for f in path.rglob("*") if f.is_file())
            else:
                files = sorted(f for f in path.iterdir() if f.is_file())
        else:
            files = [path]
        for f in files:
            if f.suffix.lower() in SUBTITLE_EXTENSIONS:
                subtitles.append(f)
            else:
                videos.append(f)
    return videos, subtitles


_LANG_TAG_RE = re.compile(r'^[a-zA-Z]{2,3}(-[a-zA-Z]{2,4})?$')


def _extract_lang_tag(path: Path) -> str | None:
    stem = path.stem
    if '.' not in stem:
        return None
    candidate = stem.rsplit('.', 1)[1]
    return candidate if _LANG_TAG_RE.match(candidate) else None


def _lang_matches(path: Path, codes: list[str]) -> bool:
    tag = _extract_lang_tag(path)
    if tag is None:
        try:
            from submatch import subtitle as _subtitle, language as _language
            subs = _subtitle.parse(path)
            text = ' '.join(s.text for s in subs[:50])
            tag = _language.detect_from_text(text)
        except Exception:
            return True
        if not tag:
            return True
    return any(tag.lower().startswith(c.lower()) for c in codes)


def filter_pairs(
    pairs: list[tuple[Path, Path]],
    sub_langs: list[str] | None = None,
    glob_pattern: str | None = None,
) -> list[tuple[Path, Path]]:
    result = []
    for video, sub in pairs:
        if glob_pattern and not fnmatch.fnmatch(sub.name, glob_pattern):
            continue
        if sub_langs and not _lang_matches(sub, sub_langs):
            continue
        result.append((video, sub))
    return result


def resolve_pairs(
    videos: list[Path],
    subtitles: list[Path],
) -> list[tuple[Path, Path]]:
    import sys
    pairs: list[tuple[Path, Path]] = []

    if videos:
        video_to_subs: dict[Path, list[Path]] = {}
        for sub in subtitles:
            matches = [
                v for v in videos
                if sub.stem == v.stem or sub.stem.startswith(v.stem + ".")
            ]
            if not matches:
                print(f"Warning: no matching video for subtitle: {sub.name}", file=sys.stderr)
                continue
            best = max(matches, key=lambda v: len(v.stem))
            video_to_subs.setdefault(best, []).append(sub)

        for video in videos:
            if video in video_to_subs:
                for sub in sorted(video_to_subs[video]):
                    pairs.append((video, sub))
            else:
                discovered = [s for v, s in find_pairs(video.parent) if v == video]
                if not discovered:
                    print(f"Warning: no subtitles found for video: {video.name}", file=sys.stderr)
                else:
                    pairs.extend((video, s) for s in discovered)
    else:
        for sub in subtitles:
            matched = [v for v, s in find_pairs(sub.parent) if s == sub]
            if not matched:
                print(f"Warning: no matching video for subtitle: {sub.name}", file=sys.stderr)
                continue
            pairs.append((matched[0], sub))

    return sorted(pairs)

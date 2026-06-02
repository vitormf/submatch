from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import pysubs2


@dataclass
class Subtitle:
    index: int
    start_ms: int
    end_ms: int
    text: str


_IMAGE_EXTENSIONS = frozenset({".sub", ".sup"})


def parse(path: Path) -> list[Subtitle]:
    """Parse a subtitle file into a list of Subtitle objects.

    Supports SRT, WebVTT, ASS/SSA, and any other format recognised by pysubs2.
    Styling tags are stripped; comment events are excluded.
    Returns an empty list for unrecognised or malformed files.
    """
    try:
        subs = pysubs2.load(str(path))
    except Exception:
        return []

    result = []
    for i, event in enumerate(subs):
        if event.type != "Dialogue":
            continue
        text = event.plaintext.strip()
        if not text:
            continue
        result.append(Subtitle(
            index=i + 1,
            start_ms=event.start,
            end_ms=event.end,
            text=text,
        ))
    return result


def is_image_based(path: Path) -> bool:
    """Return True if *path* is an image-based subtitle format (VOBSUB or PGS)."""
    return path.suffix.lower() in _IMAGE_EXTENSIONS

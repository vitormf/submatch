from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Subtitle:
    index: int
    start_ms: int
    end_ms: int
    text: str


_TIMESTAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


def _to_ms(h: str, m: str, s: str, ms: str) -> int:
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1_000 + int(ms)


def parse(path: Path) -> list[Subtitle]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    subtitles = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        match = _TIMESTAMP_RE.search(lines[1])
        if not match:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        start_ms = _to_ms(*match.group(1, 2, 3, 4))
        end_ms = _to_ms(*match.group(5, 6, 7, 8))
        text_lines = [ln.strip() for ln in lines[2:] if ln.strip()]
        subtitles.append(Subtitle(
            index=index,
            start_ms=start_ms,
            end_ms=end_ms,
            text=" ".join(text_lines),
        ))
    return subtitles

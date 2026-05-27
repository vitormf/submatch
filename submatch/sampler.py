from __future__ import annotations
from dataclasses import dataclass
from submatch.subtitle import Subtitle


@dataclass
class Segment:
    start_ms: int
    end_ms: int
    subtitle_text: str
    word_count: int


def auto_segment_count(duration_ms: int) -> int:
    minutes = duration_ms / 60_000
    if minutes < 30:
        return 5
    if minutes <= 90:
        return 8
    return 12


def select_segments(
    subtitles: list[Subtitle],
    duration_ms: int,
    n: int | None = None,
    window_ms: int = 30_000,
) -> list[Segment]:
    if n is None:
        n = auto_segment_count(duration_ms)

    margin_ms = int(duration_ms * 0.05)
    start_bound = margin_ms
    end_bound = duration_ms - margin_ms - window_ms

    if start_bound >= end_bound:
        start_bound = 0
        end_bound = max(0, duration_ms - window_ms)

    usable = max(1, end_bound - start_bound)
    zone_size = usable // n
    segments = []

    for zone_idx in range(n):
        zone_start = start_bound + zone_idx * zone_size
        zone_end = start_bound + (zone_idx + 1) * zone_size
        seg = _best_window_in_range(subtitles, zone_start, zone_end, window_ms)
        segments.append(seg)

    return sorted(segments, key=lambda s: s.start_ms)


def _best_window_in_range(
    subtitles: list[Subtitle],
    range_start: int,
    range_end: int,
    window_ms: int,
) -> Segment:
    candidates = [
        s.start_ms for s in subtitles
        if range_start <= s.start_ms <= range_end
    ]
    if not candidates:
        candidates = [range_start]

    best_start = range_start
    best_count = -1
    best_text = ""

    for win_start in candidates:
        win_end = win_start + window_ms
        window_subs = [
            s for s in subtitles
            if s.start_ms < win_end and s.end_ms > win_start
        ]
        word_count = sum(len(s.text.split()) for s in window_subs)
        if word_count > best_count:
            best_count = word_count
            best_start = win_start
            best_text = " ".join(s.text for s in window_subs)

    return Segment(
        start_ms=best_start,
        end_ms=best_start + window_ms,
        subtitle_text=best_text,
        word_count=max(0, best_count),
    )

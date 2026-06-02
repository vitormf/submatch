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


def segments_from_starts(
    subtitles: list[Subtitle],
    starts_ms: list[int],
    window_ms: int = 30_000,
) -> list[Segment]:
    """Build Segment objects at fixed video timestamps using text from a (re-synced) subtitle.

    Used when transcriptions are already cached: reconstructs the subtitle text that
    falls inside each pre-determined window without re-running segment selection.
    """
    result = []
    for start in starts_ms:
        end = start + window_ms
        words = [s for s in subtitles if s.start_ms < end and s.end_ms > start]
        text = " ".join(s.text for s in words)
        result.append(Segment(
            start_ms=start,
            end_ms=end,
            subtitle_text=text,
            word_count=len(text.split()) if text else 0,
        ))
    return result


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


def audio_candidate_segments(
    speech_regions: list[tuple[int, int]],
    duration_ms: int,
    n_zones: int,
    candidates_per_zone: int = 2,
    window_ms: int = 30_000,
    step_ms: int = 5_000,
) -> list[list[int]]:
    """Return up to candidates_per_zone candidate window start positions per zone.

    Zones cover [5%, 95%] of the video. Within each zone, windows are scored by
    overlap with speech_regions. Falls back to even spacing when speech_regions is empty.
    Returns a list of n_zones lists; each inner list has 1..candidates_per_zone start_ms values.
    """
    margin_ms = int(duration_ms * 0.05)
    usable_start = margin_ms
    usable_end = duration_ms - margin_ms
    usable = max(1, usable_end - usable_start)
    zone_size = usable // n_zones

    result: list[list[int]] = []

    for zone_idx in range(n_zones):
        zone_start = usable_start + zone_idx * zone_size
        zone_end = usable_start + (zone_idx + 1) * zone_size

        if not speech_regions:
            step = max(step_ms, zone_size // (candidates_per_zone + 1))
            fallback = []
            for i in range(1, candidates_per_zone + 1):
                start = zone_start + step * i
                if start + window_ms <= duration_ms:
                    fallback.append(start)
            if not fallback and zone_start + window_ms <= duration_ms:
                fallback = [zone_start]
            result.append(fallback)
            continue

        # Score every step_ms window that starts within the zone
        scored: list[tuple[int, int]] = []  # (overlap_ms, start_ms)
        pos = zone_start
        while pos <= zone_end and pos + window_ms <= duration_ms:
            win_end = pos + window_ms
            overlap = sum(
                min(win_end, r_end) - max(pos, r_start)
                for r_start, r_end in speech_regions
                if r_end > pos and r_start < win_end
            )
            scored.append((overlap, pos))
            pos += step_ms

        if not scored and zone_start + window_ms <= duration_ms:  # pragma: no cover
            scored = [(0, zone_start)]  # pragma: no cover

        scored.sort(reverse=True)

        candidates: list[int] = []
        for _, start in scored:
            if len(candidates) >= candidates_per_zone:
                break
            if not any(abs(start - c) < window_ms for c in candidates):
                candidates.append(start)

        result.append(candidates)

    return result

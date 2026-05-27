from submatch.sampler import auto_segment_count, select_segments, Segment
from submatch.subtitle import Subtitle


def _make_subtitles(n: int, duration_ms: int) -> list[Subtitle]:
    step = duration_ms // n
    return [
        Subtitle(
            index=i + 1,
            start_ms=i * step + 1_000,
            end_ms=i * step + 3_000,
            text=f"This is subtitle number {i + 1} with some dialogue words.",
        )
        for i in range(n)
    ]


def test_auto_segment_count_short():
    assert auto_segment_count(20 * 60 * 1_000) == 5


def test_auto_segment_count_medium():
    assert auto_segment_count(60 * 60 * 1_000) == 8


def test_auto_segment_count_long():
    assert auto_segment_count(120 * 60 * 1_000) == 12


def test_select_segments_returns_n():
    duration_ms = 90 * 60 * 1_000
    subs = _make_subtitles(60, duration_ms)
    result = select_segments(subs, duration_ms, n=8)
    assert len(result) == 8


def test_select_segments_sorted_by_start():
    duration_ms = 90 * 60 * 1_000
    subs = _make_subtitles(60, duration_ms)
    result = select_segments(subs, duration_ms, n=8)
    starts = [s.start_ms for s in result]
    assert starts == sorted(starts)


def test_select_segments_returns_segment_dataclasses():
    duration_ms = 90 * 60 * 1_000
    subs = _make_subtitles(60, duration_ms)
    result = select_segments(subs, duration_ms, n=5)
    assert all(isinstance(s, Segment) for s in result)


def test_select_segments_auto_n_uses_duration():
    duration_ms = 60 * 60 * 1_000  # 60 min → 8 segments
    subs = _make_subtitles(60, duration_ms)
    result = select_segments(subs, duration_ms, n=None)
    assert len(result) == 8


def test_select_segments_sparse_fallback():
    duration_ms = 90 * 60 * 1_000
    subs = _make_subtitles(2, duration_ms)
    result = select_segments(subs, duration_ms, n=5)
    assert len(result) == 5


def test_select_segments_very_short_video_boundary_correction():
    # duration < window triggers start_bound >= end_bound correction
    duration_ms = 20_000  # 20s, less than 30s window
    subs = _make_subtitles(2, duration_ms)
    result = select_segments(subs, duration_ms, n=2)
    assert len(result) == 2

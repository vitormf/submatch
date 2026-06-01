from submatch.sampler import auto_segment_count, select_segments, Segment, audio_candidate_segments
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


def test_audio_candidate_segments_empty_regions_falls_back_to_even_spacing():
    duration_ms = 600_000  # 10 minutes
    candidates = audio_candidate_segments([], duration_ms, n_zones=3, candidates_per_zone=2)
    assert len(candidates) == 3
    for zone in candidates:
        assert len(zone) >= 1
        for start in zone:
            assert 0 <= start
            assert start + 30_000 <= duration_ms


def test_audio_candidate_segments_picks_from_speech_regions():
    duration_ms = 600_000
    speech_regions = [(120_000, 480_000)]
    candidates = audio_candidate_segments(speech_regions, duration_ms, n_zones=3, candidates_per_zone=2)
    assert len(candidates) == 3
    for zone in candidates:
        assert len(zone) >= 1
        for start in zone:
            assert start + 30_000 <= duration_ms


def test_audio_candidate_segments_returns_two_per_zone():
    duration_ms = 3_600_000  # 60 minutes — wide enough for 2 non-overlapping windows per zone
    speech_regions = [(0, 3_600_000)]
    candidates = audio_candidate_segments(speech_regions, duration_ms, n_zones=3, candidates_per_zone=2)
    assert len(candidates) == 3
    for zone in candidates:
        # Each zone should have 2 candidates that don't overlap
        assert len(zone) == 2
        starts = sorted(zone)
        assert starts[1] - starts[0] >= 30_000


def test_audio_candidate_segments_candidates_within_video_bounds():
    duration_ms = 5_400_000  # 90 minutes
    speech_regions = [(300_000, 5_100_000)]
    candidates = audio_candidate_segments(speech_regions, duration_ms, n_zones=8, candidates_per_zone=2)
    for zone in candidates:
        for start in zone:
            assert start >= 0
            assert start + 30_000 <= duration_ms

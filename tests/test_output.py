import json
from submatch.output import (
    SegmentResult, MatchResult, format_json, print_human,
    _ms_to_ts, _bar,
)
from submatch.language import LanguageResult
from submatch.sync import SyncResult
from pathlib import Path


def _make_result() -> MatchResult:
    lang = LanguageResult(
        audio="en",
        subtitle_detected="en",
        subtitle_filename="en",
        video_metadata=None,
        expected=None,
        mismatch=False,
        mismatch_details=[],
    )
    seg = SegmentResult(
        index=1,
        start_ms=10_000,
        score=0.75,
        wer=0.3,
        subtitle_text="Hello world",
        transcription="hello world",
    )
    return MatchResult(
        confidence=0.75,
        passed=True,
        threshold=0.35,
        language=lang,
        sync=None,
        segments=[seg],
        model="base",
    )


def test_format_json_top_level_keys():
    data = json.loads(format_json(_make_result()))
    for key in ("confidence", "passed", "threshold", "language", "segments", "model"):
        assert key in data


def test_format_json_confidence_value():
    data = json.loads(format_json(_make_result()))
    assert data["confidence"] == 0.75
    assert data["passed"] is True


def test_format_json_segment_fields():
    data = json.loads(format_json(_make_result()))
    seg = data["segments"][0]
    for key in ("index", "start_ms", "score", "wer", "subtitle_text", "transcription"):
        assert key in seg


def test_format_json_language_fields():
    data = json.loads(format_json(_make_result()))
    lang = data["language"]
    for key in ("audio", "subtitle_detected", "subtitle_filename", "mismatch"):
        assert key in lang


def test_format_json_sync_none():
    data = json.loads(format_json(_make_result()))
    assert data["sync"] is None


def test_format_json_with_sync():
    result = _make_result()
    result.sync = SyncResult(
        synced_srt_path=Path("/tmp/synced.srt"),
        offset_seconds=5.0,
        drift_detected=True,
    )
    data = json.loads(format_json(result))
    assert data["sync"]["offset_seconds"] == 5.0
    assert data["sync"]["drift_detected"] is True
    assert data["sync"]["synced_srt_path"] == "/tmp/synced.srt"


# ── _ms_to_ts ────────────────────────────────────────────────────────────────

def test_ms_to_ts_zero():
    assert _ms_to_ts(0) == "00:00:00"


def test_ms_to_ts_one_hour():
    assert _ms_to_ts(3_600_000) == "01:00:00"


def test_ms_to_ts_mixed():
    assert _ms_to_ts(90_061_000) == "25:01:01"


# ── _bar ─────────────────────────────────────────────────────────────────────

def test_bar_full():
    assert _bar(1.0) == "█" * 10


def test_bar_empty():
    assert _bar(0.0) == "░" * 10


def test_bar_half():
    bar = _bar(0.5)
    assert bar.count("█") == 5
    assert bar.count("░") == 5


# ── print_human ───────────────────────────────────────────────────────────────

def _make_result_with_sync(*, drift: bool, offset: float) -> MatchResult:
    result = _make_result()
    result.sync = SyncResult(
        synced_srt_path=Path("/tmp/s.srt"),
        offset_seconds=offset,
        drift_detected=drift,
    )
    return result


def _make_result_with_mismatch() -> MatchResult:
    lang = LanguageResult(
        audio="en", subtitle_detected="pt", subtitle_filename=None,
        video_metadata=None, expected=None,
        mismatch=True, mismatch_details=["audio=en but subtitle text detected as pt"],
    )
    result = _make_result()
    result.language = lang
    return result


def test_print_human_no_sync(capsys):
    print_human(_make_result())
    assert "Skipped" in capsys.readouterr().out


def test_print_human_with_drift(capsys):
    print_human(_make_result_with_sync(drift=True, offset=23.4))
    out = capsys.readouterr().out
    assert "Drift detected" in out
    assert "23.4" in out


def test_print_human_no_drift(capsys):
    print_human(_make_result_with_sync(drift=False, offset=0.5))
    assert "No significant drift" in capsys.readouterr().out


def test_print_human_mismatch_warning(capsys):
    print_human(_make_result_with_mismatch())
    assert "⚠" in capsys.readouterr().out


def test_print_human_verbose_shows_texts(capsys):
    print_human(_make_result(), verbose=True)
    out = capsys.readouterr().out
    assert "subtitle:" in out
    assert "transcription:" in out


def test_print_human_passed_shows_checkmark(capsys):
    print_human(_make_result())
    assert "✓" in capsys.readouterr().out


def test_print_human_failed_shows_cross(capsys):
    result = _make_result()
    result.passed = False
    result.confidence = 0.1
    print_human(result)
    assert "✗" in capsys.readouterr().out


def test_print_human_shows_video_metadata_row(capsys):
    result = _make_result()
    result.language = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata="en", expected=None, mismatch=False, mismatch_details=[],
    )
    print_human(result)
    assert "Video metadata:" in capsys.readouterr().out

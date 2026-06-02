from submatch.types import MatchState, MatchResult, SegmentResult
from submatch.output import print_human, _ms_to_ts, _bar
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
    result = MatchResult(
        confidence=0.75,
        passed=True,
        threshold=0.35,
        language=lang,
        sync=None,
        segments=[seg],
        model="base",
    )
    result.state = MatchState.PASS
    return result


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
    assert "skipped" in capsys.readouterr().out


def test_print_human_with_drift(capsys):
    print_human(_make_result_with_sync(drift=True, offset=23.4))
    out = capsys.readouterr().out
    assert "23.4" in out
    assert "⚠" in out


def test_print_human_no_drift(capsys):
    print_human(_make_result_with_sync(drift=False, offset=0.5))
    assert "no drift" in capsys.readouterr().out


def test_print_human_mismatch_warning(capsys):
    print_human(_make_result_with_mismatch())
    assert "⚠" in capsys.readouterr().out


def test_print_human_verbose_shows_texts(capsys):
    print_human(_make_result(), verbose=True)
    out = capsys.readouterr().out
    assert "sub:" in out
    assert "asr:" in out


def test_print_human_passed_shows_checkmark(capsys):
    print_human(_make_result())
    assert "✓" in capsys.readouterr().out


def test_print_human_failed_shows_cross(capsys):
    result = _make_result()
    result.passed = False
    result.confidence = 0.1
    result.state = MatchState.FAIL
    print_human(result)
    assert "✗" in capsys.readouterr().out


def test_print_human_shows_video_metadata(capsys):
    result = _make_result()
    result.language = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata="fr", expected=None, mismatch=False, mismatch_details=[],
    )
    print_human(result)
    assert "meta=fr" in capsys.readouterr().out


# ── batch output ──────────────────────────────────────────────────────────────

from submatch.types import BatchPairResult  # noqa: E402
from submatch.output import print_batch_compact, print_batch_summary  # noqa: E402


def _make_batch_pairs() -> list[BatchPairResult]:
    passed_result = _make_result()  # already has state=MatchState.PASS
    passed = BatchPairResult(
        video=Path("show.mkv"),
        subtitle=Path("show.en.srt"),
        result=passed_result,
        error=None,
    )
    failed_result = _make_result()
    failed_result.passed = False
    failed_result.confidence = 0.10
    failed_result.state = MatchState.FAIL
    failed = BatchPairResult(
        video=Path("show.mkv"),
        subtitle=Path("show.pt.srt"),
        result=failed_result,
        error=None,
    )
    errored = BatchPairResult(
        video=Path("broken.mkv"),
        subtitle=Path("broken.srt"),
        result=None,
        error="no audio track",
    )
    return [passed, failed, errored]


def test_print_batch_compact_shows_pass(capsys):
    print_batch_compact(_make_batch_pairs())
    assert "PASS" in capsys.readouterr().out


def test_print_batch_compact_shows_fail(capsys):
    print_batch_compact(_make_batch_pairs())
    assert "FAIL" in capsys.readouterr().out


def test_print_batch_compact_shows_error(capsys):
    print_batch_compact(_make_batch_pairs())
    assert "ERROR" in capsys.readouterr().out


def test_print_batch_compact_shows_filenames(capsys):
    print_batch_compact(_make_batch_pairs())
    out = capsys.readouterr().out
    assert "show.en.srt" in out
    assert "show.pt.srt" in out


def test_print_batch_summary_counts(capsys):
    print_batch_summary(_make_batch_pairs())
    out = capsys.readouterr().out
    assert "1 PASS" in out
    assert "1 FAIL" in out
    assert "1 error" in out


def test_print_batch_compact_shows_score(capsys):
    print_batch_compact(_make_batch_pairs())
    out = capsys.readouterr().out
    assert "0.75" in out   # passed pair score
    assert "0.10" in out   # failed pair score


def test_print_batch_summary_empty(capsys):
    print_batch_summary([])
    assert "0 processed" in capsys.readouterr().out


# ── cross-language fields ─────────────────────────────────────────────────────

def test_print_human_cross_language_shows_header(capsys):
    result = _make_result()
    result.cross_language = True
    result.subtitle_language = "pt"
    result.language = LanguageResult(
        audio="en", subtitle_detected="pt", subtitle_filename="pt",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    print_human(result)
    out = capsys.readouterr().out
    assert "en→pt" in out or "en" in out
    assert "pt" in out


def test_print_human_same_language_no_cross_header(capsys):
    print_human(_make_result())
    out = capsys.readouterr().out
    assert "cross-language" not in out


# ── fmt_progress_result ───────────────────────────────────────────────────────

from submatch.output import fmt_progress_result  # noqa: E402


def test_fmt_progress_result_pass():
    line = fmt_progress_result(_make_result(), None, "movie.en.srt", 42.0)
    assert "PASS" in line
    assert "0.75" in line
    assert "movie.en.srt" in line
    assert "42s" in line


def test_fmt_progress_result_error():
    line = fmt_progress_result(None, "no audio track", "broken.srt", 5.0)
    assert "ERROR" in line
    assert "broken.srt" in line
    assert "5s" in line


def test_fmt_progress_result_fail():
    result = _make_result()
    result.passed = False
    result.confidence = 0.10
    result.state = MatchState.FAIL
    line = fmt_progress_result(result, None, "movie.pt.srt", 30.0)
    assert "FAIL" in line
    assert "0.10" in line


# ── audio_track fields ────────────────────────────────────────────────────────

def test_match_result_default_audio_track_fields():
    result = _make_result()
    assert result.audio_track_index == 0
    assert result.audio_track_lang is None


def test_print_human_omits_track_line_when_default(capsys):
    result = _make_result()
    print_human(result)
    out = capsys.readouterr().out
    assert "track" not in out


def test_print_human_shows_track_line_with_lang(capsys):
    result = _make_result()
    result.audio_track_index = 1
    result.audio_track_lang = "jpn"
    print_human(result)
    out = capsys.readouterr().out
    assert "track" in out
    assert "a:1" in out
    assert "jpn" in out


def test_print_human_shows_track_line_no_lang(capsys):
    result = _make_result()
    result.audio_track_index = 2
    result.audio_track_lang = None
    print_human(result)
    out = capsys.readouterr().out
    assert "track" in out
    assert "a:2" in out


def test_print_human_shows_track_line_when_lang_known_but_index_zero(capsys):
    """Show track line if lang is known even when index is 0."""
    result = _make_result()
    result.audio_track_index = 0
    result.audio_track_lang = "eng"
    print_human(result)
    out = capsys.readouterr().out
    assert "track" in out
    assert "a:0" in out
    assert "eng" in out


def test_print_human_shows_resynced_in_meta(capsys):
    """resynced=True should add 'resynced' to the meta line."""
    result = _make_result()
    result.resynced = True
    print_human(result)
    out = capsys.readouterr().out
    assert "resynced" in out

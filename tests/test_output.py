import json
from submatch.output import SegmentResult, MatchResult, format_json
from submatch.language import LanguageResult


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

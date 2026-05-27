from pathlib import Path
from unittest.mock import patch
from submatch.language import (
    detect_from_filename,
    detect_from_text,
    build_result,
    LanguageResult,
)


def test_detect_from_filename_two_letter_code():
    assert detect_from_filename(Path("movie.en.srt")) == "en"


def test_detect_from_filename_portuguese():
    assert detect_from_filename(Path("movie.pt.srt")) == "pt"


def test_detect_from_filename_long_name():
    assert detect_from_filename(Path("Movie.2020.BluRay.1080p.en.srt")) == "en"


def test_detect_from_filename_language_name():
    assert detect_from_filename(Path("movie.English.srt")) == "en"


def test_detect_from_filename_no_code():
    assert detect_from_filename(Path("movie.srt")) is None


def test_detect_from_text_calls_langdetect():
    with patch("submatch.language._langdetect") as mock:
        mock.return_value = "en"
        result = detect_from_text("Hello this is some English text")
    assert result == "en"
    mock.assert_called_once()


def test_detect_from_text_returns_none_on_error():
    with patch("submatch.language._langdetect", side_effect=Exception("fail")):
        result = detect_from_text("x")
    assert result is None


def test_build_result_no_mismatch():
    result = build_result(
        audio="en",
        subtitle_detected="en",
        subtitle_filename="en",
        video_meta=None,
        expected=None,
    )
    assert not result.mismatch
    assert result.mismatch_details == []


def test_build_result_detects_mismatch():
    result = build_result(
        audio="en",
        subtitle_detected="pt",
        subtitle_filename=None,
        video_meta=None,
        expected=None,
    )
    assert result.mismatch
    assert len(result.mismatch_details) >= 1


def test_build_result_expected_overrides():
    result = build_result(
        audio="en",
        subtitle_detected="en",
        subtitle_filename=None,
        video_meta=None,
        expected="pt",
    )
    assert result.mismatch

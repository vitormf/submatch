import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from submatch.language import (
    detect_from_filename,
    detect_from_text,
    detect_from_video,
    build_result,
    normalize_lang,
    to_tesseract_lang,
    _langdetect,
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


def test_build_result_subtitle_internal_mismatch():
    """Subtitle filename claims 'en' but text is detected as 'fr' → mismatch."""
    result = build_result(
        audio="fr",
        subtitle_detected="fr",
        subtitle_filename="en",
        video_meta=None,
        expected=None,
    )
    assert result.mismatch
    assert any("filename says en" in d for d in result.mismatch_details)


def test_build_result_cross_language_no_mismatch():
    """English audio with Portuguese subtitle is valid cross-language — no warning."""
    result = build_result(
        audio="en",
        subtitle_detected="pt",
        subtitle_filename="pt",
        video_meta=None,
        expected=None,
    )
    assert not result.mismatch
    assert result.mismatch_details == []


def test_build_result_expected_overrides():
    """Explicit expected language doesn't match subtitle → mismatch."""
    result = build_result(
        audio="en",
        subtitle_detected="en",
        subtitle_filename=None,
        video_meta=None,
        expected="pt",
    )
    assert result.mismatch


def test_build_result_video_meta_mismatch():
    """Whisper says audio=en but ffprobe says pt → mismatch."""
    result = build_result(
        audio="en",
        subtitle_detected=None,
        subtitle_filename=None,
        video_meta="pt",
        expected=None,
    )
    assert result.mismatch
    assert any("video metadata" in d for d in result.mismatch_details)


def test_build_result_subtitle_filename_only_no_mismatch():
    """Subtitle filename saying 'pt' with English audio is cross-language — no warning."""
    result = build_result(
        audio="en",
        subtitle_detected=None,
        subtitle_filename="pt",
        video_meta=None,
        expected=None,
    )
    assert not result.mismatch


def test_build_result_multiple_mismatches():
    """Subtitle filename vs detected text mismatch plus audio vs video_meta mismatch."""
    result = build_result(
        audio="en",
        subtitle_detected="pt",
        subtitle_filename="es",
        video_meta="fr",
        expected=None,
    )
    assert len(result.mismatch_details) == 2


def test_detect_from_video_returns_language(tmp_path):
    streams = json.dumps({"streams": [{"tags": {"language": "eng"}}]})
    mock_result = MagicMock(stdout=streams)
    with patch("submatch.language.subprocess.run", return_value=mock_result):
        assert detect_from_video(Path("video.mp4")) == "en"


def test_detect_from_video_three_letter_german():
    streams = json.dumps({"streams": [{"tags": {"language": "deu"}}]})
    with patch("submatch.language.subprocess.run", return_value=MagicMock(stdout=streams)):
        assert detect_from_video(Path("video.mp4")) == "de"


def test_detect_from_video_unknown_three_letter_passthrough():
    streams = json.dumps({"streams": [{"tags": {"language": "xyz"}}]})
    with patch("submatch.language.subprocess.run", return_value=MagicMock(stdout=streams)):
        assert detect_from_video(Path("video.mp4")) == "xyz"


def test_detect_from_filename_three_letter_code():
    assert detect_from_filename(Path("movie.eng.srt")) == "en"


def test_detect_from_filename_three_letter_german():
    assert detect_from_filename(Path("movie.deu.srt")) == "de"


def test_detect_from_filename_three_letter_french_b():
    assert detect_from_filename(Path("movie.fre.srt")) == "fr"


def test_normalize_lang_three_to_two():
    assert normalize_lang("eng") == "en"
    assert normalize_lang("por") == "pt"
    assert normalize_lang("deu") == "de"
    assert normalize_lang("ger") == "de"


def test_normalize_lang_two_letter_passthrough():
    assert normalize_lang("en") == "en"
    assert normalize_lang("pt") == "pt"


def test_normalize_lang_none():
    assert normalize_lang(None) is None


def test_normalize_lang_unknown_passthrough():
    assert normalize_lang("xyz") == "xyz"


def test_normalize_lang_uppercase():
    assert normalize_lang("ENG") == "en"
    assert normalize_lang("DEU") == "de"


def test_detect_from_video_und_returns_none():
    streams = json.dumps({"streams": [{"tags": {"language": "und"}}]})
    with patch("submatch.language.subprocess.run", return_value=MagicMock(stdout=streams)):
        assert detect_from_video(Path("video.mp4")) is None


def test_detect_from_video_no_streams():
    with patch("submatch.language.subprocess.run",
               return_value=MagicMock(stdout=json.dumps({"streams": []}))):
        assert detect_from_video(Path("video.mp4")) is None


def test_detect_from_video_exception_returns_none():
    with patch("submatch.language.subprocess.run", side_effect=Exception("fail")):
        assert detect_from_video(Path("video.mp4")) is None


def test_langdetect_delegates_to_langdetect_library():
    with patch("langdetect.detect", return_value="fr"):
        assert _langdetect("Bonjour le monde") == "fr"


def test_to_tesseract_lang_english():
    assert to_tesseract_lang("en") == "eng"


def test_to_tesseract_lang_portuguese():
    assert to_tesseract_lang("pt") == "por"


def test_to_tesseract_lang_japanese():
    assert to_tesseract_lang("ja") == "jpn"


def test_to_tesseract_lang_chinese():
    assert to_tesseract_lang("zh") == "chi_sim"


def test_to_tesseract_lang_unknown_falls_back_to_eng():
    assert to_tesseract_lang("xx") == "eng"


def test_to_tesseract_lang_three_letter_code():
    # Caller is expected to pass ISO 639-1; three-letter input is unknown → "eng"
    assert to_tesseract_lang("eng") == "eng"


def test_to_tesseract_lang_uppercase():
    assert to_tesseract_lang("EN") == "eng"

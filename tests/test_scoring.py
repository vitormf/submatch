from __future__ import annotations
from unittest.mock import MagicMock, patch


def test_scoring_functions_importable():
    from submatch.scoring import (
        _score_pair,
        _determine_state,
        _get_embed_model,
        _is_cross_language,
        _cache_config,
        _audio_driven_transcribe,
    )
    assert callable(_score_pair)
    assert callable(_determine_state)
    assert callable(_get_embed_model)
    assert callable(_is_cross_language)
    assert callable(_cache_config)
    assert callable(_audio_driven_transcribe)


def test_determine_state_no_segments():
    from submatch.scoring import _determine_state
    from submatch.types import MatchResult, MatchState
    from submatch.language import LanguageResult
    lang = LanguageResult(
        audio=None, subtitle_detected=None, subtitle_filename=None,
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    result = MatchResult(
        confidence=0.5, passed=True, threshold=0.35,
        language=lang, sync=None, segments=[], model="base",
    )
    assert _determine_state(result) == MatchState.UNSURE


def test_determine_state_failed():
    from submatch.scoring import _determine_state
    from submatch.types import MatchResult, MatchState
    from submatch.language import LanguageResult
    lang = LanguageResult(
        audio=None, subtitle_detected=None, subtitle_filename=None,
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    result = MatchResult(
        confidence=0.1, passed=False, threshold=0.35,
        language=lang, sync=None, segments=[MagicMock()], model="base",
    )
    assert _determine_state(result) == MatchState.FAIL


def test_is_cross_language_same():
    from submatch.scoring import _is_cross_language
    assert _is_cross_language("en", "en") is False


def test_is_cross_language_different():
    from submatch.scoring import _is_cross_language
    assert _is_cross_language("en", "pt") is True


def test_is_cross_language_none():
    from submatch.scoring import _is_cross_language
    assert _is_cross_language(None, "en") is False
    assert _is_cross_language("en", None) is False


def test_score_pair_cache_hit_uses_cached_transcriptions(tmp_path):
    """When _disk_hit is not None, transcription_pairs come from the cache (lines 281-287)."""
    from unittest.mock import patch, MagicMock
    from submatch.scoring import _score_pair
    from submatch.pipeline import PipelineConfig
    from submatch.cache import VideoCache
    from submatch.language import LanguageResult

    video = tmp_path / "movie.mkv"
    video.touch()
    sub = tmp_path / "movie.srt"
    sub.write_text("")

    cached = VideoCache(
        segment_starts=[60_000],
        transcriptions=["hello world cached text"],
        audio_lang="en",
        audio_track_index=0,
        audio_track_lang=None,
    )
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    config = PipelineConfig(
        model="base", threshold=0.35, segments=None, sync=False,
        language=None, verbose=False, audio_track=None,
        cross_threshold=None, resync=False,
        drift_threshold=2.0, use_cache=True,
        cache_ttl_days=30, cache_max_mb=200, cache_dir=tmp_path,
    )

    with patch("submatch.cache.load", return_value=cached), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=3_600_000), \
         patch("submatch.scoring.subtitle.parse", return_value=[]), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=[]), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang), \
         patch("submatch.scoring.transcribe.transcribe_segment") as mock_transcribe:
        result, cache_out = _score_pair(video, sub, config, MagicMock())

    mock_transcribe.assert_not_called()
    assert cache_out is cached


def test_score_pair_sync_runtime_error_keeps_fail(tmp_path):
    """When lazy sync triggers but ffs raises RuntimeError, keep the FAIL result (lines 348-350)."""
    from unittest.mock import patch, MagicMock
    from submatch.scoring import _score_pair
    from submatch.pipeline import PipelineConfig
    from submatch.sampler import Segment
    from submatch.subtitle import Subtitle
    from submatch.language import LanguageResult
    from submatch.types import MatchState

    video = tmp_path / "movie.mkv"
    video.touch()
    sub = tmp_path / "movie.srt"
    sub.write_text("")

    subs = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    mock_wav = MagicMock()
    mock_wav.unlink = MagicMock()
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    config = PipelineConfig(
        sync=True, segments=None, model="base", language=None,
        cross_threshold=None, threshold=0.35, resync=False, drift_threshold=2.0,
        device="cpu", use_cache=False,
    )

    with patch("submatch.scoring.audio.get_duration_ms", return_value=3_600_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=mock_wav), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.subtitle.parse", return_value=subs), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang), \
         patch("submatch.scoring.compare.aggregate", return_value=0.0), \
         patch("submatch.scoring.sync.sync_subtitle",
               side_effect=RuntimeError("ffs not available")):
        result, _ = _score_pair(video, sub, config, MagicMock())

    assert result.state == MatchState.FAIL


# ── _score_pair verbose output branches ──────────────────────────────────────


def _mock_base(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare):
    from submatch.sampler import Segment
    from submatch.subtitle import Subtitle
    from submatch.transcribe import TranscriptionResult
    from submatch.compare import SegmentScore
    mock_sub.parse.return_value = [Subtitle(index=1, start_ms=1000, end_ms=5000, text="hello")]
    mock_sub.is_image_based.return_value = False
    mock_lang.detect_from_filename.return_value = "en"
    mock_lang.detect_from_text.return_value = "en"
    mock_lang.detect_from_video.return_value = None
    mock_lang.build_result.return_value = MagicMock()
    mock_audio.get_duration_ms.return_value = 60_000
    mock_sampler.auto_segment_count.return_value = 1
    mock_sampler.select_segments.return_value = [
        Segment(start_ms=10_000, end_ms=40_000, subtitle_text="hello", word_count=1)
    ]
    mock_audio.extract_segment.return_value = MagicMock()
    mock_transcribe.transcribe_segment.return_value = TranscriptionResult(
        text="hello", language="en", no_speech_prob=0.1, avg_logprob=0.5
    )
    mock_compare.token_f1.return_value = SegmentScore(f1=0.9, wer=0.1, subtitle_tokens=1)
    mock_compare.aggregate.return_value = 0.9
    mock_compare.SegmentScore = SegmentScore


@patch("submatch.scoring.sync")
@patch("submatch.scoring.subtitle")
@patch("submatch.scoring.language")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.compare")
def test_score_pair_verbose_sync_error_prints_warning(
    mock_compare, mock_transcribe, mock_sampler, mock_audio, mock_lang, mock_sub, mock_sync,
    tmp_path, capsys
):
    # Covers scoring.py: verbose warning when lazy sync triggers (FAIL result) and ffs raises
    from submatch.scoring import _score_pair
    from submatch.pipeline import PipelineConfig
    _mock_base(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    # Lazy sync only triggers on FAIL; force aggregate to 0.0 so the first pass fails
    mock_compare.aggregate.return_value = 0.0
    mock_sync.sync_subtitle.side_effect = RuntimeError("ffsubsync crashed")

    video = tmp_path / "v.mkv"
    sub = tmp_path / "s.srt"
    sub.write_text("")
    config = PipelineConfig(sync=True, use_cache=False, verbose=True, device="cpu")
    _score_pair(video, sub, config, MagicMock())
    assert "ffsubsync crashed" in capsys.readouterr().err


@patch("submatch.scoring.sync")
@patch("submatch.scoring.subtitle")
@patch("submatch.scoring.language")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.compare")
def test_score_pair_verbose_segment_progress_and_final_newline(
    mock_compare, mock_transcribe, mock_sampler, mock_audio, mock_lang, mock_sub, mock_sync,
    tmp_path, capsys
):
    # Covers lines 188 (verbose segment progress) and 205 (final newline)
    from submatch.scoring import _score_pair
    from submatch.pipeline import PipelineConfig
    _mock_base(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)

    video = tmp_path / "v.mkv"
    sub = tmp_path / "s.srt"
    sub.write_text("")
    config = PipelineConfig(sync=False, use_cache=False, verbose=True, on_segment=None, device="cpu")
    _score_pair(video, sub, config, MagicMock())
    err = capsys.readouterr().err
    assert "[1/1]" in err


@patch("submatch.scoring.sync")
@patch("submatch.scoring.subtitle")
@patch("submatch.scoring.language")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.compare")
def test_score_pair_verbose_segment_exception_warning(
    mock_compare, mock_transcribe, mock_sampler, mock_audio, mock_lang, mock_sub, mock_sync,
    tmp_path, capsys
):
    # Covers lines 200-203: verbose warning when audio.extract_segment raises
    from submatch.scoring import _score_pair
    from submatch.pipeline import PipelineConfig
    _mock_base(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    mock_audio.extract_segment.side_effect = RuntimeError("audio extraction failed")

    video = tmp_path / "v.mkv"
    sub = tmp_path / "s.srt"
    sub.write_text("")
    config = PipelineConfig(sync=False, use_cache=False, verbose=True, on_segment=None, device="cpu")
    _score_pair(video, sub, config, MagicMock())
    assert "audio extraction failed" in capsys.readouterr().err


def test_resolve_ocr_lang_from_filename():
    from submatch.scoring import _resolve_ocr_lang
    from pathlib import Path
    from unittest.mock import patch

    with patch("submatch.scoring.language.detect_from_filename", return_value="pt"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None):
        result = _resolve_ocr_lang(Path("movie.pt.sub"), Path("movie.mkv"))
    assert result == "por"


def test_resolve_ocr_lang_from_video_metadata():
    from submatch.scoring import _resolve_ocr_lang
    from pathlib import Path
    from unittest.mock import patch

    with patch("submatch.scoring.language.detect_from_filename", return_value=None), \
         patch("submatch.scoring.language.detect_from_video", return_value="ja"):
        result = _resolve_ocr_lang(Path("movie.sub"), Path("movie.mkv"))
    assert result == "jpn"


def test_resolve_ocr_lang_returns_none_when_undetected():
    from submatch.scoring import _resolve_ocr_lang
    from pathlib import Path
    from unittest.mock import patch

    with patch("submatch.scoring.language.detect_from_filename", return_value=None), \
         patch("submatch.scoring.language.detect_from_video", return_value=None):
        result = _resolve_ocr_lang(Path("movie.sub"), Path("movie.mkv"))
    assert result is None


def test_score_pair_calls_ocr_for_image_subtitle(tmp_path):
    """When subtitle is image-based, ocr_window is called once per segment."""
    from unittest.mock import patch, MagicMock
    from submatch.scoring import _score_pair
    from submatch.pipeline import PipelineConfig
    from submatch.cache import VideoCache
    from submatch.language import LanguageResult

    video = tmp_path / "movie.mkv"
    video.touch()
    sub = tmp_path / "movie.sub"
    sub.touch()

    lang_result = LanguageResult(
        audio=None, subtitle_detected=None, subtitle_filename=None,
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    cached = VideoCache(
        segment_starts=[0, 30_000],
        transcriptions=["hello world", "goodbye"],
        audio_lang="en",
        audio_track_index=0,
        audio_track_lang=None,
    )

    with patch("submatch.scoring.subtitle.parse", return_value=[]), \
         patch("submatch.scoring.subtitle.is_image_based", return_value=True), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.detect_from_text", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang_result), \
         patch("submatch.scoring.ocr.pytesseract", new=MagicMock()), \
         patch("submatch.scoring.ocr.ocr_window", return_value="hello world") as mock_ocr:
        config = PipelineConfig(use_cache=False, sync=False)
        result, _ = _score_pair(video, sub, config, MagicMock(), video_cache=cached)

    assert mock_ocr.call_count == 2
    assert len(result.segments) > 0  # OCR text flowed through to segment scoring


def test_score_pair_ocr_exception_on_one_segment_continues(tmp_path):
    """If ocr_window raises for one segment, remaining segments are still processed."""
    from unittest.mock import patch, MagicMock
    from submatch.scoring import _score_pair
    from submatch.pipeline import PipelineConfig
    from submatch.cache import VideoCache
    from submatch.language import LanguageResult

    video = tmp_path / "movie.mkv"
    video.touch()
    sub = tmp_path / "movie.sub"
    sub.touch()

    lang_result = LanguageResult(
        audio=None, subtitle_detected=None, subtitle_filename=None,
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    cached = VideoCache(
        segment_starts=[0, 30_000],
        transcriptions=["hello", "world"],
        audio_lang="en",
        audio_track_index=0,
        audio_track_lang=None,
    )

    with patch("submatch.scoring.subtitle.parse", return_value=[]), \
         patch("submatch.scoring.subtitle.is_image_based", return_value=True), \
         patch("submatch.scoring.language.detect_from_filename", return_value=None), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.detect_from_text", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang_result), \
         patch("submatch.scoring.ocr.pytesseract", new=MagicMock()), \
         patch("submatch.scoring.ocr.ocr_window",
               side_effect=[RuntimeError("frame extraction failed"), "world text"]) as mock_ocr:
        config = PipelineConfig(use_cache=False, sync=False)
        _score_pair(video, sub, config, MagicMock(), video_cache=cached)

    # Both segments were attempted despite the first raising
    assert mock_ocr.call_count == 2


def test_score_pair_warns_when_pytesseract_missing(tmp_path, capsys):
    """When pytesseract is None, a warning is printed for image-based subtitles."""
    from unittest.mock import patch, MagicMock
    from submatch.scoring import _score_pair
    from submatch.pipeline import PipelineConfig
    from submatch.cache import VideoCache
    from submatch.language import LanguageResult

    video = tmp_path / "movie.mkv"
    video.touch()
    sub = tmp_path / "movie.sub"
    sub.touch()

    lang_result = LanguageResult(
        audio=None, subtitle_detected=None, subtitle_filename=None,
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    cached = VideoCache(
        segment_starts=[0],
        transcriptions=["hello"],
        audio_lang="en",
        audio_track_index=0,
        audio_track_lang=None,
    )

    with patch("submatch.scoring.subtitle.parse", return_value=[]), \
         patch("submatch.scoring.subtitle.is_image_based", return_value=True), \
         patch("submatch.scoring.language.detect_from_filename", return_value=None), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.detect_from_text", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang_result), \
         patch("submatch.scoring.ocr.pytesseract", new=None), \
         patch("submatch.scoring.ocr.ocr_window") as mock_ocr:
        config = PipelineConfig(use_cache=False, sync=False, verbose=True)
        _score_pair(video, sub, config, MagicMock(), video_cache=cached)

    mock_ocr.assert_not_called()
    captured = capsys.readouterr()
    assert "pytesseract" in captured.err


def test_score_pair_does_not_call_ocr_for_text_subtitle(tmp_path):
    """When subtitle is text-based (SRT), ocr_window must not be called."""
    from unittest.mock import patch, MagicMock
    from submatch.scoring import _score_pair
    from submatch.pipeline import PipelineConfig
    from submatch.cache import VideoCache
    from submatch.language import LanguageResult

    video = tmp_path / "movie.mkv"
    video.touch()
    sub = tmp_path / "movie.srt"
    sub.touch()

    lang_result = LanguageResult(
        audio=None, subtitle_detected=None, subtitle_filename=None,
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    cached = VideoCache(
        segment_starts=[0],
        transcriptions=["hello"],
        audio_lang="en",
        audio_track_index=0,
        audio_track_lang=None,
    )

    with patch("submatch.scoring.subtitle.parse", return_value=[]), \
         patch("submatch.scoring.subtitle.is_image_based", return_value=False), \
         patch("submatch.scoring.language.detect_from_filename", return_value=None), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.detect_from_text", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang_result), \
         patch("submatch.scoring.ocr.ocr_window") as mock_ocr:
        config = PipelineConfig(use_cache=False, sync=False)
        _score_pair(video, sub, config, MagicMock(), video_cache=cached)

    mock_ocr.assert_not_called()

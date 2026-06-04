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
        _audio_lang_from_votes,
    )
    assert callable(_score_pair)
    assert callable(_determine_state)
    assert callable(_get_embed_model)
    assert callable(_is_cross_language)
    assert callable(_cache_config)
    assert callable(_audio_driven_transcribe)
    assert callable(_audio_lang_from_votes)


def test_audio_lang_from_votes_exact_majority():
    # 6/12 votes for 'ja' — exactly 50% — should still win (plurality ≥ 50%).
    # Previously failed because the check was `top_count * 2 > total` (strict).
    from submatch.scoring import _audio_lang_from_votes
    votes = ["ja"] * 6 + ["en"] * 3 + ["ko"] * 1 + ["zh"] * 1 + ["en"] * 1
    assert _audio_lang_from_votes(votes) == "ja"


def test_audio_lang_from_votes_clear_majority():
    from submatch.scoring import _audio_lang_from_votes
    assert _audio_lang_from_votes(["ja"] * 7 + ["en"] * 3) == "ja"


def test_audio_lang_from_votes_no_majority():
    # Genuine tie: neither language has ≥ 50% without also tying → return None.
    from submatch.scoring import _audio_lang_from_votes
    assert _audio_lang_from_votes(["ja"] * 5 + ["en"] * 5) is None


def test_audio_lang_from_votes_empty():
    from submatch.scoring import _audio_lang_from_votes
    assert _audio_lang_from_votes([]) is None


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
        config = PipelineConfig(use_cache=False, sync=False, verbose=True)
        _score_pair(video, sub, config, MagicMock(), video_cache=cached)

    # Both segments were attempted despite the first raising
    assert mock_ocr.call_count == 2


def test_score_pair_exits_when_tesseract_missing(tmp_path, capsys):
    """When tesseract binary is missing, sys.exit(2) with install instructions."""
    import pytest
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
         patch("submatch.scoring.ocr.is_tesseract_available", return_value=False), \
         patch("submatch.scoring.ocr.ocr_window") as mock_ocr:
        config = PipelineConfig(use_cache=False, sync=False)
        with pytest.raises(SystemExit) as exc_info:
            _score_pair(video, sub, config, MagicMock(), video_cache=cached)

    assert exc_info.value.code == 2
    mock_ocr.assert_not_called()
    captured = capsys.readouterr()
    assert "Tesseract" in captured.err
    assert "tesseract-ocr.github.io" in captured.err


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


def test_default_cross_threshold_constant():
    from submatch.scoring import _DEFAULT_CROSS_THRESHOLD
    assert _DEFAULT_CROSS_THRESHOLD == 0.20


def test_build_match_result_cross_language_uses_default_threshold():
    """When cross_threshold is None and audio/subtitle differ, effective threshold is 0.20."""
    from unittest.mock import MagicMock, patch
    from pathlib import Path
    from submatch.scoring import _build_match_result
    from submatch.pipeline import PipelineConfig
    from submatch.sampler import Segment
    from submatch.language import LanguageResult
    from submatch.compare import SegmentScore

    lang = LanguageResult(
        audio="ja", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    seg = Segment(60_000, 90_000, "hello world", 2)
    # confidence=0.22 sits above 0.20 (new cross-language default) but below 0.35 (old fallback)
    config = PipelineConfig(model="base", threshold=0.35, cross_threshold=None)

    from submatch.scoring import TranscriptionEntry
    with patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang), \
         patch("submatch.scoring.compare.aggregate", return_value=0.22), \
         patch("submatch.scoring.embeddings.cross_language_score",
               return_value=SegmentScore(f1=0.22, wer=0.78, subtitle_tokens=2)), \
         patch("submatch.scoring._get_embed_model", return_value=MagicMock()):
        result = _build_match_result(
            transcription_pairs=[TranscriptionEntry(index=1, segment=seg, text="こんにちは 世界", audio_language="ja")],
            subtitle_sample="hello world",
            subtitle_lang="en",
            audio_lang="ja",
            subtitle_path=Path("movie.en.srt"),
            video=Path("movie.mkv"),
            config=config,
            audio_track_index=0,
            audio_track_lang=None,
        )

    assert result.threshold == 0.20
    assert result.passed is True


def test_build_match_result_same_language_still_uses_threshold():
    """When same-language, threshold stays at config.threshold (0.35)."""
    from unittest.mock import patch
    from pathlib import Path
    from submatch.scoring import _build_match_result
    from submatch.pipeline import PipelineConfig
    from submatch.sampler import Segment
    from submatch.language import LanguageResult
    from submatch.compare import SegmentScore

    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    seg = Segment(60_000, 90_000, "hello world", 2)
    config = PipelineConfig(model="base", threshold=0.35, cross_threshold=None)

    from submatch.scoring import TranscriptionEntry
    with patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang), \
         patch("submatch.scoring.compare.aggregate", return_value=0.22), \
         patch("submatch.scoring.compare.token_f1",
               return_value=SegmentScore(f1=0.22, wer=0.78, subtitle_tokens=2)):
        result = _build_match_result(
            transcription_pairs=[TranscriptionEntry(index=1, segment=seg, text="hello world", audio_language="en")],
            subtitle_sample="hello world",
            subtitle_lang="en",
            audio_lang="en",
            subtitle_path=Path("movie.en.srt"),
            video=Path("movie.mkv"),
            config=config,
            audio_track_index=0,
            audio_track_lang=None,
        )

    assert result.threshold == 0.35
    assert result.passed is False


def test_build_match_result_explicit_cross_threshold_overrides_default():
    """When cross_threshold is explicitly set, it overrides the 0.20 default."""
    from unittest.mock import MagicMock, patch
    from pathlib import Path
    from submatch.scoring import _build_match_result
    from submatch.pipeline import PipelineConfig
    from submatch.sampler import Segment
    from submatch.language import LanguageResult
    from submatch.compare import SegmentScore

    lang = LanguageResult(
        audio="ja", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    seg = Segment(60_000, 90_000, "hello world", 2)
    config = PipelineConfig(model="base", threshold=0.35, cross_threshold=0.30)

    from submatch.scoring import TranscriptionEntry
    with patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang), \
         patch("submatch.scoring.compare.aggregate", return_value=0.22), \
         patch("submatch.scoring.embeddings.cross_language_score",
               return_value=SegmentScore(f1=0.22, wer=0.78, subtitle_tokens=2)), \
         patch("submatch.scoring._get_embed_model", return_value=MagicMock()):
        result = _build_match_result(
            transcription_pairs=[TranscriptionEntry(index=1, segment=seg, text="こんにちは 世界", audio_language="ja")],
            subtitle_sample="hello world",
            subtitle_lang="en",
            audio_lang="ja",
            subtitle_path=Path("movie.en.srt"),
            video=Path("movie.mkv"),
            config=config,
            audio_track_index=0,
            audio_track_lang=None,
        )

    assert result.threshold == 0.30
    assert result.passed is False  # 0.22 < 0.30


def test_gather_transcriptions_importable():
    from submatch.scoring import _gather_transcriptions
    assert callable(_gather_transcriptions)


def test_audio_driven_transcribe_uses_audio_track_duration_for_candidates(tmp_path):
    """Regression for PYTHON-C: when audio track is shorter than the container format,
    candidate positions must be bounded by the audio track duration, not the format duration.
    Without this fix, candidates near the end of the format duration fall past the end of
    the audio track, causing ffmpeg CalledProcessError on every candidate in those zones."""
    from submatch.scoring import _audio_driven_transcribe
    from submatch.pipeline import PipelineConfig
    from submatch.transcribe import TranscriptionResult
    from unittest.mock import patch, MagicMock

    video = tmp_path / "v.mkv"
    video.touch()

    # Simulate a 2-hour video where the audio track ends at 40 minutes
    format_duration_ms = 7_200_000   # 2 hours
    audio_track_duration_ms = 2_400_000  # 40 minutes

    good_trans = TranscriptionResult(text="hello world", language="en", no_speech_prob=0.1, avg_logprob=0.0)
    mock_wav = MagicMock()
    mock_wav.unlink = MagicMock()

    config = PipelineConfig(use_cache=True, sync=False, device="cpu")

    with patch("submatch.scoring.audio.get_duration_ms", return_value=format_duration_ms), \
         patch("submatch.scoring.audio.get_audio_track_duration_ms", return_value=audio_track_duration_ms) as mock_atd, \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.audio.extract_segment", return_value=mock_wav) as mock_extract, \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=good_trans):
        starts, _, _, _ = _audio_driven_transcribe(
            video, audio_track_index=0, n_seg=12, model=MagicMock(),
            config=config, duration_ms=format_duration_ms,
        )

    # All candidate start positions must be within the audio track duration
    for call_args in mock_extract.call_args_list:
        start_ms = call_args[0][1]  # positional arg: start_ms
        assert start_ms + 30_000 <= audio_track_duration_ms, (
            f"Candidate at {start_ms}ms extends past audio track end "
            f"({audio_track_duration_ms}ms): would cause ffmpeg CalledProcessError"
        )

    mock_atd.assert_called_once_with(video, 0)


def test_audio_driven_transcribe_silent_segments_do_not_vote(tmp_path, monkeypatch):
    """When most zones are silent (no accepted segment), those zones cast no vote,
    so the final audio_lang reflects only the zones with real speech."""
    from pathlib import Path
    from submatch.scoring import _audio_driven_transcribe
    from submatch import transcribe as _transcribe, audio as _audio, sampler as _sampler
    from submatch.pipeline import PipelineConfig

    call_count = [0]
    def fake_transcribe(model, wav_path):
        i = call_count[0]
        call_count[0] += 1
        r = _transcribe.TranscriptionResult(text="", language="en", no_speech_prob=0.95, avg_logprob=-2.0)
        if i == 0:
            r = _transcribe.TranscriptionResult(
                text="안녕하세요 반갑습니다 잘 있었어요",
                language="ko",
                no_speech_prob=0.1,
                avg_logprob=-0.5,
            )
        return r

    fake_wav = tmp_path / "seg.wav"
    fake_wav.write_bytes(b"RIFF")

    monkeypatch.setattr(_transcribe, "transcribe_segment", fake_transcribe)
    monkeypatch.setattr(_audio, "detect_speech_regions", lambda *a, **kw: [])
    monkeypatch.setattr(_audio, "get_audio_track_duration_ms", lambda *a, **kw: 3_600_000)
    monkeypatch.setattr(_audio, "extract_segment", lambda *a, **kw: fake_wav)
    monkeypatch.setattr(_sampler, "audio_candidate_segments", lambda *a, **kw: [
        [0], [300_000], [600_000], [900_000], [1_200_000]
    ])

    config = PipelineConfig(model="base", verbose=False)
    _, _, audio_lang, _ = _audio_driven_transcribe(Path("/fake/movie.mp4"), 0, 5, None, config, duration_ms=3_600_000)
    assert audio_lang == "ko"


def test_build_match_result_per_segment_cross_language(monkeypatch):
    """Each segment uses its own audio_language for cross/same scoring decision."""
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from submatch.scoring import _build_match_result, TranscriptionEntry
    from submatch import language as _language, embeddings as _emb, compare as _cmp
    from submatch.pipeline import PipelineConfig

    def _seg(text):
        s = MagicMock()
        s.subtitle_text = text
        s.start_ms = 0
        s.word_count = len(text.split())
        return s

    entries = [
        TranscriptionEntry(index=1, segment=_seg("hello world"), text="hello world", audio_language="en"),
        TranscriptionEntry(index=2, segment=_seg("안녕하세요 반갑습니다"), text="안녕하세요", audio_language="ko"),
    ]

    embed_calls = []
    f1_calls = []

    def fake_cross(sub, asr, model):
        embed_calls.append((sub, asr))
        r = MagicMock(); r.f1 = 0.8; r.wer = 0.2
        return r

    def fake_f1(sub, asr):
        f1_calls.append((sub, asr))
        r = MagicMock(); r.f1 = 0.9; r.wer = 0.1
        return r

    config = PipelineConfig(model="base", verbose=False)

    with patch.object(_emb, "cross_language_score", side_effect=fake_cross), \
         patch.object(_cmp, "token_f1", side_effect=fake_f1), \
         patch("submatch.scoring._get_embed_model", return_value=MagicMock()), \
         patch.object(_language, "detect_from_text", return_value="en"), \
         patch.object(_language, "detect_from_filename", return_value="en"), \
         patch.object(_language, "detect_from_video", return_value=None):
        result = _build_match_result(
            entries,
            subtitle_sample="hello world",
            subtitle_lang="en",
            audio_lang=None,
            subtitle_path=Path("sub.en.srt"),
            video=Path("video.mp4"),
            config=config,
            audio_track_index=0,
            audio_track_lang=None,
        )

    assert len(f1_calls) == 1, "English segment should use token_f1"
    assert len(embed_calls) == 1, "Korean segment should use embeddings"
    assert result.segments[0].audio_language == "en"
    assert result.segments[1].audio_language == "ko"
    assert result.cross_language is True


def test_no_cache_path_silent_segments_do_not_vote(tmp_path, monkeypatch):
    """In --no-cache mode, silent segments should not vote for audio language."""
    from pathlib import Path
    from submatch.scoring import _gather_transcriptions
    from submatch import transcribe as _transcribe, audio as _audio, sampler as _sampler, subtitle as _subtitle
    from submatch.pipeline import PipelineConfig

    call_count = [0]
    def fake_transcribe(model, wav_path):
        i = call_count[0]
        call_count[0] += 1
        if i == 0:
            return _transcribe.TranscriptionResult(
                text="안녕하세요 반갑습니다 잘 있었어요",
                language="ko", no_speech_prob=0.1, avg_logprob=-0.5,
            )
        return _transcribe.TranscriptionResult(
            text="", language="en", no_speech_prob=0.95, avg_logprob=-2.0,
        )

    fake_wav = tmp_path / "seg.wav"
    fake_wav.write_bytes(b"RIFF")

    from unittest.mock import MagicMock
    monkeypatch.setattr(_transcribe, "transcribe_segment", fake_transcribe)
    monkeypatch.setattr(_audio, "get_duration_ms", lambda *a, **kw: 3_600_000)
    monkeypatch.setattr(_audio, "extract_segment", lambda *a, **kw: fake_wav)

    fake_segs = [MagicMock(start_ms=i * 300_000, word_count=3, subtitle_text="test") for i in range(5)]
    monkeypatch.setattr(_sampler, "select_segments", lambda *a, **kw: fake_segs)

    config = PipelineConfig(model="base", verbose=False, use_cache=False, segments=5)
    entries, _, audio_lang = _gather_transcriptions(
        video=Path("/fake/movie.mp4"),
        subtitles=[],
        audio_track_index=0,
        audio_track_lang=None,
        config=config,
        model=None,
    )
    assert audio_lang == "ko"


def test_gather_transcriptions_uses_existing_cache(tmp_path):
    """When video_cache is provided, returns its data without touching audio."""
    from submatch.scoring import _gather_transcriptions
    from submatch.cache import VideoCache
    from submatch.sampler import Segment
    from unittest.mock import patch, MagicMock

    video = tmp_path / "v.mkv"
    video.touch()

    cache = VideoCache(
        segment_starts=[60_000],
        transcriptions=["hello from cache"],
        audio_lang="en",
        audio_track_index=0,
        audio_track_lang=None,
    )

    with patch("submatch.scoring.sampler.segments_from_starts",
               return_value=[Segment(60_000, 90_000, "hello from cache", 3)]), \
         patch("submatch.scoring.audio.get_duration_ms") as mock_dur:
        pairs, out_cache, audio_lang = _gather_transcriptions(
            video, [], 0, None, MagicMock(), MagicMock(), video_cache=cache
        )

    mock_dur.assert_not_called()
    assert out_cache is cache
    assert audio_lang == "en"
    assert len(pairs) == 1
    assert pairs[0].text == "hello from cache"

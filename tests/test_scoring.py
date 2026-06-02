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


# ── _score_pair verbose output branches ──────────────────────────────────────


def _mock_base(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare):
    from submatch.sampler import Segment
    from submatch.subtitle import Subtitle
    from submatch.transcribe import TranscriptionResult
    from submatch.compare import SegmentScore
    mock_sub.parse.return_value = [Subtitle(index=1, start_ms=1000, end_ms=5000, text="hello")]
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
    # Covers line 171: verbose warning when sync.sync_subtitle raises RuntimeError
    from submatch.scoring import _score_pair
    from submatch.pipeline import PipelineConfig
    _mock_base(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
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

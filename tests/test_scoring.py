from __future__ import annotations
from unittest.mock import MagicMock


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

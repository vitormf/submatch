"""
Integration tests — require network (first run only), ffmpeg, and openai-whisper.

Assets are downloaded to tests/fixtures/ and cached between runs.
Run with: make integration-test
"""
import pytest
from pathlib import Path

from submatch import audio, compare, sampler, subtitle, transcribe

pytestmark = pytest.mark.integration


def _score(video: Path, subtitle: Path, model, n: int = 2) -> tuple[float, str]:
    """Run pipeline and return (confidence, audio_language)."""
    duration_ms = audio.get_duration_ms(video)
    subtitles = subtitle.parse(subtitle)
    segments = sampler.select_segments(subtitles, duration_ms, n=n)

    scores: list[compare.SegmentScore] = []
    audio_lang: str | None = None

    for i, seg in enumerate(segments):
        wav = audio.extract_segment(video, seg.start_ms, 30_000)
        try:
            trans = transcribe.transcribe_segment(model, wav)
            if i == 0:
                audio_lang = trans.language
            scores.append(compare.token_f1(seg.subtitle_text, trans.text))
        finally:
            wav.unlink(missing_ok=True)

    return compare.aggregate(scores), audio_lang or "unknown"


def test_matching_subtitle_passes_threshold(
    made_in_america_video, made_in_america_srt, whisper_tiny
):
    """Correct subtitle for this video should score above 0.25 with the tiny model."""
    confidence, _ = _score(made_in_america_video, made_in_america_srt, whisper_tiny)
    assert confidence >= 0.25, (
        f"Matching subtitle scored {confidence:.2f} (tiny model), expected >= 0.25"
    )


def test_mismatched_subtitle_scores_lower_than_matching(
    made_in_america_video, made_in_america_srt, nasa_venus_srt, whisper_tiny
):
    """NASA Venus subtitle (planets/chemistry) should score lower than the correct one."""
    matching, _ = _score(made_in_america_video, made_in_america_srt, whisper_tiny)
    mismatched, _ = _score(made_in_america_video, nasa_venus_srt, whisper_tiny)
    assert mismatched < matching, (
        f"Mismatch score ({mismatched:.2f}) should be lower than match ({matching:.2f})"
    )


def test_audio_language_detected_as_english(
    made_in_america_video, made_in_america_srt, whisper_tiny
):
    _, lang = _score(made_in_america_video, made_in_america_srt, whisper_tiny, n=1)
    assert lang == "en", f"Expected audio language 'en', got '{lang}'"


def test_mismatched_subtitle_below_default_threshold(
    made_in_america_video, nasa_venus_srt, whisper_tiny
):
    """A clearly wrong subtitle should fall below the default 0.35 threshold."""
    confidence, _ = _score(made_in_america_video, nasa_venus_srt, whisper_tiny)
    assert confidence < 0.35, (
        f"Mismatched subtitle scored {confidence:.2f}, expected < 0.35"
    )

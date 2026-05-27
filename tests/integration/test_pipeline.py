"""
Integration tests — require network (first run only), ffmpeg, and openai-whisper.

Assets are downloaded to tests/fixtures/ and cached between runs.
Run with: make integration-test

Videos: WIKITONGUES project on Wikimedia Commons (CC BY-SA 4.0 / CC BY 3.0).
  - Gereon speaking German  — German audio, subtitles in de/en/pt-br
  - María speaking Guarani  — Guarani audio, subtitles in gn/en/es/de
"""
import pytest
from pathlib import Path

from submatch import audio, compare, embeddings, sampler, subtitle, transcribe

pytestmark = pytest.mark.integration


def _score(video: Path, subtitle_path: Path, model, n: int = 2) -> tuple[float, str]:
    """Run the same-language pipeline and return (confidence, audio_language)."""
    duration_ms = audio.get_duration_ms(video)
    subs = subtitle.parse(subtitle_path)
    segments = sampler.select_segments(subs, duration_ms, n=n)

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


def _score_cross_language(
    video: Path, subtitle_path: Path, whisper_model, embed_model, n: int = 2,
) -> tuple[float, str]:
    """Run the cross-language pipeline and return (confidence, audio_language)."""
    duration_ms = audio.get_duration_ms(video)
    subs = subtitle.parse(subtitle_path)
    segments = sampler.select_segments(subs, duration_ms, n=n)

    scores: list[compare.SegmentScore] = []
    audio_lang: str | None = None

    for i, seg in enumerate(segments):
        wav = audio.extract_segment(video, seg.start_ms, 30_000)
        try:
            trans = transcribe.transcribe_segment(whisper_model, wav)
            if i == 0:
                audio_lang = trans.language
            scores.append(
                embeddings.cross_language_score(seg.subtitle_text, trans.text, embed_model)
            )
        finally:
            wav.unlink(missing_ok=True)

    return compare.aggregate(scores), audio_lang or "unknown"


# ── German video — same-language tests ───────────────────────────────────────

def test_german_native_subtitle_passes_threshold(
    german_video, german_de_srt, whisper_tiny,
):
    """German subtitle for German audio should score above 0.25 with tiny model."""
    confidence, _ = _score(german_video, german_de_srt, whisper_tiny)
    assert confidence >= 0.25, (
        f"Native German subtitle scored {confidence:.2f} (tiny model), expected >= 0.25"
    )


def test_german_audio_detected_as_german(german_video, german_de_srt, whisper_tiny):
    """Whisper should identify the audio language as German."""
    _, lang = _score(german_video, german_de_srt, whisper_tiny, n=1)
    assert lang == "de", f"Expected audio language 'de', got '{lang}'"


def test_german_mismatched_subtitle_scores_lower(
    german_video, german_de_srt, guarani_de_srt, whisper_tiny,
):
    """German subtitle about Paraguay should score lower than the native German subtitle."""
    matching, _ = _score(german_video, german_de_srt, whisper_tiny)
    mismatched, _ = _score(german_video, guarani_de_srt, whisper_tiny)
    assert mismatched < matching, (
        f"Mismatch ({mismatched:.2f}) should be lower than match ({matching:.2f})"
    )


# ── German video — subtitle format tests ─────────────────────────────────────

def test_german_vtt_subtitle_passes_threshold(german_video, german_de_vtt, whisper_tiny):
    """WebVTT German subtitle should score as well as the SRT equivalent."""
    confidence, _ = _score(german_video, german_de_vtt, whisper_tiny)
    assert confidence >= 0.25, (
        f"VTT subtitle scored {confidence:.2f} (tiny model), expected >= 0.25"
    )


def test_german_ass_subtitle_passes_threshold(german_video, german_de_ass, whisper_tiny):
    """ASS subtitle (converted from SRT) should score above 0.25 with tiny model."""
    confidence, _ = _score(german_video, german_de_ass, whisper_tiny)
    assert confidence >= 0.25, (
        f"ASS subtitle scored {confidence:.2f} (tiny model), expected >= 0.25"
    )


# ── German video — cross-language tests ──────────────────────────────────────

def test_cross_language_german_english_passes_threshold(
    german_video, german_en_srt, whisper_tiny, embed_model,
):
    """English translation of German audio should score above 0.10 via embeddings."""
    confidence, _ = _score_cross_language(german_video, german_en_srt, whisper_tiny, embed_model)
    assert confidence >= 0.10, (
        f"DE audio + EN subtitle scored {confidence:.2f} (tiny model), expected >= 0.10"
    )


def test_cross_language_german_ptbr_passes_threshold(
    german_video, german_ptbr_srt, whisper_tiny, embed_model,
):
    """Brazilian Portuguese translation of German audio should score above 0.10 via embeddings."""
    confidence, _ = _score_cross_language(
        german_video, german_ptbr_srt, whisper_tiny, embed_model,
    )
    assert confidence >= 0.10, (
        f"DE audio + PT-BR subtitle scored {confidence:.2f} (tiny model), expected >= 0.10"
    )


def test_cross_language_matching_translation_scores_higher_than_mismatch(
    german_video, german_en_srt, guarani_en_srt, whisper_tiny, embed_model,
):
    """Correct English translation of German audio should outscore an unrelated English subtitle."""
    matching, _ = _score_cross_language(
        german_video, german_en_srt, whisper_tiny, embed_model,
    )
    mismatched, _ = _score_cross_language(
        german_video, guarani_en_srt, whisper_tiny, embed_model,
    )
    assert mismatched < matching, (
        f"Mismatch ({mismatched:.2f}) should be lower than match ({matching:.2f})"
    )


def test_cross_language_audio_detected_as_german(
    german_video, german_en_srt, whisper_tiny, embed_model,
):
    """Audio language detection should return German even when running the cross-language path."""
    _, lang = _score_cross_language(
        german_video, german_en_srt, whisper_tiny, embed_model, n=1,
    )
    assert lang == "de", f"Expected audio language 'de', got '{lang}'"


# ── Guarani video — cross-language tests ─────────────────────────────────────
# Guarani has limited Whisper support; scores may be low. These tests use relative
# comparisons and conservative thresholds to stay robust across model versions.

def test_guarani_spanish_subtitle_scores_higher_than_german_mismatch(
    guarani_video, guarani_es_srt, german_de_srt, whisper_tiny, embed_model,
):
    """Spanish translation of Guarani audio should outscore an unrelated German subtitle."""
    matching, _ = _score_cross_language(
        guarani_video, guarani_es_srt, whisper_tiny, embed_model,
    )
    mismatched, _ = _score_cross_language(
        guarani_video, german_de_srt, whisper_tiny, embed_model,
    )
    assert mismatched < matching, (
        f"Mismatch ({mismatched:.2f}) should be lower than match ({matching:.2f})"
    )


def test_guarani_english_subtitle_scores_higher_than_german_mismatch(
    guarani_video, guarani_en_srt, german_de_srt, whisper_tiny, embed_model,
):
    """English translation of Guarani audio should outscore an unrelated German subtitle."""
    matching, _ = _score_cross_language(
        guarani_video, guarani_en_srt, whisper_tiny, embed_model,
    )
    mismatched, _ = _score_cross_language(
        guarani_video, german_de_srt, whisper_tiny, embed_model,
    )
    assert mismatched < matching, (
        f"Mismatch ({mismatched:.2f}) should be lower than match ({matching:.2f})"
    )

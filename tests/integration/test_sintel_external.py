"""
Integration tests: Sintel 720p (English audio) against external subtitle SRTs.

Covers subtitle language scoring for languages not covered by Wikitongues fixtures.
All tests are cross-language: English audio vs. translated subtitle text.

Source: Sintel (Blender Foundation, CC BY 3.0), external SRTs from Wikimedia TimedText.
"""
import pytest
from pathlib import Path

from submatch import audio, compare, embeddings, sampler, subtitle, transcribe

pytestmark = pytest.mark.integration


def _score_cross_language(
    video: Path, subtitle_path: Path, whisper_model, embed_model, n: int = 2,
) -> float:
    """Run the cross-language pipeline and return confidence score."""
    duration_ms = audio.get_duration_ms(video)
    subs = subtitle.parse(subtitle_path)
    segments = sampler.select_segments(subs, duration_ms, n=n)

    scores: list[compare.SegmentScore] = []
    for seg in segments:
        wav = audio.extract_segment(video, seg.start_ms, 30_000)
        try:
            trans = transcribe.transcribe_segment(whisper_model, wav)
            scores.append(
                embeddings.cross_language_score(seg.subtitle_text, trans.text, embed_model)
            )
        finally:
            wav.unlink(missing_ok=True)

    return compare.aggregate(scores)


# Each test uses sintel_720p_video (English audio) + external SRT in the target language.
# Threshold 0.10 — conservative for cross-language embedding similarity.

def test_sintel_ja_cross_language(sintel_720p_video, sintel_ja_srt, whisper_tiny, embed_model):
    """Japanese subtitle against Sintel English audio: cross-language score >= 0.10."""
    confidence = _score_cross_language(sintel_720p_video, sintel_ja_srt, whisper_tiny, embed_model)
    assert confidence >= 0.10, f"EN audio + JA subtitle scored {confidence:.2f}, expected >= 0.10"


def test_sintel_hi_cross_language(sintel_720p_video, sintel_hi_srt, whisper_tiny, embed_model):
    """Hindi subtitle against Sintel English audio: cross-language score >= 0.10."""
    confidence = _score_cross_language(sintel_720p_video, sintel_hi_srt, whisper_tiny, embed_model)
    assert confidence >= 0.10, f"EN audio + HI subtitle scored {confidence:.2f}, expected >= 0.10"


def test_sintel_ar_cross_language(sintel_720p_video, sintel_ar_srt, whisper_tiny, embed_model):
    """Arabic (right-to-left) subtitle against Sintel English audio: cross-language score >= 0.10."""
    confidence = _score_cross_language(sintel_720p_video, sintel_ar_srt, whisper_tiny, embed_model)
    assert confidence >= 0.10, f"EN audio + AR subtitle scored {confidence:.2f}, expected >= 0.10"


def test_sintel_zh_hans_cross_language(
    sintel_720p_video, sintel_zh_hans_srt, whisper_tiny, embed_model,
):
    """Chinese Simplified subtitle against Sintel English audio: cross-language score >= 0.10."""
    confidence = _score_cross_language(
        sintel_720p_video, sintel_zh_hans_srt, whisper_tiny, embed_model,
    )
    assert confidence >= 0.10, f"EN audio + ZH-HANS subtitle scored {confidence:.2f}, expected >= 0.10"


def test_sintel_cs_cross_language(sintel_720p_video, sintel_cs_srt, whisper_tiny, embed_model):
    """Czech subtitle against Sintel English audio: cross-language score >= 0.10."""
    confidence = _score_cross_language(sintel_720p_video, sintel_cs_srt, whisper_tiny, embed_model)
    assert confidence >= 0.10, f"EN audio + CS subtitle scored {confidence:.2f}, expected >= 0.10"


def test_sintel_sv_cross_language(sintel_720p_video, sintel_sv_srt, whisper_tiny, embed_model):
    """Swedish subtitle against Sintel English audio: cross-language score >= 0.10."""
    confidence = _score_cross_language(sintel_720p_video, sintel_sv_srt, whisper_tiny, embed_model)
    assert confidence >= 0.10, f"EN audio + SV subtitle scored {confidence:.2f}, expected >= 0.10"


def test_sintel_da_cross_language(sintel_720p_video, sintel_da_srt, whisper_tiny, embed_model):
    """Danish subtitle against Sintel English audio: cross-language score >= 0.10."""
    confidence = _score_cross_language(sintel_720p_video, sintel_da_srt, whisper_tiny, embed_model)
    assert confidence >= 0.10, f"EN audio + DA subtitle scored {confidence:.2f}, expected >= 0.10"


def test_sintel_el_cross_language(sintel_720p_video, sintel_el_srt, whisper_tiny, embed_model):
    """Greek subtitle against Sintel English audio: cross-language score >= 0.10."""
    confidence = _score_cross_language(sintel_720p_video, sintel_el_srt, whisper_tiny, embed_model)
    assert confidence >= 0.10, f"EN audio + EL subtitle scored {confidence:.2f}, expected >= 0.10"


def test_sintel_he_cross_language(sintel_720p_video, sintel_he_srt, whisper_tiny, embed_model):
    """Hebrew (right-to-left) subtitle against Sintel English audio: cross-language score >= 0.10."""
    confidence = _score_cross_language(sintel_720p_video, sintel_he_srt, whisper_tiny, embed_model)
    assert confidence >= 0.10, f"EN audio + HE subtitle scored {confidence:.2f}, expected >= 0.10"


def test_sintel_id_cross_language(sintel_720p_video, sintel_id_srt, whisper_tiny, embed_model):
    """Indonesian subtitle against Sintel English audio: cross-language score >= 0.10."""
    confidence = _score_cross_language(sintel_720p_video, sintel_id_srt, whisper_tiny, embed_model)
    assert confidence >= 0.10, f"EN audio + ID subtitle scored {confidence:.2f}, expected >= 0.10"


def test_sintel_ro_cross_language(sintel_720p_video, sintel_ro_srt, whisper_tiny, embed_model):
    """Romanian subtitle against Sintel English audio: cross-language score >= 0.10."""
    confidence = _score_cross_language(sintel_720p_video, sintel_ro_srt, whisper_tiny, embed_model)
    assert confidence >= 0.10, f"EN audio + RO subtitle scored {confidence:.2f}, expected >= 0.10"


def test_sintel_uk_cross_language(sintel_720p_video, sintel_uk_srt, whisper_tiny, embed_model):
    """Ukrainian (Cyrillic) subtitle against Sintel English audio: cross-language score >= 0.10."""
    confidence = _score_cross_language(sintel_720p_video, sintel_uk_srt, whisper_tiny, embed_model)
    assert confidence >= 0.10, f"EN audio + UK subtitle scored {confidence:.2f}, expected >= 0.10"


def test_sintel_fi_cross_language(sintel_720p_video, sintel_fi_srt, whisper_tiny, embed_model):
    """Finnish subtitle against Sintel English audio: cross-language score >= 0.10."""
    confidence = _score_cross_language(sintel_720p_video, sintel_fi_srt, whisper_tiny, embed_model)
    assert confidence >= 0.10, f"EN audio + FI subtitle scored {confidence:.2f}, expected >= 0.10"

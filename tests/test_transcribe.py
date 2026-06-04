import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from submatch.transcribe import load_model, transcribe_segment, TranscriptionResult


def _mock_whisper(lang_probs=None, transcribe_result=None, audio=None):
    """Return a (mock_whisper_module, mock_model) pair with sensible defaults."""
    if lang_probs is None:
        lang_probs = {"en": 0.9, "fr": 0.1}
    if transcribe_result is None:
        transcribe_result = {"text": "hello world", "language": "en", "segments": []}
    if audio is None:
        audio = np.zeros(16000, dtype=np.float32)

    mock_whisper = MagicMock()
    mock_whisper.load_audio.return_value = audio
    mock_whisper.pad_or_trim.return_value = audio
    mock_whisper.log_mel_spectrogram.return_value = MagicMock()

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.detect_language.return_value = (None, lang_probs)
    mock_model.transcribe.return_value = transcribe_result

    return mock_whisper, mock_model


def test_load_model_selects_mps_when_available():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    mock_whisper = MagicMock()

    with patch.dict(sys.modules, {"torch": mock_torch, "whisper": mock_whisper}):
        load_model("base")

    mock_whisper.load_model.assert_called_once_with("base", device="mps")


def test_load_model_selects_cpu_when_mps_unavailable():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    mock_whisper = MagicMock()

    with patch.dict(sys.modules, {"torch": mock_torch, "whisper": mock_whisper}):
        load_model("tiny")

    mock_whisper.load_model.assert_called_once_with("tiny", device="cpu")


def test_load_model_returns_whisper_model():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    mock_whisper = MagicMock()
    expected_model = MagicMock()
    mock_whisper.load_model.return_value = expected_model

    with patch.dict(sys.modules, {"torch": mock_torch, "whisper": mock_whisper}):
        result = load_model("base")

    assert result is expected_model


def test_load_model_uses_explicit_device():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    mock_whisper = MagicMock()

    with patch.dict(sys.modules, {"torch": mock_torch, "whisper": mock_whisper}):
        load_model("base", device="cpu")

    mock_whisper.load_model.assert_called_once_with("base", device="cpu")


def test_load_model_auto_selects_cuda_when_available():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.backends.mps.is_available.return_value = False
    mock_whisper = MagicMock()

    with patch.dict(sys.modules, {"torch": mock_torch, "whisper": mock_whisper}):
        load_model("base")

    mock_whisper.load_model.assert_called_once_with("base", device="cuda")


def _patch_whisper(mock_whisper):
    return patch.dict(sys.modules, {"whisper": mock_whisper})


def test_transcribe_segment_strips_whitespace(tmp_path):
    wav = tmp_path / "seg.wav"
    wav.touch()
    mock_whisper, mock_model = _mock_whisper(
        transcribe_result={"text": "  hello world  ", "language": "en", "segments": []}
    )
    with _patch_whisper(mock_whisper):
        result = transcribe_segment(mock_model, wav)

    assert isinstance(result, TranscriptionResult)
    assert result.text == "hello world"
    assert result.language == "en"


def test_transcribe_segment_passes_fp16_false(tmp_path):
    wav = tmp_path / "seg.wav"
    wav.touch()
    mock_whisper, mock_model = _mock_whisper(
        transcribe_result={"text": "test", "language": "fr", "segments": []}
    )
    with _patch_whisper(mock_whisper):
        transcribe_segment(mock_model, wav)

    args, kwargs = mock_model.transcribe.call_args
    assert kwargs.get("fp16") is False


def test_transcribe_segment_captures_no_speech_prob(tmp_path):
    wav = tmp_path / "seg.wav"
    wav.touch()
    mock_whisper, mock_model = _mock_whisper(
        transcribe_result={
            "text": "hello world",
            "language": "en",
            "segments": [
                {"no_speech_prob": 0.1},
                {"no_speech_prob": 0.3},
            ],
        }
    )
    with _patch_whisper(mock_whisper):
        result = transcribe_segment(mock_model, wav)

    assert result.no_speech_prob == pytest.approx(0.2)


def test_transcribe_segment_no_segments_gives_no_speech_prob_one(tmp_path):
    wav = tmp_path / "seg.wav"
    wav.touch()
    mock_whisper, mock_model = _mock_whisper(
        transcribe_result={"text": "", "language": "en", "segments": []}
    )
    with _patch_whisper(mock_whisper):
        result = transcribe_segment(mock_model, wav)

    assert result.no_speech_prob == 1.0


def test_transcribe_segment_captures_avg_logprob(tmp_path):
    wav = tmp_path / "seg.wav"
    wav.touch()
    mock_whisper, mock_model = _mock_whisper(
        transcribe_result={
            "text": "hello world",
            "language": "en",
            "segments": [
                {"no_speech_prob": 0.1, "avg_logprob": -0.4},
                {"no_speech_prob": 0.2, "avg_logprob": -0.8},
            ],
        }
    )
    with _patch_whisper(mock_whisper):
        result = transcribe_segment(mock_model, wav)

    assert result.avg_logprob == pytest.approx(-0.6)


def test_transcribe_segment_no_segments_gives_avg_logprob_zero(tmp_path):
    wav = tmp_path / "seg.wav"
    wav.touch()
    mock_whisper, mock_model = _mock_whisper(
        transcribe_result={"text": "", "language": "en", "segments": []}
    )
    with _patch_whisper(mock_whisper):
        result = transcribe_segment(mock_model, wav)

    assert result.avg_logprob == 0.0


def test_transcribe_segment_captures_language_prob(tmp_path):
    wav = tmp_path / "seg.wav"
    wav.touch()
    mock_whisper, mock_model = _mock_whisper(
        lang_probs={"fr": 0.92, "en": 0.05, "de": 0.03},
        transcribe_result={"text": "bonjour monde", "language": "fr", "segments": []}
    )
    with _patch_whisper(mock_whisper):
        result = transcribe_segment(mock_model, wav)

    assert result.language == "fr"
    assert result.language_prob == pytest.approx(0.92)


def test_transcribe_segment_low_confidence_language(tmp_path):
    """When Whisper is uncertain (unsupported language), language_prob stays low."""
    wav = tmp_path / "seg.wav"
    wav.touch()
    mock_whisper, mock_model = _mock_whisper(
        lang_probs={"ne": 0.12, "ar": 0.10, "zh": 0.09},
        transcribe_result={"text": "garbage hallucination", "language": "ne", "segments": []}
    )
    with _patch_whisper(mock_whisper):
        result = transcribe_segment(mock_model, wav)

    assert result.language == "ne"
    assert result.language_prob == pytest.approx(0.12)

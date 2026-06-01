import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from submatch.transcribe import load_model, transcribe_segment, TranscriptionResult


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


def test_transcribe_segment_strips_whitespace():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "  hello world  ", "language": "en"}

    result = transcribe_segment(mock_model, Path("/tmp/audio.wav"))

    assert isinstance(result, TranscriptionResult)
    assert result.text == "hello world"
    assert result.language == "en"


def test_transcribe_segment_passes_fp16_false():
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "test", "language": "fr"}

    transcribe_segment(mock_model, Path("/tmp/audio.wav"))

    mock_model.transcribe.assert_called_once_with("/tmp/audio.wav", fp16=False)


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


def test_transcribe_segment_captures_no_speech_prob(tmp_path):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "  hello world  ",
        "language": "en",
        "segments": [
            {"no_speech_prob": 0.1},
            {"no_speech_prob": 0.3},
        ],
    }
    wav = tmp_path / "seg.wav"
    wav.touch()
    result = transcribe_segment(mock_model, wav)
    assert result.no_speech_prob == pytest.approx(0.2)


def test_transcribe_segment_no_segments_gives_no_speech_prob_one(tmp_path):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "",
        "language": "en",
        "segments": [],
    }
    wav = tmp_path / "seg.wav"
    wav.touch()
    result = transcribe_segment(mock_model, wav)
    assert result.no_speech_prob == 1.0

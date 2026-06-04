from __future__ import annotations
import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TranscriptionResult:
    text: str
    language: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    language_prob: float = 1.0


def load_model(model_name: str = "base", device: str | None = None) -> Any:
    import warnings
    import torch
    import whisper
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        return whisper.load_model(model_name, device=device)


def transcribe_segment(model: Any, audio_path: Path) -> TranscriptionResult:
    import warnings
    import whisper
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio = whisper.load_audio(str(audio_path))
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio)).to(model.device)
        _, lang_probs = model.detect_language(mel)
        result = model.transcribe(audio, fp16=False)
    language = result["language"]
    language_prob = lang_probs.get(language, 0.0)
    segs = result.get("segments", [])
    if segs:
        no_speech_prob = sum(s.get("no_speech_prob", 0.0) for s in segs) / len(segs)
        avg_logprob = sum(s.get("avg_logprob", 0.0) for s in segs) / len(segs)
    else:
        no_speech_prob = 1.0
        avg_logprob = 0.0
    return TranscriptionResult(
        text=result["text"].strip(),
        language=language,
        no_speech_prob=no_speech_prob,
        avg_logprob=avg_logprob,
        language_prob=language_prob,
    )

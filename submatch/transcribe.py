from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TranscriptionResult:
    text: str
    language: str


def load_model(model_name: str = "base") -> Any:
    import torch
    import whisper
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return whisper.load_model(model_name, device=device)


def transcribe_segment(model: Any, audio_path: Path) -> TranscriptionResult:
    result = model.transcribe(str(audio_path), fp16=False)
    return TranscriptionResult(
        text=result["text"].strip(),
        language=result["language"],
    )

"""
End-to-end smoke test: runs the real pipeline against a synthetic video created
by ffmpeg. Whisper is mocked so the test stays fast and offline, but every
other component (ffmpeg audio extraction, duration detection, sampler, compare)
executes for real.
"""
from __future__ import annotations
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from submatch import run, MatchResult
from submatch.pipeline import PipelineConfig
from submatch.transcribe import TranscriptionResult


@pytest.fixture(scope="module")
def smoke_video(tmp_path_factory):
    """10-second black video with a 440 Hz sine tone, created by ffmpeg."""
    out = tmp_path_factory.mktemp("smoke") / "smoke.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:r=10:d=10",
            "-c:v", "libx264", "-c:a", "aac",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture(scope="module")
def smoke_sub(tmp_path_factory):
    """Simple SRT subtitle for the smoke video."""
    sub = tmp_path_factory.mktemp("smoke") / "smoke.en.srt"
    sub.write_text(
        "1\n00:00:01,000 --> 00:00:04,000\nHello world.\n\n"
        "2\n00:00:05,000 --> 00:00:08,000\nGoodbye world.\n\n"
    )
    return sub


def test_run_returns_match_result(smoke_video, smoke_sub):
    """Full pipeline executes without error; transcription is mocked."""
    mock_transcription = TranscriptionResult(
        text="hello world", language="en", no_speech_prob=0.05, avg_logprob=0.4,
    )
    config = PipelineConfig(
        model="tiny", use_cache=False, sync=False, device="cpu", verbose=False,
    )

    with patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_transcription), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()):
        result = run(smoke_video, smoke_sub, config)

    assert isinstance(result, MatchResult)
    assert 0.0 <= result.confidence <= 1.0
    assert result.model == "tiny"
    assert result.language is not None


def test_run_with_matching_subtitle_passes(smoke_video, smoke_sub):
    """High-F1 transcription should yield a passing result."""
    mock_transcription = TranscriptionResult(
        text="hello world", language="en", no_speech_prob=0.05, avg_logprob=0.4,
    )
    config = PipelineConfig(
        model="tiny", use_cache=False, sync=False, device="cpu", threshold=0.1,
    )

    with patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_transcription), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()):
        result = run(smoke_video, smoke_sub, config)

    assert result.passed


def test_run_with_mismatched_subtitle_fails(smoke_video, smoke_sub):
    """Low-F1 transcription against unrelated subtitle should not pass at high threshold."""
    mock_transcription = TranscriptionResult(
        text="completely different content nothing matches here",
        language="en", no_speech_prob=0.05, avg_logprob=0.4,
    )
    config = PipelineConfig(
        model="tiny", use_cache=False, sync=False, device="cpu", threshold=0.9,
    )

    with patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_transcription), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()):
        result = run(smoke_video, smoke_sub, config)

    assert not result.passed

import subprocess
import pytest
from pathlib import Path
from submatch.audio import get_duration_ms, has_audio_track, extract_segment


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("video") / "test.mp4"
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
def silent_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("video") / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:r=10:d=10",
            "-c:v", "libx264", "-an",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


def test_get_duration_ms(tiny_video):
    duration = get_duration_ms(tiny_video)
    assert 9_000 <= duration <= 11_000


def test_has_audio_track_true(tiny_video):
    assert has_audio_track(tiny_video) is True


def test_has_audio_track_false(silent_video):
    assert has_audio_track(silent_video) is False


def test_extract_segment_creates_wav(tiny_video, tmp_path):
    wav = extract_segment(tiny_video, start_ms=0, duration_ms=3_000)
    try:
        assert wav.exists()
        assert wav.suffix == ".wav"
        assert wav.stat().st_size > 0
    finally:
        wav.unlink(missing_ok=True)


def test_extract_segment_respects_duration(tiny_video):
    wav = extract_segment(tiny_video, start_ms=0, duration_ms=3_000)
    try:
        duration = get_duration_ms(wav)
        assert 2_500 <= duration <= 3_500
    finally:
        wav.unlink(missing_ok=True)

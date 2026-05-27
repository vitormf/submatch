import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from submatch.sync import sync_subtitle, _compute_offset, SyncResult, DRIFT_THRESHOLD_SECONDS
from tests.conftest import SAMPLE_SRT

_SHIFTED_SRT = """\
1
00:00:06,000 --> 00:00:08,500
Hello, world.

2
00:00:10,000 --> 00:00:13,000
This is a test subtitle.
With two lines.

3
00:00:15,000 --> 00:00:17,000
Goodbye.
"""


# ── _compute_offset ──────────────────────────────────────────────────────────

def test_compute_offset_identical_files(tmp_path):
    f = tmp_path / "sub.srt"
    f.write_text(SAMPLE_SRT)
    assert _compute_offset(f, f) == pytest.approx(0.0)


def test_compute_offset_five_second_shift(tmp_path):
    original = tmp_path / "original.srt"
    original.write_text(SAMPLE_SRT)
    shifted = tmp_path / "shifted.srt"
    shifted.write_text(_SHIFTED_SRT)
    assert _compute_offset(original, shifted) == pytest.approx(5.0)


def test_compute_offset_empty_files_return_zero(tmp_path):
    f = tmp_path / "empty.srt"
    f.write_text("")
    assert _compute_offset(f, f) == pytest.approx(0.0)


# ── sync_subtitle ────────────────────────────────────────────────────────────

def test_sync_subtitle_success(tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text(SAMPLE_SRT)
    output = tmp_path / "synced.srt"
    output.write_text(SAMPLE_SRT)  # same content → offset=0, no drift

    with patch("submatch.sync.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = sync_subtitle(video, subtitle, output)

    assert isinstance(result, SyncResult)
    assert result.synced_srt_path == output
    assert result.offset_seconds == pytest.approx(0.0)
    assert result.drift_detected is False


def test_sync_subtitle_failure_raises(tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text(SAMPLE_SRT)
    output = tmp_path / "synced.srt"

    with patch("submatch.sync.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="ffs error msg")
        with pytest.raises(RuntimeError, match="ffsubsync failed"):
            sync_subtitle(video, subtitle, output)


def test_sync_subtitle_drift_detected(tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text(SAMPLE_SRT)
    output = tmp_path / "synced.srt"
    output.write_text(_SHIFTED_SRT)  # 5s offset exceeds 2s threshold

    with patch("submatch.sync.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = sync_subtitle(video, subtitle, output)

    assert result.drift_detected is True
    assert result.offset_seconds > DRIFT_THRESHOLD_SECONDS


def test_sync_subtitle_auto_output_path(tmp_path):
    """output_path=None triggers internal tempfile creation."""
    video = tmp_path / "video.mp4"
    video.touch()
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text(SAMPLE_SRT)

    with patch("submatch.sync.subprocess.run") as mock_run:
        def write_and_return(cmd, **kwargs):
            out_path = Path(cmd[cmd.index("-o") + 1])
            out_path.write_text(SAMPLE_SRT)
            return MagicMock(returncode=0, stderr="")
        mock_run.side_effect = write_and_return
        result = sync_subtitle(video, subtitle)

    assert result.synced_srt_path.exists()
    result.synced_srt_path.unlink(missing_ok=True)

import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from submatch.embedded import list_subtitle_tracks, extract_subtitle_track


def _ffprobe_out(streams: list[dict]) -> str:
    return json.dumps({"streams": streams})


def _mock_run(stdout: str):
    m = MagicMock()
    m.stdout = stdout
    return m


def test_list_subtitle_tracks_empty():
    with patch("subprocess.run", return_value=_mock_run(_ffprobe_out([]))):
        result = list_subtitle_tracks(Path("movie.mkv"))
    assert result == []


def test_list_subtitle_tracks_single_with_lang():
    stream = {"tags": {"language": "eng", "title": "English"}}
    with patch("subprocess.run", return_value=_mock_run(_ffprobe_out([stream]))):
        result = list_subtitle_tracks(Path("movie.mkv"))
    assert result == [{"index": 0, "lang": "eng", "title": "English"}]


def test_list_subtitle_tracks_multiple():
    streams = [
        {"tags": {"language": "eng"}},
        {"tags": {"language": "jpn", "title": "Japanese"}},
        {},  # no tags at all
    ]
    with patch("subprocess.run", return_value=_mock_run(_ffprobe_out(streams))):
        result = list_subtitle_tracks(Path("movie.mkv"))
    assert result == [
        {"index": 0, "lang": "eng", "title": None},
        {"index": 1, "lang": "jpn", "title": "Japanese"},
        {"index": 2, "lang": None, "title": None},
    ]


def test_list_subtitle_tracks_ffprobe_command():
    with patch("subprocess.run", return_value=_mock_run(_ffprobe_out([]))) as mock_run:
        list_subtitle_tracks(Path("movie.mkv"))
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ffprobe"
    assert "-select_streams" in cmd
    assert "s" in cmd
    assert "movie.mkv" in cmd


def _mock_proc(returncode=0):
    proc = MagicMock()
    proc.communicate.return_value = (b"", b"")
    proc.returncode = returncode
    proc.pid = 12345
    return proc


def test_extract_subtitle_track_calls_ffmpeg(tmp_path):
    dest = tmp_path / "sub.srt"
    with patch("submatch.embedded.subprocess.Popen", return_value=_mock_proc()) as mock_popen:
        extract_subtitle_track(Path("movie.mkv"), 0, dest)
    cmd = mock_popen.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert "-map" in cmd
    idx = cmd.index("-map")
    assert cmd[idx + 1] == "0:s:0"
    assert "-c:s" in cmd
    assert "srt" in cmd
    assert str(dest) in cmd


def test_extract_subtitle_track_index_2(tmp_path):
    dest = tmp_path / "sub.srt"
    with patch("submatch.embedded.subprocess.Popen", return_value=_mock_proc()) as mock_popen:
        extract_subtitle_track(Path("movie.mkv"), 2, dest)
    cmd = mock_popen.call_args[0][0]
    idx = cmd.index("-map")
    assert cmd[idx + 1] == "0:s:2"


def test_extract_subtitle_track_propagates_error(tmp_path):
    dest = tmp_path / "sub.srt"
    with patch("submatch.embedded.subprocess.Popen", return_value=_mock_proc(returncode=1)):
        with pytest.raises(subprocess.CalledProcessError):
            extract_subtitle_track(Path("movie.mkv"), 0, dest)


def test_extract_subtitle_track_uses_process_group(tmp_path):
    """ffmpeg must be launched in its own process group (preexec_fn=os.setsid)."""
    import os
    dest = tmp_path / "sub.srt"
    with patch("submatch.embedded.subprocess.Popen", return_value=_mock_proc()) as mock_popen:
        extract_subtitle_track(Path("movie.mkv"), 0, dest)
    assert mock_popen.call_args.kwargs.get("preexec_fn") is os.setsid


def test_extract_subtitle_track_keyboard_interrupt_kills_process_group(tmp_path):
    """KeyboardInterrupt must kill the ffmpeg process group and re-raise."""
    import signal
    dest = tmp_path / "sub.srt"

    proc = MagicMock()
    proc.communicate.side_effect = KeyboardInterrupt
    proc.pid = 12345

    with patch("submatch.embedded.subprocess.Popen", return_value=proc), \
         patch("submatch.embedded.os.getpgid", return_value=12345) as mock_getpgid, \
         patch("submatch.embedded.os.killpg") as mock_killpg:
        with pytest.raises(KeyboardInterrupt):
            extract_subtitle_track(Path("movie.mkv"), 0, dest)

    mock_getpgid.assert_called_once_with(12345)
    mock_killpg.assert_called_once_with(12345, signal.SIGTERM)
    proc.wait.assert_called_once()

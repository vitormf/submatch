import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from submatch.embedded import list_subtitle_tracks, extract_subtitle_track, extract_all_subtitle_tracks


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


# --- extract_all_subtitle_tracks ---

def test_extract_all_subtitle_tracks_single_ffmpeg_call(tmp_path):
    """All tracks must be extracted in a single ffmpeg invocation."""
    tracks = [
        {"index": 0, "lang": "eng", "title": None},
        {"index": 1, "lang": "jpn", "title": None},
        {"index": 2, "lang": "por", "title": None},
    ]
    with patch("submatch.embedded.subprocess.Popen", return_value=_mock_proc()) as mock_popen:
        extract_all_subtitle_tracks(Path("movie.mkv"), tracks, tmp_path)
    assert mock_popen.call_count == 1


def test_extract_all_subtitle_tracks_maps_all_streams(tmp_path):
    """Command must contain a -map 0:s:N for each track."""
    tracks = [
        {"index": 0, "lang": "eng", "title": None},
        {"index": 1, "lang": "jpn", "title": None},
    ]
    with patch("submatch.embedded.subprocess.Popen", return_value=_mock_proc()) as mock_popen:
        extract_all_subtitle_tracks(Path("movie.mkv"), tracks, tmp_path)
    cmd = mock_popen.call_args[0][0]
    maps = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-map"]
    assert "0:s:0" in maps
    assert "0:s:1" in maps


def test_extract_all_subtitle_tracks_returns_paths(tmp_path):
    """Return value must map each track index to its output SRT path."""
    tracks = [
        {"index": 0, "lang": "eng", "title": None},
        {"index": 1, "lang": "jpn", "title": None},
    ]
    with patch("submatch.embedded.subprocess.Popen", return_value=_mock_proc()):
        result = extract_all_subtitle_tracks(Path("movie.mkv"), tracks, tmp_path)
    assert set(result.keys()) == {0, 1}
    for path in result.values():
        assert str(path).endswith(".srt")
        assert str(tmp_path) in str(path)


def test_extract_all_subtitle_tracks_empty(tmp_path):
    """Empty track list must return empty dict without calling ffmpeg."""
    with patch("submatch.embedded.subprocess.Popen") as mock_popen:
        result = extract_all_subtitle_tracks(Path("movie.mkv"), [], tmp_path)
    assert result == {}
    mock_popen.assert_not_called()


def test_extract_all_subtitle_tracks_propagates_error(tmp_path):
    tracks = [{"index": 0, "lang": "eng", "title": None}]
    with patch("submatch.embedded.subprocess.Popen", return_value=_mock_proc(returncode=1)):
        with pytest.raises(subprocess.CalledProcessError):
            extract_all_subtitle_tracks(Path("movie.mkv"), tracks, tmp_path)


def test_extract_all_subtitle_tracks_keyboard_interrupt_kills_process_group(tmp_path):
    import signal
    tracks = [{"index": 0, "lang": "eng", "title": None}]
    proc = MagicMock()
    proc.communicate.side_effect = KeyboardInterrupt
    proc.pid = 12345
    with patch("submatch.embedded.subprocess.Popen", return_value=proc), \
         patch("submatch.embedded.os.getpgid", return_value=12345) as mock_getpgid, \
         patch("submatch.embedded.os.killpg") as mock_killpg:
        with pytest.raises(KeyboardInterrupt):
            extract_all_subtitle_tracks(Path("movie.mkv"), tracks, tmp_path)
    mock_getpgid.assert_called_once_with(12345)
    mock_killpg.assert_called_once_with(12345, signal.SIGTERM)
    proc.wait.assert_called_once()

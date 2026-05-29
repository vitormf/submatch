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


def test_extract_subtitle_track_calls_ffmpeg(tmp_path):
    dest = tmp_path / "sub.srt"
    with patch("subprocess.run") as mock_run:
        extract_subtitle_track(Path("movie.mkv"), 0, dest)
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert "-map" in cmd
    idx = cmd.index("-map")
    assert cmd[idx + 1] == "0:s:0"
    assert "-c:s" in cmd
    assert "srt" in cmd
    assert str(dest) in cmd


def test_extract_subtitle_track_index_2(tmp_path):
    dest = tmp_path / "sub.srt"
    with patch("subprocess.run") as mock_run:
        extract_subtitle_track(Path("movie.mkv"), 2, dest)
    cmd = mock_run.call_args[0][0]
    idx = cmd.index("-map")
    assert cmd[idx + 1] == "0:s:2"


def test_extract_subtitle_track_propagates_error(tmp_path):
    dest = tmp_path / "sub.srt"
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
        with pytest.raises(subprocess.CalledProcessError):
            extract_subtitle_track(Path("movie.mkv"), 0, dest)

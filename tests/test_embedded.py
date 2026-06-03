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
    assert result == [{"index": 0, "global_index": 0, "lang": "eng", "title": "English", "codec": None}]


def test_list_subtitle_tracks_multiple():
    streams = [
        {"tags": {"language": "eng"}},
        {"tags": {"language": "jpn", "title": "Japanese"}},
        {},  # no tags at all
    ]
    with patch("subprocess.run", return_value=_mock_run(_ffprobe_out(streams))):
        result = list_subtitle_tracks(Path("movie.mkv"))
    assert result == [
        {"index": 0, "global_index": 0, "lang": "eng", "title": None, "codec": None},
        {"index": 1, "global_index": 1, "lang": "jpn", "title": "Japanese", "codec": None},
        {"index": 2, "global_index": 2, "lang": None, "title": None, "codec": None},
    ]


def test_list_subtitle_tracks_global_index_from_ffprobe():
    """global_index reflects the stream's position among ALL streams, not just subtitles."""
    stream = {"index": 5, "codec_name": "dvd_subtitle", "tags": {"language": "eng"}}
    with patch("subprocess.run", return_value=_mock_run(_ffprobe_out([stream]))):
        result = list_subtitle_tracks(Path("movie.mkv"))
    assert result[0]["index"] == 0         # subtitle-relative (for -map 0:s:0)
    assert result[0]["global_index"] == 5  # global (for mkvextract tracks 5:dest)


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


# --- codec detection and image track extraction ---

def test_list_subtitle_tracks_includes_codec():
    stream = {"codec_name": "dvd_subtitle", "tags": {"language": "eng"}}
    with patch("subprocess.run", return_value=_mock_run(_ffprobe_out([stream]))):
        result = list_subtitle_tracks(Path("movie.mkv"))
    assert result[0]["codec"] == "dvd_subtitle"


def test_list_subtitle_tracks_codec_none_when_absent():
    stream = {"tags": {"language": "eng"}}
    with patch("subprocess.run", return_value=_mock_run(_ffprobe_out([stream]))):
        result = list_subtitle_tracks(Path("movie.mkv"))
    assert result[0]["codec"] is None


def test_extract_all_subtitle_tracks_image_track_produces_sub(tmp_path):
    """dvd_subtitle tracks must be extracted via mkvextract to .sub (VOBSUB format).

    ffmpeg maps .sub to the microdvd muxer (wrong codec), so mkvextract is required
    to write the correct VOBSUB .sub + .idx pair.
    """
    tracks = [{"index": 0, "global_index": 5, "lang": "eng", "title": None, "codec": "dvd_subtitle"}]
    with patch("submatch.embedded.subprocess.Popen", return_value=_mock_proc()) as mock_popen:
        result = extract_all_subtitle_tracks(Path("movie.mkv"), tracks, tmp_path)
    assert result[0].suffix == ".sub"
    cmd = mock_popen.call_args[0][0]
    assert cmd[0] == "mkvextract"
    assert "tracks" in cmd
    # mkvextract track spec: "global_idx:dest_path"
    track_spec = next(a for a in cmd if ":" in a and a[0].isdigit())
    assert track_spec.startswith("5:")


def test_extract_all_subtitle_tracks_image_track_skips_on_error(tmp_path):
    """ffmpeg error on an image track must be swallowed — track is skipped, not raised."""
    tracks = [{"index": 0, "lang": "eng", "title": None, "codec": "dvd_subtitle"}]
    with patch("submatch.embedded.subprocess.Popen", return_value=_mock_proc(returncode=1)):
        result = extract_all_subtitle_tracks(Path("movie.mkv"), tracks, tmp_path)
    assert result == {}


def test_extract_all_subtitle_tracks_image_error_does_not_block_text_tracks(tmp_path):
    """A failing image track must not prevent successful text track extraction."""
    tracks = [
        {"index": 0, "lang": "eng", "title": None, "codec": "dvd_subtitle"},
        {"index": 1, "lang": "eng", "title": None, "codec": "subrip"},
    ]
    # Text-track ffmpeg pass succeeds; image-track ffmpeg pass fails
    good_proc = _mock_proc(returncode=0)
    bad_proc = _mock_proc(returncode=1)
    with patch("submatch.embedded.subprocess.Popen", side_effect=[good_proc, bad_proc]):
        result = extract_all_subtitle_tracks(Path("movie.mkv"), tracks, tmp_path)
    assert 1 in result   # text track extracted
    assert 0 not in result  # image track skipped


def test_extract_all_subtitle_tracks_image_track_keyboard_interrupt_kills_process_group(tmp_path):
    """KeyboardInterrupt during image track extraction must kill the process group and re-raise."""
    import signal
    tracks = [{"index": 0, "lang": "eng", "title": None, "codec": "dvd_subtitle"}]
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


def test_extract_all_subtitle_tracks_pgs_track_produces_sup(tmp_path):
    """hdmv_pgs_subtitle tracks must be extracted to .sup."""
    tracks = [{"index": 0, "lang": "eng", "title": None, "codec": "hdmv_pgs_subtitle"}]
    with patch("submatch.embedded.subprocess.Popen", return_value=_mock_proc()):
        result = extract_all_subtitle_tracks(Path("movie.mkv"), tracks, tmp_path)
    assert result[0].suffix == ".sup"


def test_extract_all_subtitle_tracks_image_uses_copy_codec(tmp_path):
    """PGS (hdmv_pgs_subtitle) must use ffmpeg with -c:s copy (not mkvextract)."""
    tracks = [{"index": 0, "lang": "eng", "title": None, "codec": "hdmv_pgs_subtitle"}]
    with patch("submatch.embedded.subprocess.Popen", return_value=_mock_proc()) as mock_popen:
        extract_all_subtitle_tracks(Path("movie.mkv"), tracks, tmp_path)
    cmd = mock_popen.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert "-c:s" in cmd
    cs_idx = cmd.index("-c:s")
    assert cmd[cs_idx + 1] == "copy"


def test_extract_all_subtitle_tracks_mixed_text_and_image(tmp_path):
    """Text and image tracks together: text → .srt, image → .sub."""
    tracks = [
        {"index": 0, "lang": "eng", "title": None, "codec": "subrip"},
        {"index": 1, "lang": "eng", "title": None, "codec": "dvd_subtitle"},
    ]
    with patch("submatch.embedded.subprocess.Popen", return_value=_mock_proc()):
        result = extract_all_subtitle_tracks(Path("movie.mkv"), tracks, tmp_path)
    assert result[0].suffix == ".srt"
    assert result[1].suffix == ".sub"


def test_extract_all_subtitle_tracks_unknown_codec_treated_as_text(tmp_path):
    """Tracks with unknown or missing codec fall back to SRT extraction."""
    tracks = [{"index": 0, "lang": "eng", "title": None, "codec": None}]
    with patch("submatch.embedded.subprocess.Popen", return_value=_mock_proc()):
        result = extract_all_subtitle_tracks(Path("movie.mkv"), tracks, tmp_path)
    assert result[0].suffix == ".srt"

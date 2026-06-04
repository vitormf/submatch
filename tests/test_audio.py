import json
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from submatch.audio import get_duration_ms, get_audio_track_duration_ms, has_audio_track, extract_segment, list_audio_tracks, resolve_audio_track, detect_speech_regions


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


def _mock_ffprobe(data: dict) -> MagicMock:
    m = MagicMock()
    m.stdout = json.dumps(data)
    return m


def test_get_duration_ms_falls_back_to_stream_duration_when_format_missing():
    """Regression for PYTHON-D: KeyError when format dict has no 'duration' key.

    Some containers (raw MPEG-TS, certain MP4) omit duration at the format level.
    get_duration_ms must fall back to the maximum stream-level duration.
    """
    data = {
        "format": {"filename": "video.ts", "nb_streams": 2},  # no 'duration' key
        "streams": [
            {"codec_type": "video", "duration": "3600.0"},
            {"codec_type": "audio", "duration": "3598.5"},
        ],
    }
    with patch("submatch.audio.subprocess.run", return_value=_mock_ffprobe(data)):
        result = get_duration_ms(Path("video.ts"))
    assert result == 3_600_000


def test_get_duration_ms_prefers_format_duration_over_streams():
    """Format-level duration takes precedence when present."""
    data = {
        "format": {"duration": "120.0"},
        "streams": [{"codec_type": "audio", "duration": "100.0"}],
    }
    with patch("submatch.audio.subprocess.run", return_value=_mock_ffprobe(data)):
        result = get_duration_ms(Path("video.mp4"))
    assert result == 120_000


def test_get_duration_ms_raises_when_no_duration_anywhere():
    """Raises ValueError with clear message when neither format nor streams have duration."""
    data = {"format": {}, "streams": [{"codec_type": "video"}]}
    with patch("submatch.audio.subprocess.run", return_value=_mock_ffprobe(data)):
        with pytest.raises(ValueError, match="duration"):
            get_duration_ms(Path("video.mp4"))


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


def _ffprobe_audio_response(streams: list[dict]) -> str:
    return json.dumps({"streams": streams})


def test_list_audio_tracks_two_streams(tmp_path):
    video = tmp_path / "v.mkv"
    video.touch()
    response = _ffprobe_audio_response([
        {"tags": {"language": "eng"}},
        {"tags": {"language": "jpn"}},
    ])
    with patch("submatch.audio.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=response, returncode=0)
        tracks = list_audio_tracks(video)
    assert tracks == [{"index": 0, "lang": "eng"}, {"index": 1, "lang": "jpn"}]


def test_list_audio_tracks_no_language_tag(tmp_path):
    video = tmp_path / "v.mkv"
    video.touch()
    response = _ffprobe_audio_response([{"codec_type": "audio"}])
    with patch("submatch.audio.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=response, returncode=0)
        tracks = list_audio_tracks(video)
    assert tracks == [{"index": 0, "lang": None}]


def test_list_audio_tracks_empty(tmp_path):
    video = tmp_path / "v.mkv"
    video.touch()
    response = _ffprobe_audio_response([])
    with patch("submatch.audio.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=response, returncode=0)
        tracks = list_audio_tracks(video)
    assert tracks == []


def _two_track_video(tmp_path):
    video = tmp_path / "v.mkv"
    video.touch()
    tracks = [{"index": 0, "lang": "eng"}, {"index": 1, "lang": "jpn"}]
    return video, tracks


def test_resolve_audio_track_integer_valid(tmp_path):
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "1")
    assert idx == 1
    assert lang == "jpn"


def test_resolve_audio_track_integer_zero(tmp_path):
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "0")
    assert idx == 0
    assert lang == "eng"


def test_resolve_audio_track_integer_out_of_range(tmp_path):
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        with pytest.raises(SystemExit) as exc:
            resolve_audio_track(video, "5")
    assert exc.value.code == 2


def test_resolve_audio_track_language_exact_match(tmp_path):
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "jpn")
    assert idx == 1
    assert lang == "jpn"


def test_resolve_audio_track_language_iso_639_1_to_2(tmp_path):
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "jp")
    assert idx == 1
    assert lang == "jpn"


def test_resolve_audio_track_language_en_matches_eng(tmp_path):
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "en")
    assert idx == 0
    assert lang == "eng"


def test_resolve_audio_track_preference_list_first_match(tmp_path):
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "jp,en")
    assert idx == 1  # jp matched first


def test_resolve_audio_track_preference_list_fallback_to_second(tmp_path):
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "fr,en")
    assert idx == 0  # fr not found, en matched


def test_resolve_audio_track_no_match_falls_back_to_track_0(tmp_path, capsys):
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "fr,de")
    assert idx == 0
    assert lang == "eng"
    captured = capsys.readouterr()
    assert "Warning" in captured.err


def test_resolve_audio_track_case_insensitive(tmp_path):
    video, tracks = _two_track_video(tmp_path)
    with patch("submatch.audio.list_audio_tracks", return_value=tracks):
        idx, lang = resolve_audio_track(video, "JP")
    assert idx == 1


def test_resolve_audio_track_ffprobe_failure_falls_back(tmp_path, capsys):
    video = tmp_path / "v.mkv"
    video.touch()
    with patch("submatch.audio.list_audio_tracks", side_effect=RuntimeError("ffprobe fail")):
        idx, lang = resolve_audio_track(video, "jp")
    assert idx == 0
    assert lang is None
    captured = capsys.readouterr()
    assert "Warning" in captured.err


def _mock_proc(returncode=0):
    proc = MagicMock()
    proc.communicate.return_value = (b"", b"")
    proc.returncode = returncode
    proc.pid = 12345
    return proc


def test_extract_segment_default_track_no_map_flag(tmp_path):
    """audio_track=0 must NOT add a -map flag."""
    video = tmp_path / "v.mp4"
    video.touch()
    captured_cmds = []

    def make_proc(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        return _mock_proc()

    with patch("submatch.audio.subprocess.Popen", side_effect=make_proc):
        from submatch.audio import extract_segment
        wav = extract_segment(video, 0, 3_000, audio_track=0)
        wav.unlink(missing_ok=True)

    assert captured_cmds
    assert "-map" not in captured_cmds[0]


def test_extract_segment_nonzero_track_has_map_flag(tmp_path):
    """audio_track=2 must add '-map' '0:a:2'."""
    video = tmp_path / "v.mp4"
    video.touch()
    captured_cmds = []

    def make_proc(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        return _mock_proc()

    with patch("submatch.audio.subprocess.Popen", side_effect=make_proc):
        from submatch.audio import extract_segment
        wav = extract_segment(video, 0, 3_000, audio_track=2)
        wav.unlink(missing_ok=True)

    assert captured_cmds
    cmd = captured_cmds[0]
    assert "-map" in cmd
    assert cmd[cmd.index("-map") + 1] == "0:a:2"


def test_extract_segment_uses_process_group(tmp_path):
    """ffmpeg must be launched in its own process group (preexec_fn=os.setsid)."""
    import os
    video = tmp_path / "v.mp4"
    video.touch()

    with patch("submatch.audio.subprocess.Popen", return_value=_mock_proc()) as mock_popen:
        from submatch.audio import extract_segment
        wav = extract_segment(video, 0, 3_000)
        wav.unlink(missing_ok=True)

    assert mock_popen.call_args.kwargs.get("preexec_fn") is os.setsid


def test_extract_segment_keyboard_interrupt_kills_process_group(tmp_path):
    """KeyboardInterrupt must kill the ffmpeg process group and re-raise."""
    import signal
    video = tmp_path / "v.mp4"
    video.touch()

    proc = MagicMock()
    proc.communicate.side_effect = KeyboardInterrupt
    proc.pid = 12345

    with patch("submatch.audio.subprocess.Popen", return_value=proc), \
         patch("submatch.audio.os.getpgid", return_value=12345) as mock_getpgid, \
         patch("submatch.audio.os.killpg") as mock_killpg:
        from submatch.audio import extract_segment
        with pytest.raises(KeyboardInterrupt):
            extract_segment(video, 0, 3_000)

    mock_getpgid.assert_called_once_with(12345)
    mock_killpg.assert_called_once_with(12345, signal.SIGTERM)
    proc.wait.assert_called_once()


def test_detect_speech_regions_parses_silence_correctly():
    stderr = (
        "  Duration: 00:01:30.00, start: 0.000000, bitrate: 1000 kb/s\n"
        "[silencedetect @ 0x...] silence_start: 5.0\n"
        "[silencedetect @ 0x...] silence_end: 10.0 | silence_duration: 5.0\n"
        "[silencedetect @ 0x...] silence_start: 60.0\n"
        "[silencedetect @ 0x...] silence_end: 70.0 | silence_duration: 10.0\n"
    )
    mock_result = MagicMock()
    mock_result.stderr = stderr

    with patch("submatch.audio.subprocess.run", return_value=mock_result):
        regions = detect_speech_regions(Path("video.mkv"), audio_track=0)

    # Non-silent: [0,5s], [10s,60s], [70s,90s]
    assert regions == [(0, 5000), (10000, 60000), (70000, 90000)]


def test_detect_speech_regions_video_starts_with_silence():
    stderr = (
        "  Duration: 00:01:00.00, start: 0.000000, bitrate: 1000 kb/s\n"
        "[silencedetect @ 0x...] silence_start: 0.0\n"
        "[silencedetect @ 0x...] silence_end: 5.0 | silence_duration: 5.0\n"
    )
    mock_result = MagicMock()
    mock_result.stderr = stderr

    with patch("submatch.audio.subprocess.run", return_value=mock_result):
        regions = detect_speech_regions(Path("video.mkv"), audio_track=0)

    # Non-silent: [5s, 60s]
    assert regions == [(5000, 60000)]


def test_detect_speech_regions_returns_empty_on_failure():
    with patch("submatch.audio.subprocess.run", side_effect=Exception("ffmpeg not found")):
        regions = detect_speech_regions(Path("video.mkv"), audio_track=0)
    assert regions == []


def test_detect_speech_regions_returns_empty_when_no_duration():
    mock_result = MagicMock()
    mock_result.stderr = "some output without duration info"
    with patch("submatch.audio.subprocess.run", return_value=mock_result):
        regions = detect_speech_regions(Path("video.mkv"), audio_track=0)
    assert regions == []


def test_lang_match_returns_false_for_empty_track_lang():
    from submatch.audio import _lang_match
    assert _lang_match("en", None) is False
    assert _lang_match("en", "") is False


def test_extract_segment_raises_on_nonzero_returncode(tmp_path):
    video = tmp_path / "v.mp4"
    video.touch()
    proc = MagicMock()
    proc.returncode = 1
    proc.communicate.return_value = (b"", b"error output")
    with patch("submatch.audio.subprocess.Popen", return_value=proc), \
         pytest.raises(subprocess.CalledProcessError):
        extract_segment(video, start_ms=0, duration_ms=5_000)


# --- get_audio_track_duration_ms ---

def test_get_audio_track_duration_ms_returns_stream_duration(tmp_path):
    """Should return the audio stream duration in ms."""
    response = json.dumps({"streams": [{"duration": "3600.5"}]})
    with patch("submatch.audio.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=response, returncode=0)
        result = get_audio_track_duration_ms(Path("video.mkv"), audio_track=0)
    assert result == 3600500


def test_get_audio_track_duration_ms_returns_none_for_missing_stream(tmp_path):
    """When requested track index doesn't exist, return None."""
    response = json.dumps({"streams": []})
    with patch("submatch.audio.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=response, returncode=0)
        result = get_audio_track_duration_ms(Path("video.mkv"), audio_track=0)
    assert result is None


def test_get_audio_track_duration_ms_returns_none_when_duration_absent(tmp_path):
    """Some streams don't report a duration field — return None."""
    response = json.dumps({"streams": [{"codec_type": "audio"}]})
    with patch("submatch.audio.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=response, returncode=0)
        result = get_audio_track_duration_ms(Path("video.mkv"), audio_track=0)
    assert result is None


def test_get_audio_track_duration_ms_selects_correct_track(tmp_path):
    """When multiple audio tracks exist, return duration of the requested one."""
    response = json.dumps({"streams": [
        {"duration": "1000.0"},
        {"duration": "2500.0"},
    ]})
    with patch("submatch.audio.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=response, returncode=0)
        result = get_audio_track_duration_ms(Path("video.mkv"), audio_track=1)
    assert result == 2500000

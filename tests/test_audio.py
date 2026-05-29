import json
import subprocess
import pytest
from unittest.mock import patch, MagicMock
from submatch.audio import get_duration_ms, has_audio_track, extract_segment, list_audio_tracks, resolve_audio_track


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


def test_extract_segment_default_track_no_map_flag(tmp_path):
    """audio_track=0 must NOT add a -map flag."""
    video = tmp_path / "v.mp4"
    video.touch()
    captured_cmds = []

    def fake_run(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        return MagicMock(returncode=0)

    with patch("submatch.audio.subprocess.run", side_effect=fake_run):
        import submatch.audio as _audio
        import tempfile as _tf
        with patch.object(_tf, "NamedTemporaryFile") as mock_ntf:
            mock_ntf.return_value.name = str(tmp_path / "out.wav")
            mock_ntf.return_value.close = lambda: None
            (tmp_path / "out.wav").write_bytes(b"")
            _audio.extract_segment(video, 0, 3_000, audio_track=0)

    assert captured_cmds
    cmd = captured_cmds[0]
    assert "-map" not in cmd


def test_extract_segment_nonzero_track_has_map_flag(tmp_path):
    """audio_track=2 must add '-map' '0:a:2'."""
    video = tmp_path / "v.mp4"
    video.touch()
    captured_cmds = []

    def fake_run(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        return MagicMock(returncode=0)

    with patch("submatch.audio.subprocess.run", side_effect=fake_run):
        import submatch.audio as _audio
        import tempfile as _tf
        with patch.object(_tf, "NamedTemporaryFile") as mock_ntf:
            mock_ntf.return_value.name = str(tmp_path / "out.wav")
            mock_ntf.return_value.close = lambda: None
            (tmp_path / "out.wav").write_bytes(b"")
            _audio.extract_segment(video, 0, 3_000, audio_track=2)

    assert captured_cmds
    cmd = captured_cmds[0]
    assert "-map" in cmd
    assert cmd[cmd.index("-map") + 1] == "0:a:2"

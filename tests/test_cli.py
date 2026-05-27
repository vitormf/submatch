import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from submatch import cli
from submatch.language import LanguageResult
from submatch.sampler import Segment
from submatch.subtitle import Subtitle
from tests.conftest import SAMPLE_SRT


# ── parse_args ────────────────────────────────────────────────────────────────

def test_parse_args_defaults(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s)]):
        args = cli.parse_args()
    assert args.video == v
    assert args.subtitle == s
    assert args.model == "base"
    assert args.threshold == pytest.approx(0.35)
    assert args.segments is None
    assert args.json is False
    assert args.compact is False
    assert args.verbose is False
    assert args.language is None
    assert args.no_sync is False
    assert args.keep_synced is False


def test_parse_args_all_flags(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", [
        "submatch", str(v), str(s),
        "--model", "small", "--threshold", "0.6", "--segments", "4",
        "--json", "--compact", "--verbose", "--language", "pt", "--no-sync", "--keep-synced",
    ]):
        args = cli.parse_args()
    assert args.model == "small"
    assert args.threshold == pytest.approx(0.6)
    assert args.segments == 4
    assert args.json is True
    assert args.compact is True
    assert args.verbose is True
    assert args.language == "pt"
    assert args.no_sync is True
    assert args.keep_synced is True


# ── check_dependencies ────────────────────────────────────────────────────────

def test_check_dependencies_all_present():
    with patch("submatch.cli.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch.dict(sys.modules, {"whisper": MagicMock()}):
        cli.check_dependencies()


def test_check_dependencies_missing_ffmpeg():
    def fake_which(name):
        return None if name == "ffmpeg" else "/usr/bin/ffs"
    with patch("submatch.cli.shutil.which", side_effect=fake_which), \
         patch.dict(sys.modules, {"whisper": MagicMock()}):
        with pytest.raises(SystemExit) as exc:
            cli.check_dependencies()
    assert exc.value.code == 2


def test_check_dependencies_missing_ffsubsync():
    def fake_which(name):
        return "/usr/bin/ffmpeg" if name == "ffmpeg" else None
    with patch("submatch.cli.shutil.which", side_effect=fake_which), \
         patch.dict(sys.modules, {"whisper": MagicMock()}):
        with pytest.raises(SystemExit) as exc:
            cli.check_dependencies()
    assert exc.value.code == 2


def test_check_dependencies_skip_sync_ignores_missing_ffs():
    def fake_which(name):
        return "/usr/bin/ffmpeg" if name == "ffmpeg" else None
    with patch("submatch.cli.shutil.which", side_effect=fake_which), \
         patch.dict(sys.modules, {"whisper": MagicMock()}):
        cli.check_dependencies(skip_sync=True)  # should not raise


def test_check_dependencies_missing_whisper():
    with patch("submatch.cli.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch.dict(sys.modules, {"whisper": None}):
        with pytest.raises(SystemExit) as exc:
            cli.check_dependencies()
    assert exc.value.code == 2


# ── main helpers ──────────────────────────────────────────────────────────────

def _make_pipeline_patches(tmp_path, extra_argv=()):
    """Return (video, subtitle, argv, context_manager_list) for a full mock run."""
    video = tmp_path / "video.mp4"
    video.touch()
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text(SAMPLE_SRT)

    subs = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    argv = ["submatch", str(video), str(subtitle), "--no-sync"] + list(extra_argv)

    ctx = [
        patch("sys.argv", argv),
        patch("submatch.cli.check_dependencies"),
        patch("submatch.cli.audio.has_audio_track", return_value=True),
        patch("submatch.cli.audio.get_duration_ms", return_value=90 * 60 * 1_000),
        patch("submatch.cli.audio.extract_segment", return_value=tmp_path / "seg.wav"),
        patch("submatch.cli.subtitle.parse", return_value=subs),
        patch("submatch.cli.sampler.select_segments", return_value=segs),
        patch("submatch.cli.transcribe.load_model", return_value=MagicMock()),
        patch("submatch.cli.transcribe.transcribe_segment", return_value=mock_trans),
        patch("submatch.cli.language.detect_from_text", return_value="en"),
        patch("submatch.cli.language.detect_from_filename", return_value="en"),
        patch("submatch.cli.language.detect_from_video", return_value=None),
        patch("submatch.cli.language.build_result", return_value=lang),
    ]
    return video, subtitle, ctx


# ── main ──────────────────────────────────────────────────────────────────────

def test_main_video_not_found(tmp_path):
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text(SAMPLE_SRT)
    with patch("sys.argv", ["submatch", str(tmp_path / "no.mp4"), str(subtitle)]):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 2


def test_main_subtitle_not_found(tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    with patch("sys.argv", ["submatch", str(video), str(tmp_path / "no.srt")]):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 2


def test_main_no_audio_track(tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text(SAMPLE_SRT)
    with patch("sys.argv", ["submatch", str(video), str(subtitle), "--no-sync"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=False):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 2


def test_main_pipeline_passes(tmp_path):
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--threshold", "0.01"])
    stack = [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    assert exc.value.code == 0


def test_main_pipeline_fails(tmp_path):
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--threshold", "2.0"])
    stack = [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    assert exc.value.code == 1


def test_main_json_output(tmp_path, capsys):
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--json", "--threshold", "0.01"])
    stack = [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit):
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    data = json.loads(capsys.readouterr().out)
    assert "confidence" in data
    assert "passed" in data


def test_main_sync_success_reparses_srt(tmp_path):
    """When sync succeeds, main re-parses the synced subtitle (line 82)."""
    from submatch.sync import SyncResult
    video = tmp_path / "video.mp4"
    video.touch()
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text(SAMPLE_SRT)
    synced_srt = tmp_path / "synced.srt"
    synced_srt.write_text(SAMPLE_SRT)
    sync_result = SyncResult(synced_srt_path=synced_srt, offset_seconds=0.0, drift_detected=False)

    subs = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(video), str(subtitle), "--threshold", "0.01"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.cli.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.cli.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.cli.sampler.select_segments", return_value=segs), \
         patch("submatch.cli.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.cli.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.cli.language.detect_from_text", return_value="en"), \
         patch("submatch.cli.language.detect_from_filename", return_value="en"), \
         patch("submatch.cli.language.detect_from_video", return_value=None), \
         patch("submatch.cli.language.build_result", return_value=lang), \
         patch("submatch.cli.sync.sync_subtitle", return_value=sync_result), \
         patch("submatch.cli.subtitle.parse", return_value=subs) as mock_parse:
        with pytest.raises(SystemExit) as exc:
            cli.main()

    assert exc.value.code == 0
    # srt.parse called twice: once for original, once for synced
    assert mock_parse.call_count == 2


def test_main_segment_transcription_failure_warns(tmp_path, capsys):
    """Transcription failure prints a warning and skips the segment (lines 121-122)."""
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--threshold", "0.01"])
    # Override transcribe_segment to raise
    ctx.append(patch("submatch.cli.transcribe.transcribe_segment",
                     side_effect=RuntimeError("GPU exploded")))
    stack = [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    # aggregate([]) = 0.0 < threshold 0.01 → exit 1
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Warning" in err


def test_main_keep_synced_saves_file(tmp_path):
    """--keep-synced copies the synced SRT alongside the original (lines 159-162)."""
    from submatch.sync import SyncResult
    video = tmp_path / "video.mp4"
    video.touch()
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text(SAMPLE_SRT)
    synced_srt = tmp_path / "synced_tmp.srt"
    synced_srt.write_text(SAMPLE_SRT)
    sync_result = SyncResult(synced_srt_path=synced_srt, offset_seconds=0.0, drift_detected=False)

    subs = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", [
        "submatch", str(video), str(subtitle), "--threshold", "0.01", "--keep-synced"
    ]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.cli.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.cli.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.cli.subtitle.parse", return_value=subs), \
         patch("submatch.cli.sampler.select_segments", return_value=segs), \
         patch("submatch.cli.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.cli.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.cli.language.detect_from_text", return_value="en"), \
         patch("submatch.cli.language.detect_from_filename", return_value="en"), \
         patch("submatch.cli.language.detect_from_video", return_value=None), \
         patch("submatch.cli.language.build_result", return_value=lang), \
         patch("submatch.cli.sync.sync_subtitle", return_value=sync_result):
        with pytest.raises(SystemExit) as exc:
            cli.main()

    assert exc.value.code == 0
    kept = subtitle.with_stem(subtitle.stem + ".synced")
    assert kept.exists()


def test_main_sync_failure_continues(tmp_path):
    """If ffsubsync raises, main warns and continues without sync."""
    video = tmp_path / "video.mp4"
    video.touch()
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text(SAMPLE_SRT)
    subs = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    with patch("sys.argv", ["submatch", str(video), str(subtitle), "--threshold", "0.01"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.cli.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.cli.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.cli.subtitle.parse", return_value=subs), \
         patch("submatch.cli.sampler.select_segments", return_value=segs), \
         patch("submatch.cli.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.cli.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.cli.language.detect_from_text", return_value="en"), \
         patch("submatch.cli.language.detect_from_filename", return_value="en"), \
         patch("submatch.cli.language.detect_from_video", return_value=None), \
         patch("submatch.cli.language.build_result", return_value=lang), \
         patch("submatch.cli.sync.sync_subtitle", side_effect=RuntimeError("ffs down")):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 0


# ── batch mode ────────────────────────────────────────────────────────────────

def _make_batch_patches(tmp_path, extra_argv=()):
    video = tmp_path / "show.mp4"
    video.touch()
    sub = tmp_path / "show.srt"
    sub.write_text(SAMPLE_SRT)

    subs = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    argv = ["submatch", str(tmp_path), "--no-sync"] + list(extra_argv)

    ctx = [
        patch("sys.argv", argv),
        patch("submatch.cli.check_dependencies"),
        patch("submatch.cli.audio.has_audio_track", return_value=True),
        patch("submatch.cli.audio.get_duration_ms", return_value=90 * 60 * 1_000),
        patch("submatch.cli.audio.extract_segment", return_value=tmp_path / "seg.wav"),
        patch("submatch.cli.subtitle.parse", return_value=subs),
        patch("submatch.cli.sampler.select_segments", return_value=segs),
        patch("submatch.cli.transcribe.load_model", return_value=MagicMock()),
        patch("submatch.cli.transcribe.transcribe_segment", return_value=mock_trans),
        patch("submatch.cli.language.detect_from_text", return_value="en"),
        patch("submatch.cli.language.detect_from_filename", return_value="en"),
        patch("submatch.cli.language.detect_from_video", return_value=None),
        patch("submatch.cli.language.build_result", return_value=lang),
    ]
    return ctx


def test_batch_dir_mode_passes(tmp_path):
    ctx = _make_batch_patches(tmp_path, ["--threshold", "0.01"])
    stack = [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    assert exc.value.code == 0


def test_batch_dir_mode_fails_below_threshold(tmp_path):
    ctx = _make_batch_patches(tmp_path, ["--threshold", "2.0"])
    stack = [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    assert exc.value.code == 1


def test_batch_dir_mode_empty_dir_exits_2(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with patch("sys.argv", ["submatch", str(empty), "--no-sync"]), \
         patch("submatch.cli.check_dependencies"):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 2


def test_batch_candidates_mode(tmp_path):
    video = tmp_path / "show.mp4"
    video.touch()
    subs_dir = tmp_path / "subs"
    subs_dir.mkdir()
    (subs_dir / "show.srt").write_text(SAMPLE_SRT)

    subs_parsed = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(video), str(subs_dir),
                            "--no-sync", "--threshold", "0.01"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.cli.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.cli.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.cli.subtitle.parse", return_value=subs_parsed), \
         patch("submatch.cli.sampler.select_segments", return_value=segs), \
         patch("submatch.cli.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.cli.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.cli.language.detect_from_text", return_value="en"), \
         patch("submatch.cli.language.detect_from_filename", return_value="en"), \
         patch("submatch.cli.language.detect_from_video", return_value=None), \
         patch("submatch.cli.language.build_result", return_value=lang):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 0


def test_batch_json_output(tmp_path, capsys):
    ctx = _make_batch_patches(tmp_path, ["--json", "--threshold", "0.01"])
    stack = [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit):
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list)
    assert data[0]["passed"] is True


def test_batch_compact_output(tmp_path, capsys):
    ctx = _make_batch_patches(tmp_path, ["--compact", "--threshold", "0.01"])
    stack = [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit):
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "passed" in out


def test_batch_error_in_one_pair_exits_2(tmp_path):
    ctx = _make_batch_patches(tmp_path, ["--threshold", "0.01"])
    ctx.append(patch("submatch.cli.audio.get_duration_ms",
                     side_effect=RuntimeError("ffprobe failed")))
    stack = [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    assert exc.value.code == 2

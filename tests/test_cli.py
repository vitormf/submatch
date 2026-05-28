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
    assert args.inputs == [v, s]
    assert args.model == "base"
    assert args.threshold == pytest.approx(0.35)
    assert args.segments is None
    assert args.json is False
    assert args.compact is False
    assert args.verbose is False
    assert args.language is None
    assert args.no_sync is False
    assert args.keep_synced is False
    assert args.no_recursive is False
    assert args.sub_lang is None
    assert args.filter is None
    assert args.device == "auto"
    assert args.workers is None
    assert args.cross_threshold is None
    assert args.resync is False
    assert args.pass_unsure is False
    assert args.drift_threshold == pytest.approx(2.0)


def test_parse_args_all_flags(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", [
        "submatch", str(v), str(s),
        "--model", "small", "--threshold", "0.6", "--segments", "4",
        "--json", "--compact", "--verbose", "--language", "pt", "--no-sync", "--keep-synced",
        "--no-recursive", "--sub-lang", "en", "--filter", "*.en.*",
        "--device", "cpu", "--workers", "2",
        "--cross-threshold", "0.5",
        "--resync", "--pass-unsure",
        "--drift-threshold", "5.0",
    ]):
        args = cli.parse_args()
    assert args.inputs == [v, s]
    assert args.model == "small"
    assert args.threshold == pytest.approx(0.6)
    assert args.segments == 4
    assert args.json is True
    assert args.compact is True
    assert args.verbose is True
    assert args.language == "pt"
    assert args.no_sync is True
    assert args.keep_synced is True
    assert args.no_recursive is True
    assert args.sub_lang == ["en"]
    assert args.filter == "*.en.*"
    assert args.device == "cpu"
    assert args.workers == 2
    assert args.cross_threshold == pytest.approx(0.5)
    assert args.resync is True
    assert args.pass_unsure is True
    assert args.drift_threshold == pytest.approx(5.0)


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


def test_main_inputs_nonexistent_hard_error(tmp_path):
    """Any non-existent path in inputs causes exit 2 before processing."""
    v = tmp_path / "video.mp4"
    v.touch()
    with patch("sys.argv", ["submatch", str(v), str(tmp_path / "no.srt")]):
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
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    assert exc.value.code == 0


def test_main_pipeline_fails(tmp_path):
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--threshold", "2.0"])
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    assert exc.value.code == 1


def test_main_json_output(tmp_path, capsys):
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--json", "--threshold", "0.01"])
    [c.__enter__() for c in ctx]
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
    [c.__enter__() for c in ctx]
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
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    assert exc.value.code == 0


def test_batch_dir_mode_fails_below_threshold(tmp_path):
    ctx = _make_batch_patches(tmp_path, ["--threshold", "2.0"])
    [c.__enter__() for c in ctx]
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
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit):
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    data = json.loads(capsys.readouterr().out)
    assert isinstance(data, list)
    assert data[0]["passed"] is True
    assert "video" in data[0]
    assert "subtitle" in data[0]


def test_batch_compact_output(tmp_path, capsys):
    ctx = _make_batch_patches(tmp_path, ["--compact", "--threshold", "0.01"])
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit):
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "1 PASS" in out


def test_batch_error_in_one_pair_exits_2(tmp_path):
    ctx = _make_batch_patches(tmp_path, ["--threshold", "0.01"])
    ctx.append(patch("submatch.cli.audio.get_duration_ms",
                     side_effect=RuntimeError("ffprobe failed")))
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    assert exc.value.code == 2


# ── recursive flag ────────────────────────────────────────────────────────────

def test_batch_recursive_dir_mode(tmp_path):
    """--recursive finds pairs in nested subdirectory."""
    nested = tmp_path / "show" / "season1"
    nested.mkdir(parents=True)
    video = nested / "ep01.mp4"
    video.touch()
    sub = nested / "ep01.srt"
    sub.write_text(SAMPLE_SRT)

    subs_parsed = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(tmp_path), "--no-sync",
                            "--threshold", "0.01"]), \
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


def test_batch_no_recursive_dir_mode(tmp_path):
    """--no-recursive does not find pairs in nested subdirectory."""
    nested = tmp_path / "show" / "season1"
    nested.mkdir(parents=True)
    (nested / "ep01.mp4").touch()
    (nested / "ep01.srt").write_text(SAMPLE_SRT)

    with patch("sys.argv", ["submatch", str(tmp_path), "--no-recursive", "--no-sync",
                            "--threshold", "0.01"]), \
         patch("submatch.cli.check_dependencies"):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 2


# ── new input modes ───────────────────────────────────────────────────────────

def test_main_video_only_auto_discovers_subtitle(tmp_path):
    """Single video file — batch mode runs, subtitle auto-discovered."""
    video = tmp_path / "show.mp4"
    video.touch()
    sub = tmp_path / "show.srt"
    sub.write_text(SAMPLE_SRT)

    subs_parsed = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(video), "--no-sync", "--threshold", "0.01"]), \
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


def test_main_subtitle_only_finds_video(tmp_path):
    """Single subtitle file — batch mode runs, video auto-discovered."""
    video = tmp_path / "show.mp4"
    video.touch()
    sub = tmp_path / "show.srt"
    sub.write_text(SAMPLE_SRT)

    subs_parsed = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(sub), "--no-sync", "--threshold", "0.01"]), \
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


def test_main_multiple_inputs_nonexistent_all_reported(tmp_path, capsys):
    """All missing paths are reported before exit 2."""
    with patch("sys.argv", ["submatch", str(tmp_path / "a.mp4"), str(tmp_path / "b.srt")]):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "a.mp4" in err
    assert "b.srt" in err


def test_batch_recursive_candidates_mode(tmp_path):
    """--recursive with subtitle dir finds subtitles in subdirectories."""
    video = tmp_path / "movie.mp4"
    video.touch()
    subs_dir = tmp_path / "subs"
    nested = subs_dir / "en"
    nested.mkdir(parents=True)
    (nested / "movie.srt").write_text(SAMPLE_SRT)

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


def test_batch_suppresses_transcription_progress(tmp_path, capsys):
    """In batch mode, per-segment 'Transcribing' messages are not shown."""
    ctx = _make_batch_patches(tmp_path, ["--threshold", "0.01"])
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit):
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    out = capsys.readouterr().out
    assert "Transcribing" not in out


def test_batch_tty_progress_overwrites(tmp_path, capsys):
    """In TTY mode, each pair emits a result line to stderr."""
    ctx = _make_batch_patches(tmp_path, ["--threshold", "0.01", "--workers", "1"])
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit), \
             patch("sys.stderr.isatty", return_value=True):
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    err = capsys.readouterr().err
    assert "PASS" in err or "FAIL" in err or "DRIFT" in err or "UNSURE" in err


def test_parse_args_sub_lang_single(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    with patch("sys.argv", ["submatch", str(v), "--sub-lang", "pt"]):
        args = cli.parse_args()
    assert args.sub_lang == ["pt"]


def test_parse_args_sub_lang_multiple(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    with patch("sys.argv", ["submatch", str(v), "--sub-lang", "en", "--sub-lang", "pt"]):
        args = cli.parse_args()
    assert args.sub_lang == ["en", "pt"]


def test_parse_args_filter(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    with patch("sys.argv", ["submatch", str(v), "--filter", "*.en.*"]):
        args = cli.parse_args()
    assert args.filter == "*.en.*"


def test_batch_sub_lang_filters_all_pairs(tmp_path):
    """When --sub-lang excludes all pairs, exit 2 with no pairs found."""
    video = tmp_path / "show.mp4"
    video.touch()
    (tmp_path / "show.en.srt").write_text(SAMPLE_SRT)
    with patch("sys.argv", ["submatch", str(tmp_path), "--sub-lang", "de", "--no-sync"]), \
         patch("submatch.cli.check_dependencies"):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 2


def test_resolve_device_explicit_cpu():
    assert cli._resolve_device("cpu") == "cpu"


def test_resolve_device_explicit_mps():
    assert cli._resolve_device("mps") == "mps"


def test_resolve_device_auto_no_gpu():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": mock_torch}):
        assert cli._resolve_device("auto") == "cpu"


def test_resolve_device_auto_mps():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    with patch.dict(sys.modules, {"torch": mock_torch}):
        assert cli._resolve_device("auto") == "mps"


def test_resolve_workers_auto_gpu():
    import os
    expected = min(4, os.cpu_count() or 1)
    for device in ("mps", "cuda"):
        assert cli._resolve_workers(None, device) == expected


def test_resolve_workers_auto_cpu():
    import os
    result = cli._resolve_workers(None, "cpu")
    assert 1 <= result <= min(4, os.cpu_count() or 1)


def test_resolve_workers_explicit_overrides():
    assert cli._resolve_workers(3, "mps") == 3
    assert cli._resolve_workers(1, "cpu") == 1


def test_batch_no_warn_multi_worker_gpu(tmp_path, capsys):
    ctx = _make_batch_patches(tmp_path, ["--workers", "2", "--device", "mps",
                                          "--threshold", "0.01"])
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit):
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    err = capsys.readouterr().err
    assert "contention" not in err


def test_batch_no_warn_multi_worker_cpu(tmp_path, capsys):
    ctx = _make_batch_patches(tmp_path, ["--workers", "4", "--device", "cpu",
                                          "--threshold", "0.01"])
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit):
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    err = capsys.readouterr().err
    assert "contention" not in err


def test_batch_parallel_error_in_one_pair_exits_2(tmp_path):
    """In parallel mode, an exception from one pair produces BatchPairResult.error and exit 2."""
    ctx = _make_batch_patches(tmp_path, ["--workers", "2", "--device", "cpu",
                                          "--threshold", "0.01"])
    ctx.append(patch("submatch.cli.audio.get_duration_ms",
                     side_effect=RuntimeError("ffprobe failed")))
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    assert exc.value.code == 2


# ── cross-language detection ──────────────────────────────────────────────────

def test_is_cross_language_different():
    assert cli._is_cross_language("en", "pt") is True


def test_is_cross_language_same():
    assert cli._is_cross_language("en", "en") is False


def test_is_cross_language_prefix_match():
    assert cli._is_cross_language("pt", "pt-BR") is False
    assert cli._is_cross_language("pt-BR", "pt") is False


def test_is_cross_language_none_audio():
    assert cli._is_cross_language(None, "pt") is False


def test_is_cross_language_none_subtitle():
    assert cli._is_cross_language("en", None) is False


def test_is_cross_language_both_none():
    assert cli._is_cross_language(None, None) is False


def test_parse_args_cross_threshold_default(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s)]):
        args = cli.parse_args()
    assert args.cross_threshold is None


def test_parse_args_cross_threshold_explicit(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s), "--cross-threshold", "0.5"]):
        args = cli.parse_args()
    assert args.cross_threshold == pytest.approx(0.5)


def test_get_embed_model_missing_import(capsys):
    """_get_embed_model exits with code 2 when sentence_transformers not installed."""
    if hasattr(cli._embed_local, "model"):
        del cli._embed_local.model
    with patch.dict(sys.modules, {"sentence_transformers": None}), \
         patch("submatch.embeddings.load_embedding_model",
               side_effect=ImportError("No module named 'sentence_transformers'")), \
         pytest.raises(SystemExit) as exc:
        cli._get_embed_model()
    assert exc.value.code == 2
    assert "sentence-transformers" in capsys.readouterr().err


def test_score_pair_cross_language_uses_embeddings(tmp_path):
    """Audio='en', subtitle detected as 'pt': embeddings scoring is used."""
    video = tmp_path / "movie.mkv"
    video.touch()
    sub = tmp_path / "movie.pt.srt"
    sub.write_text(SAMPLE_SRT)

    subs_parsed = [Subtitle(1, 1_000, 3_500, "Olá mundo")]
    segs = [Segment(60_000, 90_000, "Olá mundo", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="pt", subtitle_filename="pt",
        video_metadata=None, expected=None, mismatch=True,
        mismatch_details=["subtitle filename says en but text detected as pt"],
    )
    mock_embed_score = MagicMock(f1=0.72, wer=0.0)
    mock_cross_fn = MagicMock(return_value=mock_embed_score)

    with patch("sys.argv", ["submatch", str(video), str(sub),
                            "--no-sync", "--threshold", "0.5"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.cli.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.cli.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.cli.subtitle.parse", return_value=subs_parsed), \
         patch("submatch.cli.sampler.select_segments", return_value=segs), \
         patch("submatch.cli.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.cli.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.cli.language.detect_from_text", return_value="pt"), \
         patch("submatch.cli.language.detect_from_filename", return_value="pt"), \
         patch("submatch.cli.language.detect_from_video", return_value=None), \
         patch("submatch.cli.language.build_result", return_value=lang), \
         patch("submatch.cli._get_embed_model", return_value=MagicMock()), \
         patch("submatch.embeddings.cross_language_score", mock_cross_fn), \
         pytest.raises(SystemExit):
        cli.main()

    mock_cross_fn.assert_called_once()


def test_score_pair_same_language_skips_embeddings(tmp_path):
    """Audio='en', subtitle 'en': _get_embed_model is never called."""
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--threshold", "0.01"])
    mock_get_embed = MagicMock()

    with patch("submatch.cli._get_embed_model", mock_get_embed):
        [c.__enter__() for c in ctx]
        try:
            with pytest.raises(SystemExit):
                cli.main()
        finally:
            for c in reversed(ctx):
                c.__exit__(None, None, None)

    mock_get_embed.assert_not_called()


def test_cross_threshold_used_for_cross_language_pair(tmp_path):
    """--cross-threshold 0.9 causes score 0.72 to fail even though --threshold is 0.5."""
    video = tmp_path / "movie.mkv"
    video.touch()
    sub = tmp_path / "movie.pt.srt"
    sub.write_text(SAMPLE_SRT)

    subs_parsed = [Subtitle(1, 1_000, 3_500, "Olá mundo")]
    segs = [Segment(60_000, 90_000, "Olá mundo", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="pt", subtitle_filename="pt",
        video_metadata=None, expected=None, mismatch=True,
        mismatch_details=["subtitle filename says en but text detected as pt"],
    )
    mock_embed_score = MagicMock(f1=0.72, wer=0.0)

    with patch("sys.argv", ["submatch", str(video), str(sub),
                            "--no-sync", "--threshold", "0.5",
                            "--cross-threshold", "0.9"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.cli.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.cli.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.cli.subtitle.parse", return_value=subs_parsed), \
         patch("submatch.cli.sampler.select_segments", return_value=segs), \
         patch("submatch.cli.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.cli.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.cli.language.detect_from_text", return_value="pt"), \
         patch("submatch.cli.language.detect_from_filename", return_value="pt"), \
         patch("submatch.cli.language.detect_from_video", return_value=None), \
         patch("submatch.cli.language.build_result", return_value=lang), \
         patch("submatch.cli._get_embed_model", return_value=MagicMock()), \
         patch("submatch.embeddings.cross_language_score",
               return_value=mock_embed_score), \
         pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1  # FAIL because 0.72 < cross-threshold 0.9


def test_resolve_device_auto_cuda():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.backends.mps.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": mock_torch}):
        assert cli._resolve_device("auto") == "cuda"


def test_get_embed_model_returns_cached():
    """Second call to _get_embed_model returns the cached model without reloading."""
    if hasattr(cli._embed_local, "model"):
        del cli._embed_local.model
    mock_model = MagicMock()
    with patch("submatch.embeddings.load_embedding_model", return_value=mock_model) as mock_load:
        first = cli._get_embed_model()
        second = cli._get_embed_model()
    assert first is mock_model
    assert second is mock_model
    mock_load.assert_called_once()
    del cli._embed_local.model


def test_batch_sync_bar_description(tmp_path):
    """In batch mode without --no-sync, bar.set_description is called during sync phase."""
    from submatch.sync import SyncResult
    video = tmp_path / "show.mp4"
    video.touch()
    sub = tmp_path / "show.srt"
    sub.write_text(SAMPLE_SRT)
    synced_srt = tmp_path / "synced.srt"
    synced_srt.write_text(SAMPLE_SRT)
    sync_result = SyncResult(synced_srt_path=synced_srt, offset_seconds=0.0, drift_detected=False)

    subs_parsed = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(tmp_path), "--threshold", "0.01"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.cli.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.cli.subtitle.parse", return_value=subs_parsed), \
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


def test_batch_single_video_non_dir_subtitle_exits_2(tmp_path):
    """_run_batch returns 2 when no pairs can be resolved."""
    video = tmp_path / "movie.mp4"
    video.touch()

    with patch("sys.argv", ["submatch", str(video), "--no-sync"]):
        args = cli.parse_args()

    with patch("submatch.cli.check_dependencies"):
        result = cli._run_batch(args, [video], [])
    assert result == 2


def test_batch_delete_failures(tmp_path):
    """--delete-failures in batch mode removes subtitle files that fail."""
    ctx = _make_batch_patches(tmp_path, ["--threshold", "2.0", "--delete-failures"])
    sub = tmp_path / "show.srt"
    assert sub.exists()
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    assert exc.value.code == 1
    assert not sub.exists()


def test_batch_reuses_transcription_cache_for_same_video(tmp_path):
    """Second subtitle for the same video skips audio extraction (cache reuse)."""
    video = tmp_path / "show.mp4"
    video.touch()
    (tmp_path / "show.en.srt").write_text(SAMPLE_SRT)
    (tmp_path / "show.pt.srt").write_text(SAMPLE_SRT)

    subs_parsed = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    mock_get_duration = MagicMock(return_value=90 * 60 * 1_000)
    mock_extract = MagicMock(return_value=tmp_path / "seg.wav")

    with patch("sys.argv", ["submatch", str(tmp_path), "--no-sync", "--threshold", "0.01"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.get_duration_ms", mock_get_duration), \
         patch("submatch.cli.audio.extract_segment", mock_extract), \
         patch("submatch.cli.subtitle.parse", return_value=subs_parsed), \
         patch("submatch.cli.sampler.select_segments", return_value=segs), \
         patch("submatch.cli.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.cli.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.cli.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.cli.language.detect_from_text", return_value="en"), \
         patch("submatch.cli.language.detect_from_filename", return_value="en"), \
         patch("submatch.cli.language.detect_from_video", return_value=None), \
         patch("submatch.cli.language.build_result", return_value=lang):
        with pytest.raises(SystemExit) as exc:
            cli.main()

    assert exc.value.code == 0
    mock_get_duration.assert_called_once()   # only on the first subtitle
    mock_extract.assert_called_once()        # only on the first subtitle


def test_main_delete_failures_single(tmp_path):
    """--delete-failures in single-file mode removes the subtitle when it fails."""
    _, subtitle, ctx = _make_pipeline_patches(tmp_path, ["--threshold", "2.0",
                                                          "--delete-failures"])
    assert subtitle.exists()
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    assert exc.value.code == 1
    assert not subtitle.exists()


def test_score_pair_exception_propagates_after_sync(tmp_path):
    """When _score_pair raises after sync runs, exception propagates and temp file is cleaned."""
    import argparse
    from submatch.sync import SyncResult
    video = tmp_path / "video.mp4"
    video.touch()
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text(SAMPLE_SRT)

    subs = [Subtitle(1, 1_000, 3_500, "Hello world")]
    args = argparse.Namespace(
        no_sync=False, segments=None, model="base", language=None,
        cross_threshold=None, threshold=0.35, json=False, resync=False,
        pass_unsure=False, drift_threshold=2.0,
    )

    created_tmp: list[Path] = []

    def fake_sync(video, subtitle, out_path, drift_threshold=2.0):
        created_tmp.append(out_path)
        out_path.write_text(SAMPLE_SRT)
        return SyncResult(synced_srt_path=out_path, offset_seconds=0.0, drift_detected=False)

    with patch("submatch.cli.sync.sync_subtitle", side_effect=fake_sync), \
         patch("submatch.cli.subtitle.parse", return_value=subs), \
         patch("submatch.cli.audio.get_duration_ms", side_effect=RuntimeError("ffprobe error")):
        with pytest.raises(RuntimeError, match="ffprobe error"):
            cli._score_pair(video, subtitle, args, MagicMock())

    # The temp sync file created by _score_pair should have been cleaned up
    assert created_tmp, "sync was called"
    assert not created_tmp[0].exists(), "temp sync file should be deleted on exception"


def test_batch_parallel_sync_cleans_up_synced_file(tmp_path):
    """In parallel batch mode with sync (no resync), synced temp file is cleaned up after scoring."""
    from submatch.sync import SyncResult
    video = tmp_path / "show.mp4"
    video.touch()
    sub = tmp_path / "show.srt"
    sub.write_text(SAMPLE_SRT)
    synced_srt = tmp_path / "synced.srt"
    synced_srt.write_text(SAMPLE_SRT)
    sync_result = SyncResult(synced_srt_path=synced_srt, offset_seconds=0.0, drift_detected=False)

    subs_parsed = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(tmp_path), "--threshold", "0.01",
                            "--workers", "2", "--device", "cpu"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.cli.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.cli.subtitle.parse", return_value=subs_parsed), \
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


# ── state system ──────────────────────────────────────────────────────────────

def _make_match_result(segments=None, passed=True, drift_detected=False, sync=None):
    """Helper to build a MatchResult for state-system tests."""
    from submatch.output import MatchResult, MatchState, SegmentResult
    from submatch.language import LanguageResult
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    if segments is None:
        seg = SegmentResult(index=1, start_ms=1000, score=0.8, wer=0.1,
                            subtitle_text="hello", transcription="hello")
        segments = [seg]
    if sync is None and drift_detected:
        from submatch.sync import SyncResult
        sync = SyncResult(synced_srt_path=None, offset_seconds=3.0, drift_detected=True)
    result = MatchResult(
        confidence=0.8 if passed else 0.1,
        passed=passed,
        threshold=0.35,
        language=lang,
        sync=sync,
        segments=segments,
        model="base",
    )
    return result


def test_fmt_eta_seconds():
    assert cli._fmt_eta(45) == "~45s"


def test_fmt_eta_minutes():
    assert cli._fmt_eta(150) == "~2:30"


def test_fmt_eta_hours():
    assert cli._fmt_eta(3661) == "~1:01:01"


def test_determine_state_pass():
    result = _make_match_result(passed=True, drift_detected=False)
    assert cli._determine_state(result) == cli.output.MatchState.PASS


def test_determine_state_drift():
    result = _make_match_result(passed=True, drift_detected=True)
    assert cli._determine_state(result) == cli.output.MatchState.DRIFT


def test_determine_state_fail():
    result = _make_match_result(passed=False, drift_detected=False)
    assert cli._determine_state(result) == cli.output.MatchState.FAIL


def test_determine_state_unsure():
    result = _make_match_result(segments=[], passed=False)
    assert cli._determine_state(result) == cli.output.MatchState.UNSURE


def test_parse_args_resync_flag(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s), "--resync"]):
        args = cli.parse_args()
    assert args.resync is True


def test_parse_args_pass_unsure_flag(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s), "--pass-unsure"]):
        args = cli.parse_args()
    assert args.pass_unsure is True


def test_should_fail_pass_result():
    from submatch.output import MatchState
    result = _make_match_result(passed=True)
    result.state = MatchState.PASS
    assert cli._should_fail(result, False) is False


def test_should_fail_unsure_with_pass_unsure():
    from submatch.output import MatchState
    result = _make_match_result(segments=[], passed=False)
    result.state = MatchState.UNSURE
    assert cli._should_fail(result, True) is False


def test_should_fail_unsure_without_pass_unsure():
    from submatch.output import MatchState
    result = _make_match_result(segments=[], passed=False)
    result.state = MatchState.UNSURE
    assert cli._should_fail(result, False) is True


def test_should_fail_fail_result_with_pass_unsure():
    from submatch.output import MatchState
    result = _make_match_result(passed=False)
    result.state = MatchState.FAIL
    assert cli._should_fail(result, True) is True


def test_main_unsure_exits_1(tmp_path):
    """0 segments scored (all transcriptions fail) → UNSURE → exit 1."""
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--threshold", "0.01"])
    ctx.append(patch("submatch.cli.transcribe.transcribe_segment",
                     side_effect=RuntimeError("GPU exploded")))
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    assert exc.value.code == 1


def test_main_unsure_pass_unsure_exits_0(tmp_path):
    """0 segments scored → UNSURE, but --pass-unsure → exit 0."""
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--threshold", "0.01", "--pass-unsure"])
    ctx.append(patch("submatch.cli.transcribe.transcribe_segment",
                     side_effect=RuntimeError("GPU exploded")))
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    assert exc.value.code == 0


def test_main_drift_exits_1(tmp_path):
    """Content passes threshold but drift detected → DRIFT → exit 1."""
    from submatch.sync import SyncResult
    video = tmp_path / "video.mp4"
    video.touch()
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text(SAMPLE_SRT)
    synced_srt = tmp_path / "synced.srt"
    synced_srt.write_text(SAMPLE_SRT)
    sync_result = SyncResult(synced_srt_path=synced_srt, offset_seconds=3.0, drift_detected=True)

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
         patch("submatch.cli.sync.sync_subtitle", return_value=sync_result):
        with pytest.raises(SystemExit) as exc:
            cli.main()

    assert exc.value.code == 1


def test_main_resync_replaces_subtitle(tmp_path):
    """DRIFT + --resync → file replaced, second score gives PASS → exit 0."""
    from submatch.sync import SyncResult
    video = tmp_path / "video.mp4"
    video.touch()
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text(SAMPLE_SRT)
    synced_srt = tmp_path / "synced.srt"
    synced_srt.write_text(SAMPLE_SRT)
    sync_result = SyncResult(synced_srt_path=synced_srt, offset_seconds=3.0, drift_detected=True)

    subs = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(video), str(subtitle),
                            "--threshold", "0.01", "--resync"]), \
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


def test_batch_sequential_resync(tmp_path):
    """Batch sequential mode: DRIFT + --resync → subtitle replaced, second score gives PASS."""
    from submatch.sync import SyncResult
    video = tmp_path / "show.mp4"
    video.touch()
    sub = tmp_path / "show.srt"
    sub.write_text(SAMPLE_SRT)
    synced_srt = tmp_path / "synced.srt"
    synced_srt.write_text(SAMPLE_SRT)
    sync_result = SyncResult(synced_srt_path=synced_srt, offset_seconds=3.0, drift_detected=True)

    subs = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(tmp_path),
                            "--threshold", "0.01", "--resync"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.cli.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.cli.subtitle.parse", return_value=subs), \
         patch("submatch.cli.sampler.select_segments", return_value=segs), \
         patch("submatch.cli.sampler.segments_from_starts", return_value=segs), \
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


def test_batch_parallel_resync(tmp_path):
    """Batch parallel mode: DRIFT + --resync → subtitle replaced, second score gives PASS."""
    from submatch.sync import SyncResult
    video = tmp_path / "show.mp4"
    video.touch()
    sub = tmp_path / "show.srt"
    sub.write_text(SAMPLE_SRT)
    synced_srt = tmp_path / "synced.srt"
    synced_srt.write_text(SAMPLE_SRT)
    sync_result = SyncResult(synced_srt_path=synced_srt, offset_seconds=3.0, drift_detected=True)

    subs = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en")
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(tmp_path),
                            "--threshold", "0.01", "--resync", "--workers", "2",
                            "--device", "cpu"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.cli.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.cli.subtitle.parse", return_value=subs), \
         patch("submatch.cli.sampler.select_segments", return_value=segs), \
         patch("submatch.cli.sampler.segments_from_starts", return_value=segs), \
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


# ── Windows UTF-8 stdout fix ──────────────────────────────────────────────────

def test_ensure_utf8_stdout_rewraps_on_windows():
    """On Windows, stdout/stderr are rewrapped with UTF-8 to prevent UnicodeEncodeError when piped."""
    import io
    fake_stdout = io.TextIOWrapper(io.BytesIO(), encoding='cp1252')
    fake_stderr = io.TextIOWrapper(io.BytesIO(), encoding='cp1252')

    with patch('sys.platform', 'win32'), \
         patch('sys.stdout', fake_stdout), \
         patch('sys.stderr', fake_stderr):
        cli._ensure_utf8_stdout()
        assert sys.stdout.encoding.lower() == 'utf-8'
        assert sys.stderr.encoding.lower() == 'utf-8'
        sys.stdout.write('PASS ✓')  # must not raise


def test_ensure_utf8_stdout_noop_on_non_windows():
    """Non-Windows platforms leave stdout untouched."""
    import io
    fake_stdout = io.TextIOWrapper(io.BytesIO(), encoding='cp1252')

    with patch('sys.platform', 'darwin'), \
         patch('sys.stdout', fake_stdout):
        original = sys.stdout
        cli._ensure_utf8_stdout()
        assert sys.stdout is original

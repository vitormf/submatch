import concurrent.futures
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from submatch import cli
from submatch import pipeline
from submatch import scoring
from submatch.language import LanguageResult
from submatch.sampler import Segment
from submatch.subtitle import Subtitle
from submatch.types import MatchState
from tests.conftest import SAMPLE_SRT


# ── check_dependencies ────────────────────────────────────────────────────────

def test_check_dependencies_all_present():
    with patch("submatch.cli.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("submatch.cli.gpu.check_gpu_mismatch", return_value=None), \
         patch.dict(sys.modules, {"whisper": MagicMock()}):
        cli.check_dependencies()


def test_check_dependencies_missing_ffmpeg():
    def fake_which(name):
        return None if name == "ffmpeg" else "/usr/bin/ffs"
    with patch("submatch.cli.shutil.which", side_effect=fake_which), \
         patch("submatch.cli.gpu.check_gpu_mismatch", return_value=None), \
         patch.dict(sys.modules, {"whisper": MagicMock()}):
        with pytest.raises(SystemExit) as exc:
            cli.check_dependencies()
    assert exc.value.code == 2


def test_check_dependencies_missing_ffsubsync():
    def fake_which(name):
        return "/usr/bin/ffmpeg" if name == "ffmpeg" else None
    with patch("submatch.cli.shutil.which", side_effect=fake_which), \
         patch("submatch.cli.gpu.check_gpu_mismatch", return_value=None), \
         patch.dict(sys.modules, {"whisper": MagicMock()}):
        with pytest.raises(SystemExit) as exc:
            cli.check_dependencies()
    assert exc.value.code == 2


def test_check_dependencies_skip_sync_ignores_missing_ffs():
    def fake_which(name):
        return "/usr/bin/ffmpeg" if name == "ffmpeg" else None
    with patch("submatch.cli.shutil.which", side_effect=fake_which), \
         patch("submatch.cli.gpu.check_gpu_mismatch", return_value=None), \
         patch.dict(sys.modules, {"whisper": MagicMock()}):
        cli.check_dependencies(skip_sync=True)  # should not raise


def test_check_dependencies_missing_whisper():
    with patch("submatch.cli.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("submatch.cli.gpu.check_gpu_mismatch", return_value=None), \
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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    argv = ["submatch", str(video), str(subtitle), "--no-sync"] + list(extra_argv)

    ctx = [
        patch("sys.argv", argv),
        patch("submatch.cli.check_dependencies"),
        patch("submatch.cli.audio.has_audio_track", return_value=True),
        patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000),
        patch("submatch.scoring.audio.get_audio_track_duration_ms", return_value=90 * 60 * 1_000),
        patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"),
        patch("submatch.scoring.audio.detect_speech_regions", return_value=[]),
        patch("submatch.scoring.subtitle.parse", return_value=subs),
        patch("submatch.scoring.sampler.select_segments", return_value=segs),
        patch("submatch.scoring.sampler.audio_candidate_segments",
              return_value=[[60_000]]),
        patch("submatch.scoring.sampler.segments_from_starts", return_value=segs),
        patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()),
        patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans),
        patch("submatch.scoring.language.detect_from_text", return_value="en"),
        patch("submatch.scoring.language.detect_from_filename", return_value="en"),
        patch("submatch.scoring.language.detect_from_video", return_value=None),
        patch("submatch.scoring.language.build_result", return_value=lang),
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


def test_main_calls_telemetry_init(tmp_path):
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--no-sync"])
    with patch("submatch.cli.telemetry.init") as mock_init, \
         patch("submatch.cli.telemetry.set_mode"):
        [c.__enter__() for c in ctx]
        try:
            with pytest.raises(SystemExit):
                cli.main()
        finally:
            for c in reversed(ctx):
                c.__exit__(None, None, None)
    mock_init.assert_called_once()


def test_main_sets_mode_single(tmp_path):
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--no-sync"])
    with patch("submatch.cli.telemetry.init"), \
         patch("submatch.cli.telemetry.set_mode") as mock_set_mode:
        [c.__enter__() for c in ctx]
        try:
            with pytest.raises(SystemExit):
                cli.main()
        finally:
            for c in reversed(ctx):
                c.__exit__(None, None, None)
    mock_set_mode.assert_called_with("single")


def test_main_json_output(tmp_path):
    json_out = tmp_path / "out.json"
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--json", str(json_out), "--threshold", "0.01"])
    with patch("submatch.report.write_json") as mock_write_json:
        [c.__enter__() for c in ctx]
        try:
            with pytest.raises(SystemExit):
                cli.main()
        finally:
            for c in reversed(ctx):
                c.__exit__(None, None, None)
    mock_write_json.assert_called_once()
    assert mock_write_json.call_args[0][1] == str(json_out)


def test_main_sync_success_reparses_srt(tmp_path):
    """When first pass fails and sync succeeds, main re-parses the synced subtitle."""
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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(video), str(subtitle), "--threshold", "0.01"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.audio_candidate_segments", return_value=[[60_000]]), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang), \
         patch("submatch.scoring.sync.sync_subtitle", return_value=sync_result), \
         patch("submatch.scoring.compare.aggregate", side_effect=[0.0, 1.0]), \
         patch("submatch.scoring.subtitle.parse", return_value=subs) as mock_parse:
        with pytest.raises(SystemExit):
            cli.main()

    # first pass fails (aggregate=0.0), sync runs, second pass passes (aggregate=1.0)
    # parse called twice: once for original, once for synced
    assert mock_parse.call_count == 2


def test_main_segment_transcription_failure_warns(tmp_path, capsys):
    """Transcription failure prints a warning in verbose mode and skips the segment."""
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--threshold", "0.01", "--verbose"])
    # Override transcribe_segment to raise
    ctx.append(patch("submatch.scoring.transcribe.transcribe_segment",
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


def test_main_segment_failure_calls_telemetry_capture(tmp_path):
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--no-sync"])
    boom = RuntimeError("whisper exploded")
    # Override the transcribe_segment mock to raise so capture() is triggered
    ctx.append(patch("submatch.scoring.transcribe.transcribe_segment", side_effect=boom))
    with patch("submatch.cli.telemetry.init"), \
         patch("submatch.cli.telemetry.set_mode"), \
         patch("submatch.pipeline.telemetry.capture") as mock_capture:
        [c.__enter__() for c in ctx]
        try:
            with pytest.raises(SystemExit):
                cli.main()
        finally:
            for c in reversed(ctx):
                c.__exit__(None, None, None)
    mock_capture.assert_called_with(boom)


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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", [
        "submatch", str(video), str(subtitle), "--threshold", "0.01", "--keep-synced"
    ]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.subtitle.parse", return_value=subs), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.audio_candidate_segments", return_value=[[60_000]]), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang), \
         patch("submatch.scoring.sync.sync_subtitle", return_value=sync_result), \
         patch("submatch.scoring.compare.aggregate", side_effect=[0.0, 1.0]):
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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    with patch("sys.argv", ["submatch", str(video), str(subtitle), "--threshold", "0.01"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.subtitle.parse", return_value=subs), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.audio_candidate_segments", return_value=[[60_000]]), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang), \
         patch("submatch.scoring.sync.sync_subtitle", side_effect=RuntimeError("ffs down")):
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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    argv = ["submatch", str(tmp_path), "--no-sync"] + list(extra_argv)

    ctx = [
        patch("sys.argv", argv),
        patch("submatch.cli.check_dependencies"),
        patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000),
        patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"),
        patch("submatch.scoring.audio.detect_speech_regions", return_value=[]),
        patch("submatch.scoring.subtitle.parse", return_value=subs),
        patch("submatch.scoring.sampler.select_segments", return_value=segs),
        patch("submatch.scoring.sampler.audio_candidate_segments",
              return_value=[[60_000]]),
        patch("submatch.scoring.sampler.segments_from_starts", return_value=segs),
        patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()),
        patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans),
        patch("submatch.scoring.language.detect_from_text", return_value="en"),
        patch("submatch.scoring.language.detect_from_filename", return_value="en"),
        patch("submatch.scoring.language.detect_from_video", return_value=None),
        patch("submatch.scoring.language.build_result", return_value=lang),
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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(video), str(subs_dir),
                            "--no-sync", "--threshold", "0.01"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.subtitle.parse", return_value=subs_parsed), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.audio_candidate_segments", return_value=[[60_000]]), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 0


def test_batch_json_output(tmp_path):
    json_out = tmp_path / "out.json"
    ctx = _make_batch_patches(tmp_path, ["--json", str(json_out), "--threshold", "0.01"])
    with patch("submatch.report.write_json") as mock_write_json:
        [c.__enter__() for c in ctx]
        try:
            with pytest.raises(SystemExit):
                cli.main()
        finally:
            for c in reversed(ctx):
                c.__exit__(None, None, None)
    mock_write_json.assert_called_once()
    assert mock_write_json.call_args[0][1] == str(json_out)


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


def _make_two_pair_patches(tmp_path, extra_argv=()):
    """Two videos, one subtitle each — produces two pairs for inline-output tests."""
    for name in ("alpha.mp4", "beta.mp4"):
        (tmp_path / name).touch()
    (tmp_path / "alpha.srt").write_text(SAMPLE_SRT)
    (tmp_path / "beta.srt").write_text(SAMPLE_SRT)

    subs = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    argv = ["submatch", str(tmp_path), "--no-sync"] + list(extra_argv)

    return [
        patch("sys.argv", argv),
        patch("submatch.cli.check_dependencies"),
        patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000),
        patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"),
        patch("submatch.scoring.audio.detect_speech_regions", return_value=[]),
        patch("submatch.scoring.subtitle.parse", return_value=subs),
        patch("submatch.scoring.sampler.select_segments", return_value=segs),
        patch("submatch.scoring.sampler.audio_candidate_segments",
              return_value=[[60_000]]),
        patch("submatch.scoring.sampler.segments_from_starts", return_value=segs),
        patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()),
        patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans),
        patch("submatch.scoring.language.detect_from_text", return_value="en"),
        patch("submatch.scoring.language.detect_from_filename", return_value="en"),
        patch("submatch.scoring.language.detect_from_video", return_value=None),
        patch("submatch.scoring.language.build_result", return_value=lang),
    ]


def _run_patches(ctx):
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit):
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)


def test_batch_verbose_inline_sequential(tmp_path, capsys):
    """Standard mode, workers=1: both subtitles appear in stdout before the summary."""
    _run_patches(_make_two_pair_patches(tmp_path, ["--threshold", "0.01", "--workers", "1"]))
    out = capsys.readouterr().out
    assert "alpha.srt" in out
    assert "beta.srt" in out
    summary_pos = out.find("Results:")
    assert out.find("alpha.srt") < summary_pos
    assert out.find("beta.srt") < summary_pos


def test_batch_compact_inline_sequential(tmp_path, capsys):
    """Compact mode, workers=1: both subtitles appear in stdout before the summary."""
    _run_patches(_make_two_pair_patches(tmp_path, ["--compact", "--threshold", "0.01", "--workers", "1"]))
    out = capsys.readouterr().out
    assert "alpha.srt" in out
    assert "beta.srt" in out
    summary_pos = out.find("Results:")
    assert out.find("alpha.srt") < summary_pos
    assert out.find("beta.srt") < summary_pos


def test_batch_verbose_inline_parallel(tmp_path, capsys):
    """Standard mode, workers=2: both subtitles appear in stdout before the summary."""
    _run_patches(_make_two_pair_patches(tmp_path, ["--threshold", "0.01", "--workers", "2", "--device", "cpu"]))
    out = capsys.readouterr().out
    assert "alpha.srt" in out
    assert "beta.srt" in out
    summary_pos = out.find("Results:")
    assert out.find("alpha.srt") < summary_pos
    assert out.find("beta.srt") < summary_pos


def test_batch_compact_inline_parallel(tmp_path, capsys):
    """Compact mode, workers=2: both subtitles appear in stdout before the summary."""
    _run_patches(_make_two_pair_patches(tmp_path, ["--compact", "--threshold", "0.01", "--workers", "2", "--device", "cpu"]))
    out = capsys.readouterr().out
    assert "alpha.srt" in out
    assert "beta.srt" in out
    summary_pos = out.find("Results:")
    assert out.find("alpha.srt") < summary_pos
    assert out.find("beta.srt") < summary_pos


def test_batch_verbose_printed_after_each_score_sequential(tmp_path, capsys):
    """Sequential: print_human is called right after each _score_pair, not deferred."""
    events = []
    original_score_pair = scoring._score_pair

    def tracked_score(video, sub, *args, **kwargs):
        result = original_score_pair(video, sub, *args, **kwargs)
        events.append("scored")
        return result

    from submatch import output as _output
    original_print_human = _output.print_human

    def tracked_print(*args, **kwargs):
        events.append("printed")
        return original_print_human(*args, **kwargs)

    ctx = _make_two_pair_patches(tmp_path, ["--threshold", "0.01", "--workers", "1"])
    ctx += [
        patch("submatch.scoring._score_pair", side_effect=tracked_score),
        patch("submatch.output.print_human", side_effect=tracked_print),
    ]
    _run_patches(ctx)

    assert events == ["scored", "printed", "scored", "printed"]


def test_batch_compact_printed_after_each_score_sequential(tmp_path, capsys):
    """Sequential compact: print_batch_compact is called right after each _score_pair."""
    events = []
    original_score_pair = scoring._score_pair

    def tracked_score(video, sub, *args, **kwargs):
        result = original_score_pair(video, sub, *args, **kwargs)
        events.append("scored")
        return result

    from submatch import output as _output
    original_print_compact = _output.print_batch_compact

    def tracked_print(*args, **kwargs):
        events.append("printed")
        return original_print_compact(*args, **kwargs)

    ctx = _make_two_pair_patches(tmp_path, ["--compact", "--threshold", "0.01", "--workers", "1"])
    ctx += [
        patch("submatch.scoring._score_pair", side_effect=tracked_score),
        patch("submatch.output.print_batch_compact", side_effect=tracked_print),
    ]
    _run_patches(ctx)

    assert events == ["scored", "printed", "scored", "printed"]


def test_batch_verbose_printed_inside_as_completed_parallel(tmp_path, capsys):
    """Parallel: print_human is called inside the as_completed loop, not after it."""
    print_count = [0]
    counts_at_loop_end = []

    from submatch import output as _output
    original_print_human = _output.print_human

    def tracked_print(*args, **kwargs):
        print_count[0] += 1
        return original_print_human(*args, **kwargs)

    original_as_completed = concurrent.futures.as_completed

    def tracking_as_completed(fs, **kwargs):
        yield from original_as_completed(fs, **kwargs)
        counts_at_loop_end.append(print_count[0])

    ctx = _make_two_pair_patches(tmp_path, ["--threshold", "0.01", "--workers", "2", "--device", "cpu"])
    ctx += [
        patch("submatch.output.print_human", side_effect=tracked_print),
        patch("concurrent.futures.as_completed", side_effect=tracking_as_completed),
    ]
    _run_patches(ctx)

    assert print_count[0] == 2                       # both pairs printed
    assert counts_at_loop_end == [print_count[0]]    # all prints happened inside the loop


def test_batch_compact_printed_inside_as_completed_parallel(tmp_path, capsys):
    """Parallel compact: print_batch_compact is called inside the as_completed loop."""
    print_count = [0]
    counts_at_loop_end = []

    from submatch import output as _output
    original_print_compact = _output.print_batch_compact

    def tracked_print(*args, **kwargs):
        print_count[0] += 1
        return original_print_compact(*args, **kwargs)

    original_as_completed = concurrent.futures.as_completed

    def tracking_as_completed(fs, **kwargs):
        yield from original_as_completed(fs, **kwargs)
        counts_at_loop_end.append(print_count[0])

    ctx = _make_two_pair_patches(tmp_path, ["--compact", "--threshold", "0.01", "--workers", "2", "--device", "cpu"])
    ctx += [
        patch("submatch.output.print_batch_compact", side_effect=tracked_print),
        patch("concurrent.futures.as_completed", side_effect=tracking_as_completed),
    ]
    _run_patches(ctx)

    assert print_count[0] == 2
    assert counts_at_loop_end == [print_count[0]]


def test_batch_error_in_one_pair_exits_2(tmp_path):
    ctx = _make_batch_patches(tmp_path, ["--threshold", "0.01"])
    ctx.append(patch("submatch.scoring.audio.get_duration_ms",
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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(tmp_path), "--no-sync",
                            "--threshold", "0.01"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.subtitle.parse", return_value=subs_parsed), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.audio_candidate_segments", return_value=[[60_000]]), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang):
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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(video), "--no-sync", "--threshold", "0.01"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.subtitle.parse", return_value=subs_parsed), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.audio_candidate_segments", return_value=[[60_000]]), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang):
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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(sub), "--no-sync", "--threshold", "0.01"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.subtitle.parse", return_value=subs_parsed), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.audio_candidate_segments", return_value=[[60_000]]), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang):
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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(video), str(subs_dir),
                            "--no-sync", "--threshold", "0.01"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.subtitle.parse", return_value=subs_parsed), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.audio_candidate_segments", return_value=[[60_000]]), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang):
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
    """In TTY mode, each pair emits a result line (status visible in stdout via print_human)."""
    ctx = _make_batch_patches(tmp_path, ["--threshold", "0.01", "--workers", "1"])
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit), \
             patch("sys.stderr.isatty", return_value=True):
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "PASS" in combined or "FAIL" in combined or "DRIFT" in combined or "UNSURE" in combined


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
    assert pipeline._resolve_device("cpu") == "cpu"


def test_resolve_device_explicit_mps():
    assert pipeline._resolve_device("mps") == "mps"


def test_resolve_device_auto_no_gpu():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": mock_torch}):
        assert pipeline._resolve_device("auto") == "cpu"


def test_resolve_device_auto_mps_falls_back_to_cpu():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    with patch.dict(sys.modules, {"torch": mock_torch}):
        assert pipeline._resolve_device("auto") == "cpu"


def test_resolve_workers_auto_cuda_is_single():
    assert pipeline._resolve_workers(None, "cuda") == 1


def test_resolve_workers_auto_mps_uses_multi():
    import os
    assert pipeline._resolve_workers(None, "mps") == min(4, os.cpu_count() or 1)


def test_resolve_workers_auto_cpu():
    import os
    result = pipeline._resolve_workers(None, "cpu")
    assert 1 <= result <= min(4, os.cpu_count() or 1)


def test_resolve_workers_explicit_overrides():
    assert pipeline._resolve_workers(3, "mps") == 3
    assert pipeline._resolve_workers(1, "cpu") == 1


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
    ctx.append(patch("submatch.scoring.audio.get_duration_ms",
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
    assert scoring._is_cross_language("en", "pt") is True


def test_is_cross_language_same():
    assert scoring._is_cross_language("en", "en") is False


def test_is_cross_language_prefix_match():
    assert scoring._is_cross_language("pt", "pt-BR") is False
    assert scoring._is_cross_language("pt-BR", "pt") is False


def test_is_cross_language_none_audio():
    assert scoring._is_cross_language(None, "pt") is False


def test_is_cross_language_none_subtitle():
    assert scoring._is_cross_language("en", None) is False


def test_is_cross_language_both_none():
    assert scoring._is_cross_language(None, None) is False


def test_get_embed_model_missing_import():
    """scoring._get_embed_model raises ImportError when sentence_transformers not installed."""
    if hasattr(scoring._embed_local, "model"):
        del scoring._embed_local.model
    with patch.dict(sys.modules, {"sentence_transformers": None}), \
         patch("submatch.embeddings.load_embedding_model",
               side_effect=ImportError("No module named 'sentence_transformers'")), \
         pytest.raises(ImportError, match="sentence-transformers"):
        scoring._get_embed_model()


def test_score_pair_cross_language_uses_embeddings(tmp_path):
    """Audio='en', subtitle detected as 'pt': embeddings scoring is used."""
    video = tmp_path / "movie.mkv"
    video.touch()
    sub = tmp_path / "movie.pt.srt"
    sub.write_text(SAMPLE_SRT)

    subs_parsed = [Subtitle(1, 1_000, 3_500, "Olá mundo")]
    segs = [Segment(60_000, 90_000, "Olá mundo", 2)]
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
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
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.subtitle.parse", return_value=subs_parsed), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.audio_candidate_segments", return_value=[[60_000]]), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="pt"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="pt"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang), \
         patch("submatch.scoring._get_embed_model", return_value=MagicMock()), \
         patch("submatch.embeddings.cross_language_score", mock_cross_fn), \
         pytest.raises(SystemExit):
        cli.main()

    mock_cross_fn.assert_called_once()


def test_score_pair_same_language_skips_embeddings(tmp_path):
    """Audio='en', subtitle 'en': _get_embed_model is never called."""
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--threshold", "0.01"])
    mock_get_embed = MagicMock()

    with patch("submatch.scoring._get_embed_model", mock_get_embed):
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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
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
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.subtitle.parse", return_value=subs_parsed), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.audio_candidate_segments", return_value=[[60_000]]), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="pt"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="pt"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang), \
         patch("submatch.scoring._get_embed_model", return_value=MagicMock()), \
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
        assert pipeline._resolve_device("auto") == "cuda"


def test_get_embed_model_returns_cached():
    """Second call to _get_embed_model returns the cached model without reloading."""
    if hasattr(scoring._embed_local, "model"):
        del scoring._embed_local.model
    mock_model = MagicMock()
    with patch("submatch.embeddings.load_embedding_model", return_value=mock_model) as mock_load:
        first = scoring._get_embed_model()
        second = scoring._get_embed_model()
    assert first is mock_model
    assert second is mock_model
    mock_load.assert_called_once()
    del scoring._embed_local.model


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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(tmp_path), "--threshold", "0.01"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.subtitle.parse", return_value=subs_parsed), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.audio_candidate_segments", return_value=[[60_000]]), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang), \
         patch("submatch.scoring.sync.sync_subtitle", return_value=sync_result):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 0


def test_batch_single_video_non_dir_subtitle_exits_2(tmp_path):
    """_run_batch returns 2 when no pairs can be resolved."""
    video = tmp_path / "movie.mp4"
    video.touch()

    with patch("sys.argv", ["submatch", str(video), "--no-sync"]), \
         patch("submatch.config.load_config", return_value={}):
        args = cli.parse_args()

    with patch("submatch.cli.check_dependencies"):
        result = cli._run_batch(args, [video], [])
    assert result == 2


def test_run_batch_accepts_prebuilt_pairs(tmp_path):
    """When pairs= is provided, resolve_pairs is not called."""
    video = tmp_path / "movie.mp4"
    sub = tmp_path / "movie.srt"
    video.touch()
    sub.touch()

    with patch("sys.argv", ["submatch", str(video), str(sub), "--no-sync"]):
        args = cli.parse_args()

    with patch("submatch.cli.check_dependencies"), \
         patch("submatch.batch.resolve_pairs") as mock_resolve, \
         patch("submatch.scoring._score_pair", side_effect=Exception("stop")):
        cli._run_batch(args, [], [], pairs=[(video, sub)])

    mock_resolve.assert_not_called()


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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    mock_get_duration = MagicMock(return_value=90 * 60 * 1_000)
    mock_extract = MagicMock(return_value=tmp_path / "seg.wav")

    with patch("sys.argv", ["submatch", str(tmp_path), "--no-sync", "--threshold", "0.01"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.scoring.audio.get_duration_ms", mock_get_duration), \
         patch("submatch.scoring.audio.extract_segment", mock_extract), \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.subtitle.parse", return_value=subs_parsed), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.audio_candidate_segments", return_value=[[60_000]]), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang):
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
    """Exception raised after sync completes propagates and the temp sync file is cleaned up."""
    from submatch.sync import SyncResult
    video = tmp_path / "video.mp4"
    video.touch()
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text(SAMPLE_SRT)

    subs = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    mock_wav = MagicMock()
    mock_wav.unlink = MagicMock()
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )
    config = pipeline.PipelineConfig(
        sync=True, segments=None, model="base", language=None,
        cross_threshold=None, threshold=0.35, resync=False, drift_threshold=2.0,
        device="cpu", use_cache=False,
    )

    created_tmp: list[Path] = []

    def fake_sync(video, subtitle, out_path, drift_threshold=2.0, audio_track=0):
        created_tmp.append(out_path)
        out_path.write_text(SAMPLE_SRT)
        return SyncResult(synced_srt_path=out_path, offset_seconds=0.0, drift_detected=False)

    # First aggregate call returns 0.0 (first pass fails → sync triggers).
    # Second aggregate call raises so the exception escapes the inner RuntimeError handler.
    with patch("submatch.scoring.sync.sync_subtitle", side_effect=fake_sync), \
         patch("submatch.scoring.subtitle.parse", return_value=subs), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=mock_wav), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang), \
         patch("submatch.scoring.compare.aggregate",
               side_effect=[0.0, ValueError("scoring error")]):
        with pytest.raises(ValueError, match="scoring error"):
            scoring._score_pair(video, subtitle, config, MagicMock())

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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(tmp_path), "--threshold", "0.01",
                            "--workers", "2", "--device", "cpu"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.subtitle.parse", return_value=subs_parsed), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.audio_candidate_segments", return_value=[[60_000]]), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang), \
         patch("submatch.scoring.sync.sync_subtitle", return_value=sync_result):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 0


# ── state system ──────────────────────────────────────────────────────────────

def _make_match_result(segments=None, passed=True, drift_detected=False, sync=None):
    """Helper to build a MatchResult for state-system tests."""
    from submatch.types import MatchResult, SegmentResult
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


# ── _ProgressTracker ──────────────────────────────────────────────────────────

def test_progress_tracker_eta_header_no_ema():
    from submatch.cli import _ProgressTracker
    tracker = _ProgressTracker(n_total=5)
    assert tracker.eta_header() == "[1/5]"


def test_progress_tracker_eta_header_with_ema():
    from submatch.cli import _ProgressTracker
    tracker = _ProgressTracker(n_total=5)
    tracker.ema_pair_time = 10.0
    tracker.pair_idx = 1
    header = tracker.eta_header()
    assert "[2/5" in header
    assert "%" in header


def test_progress_tracker_advance_updates_state():
    from submatch.cli import _ProgressTracker
    tracker = _ProgressTracker(n_total=3)
    tracker.advance(10.0)
    assert tracker.pair_idx == 1
    assert tracker.ema_pair_time == 10.0
    tracker.advance(5.0)
    assert tracker.pair_idx == 2
    assert abs(tracker.ema_pair_time - (0.3 * 5.0 + 0.7 * 10.0)) < 0.001


def test_determine_state_pass():
    result = _make_match_result(passed=True, drift_detected=False)
    assert scoring._determine_state(result) == MatchState.PASS


def test_determine_state_drift():
    result = _make_match_result(passed=True, drift_detected=True)
    assert scoring._determine_state(result) == MatchState.DRIFT


def test_determine_state_fail():
    result = _make_match_result(passed=False, drift_detected=False)
    assert scoring._determine_state(result) == MatchState.FAIL


def test_determine_state_unsure():
    result = _make_match_result(segments=[], passed=False)
    assert scoring._determine_state(result) == MatchState.UNSURE


def test_main_unsure_exits_1(tmp_path):
    """0 segments scored (all transcriptions fail) → UNSURE → exit 1."""
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--threshold", "0.01"])
    ctx.append(patch("submatch.scoring.transcribe.transcribe_segment",
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
    ctx.append(patch("submatch.scoring.transcribe.transcribe_segment",
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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(video), str(subtitle), "--threshold", "0.01"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.scoring.subtitle.parse", return_value=subs), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang), \
         patch("submatch.scoring.sync.sync_subtitle", return_value=sync_result):
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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(video), str(subtitle),
                            "--threshold", "0.01", "--resync"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.subtitle.parse", return_value=subs), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.audio_candidate_segments", return_value=[[60_000]]), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang), \
         patch("submatch.scoring.sync.sync_subtitle", return_value=sync_result):
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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(tmp_path),
                            "--threshold", "0.01", "--resync"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.scoring.subtitle.parse", return_value=subs), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang), \
         patch("submatch.scoring.sync.sync_subtitle", return_value=sync_result):
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
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(tmp_path),
                            "--threshold", "0.01", "--resync", "--workers", "2",
                            "--device", "cpu"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.scoring.subtitle.parse", return_value=subs), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang), \
         patch("submatch.scoring.sync.sync_subtitle", return_value=sync_result):
        with pytest.raises(SystemExit) as exc:
            cli.main()

    assert exc.value.code == 0


# ── _print_run_summary ────────────────────────────────────────────────────────

def test_run_summary_single_pair(tmp_path, capsys):
    v = tmp_path / "movie.mkv"
    s = tmp_path / "movie.en.srt"
    cli._print_run_summary([(v, s)])
    err = capsys.readouterr().err
    assert "Checking:" in err
    assert "movie.mkv" in err
    assert "movie.en.srt" in err


def test_run_summary_short_list(tmp_path, capsys):
    pairs = [(tmp_path / f"v{i}.mkv", tmp_path / f"v{i}.srt") for i in range(3)]
    cli._print_run_summary(pairs)
    err = capsys.readouterr().err
    assert "Checking 3 pairs:" in err
    assert "v0.mkv" in err
    assert "v2.mkv" in err


def test_run_summary_long_list(tmp_path, capsys):
    pairs = [(tmp_path / f"v{i}.mkv", tmp_path / f"v{i}.srt") for i in range(10)]
    cli._print_run_summary(pairs)
    err = capsys.readouterr().err
    assert "Checking 10 pairs" in err
    assert "10 videos" in err
    assert "10 subtitles" in err


def test_run_summary_long_list_counts_unique_videos(tmp_path, capsys):
    v = tmp_path / "movie.mkv"
    pairs = [(v, tmp_path / f"movie.{lang}.srt") for lang in ("en", "pt", "de",
                                                                "fr", "es", "it",
                                                                "ja", "zh", "ko")]
    cli._print_run_summary(pairs)
    err = capsys.readouterr().err
    assert "1 video," in err
    assert "9 subtitles" in err


def test_main_single_pair_prints_summary(tmp_path, capsys):
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--threshold", "0.01"])
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit):
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    err = capsys.readouterr().err
    assert "Checking:" in err
    assert "video.mp4" in err
    assert "sub.srt" in err


def test_main_single_pair_json_still_shows_summary(tmp_path, capsys):
    json_out = tmp_path / "out.json"
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--json", str(json_out), "--threshold", "0.01"])
    with patch("submatch.report.write_json"):
        [c.__enter__() for c in ctx]
        try:
            with pytest.raises(SystemExit):
                cli.main()
        finally:
            for c in reversed(ctx):
                c.__exit__(None, None, None)
    err = capsys.readouterr().err
    assert "Checking:" in err


def test_batch_dir_mode_prints_summary(tmp_path, capsys):
    ctx = _make_batch_patches(tmp_path, ["--threshold", "0.01"])
    [c.__enter__() for c in ctx]
    try:
        with pytest.raises(SystemExit):
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    err = capsys.readouterr().err
    assert "Checking" in err
    assert "show.mp4" in err
    assert "show.srt" in err


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


# ── audio track threading ──────────────────────────────────────────────────────

def test_score_pair_resolve_audio_track_called_once_per_video(tmp_path):
    """resolve_audio_track is called exactly once even for two subtitles sharing a video."""
    video = tmp_path / "show.mp4"
    video.touch()
    sub1 = tmp_path / "show.en.srt"
    sub2 = tmp_path / "show.jp.srt"
    sub1.write_text(SAMPLE_SRT)
    sub2.write_text(SAMPLE_SRT)

    subs = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(tmp_path),
                             "--no-sync", "--compact", "--audio-track", "1"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav"), \
         patch("submatch.scoring.audio.resolve_audio_track", return_value=(1, "jpn")) as mock_resolve, \
         patch("submatch.scoring.subtitle.parse", return_value=subs), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.scoring.sampler.segments_from_starts", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang):
        with pytest.raises(SystemExit):
            cli.main()

    assert mock_resolve.call_count == 1


def test_score_pair_passes_audio_track_to_extract_segment(tmp_path):
    """extract_segment is called with audio_track=1 when --audio-track 1 is used."""
    video = tmp_path / "video.mp4"
    video.touch()
    sub = tmp_path / "sub.srt"
    sub.write_text(SAMPLE_SRT)

    subs = [Subtitle(1, 1_000, 3_500, "Hello world")]
    segs = [Segment(60_000, 90_000, "Hello world", 2)]
    mock_trans = MagicMock(text="hello world", language="en", no_speech_prob=0.0, avg_logprob=0.5)
    lang = LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )

    with patch("sys.argv", ["submatch", str(video), str(sub), "--no-sync", "--audio-track", "1"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=90 * 60 * 1_000), \
         patch("submatch.scoring.audio.extract_segment", return_value=tmp_path / "seg.wav") as mock_extract, \
         patch("submatch.scoring.audio.resolve_audio_track", return_value=(1, "jpn")), \
         patch("submatch.scoring.subtitle.parse", return_value=subs), \
         patch("submatch.scoring.sampler.select_segments", return_value=segs), \
         patch("submatch.pipeline.transcribe.load_model", return_value=MagicMock()), \
         patch("submatch.scoring.transcribe.transcribe_segment", return_value=mock_trans), \
         patch("submatch.scoring.language.detect_from_text", return_value="en"), \
         patch("submatch.scoring.language.detect_from_filename", return_value="en"), \
         patch("submatch.scoring.language.detect_from_video", return_value=None), \
         patch("submatch.scoring.language.build_result", return_value=lang):
        with pytest.raises(SystemExit):
            cli.main()

    assert mock_extract.call_count >= 1
    for call in mock_extract.call_args_list:
        assert call.kwargs.get("audio_track") == 1 or (len(call.args) > 3 and call.args[3] == 1)


def test_embedded_with_resync_exits_2(tmp_path, capsys):
    v = tmp_path / "v.mkv"
    v.touch()
    with patch("sys.argv", ["submatch", str(v), "--embedded", "--resync"]):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 2
    assert "incompatible" in capsys.readouterr().err.lower()


def test_embedded_with_keep_synced_exits_2(tmp_path, capsys):
    v = tmp_path / "v.mkv"
    v.touch()
    with patch("sys.argv", ["submatch", str(v), "--embedded", "--keep-synced"]):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 2


# ── _run_embedded ──────────────────────────────────────────────────────────────

def _make_embedded_args(tmp_path, extra_flags=None):
    v = tmp_path / "movie.mkv"
    v.touch()
    argv = ["submatch", str(v), "--embedded", "--no-sync"]
    if extra_flags:
        argv += extra_flags
    with patch("sys.argv", argv):
        args = cli.parse_args()
    return args, v


def test_run_embedded_no_tracks_returns_2(tmp_path):
    args, video = _make_embedded_args(tmp_path)
    with patch("submatch.embedded.list_subtitle_tracks", return_value=[]):
        result = cli._run_embedded(args, [video])
    assert result == 2


def test_run_embedded_extraction_failure_skips_track(tmp_path, capsys):
    args, video = _make_embedded_args(tmp_path)
    tracks = [{"index": 0, "lang": "eng", "title": None}]
    with patch("submatch.embedded.list_subtitle_tracks", return_value=tracks), \
         patch("submatch.embedded.extract_all_subtitle_tracks",
               side_effect=Exception("codec error")):
        result = cli._run_embedded(args, [video])
    assert result == 2
    assert "Warning" in capsys.readouterr().err


def test_run_embedded_sub_lang_filters_tracks(tmp_path):
    args, video = _make_embedded_args(tmp_path, ["--sub-lang", "en"])
    tracks = [
        {"index": 0, "lang": "eng", "title": None},
        {"index": 1, "lang": "jpn", "title": None},
    ]
    extracted_tracks = []

    def fake_extract_all(v, trks, dest_dir):
        result = {}
        for t in trks:
            extracted_tracks.append(t["index"])
            dest = dest_dir / f"embedded_s{t['index']}_{t['lang'] or 'und'}.srt"
            dest.touch()
            result[t["index"]] = dest
        return result

    with patch("submatch.embedded.list_subtitle_tracks", return_value=tracks), \
         patch("submatch.embedded.extract_all_subtitle_tracks", side_effect=fake_extract_all), \
         patch("submatch.cli._run_batch", return_value=0):
        cli._run_embedded(args, [video])

    assert extracted_tracks == [0]  # only English track extracted


def test_run_embedded_sub_lang_includes_untagged_tracks(tmp_path):
    args, video = _make_embedded_args(tmp_path, ["--sub-lang", "en"])
    tracks = [
        {"index": 0, "lang": None, "title": None},   # unknown lang → included
        {"index": 1, "lang": "jpn", "title": None},  # Japanese → excluded
    ]
    extracted_tracks = []

    def fake_extract_all(v, trks, dest_dir):
        result = {}
        for t in trks:
            extracted_tracks.append(t["index"])
            dest = dest_dir / f"embedded_s{t['index']}_und.srt"
            dest.touch()
            result[t["index"]] = dest
        return result

    with patch("submatch.embedded.list_subtitle_tracks", return_value=tracks), \
         patch("submatch.embedded.extract_all_subtitle_tracks", side_effect=fake_extract_all), \
         patch("submatch.cli._run_batch", return_value=0):
        cli._run_embedded(args, [video])

    assert extracted_tracks == [0]


def test_run_embedded_temp_files_cleaned_up_on_success(tmp_path):
    args, video = _make_embedded_args(tmp_path)
    tracks = [{"index": 0, "lang": "eng", "title": None}]
    captured_dirs = []

    def fake_extract_all(v, trks, dest_dir):
        captured_dirs.append(dest_dir.parent)  # the root tmp dir
        result = {}
        for t in trks:
            dest = dest_dir / f"embedded_s{t['index']}_eng.srt"
            dest.touch()
            result[t["index"]] = dest
        return result

    with patch("submatch.embedded.list_subtitle_tracks", return_value=tracks), \
         patch("submatch.embedded.extract_all_subtitle_tracks", side_effect=fake_extract_all), \
         patch("submatch.cli._run_batch", return_value=0):
        cli._run_embedded(args, [video])

    assert captured_dirs, "extract was never called"
    assert not captured_dirs[0].exists(), "temp dir was not cleaned up"


def test_run_embedded_temp_files_cleaned_up_on_error(tmp_path):
    args, video = _make_embedded_args(tmp_path)
    tracks = [{"index": 0, "lang": "eng", "title": None}]
    captured_dirs = []

    def fake_extract_all(v, trks, dest_dir):
        captured_dirs.append(dest_dir.parent)
        result = {}
        for t in trks:
            dest = dest_dir / f"embedded_s{t['index']}_eng.srt"
            dest.touch()
            result[t["index"]] = dest
        return result

    with patch("submatch.embedded.list_subtitle_tracks", return_value=tracks), \
         patch("submatch.embedded.extract_all_subtitle_tracks", side_effect=fake_extract_all), \
         patch("submatch.cli._run_batch", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            cli._run_embedded(args, [video])

    assert not captured_dirs[0].exists(), "temp dir was not cleaned up after error"


def test_run_embedded_list_tracks_failure_skips_video(tmp_path, capsys):
    args, video = _make_embedded_args(tmp_path)
    with patch("submatch.embedded.list_subtitle_tracks", side_effect=Exception("ffprobe failed")):
        result = cli._run_embedded(args, [video])
    assert result == 2
    assert "Warning" in capsys.readouterr().err


def test_main_embedded_dispatches_to_run_embedded(tmp_path, capsys):
    v = tmp_path / "movie.mkv"
    v.touch()
    with patch("sys.argv", ["submatch", str(v), "--embedded", "--no-sync"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli._run_embedded", return_value=0) as mock_embedded:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 0
    mock_embedded.assert_called_once()
    call_videos = mock_embedded.call_args[0][1]
    assert v in call_videos


# ── watch mode ─────────────────────────────────────────────────────────────────

def test_main_watch_non_directory_exits_2(tmp_path, capsys):
    v = tmp_path / "movie.mkv"
    v.touch()
    with patch("sys.argv", ["submatch", str(v), "--watch"]), \
         patch("submatch.config.load_config", return_value={}), \
         patch("submatch.cli.check_dependencies"):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 2
    assert "directory" in capsys.readouterr().err.lower()


def test_main_watch_multiple_inputs_exits_2(tmp_path, capsys):
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    with patch("sys.argv", ["submatch", str(d1), str(d2), "--watch"]), \
         patch("submatch.config.load_config", return_value={}), \
         patch("submatch.cli.check_dependencies"):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 2
    assert "directory" in capsys.readouterr().err.lower()


def test_main_watch_dispatches_to_run_watch(tmp_path):
    with patch("sys.argv", ["submatch", str(tmp_path), "--watch", "--no-sync"]), \
         patch("submatch.config.load_config", return_value={}), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.watch.run_watch", return_value=0) as mock_watch:
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 0
    mock_watch.assert_called_once()
    assert mock_watch.call_args[0][1] == tmp_path


def test_poll_without_watch_warns(tmp_path, capsys):
    v = tmp_path / "movie.mkv"
    s = tmp_path / "movie.en.srt"
    v.touch()
    s.touch()
    with patch("sys.argv", ["submatch", str(v), str(s), "--poll", "--no-sync"]), \
         patch("submatch.config.load_config", return_value={}), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.scoring._score_pair", side_effect=SystemExit(0)):
        with pytest.raises(SystemExit):
            cli.main()
    assert "warning" in capsys.readouterr().err.lower()


# ── _audio_driven_transcribe ───────────────────────────────────────────────────

def test_audio_driven_transcribe_retries_on_bad_no_speech_prob(tmp_path):
    """Quality gate rejects high no_speech_prob and retries with next candidate."""
    from submatch.scoring import _audio_driven_transcribe
    from submatch.pipeline import PipelineConfig
    from submatch.transcribe import TranscriptionResult

    bad = TranscriptionResult(text="uh", language="en", no_speech_prob=0.9)
    good = TranscriptionResult(text="she left before I could say goodbye", language="en", no_speech_prob=0.1)
    responses = [bad, good]

    mock_wav = MagicMock()
    mock_wav.unlink = MagicMock()

    with patch("submatch.scoring.audio.extract_segment", return_value=mock_wav), \
         patch("submatch.scoring.transcribe.transcribe_segment", side_effect=responses), \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=3_600_000), \
         patch("submatch.scoring.sampler.audio_candidate_segments",
               return_value=[[100_000, 200_000]]):
        starts, texts, lang = _audio_driven_transcribe(
            video=Path("fake.mkv"),
            audio_track_index=0,
            n_seg=1,
            model=MagicMock(),
            config=PipelineConfig(),
        )

    assert len(starts) == 1
    assert texts[0] == "she left before I could say goodbye"


def test_audio_driven_transcribe_fallback_uses_most_words(tmp_path):
    """When all candidates fail quality gate, uses the one with most words."""
    from submatch.scoring import _audio_driven_transcribe
    from submatch.pipeline import PipelineConfig
    from submatch.transcribe import TranscriptionResult

    short = TranscriptionResult(text="uh", language="en", no_speech_prob=0.95)
    longer = TranscriptionResult(text="she left before I could say goodbye there", language="en", no_speech_prob=0.8)
    responses = [short, longer]

    mock_wav = MagicMock()
    mock_wav.unlink = MagicMock()

    with patch("submatch.scoring.audio.extract_segment", return_value=mock_wav), \
         patch("submatch.scoring.transcribe.transcribe_segment", side_effect=responses), \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=3_600_000), \
         patch("submatch.scoring.sampler.audio_candidate_segments",
               return_value=[[100_000, 200_000]]):
        starts, texts, lang = _audio_driven_transcribe(
            video=Path("fake.mkv"),
            audio_track_index=0,
            n_seg=1,
            model=MagicMock(),
            config=PipelineConfig(),
        )

    assert len(starts) == 1
    assert texts[0] == "she left before I could say goodbye there"


def test_audio_driven_transcribe_tied_lang_votes_returns_none():
    """A 1:1 tie between zone language votes must not produce a false cross_language."""
    from submatch.scoring import _audio_driven_transcribe
    from submatch.pipeline import PipelineConfig
    from submatch.transcribe import TranscriptionResult

    zone1 = TranscriptionResult(text="hello world foo bar", language="ko", no_speech_prob=0.1)
    zone2 = TranscriptionResult(text="look at these cakes and pies yeah", language="en", no_speech_prob=0.1)
    responses = [zone1, zone2]

    mock_wav = MagicMock()
    mock_wav.unlink = MagicMock()

    with patch("submatch.scoring.audio.extract_segment", return_value=mock_wav), \
         patch("submatch.scoring.transcribe.transcribe_segment", side_effect=responses), \
         patch("submatch.scoring.audio.detect_speech_regions", return_value=[]), \
         patch("submatch.scoring.audio.get_duration_ms", return_value=3_600_000), \
         patch("submatch.scoring.sampler.audio_candidate_segments",
               return_value=[[100_000], [200_000]]):
        _, _, lang = _audio_driven_transcribe(
            video=Path("fake.mkv"),
            audio_track_index=0,
            n_seg=2,
            model=MagicMock(),
            config=PipelineConfig(),
        )

    assert lang is None


# ── cache flags ────────────────────────────────────────────────────────────────

def test_no_cache_flag_bypasses_disk_cache():
    """--no-cache must not call cache.load or cache.store."""
    from submatch.scoring import _score_pair
    from submatch.pipeline import PipelineConfig

    config = PipelineConfig(
        model="base", threshold=0.35, segments=None, sync=False,
        language=None, verbose=False, audio_track=None,
        cross_threshold=None, resync=False,
        drift_threshold=2.0, use_cache=False,
    )
    model = MagicMock()

    with patch("submatch.cache.load") as mock_load, \
         patch("submatch.cache.store") as mock_store, \
         patch("submatch.scoring.audio.get_duration_ms", return_value=3_600_000), \
         patch("submatch.scoring.sampler.select_segments", return_value=[]), \
         patch("submatch.scoring.subtitle.parse", return_value=[]), \
         patch("submatch.scoring.sync.sync_subtitle", side_effect=RuntimeError("skip")):
        try:
            _score_pair(Path("fake.mkv"), Path("fake.srt"), config, model)
        except Exception:
            pass
        mock_load.assert_not_called()
        mock_store.assert_not_called()


def test_cache_hit_skips_transcription(tmp_path):
    """On a disk cache hit, transcribe_segment must not be called."""
    from submatch.scoring import _score_pair
    from submatch.pipeline import PipelineConfig
    from submatch.cache import VideoCache

    config = PipelineConfig(
        model="base", threshold=0.35, segments=None, sync=False,
        language=None, verbose=False, audio_track=None,
        cross_threshold=None, resync=False,
        drift_threshold=2.0, use_cache=True,
        cache_ttl_days=30, cache_max_mb=200, cache_dir=tmp_path,
    )
    model = MagicMock()

    cached = VideoCache(
        segment_starts=[100_000],
        transcriptions=["hello world this is a test sentence"],
        audio_lang="en",
        audio_track_index=0,
        audio_track_lang=None,
    )

    with patch("submatch.cache.load", return_value=cached), \
         patch("submatch.scoring.transcribe.transcribe_segment") as mock_transcribe, \
         patch("submatch.scoring.audio.get_duration_ms", return_value=3_600_000), \
         patch("submatch.scoring.subtitle.parse", return_value=[]), \
         patch("submatch.scoring.sync.sync_subtitle", side_effect=RuntimeError("skip")):
        try:
            _score_pair(Path("fake.mkv"), Path("fake.srt"), config, model)
        except Exception:
            pass
        mock_transcribe.assert_not_called()


def test_clear_cache_flag(tmp_path, capsys):
    """--clear-cache deletes cache entries and exits 0."""
    import subprocess
    import sys as _sys
    cache_dir = tmp_path / "submatch"
    cache_dir.mkdir()
    (cache_dir / "abc123.json").write_text('{"test": 1}')

    import os
    env = {**os.environ, "SUBMATCH_CACHE_DIR": str(cache_dir)}
    result = subprocess.run(
        [_sys.executable, "-m", "submatch.cli", "--clear-cache"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert list(cache_dir.glob("*.json")) == []


def test_check_dependencies_prints_gpu_warning(capsys):
    with patch("submatch.cli.gpu.check_gpu_mismatch", return_value="Warning: NVIDIA GPU detected"), \
         patch("submatch.cli.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch.dict(sys.modules, {"whisper": MagicMock()}):
        cli.check_dependencies(skip_sync=True)
    assert "NVIDIA GPU detected" in capsys.readouterr().err


def test_check_dependencies_no_output_when_no_gpu_warning(capsys):
    with patch("submatch.cli.gpu.check_gpu_mismatch", return_value=None), \
         patch("submatch.cli.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch.dict(sys.modules, {"whisper": MagicMock()}):
        cli.check_dependencies(skip_sync=True)
    assert "NVIDIA" not in capsys.readouterr().err


# ── additional coverage tests ─────────────────────────────────────────────────

def test_print_run_summary_empty_list(capsys):
    # Covers line 62: early return when n == 0
    cli._print_run_summary([])
    assert capsys.readouterr().err == ""


def test_write_reports_csv_and_html(tmp_path):
    # Covers lines 84, 86: csv and html branches of _write_reports
    import argparse
    args = argparse.Namespace(json=None, csv=tmp_path / "out.csv", html=tmp_path / "out.html")
    with patch("submatch.report.write_csv") as mock_csv, \
         patch("submatch.report.write_html") as mock_html:
        cli._write_reports([], args)
    mock_csv.assert_called_once_with([], tmp_path / "out.csv")
    mock_html.assert_called_once_with([], tmp_path / "out.html")


def test_run_batch_import_error_returns_2(tmp_path, capsys):
    # Covers lines 171-173: ImportError in _run_batch returns exit code 2
    ctx = _make_batch_patches(tmp_path, ["--threshold", "0.01"])
    [c.__enter__() for c in ctx]
    try:
        with patch("submatch.cli._pipeline.run_batch",
                   side_effect=ImportError("sentence-transformers not installed")), \
             pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    assert exc.value.code == 2
    assert "sentence-transformers not installed" in capsys.readouterr().err


def test_main_clear_cache_exits_0(tmp_path, capsys):
    # Covers lines 250-253: --clear-cache path in main()
    with patch("sys.argv", ["submatch", "--clear-cache"]), \
         patch("submatch.cli.telemetry.init"), \
         patch("submatch.cli._scoring._cache_config",
               return_value={"dir": tmp_path / "cache"}), \
         patch("submatch.cli._cache_module.clear", return_value=5), \
         pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert "Cleared 5" in capsys.readouterr().out


def test_main_no_inputs_exits_2(capsys):
    # Covers lines 256-257: missing inputs error in main()
    with patch("sys.argv", ["submatch"]), \
         patch("submatch.cli.telemetry.init"), \
         pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "required" in capsys.readouterr().err


def test_main_single_import_error_exits_2(tmp_path, capsys):
    # Covers lines 314-315: ImportError in single-pair run
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--threshold", "0.01"])
    [c.__enter__() for c in ctx]
    try:
        with patch("submatch.cli._pipeline.run",
                   side_effect=ImportError("sentence-transformers not installed")), \
             pytest.raises(SystemExit) as exc:
            cli.main()
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    assert exc.value.code == 2
    assert "sentence-transformers not installed" in capsys.readouterr().err


def test_run_batch_import_error_calls_telemetry_capture(tmp_path):
    ctx = _make_batch_patches(tmp_path, ["--threshold", "0.01"])
    [c.__enter__() for c in ctx]
    try:
        exc = ImportError("sentence-transformers not installed")
        with patch("submatch.cli._pipeline.run_batch", side_effect=exc), \
             patch("submatch.cli.telemetry.capture") as mock_capture, \
             pytest.raises(SystemExit):
            cli.main()
        mock_capture.assert_called_once_with(exc)
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)


def test_main_single_import_error_calls_telemetry_capture(tmp_path):
    _, _, ctx = _make_pipeline_patches(tmp_path, ["--threshold", "0.01"])
    [c.__enter__() for c in ctx]
    try:
        exc = ImportError("sentence-transformers not installed")
        with patch("submatch.cli._pipeline.run", side_effect=exc), \
             patch("submatch.cli.telemetry.capture") as mock_capture, \
             pytest.raises(SystemExit):
            cli.main()
        mock_capture.assert_called_once_with(exc)
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)


def test_main_uses_static_ffmpeg_when_ffmpeg_missing(capsys):
    # Covers lines 243-244: static_ffmpeg.add_paths() called when ffmpeg not on PATH
    mock_sfmpeg = MagicMock()
    with patch("submatch.cli.shutil.which", return_value=None), \
         patch.dict(sys.modules, {"static_ffmpeg": mock_sfmpeg}), \
         patch("sys.argv", ["submatch"]), \
         patch("submatch.cli.telemetry.init"), \
         pytest.raises(SystemExit):
        cli.main()
    mock_sfmpeg.add_paths.assert_called_once()


def test_main_prints_subtitle_resynced_message(tmp_path, capsys):
    """result.resynced=True triggers 'Subtitle resynced' in main() single-pair path."""
    video = tmp_path / "video.mp4"
    video.touch()
    subtitle = tmp_path / "sub.srt"
    subtitle.write_text(SAMPLE_SRT)

    mock_result = MagicMock()
    mock_result.resynced = True
    mock_result.passed = True
    mock_result.state = MatchState.PASS
    mock_result.sync = None

    with patch("sys.argv", ["submatch", str(video), str(subtitle), "--no-sync"]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=True), \
         patch("submatch.cli._pipeline.run", return_value=mock_result), \
         patch("submatch.cli.output.print_human"), \
         patch("submatch.config.load_config", return_value={}), \
         pytest.raises(SystemExit):
        cli.main()
    out = capsys.readouterr().out
    assert "Subtitle resynced" in out

from __future__ import annotations
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from submatch import watch
from submatch.pipeline import PipelineConfig


# ── helpers ────────────────────────────────────────────────────────────────────

def test_find_pairs_recursive(tmp_path):
    from submatch import batch
    with patch.object(batch, "find_pairs_recursive", return_value=[]) as mock:
        watch._find_pairs(tmp_path, recursive=True)
    mock.assert_called_once_with(tmp_path)


def test_find_pairs_non_recursive(tmp_path):
    from submatch import batch
    with patch.object(batch, "find_pairs", return_value=[]) as mock:
        watch._find_pairs(tmp_path, recursive=False)
    mock.assert_called_once_with(tmp_path)


def test_score_and_print_cleans_up_sync_tmp(tmp_path, capsys):
    video = tmp_path / "movie.mkv"
    sub = tmp_path / "movie.en.srt"
    sync_tmp = tmp_path / "sync.srt"
    sync_tmp.touch()

    result = MagicMock()
    result.sync.synced_srt_path = sync_tmp

    with patch("submatch.pipeline.run", return_value=result), \
         patch("submatch.output.print_human"):
        watch._score_and_print(video, sub, MagicMock())

    assert not sync_tmp.exists()


def test_score_and_print_handles_exception(tmp_path, capsys):
    video = tmp_path / "movie.mkv"
    sub = tmp_path / "movie.en.srt"

    with patch("submatch.pipeline.run", side_effect=RuntimeError("boom")):
        watch._score_and_print(video, sub, MagicMock())  # must not raise

    assert "Error" in capsys.readouterr().err


# ── _score_existing ────────────────────────────────────────────────────────────

def test_score_existing_returns_all_pairs(tmp_path):
    video = tmp_path / "movie.mkv"
    sub = tmp_path / "movie.en.srt"

    args = MagicMock()
    args.no_recursive = False
    args.sub_lang = None
    args.filter = None
    config = MagicMock()

    with patch("submatch.watch._find_pairs", return_value=[(video, sub)]), \
         patch("submatch.watch._score_and_print") as mock_score:
        result = watch._score_existing(args, tmp_path, config)

    assert result == {(video, sub)}
    mock_score.assert_called_once_with(video, sub, config)


def test_score_existing_applies_filter(tmp_path):
    video = tmp_path / "movie.mkv"
    sub_en = tmp_path / "movie.en.srt"
    sub_pt = tmp_path / "movie.pt.srt"

    args = MagicMock()
    args.no_recursive = False
    args.sub_lang = ["en"]
    args.filter = None
    config = MagicMock()

    from submatch import batch
    with patch("submatch.watch._find_pairs", return_value=[(video, sub_en), (video, sub_pt)]), \
         patch.object(batch, "filter_pairs", return_value=[(video, sub_en)]), \
         patch("submatch.watch._score_and_print") as mock_score:
        result = watch._score_existing(args, tmp_path, config)

    assert result == {(video, sub_en)}
    mock_score.assert_called_once_with(video, sub_en, config)


# ── _poll_loop ─────────────────────────────────────────────────────────────────

def test_poll_loop_scores_new_pairs(tmp_path):
    video = tmp_path / "movie.mkv"
    sub = tmp_path / "movie.en.srt"

    args = MagicMock()
    args.no_recursive = False
    args.sub_lang = None
    args.filter = None
    config = MagicMock()
    known: set = set()
    scored = []

    sleep_count = [0]

    def fake_sleep(n):
        sleep_count[0] += 1
        if sleep_count[0] >= 2:
            raise KeyboardInterrupt

    with patch("submatch.watch._find_pairs", return_value=[(video, sub)]), \
         patch("submatch.watch._score_and_print",
               side_effect=lambda v, s, c: scored.append((v, s))), \
         patch("time.sleep", side_effect=fake_sleep):
        with pytest.raises(KeyboardInterrupt):
            watch._poll_loop(config, tmp_path, known, args, 10)

    assert (video, sub) in known
    assert (video, sub) in scored


def test_poll_loop_skips_known_pairs(tmp_path):
    video = tmp_path / "movie.mkv"
    sub = tmp_path / "movie.en.srt"

    args = MagicMock()
    args.no_recursive = False
    args.sub_lang = None
    args.filter = None
    known = {(video, sub)}

    with patch("submatch.watch._find_pairs", return_value=[(video, sub)]), \
         patch("submatch.watch._score_and_print") as mock_score, \
         patch("time.sleep", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            watch._poll_loop(MagicMock(), tmp_path, known, args, 10)

    mock_score.assert_not_called()


# ── _SubtitleEventHandler ──────────────────────────────────────────────────────

def _make_handler(args=None, known=None):
    if args is None:
        args = MagicMock()
        args.sub_lang = None
        args.filter = None
        args.verbose = False
    if known is None:
        known = set()
    return watch._SubtitleEventHandler(MagicMock(), known, threading.Lock(), args)


def _event(src_path: Path, is_directory: bool = False):
    e = MagicMock()
    e.src_path = str(src_path)
    e.is_directory = is_directory
    return e


def test_event_handler_subtitle_scores_pair(tmp_path):
    video = tmp_path / "movie.mkv"
    sub = tmp_path / "movie.en.srt"
    known: set = set()
    handler = _make_handler(known=known)

    with patch("submatch.batch.find_pairs", return_value=[(video, sub)]), \
         patch("submatch.watch._score_and_print") as mock_score:
        handler.on_created(_event(sub))

    assert (video, sub) in known
    mock_score.assert_called_once()


def test_event_handler_video_scores_pair(tmp_path):
    video = tmp_path / "movie.mkv"
    sub = tmp_path / "movie.en.srt"
    known: set = set()
    handler = _make_handler(known=known)

    with patch("submatch.batch.find_pairs", return_value=[(video, sub)]), \
         patch("submatch.watch._score_and_print") as mock_score:
        handler.on_created(_event(video))

    assert (video, sub) in known
    mock_score.assert_called_once()


def test_event_handler_unknown_extension_skips(tmp_path):
    txt = tmp_path / "readme.txt"
    handler = _make_handler()

    with patch("submatch.watch._score_and_print") as mock_score:
        handler.on_created(_event(txt))

    mock_score.assert_not_called()


def test_event_handler_directory_event_skips(tmp_path):
    handler = _make_handler()

    with patch("submatch.watch._score_and_print") as mock_score:
        handler.on_created(_event(tmp_path, is_directory=True))

    mock_score.assert_not_called()


def test_event_handler_skips_known_pair(tmp_path):
    video = tmp_path / "movie.mkv"
    sub = tmp_path / "movie.en.srt"
    known = {(video, sub)}
    handler = _make_handler(known=known)

    with patch("submatch.batch.find_pairs", return_value=[(video, sub)]), \
         patch("submatch.watch._score_and_print") as mock_score:
        handler.on_created(_event(sub))

    mock_score.assert_not_called()


def test_event_handler_applies_sub_lang_filter(tmp_path):
    video = tmp_path / "movie.mkv"
    sub_en = tmp_path / "movie.en.srt"
    sub_pt = tmp_path / "movie.pt.srt"
    known: set = set()

    args = MagicMock()
    args.sub_lang = ["en"]
    args.filter = None
    args.verbose = False
    handler = watch._SubtitleEventHandler(MagicMock(), known, threading.Lock(), args)

    with patch("submatch.batch.find_pairs", return_value=[(video, sub_en), (video, sub_pt)]), \
         patch("submatch.batch.filter_pairs", return_value=[(video, sub_en)]), \
         patch("submatch.watch._score_and_print") as mock_score:
        handler.on_created(_event(sub_en))

    mock_score.assert_called_once()
    assert (video, sub_en) in known
    assert (video, sub_pt) not in known


# ── _native_watch ──────────────────────────────────────────────────────────────

def test_native_watch_oserror_propagates(tmp_path):
    args = MagicMock()
    args.no_recursive = False
    known: set = set()

    mock_observer = MagicMock()
    mock_observer.start.side_effect = OSError("inotify not supported")

    with patch("watchdog.observers.Observer", return_value=mock_observer):
        with pytest.raises(OSError):
            watch._native_watch(MagicMock(), tmp_path, known, args)


def test_native_watch_stops_observer_on_keyboard_interrupt(tmp_path):
    args = MagicMock()
    args.no_recursive = False
    known: set = set()

    mock_observer = MagicMock()
    mock_observer.is_alive.side_effect = [True, KeyboardInterrupt]

    with patch("watchdog.observers.Observer", return_value=mock_observer):
        with pytest.raises(KeyboardInterrupt):
            watch._native_watch(MagicMock(), tmp_path, known, args)

    mock_observer.stop.assert_called_once()
    mock_observer.join.assert_called()


# ── run_watch ──────────────────────────────────────────────────────────────────

def _watch_args(poll: bool = False, interval: int = 10):
    args = MagicMock()
    args.poll = poll
    args.interval = interval
    args.model = "base"
    args.device = "auto"
    args.no_recursive = False
    args.sub_lang = None
    args.filter = None
    args.verbose = False
    return args


def test_run_watch_prints_startup_message(tmp_path, capsys):
    args = _watch_args(poll=True)

    with patch("submatch.pipeline._resolve_device", return_value="cpu"), \
         patch("submatch.pipeline._get_model", return_value=MagicMock()), \
         patch("submatch.cli._args_to_config", return_value=MagicMock()), \
         patch("submatch.watch._score_existing", return_value=set()), \
         patch("submatch.watch._poll_loop", side_effect=KeyboardInterrupt):
        watch.run_watch(args, tmp_path)

    err = capsys.readouterr().err
    assert "Watching" in err
    assert str(tmp_path) in err


def test_run_watch_poll_calls_poll_loop(tmp_path):
    args = _watch_args(poll=True)

    with patch("submatch.pipeline._resolve_device", return_value="cpu"), \
         patch("submatch.pipeline._get_model", return_value=MagicMock()), \
         patch("submatch.cli._args_to_config", return_value=MagicMock()), \
         patch("submatch.watch._score_existing", return_value=set()), \
         patch("submatch.watch._poll_loop", side_effect=KeyboardInterrupt) as mock_poll:
        watch.run_watch(args, tmp_path)

    mock_poll.assert_called_once()


def test_run_watch_default_calls_native_watch(tmp_path):
    args = _watch_args(poll=False)

    with patch("submatch.pipeline._resolve_device", return_value="cpu"), \
         patch("submatch.pipeline._get_model", return_value=MagicMock()), \
         patch("submatch.cli._args_to_config", return_value=MagicMock()), \
         patch("submatch.watch._score_existing", return_value=set()), \
         patch("submatch.watch._native_watch", side_effect=KeyboardInterrupt) as mock_native:
        watch.run_watch(args, tmp_path)

    mock_native.assert_called_once()


def test_run_watch_keyboard_interrupt_returns_0(tmp_path, capsys):
    args = _watch_args(poll=True)

    with patch("submatch.pipeline._resolve_device", return_value="cpu"), \
         patch("submatch.pipeline._get_model", return_value=MagicMock()), \
         patch("submatch.cli._args_to_config", return_value=MagicMock()), \
         patch("submatch.watch._score_existing", return_value=set()), \
         patch("submatch.watch._poll_loop", side_effect=KeyboardInterrupt):
        result = watch.run_watch(args, tmp_path)

    assert result == 0
    assert "Stopped" in capsys.readouterr().err


def test_run_watch_watchdog_oserror_returns_2(tmp_path, capsys):
    args = _watch_args(poll=False)

    with patch("submatch.pipeline._resolve_device", return_value="cpu"), \
         patch("submatch.pipeline._get_model", return_value=MagicMock()), \
         patch("submatch.cli._args_to_config", return_value=MagicMock()), \
         patch("submatch.watch._score_existing", return_value=set()), \
         patch("submatch.watch._native_watch", side_effect=OSError("not supported")):
        result = watch.run_watch(args, tmp_path)

    assert result == 2
    assert "--poll" in capsys.readouterr().err

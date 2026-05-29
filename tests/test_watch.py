from __future__ import annotations
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from submatch import watch


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

    from submatch import output
    result = MagicMock()
    result.sync.synced_srt_path = sync_tmp
    result.state = output.MatchState.PASS

    args = MagicMock()
    args.verbose = False

    with patch("submatch.cli._score_pair", return_value=(result, MagicMock())), \
         patch("submatch.output.print_human"):
        watch._score_and_print(video, sub, args, MagicMock())

    assert not sync_tmp.exists()


def test_score_and_print_handles_exception(tmp_path, capsys):
    video = tmp_path / "movie.mkv"
    sub = tmp_path / "movie.en.srt"
    args = MagicMock()
    args.verbose = False

    with patch("submatch.cli._score_pair", side_effect=RuntimeError("boom")):
        watch._score_and_print(video, sub, args, MagicMock())  # must not raise

    assert "Error" in capsys.readouterr().err


# ── _score_existing ────────────────────────────────────────────────────────────

def test_score_existing_returns_all_pairs(tmp_path):
    video = tmp_path / "movie.mkv"
    sub = tmp_path / "movie.en.srt"

    args = MagicMock()
    args.no_recursive = False
    args.sub_lang = None
    args.filter = None
    model = MagicMock()

    with patch("submatch.watch._find_pairs", return_value=[(video, sub)]), \
         patch("submatch.watch._score_and_print") as mock_score:
        result = watch._score_existing(args, tmp_path, model)

    assert result == {(video, sub)}
    mock_score.assert_called_once_with(video, sub, args, model)


def test_score_existing_applies_filter(tmp_path):
    video = tmp_path / "movie.mkv"
    sub_en = tmp_path / "movie.en.srt"
    sub_pt = tmp_path / "movie.pt.srt"

    args = MagicMock()
    args.no_recursive = False
    args.sub_lang = ["en"]
    args.filter = None
    model = MagicMock()

    from submatch import batch
    with patch("submatch.watch._find_pairs", return_value=[(video, sub_en), (video, sub_pt)]), \
         patch.object(batch, "filter_pairs", return_value=[(video, sub_en)]) as mock_filter, \
         patch("submatch.watch._score_and_print") as mock_score:
        result = watch._score_existing(args, tmp_path, model)

    assert result == {(video, sub_en)}
    mock_score.assert_called_once_with(video, sub_en, args, model)


# ── _poll_loop ─────────────────────────────────────────────────────────────────

def test_poll_loop_scores_new_pairs(tmp_path):
    video = tmp_path / "movie.mkv"
    sub = tmp_path / "movie.en.srt"

    args = MagicMock()
    args.no_recursive = False
    args.sub_lang = None
    args.filter = None
    model = MagicMock()
    known: set = set()
    scored = []

    sleep_count = [0]

    def fake_sleep(n):
        sleep_count[0] += 1
        if sleep_count[0] >= 2:
            raise KeyboardInterrupt

    with patch("submatch.watch._find_pairs", return_value=[(video, sub)]), \
         patch("submatch.watch._score_and_print",
               side_effect=lambda v, s, a, m: scored.append((v, s))), \
         patch("time.sleep", side_effect=fake_sleep):
        with pytest.raises(KeyboardInterrupt):
            watch._poll_loop(args, tmp_path, known, model, 10)

    assert (video, sub) in known
    assert (video, sub) in scored


def test_poll_loop_skips_known_pairs(tmp_path):
    video = tmp_path / "movie.mkv"
    sub = tmp_path / "movie.en.srt"

    args = MagicMock()
    args.no_recursive = False
    args.sub_lang = None
    args.filter = None
    model = MagicMock()
    known = {(video, sub)}

    with patch("submatch.watch._find_pairs", return_value=[(video, sub)]), \
         patch("submatch.watch._score_and_print") as mock_score, \
         patch("time.sleep", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            watch._poll_loop(args, tmp_path, known, model, 10)

    mock_score.assert_not_called()

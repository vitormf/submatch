from __future__ import annotations
from unittest.mock import patch

import pytest

from submatch.args import parse_args


def test_parse_args_defaults(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s)]), \
         patch("submatch.config.load_config", return_value={}):
        args = parse_args()
    assert args.inputs == [v, s]
    assert args.model == "base"
    assert args.threshold == pytest.approx(0.35)
    assert args.segments is None
    assert args.json is None
    assert args.csv is None
    assert args.html is None
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
    assert args.audio_track is None
    assert args.no_cache is False
    assert args.clear_cache is False


def test_parse_args_all_flags(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", [
        "submatch", str(v), str(s),
        "--model", "small", "--threshold", "0.6", "--segments", "4",
        "--json", "out.json", "--csv", "out.csv", "--html", "out.html",
        "--compact", "--verbose", "--language", "pt", "--no-sync", "--keep-synced",
        "--no-recursive", "--sub-lang", "en", "--filter", "*.en.*",
        "--device", "cpu", "--workers", "2",
        "--cross-threshold", "0.5",
        "--resync", "--pass-unsure",
        "--drift-threshold", "5.0",
        "--audio-track", "jp,en",
    ]), patch("submatch.config.load_config", return_value={}):
        args = parse_args()
    assert args.inputs == [v, s]
    assert args.model == "small"
    assert args.threshold == pytest.approx(0.6)
    assert args.segments == 4
    assert args.json == "out.json"
    assert args.csv == "out.csv"
    assert args.html == "out.html"
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
    assert args.audio_track == "jp,en"


def test_parse_args_uses_config_value(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s)]), \
         patch("submatch.config.load_config", return_value={"model": "small", "threshold": 0.5}):
        args = parse_args()
    assert args.model == "small"
    assert args.threshold == pytest.approx(0.5)


def test_parse_args_cli_overrides_config(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s), "--model", "large"]), \
         patch("submatch.config.load_config", return_value={"model": "small"}):
        args = parse_args()
    assert args.model == "large"


def test_parse_args_sub_lang_from_config(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s)]), \
         patch("submatch.config.load_config", return_value={"sub_lang": ["pt", "en"]}):
        args = parse_args()
    assert args.sub_lang == ["pt", "en"]


def test_parse_args_sub_lang_cli_replaces_config(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s), "--sub-lang", "fr"]), \
         patch("submatch.config.load_config", return_value={"sub_lang": ["pt", "en"]}):
        args = parse_args()
    assert args.sub_lang == ["fr"]


def test_parse_args_sub_lang_string_from_config(tmp_path):
    """sub_lang as bare string in config is wrapped in a list, not split into chars."""
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s)]), \
         patch("submatch.config.load_config", return_value={"sub_lang": "pt"}):
        args = parse_args()
    assert args.sub_lang == ["pt"]


def test_parse_args_audio_track_integer(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s), "--audio-track", "2"]), \
         patch("submatch.config.load_config", return_value={}):
        args = parse_args()
    assert args.audio_track == "2"


def test_parse_args_audio_track_language_preference(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s), "--audio-track", "jp,en,pt"]), \
         patch("submatch.config.load_config", return_value={}):
        args = parse_args()
    assert args.audio_track == "jp,en,pt"


def test_parse_args_sub_lang_single(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    with patch("sys.argv", ["submatch", str(v), "--sub-lang", "pt"]), \
         patch("submatch.config.load_config", return_value={}):
        args = parse_args()
    assert args.sub_lang == ["pt"]


def test_parse_args_sub_lang_multiple(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    with patch("sys.argv", ["submatch", str(v), "--sub-lang", "en", "--sub-lang", "pt"]), \
         patch("submatch.config.load_config", return_value={}):
        args = parse_args()
    assert args.sub_lang == ["en", "pt"]


def test_parse_args_filter(tmp_path):
    v = tmp_path / "v"
    v.mkdir()
    with patch("sys.argv", ["submatch", str(v), "--filter", "*.en.*"]), \
         patch("submatch.config.load_config", return_value={}):
        args = parse_args()
    assert args.filter == "*.en.*"


def test_parse_args_cross_threshold_default(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s)]), \
         patch("submatch.config.load_config", return_value={}):
        args = parse_args()
    assert args.cross_threshold is None


def test_parse_args_cross_threshold_explicit(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s), "--cross-threshold", "0.5"]), \
         patch("submatch.config.load_config", return_value={}):
        args = parse_args()
    assert args.cross_threshold == pytest.approx(0.5)


def test_parse_args_resync_flag(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s), "--resync"]), \
         patch("submatch.config.load_config", return_value={}):
        args = parse_args()
    assert args.resync is True


def test_parse_args_pass_unsure_flag(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s), "--pass-unsure"]), \
         patch("submatch.config.load_config", return_value={}):
        args = parse_args()
    assert args.pass_unsure is True


def test_parse_args_json_file(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s), "--json", "out.json"]), \
         patch("submatch.config.load_config", return_value={}):
        args = parse_args()
    assert args.json == "out.json"


def test_parse_args_bare_json_is_error(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s), "--json"]), \
         patch("submatch.config.load_config", return_value={}), \
         pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2


def test_parse_args_csv_html(tmp_path):
    v, s = tmp_path / "v.mp4", tmp_path / "s.srt"
    with patch("sys.argv", ["submatch", str(v), str(s),
                            "--csv", "out.csv", "--html", "out.html"]), \
         patch("submatch.config.load_config", return_value={}):
        args = parse_args()
    assert args.csv == "out.csv"
    assert args.html == "out.html"


def test_parse_args_embedded_default_false(tmp_path):
    v = tmp_path / "v.mkv"
    v.touch()
    with patch("sys.argv", ["submatch", str(v)]):
        args = parse_args()
    assert args.embedded is False


def test_parse_args_embedded_true(tmp_path):
    v = tmp_path / "v.mkv"
    v.touch()
    with patch("sys.argv", ["submatch", str(v), "--embedded"]):
        args = parse_args()
    assert args.embedded is True


def test_parse_args_watch_defaults(tmp_path):
    v = tmp_path / "v.mkv"
    v.touch()
    with patch("sys.argv", ["submatch", str(v)]), \
         patch("submatch.config.load_config", return_value={}):
        args = parse_args()
    assert args.watch is False
    assert args.poll is False
    assert args.interval == 10


def test_parse_args_watch_flags(tmp_path):
    d = tmp_path
    with patch("sys.argv", ["submatch", str(d), "--watch", "--poll", "--interval", "30"]), \
         patch("submatch.config.load_config", return_value={}):
        args = parse_args()
    assert args.watch is True
    assert args.poll is True
    assert args.interval == 30


def test_args_to_config_importable_from_args():
    from submatch.args import _args_to_config
    assert callable(_args_to_config)

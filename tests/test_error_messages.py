"""
Tests that verify the content of user-facing error messages, not just exit codes.
These ensure errors remain actionable: they name the problem, name the file/value,
and point toward a fix.
"""
from __future__ import annotations
import sys
from unittest.mock import MagicMock, patch

import pytest

from submatch import cli, config


# ── dependency error messages ────────────────────────────────────────────────

def test_missing_ffmpeg_message_includes_url(capsys):
    def fake_which(name):
        return None if name == "ffmpeg" else "/usr/bin/ffs"
    with patch("submatch.cli.shutil.which", side_effect=fake_which), \
         patch("submatch.cli.gpu.check_gpu_mismatch", return_value=None), \
         patch.dict(sys.modules, {"whisper": MagicMock()}), \
         pytest.raises(SystemExit):
        cli.check_dependencies()
    err = capsys.readouterr().err
    assert "ffmpeg" in err
    assert "ffmpeg.org" in err


def test_missing_whisper_message_includes_install_command(capsys):
    with patch("submatch.cli.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("submatch.cli.gpu.check_gpu_mismatch", return_value=None), \
         patch.dict(sys.modules, {"whisper": None}), \
         pytest.raises(SystemExit):
        cli.check_dependencies()
    err = capsys.readouterr().err
    assert "openai-whisper" in err
    assert "pip install" in err


# ── config error messages ─────────────────────────────────────────────────────

def test_invalid_model_message_includes_valid_choices(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    cfg.write_text('model = "gigantic"\n')
    with patch.object(config, "_USER_CONFIG", cfg), \
         patch.object(config, "_PROJECT_CONFIG", tmp_path / "no.toml"), \
         pytest.raises(SystemExit):
        config.load_config()
    err = capsys.readouterr().err
    assert "gigantic" in err
    assert "base" in err or "tiny" in err  # lists valid options


def test_invalid_device_message_includes_valid_choices(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    cfg.write_text('device = "tpu"\n')
    with patch.object(config, "_USER_CONFIG", cfg), \
         patch.object(config, "_PROJECT_CONFIG", tmp_path / "no.toml"), \
         pytest.raises(SystemExit):
        config.load_config()
    err = capsys.readouterr().err
    assert "tpu" in err
    assert "cpu" in err  # lists valid options


def test_invalid_toml_message_includes_file_path(tmp_path, capsys):
    cfg = tmp_path / "myconfig.toml"
    cfg.write_text("not valid ][[\n")
    with patch.object(config, "_USER_CONFIG", cfg), \
         patch.object(config, "_PROJECT_CONFIG", tmp_path / "no.toml"), \
         pytest.raises(SystemExit):
        config.load_config()
    err = capsys.readouterr().err
    assert str(cfg) in err
    assert "invalid TOML" in err


def test_unknown_config_key_message_includes_key_name(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    cfg.write_text('typo_threshold = 0.5\n')
    with patch.object(config, "_USER_CONFIG", cfg), \
         patch.object(config, "_PROJECT_CONFIG", tmp_path / "no.toml"):
        config.load_config()
    err = capsys.readouterr().err
    assert "typo_threshold" in err


# ── CLI pipeline error messages ───────────────────────────────────────────────

def test_no_audio_track_message_includes_video_filename(tmp_path, capsys):
    video = tmp_path / "silent_movie.mp4"
    video.touch()
    sub = tmp_path / "movie.srt"
    sub.write_text("1\n00:00:01,000 --> 00:00:03,000\nHello.\n\n")

    with patch("sys.argv", ["submatch", str(video), str(sub)]), \
         patch("submatch.cli.check_dependencies"), \
         patch("submatch.cli.audio.has_audio_track", return_value=False), \
         patch("submatch.config.load_config", return_value={}), \
         pytest.raises(SystemExit):
        cli.main()
    err = capsys.readouterr().err
    assert "silent_movie.mp4" in err


def test_missing_input_message_includes_each_filename(tmp_path, capsys):
    ghost_video = tmp_path / "ghost.mkv"
    ghost_sub = tmp_path / "ghost.srt"

    with patch("sys.argv", ["submatch", str(ghost_video), str(ghost_sub)]), \
         pytest.raises(SystemExit):
        cli.main()
    err = capsys.readouterr().err
    assert "ghost.mkv" in err
    assert "ghost.srt" in err


def test_embedded_with_resync_message_names_both_flags(tmp_path, capsys):
    video = tmp_path / "movie.mkv"
    video.touch()

    with patch("sys.argv", ["submatch", "--embedded", "--resync", str(video)]), \
         pytest.raises(SystemExit):
        cli.main()
    err = capsys.readouterr().err
    assert "--embedded" in err
    assert "--resync" in err or "resync" in err.lower()

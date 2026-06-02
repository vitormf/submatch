import pytest
from unittest.mock import patch

from submatch import config


def test_load_config_no_files(tmp_path):
    with patch.object(config, "_USER_CONFIG", tmp_path / "no_user.toml"), \
         patch.object(config, "_PROJECT_CONFIG", tmp_path / "no_proj.toml"):
        result = config.load_config()
    assert result == {}


def test_load_config_user_only(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('model = "small"\nthreshold = 0.4\n')
    with patch.object(config, "_USER_CONFIG", cfg), \
         patch.object(config, "_PROJECT_CONFIG", tmp_path / "no.toml"):
        result = config.load_config()
    assert result == {"model": "small", "threshold": pytest.approx(0.4)}


def test_load_config_project_only(tmp_path):
    cfg = tmp_path / "submatch.toml"
    cfg.write_text('model = "medium"\n')
    with patch.object(config, "_USER_CONFIG", tmp_path / "no.toml"), \
         patch.object(config, "_PROJECT_CONFIG", cfg):
        result = config.load_config()
    assert result == {"model": "medium"}


def test_load_config_project_overrides_user(tmp_path):
    user_cfg = tmp_path / "config.toml"
    user_cfg.write_text('model = "base"\nthreshold = 0.3\n')
    proj_cfg = tmp_path / "submatch.toml"
    proj_cfg.write_text('model = "large"\n')
    with patch.object(config, "_USER_CONFIG", user_cfg), \
         patch.object(config, "_PROJECT_CONFIG", proj_cfg):
        result = config.load_config()
    assert result["model"] == "large"
    assert result["threshold"] == pytest.approx(0.3)


def test_load_config_unknown_key_warns_and_excludes(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    cfg.write_text('unknown_flag = true\nmodel = "small"\n')
    with patch.object(config, "_USER_CONFIG", cfg), \
         patch.object(config, "_PROJECT_CONFIG", tmp_path / "no.toml"):
        result = config.load_config()
    assert "unknown_flag" not in result
    assert result == {"model": "small"}
    assert "unknown config key 'unknown_flag'" in capsys.readouterr().err


def test_load_config_invalid_toml_exits(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    cfg.write_text("not valid toml ][[\n")
    with patch.object(config, "_USER_CONFIG", cfg), \
         patch.object(config, "_PROJECT_CONFIG", tmp_path / "no.toml"), \
         pytest.raises(SystemExit) as exc:
        config.load_config()
    assert exc.value.code == 2
    assert "invalid TOML" in capsys.readouterr().err


def test_load_config_sub_lang_list(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('sub_lang = ["pt", "en"]\n')
    with patch.object(config, "_USER_CONFIG", cfg), \
         patch.object(config, "_PROJECT_CONFIG", tmp_path / "no.toml"):
        result = config.load_config()
    assert result["sub_lang"] == ["pt", "en"]


def test_load_config_boolean_flags(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('no_sync = true\npass_unsure = false\n')
    with patch.object(config, "_USER_CONFIG", cfg), \
         patch.object(config, "_PROJECT_CONFIG", tmp_path / "no.toml"):
        result = config.load_config()
    assert result["no_sync"] is True
    assert result["pass_unsure"] is False


def test_load_config_invalid_model_exits(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('model = "gigantic"\n')
    with patch.object(config, "_USER_CONFIG", cfg), \
         patch.object(config, "_PROJECT_CONFIG", tmp_path / "no.toml"), \
         pytest.raises(SystemExit) as exc:
        config.load_config()
    assert exc.value.code == 2


def test_load_config_invalid_device_exits(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('device = "tpu"\n')
    with patch.object(config, "_USER_CONFIG", cfg), \
         patch.object(config, "_PROJECT_CONFIG", tmp_path / "no.toml"), \
         pytest.raises(SystemExit) as exc:
        config.load_config()
    assert exc.value.code == 2


def test_cache_ttl_days_accepted(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("cache_ttl_days = 7\n")
    with patch.object(config, "_USER_CONFIG", cfg_file), \
         patch.object(config, "_PROJECT_CONFIG", tmp_path / "none.toml"):
        result = config.load_config()
    assert result["cache_ttl_days"] == 7


def test_cache_max_mb_accepted(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("cache_max_mb = 100\n")
    with patch.object(config, "_USER_CONFIG", cfg_file), \
         patch.object(config, "_PROJECT_CONFIG", tmp_path / "none.toml"):
        result = config.load_config()
    assert result["cache_max_mb"] == 100


def test_cache_dir_accepted(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('cache_dir = "/tmp/my_cache"\n')
    with patch.object(config, "_USER_CONFIG", cfg_file), \
         patch.object(config, "_PROJECT_CONFIG", tmp_path / "none.toml"):
        result = config.load_config()
    assert result["cache_dir"] == "/tmp/my_cache"


def test_load_config_telemetry_false(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("telemetry = false\n")
    with patch.object(config, "_USER_CONFIG", cfg), \
         patch.object(config, "_PROJECT_CONFIG", tmp_path / "no.toml"):
        result = config.load_config()
    assert result == {"telemetry": False}


def test_load_config_unknown_key_warns_not_telemetry(tmp_path, capsys):
    cfg = tmp_path / "config.toml"
    cfg.write_text('telemetry = false\nmodel = "base"\n')
    with patch.object(config, "_USER_CONFIG", cfg), \
         patch.object(config, "_PROJECT_CONFIG", tmp_path / "no.toml"):
        result = config.load_config()
    captured = capsys.readouterr()
    assert "telemetry" not in captured.err
    assert result["telemetry"] is False


def test_load_config_oserror_exits(tmp_path, capsys):
    # Covers lines 38-40: OSError when opening config file
    cfg = tmp_path / "config.toml"
    cfg.write_text('model = "base"\n')
    cfg.chmod(0o000)  # unreadable — open() raises PermissionError (OSError subclass)
    try:
        with patch.object(config, "_USER_CONFIG", cfg), \
             patch.object(config, "_PROJECT_CONFIG", tmp_path / "no.toml"), \
             pytest.raises(SystemExit) as exc:
            config.load_config()
        assert exc.value.code == 2
        assert "cannot read config file" in capsys.readouterr().err
    finally:
        cfg.chmod(0o644)

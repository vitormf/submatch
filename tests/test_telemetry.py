from __future__ import annotations
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from submatch import telemetry


@pytest.fixture(autouse=True)
def reset_telemetry():
    telemetry._enabled = False
    yield
    telemetry._enabled = False


def _args(**kwargs):
    defaults = dict(telemetry=True, model="base", no_sync=False,
                    device="auto", workers=None, threshold=0.35)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ── opt-out ───────────────────────────────────────────────────────────────────

def test_env_var_disables_init(monkeypatch):
    monkeypatch.setenv("SUBMATCH_NO_TELEMETRY", "1")
    mock_sdk = MagicMock()
    with patch.dict(sys.modules, {"sentry_sdk": mock_sdk}):
        telemetry.init(_args())
    mock_sdk.init.assert_not_called()
    assert not telemetry._enabled


def test_config_false_disables_init():
    mock_sdk = MagicMock()
    with patch.dict(sys.modules, {"sentry_sdk": mock_sdk}):
        telemetry.init(_args(telemetry=False))
    mock_sdk.init.assert_not_called()
    assert not telemetry._enabled


def test_config_true_enables_init():
    mock_sdk = MagicMock()
    with patch.dict(sys.modules, {"sentry_sdk": mock_sdk}):
        telemetry.init(_args(telemetry=True))
    mock_sdk.init.assert_called_once()
    assert telemetry._enabled


def test_no_telemetry_attr_defaults_to_enabled():
    args = SimpleNamespace(model="base", no_sync=False, device="auto",
                           workers=None, threshold=0.35)
    mock_sdk = MagicMock()
    with patch.dict(sys.modules, {"sentry_sdk": mock_sdk}):
        telemetry.init(args)
    mock_sdk.init.assert_called_once()
    assert telemetry._enabled


# ── PII scrubbing ─────────────────────────────────────────────────────────────

def test_scrub_path_in_exception_value():
    event = {
        "exception": {
            "values": [{"value": "/home/user/movies/film.mkv failed"}]
        }
    }
    result = telemetry._scrub_pii(event, {})
    assert result["exception"]["values"][0]["value"] == "<path>"


def test_scrub_non_path_string_unchanged():
    event = {
        "exception": {
            "values": [{"value": "index out of range"}]
        }
    }
    result = telemetry._scrub_pii(event, {})
    assert result["exception"]["values"][0]["value"] == "index out of range"


def test_scrub_path_in_frame_vars():
    event = {
        "exception": {
            "values": [{
                "stacktrace": {
                    "frames": [{"vars": {"path": "/tmp/foo.wav", "count": 3}}]
                }
            }]
        }
    }
    result = telemetry._scrub_pii(event, {})
    frame_vars = result["exception"]["values"][0]["stacktrace"]["frames"][0]["vars"]
    assert frame_vars["path"] == "<path>"
    assert frame_vars["count"] == 3


def test_scrub_path_in_extra():
    event = {"extra": {"file": "/home/user/sub.srt", "score": 0.9}}
    result = telemetry._scrub_pii(event, {})
    assert result["extra"]["file"] == "<path>"
    assert result["extra"]["score"] == pytest.approx(0.9)


def test_scrub_nested_dict_in_extra():
    event = {"extra": {"info": {"path": "/tmp/x.wav", "ok": True}}}
    result = telemetry._scrub_pii(event, {})
    assert result["extra"]["info"]["path"] == "<path>"
    assert result["extra"]["info"]["ok"] is True


def test_scrub_windows_path():
    event = {"exception": {"values": [{"value": r"C:\Users\bob\file.mkv"}]}}
    result = telemetry._scrub_pii(event, {})
    assert result["exception"]["values"][0]["value"] == "<path>"


def test_scrub_frame_abs_path():
    event = {
        "exception": {
            "values": [{
                "stacktrace": {
                    "frames": [{
                        "abs_path": "/home/bob/.local/lib/python3.12/submatch/cli.py",
                        "filename": "submatch/cli.py",
                        "vars": {},
                    }]
                }
            }]
        }
    }
    result = telemetry._scrub_pii(event, {})
    frame = result["exception"]["values"][0]["stacktrace"]["frames"][0]
    assert frame["abs_path"] == "<path>"
    assert frame["filename"] == "submatch/cli.py"  # relative path — no separator to scrub


# ── capture ───────────────────────────────────────────────────────────────────

def test_capture_noop_when_disabled():
    mock_sdk = MagicMock()
    with patch.dict(sys.modules, {"sentry_sdk": mock_sdk}):
        telemetry.capture(ValueError("test"))
    mock_sdk.capture_exception.assert_not_called()


def test_capture_sends_when_enabled():
    telemetry._enabled = True
    exc = ValueError("test")
    mock_sdk = MagicMock()
    with patch.dict(sys.modules, {"sentry_sdk": mock_sdk}):
        telemetry.capture(exc)
    mock_sdk.capture_exception.assert_called_once_with(exc)


# ── tags ──────────────────────────────────────────────────────────────────────

def test_tags_set_from_args():
    mock_sdk = MagicMock()
    args = _args(model="small", no_sync=True, device="cuda", workers=2, threshold=0.5)
    with patch.dict(sys.modules, {"sentry_sdk": mock_sdk}):
        telemetry.init(args)

    tag_calls = {c[0][0]: c[0][1] for c in mock_sdk.set_tag.call_args_list}
    assert tag_calls["submatch.model"] == "small"
    assert tag_calls["submatch.no_sync"] == "true"

    extra_calls = {c[0][0]: c[0][1] for c in mock_sdk.set_extra.call_args_list}
    assert extra_calls["device"] == "cuda"
    assert extra_calls["workers"] == 2
    assert extra_calls["threshold"] == pytest.approx(0.5)


def test_set_mode_noop_when_disabled():
    mock_sdk = MagicMock()
    with patch.dict(sys.modules, {"sentry_sdk": mock_sdk}):
        telemetry.set_mode("single")
    mock_sdk.set_tag.assert_not_called()


def test_set_mode_sets_tag_when_enabled():
    telemetry._enabled = True
    mock_sdk = MagicMock()
    with patch.dict(sys.modules, {"sentry_sdk": mock_sdk}):
        telemetry.set_mode("batch")
    mock_sdk.set_tag.assert_called_once_with("submatch.mode", "batch")

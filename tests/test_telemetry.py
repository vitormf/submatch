from __future__ import annotations
import os
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


@pytest.fixture()
def allow_telemetry(monkeypatch):
    """Remove the test-suite opt-out and simulate a release install."""
    monkeypatch.delenv("SUBMATCH_NO_TELEMETRY", raising=False)
    monkeypatch.setattr(telemetry, "_get_direct_url", lambda: None)


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


def test_config_true_enables_init(allow_telemetry):
    mock_sdk = MagicMock()
    with patch.dict(sys.modules, {"sentry_sdk": mock_sdk}):
        telemetry.init(_args(telemetry=True))
    mock_sdk.init.assert_called_once()
    assert telemetry._enabled


def test_no_telemetry_attr_defaults_to_enabled(allow_telemetry):
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

def test_tags_set_from_args(allow_telemetry):
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


# ── dev install detection ─────────────────────────────────────────────────────

def test_dev_install_detected_when_editable(allow_telemetry):
    fake_direct_url = '{"url": "file:///home/user/submatch", "dir_info": {"editable": true}}'
    mock_dist = MagicMock()
    mock_dist.read_text.return_value = fake_direct_url
    with patch("submatch.telemetry._get_direct_url", return_value=fake_direct_url):
        mock_sdk = MagicMock()
        with patch.dict(sys.modules, {"sentry_sdk": mock_sdk}):
            telemetry.init(_args())
    mock_sdk.init.assert_not_called()
    assert not telemetry._enabled


def test_dev_install_not_detected_when_release(allow_telemetry):
    with patch("submatch.telemetry._get_direct_url", return_value=None):
        mock_sdk = MagicMock()
        with patch.dict(sys.modules, {"sentry_sdk": mock_sdk}):
            telemetry.init(_args())
    mock_sdk.init.assert_called_once()
    assert telemetry._enabled


def test_dev_install_not_detected_when_non_editable_direct_url(allow_telemetry):
    # pip install . (non-editable from source dir) should still enable telemetry
    non_editable = '{"url": "file:///home/user/submatch", "dir_info": {"editable": false}}'
    with patch("submatch.telemetry._get_direct_url", return_value=non_editable):
        mock_sdk = MagicMock()
        with patch.dict(sys.modules, {"sentry_sdk": mock_sdk}):
            telemetry.init(_args())
    mock_sdk.init.assert_called_once()
    assert telemetry._enabled


# ── test-suite isolation ───────────────────────────────────────────────────────

def test_submatch_no_telemetry_set_in_test_suite():
    """SUBMATCH_NO_TELEMETRY must be set so tests never reach the real Sentry SDK."""
    assert os.environ.get("SUBMATCH_NO_TELEMETRY"), (
        "SUBMATCH_NO_TELEMETRY is not set. "
        "Tests that call cli.main() without patching telemetry.init() will send "
        "real events to the production Sentry project. "
        "Add `os.environ['SUBMATCH_NO_TELEMETRY'] = '1'` to tests/conftest.py."
    )


# ── direct URL detection ──────────────────────────────────────────────────────

def test_get_direct_url_returns_without_raising():
    # Covers lines 9-13: _get_direct_url is always mocked in other tests; call it directly
    result = telemetry._get_direct_url()
    assert result is None or isinstance(result, str)


def test_get_direct_url_handles_exception():
    # Covers lines 12-13: exception handler in _get_direct_url
    mock_distribution = MagicMock(side_effect=RuntimeError("not found"))
    with patch("importlib.metadata.distribution", mock_distribution):
        result = telemetry._get_direct_url()
    assert result is None


# ── scrub list and edge cases ─────────────────────────────────────────────────

def test_scrub_list_value():
    # Covers line 28: list branch of _scrub
    result = telemetry._scrub(["/home/user/file.mkv", "text", 42])
    assert result == ["<path>", "text", 42]


def test_scrub_nested_list():
    # Covers line 28 again with nesting
    result = telemetry._scrub(["/path/a", ["/path/b", "safe"]])
    assert result[0] == "<path>"
    assert result[1] == ["<path>", "safe"]


# ── init with telemetry arg ───────────────────────────────────────────────────

def test_telemetry_false_arg_disables_init(monkeypatch):
    # Covers line 60: telemetry=False short-circuits init
    # Must remove SUBMATCH_NO_TELEMETRY so that check doesn't fire first (line 57)
    monkeypatch.delenv("SUBMATCH_NO_TELEMETRY", raising=False)
    monkeypatch.setattr(telemetry, "_get_direct_url", lambda: None)
    mock_sdk = MagicMock()
    with patch.dict(sys.modules, {"sentry_sdk": mock_sdk}):
        telemetry.init(_args(telemetry=False))
    mock_sdk.init.assert_not_called()
    assert not telemetry._enabled


# ── import error handling ─────────────────────────────────────────────────────

def test_init_when_sentry_sdk_not_installed(monkeypatch):
    # Covers lines 82-83: ImportError path when sentry_sdk is missing
    monkeypatch.delenv("SUBMATCH_NO_TELEMETRY", raising=False)
    monkeypatch.setattr(telemetry, "_get_direct_url", lambda: None)
    with patch.dict(sys.modules, {"sentry_sdk": None}):
        telemetry.init(_args())
    assert not telemetry._enabled


# ── exception handling ────────────────────────────────────────────────────────

def test_set_mode_swallows_exception():
    # Covers lines 92-93: exception handler in set_mode
    telemetry._enabled = True
    mock_sdk = MagicMock()
    mock_sdk.set_tag.side_effect = RuntimeError("sentry down")
    with patch.dict(sys.modules, {"sentry_sdk": mock_sdk}):
        telemetry.set_mode("batch")  # must not raise


def test_capture_swallows_exception():
    # Covers lines 102-103: exception handler in capture
    telemetry._enabled = True
    mock_sdk = MagicMock()
    mock_sdk.capture_exception.side_effect = RuntimeError("sentry down")
    with patch.dict(sys.modules, {"sentry_sdk": mock_sdk}):
        telemetry.capture(ValueError("test"))  # must not raise

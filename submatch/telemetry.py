from __future__ import annotations
import os
import platform
import sys
from typing import Any

_SENTRY_DSN = (
    "https://105e24adec63749ae27bed49f404f225"
    "@o4511495643660288.ingest.de.sentry.io/4511495658012752"
)
_enabled = False


def _scrub(value: Any) -> Any:
    if isinstance(value, str) and ("/" in value or "\\" in value):
        return "<path>"
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    return value


def _scrub_pii(event: dict, hint: dict) -> dict:
    if "exception" in event:
        for exc_val in event["exception"].get("values", []):
            if "value" in exc_val:
                exc_val["value"] = _scrub(exc_val["value"])
            for frame in exc_val.get("stacktrace", {}).get("frames", []):
                if "vars" in frame:
                    frame["vars"] = _scrub(frame["vars"])
    if "extra" in event:
        event["extra"] = _scrub(event["extra"])
    return event


def init(args: Any) -> None:
    global _enabled
    if os.environ.get("SUBMATCH_NO_TELEMETRY"):
        return
    if getattr(args, "telemetry", True) is False:
        return
    try:
        import sentry_sdk
        from submatch import __version__

        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            send_default_pii=False,
            before_send=_scrub_pii,
        )
        sentry_sdk.set_tag("submatch.version", __version__)
        sentry_sdk.set_tag("submatch.model", getattr(args, "model", "base"))
        sentry_sdk.set_tag("submatch.no_sync", str(getattr(args, "no_sync", False)).lower())
        sentry_sdk.set_extra("python_version", sys.version)
        sentry_sdk.set_extra("platform", platform.platform())
        sentry_sdk.set_extra("device", getattr(args, "device", "auto"))
        sentry_sdk.set_extra("workers", getattr(args, "workers", None))
        sentry_sdk.set_extra("threshold", getattr(args, "threshold", 0.35))
        _enabled = True
    except ImportError:
        pass


def set_mode(mode: str) -> None:
    if not _enabled:
        return
    try:
        import sentry_sdk
        sentry_sdk.set_tag("submatch.mode", mode)
    except Exception:
        pass


def capture(exc: BaseException) -> None:
    if not _enabled:
        return
    try:
        import sentry_sdk
        sentry_sdk.capture_exception(exc)
    except Exception:
        pass

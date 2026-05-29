from __future__ import annotations
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

_USER_CONFIG = Path.home() / ".config" / "submatch" / "config.toml"
_PROJECT_CONFIG = Path("submatch.toml")

_CONFIGURABLE_KEYS = frozenset({
    "model", "threshold", "segments", "language",
    "no_sync", "keep_synced", "no_recursive", "sub_lang",
    "filter", "device", "workers", "delete_failures",
    "cross_threshold", "resync", "pass_unsure",
    "drift_threshold", "audio_track",
})


def load_config() -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in [_USER_CONFIG, _PROJECT_CONFIG]:
        if not path.is_file():
            continue
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except Exception as exc:
            print(f"Error: invalid config file {path}: {exc}", file=sys.stderr)
            sys.exit(2)
        for key in data:
            if key not in _CONFIGURABLE_KEYS:
                print(f"Warning: unknown config key '{key}' in {path}", file=sys.stderr)
        merged.update({k: v for k, v in data.items() if k in _CONFIGURABLE_KEYS})
    return merged

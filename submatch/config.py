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

_CHOICES: dict[str, frozenset[str]] = {
    "model": frozenset({"tiny", "base", "small", "medium", "large"}),
    "device": frozenset({"cpu", "mps", "cuda", "auto"}),
}


def load_config() -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in [_USER_CONFIG, _PROJECT_CONFIG]:
        if not path.is_file():
            continue
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except OSError as exc:
            print(f"Error: cannot read config file {path}: {exc}", file=sys.stderr)
            sys.exit(2)
        except tomllib.TOMLDecodeError as exc:
            print(f"Error: invalid TOML in config file {path}: {exc}", file=sys.stderr)
            sys.exit(2)
        for key in data:
            if key not in _CONFIGURABLE_KEYS:
                print(f"Warning: unknown config key '{key}' in {path}", file=sys.stderr)
        merged.update({k: v for k, v in data.items() if k in _CONFIGURABLE_KEYS})

    for key, valid in _CHOICES.items():
        if key in merged and merged[key] not in valid:
            print(
                f"Error: invalid config value {merged[key]!r} for '{key}' "
                f"(valid: {', '.join(sorted(valid))})",
                file=sys.stderr,
            )
            sys.exit(2)
    return merged

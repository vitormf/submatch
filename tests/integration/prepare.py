#!/usr/bin/env python3
"""Pre-download fixtures and models required for integration tests.

Run before the integration test suite to ensure no network I/O or model
downloads happen during the timed test run itself.

Usage:
    python tests/integration/prepare.py
"""
import os
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.integration.conftest import ASSETS, FIXTURES_DIR  # noqa: E402

_USER_AGENT = "submatch-integration-tests/1.0 (https://github.com/vitormf/submatch)"


def _download(dest: Path, url: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def prepare_fixtures() -> list[tuple[str, Exception]]:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    errors: list[tuple[str, Exception]] = []

    for name, url in ASSETS.items():
        dest = FIXTURES_DIR / name
        if dest.exists():
            print(f"  [fixtures] {name} already cached", flush=True)
            continue
        print(f"  [fixtures] Downloading {name} ...", flush=True)
        try:
            _download(dest, url)
            print(f"  [fixtures] {name} ready ({dest.stat().st_size // 1024} KB)", flush=True)
        except Exception as exc:
            errors.append((name, exc))
            if dest.exists():
                dest.unlink()
            print(f"  [fixtures] FAILED: {name}: {exc}", flush=True)

    stale = [f for f in FIXTURES_DIR.iterdir() if f.name not in ASSETS]
    for path in stale:
        print(f"  [fixtures] Removing stale: {path.name}", flush=True)
        path.unlink()

    return errors


def prepare_models() -> None:
    from submatch.transcribe import load_model

    _whisper_cache = Path(os.path.expanduser("~")) / ".cache" / "whisper"
    for model_name in ("tiny", "base"):
        if (_whisper_cache / f"{model_name}.pt").exists():
            print(f"  [models] Whisper {model_name} already cached", flush=True)
            continue
        print(f"  [models] Whisper {model_name} downloading ...", flush=True)
        load_model(model_name)
        print(f"  [models] Whisper {model_name} ready", flush=True)

    _embed_cache = (
        Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"
        / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
    )
    try:
        from submatch.embeddings import load_embedding_model
        if _embed_cache.exists():
            print("  [models] Embedding model already cached", flush=True)
        else:
            print("  [models] Embedding model downloading ...", flush=True)
            load_embedding_model()
            print("  [models] Embedding model ready", flush=True)
    except Exception as exc:
        print(f"  [models] Embedding model unavailable: {exc}", flush=True)


if __name__ == "__main__":
    print("Preparing fixtures...")
    errors = prepare_fixtures()

    print("Preparing models...")
    prepare_models()

    if errors:
        print(f"\n{len(errors)} fixture(s) failed to download:")
        for name, exc in errors:
            print(f"  - {name}: {exc}")
        sys.exit(1)

    print("\nAll fixtures and models ready.")

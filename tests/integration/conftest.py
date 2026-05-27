import shutil
import urllib.request
from pathlib import Path
import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
_USER_AGENT = "submatch-integration-tests/1.0 (https://github.com/vitormf/submatch)"

# ── Asset registry ────────────────────────────────────────────────────────────
# Single source of truth for integration test fixtures.
# Add an entry  → file is downloaded before tests run.
# Remove an entry → cached file is deleted before tests run.
ASSETS: dict[str, str] = {
    "made_in_america.webm": (
        "https://upload.wikimedia.org/wikipedia/commons/c/ce/"
        "%22Made_in_America_is_back%21%22.webm"
    ),
    "made_in_america.srt": (
        "https://commons.wikimedia.org/w/index.php"
        "?title=TimedText:%22Made_in_America_is_back!%22.webm.en.srt&action=raw"
    ),
    # NASA public domain — used as mismatched subtitle (different content, same language)
    "nasa_venus.srt": (
        "https://svs.gsfc.nasa.gov/vis/a010000/a013600/a013640/"
        "VIAM_caption.en_US.srt"
    ),
}


# ── Cache sync ────────────────────────────────────────────────────────────────

def _download(dest: Path, url: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def pytest_sessionstart(session) -> None:
    """Sync the fixtures cache before any test runs.

    Downloads files that are in the registry but missing from the cache.
    Deletes cached files that have been removed from the registry.
    """
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in ASSETS.items():
        dest = FIXTURES_DIR / name
        if not dest.exists():
            print(f"\n  [fixtures] Downloading {name} ...", flush=True)
            _download(dest, url)
            print(f"  [fixtures] {name} ready ({dest.stat().st_size // 1024} KB)", flush=True)

    stale = [f for f in FIXTURES_DIR.iterdir() if f.name not in ASSETS]
    for path in stale:
        print(f"  [fixtures] Removing stale: {path.name}", flush=True)
        path.unlink()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def made_in_america_video() -> Path:
    return FIXTURES_DIR / "made_in_america.webm"


@pytest.fixture(scope="session")
def made_in_america_srt() -> Path:
    return FIXTURES_DIR / "made_in_america.srt"


@pytest.fixture(scope="session")
def nasa_venus_srt() -> Path:
    return FIXTURES_DIR / "nasa_venus.srt"


@pytest.fixture(scope="session")
def whisper_tiny():
    from submatch.transcribe import load_model
    return load_model("tiny")

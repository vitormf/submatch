import shutil
import urllib.request
from pathlib import Path
import pytest

_USER_AGENT = "submatch-integration-tests/1.0 (https://github.com/vitormf/submatch)"

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# Public domain assets — stable permanent URLs
_ASSETS = {
    "made_in_america.webm": (
        "https://upload.wikimedia.org/wikipedia/commons/c/ce/"
        "%22Made_in_America_is_back%21%22.webm"
    ),
    "made_in_america.srt": (
        "https://commons.wikimedia.org/w/index.php"
        "?title=TimedText:%22Made_in_America_is_back!%22.webm.en.srt&action=raw"
    ),
    # NASA public domain — used as a mismatched subtitle (different content, same language)
    "nasa_venus.srt": (
        "https://svs.gsfc.nasa.gov/vis/a010000/a013600/a013640/"
        "VIAM_caption.en_US.srt"
    ),
}


def _fetch(name: str) -> Path:
    dest = FIXTURES_DIR / name
    if dest.exists():
        return dest
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n    Downloading {name} ...", flush=True)
    req = urllib.request.Request(_ASSETS[name], headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)
    return dest


@pytest.fixture(scope="session")
def made_in_america_video() -> Path:
    return _fetch("made_in_america.webm")


@pytest.fixture(scope="session")
def made_in_america_srt() -> Path:
    return _fetch("made_in_america.srt")


@pytest.fixture(scope="session")
def nasa_venus_srt() -> Path:
    return _fetch("nasa_venus.srt")


@pytest.fixture(scope="session")
def whisper_tiny():
    from submatch.transcribe import load_model
    return load_model("tiny")

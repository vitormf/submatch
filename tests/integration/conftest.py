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
#
# Sources: WIKITONGUES project on Wikimedia Commons (CC BY-SA 4.0 / CC BY 3.0).
ASSETS: dict[str, str] = {
    # WIKITONGUES — Gereon speaking German (CC BY-SA 4.0)
    # Native German speaker discussing the German language, ~3 min, 21.8 MB.
    # Subtitles available in: de (native), en, pt-br, eo.
    "wikitongues_german.webm": (
        "https://upload.wikimedia.org/wikipedia/commons/2/20/"
        "WIKITONGUES-_Gereon_speaking_German.webm"
    ),
    "wikitongues_german.de.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Gereon_speaking_German.webm"
        "&lang=de&trackformat=srt&origin=*"
    ),
    "wikitongues_german.en.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Gereon_speaking_German.webm"
        "&lang=en&trackformat=srt&origin=*"
    ),
    "wikitongues_german.pt-br.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Gereon_speaking_German.webm"
        "&lang=pt-br&trackformat=srt&origin=*"
    ),
    "wikitongues_german.de.vtt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Gereon_speaking_German.webm"
        "&lang=de&trackformat=vtt&origin=*"
    ),
    # WIKITONGUES — María speaking Guarani (CC BY 3.0)
    # Speaker from Paraguay, ~5 min, 31.6 MB.
    # Subtitles available in: gn (native), en, es, de, fr, fi, pt-br, uk.
    "wikitongues_guarani.webm": (
        "https://upload.wikimedia.org/wikipedia/commons/e/e1/"
        "WIKITONGUES-_Mar%C3%ADa_speaking_Guarani.webm"
    ),
    "wikitongues_guarani.gn.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Mar%C3%ADa_speaking_Guarani.webm"
        "&lang=gn&trackformat=srt&origin=*"
    ),
    "wikitongues_guarani.en.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Mar%C3%ADa_speaking_Guarani.webm"
        "&lang=en&trackformat=srt&origin=*"
    ),
    "wikitongues_guarani.es.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Mar%C3%ADa_speaking_Guarani.webm"
        "&lang=es&trackformat=srt&origin=*"
    ),
    # German subtitle of the Guarani video — used as cross-video mismatch control.
    "wikitongues_guarani.de.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Mar%C3%ADa_speaking_Guarani.webm"
        "&lang=de&trackformat=srt&origin=*"
    ),
    # WIKITONGUES — Omar speaking English and Jamaican Patois (CC BY 3.0)
    # English speaker discussing Jamaican Patois as heritage language, ~4 min, 85.5 MB.
    # Subtitles available in: en (native), de, es, fr, pt, it, tr, eo, fy.
    "wikitongues_english.webm": (
        "https://upload.wikimedia.org/wikipedia/commons/f/f8/"
        "WIKITONGUES-_Omar_Speaking_English_and_Jamaican_Patois.webm"
    ),
    "wikitongues_english.en.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Omar_Speaking_English_and_Jamaican_Patois.webm"
        "&lang=en&trackformat=srt&origin=*"
    ),
    "wikitongues_english.es.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Omar_Speaking_English_and_Jamaican_Patois.webm"
        "&lang=es&trackformat=srt&origin=*"
    ),
    "wikitongues_english.fr.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Omar_Speaking_English_and_Jamaican_Patois.webm"
        "&lang=fr&trackformat=srt&origin=*"
    ),
    "wikitongues_english.pt.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Omar_Speaking_English_and_Jamaican_Patois.webm"
        "&lang=pt&trackformat=srt&origin=*"
    ),
}

# Populated during pytest_sessionstart; checked by fixtures to skip dependent tests.
_DOWNLOAD_ERRORS: dict[str, str] = {}


# ── Cache sync ────────────────────────────────────────────────────────────────

def _download(dest: Path, url: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def pytest_sessionstart(session) -> None:
    """Sync the fixtures cache before any test runs.

    Downloads files that are in the registry but missing from the cache.
    Deletes cached files that have been removed from the registry.
    Prints a warning for each failed download; dependent tests will be skipped.
    """
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in ASSETS.items():
        dest = FIXTURES_DIR / name
        if dest.exists():
            continue
        print(f"\n  [fixtures] Downloading {name} ...", flush=True)
        try:
            _download(dest, url)
            print(f"  [fixtures] {name} ready ({dest.stat().st_size // 1024} KB)", flush=True)
        except Exception as exc:
            _DOWNLOAD_ERRORS[name] = str(exc)
            if dest.exists():
                dest.unlink()
            print(
                f"\n  ⚠  [fixtures] FAILED to download {name}: {exc}\n"
                f"     Tests requiring this file will be skipped.",
                flush=True,
            )

    stale = [f for f in FIXTURES_DIR.iterdir() if f.name not in ASSETS]
    for path in stale:
        print(f"  [fixtures] Removing stale: {path.name}", flush=True)
        path.unlink()


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _fixture_path(name: str) -> Path:
    if name in _DOWNLOAD_ERRORS:
        pytest.skip(
            f"Fixture '{name}' unavailable — download failed: {_DOWNLOAD_ERRORS[name]}"
        )
    return FIXTURES_DIR / name


# German video (Gereon, native German speaker)
@pytest.fixture(scope="session")
def german_video() -> Path:
    return _fixture_path("wikitongues_german.webm")


@pytest.fixture(scope="session")
def german_de_srt() -> Path:
    return _fixture_path("wikitongues_german.de.srt")


@pytest.fixture(scope="session")
def german_en_srt() -> Path:
    return _fixture_path("wikitongues_german.en.srt")


@pytest.fixture(scope="session")
def german_ptbr_srt() -> Path:
    return _fixture_path("wikitongues_german.pt-br.srt")


@pytest.fixture(scope="session")
def german_de_vtt() -> Path:
    return _fixture_path("wikitongues_german.de.vtt")


@pytest.fixture(scope="session")
def german_de_ass(german_de_srt, tmp_path_factory) -> Path:
    """ASS subtitle generated from the German SRT — exercises the ASS parser path."""
    import pysubs2
    subs = pysubs2.load(str(german_de_srt))
    out = tmp_path_factory.mktemp("ass") / "wikitongues_german.de.ass"
    subs.save(str(out))
    return out


# Guarani video (María, Guarani speaker from Paraguay)
@pytest.fixture(scope="session")
def guarani_video() -> Path:
    return _fixture_path("wikitongues_guarani.webm")


@pytest.fixture(scope="session")
def guarani_gn_srt() -> Path:
    return _fixture_path("wikitongues_guarani.gn.srt")


@pytest.fixture(scope="session")
def guarani_en_srt() -> Path:
    return _fixture_path("wikitongues_guarani.en.srt")


@pytest.fixture(scope="session")
def guarani_es_srt() -> Path:
    return _fixture_path("wikitongues_guarani.es.srt")


@pytest.fixture(scope="session")
def guarani_de_srt() -> Path:
    """German subtitle of the Guarani video — different content from the German video."""
    return _fixture_path("wikitongues_guarani.de.srt")


# Shared model fixtures
@pytest.fixture(scope="session")
def whisper_tiny():
    from submatch.transcribe import load_model
    return load_model("tiny")


# English video (Omar, English speaker discussing Jamaican Patois)
@pytest.fixture(scope="session")
def english_video() -> Path:
    return _fixture_path("wikitongues_english.webm")


@pytest.fixture(scope="session")
def english_en_srt() -> Path:
    return _fixture_path("wikitongues_english.en.srt")


@pytest.fixture(scope="session")
def english_es_srt() -> Path:
    return _fixture_path("wikitongues_english.es.srt")


@pytest.fixture(scope="session")
def english_fr_srt() -> Path:
    return _fixture_path("wikitongues_english.fr.srt")


@pytest.fixture(scope="session")
def english_pt_srt() -> Path:
    return _fixture_path("wikitongues_english.pt.srt")


# Shared model fixtures
@pytest.fixture(scope="session")
def embed_model():
    """Session-scoped multilingual sentence embedding model."""
    pytest.importorskip(
        "sentence_transformers",
        reason="sentence-transformers not installed — skipping cross-language integration tests",
    )
    from submatch.embeddings import load_embedding_model
    return load_embedding_model()

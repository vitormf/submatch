import os
import subprocess
from pathlib import Path
import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# ── Asset registry ────────────────────────────────────────────────────────────
# Single source of truth for integration test fixtures.
# Add an entry  → prepare.py will download it before tests run.
# Remove an entry → prepare.py will delete the stale cached file.
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
    # WIKITONGUES — Clara speaking French (CC BY-SA 4.0)
    # Native French speaker from Francophone Switzerland, ~35 sec, 1.1 MB.
    # Subtitles available in: fr (native), en, es.
    "wikitongues_french.webm": (
        "https://upload.wikimedia.org/wikipedia/commons/7/79/"
        "WIKITONGUES-_Clara_speaking_French.webm"
    ),
    "wikitongues_french.fr.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Clara_speaking_French.webm"
        "&lang=fr&trackformat=srt&origin=*"
    ),
    "wikitongues_french.en.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Clara_speaking_French.webm"
        "&lang=en&trackformat=srt&origin=*"
    ),
    "wikitongues_french.es.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Clara_speaking_French.webm"
        "&lang=es&trackformat=srt&origin=*"
    ),
    # WIKITONGUES — Ivy speaking Shanghainese (CC BY-SA 4.0)
    # Native Shanghainese speaker, ~27 MB.
    # Subtitles available in: zh-hans (native), zh, en.
    "wikitongues_shanghainese.webm": (
        "https://upload.wikimedia.org/wikipedia/commons/8/8d/"
        "WIKITONGUES-_Ivy_speaking_Shanghainese.webm"
    ),
    "wikitongues_shanghainese.zh-hans.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Ivy_speaking_Shanghainese.webm"
        "&lang=zh-hans&trackformat=srt&origin=*"
    ),
    "wikitongues_shanghainese.en.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Ivy_speaking_Shanghainese.webm"
        "&lang=en&trackformat=srt&origin=*"
    ),
    # WIKITONGUES — Krishna speaking Hindi (CC BY-SA 4.0)
    # Native Hindi speaker from Delhi, ~23 MB.
    # Subtitles available in: en, fa (Farsi), fr. No native hi subtitle track.
    "wikitongues_hindi.webm": (
        "https://upload.wikimedia.org/wikipedia/commons/7/7a/"
        "WIKITONGUES-_Krishna_speaking_Hindi.webm"
    ),
    "wikitongues_hindi.en.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Krishna_speaking_Hindi.webm"
        "&lang=en&trackformat=srt&origin=*"
    ),
    "wikitongues_hindi.fr.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Krishna_speaking_Hindi.webm"
        "&lang=fr&trackformat=srt&origin=*"
    ),
    # WIKITONGUES — Azariah speaking Spanish (CC BY-SA 4.0)
    # Native Spanish speaker from Honduras, ~50 sec.
    # Subtitles available in: es.  No English track.
    "wikitongues_spanish.webm": (
        "https://upload.wikimedia.org/wikipedia/commons/0/0a/"
        "WIKITONGUES-_Azariah_speaking_Spanish.webm"
    ),
    "wikitongues_spanish.es.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Azariah_speaking_Spanish.webm"
        "&lang=es&trackformat=srt&origin=*"
    ),
    # WIKITONGUES — Changjiu and Chaofen speaking Guiyangese (CC BY-SA 4.0)
    # Native speakers of Guiyang dialect (Mandarin family), ~3 min.
    # Subtitles available in: zh-hans, en.
    "wikitongues_guiyangese.webm": (
        "https://upload.wikimedia.org/wikipedia/commons/3/3b/"
        "WIKITONGUES-_Changjiu_and_Chaofen_speaking_Guiyangese.webm"
    ),
    "wikitongues_guiyangese.zh-hans.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Changjiu_and_Chaofen_speaking_Guiyangese.webm"
        "&lang=zh-hans&trackformat=srt&origin=*"
    ),
    "wikitongues_guiyangese.en.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Changjiu_and_Chaofen_speaking_Guiyangese.webm"
        "&lang=en&trackformat=srt&origin=*"
    ),
    # WIKITONGUES — Sara speaking Portuguese (CC BY-SA 4.0)
    # Speaker from Lisbon, ~85 seconds, 10.6 MB.
    # Subtitles available in: pt (native), en.
    "wikitongues_portuguese.webm": (
        "https://upload.wikimedia.org/wikipedia/commons/7/75/"
        "WIKITONGUES-_Sara_speaking_Portuguese.webm"
    ),
    "wikitongues_portuguese.pt.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Sara_speaking_Portuguese.webm"
        "&lang=pt&trackformat=srt&origin=*"
    ),
    "wikitongues_portuguese.en.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Sara_speaking_Portuguese.webm"
        "&lang=en&trackformat=srt&origin=*"
    ),
    # WIKITONGUES — Freddie speaking Portuguese (CC BY-SA 4.0)
    # Brazilian Portuguese speaker from São Paulo, ~243 seconds, 29.6 MB.
    # Subtitles available in: pt-br (native), en.
    "wikitongues_portuguese_br.webm": (
        "https://upload.wikimedia.org/wikipedia/commons/b/b2/"
        "WIKITONGUES-_Freddie_speaking_Portuguese.webm"
    ),
    "wikitongues_portuguese_br.pt-br.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Freddie_speaking_Portuguese.webm"
        "&lang=pt-br&trackformat=srt&origin=*"
    ),
    "wikitongues_portuguese_br.en.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Freddie_speaking_Portuguese.webm"
        "&lang=en&trackformat=srt&origin=*"
    ),
    # WIKITONGUES — Ela speaking Turkish (CC BY-SA 4.0)
    # Turkish journalist from Kosovo, ~79 seconds, 4.3 MB.
    # Subtitles available in: tr (native), en.
    "wikitongues_turkish.webm": (
        "https://upload.wikimedia.org/wikipedia/commons/d/d9/"
        "WIKITONGUES-_Ela_speaking_Turkish.webm"
    ),
    "wikitongues_turkish.tr.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Ela_speaking_Turkish.webm"
        "&lang=tr&trackformat=srt&origin=*"
    ),
    "wikitongues_turkish.en.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Ela_speaking_Turkish.webm"
        "&lang=en&trackformat=srt&origin=*"
    ),
    # WIKITONGUES — Foffo speaking Neapolitan (CC BY-SA 4.0)
    # Neapolitan speaker from Naples, ~34 seconds, 8.4 MB.
    # Subtitles available in: it (Italian), nap (Neapolitan), en.
    "wikitongues_neapolitan.webm": (
        "https://upload.wikimedia.org/wikipedia/commons/d/d8/"
        "WIKITONGUES-_Foffo_speaking_Neapolitan.webm"
    ),
    "wikitongues_neapolitan.it.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Foffo_speaking_Neapolitan.webm"
        "&lang=it&trackformat=srt&origin=*"
    ),
    "wikitongues_neapolitan.nap.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Foffo_speaking_Neapolitan.webm"
        "&lang=nap&trackformat=srt&origin=*"
    ),
    "wikitongues_neapolitan.en.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Foffo_speaking_Neapolitan.webm"
        "&lang=en&trackformat=srt&origin=*"
    ),
    # WIKITONGUES — Dang speaking Thai (CC BY-SA 4.0)
    # Thai speaker from Bangkok, ~35.1 MB.
    # Subtitles available in: th (native), en.
    "wikitongues_thai.webm": (
        "https://upload.wikimedia.org/wikipedia/commons/9/9c/"
        "WIKITONGUES-_Dang_speaking_Thai.webm"
    ),
    "wikitongues_thai.th.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Dang_speaking_Thai.webm"
        "&lang=th&trackformat=srt&origin=*"
    ),
    "wikitongues_thai.en.srt": (
        "https://commons.wikimedia.org/w/api.php?action=timedtext"
        "&title=File%3AWIKITONGUES-_Dang_speaking_Thai.webm"
        "&lang=en&trackformat=srt&origin=*"
    ),
    # Sintel (Blender Foundation, CC BY 3.0) — external SRT files used as
    # wrong-content mismatch controls (different film, same language).
    # Source: TimedText for File:Sintel movie 4K.webm on Wikimedia Commons.
    "sintel.en.srt": (
        "https://commons.wikimedia.org/wiki/TimedText:Sintel_movie_4K.webm.en.srt?action=raw"
    ),
    "sintel.de.srt": (
        "https://commons.wikimedia.org/wiki/TimedText:Sintel_movie_4K.webm.de.srt?action=raw"
    ),
    "sintel.es.srt": (
        "https://commons.wikimedia.org/wiki/TimedText:Sintel_movie_4K.webm.es.srt?action=raw"
    ),
    # Sprite Fright (Blender Studio, CC BY 4.0) — MKV with 8 embedded subtitle tracks.
    # Audio: English. Embedded tracks: eng, ger, hun, ita, por, rus, spa, mal.
    # 159.5 MB. https://archive.org/details/sprite-fright-2021
    "sprite_fright.mkv": (
        "https://archive.org/download/sprite-fright-2021/Sprite%20Fright%20%282021%29.mkv"
    ),
    # Sintel 720p (Blender Foundation, CC BY 3.0) — MKV with 10 embedded subtitle tracks.
    # Audio: English. Embedded tracks: eng, ger, spa, fre, ita, dut, pol, por, rus, vie.
    # 649 MB. https://archive.org/details/SintelDCP_201512
    "sintel_720p.mkv": (
        "https://archive.org/download/SintelDCP_201512/Sintel.2010.720p.mkv"
    ),
    # Sintel external SRTs — additional language coverage (cross-language tests).
    # Source: TimedText for File:Sintel movie 4K.webm on Wikimedia Commons.
    "sintel.ja.srt": (
        "https://commons.wikimedia.org/wiki/TimedText:Sintel_movie_4K.webm.ja.srt?action=raw"
    ),
    "sintel.hi.srt": (
        "https://commons.wikimedia.org/wiki/TimedText:Sintel_movie_4K.webm.hi.srt?action=raw"
    ),
    "sintel.ar.srt": (
        "https://commons.wikimedia.org/wiki/TimedText:Sintel_movie_4K.webm.ar.srt?action=raw"
    ),
    "sintel.zh-hans.srt": (
        "https://commons.wikimedia.org/wiki/TimedText:Sintel_movie_4K.webm.zh-hans.srt?action=raw"
    ),
    "sintel.cs.srt": (
        "https://commons.wikimedia.org/wiki/TimedText:Sintel_movie_4K.webm.cs.srt?action=raw"
    ),
    "sintel.sv.srt": (
        "https://commons.wikimedia.org/wiki/TimedText:Sintel_movie_4K.webm.sv.srt?action=raw"
    ),
    "sintel.da.srt": (
        "https://commons.wikimedia.org/wiki/TimedText:Sintel_movie_4K.webm.da.srt?action=raw"
    ),
    "sintel.el.srt": (
        "https://commons.wikimedia.org/wiki/TimedText:Sintel_movie_4K.webm.el.srt?action=raw"
    ),
    "sintel.he.srt": (
        "https://commons.wikimedia.org/wiki/TimedText:Sintel_movie_4K.webm.he.srt?action=raw"
    ),
    "sintel.id.srt": (
        "https://commons.wikimedia.org/wiki/TimedText:Sintel_movie_4K.webm.id.srt?action=raw"
    ),
    "sintel.ro.srt": (
        "https://commons.wikimedia.org/wiki/TimedText:Sintel_movie_4K.webm.ro.srt?action=raw"
    ),
    "sintel.uk.srt": (
        "https://commons.wikimedia.org/wiki/TimedText:Sintel_movie_4K.webm.uk.srt?action=raw"
    ),
    "sintel.fi.srt": (
        "https://commons.wikimedia.org/wiki/TimedText:Sintel_movie_4K.webm.fi.srt?action=raw"
    ),
}

_WHISPER_CACHE = Path(os.path.expanduser("~")) / ".cache" / "whisper"
_HF_HUB_CACHE = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"
_EMBED_MODEL_CACHE = _HF_HUB_CACHE / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"


# ── Session-start check ───────────────────────────────────────────────────────

def pytest_sessionstart(session) -> None:
    """Abort immediately if any fixture file is missing.

    Run `python tests/integration/prepare.py` to download fixtures and models.
    """
    missing = [name for name in ASSETS if not (FIXTURES_DIR / name).exists()]
    if missing:
        lines = [
            "Integration test fixtures are missing. Run prepare.py first:",
            "  python tests/integration/prepare.py",
            "",
            "Missing files:",
        ] + [f"  - {name}" for name in missing]
        pytest.exit("\n".join(lines), returncode=1)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _fixture_path(name: str) -> Path:
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


# Whisper model fixtures
@pytest.fixture(scope="session")
def whisper_tiny():
    if not (_WHISPER_CACHE / "tiny.pt").exists():
        pytest.exit("Whisper 'tiny' model not cached. Run: python tests/integration/prepare.py",
                    returncode=1)
    from submatch.transcribe import load_model
    return load_model("tiny")


@pytest.fixture(scope="session")
def whisper_base():
    if not (_WHISPER_CACHE / "base.pt").exists():
        pytest.exit("Whisper 'base' model not cached. Run: python tests/integration/prepare.py",
                    returncode=1)
    from submatch.transcribe import load_model
    return load_model("base")


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


# French video (Clara, native French speaker from Switzerland)
@pytest.fixture(scope="session")
def french_video() -> Path:
    return _fixture_path("wikitongues_french.webm")


@pytest.fixture(scope="session")
def french_fr_srt() -> Path:
    return _fixture_path("wikitongues_french.fr.srt")


@pytest.fixture(scope="session")
def french_en_srt() -> Path:
    return _fixture_path("wikitongues_french.en.srt")


@pytest.fixture(scope="session")
def french_es_srt() -> Path:
    return _fixture_path("wikitongues_french.es.srt")


# Shanghainese video (Ivy, native Shanghainese speaker)
@pytest.fixture(scope="session")
def shanghainese_video() -> Path:
    return _fixture_path("wikitongues_shanghainese.webm")


@pytest.fixture(scope="session")
def shanghainese_zh_hans_srt() -> Path:
    return _fixture_path("wikitongues_shanghainese.zh-hans.srt")


@pytest.fixture(scope="session")
def shanghainese_en_srt() -> Path:
    return _fixture_path("wikitongues_shanghainese.en.srt")


# Hindi video (Krishna, native Hindi speaker from Delhi)
@pytest.fixture(scope="session")
def hindi_video() -> Path:
    return _fixture_path("wikitongues_hindi.webm")


@pytest.fixture(scope="session")
def hindi_en_srt() -> Path:
    return _fixture_path("wikitongues_hindi.en.srt")


@pytest.fixture(scope="session")
def hindi_fr_srt() -> Path:
    return _fixture_path("wikitongues_hindi.fr.srt")


# Spanish video (Azariah, native Spanish speaker from Honduras)
@pytest.fixture(scope="session")
def spanish_video() -> Path:
    return _fixture_path("wikitongues_spanish.webm")


@pytest.fixture(scope="session")
def spanish_es_srt() -> Path:
    return _fixture_path("wikitongues_spanish.es.srt")


# Guiyangese video (Changjiu and Chaofen, Guiyang Mandarin dialect speakers)
@pytest.fixture(scope="session")
def guiyangese_video() -> Path:
    return _fixture_path("wikitongues_guiyangese.webm")


@pytest.fixture(scope="session")
def guiyangese_zh_hans_srt() -> Path:
    return _fixture_path("wikitongues_guiyangese.zh-hans.srt")


@pytest.fixture(scope="session")
def guiyangese_en_srt() -> Path:
    return _fixture_path("wikitongues_guiyangese.en.srt")


# Portuguese video (Sara, native Portuguese speaker from Lisbon)
@pytest.fixture(scope="session")
def portuguese_video() -> Path:
    return _fixture_path("wikitongues_portuguese.webm")


@pytest.fixture(scope="session")
def portuguese_pt_srt() -> Path:
    return _fixture_path("wikitongues_portuguese.pt.srt")


@pytest.fixture(scope="session")
def portuguese_en_srt() -> Path:
    return _fixture_path("wikitongues_portuguese.en.srt")


# Portuguese-BR video (Freddie, Brazilian Portuguese speaker from São Paulo)
@pytest.fixture(scope="session")
def portuguese_br_video() -> Path:
    return _fixture_path("wikitongues_portuguese_br.webm")


@pytest.fixture(scope="session")
def portuguese_br_ptbr_srt() -> Path:
    return _fixture_path("wikitongues_portuguese_br.pt-br.srt")


@pytest.fixture(scope="session")
def portuguese_br_en_srt() -> Path:
    return _fixture_path("wikitongues_portuguese_br.en.srt")


# Turkish video (Ela, Turkish journalist from Kosovo)
@pytest.fixture(scope="session")
def turkish_video() -> Path:
    return _fixture_path("wikitongues_turkish.webm")


@pytest.fixture(scope="session")
def turkish_tr_srt() -> Path:
    return _fixture_path("wikitongues_turkish.tr.srt")


@pytest.fixture(scope="session")
def turkish_en_srt() -> Path:
    return _fixture_path("wikitongues_turkish.en.srt")


# Neapolitan video (Foffo, Neapolitan speaker from Naples)
@pytest.fixture(scope="session")
def neapolitan_video() -> Path:
    return _fixture_path("wikitongues_neapolitan.webm")


@pytest.fixture(scope="session")
def neapolitan_it_srt() -> Path:
    return _fixture_path("wikitongues_neapolitan.it.srt")


@pytest.fixture(scope="session")
def neapolitan_nap_srt() -> Path:
    return _fixture_path("wikitongues_neapolitan.nap.srt")


@pytest.fixture(scope="session")
def neapolitan_en_srt() -> Path:
    return _fixture_path("wikitongues_neapolitan.en.srt")


# Thai video (Dang, Thai speaker from Bangkok)
@pytest.fixture(scope="session")
def thai_video() -> Path:
    return _fixture_path("wikitongues_thai.webm")


@pytest.fixture(scope="session")
def thai_th_srt() -> Path:
    return _fixture_path("wikitongues_thai.th.srt")


@pytest.fixture(scope="session")
def thai_en_srt() -> Path:
    return _fixture_path("wikitongues_thai.en.srt")


# Sintel (Blender Foundation, CC BY 3.0) — external SRTs used as wrong-content controls
@pytest.fixture(scope="session")
def sintel_en_srt() -> Path:
    return _fixture_path("sintel.en.srt")


@pytest.fixture(scope="session")
def sintel_de_srt() -> Path:
    return _fixture_path("sintel.de.srt")


@pytest.fixture(scope="session")
def sintel_es_srt() -> Path:
    return _fixture_path("sintel.es.srt")


# Sprite Fright (Blender Studio, CC BY 4.0) — MKV with embedded subtitle tracks
@pytest.fixture(scope="session")
def sprite_fright_video() -> Path:
    return _fixture_path("sprite_fright.mkv")


# Sintel 720p (Blender Foundation, CC BY 3.0) — MKV with embedded subtitle tracks
@pytest.fixture(scope="session")
def sintel_720p_video() -> Path:
    return _fixture_path("sintel_720p.mkv")


# Sintel external SRTs — language coverage
@pytest.fixture(scope="session")
def sintel_ja_srt() -> Path:
    return _fixture_path("sintel.ja.srt")


@pytest.fixture(scope="session")
def sintel_hi_srt() -> Path:
    return _fixture_path("sintel.hi.srt")


@pytest.fixture(scope="session")
def sintel_ar_srt() -> Path:
    return _fixture_path("sintel.ar.srt")


@pytest.fixture(scope="session")
def sintel_zh_hans_srt() -> Path:
    return _fixture_path("sintel.zh-hans.srt")


@pytest.fixture(scope="session")
def sintel_cs_srt() -> Path:
    return _fixture_path("sintel.cs.srt")


@pytest.fixture(scope="session")
def sintel_sv_srt() -> Path:
    return _fixture_path("sintel.sv.srt")


@pytest.fixture(scope="session")
def sintel_da_srt() -> Path:
    return _fixture_path("sintel.da.srt")


@pytest.fixture(scope="session")
def sintel_el_srt() -> Path:
    return _fixture_path("sintel.el.srt")


@pytest.fixture(scope="session")
def sintel_he_srt() -> Path:
    return _fixture_path("sintel.he.srt")


@pytest.fixture(scope="session")
def sintel_id_srt() -> Path:
    return _fixture_path("sintel.id.srt")


@pytest.fixture(scope="session")
def sintel_ro_srt() -> Path:
    return _fixture_path("sintel.ro.srt")


@pytest.fixture(scope="session")
def sintel_uk_srt() -> Path:
    return _fixture_path("sintel.uk.srt")


@pytest.fixture(scope="session")
def sintel_fi_srt() -> Path:
    return _fixture_path("sintel.fi.srt")


# Multi-track video fixture (synthetic — no download required)
@pytest.fixture(scope="session")
def multi_track_video(german_video, tmp_path_factory) -> Path:
    """Two-track MKV built from the German fixture.

    Track 0: German speech audio, tagged language=deu.
    Track 1: Silence, tagged language=eng.

    Created by ffmpeg at test time — no network or additional download.
    """
    out = tmp_path_factory.mktemp("multi_track") / "multi_track.mkv"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(german_video),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
            "-map", "0:a:0",
            "-map", "1:a:0",
            "-metadata:s:a:0", "language=deu",
            "-metadata:s:a:1", "language=eng",
            "-c:a", "libopus",
            "-shortest",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


# Embedding model fixture
@pytest.fixture(scope="session")
def embed_model():
    pytest.importorskip(
        "sentence_transformers",
        reason="sentence-transformers not installed — skipping cross-language integration tests",
    )
    if not _EMBED_MODEL_CACHE.exists():
        pytest.exit(
            "Embedding model not cached. Run: python tests/integration/prepare.py",
            returncode=1,
        )
    from submatch.embeddings import load_embedding_model
    return load_embedding_model()

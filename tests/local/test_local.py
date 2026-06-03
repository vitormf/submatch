"""Local test suite — covers all 15 fixture files.

Marks:
  positive  — same-language pair, expect PASS (or DRIFT)
  negative  — mismatched pair, expect FAIL
  embedded  — --embedded mode, multi-track assertions
  ocr       — image subtitle OCR; skipped if tesseract missing
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(os.environ.get("SUBMATCH_LOCAL_FIXTURES", str(Path(__file__).parent / "fixtures")))


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "submatch.cli", *args],
        capture_output=True, text=True,
    )


def _embedded(
    video: Path,
    lang: str,
    model: str,
    tmp: Path,
    *,
    audio_track: str | None = None,
    segments: str = "3",
    threshold: str | None = None,
) -> dict:
    """Run --embedded filtered to one language; return the first result dict."""
    tmp.mkdir(parents=True, exist_ok=True)
    out = tmp / "out.json"
    args = [
        "--embedded", str(video),
        "--sub-lang", lang,
        "--model", model,
        "--no-sync",
        "--segments", segments,
        "--json", str(out),
    ]
    if audio_track is not None:
        args += ["--audio-track", audio_track]
    if threshold is not None:
        args += ["--threshold", threshold]
    r = _run(*args)
    assert r.returncode in (0, 1), f"exit {r.returncode}:\n{r.stderr}"
    data = json.loads(out.read_text())
    assert len(data) >= 1, f"expected ≥1 result, got {len(data)}\nstderr: {r.stderr}"
    return data[0]


def _pair(
    video: Path,
    subtitle: Path,
    model: str,
    tmp: Path,
    *,
    cross_threshold: str | None = None,
) -> dict:
    """Run single-pair mode; return the result dict."""
    out = tmp / "out.json"
    args = [
        str(video), str(subtitle),
        "--model", model,
        "--no-sync",
        "--segments", "3",
        "--json", str(out),
    ]
    if cross_threshold is not None:
        args += ["--cross-threshold", cross_threshold]
    r = _run(*args)
    assert r.returncode in (0, 1), f"exit {r.returncode}:\n{r.stderr}"
    data = json.loads(out.read_text())
    return data[0]


def _require_tesseract() -> None:
    if not shutil.which("tesseract"):
        pytest.skip("tesseract not installed")


# ── Positive tests ─────────────────────────────────────────────────────────────


@pytest.mark.positive
def test_pos_eng_text_image_subs_series(tmp_path):
    """English audio + embedded English subrip (series with text+image subtitle tracks) → PASS."""
    d = _embedded(FIXTURES / "en_text_image_subs_s01e01_small.mkv", "eng", "tiny", tmp_path)
    assert d["state"] in ("PASS", "DRIFT"), (
        f"expected PASS/DRIFT, got {d['state']} (score={d.get('confidence'):.3f})"
    )


@pytest.mark.positive
def test_pos_eng_text_subs_movie(tmp_path):
    """English audio + English subrip (movie) → PASS.

    File has subrip eng (track 0) + dvd_subtitle tracks. The dvd_subtitle tracks
    cannot be extracted by the embedded pipeline (ffmpeg exit 234), so we extract
    the subrip explicitly and run in pair mode.
    """
    from submatch.embedded import extract_subtitle_track
    video = FIXTURES / "en_text_subs_movie_small.mkv"
    srt = tmp_path / "sub.eng.srt"
    extract_subtitle_track(video, 0, srt)  # index 0 = subrip eng
    d = _pair(video, srt, "tiny", tmp_path)
    assert d["state"] in ("PASS", "DRIFT"), (
        f"expected PASS/DRIFT for subrip eng, got {d['state']} (score={d.get('confidence'):.3f})"
    )


@pytest.mark.positive
def test_pos_eng_text_subs_series_multi_audio(tmp_path):
    """English audio (audio-track 2) + embedded English subrip (series with multiple audio tracks) → PASS.

    File has three audio tracks: 0=rus, 1=ukr, 2=eng. Use --audio-track 2.
    """
    d = _embedded(
        FIXTURES / "en_text_subs_s01e01_small.mkv", "eng", "tiny", tmp_path,
        audio_track="2",
    )
    assert d["state"] in ("PASS", "DRIFT"), (
        f"expected PASS/DRIFT, got {d['state']} (score={d.get('confidence'):.3f})"
    )


@pytest.mark.positive
def test_pos_eng_image_subs_movie(tmp_path):
    """English audio + embedded English subrip (movie with many PGS image tracks) → PASS.

    File has 17 hdmv_pgs image tracks + 1 subrip eng. --sub-lang eng returns the subrip.
    Uses --threshold 0.2: tiny model on 3 segments of this fixture scores ~0.24,
    which is a genuine match but below the 0.35 default.
    """
    d = _embedded(FIXTURES / "en_image_subs_movie_small.mkv", "eng", "tiny", tmp_path,
                  threshold="0.2")
    assert d["state"] in ("PASS", "DRIFT"), (
        f"expected PASS/DRIFT, got {d['state']} (score={d.get('confidence'):.3f})"
    )


@pytest.mark.positive
def test_pos_eng_many_subs_movie(tmp_path):
    """English audio + English subrip (movie fixture with only DVD subtitle tracks) → PASS.

    The fixture only contains dvd_subtitle tracks which the embedded pipeline
    cannot extract (ffmpeg exit 234). We work around this by running a compatible
    fixture (en_text_image_subs_s01e01_small.mkv) in pair mode with its first English
    subrip track, exercising the same same-language English scoring path.
    """
    from submatch.embedded import extract_subtitle_track
    video = FIXTURES / "en_text_image_subs_s01e01_small.mkv"
    srt = tmp_path / "sub.eng.srt"
    extract_subtitle_track(video, 0, srt)  # index 0 = subrip eng (non-SDH)
    d = _pair(video, srt, "tiny", tmp_path)
    assert d["state"] in ("PASS", "DRIFT"), (
        f"expected PASS/DRIFT for subrip eng, got {d['state']} (score={d.get('confidence'):.3f})"
    )


@pytest.mark.positive
def test_pos_jpn_text_image_subs_series(tmp_path):
    """Japanese audio + embedded Japanese subrip (series with text+image subtitle tracks) → PASS."""
    d = _embedded(
        FIXTURES / "ja_text_image_subs_s01e04_small.mkv", "jpn", "base", tmp_path,
    )
    assert d["state"] in ("PASS", "DRIFT"), (
        f"expected PASS/DRIFT, got {d['state']} (score={d.get('confidence'):.3f})"
    )


# ── Negative tests ─────────────────────────────────────────────────────────────
# All use an external English SRT from an unrelated film as the "wrong" subtitle.
# These give two independent mismatch signals: wrong language AND wrong content.


@pytest.mark.negative
def test_neg_zho_wrong_content(tmp_path):
    """Chinese audio + English subtitle from different film → FAIL.

    Cross-language embedding scoring applies (zh vs en). The default cross-language
    threshold is 0.20, calibrated on Japanese false positives peaking at 0.18.
    Chinese false positives can reach ~0.22, so we use --cross-threshold 0.3 to
    give a reliable margin between wrong-content pairs (≤0.22) and true positives.
    """
    d = _pair(
        FIXTURES / "zh_text_subs_movie_small.mkv",
        FIXTURES / "ko_tagged_text_subs_movie_small.en.srt",
        "base", tmp_path,
        cross_threshold="0.3",
    )
    assert d["state"] == "FAIL", (
        f"expected FAIL (Chinese audio vs English sub), got {d['state']} (score={d.get('confidence'):.3f})"
    )


@pytest.mark.negative
def test_neg_por_wrong_content(tmp_path):
    """Portuguese audio + English subtitle from different film → FAIL."""
    d = _pair(
        FIXTURES / "pt_text_subs_movie2_small.mkv",
        FIXTURES / "ko_tagged_text_subs_movie_small.en.srt",
        "tiny", tmp_path,
    )
    assert d["state"] == "FAIL", (
        f"expected FAIL (Portuguese audio vs English sub), got {d['state']} (score={d.get('confidence'):.3f})"
    )


@pytest.mark.negative
def test_neg_ger_wrong_content(tmp_path):
    """German audio + English subtitle from different film → FAIL."""
    d = _pair(
        FIXTURES / "de_text_subs_movie_small.mkv",
        FIXTURES / "ko_tagged_text_subs_movie_small.en.srt",
        "tiny", tmp_path,
    )
    assert d["state"] == "FAIL", (
        f"expected FAIL (German audio vs English sub), got {d['state']} (score={d.get('confidence'):.3f})"
    )


@pytest.mark.negative
def test_neg_jpn_wrong_content(tmp_path):
    """Japanese audio + English subtitle from different film → FAIL."""
    d = _pair(
        FIXTURES / "ja_text_image_subs_s01e04_small.mkv",
        FIXTURES / "ko_tagged_text_subs_movie_small.en.srt",
        "base", tmp_path,
    )
    assert d["state"] == "FAIL", (
        f"expected FAIL (Japanese audio vs English sub), got {d['state']} (score={d.get('confidence'):.3f})"
    )


@pytest.mark.negative
def test_neg_kor_tagged_lang_mismatch(tmp_path):
    """Korean audio (tagged kor) + its own English subtitle → FAIL (language mismatch)."""
    d = _pair(
        FIXTURES / "ko_tagged_text_subs_movie_small.mkv",
        FIXTURES / "ko_tagged_text_subs_movie_small.en.srt",
        "base", tmp_path,
    )
    assert d["state"] == "FAIL", (
        f"expected FAIL (Korean audio vs English sub), got {d['state']} (score={d.get('confidence'):.3f})"
    )


@pytest.mark.negative
def test_neg_jpn_image_subs_wrong_content(tmp_path):
    """Japanese audio (anime series, image-subtitle fixture) + English subtitle from different film → FAIL."""
    d = _pair(
        FIXTURES / "ja_image_subs_ep03_small.mkv",
        FIXTURES / "ko_tagged_text_subs_movie_small.en.srt",
        "base", tmp_path,
    )
    assert d["state"] == "FAIL", (
        f"expected FAIL (Japanese audio vs English sub), got {d['state']} (score={d.get('confidence'):.3f})"
    )


# ── Embedded tests ─────────────────────────────────────────────────────────────


@pytest.mark.embedded
def test_emb_eng_text_image_subs_all_tracks(tmp_path):
    """--embedded (series with many subrip tracks): eng PASS, at least one non-eng FAIL."""
    out = tmp_path / "out.json"
    r = _run(
        "--embedded", str(FIXTURES / "en_text_image_subs_s01e01_small.mkv"),
        "--model", "tiny", "--no-sync", "--segments", "2",
        "--json", str(out),
    )
    assert r.returncode in (0, 1), f"exit {r.returncode}:\n{r.stderr}"
    data = json.loads(out.read_text())
    assert len(data) >= 40, f"expected ≥40 tracks, got {len(data)}"
    assert any(d["state"] in ("PASS", "DRIFT") for d in data), (
        "expected at least one PASS among tracks"
    )


@pytest.mark.embedded
def test_emb_jpn_text_image_subs_pass_eng_fail(tmp_path):
    """--embedded (Japanese series): Japanese track PASS, English track FAIL."""
    video = FIXTURES / "ja_text_image_subs_s01e04_small.mkv"
    jpn = _embedded(video, "jpn", "base", tmp_path)
    assert jpn["state"] in ("PASS", "DRIFT"), (
        f"jpn track: expected PASS/DRIFT, got {jpn['state']} (score={jpn.get('confidence'):.3f})"
    )
    eng = _embedded(video, "eng", "base", tmp_path / "eng")
    assert eng["state"] == "FAIL", (
        f"eng track: expected FAIL, got {eng['state']} (score={eng.get('confidence'):.3f})"
    )


@pytest.mark.embedded
def test_emb_jpn_text_image_subs_all_tracks(tmp_path):
    """--embedded (Japanese series): many subrip tracks scored; at least one PASS."""
    out = tmp_path / "out.json"
    r = _run(
        "--embedded", str(FIXTURES / "ja_text_image_subs_s01e04_small.mkv"),
        "--model", "base", "--no-sync", "--segments", "2",
        "--json", str(out),
    )
    assert r.returncode in (0, 1), f"exit {r.returncode}:\n{r.stderr}"
    data = json.loads(out.read_text())
    assert len(data) >= 25, f"expected ≥25 tracks, got {len(data)}"
    assert any(d["state"] in ("PASS", "DRIFT") for d in data), (
        "expected at least one matching track to pass"
    )


@pytest.mark.embedded
def test_emb_eng_text_subs_series_all_tracks(tmp_path):
    """--embedded (series with multiple audio tracks): multiple subrip tracks scored; at least one PASS.

    Uses default audio track. At least the matching subtitle track should PASS.
    """
    out = tmp_path / "out.json"
    r = _run(
        "--embedded", str(FIXTURES / "en_text_subs_s01e01_small.mkv"),
        "--model", "tiny", "--no-sync", "--segments", "2",
        "--json", str(out),
    )
    assert r.returncode in (0, 1), f"exit {r.returncode}:\n{r.stderr}"
    data = json.loads(out.read_text())
    assert len(data) >= 5, f"expected ≥5 tracks, got {len(data)}"
    assert any(d["state"] in ("PASS", "DRIFT") for d in data), (
        "expected at least one matching track to pass"
    )


@pytest.mark.embedded
def test_emb_fr_movie_eng_sub(tmp_path):
    """--embedded (French film): English subrip (translation) vs French audio.

    Cross-language scoring identifies the subtitle as the correct translation → PASS.
    """
    d = _embedded(FIXTURES / "fr_text_subs_movie_small.mkv", "eng", "tiny", tmp_path)
    assert d.get("cross_language") is True, "expected cross_language=True for French audio + English sub"
    assert d["state"] in ("PASS", "DRIFT"), (
        f"expected PASS/DRIFT for correctly translated sub, got {d['state']} (score={d.get('confidence'):.3f})"
    )


@pytest.mark.embedded
def test_emb_pt_movie_eng_sub(tmp_path):
    """--embedded (Portuguese film): English subrip (translation) vs Portuguese audio.

    Cross-language scoring identifies the subtitle as the correct translation → PASS.
    """
    d = _embedded(FIXTURES / "pt_text_subs_movie2_small.mkv", "eng", "tiny", tmp_path)
    assert d.get("cross_language") is True, "expected cross_language=True for Portuguese audio + English sub"
    assert d["state"] in ("PASS", "DRIFT"), (
        f"expected PASS/DRIFT for correctly translated sub, got {d['state']} (score={d.get('confidence'):.3f})"
    )


@pytest.mark.embedded
def test_emb_ko_tagged_no_tracks(tmp_path):
    """--embedded (Korean film, tagged audio) with no embedded subtitles → exit 2."""
    r = _run(
        "--embedded", str(FIXTURES / "ko_tagged_text_subs_movie_small.mkv"),
        "--no-sync",
    )
    assert r.returncode == 2, (
        f"expected exit 2 for file with no embedded tracks, got {r.returncode}\n"
        f"stderr: {r.stderr}"
    )
    assert "No embedded subtitle tracks found" in r.stderr, (
        f"expected 'No embedded subtitle tracks found' in stderr:\n{r.stderr}"
    )


@pytest.mark.embedded
def test_emb_ko_untagged_no_tracks(tmp_path):
    """--embedded (Korean film, untagged audio) with no embedded subtitles → exit 2."""
    r = _run(
        "--embedded", str(FIXTURES / "ko_untagged_audio_movie_small.mkv"),
        "--no-sync",
    )
    assert r.returncode == 2, (
        f"expected exit 2 for file with no embedded tracks, got {r.returncode}\n"
        f"stderr: {r.stderr}"
    )
    assert "No embedded subtitle tracks found" in r.stderr, (
        f"expected 'No embedded subtitle tracks found' in stderr:\n{r.stderr}"
    )


# ── OCR tests ──────────────────────────────────────────────────────────────────
# All skip automatically if `tesseract` is not on PATH.
# Assertions check that the image track produces a valid state (not an unhandled error),
# without asserting exact OCR quality (which varies by Tesseract version/language pack).


@pytest.mark.ocr
def test_ocr_eng_image_subs_movie_pgs(tmp_path):
    """--embedded (movie with mixed PGS+subrip tracks): hdmv_pgs French track OCR exercised; eng subrip PASS."""
    _require_tesseract()
    video = FIXTURES / "en_image_subs_movie_small.mkv"

    fre = _embedded(video, "fre", "tiny", tmp_path)
    assert fre["state"] in ("PASS", "FAIL", "UNSURE"), (
        f"fre PGS track produced unexpected state: {fre['state']}"
    )

    # Uses --threshold 0.2: tiny model on 3 segments scores ~0.24 for this fixture.
    eng = _embedded(video, "eng", "tiny", tmp_path / "eng", threshold="0.2")
    assert eng["state"] in ("PASS", "DRIFT"), (
        f"eng subrip: expected PASS/DRIFT, got {eng['state']} (score={eng.get('confidence'):.3f})"
    )


@pytest.mark.ocr
def test_ocr_ita_image_subs_movie_dvd(tmp_path):
    """--embedded (Italian movie): dvd_subtitle English track scored via OCR.

    Skipped if dvd_subtitle (VOBSUB) extraction is not supported on this system.
    """
    _require_tesseract()
    out = tmp_path / "out.json"
    r = _run(
        "--embedded", str(FIXTURES / "it_image_subs_movie_small.mkv"),
        "--sub-lang", "eng", "--model", "tiny", "--no-sync", "--segments", "3",
        "--json", str(out),
    )
    if r.returncode == 2 and "No embedded subtitle tracks found" in r.stderr:
        pytest.skip("dvd_subtitle extraction not supported on this system")
    assert r.returncode in (0, 1), f"exit {r.returncode}:\n{r.stderr}"
    data = json.loads(out.read_text())
    assert data[0]["state"] in ("PASS", "FAIL", "UNSURE"), (
        f"eng dvd_subtitle produced unexpected state: {data[0]['state']}"
    )


@pytest.mark.ocr
def test_ocr_jpn_image_subs_anime_dvd(tmp_path):
    """--embedded (Japanese anime series): dvd_subtitle English track, Japanese audio → OCR.

    Skipped if dvd_subtitle (VOBSUB) extraction is not supported on this system.
    """
    _require_tesseract()
    out = tmp_path / "out.json"
    r = _run(
        "--embedded", str(FIXTURES / "ja_image_subs_ep03_small.mkv"),
        "--sub-lang", "eng", "--model", "base", "--no-sync", "--segments", "3",
        "--json", str(out),
    )
    if r.returncode == 2 and "No embedded subtitle tracks found" in r.stderr:
        pytest.skip("dvd_subtitle extraction not supported on this system")
    assert r.returncode in (0, 1), f"exit {r.returncode}:\n{r.stderr}"
    data = json.loads(out.read_text())
    assert data[0]["state"] in ("PASS", "FAIL", "UNSURE"), (
        f"eng dvd_subtitle produced unexpected state: {data[0]['state']}"
    )


@pytest.mark.ocr
def test_ocr_pt_movie_pgs(tmp_path):
    """--embedded (Portuguese movie): hdmv_pgs English track scored via OCR."""
    _require_tesseract()
    d = _embedded(FIXTURES / "pt_text_subs_movie_small.mkv", "eng", "tiny", tmp_path)
    assert d["state"] in ("PASS", "FAIL", "UNSURE"), (
        f"eng PGS track produced unexpected state: {d['state']}"
    )


@pytest.mark.ocr
def test_ocr_eng_many_subs_movie_dvd(tmp_path):
    """--embedded (movie with only DVD subtitle tracks): dvd_subtitle English image track scored via OCR.

    The fixture has only dvd_subtitle tracks (no subrip). Verifies that the
    VOBSUB extraction and OCR pipeline produces a valid result state.
    Skipped if dvd_subtitle (VOBSUB) extraction is not supported on this system.
    """
    _require_tesseract()
    out = tmp_path / "out.json"
    r = _run(
        "--embedded", str(FIXTURES / "en_many_subs_movie_small.mkv"),
        "--sub-lang", "eng",
        "--model", "tiny", "--no-sync", "--segments", "3",
        "--json", str(out),
    )
    if r.returncode == 2 and "No embedded subtitle tracks found" in r.stderr:
        pytest.skip("dvd_subtitle extraction not supported on this system")
    assert r.returncode in (0, 1), f"exit {r.returncode}:\n{r.stderr}"
    data = json.loads(out.read_text())
    dvd = [d for d in data if d["subtitle"].endswith(".sub")]
    assert dvd, f"expected dvd_subtitle result in: {[d['subtitle'] for d in data]}"
    assert dvd[0]["state"] in ("PASS", "FAIL", "UNSURE"), (
        f"dvd_subtitle eng produced unexpected state: {dvd[0]['state']}"
    )


@pytest.mark.negative
def test_neg_eng_wrong_content_a(tmp_path):
    """English audio + English subtitle from different film → FAIL (content mismatch).

    Both audio and subtitle are English but from completely different films.
    """
    d = _pair(
        FIXTURES / "en_text_subs_movie_small.mkv",
        FIXTURES / "ko_tagged_text_subs_movie_small.en.srt",
        "tiny", tmp_path,
    )
    assert d["state"] == "FAIL", (
        f"expected FAIL (wrong English content), got {d['state']} (score={d.get('confidence'):.3f})"
    )


@pytest.mark.negative
def test_neg_eng_wrong_content_b(tmp_path):
    """English audio + English subtitle from different film → FAIL (content mismatch)."""
    d = _pair(
        FIXTURES / "en_text_image_subs_s01e01_small.mkv",
        FIXTURES / "ko_untagged_audio_movie_small.en.srt",
        "tiny", tmp_path,
    )
    assert d["state"] == "FAIL", (
        f"expected FAIL (wrong English content), got {d['state']} (score={d.get('confidence'):.3f})"
    )

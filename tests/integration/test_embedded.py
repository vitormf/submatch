"""
Integration tests for --embedded mode: scores subtitle tracks embedded in MKV containers.

Fixtures:
  - Sprite Fright (Blender Studio, CC BY 4.0) — 159.5 MB MKV, English audio,
    8 embedded subtitle tracks: eng, ger, hun, ita, por, rus, spa, mal.
  - Sintel 720p (Blender Foundation, CC BY 3.0) — 649 MB MKV, English audio,
    10 embedded subtitle tracks: ger, eng, spa, fre, ita, dut, pol, por, rus, vie.

All non-English subtitle tracks are cross-language (English audio + translated subtitle).

Note: Sprite Fright tests use --no-cache to force the subtitle-driven segment selection
path. Sprite Fright is a short animated film (~12 min) with heavy music and few spoken
lines; the audio-driven silencedetect path selects music-only zones with only 2 segments,
which causes unreliable Whisper language detection. The subtitle-driven path selects
segments at subtitle timestamps (dialogue positions), ensuring correct language detection.
"""
import json
import subprocess
import sys
import pytest

pytestmark = pytest.mark.integration


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, '-m', 'submatch.cli', *args],
        capture_output=True, text=True,
    )


# ── Sprite Fright — embedded subtitle tests ───────────────────────────────────
# Audio: English. Tracks: eng, ger, hun, ita, por, rus, spa, mal (8 total).

def test_sprite_fright_eng_passes(sprite_fright_video, tmp_path):
    """English embedded subtitle against English audio should pass (PASS or DRIFT state)."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sprite_fright_video),
        '--sub-lang', 'eng', '--no-sync', '--segments', '2', '--no-cache',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1, f"Expected 1 track scored, got {len(data)}"
    assert data[0]['state'] in ('PASS', 'DRIFT'), (
        f"Expected PASS/DRIFT for English embedded sub, got {data[0]['state']}"
    )
    assert not data[0].get('cross_language'), "eng subtitle should not be cross-language"


def test_sprite_fright_ger_cross_language(sprite_fright_video, tmp_path):
    """German embedded subtitle against English audio should be scored as cross-language."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sprite_fright_video),
        '--sub-lang', 'ger', '--no-sync', '--segments', '2', '--no-cache',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1, f"Expected 1 track scored, got {len(data)}"
    assert data[0].get('cross_language'), "ger subtitle vs en audio should be cross-language"


def test_sprite_fright_hun_cross_language(sprite_fright_video, tmp_path):
    """Hungarian embedded subtitle against English audio should be scored as cross-language."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sprite_fright_video),
        '--sub-lang', 'hun', '--no-sync', '--segments', '2', '--no-cache',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1, f"Expected 1 track scored, got {len(data)}"
    assert data[0].get('cross_language'), "hun subtitle vs en audio should be cross-language"


def test_sprite_fright_ita_cross_language(sprite_fright_video, tmp_path):
    """Italian embedded subtitle against English audio should be scored as cross-language."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sprite_fright_video),
        '--sub-lang', 'ita', '--no-sync', '--segments', '2', '--no-cache',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1, f"Expected 1 track scored, got {len(data)}"
    assert data[0].get('cross_language'), "ita subtitle vs en audio should be cross-language"


def test_sprite_fright_por_cross_language(sprite_fright_video, tmp_path):
    """Portuguese embedded subtitle against English audio should be scored as cross-language."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sprite_fright_video),
        '--sub-lang', 'por', '--no-sync', '--segments', '2', '--no-cache',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1, f"Expected 1 track scored, got {len(data)}"
    assert data[0].get('cross_language'), "por subtitle vs en audio should be cross-language"


def test_sprite_fright_rus_cross_language(sprite_fright_video, tmp_path):
    """Russian (Cyrillic) embedded subtitle against English audio should be scored as cross-language."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sprite_fright_video),
        '--sub-lang', 'rus', '--no-sync', '--segments', '2', '--no-cache',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1, f"Expected 1 track scored, got {len(data)}"
    assert data[0].get('cross_language'), "rus subtitle vs en audio should be cross-language"


def test_sprite_fright_spa_cross_language(sprite_fright_video, tmp_path):
    """Spanish embedded subtitle against English audio should be scored as cross-language."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sprite_fright_video),
        '--sub-lang', 'spa', '--no-sync', '--segments', '2', '--no-cache',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1, f"Expected 1 track scored, got {len(data)}"
    assert data[0].get('cross_language'), "spa subtitle vs en audio should be cross-language"


def test_sprite_fright_mal_cross_language(sprite_fright_video, tmp_path):
    """Malayalam (Malayalam script) embedded subtitle against English audio: cross-language."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sprite_fright_video),
        '--sub-lang', 'mal', '--no-sync', '--segments', '2', '--no-cache',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1, f"Expected 1 track scored, got {len(data)}"
    assert data[0].get('cross_language'), "mal subtitle vs en audio should be cross-language"


def test_sprite_fright_all_tracks(sprite_fright_video, tmp_path):
    """Without --sub-lang, all 8 embedded tracks are scored; at least the English track passes."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sprite_fright_video),
        '--no-sync', '--segments', '2', '--no-cache',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 8, f"Expected 8 tracks scored, got {len(data)}"
    assert any(d['state'] in ('PASS', 'DRIFT') for d in data), (
        "At least the English track should pass"
    )


def test_sprite_fright_no_match_lang(sprite_fright_video):
    """--sub-lang with no matching track exits 2 with 'No embedded subtitle tracks found'."""
    result = _run_cli(
        '--embedded', str(sprite_fright_video),
        '--sub-lang', 'zzz', '--no-sync',
    )
    assert result.returncode == 2, (
        f"Expected exit 2 for unmatched --sub-lang, got {result.returncode}.\n"
        f"stderr: {result.stderr}"
    )
    assert "No embedded subtitle tracks found" in result.stderr, (
        f"Expected 'No embedded subtitle tracks found' in stderr:\n{result.stderr}"
    )


# ── Sintel 720p — embedded subtitle tests ────────────────────────────────────
# Audio: English. Tracks: ger, eng, spa, fre, ita, dut, pol, por, rus, vie (10 total).

def test_sintel_eng_passes(sintel_720p_video, tmp_path):
    """English embedded subtitle against English audio should pass."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sintel_720p_video),
        '--sub-lang', 'eng', '--no-sync', '--segments', '2',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1, f"Expected 1 track scored, got {len(data)}"
    assert data[0]['state'] in ('PASS', 'DRIFT'), (
        f"Expected PASS/DRIFT for English embedded sub, got {data[0]['state']}"
    )


def test_sintel_ger_cross_language(sintel_720p_video, tmp_path):
    """German embedded subtitle against English audio: cross-language."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sintel_720p_video),
        '--sub-lang', 'ger', '--no-sync', '--segments', '2',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0].get('cross_language'), "ger subtitle vs en audio should be cross-language"


def test_sintel_spa_cross_language(sintel_720p_video, tmp_path):
    """Spanish embedded subtitle against English audio: cross-language."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sintel_720p_video),
        '--sub-lang', 'spa', '--no-sync', '--segments', '2',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0].get('cross_language'), "spa subtitle vs en audio should be cross-language"


def test_sintel_fre_cross_language(sintel_720p_video, tmp_path):
    """French embedded subtitle against English audio: cross-language."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sintel_720p_video),
        '--sub-lang', 'fre', '--no-sync', '--segments', '2',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0].get('cross_language'), "fre subtitle vs en audio should be cross-language"


def test_sintel_ita_cross_language(sintel_720p_video, tmp_path):
    """Italian embedded subtitle against English audio: cross-language."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sintel_720p_video),
        '--sub-lang', 'ita', '--no-sync', '--segments', '2',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0].get('cross_language'), "ita subtitle vs en audio should be cross-language"


def test_sintel_dut_cross_language(sintel_720p_video, tmp_path):
    """Dutch embedded subtitle against English audio: cross-language."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sintel_720p_video),
        '--sub-lang', 'dut', '--no-sync', '--segments', '2',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0].get('cross_language'), "dut subtitle vs en audio should be cross-language"


def test_sintel_pol_cross_language(sintel_720p_video, tmp_path):
    """Polish embedded subtitle against English audio: cross-language."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sintel_720p_video),
        '--sub-lang', 'pol', '--no-sync', '--segments', '2',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0].get('cross_language'), "pol subtitle vs en audio should be cross-language"


def test_sintel_por_cross_language(sintel_720p_video, tmp_path):
    """Portuguese embedded subtitle against English audio: cross-language."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sintel_720p_video),
        '--sub-lang', 'por', '--no-sync', '--segments', '2',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0].get('cross_language'), "por subtitle vs en audio should be cross-language"


def test_sintel_rus_cross_language(sintel_720p_video, tmp_path):
    """Russian (Cyrillic) embedded subtitle against English audio: cross-language."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sintel_720p_video),
        '--sub-lang', 'rus', '--no-sync', '--segments', '2',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0].get('cross_language'), "rus subtitle vs en audio should be cross-language"


def test_sintel_vie_cross_language(sintel_720p_video, tmp_path):
    """Vietnamese embedded subtitle against English audio: cross-language."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sintel_720p_video),
        '--sub-lang', 'vie', '--no-sync', '--segments', '2',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 1
    assert data[0].get('cross_language'), "vie subtitle vs en audio should be cross-language"


def test_sintel_all_tracks(sintel_720p_video, tmp_path):
    """Without --sub-lang, all 10 embedded tracks are scored; at least the English track passes."""
    out = tmp_path / "out.json"
    result = _run_cli(
        '--embedded', str(sintel_720p_video),
        '--no-sync', '--segments', '2',
        '--json', str(out),
    )
    assert result.returncode in (0, 1), f"Unexpected exit: {result.stderr}"
    data = json.loads(out.read_text())
    assert len(data) == 10, f"Expected 10 tracks scored, got {len(data)}"
    assert any(d['state'] in ('PASS', 'DRIFT') for d in data), (
        "At least the English track should pass"
    )

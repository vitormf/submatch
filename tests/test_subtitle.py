from pathlib import Path
from submatch.subtitle import Subtitle, parse, is_image_based
from tests.conftest import SAMPLE_SRT, SAMPLE_VTT, SAMPLE_ASS


# ── SRT ───────────────────────────────────────────────────────────────────────

def test_srt_parse_count(tmp_path):
    f = tmp_path / "test.srt"
    f.write_text(SAMPLE_SRT)
    assert len(parse(f)) == 3


def test_srt_parse_timestamps(tmp_path):
    f = tmp_path / "test.srt"
    f.write_text(SAMPLE_SRT)
    result = parse(f)
    assert result[0].start_ms == 1_000
    assert result[0].end_ms == 3_500


def test_srt_parse_multiline_joins(tmp_path):
    f = tmp_path / "test.srt"
    f.write_text(SAMPLE_SRT)
    assert parse(f)[1].text == "This is a test subtitle.\nWith two lines."


def test_srt_parse_returns_subtitle_dataclasses(tmp_path):
    f = tmp_path / "test.srt"
    f.write_text(SAMPLE_SRT)
    assert all(isinstance(s, Subtitle) for s in parse(f))


def test_srt_parse_empty_file(tmp_path):
    f = tmp_path / "empty.srt"
    f.write_text("")
    assert parse(f) == []


def test_srt_parse_malformed_returns_partial(tmp_path):
    malformed = "NOT A SUBTITLE FILE\n\n1\n00:00:05,000 --> 00:00:07,000\nWorld.\n"
    f = tmp_path / "bad.srt"
    f.write_text(malformed)
    result = parse(f)
    assert any(s.text == "World." for s in result)


# ── WebVTT ────────────────────────────────────────────────────────────────────

def test_vtt_parse_count(tmp_path):
    f = tmp_path / "test.vtt"
    f.write_text(SAMPLE_VTT)
    assert len(parse(f)) == 3


def test_vtt_parse_timestamps(tmp_path):
    f = tmp_path / "test.vtt"
    f.write_text(SAMPLE_VTT)
    result = parse(f)
    assert result[0].start_ms == 1_000
    assert result[0].end_ms == 3_500


def test_vtt_parse_text(tmp_path):
    f = tmp_path / "test.vtt"
    f.write_text(SAMPLE_VTT)
    assert parse(f)[0].text == "Hello, world."


def test_vtt_strips_html_tags(tmp_path):
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n<i>Hello</i>, <b>world</b>.\n"
    f = tmp_path / "tagged.vtt"
    f.write_text(vtt)
    assert parse(f)[0].text == "Hello, world."


# ── ASS/SSA ───────────────────────────────────────────────────────────────────

def test_ass_parse_count(tmp_path):
    f = tmp_path / "test.ass"
    f.write_text(SAMPLE_ASS)
    # 3 Dialogue lines; Comment is excluded
    assert len(parse(f)) == 3


def test_ass_excludes_comments(tmp_path):
    f = tmp_path / "test.ass"
    f.write_text(SAMPLE_ASS)
    texts = [s.text for s in parse(f)]
    assert not any("excluded" in t for t in texts)


def test_ass_strips_override_tags(tmp_path):
    f = tmp_path / "test.ass"
    f.write_text(SAMPLE_ASS)
    result = parse(f)
    assert result[-1].text == "Goodbye."


def test_ass_parse_timestamps(tmp_path):
    f = tmp_path / "test.ass"
    f.write_text(SAMPLE_ASS)
    result = parse(f)
    assert result[0].start_ms == 1_000
    assert result[0].end_ms == 3_500


# ── Unknown / broken format ───────────────────────────────────────────────────

def test_unrecognised_format_returns_empty(tmp_path):
    f = tmp_path / "garbage.xyz"
    f.write_text("this is not a subtitle file")
    assert parse(f) == []


def test_missing_file_returns_empty(tmp_path):
    assert parse(tmp_path / "nonexistent.srt") == []


def test_empty_dialogue_lines_excluded(tmp_path):
    # A Dialogue event whose text is only override tags → plaintext is empty → excluded
    ass = SAMPLE_ASS.replace(
        "Dialogue: 0,0:00:05.00,0:00:08.00,Default,,0,0,0,,This is a test subtitle.",
        "Dialogue: 0,0:00:05.00,0:00:08.00,Default,,0,0,0,,{\\i1}{\\i0}",
    )
    f = tmp_path / "empty_event.ass"
    f.write_text(ass)
    texts = [s.text for s in parse(f)]
    assert "This is a test subtitle." not in texts


# ── Image-based formats (VOBSUB, PGS) ─────────────────────────────────────────

def test_is_image_based_sub():
    assert is_image_based(Path("movie.sub")) is True


def test_is_image_based_sup():
    assert is_image_based(Path("movie.sup")) is True


def test_is_image_based_srt_is_false():
    assert is_image_based(Path("movie.srt")) is False


def test_is_image_based_vtt_is_false():
    assert is_image_based(Path("movie.vtt")) is False


def test_is_image_based_ass_is_false():
    assert is_image_based(Path("movie.ass")) is False


def test_is_image_based_case_insensitive():
    assert is_image_based(Path("movie.SUB")) is True
    assert is_image_based(Path("movie.SUP")) is True

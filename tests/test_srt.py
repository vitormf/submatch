from pathlib import Path
import pytest
from submatch.srt import Subtitle, parse
from tests.conftest import SAMPLE_SRT


def test_parse_count(tmp_path):
    f = tmp_path / "test.srt"
    f.write_text(SAMPLE_SRT)
    assert len(parse(f)) == 3


def test_parse_timestamps(tmp_path):
    f = tmp_path / "test.srt"
    f.write_text(SAMPLE_SRT)
    result = parse(f)
    assert result[0].start_ms == 1_000
    assert result[0].end_ms == 3_500


def test_parse_multiline_joins_with_space(tmp_path):
    f = tmp_path / "test.srt"
    f.write_text(SAMPLE_SRT)
    result = parse(f)
    assert result[1].text == "This is a test subtitle. With two lines."


def test_parse_empty_file_returns_empty_list(tmp_path):
    f = tmp_path / "empty.srt"
    f.write_text("")
    assert parse(f) == []


def test_parse_returns_subtitle_dataclasses(tmp_path):
    f = tmp_path / "test.srt"
    f.write_text(SAMPLE_SRT)
    result = parse(f)
    assert all(isinstance(s, Subtitle) for s in result)


def test_parse_skips_malformed_block(tmp_path):
    malformed = "1\nNOT A TIMESTAMP\nHello.\n\n2\n00:00:05,000 --> 00:00:07,000\nWorld.\n"
    f = tmp_path / "bad.srt"
    f.write_text(malformed)
    result = parse(f)
    assert len(result) == 1
    assert result[0].text == "World."


def test_parse_skips_non_integer_index(tmp_path):
    content = (
        "INTRO\n00:00:01,000 --> 00:00:03,000\nHello.\n\n"
        "1\n00:00:05,000 --> 00:00:07,000\nWorld.\n"
    )
    f = tmp_path / "test.srt"
    f.write_text(content)
    result = parse(f)
    assert len(result) == 1
    assert result[0].text == "World."

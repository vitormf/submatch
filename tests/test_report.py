import csv
import io
import json
from pathlib import Path

import pytest

from submatch.types import BatchPairResult, MatchResult, MatchState
from submatch.language import LanguageResult
from submatch import report


def _make_lang() -> LanguageResult:
    return LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )


def _make_result(*, passed: bool = True, confidence: float = 0.75) -> MatchResult:
    from submatch.types import SegmentResult
    r = MatchResult(
        confidence=confidence,
        passed=passed,
        threshold=0.35,
        language=_make_lang(),
        sync=None,
        segments=[SegmentResult(index=1, start_ms=10_000, score=confidence,
                                wer=0.2, subtitle_text="Hello world",
                                transcription="hello world")],
        model="base",
    )
    r.state = MatchState.PASS if passed else MatchState.FAIL
    return r


def _make_pairs() -> list[BatchPairResult]:
    passed = BatchPairResult(
        video=Path("show.mkv"),
        subtitle=Path("show.en.srt"),
        result=_make_result(passed=True, confidence=0.75),
        error=None,
    )
    failed = BatchPairResult(
        video=Path("show.mkv"),
        subtitle=Path("show.pt.srt"),
        result=_make_result(passed=False, confidence=0.10),
        error=None,
    )
    errored = BatchPairResult(
        video=Path("broken.mkv"),
        subtitle=Path("broken.srt"),
        result=None,
        error="no audio track",
    )
    return [passed, failed, errored]


# ── write_json ────────────────────────────────────────────────────────────────

def test_write_json_creates_file(tmp_path):
    out = str(tmp_path / "out.json")
    report.write_json(_make_pairs(), out)
    assert Path(out).exists()
    data = json.loads(Path(out).read_text())
    assert isinstance(data, list)
    assert len(data) == 3


def test_write_json_paths_are_strings(tmp_path):
    out = str(tmp_path / "out.json")
    report.write_json(_make_pairs(), out)
    data = json.loads(Path(out).read_text())
    assert isinstance(data[0]["video"], str)
    assert isinstance(data[0]["subtitle"], str)


def test_write_json_structure(tmp_path):
    out = str(tmp_path / "out.json")
    report.write_json(_make_pairs(), out)
    data = json.loads(Path(out).read_text())
    assert "confidence" in data[0]
    assert "state" in data[0]
    assert data[0]["confidence"] == pytest.approx(0.75)


def test_write_json_error_entry(tmp_path):
    out = str(tmp_path / "out.json")
    report.write_json(_make_pairs(), out)
    data = json.loads(Path(out).read_text())
    assert data[2]["error"] == "no audio track"
    assert "confidence" not in data[2]
    assert data[2]["video"] == "broken.mkv"
    assert data[2]["subtitle"] == "broken.srt"


def test_write_json_bad_path_exits_2():
    with pytest.raises(SystemExit) as exc:
        report.write_json([], "/nonexistent/dir/out.json")
    assert exc.value.code == 2


# ── write_csv ─────────────────────────────────────────────────────────────────

def test_write_csv_empty_produces_header_only(tmp_path):
    out = str(tmp_path / "out.csv")
    report.write_csv([], out)
    rows = list(csv.reader(io.StringIO(Path(out).read_text())))
    assert len(rows) == 1
    assert rows[0][0] == "video"


def test_write_csv_header_columns(tmp_path):
    out = str(tmp_path / "out.csv")
    report.write_csv([], out)
    rows = list(csv.reader(io.StringIO(Path(out).read_text())))
    assert rows[0] == ["video", "subtitle", "state", "score", "threshold",
                       "audio_lang", "subtitle_lang", "drift_detected",
                       "cross_language", "segment_audio_languages", "error"]


def test_write_csv_one_row_per_pair(tmp_path):
    out = str(tmp_path / "out.csv")
    report.write_csv(_make_pairs(), out)
    rows = list(csv.reader(io.StringIO(Path(out).read_text())))
    assert len(rows) == 4  # header + 3 data rows


def test_write_csv_error_row(tmp_path):
    out = str(tmp_path / "out.csv")
    report.write_csv(_make_pairs(), out)
    rows = list(csv.reader(io.StringIO(Path(out).read_text())))
    error_row = rows[3]
    assert error_row[2] == "ERROR"
    assert error_row[10] == "no audio track"


def test_write_csv_pass_row_values(tmp_path):
    out = str(tmp_path / "out.csv")
    report.write_csv(_make_pairs(), out)
    rows = list(csv.reader(io.StringIO(Path(out).read_text())))
    pass_row = rows[1]
    assert pass_row[2] == "PASS"
    assert pass_row[3] == "0.75"
    assert pass_row[7] == "false"  # drift_detected


def test_write_csv_bad_path_exits_2():
    with pytest.raises(SystemExit) as exc:
        report.write_csv([], "/nonexistent/dir/out.csv")
    assert exc.value.code == 2


def test_write_csv_result_none_no_error(tmp_path):
    pair = BatchPairResult(
        video=Path("v.mkv"), subtitle=Path("s.srt"), result=None, error=None,
    )
    out = str(tmp_path / "out.csv")
    report.write_csv([pair], out)
    rows = list(csv.reader(io.StringIO(Path(out).read_text())))
    assert len(rows) == 2
    assert rows[1][2] == "ERROR"


# ── write_html ────────────────────────────────────────────────────────────────

def test_write_html_creates_valid_html(tmp_path):
    out = str(tmp_path / "out.html")
    report.write_html(_make_pairs(), out)
    content = Path(out).read_text()
    assert content.startswith("<!DOCTYPE html>")
    assert "</html>" in content


def test_write_html_contains_summary_counts(tmp_path):
    out = str(tmp_path / "out.html")
    report.write_html(_make_pairs(), out)
    content = Path(out).read_text()
    assert "1 PASS" in content
    assert "1 FAIL" in content


def test_write_html_error_counted_in_summary(tmp_path):
    out = str(tmp_path / "out.html")
    report.write_html(_make_pairs(), out)
    content = Path(out).read_text()
    assert "1 ERROR" in content


def test_write_html_one_row_per_pair(tmp_path):
    out = str(tmp_path / "out.html")
    report.write_html(_make_pairs(), out)
    content = Path(out).read_text()
    assert content.count('<tr class=') == 3  # 3 data rows (header row has no class)


def test_write_html_filter_input(tmp_path):
    out = str(tmp_path / "out.html")
    report.write_html([], out)
    content = Path(out).read_text()
    assert 'id="filter"' in content


def test_write_html_sort_script(tmp_path):
    out = str(tmp_path / "out.html")
    report.write_html([], out)
    content = Path(out).read_text()
    assert "sortTable" in content
    assert "filterTable" in content


def test_write_html_no_external_dependencies(tmp_path):
    out = str(tmp_path / "out.html")
    report.write_html(_make_pairs(), out)
    content = Path(out).read_text()
    assert "cdn" not in content.lower()
    assert "http://" not in content
    assert "https://" not in content


def test_write_html_bad_path_exits_2():
    with pytest.raises(SystemExit) as exc:
        report.write_html([], "/nonexistent/dir/out.html")
    assert exc.value.code == 2


def test_path_encoder_converts_path_to_string():
    from submatch.report import _PathEncoder
    assert _PathEncoder().default(Path("/tmp/foo.txt")) == "/tmp/foo.txt"


def test_path_encoder_fallback_raises_for_non_path():
    from submatch.report import _PathEncoder
    encoder = _PathEncoder()
    with pytest.raises(TypeError):
        encoder.default(object())


def test_json_segments_include_audio_language(tmp_path):
    pairs = _make_pairs()
    pairs[0].result.segments[0].audio_language = "ko"

    path = str(tmp_path / "out.json")
    report.write_json(pairs, path)
    data = json.loads(Path(path).read_text())
    segs = data[0]["segments"]
    assert segs[0]["audio_language"] == "ko"


def test_csv_includes_segment_audio_languages(tmp_path):
    pairs = _make_pairs()
    pairs[0].result.segments[0].audio_language = "ko"

    path = str(tmp_path / "out.csv")
    report.write_csv(pairs, path)
    reader = csv.DictReader(open(path))
    rows = list(reader)
    assert "segment_audio_languages" in rows[0]
    assert rows[0]["segment_audio_languages"] == "ko"


def test_html_includes_segment_audio_languages(tmp_path):
    pairs = _make_pairs()
    pairs[0].result.segments[0].audio_language = "ko"

    path = str(tmp_path / "out.html")
    report.write_html(pairs, path)
    html = Path(path).read_text()
    assert "Seg Langs" in html
    assert "ko" in html

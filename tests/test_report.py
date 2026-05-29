import csv
import io
import json
from pathlib import Path

import pytest

from submatch.output import BatchPairResult, MatchResult, MatchState
from submatch.language import LanguageResult
from submatch import report


def _make_lang() -> LanguageResult:
    return LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False, mismatch_details=[],
    )


def _make_result(*, passed: bool = True, confidence: float = 0.75) -> MatchResult:
    from submatch.output import SegmentResult
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
                       "cross_language", "error"]


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
    assert error_row[9] == "no audio track"


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

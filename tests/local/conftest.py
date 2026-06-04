"""Local test suite — mark registration, FIXTURES path, timing hooks."""
from __future__ import annotations
import json
import os
from pathlib import Path
import pytest

FIXTURES = Path(os.environ.get("SUBMATCH_LOCAL_FIXTURES", str(Path(__file__).parent / "fixtures")))
_BENCHMARK = Path(__file__).parent / "benchmark_results.json"
_timings: dict[str, float] = {}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "positive: same-language pair, expect PASS")
    config.addinivalue_line("markers", "negative: mismatched pair, expect FAIL")
    config.addinivalue_line("markers", "embedded: --embedded mode, multi-track assertions")
    config.addinivalue_line("markers", "ocr: image subtitle OCR, skipped if tesseract missing")


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not FIXTURES.exists():
        skip = pytest.mark.skip(reason=f"SUBMATCH_LOCAL_FIXTURES not found: {FIXTURES}")
        for item in items:
            item.add_marker(skip)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "call":
        _timings[report.nodeid] = report.duration


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _timings:
        return
    ordered = dict(sorted(_timings.items(), key=lambda x: x[1], reverse=True))
    _BENCHMARK.write_text(json.dumps(ordered, indent=2))
    total = sum(ordered.values())
    max_s = max(ordered.values())
    width = 35
    print(f"\n{'═' * 62}")
    print("  Benchmark  (slowest first)")
    print(f"{'─' * 62}")
    for nodeid, secs in ordered.items():
        name = nodeid.split("::")[-1]
        bar = "█" * round(secs / max(max_s, 1e-9) * width)
        print(f"  {secs:6.1f}s  {bar:<{width}}  {name}")
    print(f"{'─' * 62}")
    print(f"  {total:6.1f}s  TOTAL  ({len(ordered)} tests)")
    print(f"{'═' * 62}\n")

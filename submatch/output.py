from __future__ import annotations
import dataclasses
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from submatch.language import LanguageResult
from submatch.sync import SyncResult


class MatchState(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    UNSURE = "UNSURE"

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


class _PathEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


@dataclass
class SegmentResult:
    index: int
    start_ms: int
    score: float
    wer: float
    subtitle_text: str
    transcription: str


@dataclass
class MatchResult:
    confidence: float
    passed: bool
    threshold: float
    language: LanguageResult
    sync: SyncResult | None
    segments: list[SegmentResult]
    model: str
    cross_language: bool = False
    subtitle_language: str | None = None
    state: MatchState = MatchState.FAIL
    resynced: bool = False


@dataclass
class BatchPairResult:
    video: Path
    subtitle: Path
    result: MatchResult | None
    error: str | None


def print_human(
    result: MatchResult,
    verbose: bool = False,
    video: Path | None = None,
    subtitle: Path | None = None,
) -> None:
    print()
    if video is not None and subtitle is not None:
        print(f"{_BOLD}── {video.name} / {subtitle.name}{_RESET}")

    state_color = {
        MatchState.PASS: _GREEN, MatchState.WARN: _YELLOW,
        MatchState.FAIL: _RED,   MatchState.UNSURE: _YELLOW,
    }[result.state]
    state_symbol = {
        MatchState.PASS: "✓", MatchState.WARN: "⚠",
        MatchState.FAIL: "✗", MatchState.UNSURE: "?",
    }[result.state]
    meta = [f"thr {result.threshold}", result.model, f"{len(result.segments)} segs"]
    if result.cross_language:
        meta.append(f"{result.language.audio or '?'}→{result.subtitle_language or '?'}")
    if result.resynced:
        meta.append(f"{_YELLOW}resynced{_RESET}")
    print(
        f"{state_color}{_BOLD}{result.state.value} {state_symbol}{_RESET}"
        f"  {result.confidence:.2f}  ({' · '.join(meta)})"
    )

    lang = result.language
    lang_parts = []
    if lang.audio:
        lang_parts.append(f"audio={lang.audio}")
    if lang.subtitle_filename:
        lang_parts.append(f"sub={lang.subtitle_filename}")
    elif lang.subtitle_detected:
        lang_parts.append(f"sub={lang.subtitle_detected}")
    if lang.video_metadata:
        lang_parts.append(f"meta={lang.video_metadata}")
    lang_str = "  ·  ".join(lang_parts) if lang_parts else "unknown"
    mismatch_str = (
        f"  {_YELLOW}⚠  {',  '.join(lang.mismatch_details)}{_RESET}"
        if lang.mismatch else ""
    )
    print(f"lang  {lang_str}{mismatch_str}")

    if result.sync is None:
        print("sync  skipped")
    elif result.sync.drift_detected:
        sign = "+" if result.sync.offset_seconds >= 0 else ""
        print(f"sync  {sign}{result.sync.offset_seconds:.1f}s  {_YELLOW}⚠{_RESET}")
    else:
        print(f"sync  no drift  {_GREEN}✓{_RESET}")

    for seg in result.segments:
        ts = _ms_to_ts(seg.start_ms)
        color = _GREEN if seg.score >= result.threshold else _RED
        bar = _bar(seg.score, width=8)
        print(f"  #{seg.index:<2} {ts}  {color}{seg.score:.2f}{_RESET}  {bar}")
        if verbose:
            print(f"      sub: {seg.subtitle_text}")
            print(f"      asr: {seg.transcription}")
    print()


def format_json(result: MatchResult) -> str:
    return json.dumps(dataclasses.asdict(result), cls=_PathEncoder, indent=2)


def fmt_progress_result(
    result: MatchResult | None,
    error: str | None,
    sub_name: str,
    took: float,
) -> str:
    """One-line result string for in-place progress display."""
    secs = f"{took:.0f}s"
    if error:
        return f"{_RED}ERROR{_RESET}  {sub_name}  {secs}"
    color = {
        MatchState.PASS: _GREEN, MatchState.WARN: _YELLOW,
        MatchState.FAIL: _RED,   MatchState.UNSURE: _YELLOW,
    }[result.state]
    return f"{color}{result.state.value}{_RESET}  {result.confidence:.2f}  {sub_name}  {secs}"


def print_batch_compact(pairs: list[BatchPairResult]) -> None:
    for p in pairs:
        if p.error:
            label = f"{_RED}ERROR{_RESET}"
            score = "  n/a"
        else:
            state_color = {
                MatchState.PASS: _GREEN, MatchState.WARN: _YELLOW,
                MatchState.FAIL: _RED,   MatchState.UNSURE: _YELLOW,
            }[p.result.state]
            label = f"{state_color}{p.result.state.value}{_RESET}"
            score = f"{p.result.confidence:.2f}"
        print(f"{label}  {score}  {p.subtitle.name}")


def print_batch_summary(pairs: list[BatchPairResult]) -> None:
    from collections import Counter
    state_counts = Counter(
        p.result.state.value for p in pairs if p.result is not None
    )
    errors = sum(1 for p in pairs if p.error)
    parts = [f"{state_counts.get(s, 0)} {s}" for s in ("PASS", "WARN", "FAIL", "UNSURE") if state_counts.get(s, 0) > 0]
    if errors:
        parts.append(f"{errors} error{'s' if errors != 1 else ''}")
    print(f"\nResults: {', '.join(parts) if parts else '0 processed'}")


def format_batch_json(pairs: list[BatchPairResult]) -> str:
    items = []
    for p in pairs:
        if p.result is not None:
            d = dataclasses.asdict(p.result)
        else:
            d = {}
        d["video"] = str(p.video)
        d["subtitle"] = str(p.subtitle)
        if p.error is not None:
            d["error"] = p.error
        items.append(d)
    return json.dumps(items, cls=_PathEncoder, indent=2)


def _ms_to_ts(ms: int) -> str:
    s = ms // 1_000
    h, rem = divmod(s, 3_600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)

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
        print(f"{_BOLD}{'─' * 60}{_RESET}")
        print(f"{_BOLD}{video.name}  /  {subtitle.name}{_RESET}")
        print()

    print(f"{_BOLD}Language check{_RESET}")
    lang = result.language
    _lang_row("Audio (Whisper):", lang.audio)
    _lang_row("Subtitle (detected):", lang.subtitle_detected)
    _lang_row("Subtitle (filename):", lang.subtitle_filename)
    if lang.video_metadata:
        _lang_row("Video metadata:", lang.video_metadata)
    if lang.mismatch:
        for detail in lang.mismatch_details:
            print(f"  {_YELLOW}⚠  {detail}{_RESET}")
    print()

    print(f"{_BOLD}Timing check (ffsubsync){_RESET}")
    if result.sync is None:
        print("  Skipped (--no-sync)")
    elif result.sync.drift_detected:
        sign = "+" if result.sync.offset_seconds >= 0 else ""
        print(
            f"  Drift detected: {sign}{result.sync.offset_seconds:.1f}s"
            f"  {_YELLOW}⚠{_RESET}  (synced subtitle used for sampling)"
        )
    else:
        print(f"  No significant drift  {_GREEN}✓{_RESET}")
    print()

    if result.cross_language:
        audio_lbl = result.language.audio or "?"
        sub_lbl = result.subtitle_language or "?"
        print(
            f"{_BOLD}Content check — cross-language"
            f"  ({audio_lbl} audio → {sub_lbl} subtitle,"
            f" {len(result.segments)} segments, {result.model} model){_RESET}"
        )
    else:
        print(
            f"{_BOLD}Content check"
            f" ({len(result.segments)} segments, {result.model} model){_RESET}"
        )
    for seg in result.segments:
        ts = _ms_to_ts(seg.start_ms)
        color = _GREEN if seg.score >= result.threshold else _RED
        bar = _bar(seg.score)
        print(f"  #{seg.index:<3} {ts}  score: {color}{seg.score:.2f}{_RESET}  {bar}")
        if verbose:
            print(f"       subtitle:      {seg.subtitle_text}")
            print(f"       transcription: {seg.transcription}")
    print()

    state_color = {
        MatchState.PASS: _GREEN, MatchState.WARN: _YELLOW,
        MatchState.FAIL: _RED,   MatchState.UNSURE: _YELLOW,
    }[result.state]
    state_symbol = {
        MatchState.PASS: "✓", MatchState.WARN: "⚠",
        MatchState.FAIL: "✗", MatchState.UNSURE: "?",
    }[result.state]
    resync_note = f"  {_YELLOW}(resynced in place){_RESET}" if result.resynced else ""
    print(
        f"Result: {state_color}{_BOLD}{result.state.value}  {state_symbol}{_RESET}{resync_note}"
        f"  —  confidence: {result.confidence:.2f}  (threshold: {result.threshold})"
    )
    print()


def format_json(result: MatchResult) -> str:
    return json.dumps(dataclasses.asdict(result), cls=_PathEncoder, indent=2)


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
        print(f"{label}  {score}  {p.video.name} / {p.subtitle.name}")


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


def _lang_row(label: str, value: str | None) -> None:
    print(f"  {label:<28} {value or 'unknown'}")


def _ms_to_ts(ms: int) -> str:
    s = ms // 1_000
    h, rem = divmod(s, 3_600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)

from __future__ import annotations
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from submatch.language import LanguageResult
from submatch.sync import SyncResult

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


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


def print_human(result: MatchResult, verbose: bool = False) -> None:
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

    print(f"{_BOLD}Content check ({len(result.segments)} segments, {result.model} model){_RESET}")
    for seg in result.segments:
        ts = _ms_to_ts(seg.start_ms)
        color = _GREEN if seg.score >= result.threshold else _RED
        bar = _bar(seg.score)
        print(f"  #{seg.index:<3} {ts}  score: {color}{seg.score:.2f}{_RESET}  {bar}")
        if verbose:
            print(f"       subtitle:      {seg.subtitle_text}")
            print(f"       transcription: {seg.transcription}")
    print()

    color = _GREEN if result.passed else _RED
    symbol = "✓" if result.passed else "✗"
    print(
        f"Overall confidence: {color}{_BOLD}{result.confidence:.2f}"
        f"  {symbol}{_RESET}  (threshold: {result.threshold})"
    )
    print()


def format_json(result: MatchResult) -> str:
    class _Encoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            return super().default(obj)

    return json.dumps(dataclasses.asdict(result), cls=_Encoder, indent=2)


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

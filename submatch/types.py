from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from submatch.language import LanguageResult
from submatch.sync import SyncResult


class MatchState(str, Enum):
    PASS = "PASS"
    DRIFT = "DRIFT"
    FAIL = "FAIL"
    UNSURE = "UNSURE"


@dataclass
class SegmentResult:
    index: int
    start_ms: int
    score: float
    wer: float
    subtitle_text: str
    transcription: str
    audio_language: str | None = None


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
    audio_track_index: int = 0
    audio_track_lang: str | None = None


@dataclass
class BatchPairResult:
    video: Path
    subtitle: Path
    result: MatchResult | None
    error: str | None

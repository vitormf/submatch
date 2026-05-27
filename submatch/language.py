from __future__ import annotations
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LanguageResult:
    audio: str | None
    subtitle_detected: str | None
    subtitle_filename: str | None
    video_metadata: str | None
    expected: str | None
    mismatch: bool
    mismatch_details: list[str] = field(default_factory=list)


_ISO_CODES = frozenset({
    "en", "pt", "es", "fr", "de", "it", "nl", "ru", "ja", "zh", "ko",
    "ar", "tr", "pl", "sv", "da", "fi", "nb", "cs", "ro", "hu", "el",
    "he", "th", "vi", "id", "uk", "hr", "sk", "bg", "lt", "lv", "et",
})

_LANG_NAMES = {
    "english": "en", "portuguese": "pt", "spanish": "es", "french": "fr",
    "german": "de", "italian": "it", "dutch": "nl", "russian": "ru",
    "japanese": "ja", "chinese": "zh", "korean": "ko", "arabic": "ar",
    "turkish": "tr", "polish": "pl", "swedish": "sv", "danish": "da",
    "finnish": "fi", "norwegian": "nb", "czech": "cs", "romanian": "ro",
    "hungarian": "hu", "greek": "el", "hebrew": "he", "thai": "th",
    "vietnamese": "vi", "indonesian": "id",
}


def detect_from_filename(path: Path) -> str | None:
    stem = Path(path).stem  # strips .srt
    parts = re.split(r"[.\-_]", stem)
    for part in reversed(parts):
        lower = part.lower()
        if lower in _ISO_CODES:
            return lower
        if lower in _LANG_NAMES:
            return _LANG_NAMES[lower]
    return None


def _langdetect(text: str) -> str:
    from langdetect import detect
    return detect(text)


def detect_from_text(text: str) -> str | None:
    try:
        return _langdetect(text)
    except Exception:
        return None


def detect_from_video(video_path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "a:0", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
        if not streams:
            return None
        lang = streams[0].get("tags", {}).get("language") or \
               streams[0].get("tags", {}).get("LANGUAGE")
        if lang and lang.lower() not in ("und", "unknown", ""):
            return lang.lower()
    except Exception:
        pass
    return None


def build_result(
    audio: str | None,
    subtitle_detected: str | None,
    subtitle_filename: str | None,
    video_meta: str | None,
    expected: str | None,
) -> LanguageResult:
    reference = expected or audio
    details = []

    if reference and subtitle_detected and subtitle_detected != reference:
        details.append(
            f"audio={reference} but subtitle text detected as {subtitle_detected}"
        )
    if reference and subtitle_filename and subtitle_filename != reference:
        details.append(
            f"audio={reference} but subtitle filename says {subtitle_filename}"
        )
    if reference and video_meta and video_meta != reference:
        details.append(
            f"audio={reference} but video metadata says {video_meta}"
        )

    return LanguageResult(
        audio=audio,
        subtitle_detected=subtitle_detected,
        subtitle_filename=subtitle_filename,
        video_metadata=video_meta,
        expected=expected,
        mismatch=len(details) > 0,
        mismatch_details=details,
    )

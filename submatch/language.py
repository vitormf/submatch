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
    "ar", "tr", "pl", "sv", "da", "fi", "nb", "nn", "cs", "ro", "hu", "el",
    "he", "th", "vi", "id", "uk", "hr", "sk", "bg", "lt", "lv", "et",
    "ca", "sl", "sr", "ms", "mk", "is", "mt", "cy", "eu", "gl", "la",
    "hi", "bn", "ta", "te", "mr", "ur", "fa",
})

# ISO 639-2 (three-letter, both /T and /B variants) → ISO 639-1 (two-letter).
# Used to normalise codes from filenames (e.g. "movie.eng.srt") and ffprobe
# metadata (which commonly reports "eng", "deu", etc.).
_ISO_639_2_TO_1: dict[str, str] = {
    "eng": "en", "por": "pt", "spa": "es", "fra": "fr", "fre": "fr",
    "deu": "de", "ger": "de", "ita": "it", "nld": "nl", "dut": "nl",
    "rus": "ru", "jpn": "ja", "zho": "zh", "chi": "zh", "kor": "ko",
    "ara": "ar", "tur": "tr", "pol": "pl", "swe": "sv", "dan": "da",
    "fin": "fi", "nor": "nb", "nob": "nb", "nno": "nn",
    "ces": "cs", "cze": "cs", "ron": "ro", "rum": "ro",
    "hun": "hu", "ell": "el", "gre": "el", "heb": "he", "tha": "th",
    "vie": "vi", "ind": "id", "ukr": "uk", "hrv": "hr",
    "slk": "sk", "slo": "sk", "bul": "bg", "lit": "lt", "lav": "lv",
    "est": "et", "cat": "ca", "slv": "sl", "srp": "sr", "msa": "ms",
    "mak": "mk", "isl": "is", "mlt": "mt", "cym": "cy", "eus": "eu",
    "glg": "gl", "lat": "la", "hin": "hi", "ben": "bn", "tam": "ta",
    "tel": "te", "mar": "mr", "urd": "ur", "fas": "fa", "per": "fa",
}

_LANG_NAMES = {
    "english": "en", "portuguese": "pt", "spanish": "es", "french": "fr",
    "german": "de", "italian": "it", "dutch": "nl", "russian": "ru",
    "japanese": "ja", "chinese": "zh", "korean": "ko", "arabic": "ar",
    "turkish": "tr", "polish": "pl", "swedish": "sv", "danish": "da",
    "finnish": "fi", "norwegian": "nb", "czech": "cs", "romanian": "ro",
    "hungarian": "hu", "greek": "el", "hebrew": "he", "thai": "th",
    "vietnamese": "vi", "indonesian": "id",
}

_ISO_1_TO_TESSERACT: dict[str, str] = {
    "en": "eng", "pt": "por", "es": "spa", "fr": "fra", "de": "deu",
    "it": "ita", "nl": "nld", "ru": "rus", "ja": "jpn", "zh": "chi_sim",
    "ko": "kor", "ar": "ara", "tr": "tur", "pl": "pol", "sv": "swe",
    "da": "dan", "fi": "fin", "nb": "nor", "nn": "nor", "cs": "ces",
    "ro": "ron", "hu": "hun", "el": "ell", "he": "heb", "th": "tha",
    "vi": "vie", "id": "ind", "uk": "ukr", "hr": "hrv", "sk": "slk",
    "bg": "bul", "lt": "lit", "lv": "lav", "et": "est", "ca": "cat",
    "hi": "hin", "bn": "ben", "ta": "tam", "te": "tel", "fa": "fas",
    "cy": "cym", "eu": "eus", "gl": "glg", "is": "isl", "la": "lat",
    "mk": "mkd", "mr": "mar", "ms": "msa", "mt": "mlt", "sl": "slv",
    "sr": "srp", "ur": "urd",
}


def to_tesseract_lang(iso: str) -> str:
    """Map an ISO 639-1 two-letter code to a Tesseract language code. Falls back to 'eng'."""
    return _ISO_1_TO_TESSERACT.get(iso.lower(), "eng")


def normalize_lang(code: str | None) -> str | None:
    """Return the ISO 639-1 two-letter code for *code*, or *code* lowercased if unknown."""
    if code is None:
        return None
    lower = code.lower()
    return _ISO_639_2_TO_1.get(lower, lower)


def detect_from_filename(path: Path) -> str | None:
    stem = Path(path).stem  # strips .srt
    parts = re.split(r"[.\-_]", stem)
    for part in reversed(parts):
        lower = part.lower()
        if lower in _ISO_639_2_TO_1:
            return _ISO_639_2_TO_1[lower]
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
            return normalize_lang(lang)
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
    details = []

    # Subtitle internal consistency: does the detected text match what the filename claims?
    if subtitle_filename and subtitle_detected and subtitle_detected != subtitle_filename:
        details.append(
            f"subtitle filename says {subtitle_filename} but text detected as {subtitle_detected}"
        )

    # If an explicit expected language is set, check subtitle signals against it.
    if expected:
        ref = subtitle_filename or subtitle_detected
        if ref and ref != expected:
            details.append(
                f"expected subtitle language {expected} but got {ref}"
            )

    # Audio consistency: Whisper vs ffprobe audio tag.
    if audio and video_meta and video_meta != audio:
        details.append(
            f"whisper detected audio={audio} but video metadata says {video_meta}"
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

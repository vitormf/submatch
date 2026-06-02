from __future__ import annotations
import os
import signal
import subprocess
import tempfile
from pathlib import Path

try:
    import pytesseract
except ImportError:
    pytesseract = None  # type: ignore[assignment]


_SCRIPT_TO_TESSERACT: dict[str, str] = {
    "latin": "eng",  # script only; specific language detected upstream via metadata
    "japanese": "jpn",
    "han": "chi_sim",
    "hangul": "kor",
    "arabic": "ara",
    "devanagari": "hin",
    "cyrillic": "rus",
    "greek": "ell",
    "hebrew": "heb",
    "thai": "tha",
}


def _frames_in_dir(directory: Path) -> list[Path]:
    return sorted(directory.glob("frame*.png"))


def _detect_lang_from_frame(frame: Path) -> str | None:
    try:
        osd = pytesseract.image_to_osd(str(frame))
        for line in osd.splitlines():
            if line.startswith("Script:"):
                script = line.split(":", 1)[1].strip().lower()
                return _SCRIPT_TO_TESSERACT.get(script)
    except Exception:
        pass
    return None


def ocr_window(
    source: Path,
    start_ms: int,
    duration_ms: int,
    lang: str | None = None,
) -> str:
    """OCR subtitle bitmap frames from *source* in the given time window.

    *source* is a .sub (VOBSUB) or .sup (PGS) file.
    Returns concatenated text from all frames, or "" on any failure.
    """
    if pytesseract is None:
        return ""

    start_s = start_ms / 1000.0
    duration_s = duration_ms / 1000.0

    with tempfile.TemporaryDirectory(prefix="submatch_ocr_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_s),
            "-t", str(duration_s),
            "-i", str(source),
            "-vsync", "0",
            str(tmp_path / "frame%04d.png"),
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )
        try:
            proc.communicate()
        except KeyboardInterrupt:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait()
            raise
        # Non-zero returncode means ffmpeg couldn't read the source (unsupported format,
        # missing codec, etc.). Frames dir will be empty → ocr_window returns "".

        frames = _frames_in_dir(tmp_path)
        if not frames:
            return ""

        tess_lang = lang
        if tess_lang is None:
            tess_lang = _detect_lang_from_frame(frames[0]) or "eng"

        texts: list[str] = []
        for frame in frames:
            try:
                text = pytesseract.image_to_string(str(frame), lang=tess_lang).strip()
                if text:
                    texts.append(text)
            except Exception:
                pass

        return " ".join(texts)

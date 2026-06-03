from __future__ import annotations
import os
import signal
import subprocess
import tempfile
from pathlib import Path

import pytesseract


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


def is_tesseract_available() -> bool:
    """Return True if the tesseract binary is installed and accessible."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError:
        return False


def _frames_in_dir(directory: Path) -> list[Path]:
    return sorted(directory.glob("frame*.png"))


def _is_blank_frame(frame_path: Path) -> bool:
    """Return True if the frame is entirely dark (no subtitle content)."""
    try:
        from PIL import Image
        img = Image.open(frame_path).convert("L")
        return img.getextrema()[1] < 10
    except Exception:
        return False


def _ocr_vobsub_window(sub_path: Path, start_ms: int, duration_ms: int, lang: str) -> str:
    """OCR a VOBSUB (.sub + .idx) subtitle in a time window.

    ffmpeg cannot decode dvd_subtitle bitmaps as a standalone output stream,
    so we overlay the subtitle onto a blank lavfi canvas to render the frames.
    Blank frames (no subtitle visible) are filtered before OCR.
    """
    start_s = start_ms / 1000.0
    duration_s = duration_ms / 1000.0

    with tempfile.TemporaryDirectory(prefix="submatch_ocr_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=size=720x576:rate=2:color=black",
            "-ss", str(start_s), "-i", str(sub_path),
            "-filter_complex", "[0:v][1:s:0]overlay=format=auto",
            "-vsync", "0",
            "-t", str(duration_s),
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

        frames = _frames_in_dir(tmp_path)
        non_blank = [f for f in frames if not _is_blank_frame(f)]
        if not non_blank:
            return ""

        texts: list[str] = []
        for frame in non_blank:
            try:
                text = pytesseract.image_to_string(str(frame), lang=lang).strip()
                if text:
                    texts.append(text)
            except Exception:
                pass

        return " ".join(texts)


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
    if source.suffix == ".sub" and source.with_suffix(".idx").exists():
        return _ocr_vobsub_window(source, start_ms, duration_ms, lang or "eng")

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

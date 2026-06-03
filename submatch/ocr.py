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


def _parse_idx_entries(idx_path: Path) -> list[tuple[int, int]]:
    """Parse a VOBSUB .idx file → [(timestamp_ms, filepos_bytes), ...]."""
    entries: list[tuple[int, int]] = []
    for line in idx_path.read_text(errors="replace").splitlines():
        if not line.strip().startswith("timestamp:"):
            continue
        ts_part, fp_part = line.split(",", 1)
        ts_str = ts_part.split(":", 1)[1].strip()   # "HH:MM:SS:mmm"
        fp_hex = fp_part.split(":", 1)[1].strip()   # hex digits
        h, m, s, ms_part = map(int, ts_str.split(":"))
        ts_ms = (h * 3_600_000 + m * 60_000 + s * 1_000) + ms_part
        entries.append((ts_ms, int(fp_hex, 16)))
    return entries


def _build_mini_vobsub(
    sub_path: Path, idx_path: Path, all_entries: list[tuple[int, int]],
    window_entries: list[tuple[int, int]], start_ms: int, duration_ms: int,
    dest_dir: Path,
) -> tuple[Path, Path]:
    """Carve out subtitle entries for a time window into a new mini VOBSUB pair."""
    sub_data = sub_path.read_bytes()
    all_fps = sorted({fp for _, fp in all_entries})

    mini_sub = bytearray()
    new_fp: dict[int, int] = {}
    for ts, fp in sorted(window_entries, key=lambda x: x[1]):
        new_fp[fp] = len(mini_sub)
        pos_in_all = all_fps.index(fp)
        end = all_fps[pos_in_all + 1] if pos_in_all + 1 < len(all_fps) else len(sub_data)
        mini_sub.extend(sub_data[fp:end])

    # Rebase timestamps to window start; copy non-timestamp lines from original header
    header: list[str] = []
    for line in idx_path.read_text(errors="replace").splitlines():
        if line.strip().startswith("timestamp:"):
            break
        header.append(line)
    idx_lines = list(header)
    for ts, fp in sorted(window_entries):
        rebased = ts - start_ms
        h = rebased // 3_600_000
        m = (rebased % 3_600_000) // 60_000
        s = (rebased % 60_000) // 1_000
        ms = rebased % 1_000
        idx_lines.append(f"timestamp: {h:02d}:{m:02d}:{s:02d}:{ms:03d}, filepos: {new_fp[fp]:09x}")

    mini_sub_path = dest_dir / "mini.sub"
    mini_idx_path = dest_dir / "mini.idx"
    mini_sub_path.write_bytes(bytes(mini_sub))
    mini_idx_path.write_text("\n".join(idx_lines) + "\n")
    return mini_sub_path, mini_idx_path


def _ocr_vobsub_window(sub_path: Path, start_ms: int, duration_ms: int, lang: str) -> str:
    """OCR a VOBSUB (.sub + .idx) subtitle in a time window.

    Parses .idx timestamps to isolate only the subtitle entries in the window,
    writes a minimal .sub + .idx, then overlays onto a blank lavfi canvas to
    render each bitmap. Blank frames are filtered before OCR.
    """
    idx_path = sub_path.with_suffix(".idx")
    all_entries = _parse_idx_entries(idx_path)
    end_ms = start_ms + duration_ms
    window = [(ts, fp) for ts, fp in all_entries if start_ms <= ts < end_ms]
    if not window:
        return ""

    duration_s = duration_ms / 1000.0

    with tempfile.TemporaryDirectory(prefix="submatch_ocr_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        mini_sub, _ = _build_mini_vobsub(
            sub_path, idx_path, all_entries, window, start_ms, duration_ms, tmp_path
        )
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=size=720x576:rate=2:color=black",
            "-i", str(mini_sub),
            "-filter_complex", "[0:v][1:s:0]overlay=format=auto",
            "-vsync", "0",
            "-t", str(duration_s),
            str(frames_dir / "frame%04d.png"),
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

        frames = _frames_in_dir(frames_dir)
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

from __future__ import annotations
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

_ISO_639_1_TO_2: dict[str, str] = {
    "ar": "ara", "bg": "bul", "ca": "cat", "cs": "ces", "cy": "wel",
    "da": "dan", "de": "deu", "el": "ell", "en": "eng", "es": "spa",
    "et": "est", "eu": "eus", "fa": "fas", "fi": "fin", "fr": "fra",
    "gl": "glg", "he": "heb", "hi": "hin", "hr": "hrv", "hu": "hun",
    "hy": "hye", "id": "ind", "is": "isl", "it": "ita", "ja": "jpn",
    "jp": "jpn",
    "ka": "kat", "kk": "kaz", "kn": "kan", "ko": "kor", "lt": "lit",
    "lv": "lav", "mk": "mkd", "ml": "mal", "mn": "mon", "mr": "mar",
    "ms": "msa", "ne": "nep", "nl": "nld", "no": "nor", "pa": "pan",
    "pl": "pol", "pt": "por", "ro": "ron", "ru": "rus", "sk": "slk",
    "sl": "slv", "sq": "sqi", "sr": "srp", "sv": "swe", "sw": "swa",
    "ta": "tam", "te": "tel", "th": "tha", "tr": "tur", "uk": "ukr",
    "ur": "urd", "uz": "uzb", "vi": "vie", "zh": "zho",
}


def _lang_match(pref: str, track_lang: str | None) -> bool:
    """Return True if pref matches track_lang, handling ISO 639-1 <-> 639-2 equivalence."""
    if not track_lang:
        return False
    p = pref.lower()
    t = track_lang.lower()
    return _ISO_639_1_TO_2.get(p, p) == _ISO_639_1_TO_2.get(t, t)


def get_duration_ms(video_path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    duration_s = float(json.loads(result.stdout)["format"]["duration"])
    return int(duration_s * 1_000)


def get_audio_track_duration_ms(video_path: Path, audio_track: int = 0) -> int | None:
    """Return the duration of a specific audio stream in ms, or None if unavailable.

    Some video containers report a format duration longer than the audio track (e.g. when
    a broadcast recording is padded with video after audio ends). Using the audio stream's
    own duration prevents ffmpeg CalledProcessError when seeking into positions the audio
    track doesn't cover. Returns None on any error so callers fall back to format duration.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "a", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
        if audio_track >= len(streams):
            return None
        duration = streams[audio_track].get("duration")
        if duration is None:
            return None
        return int(float(duration) * 1_000)
    except Exception:
        return None


def has_audio_track(video_path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "a", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return len(json.loads(result.stdout).get("streams", [])) > 0


def list_audio_tracks(video_path: Path) -> list[dict]:
    """Return one dict per audio stream: {"index": int, "lang": str | None}."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "a", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    return [
        {"index": i, "lang": s.get("tags", {}).get("language")}
        for i, s in enumerate(streams)
    ]


def resolve_audio_track(video_path: Path, spec: str) -> tuple[int, str | None]:
    """Parse spec and return (track_index, track_lang).

    spec is either an integer string ("1") or a comma-separated language preference
    list ("jp,en,pt"). Falls back to track 0 with a warning if no preference matches.
    Exits 2 if an integer index is out of range.
    """
    try:
        tracks = list_audio_tracks(video_path)
    except Exception:
        print(
            f"Warning: could not list audio tracks for {video_path.name}, using track 0",
            file=sys.stderr,
        )
        return 0, None

    spec = spec.strip()

    # Integer path
    try:
        idx = int(spec)
        if idx < 0 or idx >= len(tracks):
            print(
                f"Error: audio track {idx} does not exist in {video_path.name} "
                f"({len(tracks)} track(s) available)",
                file=sys.stderr,
            )
            sys.exit(2)
        return idx, tracks[idx]["lang"]
    except ValueError:
        pass

    # Language preference list
    for pref in [p.strip() for p in spec.split(",") if p.strip()]:
        for track in tracks:
            if _lang_match(pref, track["lang"]):
                return track["index"], track["lang"]

    track_desc = ", ".join(f"a:{t['index']} ({t['lang'] or '?'})" for t in tracks) if tracks else "none"
    print(
        f"Warning: no audio track matches '{spec}' in {video_path.name} "
        f"(available: {track_desc}), using track 0",
        file=sys.stderr,
    )
    return 0, tracks[0]["lang"] if tracks else None


def extract_segment(video_path: Path, start_ms: int, duration_ms: int, audio_track: int = 0) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = Path(tmp.name)
    tmp.close()
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_ms / 1_000),
        "-i", str(video_path),
        "-t", str(duration_ms / 1_000),
        "-ar", "16000",
        "-ac", "1",
        "-vn",
    ]
    if audio_track > 0:
        cmd += ["-map", f"0:a:{audio_track}"]
    cmd.append(str(out_path))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
    try:
        proc.communicate()
    except KeyboardInterrupt:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
        raise
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return out_path


def detect_speech_regions(video_path: Path, audio_track: int = 0) -> list[tuple[int, int]]:
    """Return non-silent (speech) spans as (start_ms, end_ms) pairs.

    Uses ffmpeg silencedetect. Returns [] on any failure so callers can fall back.
    """
    try:
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-map", f"0:a:{audio_track}",
            "-af", "silencedetect=noise=-30dB:duration=0.5",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        stderr = result.stderr

        duration_match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", stderr)
        if not duration_match:
            return []
        h, m, s = duration_match.groups()
        total_ms = int((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)

        silence_starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", stderr)]
        silence_ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", stderr)]

        speech_regions: list[tuple[int, int]] = []
        prev_end_ms = 0
        for sil_start, sil_end in zip(silence_starts, silence_ends):
            sil_start_ms = int(sil_start * 1000)
            sil_end_ms = int(sil_end * 1000)
            if sil_start_ms > prev_end_ms:
                speech_regions.append((prev_end_ms, sil_start_ms))
            prev_end_ms = sil_end_ms
        if prev_end_ms < total_ms:
            speech_regions.append((prev_end_ms, total_ms))

        return [(s, e) for s, e in speech_regions if e > s]
    except Exception:
        return []

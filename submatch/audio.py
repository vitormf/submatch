from __future__ import annotations
import json
import os
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

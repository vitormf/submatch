from __future__ import annotations
import json
import os
import signal
import subprocess
from pathlib import Path

_IMAGE_CODECS = frozenset({"dvd_subtitle", "hdmv_pgs_subtitle", "dvbsub"})


def list_subtitle_tracks(video: Path) -> list[dict]:
    """Return one dict per subtitle stream: {"index", "global_index", "lang", "title", "codec"}.

    "index" is the subtitle-stream-relative index (for -map 0:s:N).
    "global_index" is the global stream index from ffprobe (for mkvextract tracks N:dest).
    """
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "s", str(video)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    return [
        {
            "index": i,
            "global_index": s.get("index", i),
            "lang": s.get("tags", {}).get("language"),
            "title": s.get("tags", {}).get("title"),
            "codec": s.get("codec_name"),
        }
        for i, s in enumerate(streams)
    ]


def extract_subtitle_track(video: Path, index: int, dest: Path) -> None:
    """Extract subtitle stream at stream index `index` to `dest` as SRT."""
    cmd = ["ffmpeg", "-y", "-i", str(video), "-map", f"0:s:{index}", "-c:s", "srt", str(dest)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
    try:
        proc.communicate()
    except KeyboardInterrupt:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
        raise
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _extract_image_track(video: Path, track: dict, dest_dir: Path) -> Path:
    """Extract one image-based subtitle track to its native format (.sub or .sup).

    dvd_subtitle uses mkvextract: ffmpeg maps .sub to the microdvd muxer (wrong format).
    mkvextract correctly writes the VOBSUB .sub + .idx pair.
    PGS (hdmv_pgs_subtitle) still uses ffmpeg -c:s copy → .sup.
    """
    idx = track["index"]
    global_idx = track.get("global_index", idx)
    lang = track.get("lang") or "und"
    codec = track.get("codec")

    if codec == "dvd_subtitle":
        dest = dest_dir / f"embedded_s{idx}_{lang}.sub"
        cmd = ["mkvextract", str(video), "tracks", f"{global_idx}:{dest}"]
    else:
        ext = ".sup" if codec == "hdmv_pgs_subtitle" else ".sub"
        dest = dest_dir / f"embedded_s{idx}_{lang}{ext}"
        cmd = ["ffmpeg", "-y", "-i", str(video), "-map", f"0:s:{idx}", "-c:s", "copy", str(dest)]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
    try:
        proc.communicate()
    except KeyboardInterrupt:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
        raise
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return dest


def extract_all_subtitle_tracks(
    video: Path, tracks: list[dict], dest_dir: Path
) -> dict[int, Path]:
    """Extract all subtitle tracks. Text tracks → SRT (one ffmpeg pass); image tracks → native format."""
    if not tracks:
        return {}

    dest_dir.mkdir(parents=True, exist_ok=True)

    text_tracks = [t for t in tracks if t.get("codec") not in _IMAGE_CODECS]
    image_tracks = [t for t in tracks if t.get("codec") in _IMAGE_CODECS]

    paths: dict[int, Path] = {}

    if text_tracks:
        cmd = ["ffmpeg", "-y", "-i", str(video)]
        for track in text_tracks:
            idx = track["index"]
            lang = track.get("lang") or "und"
            dest = dest_dir / f"embedded_s{idx}_{lang}.srt"
            paths[idx] = dest
            cmd += ["-map", f"0:s:{idx}", "-c:s", "srt", str(dest)]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
        try:
            proc.communicate()
        except KeyboardInterrupt:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait()
            raise
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

    for track in image_tracks:
        try:
            dest = _extract_image_track(video, track, dest_dir)
            paths[track["index"]] = dest
        except subprocess.CalledProcessError:
            pass

    return paths

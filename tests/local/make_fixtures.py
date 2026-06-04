#!/usr/bin/env python3
"""Re-encode videos from sources.json into tests/local/fixtures/ at minimal quality.

Keeps all audio and subtitle streams intact, with optional audio language retagging.
External subtitle files listed in sources.json are copied alongside the output video.
Skips files already processed according to .index.json.
Run from anywhere — paths are relative to this script.
"""
from __future__ import annotations
import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LOCAL = Path(__file__).parent
FIXTURES = LOCAL / "fixtures"
SOURCES_FILE = LOCAL / "sources.json"
INDEX = LOCAL / ".index.json"


def load_index() -> dict:
    if INDEX.exists():
        return json.loads(INDEX.read_text())
    return {}


def save_index(index: dict) -> None:
    INDEX.write_text(json.dumps(index, indent=2))


def load_sources() -> list[dict]:
    if not SOURCES_FILE.exists():
        print(f"No sources.json found at {SOURCES_FILE}", file=sys.stderr)
        sys.exit(1)
    return json.loads(SOURCES_FILE.read_text())


def reencode(
    source: Path,
    dest: Path,
    retag_audio: dict[str, str] | None = None,
    drop_subs: bool = False,
) -> None:
    """Re-encode source to dest at minimal quality.

    retag_audio: maps stream index string ("0") to language code ("kor"),
    applied via -metadata:s:a:<idx> language=<lang>.
    drop_subs: omit subtitle streams (use when embedded codec is incompatible with MKV).
    """
    sub_map = [] if drop_subs else ["-map", "0:s?"]
    sub_codec = [] if drop_subs else ["-c:s", "copy"]
    cmd = [
        "ffmpeg", "-y", "-i", str(source),
        "-map", "0:v:0?", "-map", "0:a", *sub_map,
        "-c:v", "libx264", "-crf", "51", "-preset", "ultrafast",
        "-vf", "scale=320:-2",
        "-c:a", "copy",
        *sub_codec,
    ]
    for stream_idx, lang in (retag_audio or {}).items():
        cmd += [f"-metadata:s:a:{stream_idx}", f"language={lang}"]
    cmd.append(str(dest))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        _, stderr = proc.communicate()
    except KeyboardInterrupt:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
        if dest.exists():
            dest.unlink()
        raise
    if proc.returncode != 0:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(stderr.decode(errors="replace").strip().splitlines()[-1])


def copy_external_subs(external_subs: list[str], out_stem: str) -> None:
    """Copy external subtitle files, renaming them to match the output video stem."""
    for sub_path_str in external_subs:
        sub = Path(sub_path_str)
        if not sub.exists():
            print(f"  warn: external sub not found: {sub}", file=sys.stderr)
            continue
        # Preserve everything after the first dot in the subtitle filename as the suffix.
        # e.g. "Movie.en.srt" → suffixes ".en.srt" → out_stem + ".en.srt"
        sub_suffix = "".join(sub.suffixes)
        dest_sub = FIXTURES / f"{out_stem}{sub_suffix}"
        if not dest_sub.exists():
            shutil.copy2(sub, dest_sub)
            print(f"  copied sub: {dest_sub.name}")


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    index = load_index()

    for entry in sources:
        src_path = Path(entry["src"])
        out_name = entry["out"]
        retag_audio = entry.get("retag_audio")
        drop_subs = entry.get("drop_subs", False)
        external_subs = entry.get("external_subs", [])
        dest = FIXTURES / out_name
        out_stem = Path(out_name).stem

        if out_name in index and dest.exists():
            print(f"skip  {out_name} — already encoded")
            copy_external_subs(external_subs, out_stem)
            continue

        if not src_path.exists():
            print(f"SKIP  {out_name} — source not found: {src_path}", file=sys.stderr)
            continue

        print(f"encode {src_path.name} → {out_name} ...", end=" ", flush=True)
        try:
            reencode(src_path, dest, retag_audio, drop_subs)
        except RuntimeError as exc:
            print(f"FAILED\n  {exc}", file=sys.stderr)
            continue

        src_mb = round(src_path.stat().st_size / 1_048_576, 1)
        out_mb = round(dest.stat().st_size / 1_048_576, 1)
        index[out_name] = {
            "src": str(src_path),
            "encoded_at": datetime.now(timezone.utc).isoformat(),
            "source_mb": src_mb,
            "output_mb": out_mb,
        }
        save_index(index)
        ratio = src_mb / max(out_mb, 0.1)
        print(f"done  ({src_mb} MB → {out_mb} MB, {ratio:.0f}x smaller)")
        copy_external_subs(external_subs, out_stem)


if __name__ == "__main__":
    main()

from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "submatch"
_DEFAULT_TTL_DAYS = 30
_DEFAULT_MAX_MB = 200


@dataclass
class VideoCache:
    segment_starts: list[int]
    transcriptions: list[str]
    audio_lang: str | None
    audio_track_index: int = 0
    audio_track_lang: str | None = None
    segment_langs: list[str | None] = field(default_factory=list)


def _cache_key(video: Path, mtime: float, model: str, n_segments: int, audio_track_index: int) -> str:
    raw = f"{video.resolve()}|{mtime}|{model}|{n_segments}|{audio_track_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load(
    video: Path,
    mtime: float,
    model: str,
    n_segments: int,
    audio_track_index: int,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> VideoCache | None:
    key = _cache_key(video, mtime, model, n_segments, audio_track_index)
    path = cache_dir / f"{key}.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vc = VideoCache(
            segment_starts=[s["start_ms"] for s in data["segments"]],
            transcriptions=[s["text"] for s in data["segments"]],
            audio_lang=data.get("audio_lang"),
            audio_track_index=data.get("audio_track_index", 0),
            audio_track_lang=data.get("audio_track_lang"),
            segment_langs=[s.get("lang") for s in data["segments"]],
        )
    except Exception:
        return None

    # Write-back last_used in separate try/except: don't discard valid cache hit on write failure
    data["last_used"] = time.time()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass  # best-effort; stale last_used is acceptable

    return vc


def store(
    video: Path,
    mtime: float,
    model: str,
    n_segments: int,
    audio_track_index: int,
    vc: VideoCache,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    ttl_days: int = _DEFAULT_TTL_DAYS,
    max_mb: int = _DEFAULT_MAX_MB,
) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(video, mtime, model, n_segments, audio_track_index)
        path = cache_dir / f"{key}.json"
        now = time.time()
        data = {
            "video_path": str(video.resolve()),
            "mtime": mtime,
            "model": model,
            "n_segments": n_segments,
            "audio_track_index": audio_track_index,
            "audio_lang": vc.audio_lang,
            "audio_track_lang": vc.audio_track_lang,
            "created_at": now,
            "last_used": now,
            "segments": [
                {"start_ms": s, "text": t, "lang": lang}
                for s, t, lang in zip(
                    vc.segment_starts,
                    vc.transcriptions,
                    (
                        list(vc.segment_langs) + [None] * (len(vc.segment_starts) - len(vc.segment_langs))
                        if vc.segment_langs
                        else [None] * len(vc.segment_starts)
                    ),
                )
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        _evict(cache_dir, ttl_days, max_mb)
    except Exception:
        pass


def clear(cache_dir: Path = _DEFAULT_CACHE_DIR) -> int:
    count = 0
    try:
        for p in cache_dir.glob("*.json"):
            p.unlink(missing_ok=True)
            count += 1
    except Exception:
        pass
    return count


def _evict(cache_dir: Path, ttl_days: int, max_mb: int) -> None:
    try:
        now = time.time()
        ttl_seconds = ttl_days * 86400
        surviving: list[tuple[float, Path]] = []

        for p in cache_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if now - data.get("created_at", 0) > ttl_seconds:
                    p.unlink(missing_ok=True)
                else:
                    surviving.append((data.get("last_used", 0), p))
            except Exception:
                p.unlink(missing_ok=True)

        max_bytes = max_mb * 1024 * 1024
        surviving.sort()  # oldest last_used first
        while len(surviving) > 1:
            total = sum(p.stat().st_size for _, p in surviving if p.exists())
            if total <= max_bytes:
                break
            _, oldest = surviving.pop(0)
            oldest.unlink(missing_ok=True)
    except Exception:
        pass

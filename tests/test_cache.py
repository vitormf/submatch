from __future__ import annotations
import json
import time
from pathlib import Path
from unittest.mock import patch
from submatch.cache import VideoCache, load, store, clear, _evict


def _make_vc(starts=(10000, 20000), texts=("hello world foo bar baz", "she left before"), lang="en"):
    return VideoCache(
        segment_starts=list(starts),
        transcriptions=list(texts),
        audio_lang=lang,
        audio_track_index=0,
        audio_track_lang=None,
    )


def test_store_and_load_round_trip(tmp_path):
    video = Path("/fake/movie.mkv")
    vc = _make_vc()
    store(video, mtime=1000.0, model="base", n_segments=2, audio_track_index=0,
          vc=vc, cache_dir=tmp_path, ttl_days=30, max_mb=200)

    result = load(video, mtime=1000.0, model="base", n_segments=2,
                  audio_track_index=0, cache_dir=tmp_path)

    assert result is not None
    assert result.segment_starts == [10000, 20000]
    assert result.transcriptions == ["hello world foo bar baz", "she left before"]
    assert result.audio_lang == "en"
    assert result.audio_track_index == 0


def test_load_returns_none_on_cache_miss(tmp_path):
    result = load(Path("/fake/movie.mkv"), mtime=1000.0, model="base",
                  n_segments=5, audio_track_index=0, cache_dir=tmp_path)
    assert result is None


def test_load_returns_none_for_different_mtime(tmp_path):
    video = Path("/fake/movie.mkv")
    vc = _make_vc()
    store(video, mtime=1000.0, model="base", n_segments=2, audio_track_index=0,
          vc=vc, cache_dir=tmp_path, ttl_days=30, max_mb=200)
    result = load(video, mtime=9999.0, model="base", n_segments=2,
                  audio_track_index=0, cache_dir=tmp_path)
    assert result is None


def test_load_returns_none_for_different_model(tmp_path):
    video = Path("/fake/movie.mkv")
    vc = _make_vc()
    store(video, mtime=1000.0, model="base", n_segments=2, audio_track_index=0,
          vc=vc, cache_dir=tmp_path, ttl_days=30, max_mb=200)
    result = load(video, mtime=1000.0, model="small", n_segments=2,
                  audio_track_index=0, cache_dir=tmp_path)
    assert result is None


def test_load_returns_none_on_corrupt_file(tmp_path):
    video = Path("/fake/movie.mkv")
    vc = _make_vc()
    store(video, mtime=1000.0, model="base", n_segments=2, audio_track_index=0,
          vc=vc, cache_dir=tmp_path, ttl_days=30, max_mb=200)
    for f in tmp_path.glob("*.json"):
        f.write_text("not valid json")
    result = load(video, mtime=1000.0, model="base", n_segments=2,
                  audio_track_index=0, cache_dir=tmp_path)
    assert result is None


def test_store_does_not_raise_on_unwritable_dir(tmp_path):
    unwritable = tmp_path / "nope"
    unwritable.mkdir()
    unwritable.chmod(0o444)
    vc = _make_vc()
    store(Path("/fake/movie.mkv"), mtime=1000.0, model="base", n_segments=2,
          audio_track_index=0, vc=vc, cache_dir=unwritable / "sub",
          ttl_days=30, max_mb=200)
    unwritable.chmod(0o755)


def test_clear_removes_all_entries(tmp_path):
    video = Path("/fake/movie.mkv")
    for i in range(3):
        vc = _make_vc(starts=(i * 1000,), texts=(f"text {i}",))
        store(video, mtime=float(i), model="base", n_segments=1, audio_track_index=0,
              vc=vc, cache_dir=tmp_path, ttl_days=30, max_mb=200)
    count = clear(tmp_path)
    assert count == 3
    assert list(tmp_path.glob("*.json")) == []


def test_evict_removes_entries_older_than_ttl(tmp_path):
    video = Path("/fake/movie.mkv")
    vc = _make_vc()
    store(video, mtime=1.0, model="base", n_segments=2, audio_track_index=0,
          vc=vc, cache_dir=tmp_path, ttl_days=30, max_mb=200)

    for p in tmp_path.glob("*.json"):
        data = json.loads(p.read_text())
        data["created_at"] = time.time() - 31 * 86400
        data["last_used"] = time.time() - 31 * 86400
        p.write_text(json.dumps(data))

    store(Path("/fake/movie2.mkv"), mtime=2.0, model="base", n_segments=2,
          audio_track_index=0, vc=vc, cache_dir=tmp_path, ttl_days=30, max_mb=200)

    remaining = list(tmp_path.glob("*.json"))
    assert len(remaining) == 1


def test_evict_lru_when_over_max_size(tmp_path):
    v1, v2 = Path("/fake/a.mkv"), Path("/fake/b.mkv")
    vc = _make_vc()

    store(v1, mtime=1.0, model="base", n_segments=2, audio_track_index=0,
          vc=vc, cache_dir=tmp_path, ttl_days=30, max_mb=200)
    store(v2, mtime=2.0, model="base", n_segments=2, audio_track_index=0,
          vc=vc, cache_dir=tmp_path, ttl_days=30, max_mb=200)

    for p in tmp_path.glob("*.json"):
        data = json.loads(p.read_text())
        if data["video_path"].endswith("a.mkv"):
            data["last_used"] = time.time() - 1000
            p.write_text(json.dumps(data))

    _evict(tmp_path, ttl_days=30, max_mb=0)

    remaining = list(tmp_path.glob("*.json"))
    assert len(remaining) == 1
    data = json.loads(remaining[0].read_text())
    assert data["video_path"].endswith("b.mkv")


def test_load_updates_last_used_on_disk(tmp_path):
    video = Path("/fake/movie.mkv")
    vc = _make_vc()
    store(video, mtime=1000.0, model="base", n_segments=2, audio_track_index=0,
          vc=vc, cache_dir=tmp_path, ttl_days=30, max_mb=200)

    files = list(tmp_path.glob("*.json"))
    original_last_used = json.loads(files[0].read_text())["last_used"]

    future_time = original_last_used + 100.0
    with patch("submatch.cache.time.time", return_value=future_time):
        result = load(video, mtime=1000.0, model="base", n_segments=2,
                      audio_track_index=0, cache_dir=tmp_path)

    assert result is not None
    updated_last_used = json.loads(files[0].read_text())["last_used"]
    assert updated_last_used == future_time

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


def test_load_writeback_failure_still_returns_cache(tmp_path):
    # Covers lines 55-56: write-back of last_used fails; valid cache is still returned
    video = Path("/fake/movie.mkv")
    vc = _make_vc()
    store(video, mtime=1000.0, model="base", n_segments=2, audio_track_index=0,
          vc=vc, cache_dir=tmp_path, ttl_days=30, max_mb=200)

    files = list(tmp_path.glob("*.json"))
    files[0].chmod(0o444)  # read-only — write-back will raise PermissionError
    try:
        result = load(video, mtime=1000.0, model="base", n_segments=2,
                      audio_track_index=0, cache_dir=tmp_path)
        assert result is not None
        assert result.transcriptions == list(vc.transcriptions)
    finally:
        files[0].chmod(0o644)


def test_clear_exception_is_swallowed():
    # Covers lines 105-106: exception in clear() is swallowed and returns 0
    from unittest.mock import MagicMock
    mock_dir = MagicMock()
    mock_dir.glob.side_effect = OSError("permission denied")
    result = clear(mock_dir)
    assert result == 0


def test_evict_handles_corrupt_json_file(tmp_path):
    # Covers lines 124-125: corrupt JSON in _evict triggers per-file exception handler
    (tmp_path / "corrupt.json").write_text("not valid json {{")
    video = Path("/fake/movie.mkv")
    vc = _make_vc()
    store(video, mtime=1000.0, model="base", n_segments=2, audio_track_index=0,
          vc=vc, cache_dir=tmp_path, ttl_days=30, max_mb=200)
    assert not (tmp_path / "corrupt.json").exists()


def test_evict_outer_exception_is_swallowed():
    # Covers lines 135-136: outer exception in _evict is swallowed (does not propagate)
    from unittest.mock import MagicMock
    mock_dir = MagicMock()
    mock_dir.glob.side_effect = OSError("permission denied")
    _evict(mock_dir, ttl_days=30, max_mb=200)  # must not raise


def test_store_and_load_preserves_segment_langs(tmp_path):
    video = Path("/fake/movie.mkv")
    vc = VideoCache(
        segment_starts=[10000, 20000],
        transcriptions=["hello world", "she left"],
        audio_lang="ko",
        audio_track_index=0,
        audio_track_lang=None,
        segment_langs=["ko", None],
    )
    store(video, mtime=1000.0, model="base", n_segments=2, audio_track_index=0,
          vc=vc, cache_dir=tmp_path, ttl_days=30, max_mb=200)
    result = load(video, mtime=1000.0, model="base", n_segments=2,
                  audio_track_index=0, cache_dir=tmp_path)
    assert result is not None
    assert result.segment_langs == ["ko", None]


def test_store_with_empty_segment_langs_writes_all_segments(tmp_path):
    """segment_langs=[] should not truncate the stored segments."""
    video = Path("/fake/movie.mkv")
    vc = VideoCache(
        segment_starts=[10000, 20000, 30000],
        transcriptions=["a", "b", "c"],
        audio_lang="en",
        audio_track_index=0,
        audio_track_lang=None,
        segment_langs=[],
    )
    store(video, mtime=1000.0, model="base", n_segments=3, audio_track_index=0,
          vc=vc, cache_dir=tmp_path, ttl_days=30, max_mb=200)
    result = load(video, mtime=1000.0, model="base", n_segments=3,
                  audio_track_index=0, cache_dir=tmp_path)
    assert result is not None
    assert len(result.segment_starts) == 3
    assert result.segment_langs == [None, None, None]


def test_load_old_cache_missing_segment_langs(tmp_path):
    """Old cache files without 'lang' in segments deserialise gracefully."""
    video = Path("/fake/movie.mkv")
    import hashlib
    import json
    import time
    raw = f"{video.resolve()}|1000.0|base|2|0"
    key = hashlib.sha256(raw.encode()).hexdigest()[:16]
    path = tmp_path / f"{key}.json"
    data = {
        "video_path": str(video.resolve()),
        "mtime": 1000.0, "model": "base", "n_segments": 2,
        "audio_track_index": 0, "audio_lang": "en", "audio_track_lang": None,
        "created_at": time.time(), "last_used": time.time(),
        "segments": [{"start_ms": 10000, "text": "hello"}, {"start_ms": 20000, "text": "world"}],
    }
    path.write_text(json.dumps(data))
    result = load(video, mtime=1000.0, model="base", n_segments=2,
                  audio_track_index=0, cache_dir=tmp_path)
    assert result is not None
    assert result.segment_langs == [None, None]

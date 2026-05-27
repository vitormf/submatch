from pathlib import Path
import pytest
from submatch.batch import find_pairs, find_subtitle_candidates, find_pairs_recursive, find_subtitle_candidates_recursive


def test_find_pairs_by_stem(tmp_path):
    (tmp_path / "movie.mkv").touch()
    (tmp_path / "movie.srt").touch()
    assert find_pairs(tmp_path) == [(tmp_path / "movie.mkv", tmp_path / "movie.srt")]


def test_find_pairs_multiple_subtitles(tmp_path):
    (tmp_path / "movie.mkv").touch()
    (tmp_path / "movie.srt").touch()
    (tmp_path / "movie.en.srt").touch()
    pairs = find_pairs(tmp_path)
    subs = {p[1].name for p in pairs}
    assert subs == {"movie.srt", "movie.en.srt"}


def test_find_pairs_ignores_unmatched_subtitle(tmp_path):
    (tmp_path / "movie.mkv").touch()
    (tmp_path / "other.srt").touch()
    assert find_pairs(tmp_path) == []


def test_find_pairs_ignores_non_video_files(tmp_path):
    (tmp_path / "movie.mkv").touch()
    (tmp_path / "movie.srt").touch()
    (tmp_path / "movie.nfo").touch()
    (tmp_path / "readme.txt").touch()
    assert len(find_pairs(tmp_path)) == 1


def test_find_pairs_sorted(tmp_path):
    (tmp_path / "b.mkv").touch()
    (tmp_path / "b.srt").touch()
    (tmp_path / "a.mkv").touch()
    (tmp_path / "a.srt").touch()
    pairs = find_pairs(tmp_path)
    assert pairs[0][0].name == "a.mkv"
    assert pairs[1][0].name == "b.mkv"


def test_find_pairs_all_video_extensions(tmp_path):
    (tmp_path / "show.mp4").touch()
    (tmp_path / "show.srt").touch()
    pairs = find_pairs(tmp_path)
    assert any(p[0].name == "show.mp4" for p in pairs)


def test_find_pairs_all_subtitle_extensions(tmp_path):
    (tmp_path / "ep.mkv").touch()
    for ext in (".srt", ".vtt", ".ass", ".ssa", ".sub"):
        (tmp_path / f"ep{ext}").touch()
    pairs = find_pairs(tmp_path)
    assert len(pairs) == 5


def test_find_subtitle_candidates(tmp_path):
    (tmp_path / "a.srt").touch()
    (tmp_path / "b.vtt").touch()
    (tmp_path / "c.ass").touch()
    result = find_subtitle_candidates(tmp_path)
    assert sorted(p.name for p in result) == ["a.srt", "b.vtt", "c.ass"]


def test_find_subtitle_candidates_filters_non_subtitles(tmp_path):
    (tmp_path / "a.srt").touch()
    (tmp_path / "readme.txt").touch()
    (tmp_path / "video.mkv").touch()
    result = find_subtitle_candidates(tmp_path)
    assert len(result) == 1
    assert result[0].name == "a.srt"


def test_find_subtitle_candidates_sorted(tmp_path):
    (tmp_path / "z.srt").touch()
    (tmp_path / "a.srt").touch()
    result = find_subtitle_candidates(tmp_path)
    assert result[0].name == "a.srt"


def test_find_pairs_doesnt_cross_episode_match(tmp_path):
    (tmp_path / "show.mkv").touch()
    (tmp_path / "show.season1.mkv").touch()
    (tmp_path / "show.srt").touch()
    (tmp_path / "show.season1.srt").touch()
    pairs = find_pairs(tmp_path)
    pair_map = {p[0].name: p[1].name for p in pairs}
    assert pair_map["show.mkv"] == "show.srt"
    assert pair_map["show.season1.mkv"] == "show.season1.srt"


def test_find_pairs_recursive_nested(tmp_path):
    show = tmp_path / "Bluey" / "Season1"
    show.mkdir(parents=True)
    (show / "ep01.mkv").touch()
    (show / "ep01.srt").touch()
    (show / "ep02.mkv").touch()
    (show / "ep02.srt").touch()
    pairs = find_pairs_recursive(tmp_path)
    assert len(pairs) == 2
    assert any(p[0].name == "ep01.mkv" for p in pairs)
    assert any(p[0].name == "ep02.mkv" for p in pairs)
    for video, sub in pairs:
        assert video.parent == show
        assert sub.parent == show


def test_find_pairs_recursive_per_directory_isolation(tmp_path):
    s01 = tmp_path / "Season1"
    s02 = tmp_path / "Season2"
    s01.mkdir()
    s02.mkdir()
    (s01 / "ep01.mkv").touch()
    (s01 / "ep01.srt").touch()
    (s02 / "ep01.mkv").touch()
    (s02 / "ep01.srt").touch()
    pairs = find_pairs_recursive(tmp_path)
    assert len(pairs) == 2
    for video, sub in pairs:
        assert video.parent == sub.parent


def test_find_pairs_recursive_flat_dir(tmp_path):
    (tmp_path / "movie.mkv").touch()
    (tmp_path / "movie.srt").touch()
    assert find_pairs_recursive(tmp_path) == find_pairs(tmp_path)


def test_find_pairs_recursive_empty(tmp_path):
    assert find_pairs_recursive(tmp_path) == []


def test_find_pairs_recursive_mixed_levels(tmp_path):
    # Files at root level
    (tmp_path / "movie.mkv").touch()
    (tmp_path / "movie.srt").touch()
    # Files in subdirectory
    nested = tmp_path / "TV Show" / "Season1"
    nested.mkdir(parents=True)
    (nested / "ep01.mkv").touch()
    (nested / "ep01.srt").touch()
    pairs = find_pairs_recursive(tmp_path)
    assert len(pairs) == 2
    names = {p[0].name for p in pairs}
    assert names == {"movie.mkv", "ep01.mkv"}


def test_find_subtitle_candidates_recursive(tmp_path):
    (tmp_path / "en").mkdir()
    (tmp_path / "pt").mkdir()
    (tmp_path / "en" / "movie.srt").touch()
    (tmp_path / "pt" / "movie.vtt").touch()
    result = find_subtitle_candidates_recursive(tmp_path)
    assert {p.name for p in result} == {"movie.srt", "movie.vtt"}


def test_find_subtitle_candidates_recursive_filters_non_subtitles(tmp_path):
    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "movie.srt").touch()
    (sub / "readme.txt").touch()
    result = find_subtitle_candidates_recursive(tmp_path)
    assert len(result) == 1
    assert result[0].name == "movie.srt"

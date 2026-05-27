from pathlib import Path
import pytest
from submatch.batch import find_pairs, find_subtitle_candidates


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

from pathlib import Path
from unittest.mock import patch, MagicMock
import submatch.batch as batch
from submatch.batch import (
    find_pairs, find_subtitle_candidates,
    find_pairs_recursive, find_subtitle_candidates_recursive,
    filter_pairs, _extract_lang_tag, _lang_matches, classify_inputs,
)


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


# ── _extract_lang_tag ────────────────────────────────────────────────────────

def test_extract_lang_tag_simple():
    assert _extract_lang_tag(Path("movie.en.srt")) == "en"


def test_extract_lang_tag_region():
    assert _extract_lang_tag(Path("movie.pt-BR.srt")) == "pt-BR"


def test_extract_lang_tag_no_lang():
    assert _extract_lang_tag(Path("movie.srt")) is None


def test_extract_lang_tag_release_group_number():
    assert _extract_lang_tag(Path("The.Year.Without.1974.srt")) is None


def test_extract_lang_tag_long_stem():
    assert _extract_lang_tag(
        Path("Frosty the Snowman (1969) (1080p BluRay).en.srt")
    ) == "en"


# ── _lang_matches ────────────────────────────────────────────────────────────

def test_lang_matches_exact(tmp_path):
    f = tmp_path / "movie.en.srt"
    f.touch()
    assert _lang_matches(f, ["en"]) is True
    assert _lang_matches(f, ["pt"]) is False


def test_lang_matches_prefix(tmp_path):
    f = tmp_path / "movie.pt-BR.srt"
    f.touch()
    assert _lang_matches(f, ["pt"]) is True
    assert _lang_matches(f, ["pt-BR"]) is True
    assert _lang_matches(f, ["en"]) is False


def test_lang_matches_no_tag_infers(tmp_path):
    f = tmp_path / "movie.srt"
    f.touch()
    with patch("submatch.subtitle.parse",
               return_value=[MagicMock(text="Hello world")]), \
         patch("submatch.language.detect_from_text", return_value="en"):
        assert _lang_matches(f, ["en"]) is True
        assert _lang_matches(f, ["pt"]) is False


def test_lang_matches_no_tag_include_on_failure(tmp_path):
    f = tmp_path / "movie.srt"
    f.touch()
    with patch("submatch.subtitle.parse", side_effect=Exception("parse error")):
        assert _lang_matches(f, ["en"]) is True


def test_lang_matches_no_tag_empty_detected(tmp_path):
    """When detect_from_text returns empty string (detection failed), include the file."""
    f = tmp_path / "movie.srt"
    f.touch()
    with patch("submatch.subtitle.parse",
               return_value=[MagicMock(text="Hello world")]), \
         patch("submatch.language.detect_from_text", return_value=""):
        assert _lang_matches(f, ["en"]) is True


# ── filter_pairs ─────────────────────────────────────────────────────────────

def test_filter_pairs_no_filters(tmp_path):
    v = tmp_path / "movie.mkv"
    s = tmp_path / "movie.srt"
    v.touch()
    s.touch()
    pairs = [(v, s)]
    assert filter_pairs(pairs) == pairs


def test_filter_pairs_sub_lang_keeps_match(tmp_path):
    v = tmp_path / "movie.mkv"
    en = tmp_path / "movie.en.srt"
    pt = tmp_path / "movie.pt-BR.srt"
    v.touch()
    en.touch()
    pt.touch()
    result = filter_pairs([(v, en), (v, pt)], sub_langs=["en"])
    assert result == [(v, en)]


def test_filter_pairs_sub_lang_prefix(tmp_path):
    v = tmp_path / "movie.mkv"
    en = tmp_path / "movie.en.srt"
    pt_br = tmp_path / "movie.pt-BR.srt"
    pt_pt = tmp_path / "movie.pt-PT.srt"
    v.touch()
    en.touch()
    pt_br.touch()
    pt_pt.touch()
    result = filter_pairs([(v, en), (v, pt_br), (v, pt_pt)], sub_langs=["pt"])
    assert {s.name for _, s in result} == {"movie.pt-BR.srt", "movie.pt-PT.srt"}


def test_filter_pairs_sub_lang_multiple_codes(tmp_path):
    v = tmp_path / "movie.mkv"
    en = tmp_path / "movie.en.srt"
    pt = tmp_path / "movie.pt.srt"
    de = tmp_path / "movie.de.srt"
    v.touch()
    en.touch()
    pt.touch()
    de.touch()
    result = filter_pairs([(v, en), (v, pt), (v, de)], sub_langs=["en", "pt"])
    assert {s.name for _, s in result} == {"movie.en.srt", "movie.pt.srt"}


def test_filter_pairs_glob(tmp_path):
    v = tmp_path / "movie.mkv"
    en = tmp_path / "movie.en.srt"
    pt = tmp_path / "movie.pt-BR.srt"
    v.touch()
    en.touch()
    pt.touch()
    result = filter_pairs([(v, en), (v, pt)], glob_pattern="*.en.*")
    assert result == [(v, en)]


def test_filter_pairs_combined(tmp_path):
    v = tmp_path / "movie.mkv"
    en_srt = tmp_path / "movie.en.srt"
    en_vtt = tmp_path / "movie.en.vtt"
    pt = tmp_path / "movie.pt.srt"
    v.touch()
    en_srt.touch()
    en_vtt.touch()
    pt.touch()
    result = filter_pairs(
        [(v, en_srt), (v, en_vtt), (v, pt)],
        sub_langs=["en"],
        glob_pattern="*.srt",
    )
    assert result == [(v, en_srt)]


# ── classify_inputs ───────────────────────────────────────────────────────────

def test_classify_inputs_single_video(tmp_path):
    v = tmp_path / "movie.mkv"
    v.touch()
    videos, subs = classify_inputs([v])
    assert videos == [v]
    assert subs == []


def test_classify_inputs_single_subtitle(tmp_path):
    s = tmp_path / "movie.en.srt"
    s.touch()
    videos, subs = classify_inputs([s])
    assert videos == []
    assert subs == [s]


def test_classify_inputs_mixed_explicit_files(tmp_path):
    v = tmp_path / "movie.mkv"
    v.touch()
    s = tmp_path / "movie.en.srt"
    s.touch()
    videos, subs = classify_inputs([v, s])
    assert videos == [v]
    assert subs == [s]


def test_classify_inputs_directory_expands_flat(tmp_path):
    v = tmp_path / "movie.mkv"
    v.touch()
    s = tmp_path / "movie.en.srt"
    s.touch()
    videos, subs = classify_inputs([tmp_path], recursive=False)
    assert videos == [v]
    assert subs == [s]


def test_classify_inputs_recursive_default_finds_nested(tmp_path):
    nested = tmp_path / "sub"
    nested.mkdir()
    s = nested / "movie.en.srt"
    s.touch()
    v = tmp_path / "movie.mkv"
    v.touch()
    videos, subs = classify_inputs([tmp_path], recursive=True)
    assert subs == [s]
    assert videos == [v]


def test_classify_inputs_no_recursive_ignores_nested(tmp_path):
    nested = tmp_path / "sub"
    nested.mkdir()
    s = nested / "movie.en.srt"
    s.touch()
    v = tmp_path / "movie.mkv"
    v.touch()
    videos, subs = classify_inputs([tmp_path], recursive=False)
    assert s not in subs
    assert videos == [v]


def test_classify_inputs_all_subtitle_extensions(tmp_path):
    for ext in (".srt", ".vtt", ".ass", ".ssa", ".sub"):
        f = tmp_path / f"movie{ext}"
        f.touch()
    videos, subs = classify_inputs([tmp_path], recursive=False)
    assert len(subs) == 5
    assert videos == []


# ── resolve_pairs ─────────────────────────────────────────────────────────────

def test_resolve_pairs_video_only_auto_discovers_subtitle(tmp_path):
    v = tmp_path / "movie.mkv"
    v.touch()
    s = tmp_path / "movie.en.srt"
    s.touch()
    pairs = batch.resolve_pairs([v], [])
    assert pairs == [(v, s)]


def test_resolve_pairs_subtitle_only_finds_video(tmp_path):
    v = tmp_path / "movie.mkv"
    v.touch()
    s = tmp_path / "movie.en.srt"
    s.touch()
    pairs = batch.resolve_pairs([], [s])
    assert pairs == [(v, s)]


def test_resolve_pairs_explicit_pair(tmp_path):
    v = tmp_path / "movie.mkv"
    v.touch()
    s = tmp_path / "movie.en.srt"
    s.touch()
    pairs = batch.resolve_pairs([v], [s])
    assert pairs == [(v, s)]


def test_resolve_pairs_video_with_multiple_explicit_subs(tmp_path):
    v = tmp_path / "movie.mkv"
    v.touch()
    s1 = tmp_path / "movie.en.srt"
    s1.touch()
    s2 = tmp_path / "movie.pt.srt"
    s2.touch()
    pairs = batch.resolve_pairs([v], [s1, s2])
    assert set(pairs) == {(v, s1), (v, s2)}


def test_resolve_pairs_unmatched_subtitle_warns_and_skips(tmp_path, capsys):
    v = tmp_path / "movie.mkv"
    v.touch()
    s = tmp_path / "other.en.srt"
    s.touch()
    pairs = batch.resolve_pairs([v], [s])
    assert pairs == []
    err = capsys.readouterr().err
    assert "Warning" in err
    assert "other.en.srt" in err


def test_resolve_pairs_video_no_subs_warns_and_skips(tmp_path, capsys):
    v = tmp_path / "movie.mkv"
    v.touch()
    pairs = batch.resolve_pairs([v], [])
    assert pairs == []
    err = capsys.readouterr().err
    assert "Warning" in err
    assert "movie.mkv" in err


def test_resolve_pairs_multiple_videos_explicit_and_auto(tmp_path):
    """v1 has an explicit subtitle; v2 has none so auto-discovers from disk."""
    v1 = tmp_path / "v1.mkv"
    v1.touch()
    v2 = tmp_path / "v2.mkv"
    v2.touch()
    s1 = tmp_path / "v1.en.srt"
    s1.touch()
    s2 = tmp_path / "v2.en.srt"
    s2.touch()
    pairs = batch.resolve_pairs([v1, v2], [s1])
    assert (v1, s1) in pairs
    assert (v2, s2) in pairs


def test_resolve_pairs_subtitle_only_no_video_warns(tmp_path, capsys):
    s = tmp_path / "orphan.en.srt"
    s.touch()
    pairs = batch.resolve_pairs([], [s])
    assert pairs == []
    err = capsys.readouterr().err
    assert "Warning" in err
    assert "orphan.en.srt" in err


def test_resolve_pairs_video_auto_discovers_multiple_subs(tmp_path):
    """A video with no explicit subs auto-discovers multiple subtitles from disk."""
    v = tmp_path / "movie.mkv"
    v.touch()
    s_en = tmp_path / "movie.en.srt"
    s_en.touch()
    s_pt = tmp_path / "movie.pt.srt"
    s_pt.touch()
    pairs = batch.resolve_pairs([v], [])
    assert set(pairs) == {(v, s_en), (v, s_pt)}


def test_resolve_pairs_warn_missing_false_suppresses_warning(tmp_path, capsys):
    v = tmp_path / "movie.mkv"
    v.touch()
    pairs = batch.resolve_pairs([v], [], warn_missing=False)
    assert pairs == []
    assert capsys.readouterr().err == ""


def test_classify_inputs_directory_skips_unknown_extensions(tmp_path):
    v = tmp_path / "movie.mkv"
    v.touch()
    s = tmp_path / "movie.en.srt"
    s.touch()
    (tmp_path / ".DS_Store").touch()
    (tmp_path / "movie.nfo").touch()
    videos, subs = classify_inputs([tmp_path], recursive=False)
    assert videos == [v]
    assert subs == [s]


def test_classify_inputs_explicit_file_trusts_user(tmp_path):
    """Explicitly passed non-subtitle file is treated as video regardless of extension."""
    f = tmp_path / "movie.nfo"
    f.touch()
    videos, subs = classify_inputs([f])
    assert videos == [f]
    assert subs == []

from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from submatch.compare import SegmentScore
from submatch.language import LanguageResult
from submatch.types import BatchPairResult, MatchResult, MatchState
from submatch.pipeline import PipelineConfig, run, run_batch
from submatch.sampler import Segment
from submatch.subtitle import Subtitle
from submatch.transcribe import TranscriptionResult

VIDEO = Path("movie.mkv")
SUB = Path("movie.en.srt")
VIDEO2 = Path("other.mkv")
SUB2 = Path("other.en.srt")


@pytest.fixture(autouse=True)
def reset_model_caches():
    from submatch import pipeline, scoring
    for store in (pipeline._model_local, scoring._embed_local):
        if hasattr(store, "model"):
            del store.model
    yield
    for store in (pipeline._model_local, scoring._embed_local):
        if hasattr(store, "model"):
            del store.model


def _seg():
    return Segment(start_ms=60_000, end_ms=90_000, subtitle_text="Hello world", word_count=2)


def _lang_result():
    return LanguageResult(
        audio="en", subtitle_detected="en", subtitle_filename="en",
        video_metadata=None, expected=None, mismatch=False,
    )


def _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare):
    mock_sub.parse.return_value = [Subtitle(index=1, start_ms=1000, end_ms=5000, text="Hello world")]
    mock_lang.detect_from_filename.return_value = "en"
    mock_lang.detect_from_text.return_value = "en"
    mock_lang.detect_from_video.return_value = None
    mock_lang.build_result.return_value = _lang_result()
    mock_audio.get_duration_ms.return_value = 1_800_000
    mock_audio.extract_segment.return_value = MagicMock()
    mock_sampler.auto_segment_count.return_value = 5
    mock_sampler.select_segments.return_value = [_seg()]
    mock_transcribe.load_model.return_value = MagicMock()
    mock_transcribe.transcribe_segment.return_value = TranscriptionResult(
        text="Hello world", language="en", no_speech_prob=0.1, avg_logprob=0.5
    )
    mock_compare.token_f1.return_value = SegmentScore(f1=0.9, wer=0.1, subtitle_tokens=2)
    mock_compare.aggregate.return_value = 0.9
    mock_compare.SegmentScore = SegmentScore


# ---- PipelineConfig ----

def test_pipeline_config_defaults():
    config = PipelineConfig()
    assert config.model == "base"
    assert config.threshold == 0.35
    assert config.cross_threshold is None
    assert config.segments is None
    assert config.language is None
    assert config.sync is True
    assert config.drift_threshold == 2.0
    assert config.device == "auto"
    assert config.audio_track is None
    assert config.workers is None
    assert config.use_cache is True
    assert config.cache_dir is None
    assert config.cache_ttl_days is None
    assert config.cache_max_mb is None
    assert config.resync is False
    assert config.verbose is False
    assert config.on_segment is None
    assert config.on_pair_complete is None


def test_pipeline_config_override():
    config = PipelineConfig(model="small", threshold=0.5, verbose=True)
    assert config.model == "small"
    assert config.threshold == 0.5
    assert config.verbose is True


# ---- run() ----

@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_run_returns_pass_result(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare):
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    config = PipelineConfig(use_cache=False, sync=False, device="cpu")
    result = run(VIDEO, SUB, config)
    assert isinstance(result, MatchResult)
    assert result.state == MatchState.PASS
    assert result.confidence == 0.9
    assert result.model == "base"
    assert len(result.segments) == 1
    assert result.segments[0].score == 0.9


@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_run_returns_fail_result(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare):
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    mock_compare.aggregate.return_value = 0.1
    config = PipelineConfig(use_cache=False, sync=False, device="cpu", threshold=0.35)
    result = run(VIDEO, SUB, config)
    assert result.state == MatchState.FAIL
    assert result.confidence == 0.1


@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_run_returns_unsure_when_no_segments(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare):
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    mock_sampler.select_segments.return_value = []
    mock_compare.aggregate.return_value = 0.0
    config = PipelineConfig(use_cache=False, sync=False, device="cpu")
    result = run(VIDEO, SUB, config)
    assert result.state == MatchState.UNSURE


@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_run_verbose_false_no_output(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare, capsys):
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    config = PipelineConfig(use_cache=False, sync=False, device="cpu", verbose=False)
    run(VIDEO, SUB, config)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_on_segment_callback_fires_per_segment(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare):
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    seg2 = Segment(start_ms=90_000, end_ms=120_000, subtitle_text="Goodbye", word_count=1)
    mock_sampler.select_segments.return_value = [_seg(), seg2]
    calls = []
    config = PipelineConfig(use_cache=False, sync=False, device="cpu", on_segment=lambda i, t: calls.append((i, t)))
    run(VIDEO, SUB, config)
    assert calls == [(1, 2), (2, 2)]


# ---- run_batch() ----

@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_run_batch_returns_list_of_results(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare):
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    config = PipelineConfig(use_cache=False, sync=False, device="cpu", workers=1)
    results = run_batch([(VIDEO, SUB)], config)
    assert len(results) == 1
    assert isinstance(results[0], BatchPairResult)
    assert results[0].result.state == MatchState.PASS
    assert results[0].video == VIDEO
    assert results[0].subtitle == SUB


@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_run_batch_reuses_cache_for_same_video(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare):
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    sub2 = Path("movie.pt.srt")
    config = PipelineConfig(use_cache=False, sync=False, device="cpu", workers=1)
    results = run_batch([(VIDEO, SUB), (VIDEO, sub2)], config)
    assert len(results) == 2
    # duration_ms only called once per video (cache reuse)
    assert mock_audio.get_duration_ms.call_count == 1


@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_run_batch_on_pair_complete_fires_per_pair(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare):
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    completed = []
    config = PipelineConfig(
        use_cache=False, sync=False, device="cpu", workers=1,
        on_pair_complete=lambda r: completed.append(r),
    )
    run_batch([(VIDEO, SUB), (VIDEO2, SUB2)], config)
    assert len(completed) == 2
    assert all(isinstance(r, BatchPairResult) for r in completed)


@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_run_batch_records_error_for_failing_pair(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare):
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    mock_audio.get_duration_ms.side_effect = RuntimeError("ffprobe failed")
    config = PipelineConfig(use_cache=False, sync=False, device="cpu", workers=1)
    results = run_batch([(VIDEO, SUB)], config)
    assert len(results) == 1
    assert results[0].error is not None
    assert "ffprobe failed" in results[0].error


@patch("submatch.pipeline._score_group_parallel")
@patch("submatch.pipeline._resolve_device", return_value="cpu")
@patch("submatch.pipeline._resolve_workers", return_value=2)
def test_run_batch_uses_parallel_when_workers_gt_1(mock_rw, mock_rd, mock_sgp):
    mock_sgp.return_value = [BatchPairResult(video=VIDEO, subtitle=SUB, result=MagicMock(), error=None)]
    config = PipelineConfig(use_cache=False, sync=False, device="cpu", workers=2)
    results = run_batch([(VIDEO, SUB)], config)
    assert mock_sgp.called
    assert len(results) == 1


# ── pass_unsure ────────────────────────────────────────────────────────────────

def _make_unsure_result():
    from submatch.language import LanguageResult
    lang = LanguageResult(audio=None, subtitle_detected=None, subtitle_filename=None,
                          video_metadata=None, expected=None, mismatch=False,
                          mismatch_details=[])
    return MatchResult(confidence=0.0, passed=False, threshold=0.35,
                       language=lang, sync=None, segments=[], model="base",
                       state=MatchState.UNSURE)


def test_pass_unsure_sets_passed_true_on_unsure(tmp_path):
    video = tmp_path / "v.mkv"
    sub = tmp_path / "s.srt"
    config = PipelineConfig(pass_unsure=True)
    result = _make_unsure_result()

    with patch("submatch.scoring._score_pair", return_value=(result, MagicMock())), \
         patch("submatch.pipeline._get_model", return_value=MagicMock()), \
         patch("submatch.pipeline._resolve_device", return_value="cpu"):
        final = run(video, sub, config)

    assert final.passed is True


def test_pass_unsure_false_leaves_passed_false(tmp_path):
    video = tmp_path / "v.mkv"
    sub = tmp_path / "s.srt"
    config = PipelineConfig(pass_unsure=False)
    result = _make_unsure_result()

    with patch("submatch.scoring._score_pair", return_value=(result, MagicMock())), \
         patch("submatch.pipeline._get_model", return_value=MagicMock()), \
         patch("submatch.pipeline._resolve_device", return_value="cpu"):
        final = run(video, sub, config)

    assert final.passed is False


def test_drift_result_always_fails(tmp_path):
    """DRIFT result → passed=False regardless of confidence."""
    from submatch.language import LanguageResult
    from submatch.sync import SyncResult
    video = tmp_path / "v.mkv"
    sub = tmp_path / "s.srt"
    lang = LanguageResult(audio="en", subtitle_detected="en", subtitle_filename="en",
                          video_metadata=None, expected=None, mismatch=False,
                          mismatch_details=[])
    sync_tmp = tmp_path / "sync.srt"
    sync_tmp.touch()
    sync_r = SyncResult(synced_srt_path=sync_tmp, offset_seconds=3.0, drift_detected=True)
    result = MatchResult(confidence=0.9, passed=True, threshold=0.35,
                         language=lang, sync=sync_r, segments=[MagicMock()],
                         model="base", state=MatchState.DRIFT)
    config = PipelineConfig()

    with patch("submatch.scoring._score_pair", return_value=(result, MagicMock())), \
         patch("submatch.pipeline._get_model", return_value=MagicMock()), \
         patch("submatch.pipeline._resolve_device", return_value="cpu"):
        final = run(video, sub, config)

    assert final.passed is False


# ── keep_synced ────────────────────────────────────────────────────────────────

def _make_pass_result_with_sync(sync_tmp):
    from submatch.language import LanguageResult
    from submatch.sync import SyncResult
    lang = LanguageResult(audio="en", subtitle_detected="en", subtitle_filename="en",
                          video_metadata=None, expected=None, mismatch=False,
                          mismatch_details=[])
    sync_r = SyncResult(synced_srt_path=sync_tmp, offset_seconds=0.5, drift_detected=False)
    return MatchResult(confidence=0.9, passed=True, threshold=0.35,
                       language=lang, sync=sync_r, segments=[MagicMock()],
                       model="base", state=MatchState.PASS)


def test_keep_synced_copies_and_deletes_tmp(tmp_path):
    video = tmp_path / "v.mkv"
    sub = tmp_path / "s.srt"
    sub.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    sync_tmp = tmp_path / "sync.srt"
    sync_tmp.write_text("[synced]")
    config = PipelineConfig(keep_synced=True)
    result = _make_pass_result_with_sync(sync_tmp)

    with patch("submatch.scoring._score_pair", return_value=(result, MagicMock())), \
         patch("submatch.pipeline._get_model", return_value=MagicMock()), \
         patch("submatch.pipeline._resolve_device", return_value="cpu"):
        run(video, sub, config)

    kept = sub.with_stem(sub.stem + ".synced")
    assert kept.exists()
    assert kept.read_text() == "[synced]"
    assert not sync_tmp.exists()


def test_keep_synced_false_deletes_tmp_only(tmp_path):
    video = tmp_path / "v.mkv"
    sub = tmp_path / "s.srt"
    sub.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    sync_tmp = tmp_path / "sync.srt"
    sync_tmp.write_text("[synced]")
    config = PipelineConfig(keep_synced=False)
    result = _make_pass_result_with_sync(sync_tmp)

    with patch("submatch.scoring._score_pair", return_value=(result, MagicMock())), \
         patch("submatch.pipeline._get_model", return_value=MagicMock()), \
         patch("submatch.pipeline._resolve_device", return_value="cpu"):
        run(video, sub, config)

    kept = sub.with_stem(sub.stem + ".synced")
    assert not kept.exists()
    assert not sync_tmp.exists()


# ── delete_failures ────────────────────────────────────────────────────────────

def _make_fail_result():
    from submatch.language import LanguageResult
    lang = LanguageResult(audio="en", subtitle_detected="en", subtitle_filename="en",
                          video_metadata=None, expected=None, mismatch=False,
                          mismatch_details=[])
    return MatchResult(confidence=0.1, passed=False, threshold=0.35,
                       language=lang, sync=None, segments=[MagicMock()],
                       model="base", state=MatchState.FAIL)


def test_delete_failures_unlinks_subtitle_on_fail(tmp_path):
    video = tmp_path / "v.mkv"
    sub = tmp_path / "s.srt"
    sub.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    config = PipelineConfig(delete_failures=True)
    result = _make_fail_result()

    with patch("submatch.scoring._score_pair", return_value=(result, MagicMock())), \
         patch("submatch.pipeline._get_model", return_value=MagicMock()), \
         patch("submatch.pipeline._resolve_device", return_value="cpu"):
        run(video, sub, config)

    assert not sub.exists()


def test_delete_failures_false_keeps_subtitle_on_fail(tmp_path):
    video = tmp_path / "v.mkv"
    sub = tmp_path / "s.srt"
    sub.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    config = PipelineConfig(delete_failures=False)
    result = _make_fail_result()

    with patch("submatch.scoring._score_pair", return_value=(result, MagicMock())), \
         patch("submatch.pipeline._get_model", return_value=MagicMock()), \
         patch("submatch.pipeline._resolve_device", return_value="cpu"):
        run(video, sub, config)

    assert sub.exists()


# ── PipelineConfig.from_toml ───────────────────────────────────────────────────

def test_from_toml_maps_model_and_threshold():
    with patch("submatch.config.load_config", return_value={"model": "tiny", "threshold": 0.5}):
        config = PipelineConfig.from_toml()
    assert config.model == "tiny"
    assert config.threshold == 0.5


def test_from_toml_uses_defaults_when_config_empty():
    with patch("submatch.config.load_config", return_value={}):
        config = PipelineConfig.from_toml()
    assert config.model == "base"
    assert config.threshold == 0.35
    assert config.sync is True
    assert config.pass_unsure is False
    assert config.keep_synced is False
    assert config.delete_failures is False


def test_from_toml_maps_new_fields():
    with patch("submatch.config.load_config", return_value={
        "pass_unsure": True,
        "keep_synced": True,
        "delete_failures": True,
    }):
        config = PipelineConfig.from_toml()
    assert config.pass_unsure is True
    assert config.keep_synced is True
    assert config.delete_failures is True


def test_from_toml_no_sync_maps_to_sync_false():
    with patch("submatch.config.load_config", return_value={"no_sync": True}):
        config = PipelineConfig.from_toml()
    assert config.sync is False


def test_from_toml_cache_dir_expands_tilde():
    with patch("submatch.config.load_config", return_value={"cache_dir": "~/.cache/sm"}):
        config = PipelineConfig.from_toml()
    assert config.cache_dir is not None
    assert not str(config.cache_dir).startswith("~")
    assert str(config.cache_dir) == str(Path("~/.cache/sm").expanduser())


def test_from_toml_cli_only_keys_are_ignored():
    """Keys like no_recursive and sub_lang have no PipelineConfig field — silently ignored."""
    with patch("submatch.config.load_config", return_value={
        "no_recursive": True,
        "sub_lang": ["en"],
        "filter": "*.en.*",
        "model": "small",
    }):
        config = PipelineConfig.from_toml()
    assert config.model == "small"
    assert not hasattr(config, "no_recursive")


# ── musical content support ───────────────────────────────────────────────────

@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.sampler")
def test_audio_driven_transcribe_rejects_low_avg_logprob(mock_sampler, mock_audio, mock_transcribe, tmp_path):
    """Quality gate rejects candidate with avg_logprob below -1.0 and tries the next."""
    from submatch.scoring import _audio_driven_transcribe
    video = tmp_path / "v.mkv"
    video.touch()
    config = PipelineConfig(use_cache=True, sync=False, device="cpu")

    mock_audio.detect_speech_regions.return_value = []
    mock_audio.extract_segment.return_value = MagicMock()

    # One zone, two candidates: first has low logprob (rejected), second is accepted
    mock_sampler.audio_candidate_segments.return_value = [[0, 30_000]]

    low = TranscriptionResult(text="singing lyrics here today", language="en",
                              no_speech_prob=0.1, avg_logprob=-2.0)
    good = TranscriptionResult(text="hello world indeed", language="en",
                               no_speech_prob=0.1, avg_logprob=0.5)
    mock_transcribe.transcribe_segment.side_effect = [low, good]

    starts, texts, _ = _audio_driven_transcribe(video, 0, 1, MagicMock(), config,
                                                duration_ms=1_800_000)
    assert texts == ["hello world indeed"]


@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_run_unsure_when_all_segments_below_coverage_threshold(
    mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare
):
    """All segments have no subtitle text but Whisper heard speech → UNSURE."""
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    sparse = Segment(start_ms=60_000, end_ms=90_000, subtitle_text="", word_count=0)
    mock_sampler.select_segments.return_value = [sparse, sparse, sparse, sparse]
    config = PipelineConfig(use_cache=False, sync=False, device="cpu")
    result = run(VIDEO, SUB, config)
    assert result.state == MatchState.UNSURE


@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_run_scores_normally_with_partial_coverage(
    mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare
):
    """Only 1 of 4 segments has subtitle content → still scores normally (not UNSURE)."""
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    good = _seg()
    sparse = Segment(start_ms=90_000, end_ms=120_000, subtitle_text="", word_count=0)
    mock_sampler.select_segments.return_value = [good, sparse, sparse, sparse]
    mock_compare.aggregate.return_value = 0.9
    config = PipelineConfig(use_cache=False, sync=False, device="cpu")
    result = run(VIDEO, SUB, config)
    assert result.state == MatchState.PASS


@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_run_excludes_sparse_segments_from_scoring(
    mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare
):
    """Sparse segments (word_count=0, trans≠empty) are not included in segment_results."""
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    good = _seg()
    sparse = Segment(start_ms=90_000, end_ms=120_000, subtitle_text="", word_count=0)
    mock_sampler.select_segments.return_value = [good, good, sparse, sparse]
    mock_compare.aggregate.return_value = 0.9
    config = PipelineConfig(use_cache=False, sync=False, device="cpu")
    result = run(VIDEO, SUB, config)
    assert len(result.segments) == 2


@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_run_scores_normally_with_odd_segment_count(
    mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare
):
    """2 good + 1 sparse → PASS (all good segments scored)."""
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    good = _seg()
    sparse = Segment(start_ms=90_000, end_ms=120_000, subtitle_text="", word_count=0)
    mock_sampler.select_segments.return_value = [good, good, sparse]
    mock_compare.aggregate.return_value = 0.9
    config = PipelineConfig(use_cache=False, sync=False, device="cpu")
    result = run(VIDEO, SUB, config)
    assert result.state == MatchState.PASS


# ── default config (config=None) ──────────────────────────────────────────────

def test_run_with_no_config_uses_defaults(tmp_path):
    # Covers pipeline.py line 158: if config is None: config = PipelineConfig()
    video = tmp_path / "v.mkv"
    sub = tmp_path / "s.srt"
    result = _make_unsure_result()
    with patch("submatch.scoring._score_pair", return_value=(result, MagicMock())), \
         patch("submatch.pipeline._get_model", return_value=MagicMock()), \
         patch("submatch.pipeline._resolve_device", return_value="cpu"):
        final = run(video, sub)  # no config argument → line 158
    assert final is not None


def test_run_batch_with_no_config_uses_defaults(tmp_path):
    # Covers pipeline.py line 181: if config is None: config = PipelineConfig()
    video = tmp_path / "v.mkv"
    sub = tmp_path / "s.srt"
    result = _make_unsure_result()
    with patch("submatch.scoring._score_pair", return_value=(result, MagicMock())), \
         patch("submatch.pipeline._get_model", return_value=MagicMock()), \
         patch("submatch.pipeline._resolve_device", return_value="cpu"), \
         patch("submatch.pipeline._resolve_workers", return_value=1):
        results = run_batch([(video, sub)])  # no config argument → line 181
    assert len(results) == 1


# ── resync branch in run_batch (single-worker) ────────────────────────────────

def test_run_batch_resync_on_drift(tmp_path):
    # Covers pipeline.py lines 195-203: resync branch when DRIFT + resync=True
    from submatch.language import LanguageResult
    from submatch.sync import SyncResult

    video = tmp_path / "v.mkv"
    sub = tmp_path / "s.srt"
    sub.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    sync_tmp = tmp_path / "sync.srt"
    sync_tmp.write_text("[synced]")

    lang = LanguageResult(audio="en", subtitle_detected="en", subtitle_filename="en",
                          video_metadata=None, expected=None, mismatch=False, mismatch_details=[])
    sync_r = SyncResult(synced_srt_path=sync_tmp, offset_seconds=3.0, drift_detected=True)
    drift_result = MatchResult(confidence=0.9, passed=True, threshold=0.35,
                               language=lang, sync=sync_r, segments=[MagicMock()],
                               model="base", state=MatchState.DRIFT)
    pass_result = MatchResult(confidence=0.9, passed=True, threshold=0.35,
                              language=lang, sync=None, segments=[MagicMock()],
                              model="base", state=MatchState.PASS)

    mock_cache = MagicMock()
    call_count = [0]

    def fake_score_pair(*args, **kwargs):
        call_count[0] += 1
        return (drift_result, mock_cache) if call_count[0] == 1 else (pass_result, mock_cache)

    config = PipelineConfig(resync=True, workers=1)
    with patch("submatch.scoring._score_pair", side_effect=fake_score_pair), \
         patch("submatch.pipeline._get_model", return_value=MagicMock()), \
         patch("submatch.pipeline._resolve_device", return_value="cpu"):
        results = run_batch([(video, sub)], config)

    assert len(results) == 1
    assert results[0].result.resynced is True
    assert call_count[0] == 2  # score_pair called twice (first drift, then resync)


# ── parallel exception handler ────────────────────────────────────────────────

def test_run_batch_parallel_worker_exception(tmp_path):
    # Covers pipeline.py lines 237-244: exception in parallel worker propagated to results
    video = tmp_path / "v.mkv"
    sub = tmp_path / "s.srt"

    completed = []
    config = PipelineConfig(workers=2, on_pair_complete=completed.append)

    with patch("submatch.pipeline._score_group_parallel",
               side_effect=RuntimeError("GPU OOM")), \
         patch("submatch.pipeline._resolve_device", return_value="cpu"):
        results = run_batch([(video, sub)], config)

    assert len(results) == 1
    assert results[0].error == "GPU OOM"
    assert results[0].result is None
    assert len(completed) == 1


# ── lazy sync ─────────────────────────────────────────────────────────────────

@patch("submatch.scoring.sync")
@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_sync_not_called_when_first_pass_passes(
    mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare, mock_sync
):
    """ffs must not run when the first scoring pass already passes."""
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    mock_compare.aggregate.return_value = 0.9
    config = PipelineConfig(use_cache=False, sync=True, device="cpu")
    result = run(VIDEO, SUB, config)
    assert result.state == MatchState.PASS
    mock_sync.sync_subtitle.assert_not_called()


@patch("submatch.scoring.sync")
@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_sync_called_when_first_pass_fails(
    mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare, mock_sync
):
    """ffs must run when first pass fails and sync=True."""
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    mock_compare.aggregate.return_value = 0.1
    from submatch.sync import SyncResult
    sync_tmp = MagicMock()
    mock_sync.sync_subtitle.return_value = SyncResult(
        synced_srt_path=sync_tmp, offset_seconds=3.0, drift_detected=True
    )
    mock_sub.parse.side_effect = [
        [Subtitle(index=1, start_ms=1000, end_ms=5000, text="Hello world")],
        [Subtitle(index=1, start_ms=4000, end_ms=8000, text="Hello world")],
    ]
    config = PipelineConfig(use_cache=False, sync=True, device="cpu")
    run(VIDEO, SUB, config)
    mock_sync.sync_subtitle.assert_called_once()


@patch("submatch.scoring.sync")
@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_sync_not_called_when_no_sync(
    mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare, mock_sync
):
    """ffs must never run when sync=False, even if first pass fails."""
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    mock_compare.aggregate.return_value = 0.1
    config = PipelineConfig(use_cache=False, sync=False, device="cpu")
    result = run(VIDEO, SUB, config)
    assert result.state == MatchState.FAIL
    mock_sync.sync_subtitle.assert_not_called()


@patch("submatch.scoring.sync")
@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_drift_state_when_sync_rescues_failing_pair(
    mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare, mock_sync
):
    """First pass FAIL + second pass PASS after ffs → DRIFT."""
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    mock_compare.aggregate.side_effect = [0.1, 0.9]
    mock_sampler.segments_from_starts.return_value = [_seg()]
    from submatch.sync import SyncResult
    sync_tmp = MagicMock()
    mock_sync.sync_subtitle.return_value = SyncResult(
        synced_srt_path=sync_tmp, offset_seconds=3.0, drift_detected=True
    )
    mock_sub.parse.side_effect = [
        [Subtitle(index=1, start_ms=1000, end_ms=5000, text="Hello world")],
        [Subtitle(index=1, start_ms=4000, end_ms=8000, text="Hello world")],
    ]
    config = PipelineConfig(use_cache=False, sync=True, device="cpu")
    result = run(VIDEO, SUB, config)
    assert result.state == MatchState.DRIFT


@patch("submatch.scoring.sync")
@patch("submatch.scoring.compare")
@patch("submatch.scoring.transcribe")
@patch("submatch.scoring.sampler")
@patch("submatch.scoring.audio")
@patch("submatch.scoring.language")
@patch("submatch.scoring.subtitle")
def test_fail_state_when_sync_does_not_rescue(
    mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare, mock_sync
):
    """First pass FAIL + second pass FAIL after ffs → FAIL."""
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    mock_compare.aggregate.side_effect = [0.1, 0.1]
    mock_sampler.segments_from_starts.return_value = [_seg()]
    from submatch.sync import SyncResult
    sync_tmp = MagicMock()
    mock_sync.sync_subtitle.return_value = SyncResult(
        synced_srt_path=sync_tmp, offset_seconds=3.0, drift_detected=True
    )
    mock_sub.parse.side_effect = [
        [Subtitle(index=1, start_ms=1000, end_ms=5000, text="Hello world")],
        [Subtitle(index=1, start_ms=4000, end_ms=8000, text="Hello world")],
    ]
    config = PipelineConfig(use_cache=False, sync=True, device="cpu")
    result = run(VIDEO, SUB, config)
    assert result.state == MatchState.FAIL


# ── resync branch in run() ───────────────────────────────────────────────────

def test_run_resync_on_drift(tmp_path):
    # Covers pipeline.py lines 163-171: resync when run() sees DRIFT + resync=True
    from submatch.language import LanguageResult
    from submatch.sync import SyncResult

    video = tmp_path / "v.mkv"
    video.touch()
    sub = tmp_path / "s.srt"
    sub.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    sync_tmp = tmp_path / "sync.srt"
    sync_tmp.write_text("[synced]")

    lang = LanguageResult(audio="en", subtitle_detected="en", subtitle_filename="en",
                          video_metadata=None, expected=None, mismatch=False, mismatch_details=[])
    sync_r = SyncResult(synced_srt_path=sync_tmp, offset_seconds=2.0, drift_detected=True)
    drift_result = MatchResult(confidence=0.9, passed=True, threshold=0.35,
                               language=lang, sync=sync_r, segments=[MagicMock()],
                               model="base", state=MatchState.DRIFT)
    pass_result = MatchResult(confidence=0.9, passed=True, threshold=0.35,
                              language=lang, sync=None, segments=[MagicMock()],
                              model="base", state=MatchState.PASS)

    call_count = [0]

    def fake_score_pair(*args, **kwargs):
        call_count[0] += 1
        return (drift_result, MagicMock()) if call_count[0] == 1 else (pass_result, MagicMock())

    config = PipelineConfig(resync=True)
    with patch("submatch.scoring._score_pair", side_effect=fake_score_pair), \
         patch("submatch.pipeline._get_model", return_value=MagicMock()), \
         patch("submatch.pipeline._resolve_device", return_value="cpu"):
        result = run(video, sub, config)

    assert result.resynced is True
    assert call_count[0] == 2


# ── resync in parallel worker path (_process_video_subs) ─────────────────────

def test_process_video_subs_resync_on_drift(tmp_path):
    # Covers pipeline.py lines 130-138: resync in _process_video_subs (workers > 1)
    from submatch.language import LanguageResult
    from submatch.sync import SyncResult

    video = tmp_path / "v.mkv"
    video.touch()
    sub = tmp_path / "s.srt"
    sub.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    sync_tmp = tmp_path / "sync.srt"
    sync_tmp.write_text("[synced]")

    lang = LanguageResult(audio="en", subtitle_detected="en", subtitle_filename="en",
                          video_metadata=None, expected=None, mismatch=False, mismatch_details=[])
    sync_r = SyncResult(synced_srt_path=sync_tmp, offset_seconds=2.0, drift_detected=True)
    drift_result = MatchResult(confidence=0.9, passed=True, threshold=0.35,
                               language=lang, sync=sync_r, segments=[MagicMock()],
                               model="base", state=MatchState.DRIFT)
    pass_result = MatchResult(confidence=0.9, passed=True, threshold=0.35,
                              language=lang, sync=None, segments=[MagicMock()],
                              model="base", state=MatchState.PASS)

    call_count = [0]

    def fake_score_pair(*args, **kwargs):
        call_count[0] += 1
        return (drift_result, MagicMock()) if call_count[0] == 1 else (pass_result, MagicMock())

    config = PipelineConfig(resync=True, workers=2)
    with patch("submatch.scoring._score_pair", side_effect=fake_score_pair), \
         patch("submatch.pipeline._get_model", return_value=MagicMock()), \
         patch("submatch.pipeline._resolve_device", return_value="cpu"):
        results = run_batch([(video, sub)], config)

    assert len(results) == 1
    assert results[0].result.resynced is True
    assert call_count[0] == 2

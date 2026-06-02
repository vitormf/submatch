from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from submatch.compare import SegmentScore
from submatch.language import LanguageResult
from submatch.output import BatchPairResult, MatchResult, MatchState
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
    from submatch import pipeline
    for store in (pipeline._model_local, pipeline._embed_local):
        if hasattr(store, "model"):
            del store.model
    yield
    for store in (pipeline._model_local, pipeline._embed_local):
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
        text="Hello world", language="en", no_speech_prob=0.1
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

@patch("submatch.pipeline.compare")
@patch("submatch.pipeline.transcribe")
@patch("submatch.pipeline.sampler")
@patch("submatch.pipeline.audio")
@patch("submatch.pipeline.language")
@patch("submatch.pipeline.subtitle")
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


@patch("submatch.pipeline.compare")
@patch("submatch.pipeline.transcribe")
@patch("submatch.pipeline.sampler")
@patch("submatch.pipeline.audio")
@patch("submatch.pipeline.language")
@patch("submatch.pipeline.subtitle")
def test_run_returns_fail_result(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare):
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    mock_compare.aggregate.return_value = 0.1
    config = PipelineConfig(use_cache=False, sync=False, device="cpu", threshold=0.35)
    result = run(VIDEO, SUB, config)
    assert result.state == MatchState.FAIL
    assert result.confidence == 0.1


@patch("submatch.pipeline.compare")
@patch("submatch.pipeline.transcribe")
@patch("submatch.pipeline.sampler")
@patch("submatch.pipeline.audio")
@patch("submatch.pipeline.language")
@patch("submatch.pipeline.subtitle")
def test_run_returns_unsure_when_no_segments(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare):
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    mock_sampler.select_segments.return_value = []
    mock_compare.aggregate.return_value = 0.0
    config = PipelineConfig(use_cache=False, sync=False, device="cpu")
    result = run(VIDEO, SUB, config)
    assert result.state == MatchState.UNSURE


@patch("submatch.pipeline.compare")
@patch("submatch.pipeline.transcribe")
@patch("submatch.pipeline.sampler")
@patch("submatch.pipeline.audio")
@patch("submatch.pipeline.language")
@patch("submatch.pipeline.subtitle")
def test_run_verbose_false_no_output(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare, capsys):
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    config = PipelineConfig(use_cache=False, sync=False, device="cpu", verbose=False)
    run(VIDEO, SUB, config)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@patch("submatch.pipeline.compare")
@patch("submatch.pipeline.transcribe")
@patch("submatch.pipeline.sampler")
@patch("submatch.pipeline.audio")
@patch("submatch.pipeline.language")
@patch("submatch.pipeline.subtitle")
def test_on_segment_callback_fires_per_segment(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare):
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    seg2 = Segment(start_ms=90_000, end_ms=120_000, subtitle_text="Goodbye", word_count=1)
    mock_sampler.select_segments.return_value = [_seg(), seg2]
    calls = []
    config = PipelineConfig(use_cache=False, sync=False, device="cpu", on_segment=lambda i, t: calls.append((i, t)))
    run(VIDEO, SUB, config)
    assert calls == [(1, 2), (2, 2)]


# ---- run_batch() ----

@patch("submatch.pipeline.compare")
@patch("submatch.pipeline.transcribe")
@patch("submatch.pipeline.sampler")
@patch("submatch.pipeline.audio")
@patch("submatch.pipeline.language")
@patch("submatch.pipeline.subtitle")
def test_run_batch_returns_list_of_results(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare):
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    config = PipelineConfig(use_cache=False, sync=False, device="cpu", workers=1)
    results = run_batch([(VIDEO, SUB)], config)
    assert len(results) == 1
    assert isinstance(results[0], BatchPairResult)
    assert results[0].result.state == MatchState.PASS
    assert results[0].video == VIDEO
    assert results[0].subtitle == SUB


@patch("submatch.pipeline.compare")
@patch("submatch.pipeline.transcribe")
@patch("submatch.pipeline.sampler")
@patch("submatch.pipeline.audio")
@patch("submatch.pipeline.language")
@patch("submatch.pipeline.subtitle")
def test_run_batch_reuses_cache_for_same_video(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare):
    _apply_mocks(mock_sub, mock_lang, mock_audio, mock_sampler, mock_transcribe, mock_compare)
    sub2 = Path("movie.pt.srt")
    config = PipelineConfig(use_cache=False, sync=False, device="cpu", workers=1)
    results = run_batch([(VIDEO, SUB), (VIDEO, sub2)], config)
    assert len(results) == 2
    # duration_ms only called once per video (cache reuse)
    assert mock_audio.get_duration_ms.call_count == 1


@patch("submatch.pipeline.compare")
@patch("submatch.pipeline.transcribe")
@patch("submatch.pipeline.sampler")
@patch("submatch.pipeline.audio")
@patch("submatch.pipeline.language")
@patch("submatch.pipeline.subtitle")
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


@patch("submatch.pipeline.compare")
@patch("submatch.pipeline.transcribe")
@patch("submatch.pipeline.sampler")
@patch("submatch.pipeline.audio")
@patch("submatch.pipeline.language")
@patch("submatch.pipeline.subtitle")
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

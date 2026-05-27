from submatch.compare import normalize, token_f1, aggregate, SegmentScore


def test_normalize_lowercases():
    assert normalize("Hello World") == ["hello", "world"]


def test_normalize_strips_punctuation():
    assert normalize("Hello, world!") == ["hello", "world"]


def test_normalize_removes_filler_words():
    result = normalize("um hello uh world")
    assert "um" not in result
    assert "uh" not in result
    assert "hello" in result
    assert "world" in result


def test_token_f1_exact_match():
    score = token_f1("hello world", "hello world")
    assert score.f1 == 1.0


def test_token_f1_no_overlap():
    score = token_f1("hello world", "goodbye moon")
    assert score.f1 == 0.0


def test_token_f1_partial_match():
    score = token_f1("hello world foo", "hello world bar")
    assert 0.0 < score.f1 < 1.0


def test_token_f1_empty_both():
    score = token_f1("", "")
    assert score.f1 == 1.0


def test_token_f1_empty_subtitle():
    score = token_f1("", "hello world")
    assert score.f1 == 0.0


def test_token_f1_subtitle_tokens_count():
    score = token_f1("hello world foo", "anything")
    assert score.subtitle_tokens == 3


def test_aggregate_exact_weights():
    scores = [
        SegmentScore(f1=1.0, wer=0.0, subtitle_tokens=10),
        SegmentScore(f1=0.0, wer=1.0, subtitle_tokens=10),
    ]
    assert aggregate(scores) == 0.5


def test_aggregate_respects_weights():
    scores = [
        SegmentScore(f1=1.0, wer=0.0, subtitle_tokens=1),
        SegmentScore(f1=0.0, wer=1.0, subtitle_tokens=9),
    ]
    result = aggregate(scores)
    assert result < 0.2


def test_aggregate_empty_returns_zero():
    assert aggregate([]) == 0.0


def test_aggregate_zero_subtitle_tokens_unweighted():
    import pytest
    scores = [
        SegmentScore(f1=0.8, wer=0.2, subtitle_tokens=0),
        SegmentScore(f1=0.4, wer=0.6, subtitle_tokens=0),
    ]
    assert aggregate(scores) == pytest.approx(0.6)


def test_word_error_rate_empty_ref_nonempty_hyp():
    from submatch.compare import _word_error_rate
    assert _word_error_rate([], ["hello"]) == 1.0


def test_word_error_rate_both_empty():
    from submatch.compare import _word_error_rate
    assert _word_error_rate([], []) == 0.0

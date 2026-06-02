"""
Property-based tests for submatch.compare using Hypothesis.

These complement the example-based tests in test_compare.py by verifying
mathematical invariants across a wide range of random inputs.
"""
from __future__ import annotations
from hypothesis import given, assume, settings
from hypothesis import strategies as st

from submatch.compare import token_f1, aggregate, _word_error_rate, SegmentScore


# Only printable ASCII words to keep transcriptions realistic
_WORDS = st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=50)
_WORD_LIST = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10),
    min_size=0,
    max_size=20,
)


@given(_WORDS, _WORDS)
def test_token_f1_score_always_in_unit_interval(a, b):
    score = token_f1(a, b)
    assert 0.0 <= score.f1 <= 1.0


@given(_WORDS)
def test_token_f1_identical_inputs_gives_perfect_score(text):
    assume(text.strip())  # skip whitespace-only strings (normalize → empty)
    score = token_f1(text, text)
    assert score.f1 == 1.0


@given(_WORDS, _WORDS)
def test_token_f1_wer_non_negative(a, b):
    score = token_f1(a, b)
    assert score.wer >= 0.0


@given(_WORDS)
def test_token_f1_identical_inputs_zero_wer(text):
    assume(text.strip())
    score = token_f1(text, text)
    assert score.wer == 0.0


@given(_WORDS)
def test_token_f1_subtitle_tokens_matches_normalize(text):
    from submatch.compare import normalize
    score = token_f1(text, "anything")
    assert score.subtitle_tokens == len(normalize(text))


@given(_WORD_LIST)
def test_word_error_rate_identical_ref_and_hyp_is_zero(words):
    assume(words)
    assert _word_error_rate(words, words) == 0.0


@given(_WORD_LIST, _WORD_LIST)
def test_word_error_rate_non_negative(ref, hyp):
    assert _word_error_rate(ref, hyp) >= 0.0


@given(st.lists(
    st.builds(
        SegmentScore,
        f1=st.floats(min_value=0.0, max_value=1.0),
        wer=st.floats(min_value=0.0, max_value=1.0),
        subtitle_tokens=st.integers(min_value=0, max_value=100),
    ),
    min_size=0,
    max_size=10,
))
def test_aggregate_always_in_unit_interval(scores):
    result = aggregate(scores)
    assert 0.0 <= result <= 1.0


def test_aggregate_empty_is_zero():
    assert aggregate([]) == 0.0


@given(st.floats(min_value=0.0, max_value=1.0), st.integers(min_value=1, max_value=20))
@settings(max_examples=50)
def test_aggregate_uniform_score_equals_that_score(f1_val, n):
    scores = [SegmentScore(f1=f1_val, wer=0.0, subtitle_tokens=5) for _ in range(n)]
    result = aggregate(scores)
    assert abs(result - f1_val) < 1e-9

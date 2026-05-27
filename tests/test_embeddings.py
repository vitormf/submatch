import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def test_normalize_cross_score_at_baseline():
    from submatch.embeddings import normalize_cross_score
    assert normalize_cross_score(0.15) == pytest.approx(0.0)


def test_normalize_cross_score_at_one():
    from submatch.embeddings import normalize_cross_score
    assert normalize_cross_score(1.0) == pytest.approx(1.0)


def test_normalize_cross_score_below_baseline_clamps_to_zero():
    from submatch.embeddings import normalize_cross_score
    assert normalize_cross_score(0.0) == 0.0
    assert normalize_cross_score(-0.5) == 0.0


def test_normalize_cross_score_midpoint():
    from submatch.embeddings import normalize_cross_score
    # (0.575 - 0.15) / 0.85 == 0.5
    assert normalize_cross_score(0.575) == pytest.approx(0.5)


def test_load_embedding_model_calls_sentence_transformer():
    mock_st = MagicMock()
    mock_model = MagicMock()
    mock_st.SentenceTransformer.return_value = mock_model
    with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
        from submatch import embeddings
        result = embeddings.load_embedding_model()
    mock_st.SentenceTransformer.assert_called_once_with(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    assert result is mock_model


def test_cross_language_score_identical_vectors():
    from submatch.embeddings import cross_language_score
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0]])
    score = cross_language_score("hello", "olá", mock_model)
    # cosine = 1.0 → normalized = (1.0 - 0.15) / 0.85 = 1.0
    assert score.f1 == pytest.approx(1.0)
    assert score.wer == 0.0
    assert score.subtitle_tokens == 1


def test_cross_language_score_orthogonal_vectors():
    from submatch.embeddings import cross_language_score
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
    score = cross_language_score("hello", "xyz", mock_model)
    # cosine = 0.0 → normalized = max(0, (0.0 - 0.15) / 0.85) = 0.0
    assert score.f1 == 0.0
    assert score.wer == 0.0


def test_cross_language_score_both_empty():
    from submatch.embeddings import cross_language_score
    mock_model = MagicMock()
    score = cross_language_score("", "", mock_model)
    assert score.f1 == 1.0
    mock_model.encode.assert_not_called()


def test_cross_language_score_subtitle_empty():
    from submatch.embeddings import cross_language_score
    mock_model = MagicMock()
    score = cross_language_score("", "hello world", mock_model)
    assert score.f1 == 0.0
    mock_model.encode.assert_not_called()


def test_cross_language_score_transcription_empty():
    from submatch.embeddings import cross_language_score
    mock_model = MagicMock()
    score = cross_language_score("olá mundo", "", mock_model)
    assert score.f1 == 0.0
    mock_model.encode.assert_not_called()


def test_cross_language_score_subtitle_token_count():
    from submatch.embeddings import cross_language_score
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0]])
    score = cross_language_score("três palavras aqui", "three words here", mock_model)
    assert score.subtitle_tokens == 3

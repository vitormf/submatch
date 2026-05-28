from __future__ import annotations
import contextlib
import io
from typing import Any
import numpy as np

from submatch.compare import normalize, SegmentScore

# Empirical cosine similarity floor for paraphrase-multilingual-MiniLM-L12-v2
# on unrelated cross-language sentence pairs. Scores at or below this baseline
# normalize to 0.0.
_BASELINE = 0.15
_SCALE = 1.0 - _BASELINE


def load_embedding_model(
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> Any:
    import logging
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    from sentence_transformers import SentenceTransformer
    with contextlib.redirect_stderr(io.StringIO()):
        return SentenceTransformer(model_name)


def normalize_cross_score(cosine: float) -> float:
    return max(0.0, (cosine - _BASELINE) / _SCALE)


def cross_language_score(
    subtitle_text: str,
    transcription: str,
    model: Any,
) -> SegmentScore:
    subtitle_tokens = len(normalize(subtitle_text))
    if not subtitle_text.strip() and not transcription.strip():
        return SegmentScore(f1=1.0, wer=0.0, subtitle_tokens=subtitle_tokens)
    if not subtitle_text.strip() or not transcription.strip():
        return SegmentScore(f1=0.0, wer=0.0, subtitle_tokens=subtitle_tokens)
    vecs = model.encode([subtitle_text, transcription])
    a, b = vecs[0], vecs[1]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / denom) if denom > 0.0 else 0.0
    return SegmentScore(
        f1=normalize_cross_score(cosine),
        wer=0.0,
        subtitle_tokens=subtitle_tokens,
    )

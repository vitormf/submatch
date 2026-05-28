from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class SegmentScore:
    f1: float
    wer: float
    subtitle_tokens: int


_FILLER_WORDS = frozenset({"um", "uh", "hmm", "hm", "ah", "oh", "er"})

# CJK Unified Ideographs and common CJK extension blocks (no spaces between chars)
_CJK_RE = re.compile(
    r"[一-鿿"      # CJK Unified Ideographs
    r"㐀-䶿"       # CJK Extension A
    r"豈-﫿"       # CJK Compatibility Ideographs
    r"぀-ゟ"       # Hiragana
    r"゠-ヿ"       # Katakana
    r"가-힯]"      # Hangul
)


def normalize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    # CJK scripts have no spaces between characters; split each character individually
    text = _CJK_RE.sub(r" \g<0> ", text)
    return [w for w in text.split() if w not in _FILLER_WORDS]


def token_f1(subtitle_text: str, transcription: str) -> SegmentScore:
    sub_tokens = normalize(subtitle_text)
    trans_tokens = normalize(transcription)

    if not sub_tokens and not trans_tokens:
        return SegmentScore(f1=1.0, wer=0.0, subtitle_tokens=0)
    if not sub_tokens or not trans_tokens:
        return SegmentScore(f1=0.0, wer=1.0, subtitle_tokens=len(sub_tokens))

    matched = sum((Counter(sub_tokens) & Counter(trans_tokens)).values())
    precision = matched / len(trans_tokens)
    recall = matched / len(sub_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    wer = _word_error_rate(sub_tokens, trans_tokens)

    return SegmentScore(f1=f1, wer=wer, subtitle_tokens=len(sub_tokens))


def aggregate(scores: list[SegmentScore]) -> float:
    if not scores:
        return 0.0
    total_weight = sum(s.subtitle_tokens for s in scores)
    if total_weight == 0:
        return sum(s.f1 for s in scores) / len(scores)
    return sum(s.f1 * s.subtitle_tokens for s in scores) / total_weight


def _word_error_rate(ref: list[str], hyp: list[str]) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n] / m

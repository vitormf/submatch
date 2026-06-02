from submatch.types import MatchState, SegmentResult, MatchResult, BatchPairResult


def test_types_importable_from_submatch_types():
    assert MatchState.PASS == "PASS"
    assert MatchState.FAIL == "FAIL"
    assert MatchState.DRIFT == "DRIFT"
    assert MatchState.UNSURE == "UNSURE"


def test_segment_result_fields():
    seg = SegmentResult(index=1, start_ms=1000, score=0.8, wer=0.1,
                        subtitle_text="hello", transcription="hello")
    assert seg.index == 1
    assert seg.score == 0.8


def test_match_result_default_state():
    from submatch.language import LanguageResult
    lang = LanguageResult(audio=None, subtitle_detected=None, subtitle_filename=None,
                          video_metadata=None, expected=None, mismatch=False,
                          mismatch_details=[])
    result = MatchResult(
        confidence=0.5, passed=True, threshold=0.35, language=lang,
        sync=None, segments=[], model="base",
    )
    assert result.state == MatchState.FAIL  # default


def test_batch_pair_result_fields():
    from pathlib import Path
    r = BatchPairResult(video=Path("v.mkv"), subtitle=Path("s.srt"),
                        result=None, error="oops")
    assert r.error == "oops"

from __future__ import annotations
import os
import sys
import tempfile
import threading
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from submatch import audio, compare, embeddings, language, ocr, sampler, subtitle, sync, telemetry, transcribe
from submatch import cache as _cache_module
from submatch.types import MatchResult, MatchState, SegmentResult

if TYPE_CHECKING:
    from submatch.pipeline import PipelineConfig


_embed_local = threading.local()


def _get_embed_model() -> Any:
    if not hasattr(_embed_local, "model"):
        try:
            _embed_local.model = embeddings.load_embedding_model()
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers not installed. "
                "Required for cross-language subtitle matching. "
                "Install with: pip install sentence-transformers"
            ) from exc
    return _embed_local.model


def _resolve_ocr_lang(subtitle_path: Path, video: Path) -> str | None:
    """Return a Tesseract language code for OCR, or None to trigger Tesseract OSD detection."""
    iso = language.detect_from_filename(subtitle_path)
    if iso:
        return language.to_tesseract_lang(iso)
    iso = language.detect_from_video(video)
    if iso:
        return language.to_tesseract_lang(iso)
    return None


def _is_cross_language(audio_lang: str | None, subtitle_lang: str | None) -> bool:
    if not audio_lang or not subtitle_lang:
        return False
    return audio_lang.split("-")[0].lower() != subtitle_lang.split("-")[0].lower()


def _determine_state(result: MatchResult) -> MatchState:
    if len(result.segments) == 0:
        return MatchState.UNSURE
    if not result.passed:
        return MatchState.FAIL
    if result.sync and result.sync.drift_detected:
        return MatchState.DRIFT
    return MatchState.PASS


def _cache_config(config: PipelineConfig) -> dict:
    dir_str = (str(config.cache_dir) if config.cache_dir else None) or os.environ.get("SUBMATCH_CACHE_DIR")
    return {
        "dir": Path(dir_str).expanduser() if dir_str else _cache_module._DEFAULT_CACHE_DIR,
        "ttl_days": config.cache_ttl_days or _cache_module._DEFAULT_TTL_DAYS,
        "max_mb": config.cache_max_mb or _cache_module._DEFAULT_MAX_MB,
    }


def _audio_driven_transcribe(
    video: Path,
    audio_track_index: int,
    n_seg: int,
    model: Any,
    config: PipelineConfig,
    duration_ms: int = 0,
) -> tuple[list[int], list[str], str | None]:
    """Select segments via audio VAD + quality gate. Returns (starts_ms, texts, audio_lang)."""
    speech_regions = audio.detect_speech_regions(video, audio_track_index)
    if not duration_ms:
        duration_ms = audio.get_duration_ms(video)
    zone_candidates = sampler.audio_candidate_segments(
        speech_regions, duration_ms, n_zones=n_seg, candidates_per_zone=2
    )

    accepted_starts: list[int] = []
    accepted_texts: list[str] = []
    lang_votes: list[str] = []
    n_zones = len(zone_candidates)

    for zone_idx, candidates in enumerate(zone_candidates):
        if config.on_segment is not None:
            config.on_segment(zone_idx + 1, n_zones)
        elif config.verbose:
            print(f"  [{zone_idx + 1}/{n_zones}]", end="\r", file=sys.stderr)

        best: tuple[int, str] | None = None
        best_words = -1
        best_lang: str | None = None
        accepted: tuple[int, str] | None = None
        accepted_lang: str | None = None

        for start_ms in candidates:
            try:
                wav_path = audio.extract_segment(video, start_ms, 30_000, audio_track=audio_track_index)
                try:
                    trans = transcribe.transcribe_segment(model, wav_path)
                finally:
                    wav_path.unlink(missing_ok=True)

                words = len(trans.text.split())
                if best is None or words > best_words:
                    best = (start_ms, trans.text)
                    best_words = words
                    best_lang = trans.language

                if trans.no_speech_prob < 0.6 and words >= 3 and trans.avg_logprob > -1.0:
                    accepted = (start_ms, trans.text)
                    accepted_lang = trans.language
                    break
            except Exception as exc:
                telemetry.capture(exc)
                if config.verbose:
                    print(f"Warning: candidate at {start_ms}ms failed: {exc}", file=sys.stderr)

        chosen = accepted if accepted is not None else best
        chosen_lang = accepted_lang if accepted is not None else best_lang
        if chosen is not None:
            accepted_starts.append(chosen[0])
            accepted_texts.append(chosen[1])
        if chosen_lang is not None:
            lang_votes.append(chosen_lang)

    return accepted_starts, accepted_texts, _audio_lang_from_votes(lang_votes)


def _audio_lang_from_votes(votes: list[str]) -> str | None:
    if not votes:
        return None
    counts = Counter(votes)
    most_common = counts.most_common(2)
    top_lang, top_count = most_common[0]
    # Reject a genuine tie at the top (two languages with equal counts).
    if len(most_common) > 1 and most_common[1][1] == top_count:
        return None
    # Accept the leader if it holds ≥ 50% of all votes. The original check
    # used strict `>`, which rejected the exactly-50% case (e.g. 6/12 votes
    # for Japanese when battle scenes confuse Whisper into tagging some
    # segments as English or Korean).
    if top_count * 2 >= len(votes):
        return top_lang
    return None


def _build_match_result(
    transcription_pairs: list[tuple[int, "sampler.Segment", str]],
    subtitle_sample: str,
    subtitle_lang: str | None,
    audio_lang: str | None,
    subtitle_path: Path,
    video: Path,
    config: "PipelineConfig",
    audio_track_index: int,
    audio_track_lang: str | None,
    sync_result: "sync.SyncResult | None" = None,
) -> MatchResult:
    cross_lang = _is_cross_language(audio_lang, subtitle_lang)
    embed_model: Any = _get_embed_model() if cross_lang else None

    # Skip windows where the subtitle has no text but Whisper heard something —
    # that pattern (empty subtitle, non-empty transcription) indicates musical
    # content where scoring against an empty reference gives a false F1=0.
    # When both are empty (silence/no-speech), F1=1.0 is a valid signal.
    scored_pairs = [
        (idx, seg, trans_text)
        for idx, seg, trans_text in transcription_pairs
        if seg.word_count > 0 or not trans_text.strip()
    ]

    segment_results: list[SegmentResult] = []
    for idx, seg, trans_text in scored_pairs:
        if cross_lang:
            score = embeddings.cross_language_score(seg.subtitle_text, trans_text, embed_model)
        else:
            score = compare.token_f1(seg.subtitle_text, trans_text)
        segment_results.append(SegmentResult(
            index=idx,
            start_ms=seg.start_ms,
            score=score.f1,
            wer=score.wer,
            subtitle_text=seg.subtitle_text,
            transcription=trans_text,
        ))

    lang_result = language.build_result(
        audio=audio_lang,
        subtitle_detected=language.detect_from_text(subtitle_sample),
        subtitle_filename=language.detect_from_filename(subtitle_path),
        video_meta=language.detect_from_video(video),
        expected=config.language,
    )

    seg_scores = [
        compare.SegmentScore(
            f1=sr.score,
            wer=sr.wer,
            subtitle_tokens=len(sr.subtitle_text.split()),
        )
        for sr in segment_results
    ]
    confidence = compare.aggregate(seg_scores)

    effective_threshold = (
        config.cross_threshold
        if (cross_lang and config.cross_threshold is not None)
        else config.threshold
    )

    match_result = MatchResult(
        confidence=confidence,
        passed=confidence >= effective_threshold,
        threshold=effective_threshold,
        language=lang_result,
        sync=sync_result,
        segments=segment_results,
        model=config.model,
        cross_language=cross_lang,
        subtitle_language=subtitle_lang,
        audio_track_index=audio_track_index,
        audio_track_lang=audio_track_lang,
    )
    match_result.state = _determine_state(match_result)
    return match_result


def _gather_transcriptions(
    video: Path,
    subtitles: list["subtitle.Subtitle"],
    audio_track_index: int,
    audio_track_lang: str | None,
    config: "PipelineConfig",
    model: Any,
    video_cache: _cache_module.VideoCache | None = None,
) -> tuple[list[tuple[int, "sampler.Segment", str]], _cache_module.VideoCache, str | None]:
    """Return (transcription_pairs, new_cache, audio_lang)."""
    if video_cache is not None:
        audio_lang = video_cache.audio_lang
        cached_segs = sampler.segments_from_starts(subtitles, video_cache.segment_starts)
        transcription_pairs = [
            (i + 1, seg, trans)
            for i, (seg, trans) in enumerate(zip(cached_segs, video_cache.transcriptions))
        ]
        return transcription_pairs, video_cache, audio_lang

    duration_ms = audio.get_duration_ms(video)
    n_seg = config.segments or sampler.auto_segment_count(duration_ms)
    audio_lang: str | None = None
    transcription_pairs: list[tuple[int, sampler.Segment, str]] = []

    if not config.use_cache:
        segs = sampler.select_segments(subtitles, duration_ms, n=config.segments)
        for i, seg in enumerate(segs):
            if config.on_segment is not None:
                config.on_segment(i + 1, len(segs))
            elif config.verbose:
                print(f"  [{i + 1}/{len(segs)}]", end="\r", file=sys.stderr)
            try:
                wav_path = audio.extract_segment(
                    video, seg.start_ms, 30_000, audio_track=audio_track_index
                )
                try:
                    trans = transcribe.transcribe_segment(model, wav_path)
                    if i == 0:
                        audio_lang = trans.language
                    transcription_pairs.append((i + 1, seg, trans.text))
                finally:
                    wav_path.unlink(missing_ok=True)
            except Exception as exc:
                telemetry.capture(exc)
                if config.verbose:
                    print(f"Warning: segment {i + 1} failed: {exc}", file=sys.stderr)
        if config.verbose:
            print()
        new_cache = _cache_module.VideoCache(
            segment_starts=[seg.start_ms for _, seg, _ in transcription_pairs],
            transcriptions=[t for _, _, t in transcription_pairs],
            audio_lang=audio_lang,
            audio_track_index=audio_track_index,
            audio_track_lang=audio_track_lang,
        )
        return transcription_pairs, new_cache, audio_lang

    _cfg = _cache_config(config)
    _mtime = video.stat().st_mtime
    _disk_hit = _cache_module.load(
        video, _mtime, config.model, n_seg, audio_track_index, _cfg["dir"]
    )
    if _disk_hit is not None:
        audio_lang = _disk_hit.audio_lang
        cached_segs = sampler.segments_from_starts(subtitles, _disk_hit.segment_starts)
        transcription_pairs = [
            (i + 1, seg, txt)
            for i, (seg, txt) in enumerate(zip(cached_segs, _disk_hit.transcriptions))
        ]
        return transcription_pairs, _disk_hit, audio_lang

    starts, texts, audio_lang = _audio_driven_transcribe(
        video, audio_track_index, n_seg, model, config,
        duration_ms=duration_ms,
    )
    cached_segs = sampler.segments_from_starts(subtitles, starts)
    transcription_pairs = [
        (i + 1, seg, txt)
        for i, (seg, txt) in enumerate(zip(cached_segs, texts))
    ]
    new_cache = _cache_module.VideoCache(
        segment_starts=starts,
        transcriptions=texts,
        audio_lang=audio_lang,
        audio_track_index=audio_track_index,
        audio_track_lang=audio_track_lang,
    )
    _cache_module.store(
        video, _mtime, config.model, n_seg, audio_track_index,
        new_cache, _cfg["dir"], _cfg["ttl_days"], _cfg["max_mb"],
    )
    return transcription_pairs, new_cache, audio_lang


def _score_pair(
    video: Path,
    subtitle_path: Path,
    config: "PipelineConfig",
    model: Any,
    video_cache: _cache_module.VideoCache | None = None,
) -> tuple[MatchResult, _cache_module.VideoCache]:
    subtitles = subtitle.parse(subtitle_path)
    subtitle_sample = " ".join(s.text for s in subtitles[:50])
    subtitle_lang = (language.detect_from_filename(subtitle_path) or
                     language.detect_from_text(subtitle_sample))

    if video_cache is not None:
        audio_track_index = video_cache.audio_track_index
        audio_track_lang = video_cache.audio_track_lang
    else:
        audio_track_index = 0
        audio_track_lang: str | None = None
        if config.audio_track:
            audio_track_index, audio_track_lang = audio.resolve_audio_track(video, config.audio_track)

    transcription_pairs, new_cache, audio_lang = _gather_transcriptions(
        video, subtitles, audio_track_index, audio_track_lang, config, model, video_cache
    )

    # OCR: populate subtitle_text for image-based subtitle tracks
    _is_image_sub = subtitle.is_image_based(subtitle_path)
    if _is_image_sub:
        if ocr.pytesseract is None:
            print("Warning: pytesseract not installed — cannot OCR image-based subtitle",
                  file=sys.stderr)
        else:
            ocr_lang = _resolve_ocr_lang(subtitle_path, video)
            for _, seg, _ in transcription_pairs:
                try:
                    seg.subtitle_text = ocr.ocr_window(
                        subtitle_path, seg.start_ms, 30_000, lang=ocr_lang
                    )
                    seg.word_count = len(seg.subtitle_text.split())
                except Exception as exc:
                    telemetry.capture(exc)
                    if config.verbose:
                        print(f"Warning: OCR failed for segment at {seg.start_ms}ms: {exc}",
                              file=sys.stderr)

    _sync_args = dict(subtitle_sample=subtitle_sample, subtitle_lang=subtitle_lang,
                      audio_lang=audio_lang, subtitle_path=subtitle_path, video=video,
                      config=config, audio_track_index=audio_track_index,
                      audio_track_lang=audio_track_lang)

    match_result = _build_match_result(transcription_pairs, **_sync_args)

    # Lazy sync: only run ffs when the first pass fails.
    _sync_tmp: Path | None = None
    try:
        if match_result.state == MatchState.FAIL and config.sync and not _is_image_sub:
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
                _sync_tmp = Path(tmp.name)
                tmp.close()
                sync_result = sync.sync_subtitle(
                    video, subtitle_path, _sync_tmp,
                    drift_threshold=config.drift_threshold,
                    audio_track=audio_track_index,
                )
                synced_subtitles = subtitle.parse(sync_result.synced_srt_path)
                starts = [seg.start_ms for _, seg, _ in transcription_pairs]
                synced_segs = sampler.segments_from_starts(synced_subtitles, starts)
                synced_pairs = [
                    (i, seg, txt)
                    for (i, _, txt), seg in zip(transcription_pairs, synced_segs)
                ]
                match_result = _build_match_result(synced_pairs, sync_result=sync_result,
                                                   **_sync_args)
            except RuntimeError as exc:
                telemetry.capture(exc)
                if config.verbose:
                    print(f"Warning: ffsubsync failed ({exc}), keeping FAIL result",
                          file=sys.stderr)
        return match_result, new_cache
    except Exception:
        if _sync_tmp is not None:
            _sync_tmp.unlink(missing_ok=True)
        raise

from __future__ import annotations
import argparse
import concurrent.futures
import copy
import dataclasses
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

from submatch import __version__
from submatch import audio, compare, embeddings, language, output, sampler, subtitle, sync, transcribe


def _ensure_utf8_stdout() -> None:
    """Switch stdout/stderr to UTF-8 on Windows to prevent UnicodeEncodeError when piped."""
    if sys.platform == 'win32':
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="submatch",
        description="Verify a subtitle file matches the audio content of a video.",
    )
    parser.add_argument("inputs", type=Path, nargs="+",
                        help="video files, subtitle files, or directories to process")
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="minimum confidence score to pass (default: 0.35)")
    parser.add_argument("--segments", type=int, default=None,
                        help="number of audio segments to sample (default: auto based on duration)")
    parser.add_argument("--json", action="store_true",
                        help="output results as JSON")
    parser.add_argument("--compact", action="store_true",
                        help="one-line-per-pair output in batch mode")
    parser.add_argument("--verbose", action="store_true",
                        help="show per-segment scores and transcriptions")
    parser.add_argument("--language", default=None,
                        help="expected subtitle language code (e.g. en, pt-BR)")
    parser.add_argument("--no-sync", action="store_true",
                        help="skip ffsubsync timing alignment")
    parser.add_argument("--keep-synced", action="store_true",
                        help="save the timing-corrected subtitle alongside the original")
    parser.add_argument("--no-recursive", action="store_true", dest="no_recursive",
                        help="do not recurse into subdirectories when expanding directories (default: recursive)")
    parser.add_argument("--sub-lang", action="append", dest="sub_lang", metavar="CODE",
                        help="only process subtitles matching this language prefix (repeatable, e.g. --sub-lang pt --sub-lang en)")
    parser.add_argument("--filter", metavar="GLOB",
                        help="only process subtitle files matching this glob pattern (e.g. '*.en.*')")
    parser.add_argument(
        "--device", choices=["cpu", "mps", "cuda", "auto"], default="auto",
        help="Whisper inference device (default: auto — CUDA > CPU)",
    )
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel pairs in batch mode (default: auto — up to 4)")
    parser.add_argument("--delete-failures", action="store_true", dest="delete_failures",
                        help="delete subtitle files that fail the match check")
    parser.add_argument(
        "--cross-threshold", type=float, default=None, dest="cross_threshold",
        help="pass/fail threshold for cross-language pairs (default: same as --threshold)",
    )
    parser.add_argument("--resync", action="store_true",
                        help="if timing drift detected (DRIFT), resync subtitle in place and re-score")
    parser.add_argument("--pass-unsure", action="store_true", dest="pass_unsure",
                        help="exit 0 for UNSURE results (insufficient transcription data)")
    parser.add_argument("--drift-threshold", type=float, default=2.0, dest="drift_threshold",
                        help="seconds of timing offset before flagging as drift (default: 2.0)")
    parser.add_argument("--timing", action="store_true",
                        help="print per-phase timing breakdown (single-pair mode only)")
    parser.add_argument(
        "--audio-track", default=None, dest="audio_track",
        help="audio track to use: integer index (0-based) or comma-separated language preference list (e.g. jp,en,pt)",
    )
    parser.add_argument("--watch", action="store_true",
                        help="monitor a directory for new video/subtitle pairs and score them as they appear")
    parser.add_argument("--poll", action="store_true",
                        help="use polling instead of native filesystem events (for network mounts)")
    parser.add_argument("--interval", type=int, default=10, metavar="N",
                        help="seconds between polls in --poll mode (default: 10)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    from submatch import config as _config
    _cfg: dict = {}
    if not any(a in sys.argv for a in ("--help", "-h", "--version")):
        _cfg = _config.load_config()
    _sub_lang_default = _cfg.pop("sub_lang", None)
    parser.set_defaults(**_cfg)

    args = parser.parse_args()

    if args.sub_lang is None and _sub_lang_default is not None:
        if isinstance(_sub_lang_default, str):
            args.sub_lang = [_sub_lang_default]
        else:
            args.sub_lang = list(_sub_lang_default)

    return args


def check_dependencies(skip_sync: bool = False) -> None:
    missing = []
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg  →  https://ffmpeg.org/download.html")
    if not skip_sync and not shutil.which("ffs"):
        missing.append("ffsubsync  →  pip install ffsubsync")
    if missing:
        print("Error: missing dependencies:", file=sys.stderr)
        for dep in missing:
            print(f"  - {dep}", file=sys.stderr)
        sys.exit(2)
    try:
        import whisper  # noqa: F401
    except ImportError:
        print("Error: openai-whisper not installed  →  pip install openai-whisper",
              file=sys.stderr)
        sys.exit(2)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_workers(requested: int | None, device: str) -> int:
    if requested is not None:
        return requested
    if device == "cuda":
        return 1
    return min(4, os.cpu_count() or 1)


@dataclasses.dataclass
class _VideoCache:
    """Transcriptions from a video's first subtitle pass, reused for subsequent subtitles."""
    segment_starts: list[int]
    transcriptions: list[str]
    audio_lang: str | None
    audio_track_index: int = 0
    audio_track_lang: str | None = None


_model_local = threading.local()


def _get_model(model_name: str, device: str):
    if not hasattr(_model_local, "model"):
        _model_local.model = transcribe.load_model(model_name, device=device)
    return _model_local.model


def _is_cross_language(audio_lang: str | None, subtitle_lang: str | None) -> bool:
    if not audio_lang or not subtitle_lang:
        return False
    return audio_lang.split("-")[0].lower() != subtitle_lang.split("-")[0].lower()


def _fmt_eta(secs: int) -> str:
    if secs < 60:
        return f"~{secs}s"
    if secs < 3600:
        return f"~{secs // 60}:{secs % 60:02d}"
    return f"~{secs // 3600}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"


def _determine_state(result: output.MatchResult) -> output.MatchState:
    if len(result.segments) == 0:
        return output.MatchState.UNSURE
    if not result.passed:
        return output.MatchState.FAIL
    if result.sync and result.sync.drift_detected:
        return output.MatchState.DRIFT
    return output.MatchState.PASS


def _should_fail(result: output.MatchResult, pass_unsure: bool) -> bool:
    if result.state == output.MatchState.PASS:
        return False
    if result.state == output.MatchState.UNSURE and pass_unsure:
        return False
    return True


_SUMMARY_THRESHOLD = 8


def _print_run_summary(pairs: list[tuple[Path, Path]], json_mode: bool) -> None:
    if json_mode:
        return
    n = len(pairs)
    if n == 0:
        return
    if n == 1:
        video, sub = pairs[0]
        print(f"Checking: {video.name} → {sub.name}", file=sys.stderr)
    elif n <= _SUMMARY_THRESHOLD:
        print(f"Checking {n} pairs:", file=sys.stderr)
        for video, sub in pairs:
            print(f"  {video.name} → {sub.name}", file=sys.stderr)
    else:
        n_videos = len({v for v, _ in pairs})
        print(
            f"Checking {n} pairs — {n_videos} video{'s' if n_videos != 1 else ''}, "
            f"{n} subtitle{'s' if n != 1 else ''}.",
            file=sys.stderr,
        )


_embed_local = threading.local()


def _get_embed_model():
    if not hasattr(_embed_local, "model"):
        try:
            _embed_local.model = embeddings.load_embedding_model()
        except ImportError:
            print(
                "Error: sentence-transformers not installed. "
                "Required for cross-language subtitle matching. "
                "Install with: pip install sentence-transformers",
                file=sys.stderr,
            )
            sys.exit(2)
    return _embed_local.model


def _score_group_parallel(
    video: Path,
    subs: list[Path],
    args: argparse.Namespace,
    model_name: str,
    device: str,
) -> list[output.BatchPairResult]:
    """Process all subtitles for one video in a single thread, sharing transcriptions."""
    model = _get_model(model_name, device)
    results = []
    cache: _VideoCache | None = None
    for sub in subs:
        try:
            result, new_cache = _score_pair(video, sub, args, model, show_progress=False,
                                            video_cache=cache)
            if result.state == output.MatchState.DRIFT and getattr(args, 'resync', False):
                synced_path = result.sync.synced_srt_path
                shutil.copy(synced_path, sub)
                synced_path.unlink(missing_ok=True)
                _ra = copy.copy(args)
                _ra.no_sync = True
                result, _ = _score_pair(video, sub, _ra, model, show_progress=False,
                                        video_cache=new_cache)
                result.state = _determine_state(result)
                result.resynced = True
            else:
                if result.sync and result.sync.synced_srt_path:
                    result.sync.synced_srt_path.unlink(missing_ok=True)
            if cache is None:
                cache = new_cache
            results.append(output.BatchPairResult(video=video, subtitle=sub,
                                                  result=result, error=None))
        except Exception as exc:
            results.append(output.BatchPairResult(video=video, subtitle=sub,
                                                  result=None, error=str(exc)))
    return results


def _score_pair(
    video: Path,
    subtitle_path: Path,
    args: argparse.Namespace,
    model,
    show_progress: bool = True,
    video_cache: _VideoCache | None = None,
    on_segment=None,
) -> tuple[output.MatchResult, _VideoCache]:
    subtitles = subtitle.parse(subtitle_path)
    subtitle_sample = " ".join(s.text for s in subtitles[:50])
    subtitle_lang = (language.detect_from_filename(subtitle_path) or
                     language.detect_from_text(subtitle_sample))

    _timing = getattr(args, 'timing', False) and show_progress
    _t_start = time.monotonic()

    def _phase(label: str, t_prev: float) -> float:
        t_now = time.monotonic()
        if _timing:
            print(f"  {label:<30} {t_now - t_prev:.2f}s", file=sys.stderr)
        return t_now

    # Resolve audio track once per video; subsequent subtitles reuse from cache.
    if video_cache is not None:
        audio_track_index = video_cache.audio_track_index
        audio_track_lang = video_cache.audio_track_lang
    else:
        audio_track_index = 0
        audio_track_lang: str | None = None
        _at_spec = getattr(args, 'audio_track', None)
        if _at_spec:
            audio_track_index, audio_track_lang = audio.resolve_audio_track(video, _at_spec)

    sync_result = None
    _sync_tmp: Path | None = None
    try:
        _t = time.monotonic()
        if not args.no_sync:
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
                _sync_tmp = Path(tmp.name)
                tmp.close()
                sync_result = sync.sync_subtitle(
                    video, subtitle_path, _sync_tmp,
                    drift_threshold=args.drift_threshold,
                    audio_track=audio_track_index,
                )
                subtitles = subtitle.parse(sync_result.synced_srt_path)
            except RuntimeError as exc:
                print(
                    f"Warning: ffsubsync failed ({exc}), proceeding without sync",
                    file=sys.stderr,
                )
        _t = _phase("sync", _t)

        # Phase 1: transcribe (first subtitle for this video) or reuse cache
        transcription_pairs: list[tuple[int, sampler.Segment, str]] = []
        new_cache: _VideoCache

        if video_cache is None:
            duration_ms = audio.get_duration_ms(video)
            segments = sampler.select_segments(subtitles, duration_ms, n=args.segments)
            n_seg = len(segments)
            audio_lang: str | None = None
            _t = _phase("segment selection", _t)

            for i, seg in enumerate(segments):
                if on_segment is not None:
                    on_segment(i + 1, n_seg)
                elif show_progress and not args.json:
                    print(f"  [{i + 1}/{n_seg}]", end="\r", file=sys.stderr)
                _t_seg = time.monotonic()
                try:
                    wav_path = audio.extract_segment(video, seg.start_ms, 30_000, audio_track=audio_track_index)
                    _t_extract = time.monotonic()
                    try:
                        trans = transcribe.transcribe_segment(model, wav_path)
                        _t_transcribe = time.monotonic()
                        if i == 0:
                            audio_lang = trans.language
                        transcription_pairs.append((i + 1, seg, trans.text))
                    finally:
                        wav_path.unlink(missing_ok=True)
                    if _timing:
                        print(
                            f"  seg {i+1}/{n_seg}  extract {_t_extract-_t_seg:.2f}s"
                            f"  transcribe {_t_transcribe-_t_extract:.2f}s",
                            file=sys.stderr,
                        )
                except Exception as exc:
                    print(f"Warning: segment {i + 1} failed: {exc}", file=sys.stderr)
            _t = _phase("transcription total", _t)

            if show_progress and not args.json:
                print()

            new_cache = _VideoCache(
                segment_starts=[seg.start_ms for _, seg, _ in transcription_pairs],
                transcriptions=[t for _, _, t in transcription_pairs],
                audio_lang=audio_lang,
                audio_track_index=audio_track_index,
                audio_track_lang=audio_track_lang,
            )
        else:
            # Reuse transcriptions from the first subtitle for this video.
            # Still re-syncs per subtitle (each has its own drift), then looks up
            # subtitle text at the pre-transcribed timestamps.
            audio_lang = video_cache.audio_lang
            cached_segs = sampler.segments_from_starts(subtitles, video_cache.segment_starts)
            transcription_pairs = [
                (i + 1, seg, trans)
                for i, (seg, trans) in enumerate(zip(cached_segs, video_cache.transcriptions))
            ]
            new_cache = video_cache
            _t = _phase("cache lookup", _t)

        # Phase 2: determine scoring mode
        cross_lang = _is_cross_language(audio_lang, subtitle_lang)
        embed_model = _get_embed_model() if cross_lang else None
        _t = _phase("embed model", _t)

        # Phase 3: score segments
        segment_results: list[output.SegmentResult] = []
        for idx, seg, trans_text in transcription_pairs:
            if cross_lang:
                score = embeddings.cross_language_score(seg.subtitle_text, trans_text, embed_model)
            else:
                score = compare.token_f1(seg.subtitle_text, trans_text)
            segment_results.append(output.SegmentResult(
                index=idx,
                start_ms=seg.start_ms,
                score=score.f1,
                wer=score.wer,
                subtitle_text=seg.subtitle_text,
                transcription=trans_text,
            ))

        _t = _phase("scoring", _t)

        lang_result = language.build_result(
            audio=audio_lang,
            subtitle_detected=language.detect_from_text(subtitle_sample),
            subtitle_filename=language.detect_from_filename(subtitle_path),
            video_meta=language.detect_from_video(video),
            expected=args.language,
        )
        _t = _phase("language detection", _t)

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
            args.cross_threshold
            if (cross_lang and args.cross_threshold is not None)
            else args.threshold
        )

        match_result = output.MatchResult(
            confidence=confidence,
            passed=confidence >= effective_threshold,
            threshold=effective_threshold,
            language=lang_result,
            sync=sync_result,
            segments=segment_results,
            model=args.model,
            cross_language=cross_lang,
            subtitle_language=subtitle_lang,
            audio_track_index=audio_track_index,
            audio_track_lang=audio_track_lang,
        )
        match_result.state = _determine_state(match_result)
        if _timing:
            print(f"  {'TOTAL':<30} {time.monotonic() - _t_start:.2f}s", file=sys.stderr)
        return match_result, new_cache
    except:
        if _sync_tmp is not None:
            _sync_tmp.unlink(missing_ok=True)
        raise


def _run_batch(
    args: argparse.Namespace,
    videos: list[Path],
    subtitles: list[Path],
    warn_missing: bool = True,
) -> int:
    from submatch import batch as _batch

    pairs_to_run = _batch.resolve_pairs(videos, subtitles, warn_missing=warn_missing)
    pairs_to_run = _batch.filter_pairs(
        pairs_to_run,
        sub_langs=args.sub_lang,
        glob_pattern=args.filter,
    )

    if not pairs_to_run:
        print("No video/subtitle pairs found.", file=sys.stderr)
        return 2

    _print_run_summary(pairs_to_run, args.json)

    device = _resolve_device(args.device)
    workers = _resolve_workers(args.workers, device)

    check_dependencies(skip_sync=args.no_sync)

    results: list[output.BatchPairResult] = []

    n_total = len(pairs_to_run)

    if not args.json:
        print(f"Loading model ({args.model})...", file=sys.stderr, flush=True)

    if workers == 1:
        model = transcribe.load_model(args.model, device=device)
        video_caches: dict[Path, _VideoCache] = {}
        _t0 = time.monotonic()
        _done = 0
        _ema_pair_time: float | None = None
        _EMA_ALPHA = 0.3
        _tty = not args.json and sys.stderr.isatty()

        def _print_progress(n: int, sub_name: str) -> None:
            if _ema_pair_time is not None:
                eta = _fmt_eta(int(_ema_pair_time * (n_total - _done)))
                pct = int(100 * _done / n_total)
                header = f"[{n}/{n_total}  {pct}%  {eta}]"
            else:
                header = f"[{n}/{n_total}]"
            line = f"{header} {sub_name}..."
            if _tty:
                print(line, end="\r", file=sys.stderr, flush=True)
            else:
                print(line, file=sys.stderr, flush=True)

        def _make_on_seg(pair_n: int, sub_name: str):
            def _cb(seg_idx: int, seg_total: int) -> None:
                if not _tty:
                    return
                if _ema_pair_time is not None:
                    pct = int(100 * _done / n_total)
                    eta = _fmt_eta(int(_ema_pair_time * (n_total - _done)))
                    header = f"[{pair_n}/{n_total}  {pct}%  {eta}]"
                else:
                    header = f"[{pair_n}/{n_total}]"
                print(f"{header} {sub_name}... {seg_idx}/{seg_total}", end="\r",
                      file=sys.stderr, flush=True)
            return _cb

        for i, (video, sub) in enumerate(pairs_to_run):
            if not args.json:
                _print_progress(i + 1, sub.name)
            _pair_t0 = time.monotonic()
            _on_seg = _make_on_seg(i + 1, sub.name)
            _match_result: output.MatchResult | None = None
            _error: str | None = None
            try:
                cache = video_caches.get(video)
                _match_result, new_cache = _score_pair(video, sub, args, model,
                                                       show_progress=False,
                                                       video_cache=cache,
                                                       on_segment=_on_seg)
                if _match_result.state == output.MatchState.DRIFT and getattr(args, 'resync', False):
                    synced_path = _match_result.sync.synced_srt_path
                    shutil.copy(synced_path, sub)
                    synced_path.unlink(missing_ok=True)
                    _ra = copy.copy(args)
                    _ra.no_sync = True
                    _match_result, _ = _score_pair(video, sub, _ra, model,
                                                   show_progress=False,
                                                   video_cache=new_cache)
                    _match_result.state = _determine_state(_match_result)
                    _match_result.resynced = True
                else:
                    if _match_result.sync and _match_result.sync.synced_srt_path:
                        _match_result.sync.synced_srt_path.unlink(missing_ok=True)
                if cache is None:
                    video_caches[video] = new_cache
                results.append(output.BatchPairResult(
                    video=video, subtitle=sub, result=_match_result, error=None,
                ))
            except Exception as exc:
                _error = str(exc)
                results.append(output.BatchPairResult(
                    video=video, subtitle=sub, result=None, error=_error,
                ))
            _took = time.monotonic() - _pair_t0
            if not args.json:
                if _tty:
                    print("\r\033[K", end="", file=sys.stderr, flush=True)
                if _error:
                    print(f"\nError: {video.name} / {sub.name}: {_error}", file=sys.stderr)
                elif args.compact:
                    output.print_batch_compact([results[-1]])
                else:
                    output.print_human(_match_result, verbose=args.verbose,
                                       video=video, subtitle=sub)
            if _ema_pair_time is None:
                _ema_pair_time = _took
            else:
                _ema_pair_time = _EMA_ALPHA * _took + (1 - _EMA_ALPHA) * _ema_pair_time
            _done += 1
    else:
        # Group pairs by video so each group shares one set of transcriptions.
        video_groups: dict[Path, list[Path]] = {}
        video_order: list[Path] = []
        for video, sub in pairs_to_run:
            if video not in video_groups:
                video_groups[video] = []
                video_order.append(video)
            video_groups[video].append(sub)

        _t0 = time.monotonic()
        _done = 0
        _done_lock = threading.Lock()
        _submit_times: dict[Path, float] = {}
        results_by_video: dict[Path, list[output.BatchPairResult]] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_video: dict[concurrent.futures.Future, Path] = {}
            for video in video_order:
                _submit_times[video] = time.monotonic()
                future = executor.submit(
                    _score_group_parallel, video, video_groups[video], args, args.model, device,
                )
                future_to_video[future] = video
            for future in concurrent.futures.as_completed(future_to_video):
                video = future_to_video[future]
                _ = time.monotonic() - _submit_times[video]
                try:
                    group = future.result()
                    results_by_video[video] = group
                except Exception as exc:
                    group = [
                        output.BatchPairResult(video=video, subtitle=sub,
                                               result=None, error=str(exc))
                        for sub in video_groups[video]
                    ]
                    results_by_video[video] = group
                if not args.json:
                    with _done_lock:
                        _done += len(group)
                        for pair_result in group:
                            if pair_result.error:
                                print(f"\nError: {pair_result.video.name} / "
                                      f"{pair_result.subtitle.name}: {pair_result.error}",
                                      file=sys.stderr)
                            elif args.compact:
                                output.print_batch_compact([pair_result])
                            else:
                                output.print_human(pair_result.result, verbose=args.verbose,
                                                   video=pair_result.video,
                                                   subtitle=pair_result.subtitle)
                        elapsed = time.monotonic() - _t0
                        pct = int(100 * _done / n_total)
                        if _done < n_total:
                            eta = _fmt_eta(int(elapsed / _done * (n_total - _done)))
                            print(f"[{_done}/{n_total}  {pct}%  {eta}]", file=sys.stderr)
                        else:
                            print(f"[{_done}/{n_total}  100%]", file=sys.stderr)
        results = []
        for video in video_order:
            results.extend(results_by_video.get(video, []))

    if args.delete_failures:
        for p in results:
            if p.result is not None and p.result.state == output.MatchState.FAIL:
                p.subtitle.unlink(missing_ok=True)
                if not args.json:
                    print(f"Deleted: {p.subtitle}")

    if args.json:
        print(output.format_batch_json(results))
    else:
        output.print_batch_summary(results)

    if any(p.error for p in results):
        return 2
    if any(p.result and _should_fail(p.result, args.pass_unsure) for p in results):
        return 1
    return 0


def main() -> None:
    _ensure_utf8_stdout()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        import static_ffmpeg
        static_ffmpeg.add_paths()

    args = parse_args()

    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        for p in missing:
            print(f"Error: not found: {p}", file=sys.stderr)
        sys.exit(2)

    from submatch import batch as _batch
    had_dirs = any(p.is_dir() for p in args.inputs)
    videos, subtitles = _batch.classify_inputs(args.inputs, recursive=not args.no_recursive)

    if args.poll and not args.watch:
        print("Warning: --poll and --interval have no effect without --watch", file=sys.stderr)

    if args.watch:
        if len(args.inputs) != 1 or not args.inputs[0].is_dir():
            print("Error: --watch requires exactly one directory argument", file=sys.stderr)
            sys.exit(2)
        check_dependencies(skip_sync=args.no_sync)
        from submatch import watch as _watch
        sys.exit(_watch.run_watch(args, args.inputs[0]))

    if not had_dirs and len(videos) == 1 and len(subtitles) == 1:
        args.video = videos[0]
        args.subtitle = subtitles[0]
    else:
        sys.exit(_run_batch(args, videos, subtitles, warn_missing=not had_dirs))

    if not args.json:
        print(f"Checking: {args.video.name} → {args.subtitle.name}", file=sys.stderr)

    check_dependencies(skip_sync=args.no_sync)

    if not audio.has_audio_track(args.video):
        print(f"Error: no audio track in {args.video}", file=sys.stderr)
        sys.exit(2)

    model = transcribe.load_model(args.model)

    result, _ = _score_pair(args.video, args.subtitle, args, model)

    if result.state == output.MatchState.DRIFT and args.resync:
        synced_path = result.sync.synced_srt_path
        shutil.copy(synced_path, args.subtitle)
        synced_path.unlink(missing_ok=True)
        _ra = copy.copy(args)
        _ra.no_sync = True
        result, _ = _score_pair(args.video, args.subtitle, _ra, model)
        result.state = _determine_state(result)
        result.resynced = True
        if not args.json:
            print(f"Subtitle resynced: {args.subtitle}")

    if args.keep_synced and result.sync and result.sync.synced_srt_path:
        kept = args.subtitle.with_stem(args.subtitle.stem + ".synced")
        shutil.copy(result.sync.synced_srt_path, kept)
        if not args.json:
            print(f"Synced subtitle saved to {kept}")

    # Cleanup synced temp file
    if result.sync and result.sync.synced_srt_path:
        result.sync.synced_srt_path.unlink(missing_ok=True)

    if args.json:
        print(output.format_json(result))
    else:
        output.print_human(result, verbose=args.verbose)

    if args.delete_failures and result.state == output.MatchState.FAIL:
        args.subtitle.unlink(missing_ok=True)
        if not args.json:
            print(f"Deleted: {args.subtitle}")

    sys.exit(0 if not _should_fail(result, args.pass_unsure) else 1)


if __name__ == "__main__":
    main()

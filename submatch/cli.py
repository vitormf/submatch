from __future__ import annotations
import argparse
import concurrent.futures
import dataclasses
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path

from tqdm import tqdm

from submatch import __version__
from submatch import audio, compare, embeddings, language, output, sampler, subtitle, sync, transcribe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="submatch",
        description="Verify a subtitle file matches the audio content of a video.",
    )
    parser.add_argument("video", type=Path,
                        help="video file to check, or a directory for batch mode")
    parser.add_argument("subtitle", type=Path, nargs="?", default=None,
                        help="subtitle file to verify, or a directory of subtitles for batch mode")
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
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="scan subdirectories when in batch mode")
    parser.add_argument("--sub-lang", action="append", dest="sub_lang", metavar="CODE",
                        help="only process subtitles matching this language prefix (repeatable, e.g. --sub-lang pt --sub-lang en)")
    parser.add_argument("--filter", metavar="GLOB",
                        help="only process subtitle files matching this glob pattern (e.g. '*.en.*')")
    parser.add_argument(
        "--device", choices=["cpu", "mps", "cuda", "auto"], default="auto",
        help="Whisper inference device (default: auto — MPS > CUDA > CPU)",
    )
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel pairs in batch mode (default: auto — 1 for GPU, up to 4 for CPU)")
    parser.add_argument("--delete-failures", action="store_true", dest="delete_failures",
                        help="delete subtitle files that fail the match check")
    parser.add_argument(
        "--cross-threshold", type=float, default=None, dest="cross_threshold",
        help="pass/fail threshold for cross-language pairs (default: same as --threshold)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser.parse_args()


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
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_workers(requested: int | None, device: str) -> int:
    if requested is not None:
        return requested
    if device in ("mps", "cuda"):
        return 1
    return min(4, os.cpu_count() or 1)


@dataclasses.dataclass
class _VideoCache:
    """Transcriptions from a video's first subtitle pass, reused for subsequent subtitles."""
    segment_starts: list[int]
    transcriptions: list[str]
    audio_lang: str | None


_model_local = threading.local()


def _get_model(model_name: str, device: str):
    if not hasattr(_model_local, "model"):
        _model_local.model = transcribe.load_model(model_name, device=device)
    return _model_local.model


def _is_cross_language(audio_lang: str | None, subtitle_lang: str | None) -> bool:
    if not audio_lang or not subtitle_lang:
        return False
    return audio_lang.split("-")[0].lower() != subtitle_lang.split("-")[0].lower()


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
    bar=None,
) -> list[output.BatchPairResult]:
    """Process all subtitles for one video in a single thread, sharing transcriptions."""
    model = _get_model(model_name, device)
    results = []
    cache: _VideoCache | None = None
    for sub in subs:
        try:
            result, cache = _score_pair(video, sub, args, model, show_progress=False,
                                        video_cache=cache)
            results.append(output.BatchPairResult(video=video, subtitle=sub,
                                                  result=result, error=None))
        except Exception as exc:
            results.append(output.BatchPairResult(video=video, subtitle=sub,
                                                  result=None, error=str(exc)))
        if bar is not None:
            bar.update(1)
    return results


def _score_pair(
    video: Path,
    subtitle_path: Path,
    args: argparse.Namespace,
    model,
    show_progress: bool = True,
    bar=None,
    video_cache: _VideoCache | None = None,
) -> tuple[output.MatchResult, _VideoCache]:
    subtitles = subtitle.parse(subtitle_path)
    subtitle_sample = " ".join(s.text for s in subtitles[:50])
    subtitle_lang = (language.detect_from_filename(subtitle_path) or
                     language.detect_from_text(subtitle_sample))

    sync_result = None
    _sync_tmp: Path | None = None
    try:
        if not args.no_sync:
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
                _sync_tmp = Path(tmp.name)
                tmp.close()
                if bar is not None:
                    bar.set_description(f"sync  [{video.name} / {subtitle_path.name}]")
                sync_result = sync.sync_subtitle(video, subtitle_path, _sync_tmp)
                subtitles = subtitle.parse(sync_result.synced_srt_path)
            except RuntimeError as exc:
                print(f"Warning: ffsubsync failed ({exc}), proceeding without sync",
                      file=sys.stderr)

        # Phase 1: transcribe (first subtitle for this video) or reuse cache
        transcription_pairs: list[tuple[int, sampler.Segment, str]] = []
        new_cache: _VideoCache

        if video_cache is None:
            duration_ms = audio.get_duration_ms(video)
            segments = sampler.select_segments(subtitles, duration_ms, n=args.segments)
            audio_lang: str | None = None

            for i, seg in enumerate(segments):
                if bar is not None:
                    bar.set_description(
                        f"[{i + 1}/{len(segments)}] {video.name} / {subtitle_path.name}"
                    )
                elif show_progress and not args.json:
                    print(f"  Transcribing segment {i + 1}/{len(segments)}...", end="\r")
                try:
                    wav_path = audio.extract_segment(video, seg.start_ms, 30_000)
                    try:
                        trans = transcribe.transcribe_segment(model, wav_path)
                        if i == 0:
                            audio_lang = trans.language
                        transcription_pairs.append((i + 1, seg, trans.text))
                    finally:
                        wav_path.unlink(missing_ok=True)
                except Exception as exc:
                    print(f"\nWarning: segment {i + 1} failed: {exc}", file=sys.stderr)

            if show_progress and not args.json:
                print()

            new_cache = _VideoCache(
                segment_starts=[seg.start_ms for _, seg, _ in transcription_pairs],
                transcriptions=[t for _, _, t in transcription_pairs],
                audio_lang=audio_lang,
            )
        else:
            # Reuse transcriptions from the first subtitle for this video.
            # Still re-syncs per subtitle (each has its own drift), then looks up
            # subtitle text at the pre-transcribed timestamps.
            if bar is not None:
                bar.set_description(f"scoring [{video.name} / {subtitle_path.name}]")
            audio_lang = video_cache.audio_lang
            cached_segs = sampler.segments_from_starts(subtitles, video_cache.segment_starts)
            transcription_pairs = [
                (i + 1, seg, trans)
                for i, (seg, trans) in enumerate(zip(cached_segs, video_cache.transcriptions))
            ]
            new_cache = video_cache

        # Phase 2: determine scoring mode
        cross_lang = _is_cross_language(audio_lang, subtitle_lang)
        embed_model = _get_embed_model() if cross_lang else None

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

        lang_result = language.build_result(
            audio=audio_lang,
            subtitle_detected=language.detect_from_text(subtitle_sample),
            subtitle_filename=language.detect_from_filename(subtitle_path),
            video_meta=language.detect_from_video(video),
            expected=args.language,
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
            args.cross_threshold
            if (cross_lang and args.cross_threshold is not None)
            else args.threshold
        )

        return output.MatchResult(
            confidence=confidence,
            passed=confidence >= effective_threshold,
            threshold=effective_threshold,
            language=lang_result,
            sync=sync_result,
            segments=segment_results,
            model=args.model,
            cross_language=cross_lang,
            subtitle_language=subtitle_lang,
        ), new_cache
    finally:
        if _sync_tmp is not None:
            _sync_tmp.unlink(missing_ok=True)


def _run_batch(args: argparse.Namespace) -> int:
    from submatch import batch as _batch

    if args.video.is_dir():
        pairs_to_run = (
            _batch.find_pairs_recursive(args.video)
            if args.recursive
            else _batch.find_pairs(args.video)
        )
    else:
        if not args.subtitle.is_dir():
            print(f"Error: expected a directory for subtitle argument, got: {args.subtitle}",
                  file=sys.stderr)
            return 2
        candidates = (
            _batch.find_subtitle_candidates_recursive(args.subtitle)
            if args.recursive
            else _batch.find_subtitle_candidates(args.subtitle)
        )
        pairs_to_run = [(args.video, c) for c in candidates]

    pairs_to_run = _batch.filter_pairs(
        pairs_to_run,
        sub_langs=args.sub_lang,
        glob_pattern=args.filter,
    )

    if not pairs_to_run:
        print("No video/subtitle pairs found.", file=sys.stderr)
        return 2

    device = _resolve_device(args.device)
    workers = _resolve_workers(args.workers, device)

    if workers > 1 and device in ("mps", "cuda"):
        print(
            f"Warning: --workers {workers} with --device {device} may cause GPU "
            "contention and hangs. Use --device cpu for parallel processing, or "
            "--workers 1 to keep GPU acceleration.",
            file=sys.stderr,
        )

    check_dependencies(skip_sync=args.no_sync)

    results: list[output.BatchPairResult] = []

    if workers == 1:
        if not args.json:
            print(f"Loading Whisper model '{args.model}'...")
        model = transcribe.load_model(args.model, device=device)
        bar = tqdm(
            total=len(pairs_to_run),
            unit="pair",
            disable=args.json or not sys.stderr.isatty(),
        )
        video_caches: dict[Path, _VideoCache] = {}
        for video, sub in pairs_to_run:
            bar.set_description(f"Batch [{video.name} / {sub.name}]")
            try:
                cache = video_caches.get(video)
                match_result, new_cache = _score_pair(video, sub, args, model,
                                                      show_progress=False, bar=bar,
                                                      video_cache=cache)
                if cache is None:
                    video_caches[video] = new_cache
                results.append(output.BatchPairResult(
                    video=video, subtitle=sub, result=match_result, error=None,
                ))
            except Exception as exc:
                results.append(output.BatchPairResult(
                    video=video, subtitle=sub, result=None, error=str(exc),
                ))
            bar.update(1)
        bar.close()
    else:
        # Group pairs by video so each group shares one set of transcriptions.
        video_groups: dict[Path, list[Path]] = {}
        video_order: list[Path] = []
        for video, sub in pairs_to_run:
            if video not in video_groups:
                video_groups[video] = []
                video_order.append(video)
            video_groups[video].append(sub)

        bar = tqdm(
            total=len(pairs_to_run),
            unit="pair",
            disable=args.json or not sys.stderr.isatty(),
        )
        results_by_video: dict[Path, list[output.BatchPairResult]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_video: dict[concurrent.futures.Future, Path] = {}
            for video in video_order:
                future = executor.submit(
                    _score_group_parallel, video, video_groups[video], args, args.model, device, bar,
                )
                future_to_video[future] = video
            for future in concurrent.futures.as_completed(future_to_video):
                video = future_to_video[future]
                try:
                    results_by_video[video] = future.result()
                except Exception as exc:
                    results_by_video[video] = [
                        output.BatchPairResult(video=video, subtitle=sub,
                                               result=None, error=str(exc))
                        for sub in video_groups[video]
                    ]
        bar.close()
        results = []
        for video in video_order:
            results.extend(results_by_video.get(video, []))

    if args.delete_failures:
        for p in results:
            if p.result is not None and not p.result.passed:
                p.subtitle.unlink(missing_ok=True)
                if not args.json:
                    tqdm.write(f"Deleted: {p.subtitle}")

    if args.json:
        print(output.format_batch_json(results))
    elif args.compact:
        output.print_batch_compact(results)
        output.print_batch_summary(results)
    else:
        for p in results:
            if p.error:
                print(f"\nError: {p.video.name} / {p.subtitle.name}: {p.error}",
                      file=sys.stderr)
            else:
                output.print_human(p.result, verbose=args.verbose,
                                   video=p.video, subtitle=p.subtitle)
        output.print_batch_summary(results)

    if any(p.error for p in results):
        return 2
    if any(not p.result.passed for p in results):
        return 1
    return 0


def main() -> None:
    args = parse_args()

    if args.video.is_dir() or (args.subtitle is not None and args.subtitle.is_dir()):
        sys.exit(_run_batch(args))

    if not args.video.exists():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(2)
    if args.subtitle is None or not args.subtitle.exists():
        print(f"Error: subtitle not found: {args.subtitle}", file=sys.stderr)
        sys.exit(2)

    check_dependencies(skip_sync=args.no_sync)

    if not audio.has_audio_track(args.video):
        print(f"Error: no audio track in {args.video}", file=sys.stderr)
        sys.exit(2)

    if not args.json:
        print(f"Loading Whisper model '{args.model}'...")
    model = transcribe.load_model(args.model)

    result, _ = _score_pair(args.video, args.subtitle, args, model)

    if args.keep_synced and result.sync:
        kept = args.subtitle.with_stem(args.subtitle.stem + ".synced")
        shutil.copy(result.sync.synced_srt_path, kept)
        if not args.json:
            print(f"Synced subtitle saved to {kept}")

    if args.json:
        print(output.format_json(result))
    else:
        output.print_human(result, verbose=args.verbose)

    if args.delete_failures and not result.passed:
        args.subtitle.unlink(missing_ok=True)
        if not args.json:
            print(f"Deleted: {args.subtitle}")

    sys.exit(0 if result.passed else 1)

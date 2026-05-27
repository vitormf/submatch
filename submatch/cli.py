from __future__ import annotations
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

from submatch import __version__
from submatch import audio, compare, language, output, sampler, subtitle, sync, transcribe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="submatch",
        description="Verify a subtitle file matches the audio content of a video.",
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("subtitle", type=Path, nargs="?", default=None)
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
    )
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--language", default=None)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--keep-synced", action="store_true")
    parser.add_argument("--recursive", "-r", action="store_true")
    parser.add_argument("--sub-lang", action="append", dest="sub_lang", metavar="CODE")
    parser.add_argument("--filter", metavar="GLOB")
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


def _score_pair(
    video: Path,
    subtitle_path: Path,
    args: argparse.Namespace,
    model,
    show_progress: bool = True,
) -> output.MatchResult:
    subtitles = subtitle.parse(subtitle_path)
    subtitle_sample = " ".join(s.text for s in subtitles[:50])

    sync_result = None
    _sync_tmp: Path | None = None
    try:
        if not args.no_sync:
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
                _sync_tmp = Path(tmp.name)
                tmp.close()
                sync_result = sync.sync_subtitle(video, subtitle_path, _sync_tmp)
                subtitles = subtitle.parse(sync_result.synced_srt_path)
            except RuntimeError as exc:
                print(f"Warning: ffsubsync failed ({exc}), proceeding without sync",
                      file=sys.stderr)

        duration_ms = audio.get_duration_ms(video)
        segments = sampler.select_segments(subtitles, duration_ms, n=args.segments)

        segment_results: list[output.SegmentResult] = []
        successful_segs: list[sampler.Segment] = []
        audio_lang: str | None = None

        for i, seg in enumerate(segments):
            if show_progress and not args.json:
                print(f"  Transcribing segment {i + 1}/{len(segments)}...", end="\r")
            try:
                wav_path = audio.extract_segment(video, seg.start_ms, 30_000)
                try:
                    trans = transcribe.transcribe_segment(model, wav_path)
                    if i == 0:
                        audio_lang = trans.language
                    score = compare.token_f1(seg.subtitle_text, trans.text)
                    segment_results.append(output.SegmentResult(
                        index=i + 1,
                        start_ms=seg.start_ms,
                        score=score.f1,
                        wer=score.wer,
                        subtitle_text=seg.subtitle_text,
                        transcription=trans.text,
                    ))
                    successful_segs.append(seg)
                finally:
                    wav_path.unlink(missing_ok=True)
            except Exception as exc:
                print(f"\nWarning: segment {i + 1} failed: {exc}", file=sys.stderr)

        if show_progress and not args.json:
            print()

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
                subtitle_tokens=len(seg.subtitle_text.split()),
            )
            for sr, seg in zip(segment_results, successful_segs)
        ]
        confidence = compare.aggregate(seg_scores)

        return output.MatchResult(
            confidence=confidence,
            passed=confidence >= args.threshold,
            threshold=args.threshold,
            language=lang_result,
            sync=sync_result,
            segments=segment_results,
            model=args.model,
        )
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

    check_dependencies(skip_sync=args.no_sync)

    if not args.json:
        print(f"Loading Whisper model '{args.model}'...")
    model = transcribe.load_model(args.model)

    results: list[output.BatchPairResult] = []
    bar = tqdm(
        total=len(pairs_to_run),
        unit="pair",
        disable=args.json or not sys.stderr.isatty(),
    )
    for video, sub in pairs_to_run:
        bar.set_description(f"Batch [{video.name} / {sub.name}]")
        if not args.json:
            tqdm.write(f"  Processing {video.name} / {sub.name} ...")
        try:
            match_result = _score_pair(video, sub, args, model, show_progress=False)
            results.append(output.BatchPairResult(
                video=video, subtitle=sub, result=match_result, error=None,
            ))
        except Exception as exc:
            results.append(output.BatchPairResult(
                video=video, subtitle=sub, result=None, error=str(exc),
            ))
        bar.update(1)
    bar.close()

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
                output.print_human(p.result, verbose=args.verbose)
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

    result = _score_pair(args.video, args.subtitle, args, model)

    if args.keep_synced and result.sync:
        kept = args.subtitle.with_stem(args.subtitle.stem + ".synced")
        shutil.copy(result.sync.synced_srt_path, kept)
        if not args.json:
            print(f"Synced subtitle saved to {kept}")

    if args.json:
        print(output.format_json(result))
    else:
        output.print_human(result, verbose=args.verbose)

    sys.exit(0 if result.passed else 1)

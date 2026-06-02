from __future__ import annotations
import argparse
import dataclasses
import shutil
import sys
import time
from pathlib import Path

from submatch import __version__
from submatch import audio, gpu, output, telemetry
from submatch import cache as _cache_module
from submatch import pipeline as _pipeline


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
    parser.add_argument("inputs", type=Path, nargs="*",
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
    parser.add_argument("--json", default=None, metavar="FILE",
                        help="write JSON report to FILE")
    parser.add_argument("--csv", default=None, metavar="FILE",
                        help="write CSV report to FILE")
    parser.add_argument("--html", default=None, metavar="FILE",
                        help="write self-contained HTML report to FILE")
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
    parser.add_argument("--embedded", action="store_true",
                        help="score embedded subtitle tracks in the video container")
    parser.add_argument("--watch", action="store_true",
                        help="monitor a directory for new video/subtitle pairs and score them as they appear")
    parser.add_argument("--poll", action="store_true",
                        help="use polling instead of native filesystem events (for network mounts)")
    parser.add_argument("--interval", type=int, default=10, metavar="N",
                        help="seconds between polls in --poll mode (default: 10)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--no-cache", action="store_true", dest="no_cache",
                        help="disable transcription cache and use subtitle-driven segment selection")
    parser.add_argument("--clear-cache", action="store_true", dest="clear_cache",
                        help="delete all cached transcriptions and exit")

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
    warning = gpu.check_gpu_mismatch()
    if warning:
        print(warning, file=sys.stderr)


def _fmt_eta(secs: int) -> str:
    if secs < 60:
        return f"~{secs}s"
    if secs < 3600:
        return f"~{secs // 60}:{secs % 60:02d}"
    return f"~{secs // 3600}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"


def _should_fail(result: output.MatchResult, pass_unsure: bool) -> bool:
    if result.state == output.MatchState.PASS:
        return False
    if result.state == output.MatchState.UNSURE and pass_unsure:
        return False
    return True


_SUMMARY_THRESHOLD = 8


def _print_run_summary(pairs: list[tuple[Path, Path]]) -> None:
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


def _write_reports(results: list[output.BatchPairResult], args: argparse.Namespace) -> None:
    from submatch import report as _report
    if args.json:
        _report.write_json(results, args.json)
    if args.csv:
        _report.write_csv(results, args.csv)
    if args.html:
        _report.write_html(results, args.html)


def _args_to_config(args: argparse.Namespace) -> _pipeline.PipelineConfig:
    return _pipeline.PipelineConfig(
        model=args.model,
        threshold=args.threshold,
        cross_threshold=getattr(args, "cross_threshold", None),
        segments=args.segments,
        language=args.language,
        sync=not args.no_sync,
        drift_threshold=args.drift_threshold,
        device=args.device,
        audio_track=getattr(args, "audio_track", None),
        workers=args.workers,
        use_cache=not getattr(args, "no_cache", False),
        cache_dir=Path(args.cache_dir).expanduser() if getattr(args, "cache_dir", None) else None,
        cache_ttl_days=getattr(args, "cache_ttl_days", None),
        cache_max_mb=getattr(args, "cache_max_mb", None),
        resync=getattr(args, "resync", False),
        verbose=args.verbose,
    )


def _run_batch(
    args: argparse.Namespace,
    videos: list[Path],
    subtitles: list[Path],
    warn_missing: bool = True,
    pairs: list[tuple[Path, Path]] | None = None,
) -> int:
    if pairs is not None:
        pairs_to_run = pairs
    else:
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

    _print_run_summary(pairs_to_run)
    check_dependencies(skip_sync=args.no_sync)

    n_total = len(pairs_to_run)
    _tty = sys.stderr.isatty()
    _pair_idx = [0]
    _ema_pair_time: list[float | None] = [None]
    _pair_start: list[float] = [time.monotonic()]
    _EMA_ALPHA = 0.3

    print(f"Loading model ({args.model})...", file=sys.stderr, flush=True)

    def _on_segment(seg_idx: int, seg_total: int) -> None:
        if not _tty:
            return
        idx = _pair_idx[0]
        if idx >= len(pairs_to_run):
            return
        sub_name = pairs_to_run[idx][1].name
        pair_n = idx + 1
        if _ema_pair_time[0] is not None:
            pct = int(100 * idx / n_total)
            eta = _fmt_eta(int(_ema_pair_time[0] * (n_total - idx)))
            header = f"[{pair_n}/{n_total}  {pct}%  {eta}]"
        else:
            header = f"[{pair_n}/{n_total}]"
        print(f"{header} {sub_name}... {seg_idx}/{seg_total}", end="\r",
              file=sys.stderr, flush=True)

    def _on_pair_complete(pair_result: output.BatchPairResult) -> None:
        took = time.monotonic() - _pair_start[0]
        if _tty:
            print("\r\033[K", end="", file=sys.stderr, flush=True)
        if pair_result.error:
            print(f"\nError: {pair_result.video.name} / "
                  f"{pair_result.subtitle.name}: {pair_result.error}",
                  file=sys.stderr)
        elif args.compact:
            output.print_batch_compact([pair_result])
        else:
            output.print_human(pair_result.result, verbose=args.verbose,
                               video=pair_result.video, subtitle=pair_result.subtitle)
        if _ema_pair_time[0] is None:
            _ema_pair_time[0] = took
        else:
            _ema_pair_time[0] = _EMA_ALPHA * took + (1 - _EMA_ALPHA) * _ema_pair_time[0]
        _pair_idx[0] += 1
        _pair_start[0] = time.monotonic()

    config = dataclasses.replace(
        _args_to_config(args),
        on_segment=_on_segment,
        on_pair_complete=_on_pair_complete,
    )

    try:
        results = _pipeline.run_batch(pairs_to_run, config)
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.delete_failures:
        for p in results:
            if p.result is not None and p.result.state == output.MatchState.FAIL:
                p.subtitle.unlink(missing_ok=True)
                print(f"Deleted: {p.subtitle}")

    output.print_batch_summary(results)
    _write_reports(results, args)

    if any(p.error for p in results):
        return 2
    if any(p.result and _should_fail(p.result, args.pass_unsure) for p in results):
        return 1
    return 0


def _run_embedded(
    args: argparse.Namespace,
    videos: list[Path],
) -> int:
    import tempfile
    from submatch import embedded as _embedded
    from submatch.audio import _lang_match

    tmp_dir = Path(tempfile.mkdtemp(prefix="submatch_embedded_"))
    pairs: list[tuple[Path, Path]] = []

    try:
        for video in videos:
            try:
                tracks = _embedded.list_subtitle_tracks(video)
            except Exception as exc:
                telemetry.capture(exc)
                print(
                    f"Warning: could not list subtitle tracks for {video.name}: {exc}",
                    file=sys.stderr,
                )
                continue

            if args.sub_lang:
                tracks = [
                    t for t in tracks
                    if t["lang"] is None
                    or any(_lang_match(c, t["lang"]) for c in args.sub_lang)
                ]

            vid_dir = tmp_dir / video.stem
            vid_dir.mkdir(exist_ok=True)

            try:
                extracted = _embedded.extract_all_subtitle_tracks(video, tracks, vid_dir)
                for idx, dest in extracted.items():
                    pairs.append((video, dest))
            except Exception as exc:
                telemetry.capture(exc)
                print(
                    f"Warning: could not extract subtitle tracks from {video.name}: {exc}",
                    file=sys.stderr,
                )

        if not pairs:
            print("No embedded subtitle tracks found.", file=sys.stderr)
            return 2

        return _run_batch(args, [], [], pairs=pairs)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    _ensure_utf8_stdout()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        import static_ffmpeg
        static_ffmpeg.add_paths()

    args = parse_args()
    telemetry.init(args)

    if getattr(args, 'clear_cache', False):
        _cfg = _pipeline._cache_config(_args_to_config(args))
        count = _cache_module.clear(_cfg["dir"])
        print(f"Cleared {count} cached transcription(s).")
        sys.exit(0)

    if not args.inputs:
        print("Error: the following arguments are required: inputs", file=sys.stderr)
        sys.exit(2)

    if args.embedded and (args.resync or args.keep_synced):
        print(
            "Error: --embedded is incompatible with --resync and --keep-synced",
            file=sys.stderr,
        )
        sys.exit(2)

    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        for p in missing:
            print(f"Error: not found: {p}", file=sys.stderr)
        sys.exit(2)

    from submatch import batch as _batch
    had_dirs = any(p.is_dir() for p in args.inputs)
    videos, subtitles = _batch.classify_inputs(args.inputs, recursive=not args.no_recursive)

    if args.embedded:
        check_dependencies(skip_sync=args.no_sync)
        telemetry.set_mode("embedded")
        sys.exit(_run_embedded(args, videos))

    if args.poll and not args.watch:
        print("Warning: --poll and --interval have no effect without --watch", file=sys.stderr)

    if args.watch:
        if len(args.inputs) != 1 or not args.inputs[0].is_dir():
            print("Error: --watch requires exactly one directory argument", file=sys.stderr)
            sys.exit(2)
        check_dependencies(skip_sync=args.no_sync)
        from submatch import watch as _watch
        telemetry.set_mode("watch")
        sys.exit(_watch.run_watch(args, args.inputs[0]))

    if not had_dirs and len(videos) == 1 and len(subtitles) == 1:
        args.video = videos[0]
        args.subtitle = subtitles[0]
        telemetry.set_mode("single")
    else:
        telemetry.set_mode("batch")
        sys.exit(_run_batch(args, videos, subtitles, warn_missing=not had_dirs))

    print(f"Checking: {args.video.name} → {args.subtitle.name}", file=sys.stderr)

    check_dependencies(skip_sync=args.no_sync)

    if not audio.has_audio_track(args.video):
        print(f"Error: no audio track in {args.video}", file=sys.stderr)
        sys.exit(2)

    config = _args_to_config(args)

    try:
        result = _pipeline.run(args.video, args.subtitle, config)
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    if result.resynced:
        print(f"Subtitle resynced: {args.subtitle}")

    if args.keep_synced and result.sync and result.sync.synced_srt_path:
        kept = args.subtitle.with_stem(args.subtitle.stem + ".synced")
        shutil.copy(result.sync.synced_srt_path, kept)
        print(f"Synced subtitle saved to {kept}")

    if result.sync and result.sync.synced_srt_path:
        result.sync.synced_srt_path.unlink(missing_ok=True)

    output.print_human(result, verbose=args.verbose)
    _write_reports(
        [output.BatchPairResult(video=args.video, subtitle=args.subtitle,
                                result=result, error=None)],
        args,
    )

    if args.delete_failures and result.state == output.MatchState.FAIL:
        args.subtitle.unlink(missing_ok=True)
        print(f"Deleted: {args.subtitle}")

    sys.exit(0 if not _should_fail(result, args.pass_unsure) else 1)


if __name__ == "__main__":
    main()

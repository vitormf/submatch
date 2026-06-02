from __future__ import annotations
import argparse
import dataclasses
import shutil
import sys
import time
from pathlib import Path

from submatch import audio, gpu, output, telemetry
from submatch import cache as _cache_module
from submatch import pipeline as _pipeline
from submatch import scoring as _scoring
from submatch.args import _args_to_config, parse_args
from submatch.types import BatchPairResult, MatchState


def _ensure_utf8_stdout() -> None:
    """Switch stdout/stderr to UTF-8 on Windows to prevent UnicodeEncodeError when piped."""
    if sys.platform == 'win32':
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')


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


_SUMMARY_THRESHOLD = 8


@dataclasses.dataclass
class _ProgressTracker:
    n_total: int
    pair_idx: int = 0
    ema_pair_time: float | None = None
    pair_start: float = dataclasses.field(default_factory=time.monotonic)

    _EMA_ALPHA = 0.3

    def eta_header(self) -> str:
        pair_n = self.pair_idx + 1
        if self.ema_pair_time is not None:
            pct = int(100 * self.pair_idx / self.n_total)
            eta = _fmt_eta(int(self.ema_pair_time * (self.n_total - self.pair_idx)))
            return f"[{pair_n}/{self.n_total}  {pct}%  {eta}]"
        return f"[{pair_n}/{self.n_total}]"

    def advance(self, took: float) -> None:
        if self.ema_pair_time is None:
            self.ema_pair_time = took
        else:
            self.ema_pair_time = self._EMA_ALPHA * took + (1 - self._EMA_ALPHA) * self.ema_pair_time
        self.pair_idx += 1
        self.pair_start = time.monotonic()


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


def _write_reports(results: list[BatchPairResult], args: argparse.Namespace) -> None:
    from submatch import report as _report
    if args.json:
        _report.write_json(results, args.json)
    if args.csv:
        _report.write_csv(results, args.csv)
    if args.html:
        _report.write_html(results, args.html)


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
    tracker = _ProgressTracker(n_total=n_total)

    print(f"Loading model ({args.model})...", file=sys.stderr, flush=True)

    def _on_segment(seg_idx: int, seg_total: int) -> None:
        if not _tty:
            return
        idx = tracker.pair_idx
        if idx >= len(pairs_to_run):  # pragma: no cover
            return  # pragma: no cover
        sub_name = pairs_to_run[idx][1].name
        print(f"{tracker.eta_header()} {sub_name}... {seg_idx}/{seg_total}", end="\r",
              file=sys.stderr, flush=True)

    def _on_pair_complete(pair_result: BatchPairResult) -> None:
        took = time.monotonic() - tracker.pair_start
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
        if (config.delete_failures and pair_result.result and
                pair_result.result.state == MatchState.FAIL):
            print(f"Deleted: {pair_result.subtitle}")
        tracker.advance(took)

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

    output.print_batch_summary(results)
    _write_reports(results, args)

    if any(p.error for p in results):
        return 2
    if any(p.result and not p.result.passed for p in results):
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
        _cfg = _scoring._cache_config(_args_to_config(args))
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

    if config.keep_synced and result.sync:
        kept = args.subtitle.with_stem(args.subtitle.stem + ".synced")
        print(f"Synced subtitle saved to {kept}")
    if config.delete_failures and result.state == MatchState.FAIL:
        print(f"Deleted: {args.subtitle}")

    output.print_human(result, verbose=args.verbose)
    _write_reports(
        [BatchPairResult(video=args.video, subtitle=args.subtitle,
                         result=result, error=None)],
        args,
    )

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":  # pragma: no cover
    main()

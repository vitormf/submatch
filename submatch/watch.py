from __future__ import annotations
import sys
import threading
import time
from pathlib import Path
import argparse

from watchdog.events import FileSystemEventHandler

from submatch import batch
from submatch.batch import SUBTITLE_EXTENSIONS, VIDEO_EXTENSIONS


def _find_pairs(directory: Path, recursive: bool) -> list[tuple[Path, Path]]:
    if recursive:
        return batch.find_pairs_recursive(directory)
    return batch.find_pairs(directory)


def _score_and_print(video: Path, sub: Path, config) -> None:
    from submatch import pipeline as _pipeline, output
    try:
        result = _pipeline.run(video, sub, config)
        if result.sync and result.sync.synced_srt_path:
            result.sync.synced_srt_path.unlink(missing_ok=True)
        output.print_human(result, verbose=config.verbose, video=video, subtitle=sub)
    except Exception as exc:
        print(f"Error: {video.name} / {sub.name}: {exc}", file=sys.stderr)


def _score_existing(
    args: argparse.Namespace, directory: Path, config
) -> set[tuple[Path, Path]]:
    recursive = not getattr(args, "no_recursive", False)
    pairs = _find_pairs(directory, recursive)
    pairs = batch.filter_pairs(
        pairs,
        sub_langs=getattr(args, "sub_lang", None),
        glob_pattern=getattr(args, "filter", None),
    )
    known: set[tuple[Path, Path]] = set()
    for video, sub in pairs:
        _score_and_print(video, sub, config)
        known.add((video, sub))
    return known


def _poll_loop(
    config,
    directory: Path,
    known_pairs: set[tuple[Path, Path]],
    args: argparse.Namespace,
    interval: int,
) -> None:
    recursive = not getattr(args, "no_recursive", False)
    while True:
        time.sleep(interval)
        pairs = _find_pairs(directory, recursive)
        pairs = batch.filter_pairs(
            pairs,
            sub_langs=getattr(args, "sub_lang", None),
            glob_pattern=getattr(args, "filter", None),
        )
        for video, sub in pairs:
            if (video, sub) not in known_pairs:
                _score_and_print(video, sub, config)
                known_pairs.add((video, sub))


class _SubtitleEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        config,
        known_pairs: set[tuple[Path, Path]],
        lock: threading.Lock,
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.config = config
        self.known_pairs = known_pairs
        self.lock = lock
        self.args = args

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        src = Path(event.src_path)
        ext = src.suffix.lower()
        if ext in SUBTITLE_EXTENSIONS:
            all_pairs = batch.find_pairs(src.parent)
            candidates = [(v, s) for v, s in all_pairs if s == src]
        elif ext in VIDEO_EXTENSIONS:
            all_pairs = batch.find_pairs(src.parent)
            candidates = [(v, s) for v, s in all_pairs if v == src]
        else:
            return
        candidates = batch.filter_pairs(
            candidates,
            sub_langs=getattr(self.args, "sub_lang", None),
            glob_pattern=getattr(self.args, "filter", None),
        )
        for video, sub in candidates:
            with self.lock:
                if (video, sub) in self.known_pairs:
                    continue
                self.known_pairs.add((video, sub))
            _score_and_print(video, sub, self.config)


def _native_watch(
    config,
    directory: Path,
    known_pairs: set[tuple[Path, Path]],
    args: argparse.Namespace,
) -> None:
    from watchdog.observers import Observer

    recursive = not getattr(args, "no_recursive", False)
    lock = threading.Lock()
    handler = _SubtitleEventHandler(config, known_pairs, lock, args)
    observer = Observer()
    observer.schedule(handler, str(directory), recursive=recursive)
    observer.start()  # raises OSError on unsupported filesystem
    try:
        while observer.is_alive():
            observer.join(timeout=1)
    finally:
        observer.stop()
        observer.join()


def run_watch(args: argparse.Namespace, directory: Path) -> int:
    from submatch import pipeline as _pipeline
    from submatch.cli import _args_to_config

    print(f"Watching {directory} — press Ctrl+C to stop", file=sys.stderr)
    device = _pipeline._resolve_device(args.device)
    print(f"Loading model ({args.model})...", file=sys.stderr, flush=True)
    _pipeline._get_model(args.model, device)

    config = _args_to_config(args)
    known_pairs = _score_existing(args, directory, config)

    try:
        if getattr(args, "poll", False):
            _poll_loop(config, directory, known_pairs, args, args.interval)
        else:
            try:
                _native_watch(config, directory, known_pairs, args)
            except OSError as exc:
                print(
                    f"Error: filesystem does not support native events ({exc}). Use --poll.",
                    file=sys.stderr,
                )
                return 2
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)

    return 0

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from submatch import __version__
from submatch import pipeline as _pipeline


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
        help="pass/fail threshold for cross-language pairs (default: 0.20)",
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
        pass_unsure=getattr(args, "pass_unsure", False),
        keep_synced=getattr(args, "keep_synced", False),
        delete_failures=getattr(args, "delete_failures", False),
        verbose=args.verbose,
    )

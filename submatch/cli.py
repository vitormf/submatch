from __future__ import annotations
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from submatch import __version__
from submatch import audio, compare, language, output, sampler, srt, sync, transcribe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="submatch",
        description="Verify a subtitle file matches the audio content of a video.",
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("subtitle", type=Path)
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
    )
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--language", default=None)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--keep-synced", action="store_true")
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


def main() -> None:
    args = parse_args()

    if not args.video.exists():
        print(f"Error: video not found: {args.video}", file=sys.stderr)
        sys.exit(2)
    if not args.subtitle.exists():
        print(f"Error: subtitle not found: {args.subtitle}", file=sys.stderr)
        sys.exit(2)

    check_dependencies(skip_sync=args.no_sync)

    if not audio.has_audio_track(args.video):
        print(f"Error: no audio track in {args.video}", file=sys.stderr)
        sys.exit(2)

    subtitles = srt.parse(args.subtitle)
    subtitle_sample = " ".join(s.text for s in subtitles[:50])

    # Timing sync
    sync_result = None
    _sync_tmp: Path | None = None
    if not args.no_sync:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
            _sync_tmp = Path(tmp.name)
            tmp.close()
            sync_result = sync.sync_subtitle(args.video, args.subtitle, _sync_tmp)
            subtitles = srt.parse(sync_result.synced_srt_path)
        except RuntimeError as exc:
            print(f"Warning: ffsubsync failed ({exc}), proceeding without sync",
                  file=sys.stderr)

    # Segment selection
    duration_ms = audio.get_duration_ms(args.video)
    segments = sampler.select_segments(subtitles, duration_ms, n=args.segments)

    # Transcription
    if not args.json:
        print(f"Loading Whisper model '{args.model}'...")
    model = transcribe.load_model(args.model)

    segment_results: list[output.SegmentResult] = []
    audio_lang: str | None = None

    for i, seg in enumerate(segments):
        if not args.json:
            print(f"  Transcribing segment {i + 1}/{len(segments)}...", end="\r")
        wav_path = audio.extract_segment(args.video, seg.start_ms, 30_000)
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
        finally:
            wav_path.unlink(missing_ok=True)

    if not args.json:
        print()

    # Language result
    lang_result = language.build_result(
        audio=audio_lang,
        subtitle_detected=language.detect_from_text(subtitle_sample),
        subtitle_filename=language.detect_from_filename(args.subtitle),
        video_meta=language.detect_from_video(args.video),
        expected=args.language,
    )

    # Confidence
    seg_scores = [
        compare.SegmentScore(
            f1=sr.score,
            wer=sr.wer,
            subtitle_tokens=len(seg.subtitle_text.split()),
        )
        for sr, seg in zip(segment_results, segments)
    ]
    confidence = compare.aggregate(seg_scores)

    result = output.MatchResult(
        confidence=confidence,
        passed=confidence >= args.threshold,
        threshold=args.threshold,
        language=lang_result,
        sync=sync_result,
        segments=segment_results,
        model=args.model,
    )

    # --keep-synced
    if args.keep_synced and sync_result:
        kept = args.subtitle.with_stem(args.subtitle.stem + ".synced")
        shutil.copy(sync_result.synced_srt_path, kept)
        if not args.json:
            print(f"Synced subtitle saved to {kept}")

    if args.json:
        print(output.format_json(result))
    else:
        output.print_human(result, verbose=args.verbose)

    if _sync_tmp is not None:
        _sync_tmp.unlink(missing_ok=True)

    sys.exit(0 if result.passed else 1)

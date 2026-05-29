from __future__ import annotations
import csv
import dataclasses
import io
import json
import sys
from pathlib import Path
from typing import Any

from submatch.output import BatchPairResult


class _PathEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def _write(path: str, content: str) -> None:
    try:
        Path(path).write_text(content, encoding="utf-8")
    except OSError as exc:
        print(f"Error: could not write {path}: {exc}", file=sys.stderr)
        sys.exit(2)


def write_json(results: list[BatchPairResult], path: str) -> None:
    items = []
    for p in results:
        d = dataclasses.asdict(p.result) if p.result is not None else {}
        d["video"] = str(p.video)
        d["subtitle"] = str(p.subtitle)
        if p.error is not None:
            d["error"] = p.error
        items.append(d)
    _write(path, json.dumps(items, cls=_PathEncoder, indent=2))


def write_csv(results: list[BatchPairResult], path: str) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "video", "subtitle", "state", "score", "threshold",
        "audio_lang", "subtitle_lang", "drift_detected", "cross_language", "error",
    ])
    for p in results:
        if p.result is None:
            writer.writerow([
                str(p.video), str(p.subtitle), "ERROR", "", "", "", "", "", "", p.error,
            ])
        else:
            r = p.result
            drift = r.sync.drift_detected if r.sync else False
            writer.writerow([
                str(p.video),
                str(p.subtitle),
                r.state.value,
                f"{r.confidence:.2f}",
                f"{r.threshold:.2f}",
                r.language.audio or "",
                r.subtitle_language or "",
                str(drift).lower(),
                str(r.cross_language).lower(),
                "",
            ])
    _write(path, buf.getvalue())

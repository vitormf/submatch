from __future__ import annotations
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from submatch.subtitle import parse as parse_subtitle

DRIFT_THRESHOLD_SECONDS = 2.0


@dataclass
class SyncResult:
    synced_srt_path: Path
    offset_seconds: float
    drift_detected: bool


def sync_subtitle(
    video_path: Path,
    subtitle_path: Path,
    output_path: Path | None = None,
    drift_threshold: float = DRIFT_THRESHOLD_SECONDS,
) -> SyncResult:
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
        output_path = Path(tmp.name)
        tmp.close()

    result = subprocess.run(
        ["ffs", str(video_path), "-i", str(subtitle_path), "-o", str(output_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffsubsync failed: {result.stderr.strip()}")

    offset = _compute_offset(subtitle_path, output_path)
    return SyncResult(
        synced_srt_path=output_path,
        offset_seconds=offset,
        drift_detected=abs(offset) > drift_threshold,
    )


def _compute_offset(original_path: Path, synced_path: Path) -> float:
    original = parse_subtitle(original_path)
    synced = parse_subtitle(synced_path)
    pairs = list(zip(original[:5], synced[:5]))
    if not pairs:
        return 0.0
    offsets = [(s.start_ms - o.start_ms) / 1_000.0 for o, s in pairs]
    return sum(offsets) / len(offsets)

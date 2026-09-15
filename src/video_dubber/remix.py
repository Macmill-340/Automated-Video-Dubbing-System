"""Remix stage: swap the dubbed audio under the original video.

The video stream is copied (``-c:v copy``): no re-encode, so even 2-hour
inputs mux in seconds. Audio is encoded to AAC for MP4 compatibility.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; install it to remix video.")
    return ffmpeg


def remix(
    video_path: str | Path,
    dub_wav: str | Path,
    out_path: str | Path,
    audio_bitrate: str = "192k",
) -> Path:
    """Mux *dub_wav* under *video_path*; return the dubbed video path."""
    ffmpeg = _require_ffmpeg()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                ffmpeg, "-y", "-i", str(video_path), "-i", str(dub_wav),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", audio_bitrate,
                "-shortest", str(out_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg remix failed: {exc.stderr}") from exc
    return out_path
